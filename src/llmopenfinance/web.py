"""Local web chat: a Flask app over the shared chat session (``finassist server``).

The app serves the user's financial data over plain HTTP, so every request passes three checks, in this order:

1. ``Host`` header (DNS-rebinding defence, :func:`host_allowed`): 400.
2. The per-launch secret token, read from the ``finassist_token`` cookie and compared in constant time: 401. The
   token reaches the browser once, as ``GET /?token=...``, which sets the ``HttpOnly``/``SameSite=Strict`` cookie
   and redirects to ``/``; no other route accepts it from the query string.
3. ``POST`` bodies must be ``application/json`` (415) and at most 64 KiB (413).

Every error is a JSON ``{"error": <localised text>}`` and every response carries ``Cache-Control: no-store``. The
token and the request data are never logged.

One assistant and one read-only store serve every request thread, so all use of the session, the assistant and
DuckDB happens while holding :attr:`WebContext.lock`; a request that finds it held gets 409 instead of queueing.
``GET /api/state`` reads only cached values and never touches the database, so it answers even while busy.
"""

from __future__ import annotations

import html
import logging
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any

from flask import Flask, Response, jsonify, redirect, request
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Response as BaseResponse

from llmopenfinance import messages_pt
from llmopenfinance.i18n import get_language, t
from llmopenfinance.session import BuildOptions, ChatSession, Factories, run_command

__all__ = ["COOKIE_NAME", "MAX_CONTENT_LENGTH", "WebContext", "create_app", "host_allowed", "new_token", "ui_strings"]

logger = logging.getLogger(__name__)

COOKIE_NAME = "finassist_token"
MAX_CONTENT_LENGTH = 64 * 1024
_LOOPBACK_NAMES = ("localhost", "127.0.0.1", "[::1]")
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})
_DEFAULT_HTTP_PORT = 80
# Commands served by POST /api/command; ``sync`` streams through its own route and ``exit`` means nothing here.
_WEB_COMMANDS = frozenset({"help", "summary", "reset", "history"})


def new_token() -> str:
    """Create the per-launch secret token.

    Returns:
        str: A random URL-safe token (32 characters).
    """
    return secrets.token_urlsafe(24)


def host_allowed(host_header: str, bound_host: str, port: int) -> bool:
    """Tell whether a request's ``Host`` header names this server (DNS-rebinding defence).

    Accepted (case-insensitive): ``localhost``, ``127.0.0.1`` and ``[::1]`` with the bound port, plus the bound host
    itself with the port when it is a specific address or name (an IPv6 address in brackets). For a wildcard bind
    (``0.0.0.0`` or ``::``) the client's ``Host`` cannot be predicted, so any non-empty value is accepted and the
    token alone protects access. On port 80 the port may be omitted, as browsers do.

    Args:
        host_header: Raw ``Host`` header value (empty when missing).
        bound_host: The ``--host`` the server is bound to.
        port: The bound port.

    Returns:
        bool: ``True`` when the request may proceed to the token check.
    """
    host = host_header.strip().lower()
    if not host:
        return False
    bound = bound_host.strip().lower()
    if bound in _WILDCARD_HOSTS:
        return True
    if ":" in bound and not bound.startswith("["):
        bound = f"[{bound}]"
    names = {*_LOOPBACK_NAMES, bound}
    allowed = {f"{name}:{port}" for name in names}
    if port == _DEFAULT_HTTP_PORT:
        allowed |= names
    return host in allowed


def ui_strings() -> dict[str, str]:
    """Return the page's localised strings (every ``web.ui.*`` catalog key) in the current language.

    Returns:
        dict[str, str]: Catalog key -> text.
    """
    return {key: t(key) for key in messages_pt.MESSAGES if key.startswith("web.ui.")}


@dataclass
class WebContext:
    """Everything the web routes share; :attr:`lock` guards the session, the assistant and the database.

    Attributes:
        session: Chat session holding the read-only store.
        assistant: Active ``FinanceAssistant``.
        factories: Resource factories (the sync route uses them).
        token: Per-launch secret token.
        host: Bound host (``--host``).
        port: Bound port.
        options: Assistant options chosen at startup.
        provider: LLM provider name shown in the status line.
        model: LLM model id shown in the status line.
        lock: Held by any request that uses the session, the assistant or DuckDB.
        transcript: ``{"role", "text"}`` turns shown on reload; in memory only, never written to disk or logs.
        status: Cached ``last_sync`` and ``counts`` served by ``/api/state``.
    """

    session: ChatSession
    assistant: Any
    factories: Factories
    token: str
    host: str
    port: int
    options: BuildOptions
    provider: str
    model: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    transcript: list[dict[str, str]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill the status cache so ``/api/state`` is never empty."""
        self.refresh_status()

    def refresh_status(self) -> None:
        """Re-read ``last_sync`` and the row counts from the store. Call only at startup or while holding the lock."""
        store = self.session.store
        self.status = {"last_sync": store.get_meta("last_sync"), "counts": store.counts()}


def _token_matches(candidate: str, token: str) -> bool:
    """Compare a presented token with the real one in constant time.

    Args:
        candidate: Token from the cookie or the query string.
        token: The per-launch token.

    Returns:
        bool: ``True`` when they are equal.
    """
    return secrets.compare_digest(candidate.encode("utf-8"), token.encode("utf-8"))


def _error(status: int, text: str) -> Response:
    """Build a JSON error response.

    Args:
        status: HTTP status code.
        text: Localised message for the user.

    Returns:
        Response: ``{"error": text}`` with ``status``.
    """
    response = jsonify(error=text)
    response.status_code = status
    return response


def _http_error_text(status: int) -> str:
    """Return the localised message for an HTTP error that Flask or Werkzeug raised (404, 405, 413, ...).

    Args:
        status: HTTP status code.

    Returns:
        str: The message.
    """
    match status:
        case 404:
            return t("web.not_found")
        case 413:
            return t("web.too_large")
        case _ if status >= 500:
            return t("web.internal_error")
        case _:
            return t("web.bad_request")


def _no_sync() -> int:
    """Stand-in sync entry point for :func:`run_command`, which only calls it for ``sync`` (never routed here).

    Raises:
        RuntimeError: Always; the sync has its own streaming route.
    """
    raise RuntimeError("sync runs through /api/sync, not /api/command")


def _login(token: str) -> BaseResponse:
    """Set the auth cookie and redirect to the page, dropping the token from the address bar.

    Args:
        token: The per-launch token.

    Returns:
        BaseResponse: 302 to ``/`` with the cookie.
    """
    response = redirect("/", code=302)
    # Plain HTTP on purpose (local server), so no ``Secure`` flag: the browser would never send the cookie back.
    response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="Strict", path="/")
    return response


def _guard(ctx: WebContext) -> BaseResponse | None:
    """Check Host, then token, then content type; return the rejection, or ``None`` to let the request through.

    Args:
        ctx: Shared web context.

    Returns:
        BaseResponse | None: An error or login-redirect response, or ``None``.
    """
    if not host_allowed(request.headers.get("Host", ""), ctx.host, ctx.port):
        return _error(400, t("web.bad_host"))
    if request.method == "GET" and request.path == "/":
        query_token = request.args.get("token")
        if query_token is not None and _token_matches(query_token, ctx.token):
            return _login(ctx.token)
    if not _token_matches(request.cookies.get(COOKIE_NAME, ""), ctx.token):
        return _error(401, t("web.unauthorized"))
    if request.method == "POST" and request.mimetype != "application/json":
        return _error(415, t("web.unsupported_media"))
    return None


def create_app(ctx: WebContext) -> Flask:
    """Build the Flask app serving ``ctx``.

    Args:
        ctx: Shared web context.

    Returns:
        Flask: The WSGI app (served by Waitress in production).
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.before_request
    def check_request() -> BaseResponse | None:
        return _guard(ctx)

    @app.after_request
    def add_headers(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Response:
        status = exc.code or 500
        return _error(status, _http_error_text(status))

    @app.errorhandler(Exception)
    def unexpected_error(_exc: Exception) -> Response:
        # The endpoint name only: never the URL, headers, cookies or body (they may carry the token or user data).
        logger.exception("Unexpected error in web endpoint %s", request.endpoint)
        return _error(500, t("web.internal_error"))

    @app.get("/")
    def index() -> Response:
        # Placeholder until the chat page exists.
        title = html.escape(t("web.ui.title"))
        page = f'<!doctype html><html lang="{get_language()}"><title>{title}</title><h1>{title}</h1></html>'
        return Response(page, mimetype="text/html")

    @app.get("/api/state")
    def state() -> Response:
        status = ctx.status
        return jsonify(
            provider=ctx.provider,
            model=ctx.model,
            language=get_language(),
            last_sync=status.get("last_sync"),
            counts=status.get("counts", {}),
            busy=ctx.lock.locked(),
            strings=ui_strings(),
            transcript=list(ctx.transcript),
        )

    @app.post("/api/command")
    def command() -> Response:
        body = request.get_json(silent=True)
        name = body.get("name") if isinstance(body, dict) else None
        if not isinstance(name, str) or name not in _WEB_COMMANDS:
            return _error(400, t("web.bad_request"))
        if not ctx.lock.acquire(blocking=False):
            return _error(409, t("web.busy"))
        try:
            notices = run_command(name, ctx.assistant, ctx.session, _no_sync)
            if name == "reset":
                ctx.transcript.clear()
        finally:
            ctx.lock.release()
        return jsonify(notices=[{"level": notice.level, "text": notice.text} for notice in notices])

    return app
