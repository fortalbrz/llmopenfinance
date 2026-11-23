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

Streaming routes (``POST /api/chat``) answer with NDJSON (``application/x-ndjson``): one JSON object per line,
``{"type": "text"|"tool"|"notice"|"error"|"done", ...}``, always ending with ``{"type": "done", "ok": bool}``. The
handler takes the lock before returning the response and the stream gives it back exactly once, when it ends,
fails, is aborted by a disconnecting client or is closed without ever being read (:func:`_stream_under_lock`).
"""

from __future__ import annotations

import html
import json
import logging
import secrets
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from flask import Flask, Response, jsonify, redirect, request, stream_with_context
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Response as BaseResponse

from llmopenfinance import messages_pt
from llmopenfinance.agent import AssistantError, ToolStatus
from llmopenfinance.chat import TextDelta
from llmopenfinance.i18n import get_language, t
from llmopenfinance.session import BuildOptions, ChatSession, Factories, Notice, parse_command, run_command

__all__ = ["COOKIE_NAME", "MAX_CONTENT_LENGTH", "WebContext", "create_app", "host_allowed", "new_token", "ui_strings"]

logger = logging.getLogger(__name__)

COOKIE_NAME = "finassist_token"
MAX_CONTENT_LENGTH = 64 * 1024
_LOOPBACK_NAMES = ("localhost", "127.0.0.1", "[::1]")
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})
_DEFAULT_HTTP_PORT = 80
# Commands served by POST /api/command; ``sync`` streams through its own route and ``exit`` means nothing here.
_WEB_COMMANDS = frozenset({"help", "summary", "reset", "history"})
_NDJSON = "application/x-ndjson"

StreamEvent = dict[str, Any]


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


def _run_web_command(ctx: WebContext, name: str) -> tuple[Notice, ...]:
    """Run ``help``, ``summary``, ``reset`` or ``history``; ``reset`` also clears the transcript. Hold the lock.

    Args:
        ctx: Shared web context.
        name: Command name (one of :data:`_WEB_COMMANDS`).

    Returns:
        tuple[Notice, ...]: Messages for the user, in display order.
    """
    notices = run_command(name, ctx.assistant, ctx.session, _no_sync)
    if name == "reset":
        ctx.transcript.clear()
    return notices


def _ndjson(event: StreamEvent) -> str:
    """Encode one stream event as an NDJSON line (non-ASCII text kept as is; the page escapes it).

    Args:
        event: The event.

    Returns:
        str: One JSON object followed by ``\\n``.
    """
    return json.dumps(event, ensure_ascii=False, default=str) + "\n"


class _ReleaseOnce:
    """Release a lock exactly once, however many of a stream's exit paths call it (thread-safe)."""

    def __init__(self, lock: threading.Lock) -> None:
        """Remember the lock, which the caller has just acquired.

        Args:
            lock: The held lock.
        """
        self._lock = lock
        self._guard = threading.Lock()
        self._held = True

    def __call__(self) -> None:
        """Release the lock on the first call; later calls do nothing."""
        with self._guard:
            if not self._held:
                return
            self._held = False
        self._lock.release()


def _ndjson_stream(
    produce: Callable[[], Generator[StreamEvent]], release: _ReleaseOnce, endpoint: str | None
) -> Iterator[str]:
    """Encode the events of ``produce()`` as NDJSON, turning an unexpected failure into ``error`` + ``done``.

    The lock is released in ``finally`` (end, failure or ``GeneratorExit`` from a disconnecting client), after the
    event generator has been closed, so its own cleanup (the assistant's history rollback) still runs under the lock.

    Args:
        produce: Creates the event generator; called on the first read, in the thread that serves the stream.
        release: Releases the request's lock.
        endpoint: Endpoint name for the log (never the request data).

    Yields:
        str: NDJSON lines.
    """
    try:
        with closing(produce()) as events:
            for event in events:
                yield _ndjson(event)
    except Exception:
        logger.exception("Unexpected error while streaming web endpoint %s", endpoint)
        yield _ndjson({"type": "error", "text": t("web.internal_error")})
        yield _ndjson({"type": "done", "ok": False})
    finally:
        release()


def _stream_under_lock(lock: threading.Lock, produce: Callable[[], Generator[StreamEvent]]) -> Response:
    """Take ``lock`` without blocking and stream ``produce()`` as NDJSON while holding it.

    The lock is released exactly once: by the stream's ``finally`` once it has started, or by the response's
    ``close`` (``call_on_close``) when the WSGI server closes it without ever reading it, since a generator that never
    started does not run its ``finally``.

    Args:
        lock: The lock guarding the session, the assistant and DuckDB.
        produce: Creates the event generator, which must end with a ``done`` event.

    Returns:
        Response: The NDJSON stream, or a JSON 409 when the lock is held (nothing is queued).
    """
    if not lock.acquire(blocking=False):
        return _error(409, t("web.busy"))
    release = _ReleaseOnce(lock)
    try:
        stream = _ndjson_stream(produce, release, request.endpoint)
        response = Response(stream_with_context(stream), mimetype=_NDJSON)
        response.call_on_close(release)
    except BaseException:
        release()
        raise
    return response


def _chat_events(ctx: WebContext, message: str) -> Generator[StreamEvent]:
    """Stream the assistant's answer to ``message``; record the turn in the transcript only when it succeeded.

    Closing this generator closes ``assistant.ask``, whose cleanup rolls the unfinished turn out of the history.

    Args:
        ctx: Shared web context (the caller holds its lock).
        message: The user's question.

    Yields:
        StreamEvent: ``text``, ``tool`` and ``error`` events, then ``done``.
    """
    parts: list[str] = []
    ok = True
    with closing(ctx.assistant.ask(message)) as answer:
        for event in answer:
            if isinstance(event, TextDelta):
                parts.append(event.text)
                yield {"type": "text", "text": event.text}
            elif isinstance(event, ToolStatus):
                yield {"type": "tool", "name": event.call.name, "arguments": dict(event.call.arguments)}
            elif isinstance(event, AssistantError):
                ok = False
                yield {"type": "error", "text": event.message}
    if ok:
        ctx.transcript.extend([{"role": "user", "text": message}, {"role": "assistant", "text": "".join(parts)}])
    yield {"type": "done", "ok": ok}


def _command_events(ctx: WebContext, name: str) -> Generator[StreamEvent]:
    """Stream the outcome of a chat command typed in the message box.

    ``sync`` only points to the Sync button for now, and ``exit`` explains how to leave: the server keeps running.

    Args:
        ctx: Shared web context (the caller holds its lock).
        name: Command name from :func:`~llmopenfinance.session.parse_command`.

    Yields:
        StreamEvent: ``notice`` events, then ``done``.
    """
    ok = True
    match name:
        case "sync":
            notices: tuple[Notice, ...] = (Notice("warn", t("web.sync_via_button")),)
            ok = False
        case "exit":
            notices = (Notice("dim", t("web.exit_hint")),)
        case _:
            notices = _run_web_command(ctx, name)
    for notice in notices:
        yield {"type": "notice", "level": notice.level, "text": notice.text}
    yield {"type": "done", "ok": ok}


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
            notices = _run_web_command(ctx, name)
        finally:
            ctx.lock.release()
        return jsonify(notices=[{"level": notice.level, "text": notice.text} for notice in notices])

    @app.post("/api/chat")
    def chat() -> Response:
        body = request.get_json(silent=True)
        message = body.get("message") if isinstance(body, dict) else None
        if not isinstance(message, str) or not message.strip():
            return _error(400, t("web.bad_request"))
        message = message.strip()
        name = parse_command(message)
        if name is not None:
            return _stream_under_lock(ctx.lock, partial(_command_events, ctx, name))
        return _stream_under_lock(ctx.lock, partial(_chat_events, ctx, message))

    return app
