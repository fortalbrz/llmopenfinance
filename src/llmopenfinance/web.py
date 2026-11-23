"""Local web chat: a Flask app over the shared chat session (``llmopenfinance server``).

The app serves the user's financial data over plain HTTP, so every request passes three checks, in this order:

1. ``Host`` header (DNS-rebinding defence, :func:`host_allowed`): 400.
2. The per-launch secret token, read from the ``llmopenfinance_token`` cookie and compared in constant time: 401. The
   token reaches the browser once, as ``GET /?token=...``, which sets the ``HttpOnly``/``SameSite=Strict`` cookie
   and redirects to ``/``; no other route accepts it from the query string.
3. ``POST`` bodies must be ``application/json`` (415) and at most 64 KiB (413).

Every error is a JSON ``{"error": <localised text>}`` and every response carries ``Cache-Control: no-store``. The
token and the request data are never logged. ``GET /`` serves the chat page (:mod:`llmopenfinance.web_page`) with a
fresh nonce and a ``Content-Security-Policy`` that allows only its nonce'd inline style and script and ``fetch`` to
this origin; every other response gets a CSP that allows nothing at all.

One assistant and one read-only store serve every request thread, so all use of the session, the assistant and
DuckDB happens while holding :attr:`WebContext.lock`; a request that finds it held gets 409 instead of queueing.
``GET /api/state`` reads only cached values and never touches the database, so it answers even while busy.

Streaming routes (``POST /api/chat`` and ``POST /api/sync``) answer with NDJSON (``application/x-ndjson``): one JSON
object per line, ``{"type": "text"|"tool"|"notice"|"progress"|"error"|"done", ...}``, always ending with
``{"type": "done", "ok": bool}``. The handler takes the lock before returning the response and the stream gives it
back exactly once, when it ends, fails, is aborted by a disconnecting client or is closed without ever being read
(:func:`~llmopenfinance.web_stream.stream_under_lock`). A sync cannot be cancelled: an aborted sync stream keeps the
lock until the sync has finished and the read-only store has been reopened (:func:`_sync_events`).
"""

from __future__ import annotations

import logging
import os
import queue
import secrets
import socket
import threading
from collections.abc import Callable, Generator
from contextlib import closing
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from flask import Flask, Response, jsonify, redirect, request
from werkzeug.exceptions import HTTPException
from werkzeug.wrappers import Response as BaseResponse

from llmopenfinance import messages_pt
from llmopenfinance.agent import AssistantError, ToolStatus
from llmopenfinance.chat import TextDelta
from llmopenfinance.constants import DEFAULT_WEB_THREADS
from llmopenfinance.i18n import get_language, t
from llmopenfinance.session import BuildOptions, ChatSession, Factories, Notice, parse_command, run_command, run_sync
from llmopenfinance.web_page import render_page
from llmopenfinance.web_stream import StreamEvent, json_error, release_pending, stream_under_lock

if TYPE_CHECKING:
    from llmopenfinance.sync import ItemResult

__all__ = [
    "COOKIE_NAME",
    "MAX_CONTENT_LENGTH",
    "WebContext",
    "create_app",
    "host_allowed",
    "new_token",
    "serve",
    "ui_strings",
]

logger = logging.getLogger(__name__)

COOKIE_NAME = "llmopenfinance_token"
MAX_CONTENT_LENGTH = 64 * 1024
# Waitress buffers a whole request body before the app can answer 413 (its default limit is 1 GiB): cap it just above.
_WAITRESS_BODY_LIMIT = MAX_CONTENT_LENGTH + 4096
_LOOPBACK_NAMES = ("localhost", "127.0.0.1", "[::1]")
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})
_DEFAULT_HTTP_PORT = 80
# Commands served by POST /api/command; ``sync`` streams through its own route and ``exit`` means nothing here.
_WEB_COMMANDS = frozenset({"help", "summary", "reset", "history"})
_SYNC_THREAD_NAME = "llmopenfinance-web-sync"
# Every response that is not the page: nothing may load, run or frame it.
_API_CSP = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"


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


def _page_csp(nonce: str) -> str:
    """Build the page's ``Content-Security-Policy``: only its nonce'd inline style and script, ``fetch`` to itself.

    Args:
        nonce: The request's nonce (also on the page's ``<style>`` and ``<script>`` elements).

    Returns:
        str: The header value (no ``unsafe-inline`` or ``unsafe-eval``).
    """
    return (
        f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )


def _token_matches(candidate: str, token: str) -> bool:
    """Compare a presented token with the real one in constant time.

    Args:
        candidate: Token from the cookie or the query string.
        token: The per-launch token.

    Returns:
        bool: ``True`` when they are equal.
    """
    return secrets.compare_digest(candidate.encode("utf-8"), token.encode("utf-8"))


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


def _notice_event(notice: Notice) -> StreamEvent:
    """Turn a session notice into a ``notice`` stream event.

    Args:
        notice: The notice.

    Returns:
        StreamEvent: ``{"type": "notice", "level", "text"}``.
    """
    return {"type": "notice", "level": notice.level, "text": notice.text}


def _progress_event(result: ItemResult) -> StreamEvent:
    """Turn one item's sync result into a ``progress`` stream event (the data the CLI prints per item).

    Args:
        result: The item's result.

    Returns:
        StreamEvent: ``{"type": "progress", "item_id", "ok", "skipped", "counts", "warnings", "error"}``.
    """
    return {
        "type": "progress",
        "item_id": result.item_id,
        "ok": result.ok,
        "skipped": result.skipped,
        "counts": dict(result.counts),
        "warnings": list(result.warnings),
        "error": result.error,
    }


def _command_events(ctx: WebContext, name: str) -> Generator[StreamEvent]:
    """Stream the outcome of a chat command typed in the message box (``sync`` has its own stream).

    ``exit`` only explains how to leave: the server keeps running.

    Args:
        ctx: Shared web context (the caller holds its lock).
        name: Command name from :func:`~llmopenfinance.session.parse_command`, other than ``sync``.

    Yields:
        StreamEvent: ``notice`` events, then ``done``.
    """
    notices = (Notice("dim", t("web.exit_hint")),) if name == "exit" else _run_web_command(ctx, name)
    for notice in notices:
        yield _notice_event(notice)
    yield {"type": "done", "ok": True}


@dataclass(frozen=True)
class _WorkerDone:
    """Last item a sync worker puts on its queue.

    Attributes:
        ok: ``True`` when the sync's exit code was 0.
    """

    ok: bool


def _sync_worker(ctx: WebContext, *, force: bool, put: Callable[[StreamEvent | _WorkerDone], None]) -> None:
    """Run the sync with the read-only store closed (it is reopened in every case), reporting through ``put``.

    Args:
        ctx: Shared web context (the request thread holds its lock for the whole run).
        force: Re-download unchanged items.
        put: Receives each ``notice``, ``progress`` and ``error`` event, then always a :class:`_WorkerDone`.
    """
    ok = False
    try:
        sync = partial(
            run_sync,
            ctx.factories,
            force=force,
            notify=lambda notice: put(_notice_event(notice)),
            on_item=lambda result: put(_progress_event(result)),
        )
        ok = ctx.session.resync(sync) == 0
    except Exception:
        logger.exception("Unexpected error in the web sync")
        put({"type": "error", "text": t("web.internal_error")})
    finally:
        put(_WorkerDone(ok=ok))


def _after_sync(ctx: WebContext) -> bool:
    """Start a new conversation over the fresh data and re-read the status, whatever the sync's outcome (as ``/sync``).

    Args:
        ctx: Shared web context (the caller holds its lock).

    Returns:
        bool: ``False`` when this failed (the error is logged).
    """
    try:
        ctx.assistant.refresh()
        ctx.transcript.clear()
        ctx.refresh_status()
    except Exception:
        logger.exception("Unexpected error while refreshing after the web sync")
        return False
    return True


def _sync_events(ctx: WebContext, *, force: bool) -> Generator[StreamEvent]:
    """Run a sync in a worker thread and stream its notices and per-item progress as they happen.

    The sync's callbacks are synchronous, so the worker feeds a queue that this generator drains. The sync cannot be
    cancelled: when the client disconnects (``GeneratorExit``) this generator still waits for the worker, so the
    caller's lock is only given back once the sync has finished and the read-only store has been reopened. The
    conversation, transcript and status are refreshed in every case, under the lock.

    Args:
        ctx: Shared web context (the caller holds its lock).
        force: Re-download unchanged items.

    Yields:
        StreamEvent: ``notice``, ``progress`` and ``error`` events, then ``done`` (``ok`` when the exit code is 0).
    """
    events: queue.SimpleQueue[StreamEvent | _WorkerDone] = queue.SimpleQueue()
    worker = threading.Thread(
        target=_sync_worker, args=(ctx,), kwargs={"force": force, "put": events.put}, name=_SYNC_THREAD_NAME
    )
    # A daemon, so Ctrl+C stops the server without waiting for a sync; DuckDB recovers an interrupted write.
    worker.daemon = True
    worker.start()
    ok = False
    try:
        while not isinstance(item := events.get(), _WorkerDone):
            yield item
        ok = item.ok
    finally:
        worker.join()
        refreshed = _after_sync(ctx)
    if not refreshed:
        yield {"type": "error", "text": t("web.internal_error")}
    yield {"type": "done", "ok": ok and refreshed}


def _sync_force() -> bool | None:
    """Read ``force`` from the request's optional JSON body (no body, ``null`` and ``{}`` mean ``False``).

    Returns:
        bool | None: The flag, or ``None`` when the body is not an object or ``force`` is not a boolean.
    """
    body = request.get_json() if request.get_data() else None  # malformed JSON raises a 400 BadRequest
    if body is None:
        return False
    force = body.get("force", False) if isinstance(body, dict) else None
    return force if isinstance(force, bool) else None


def _guard(ctx: WebContext) -> BaseResponse | None:
    """Check Host, then token, then content type; return the rejection, or ``None`` to let the request through.

    Args:
        ctx: Shared web context.

    Returns:
        BaseResponse | None: An error or login-redirect response, or ``None``.
    """
    if not host_allowed(request.headers.get("Host", ""), ctx.host, ctx.port):
        return json_error(400, t("web.bad_host"))
    if request.method == "GET" and request.path == "/":
        query_token = request.args.get("token")
        if query_token is not None and _token_matches(query_token, ctx.token):
            return _login(ctx.token)
    if not _token_matches(request.cookies.get(COOKIE_NAME, ""), ctx.token):
        return json_error(401, t("web.unauthorized"))
    if request.method == "POST" and request.mimetype != "application/json":
        return json_error(415, t("web.unsupported_media"))
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
        response.headers.setdefault("Content-Security-Policy", _API_CSP)  # the page sets its own, nonce'd one
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException) -> Response:
        # A failing response hook reaches here as a 500 (InternalServerError) after the view took the lock.
        release_pending()
        status = exc.code or 500
        return json_error(status, _http_error_text(status))

    @app.errorhandler(Exception)
    def unexpected_error(_exc: Exception) -> Response:
        release_pending()
        # The endpoint name only: never the URL, headers, cookies or body (they may carry the token or user data).
        logger.exception("Unexpected error in web endpoint %s", request.endpoint)
        return json_error(500, t("web.internal_error"))

    @app.get("/")
    def index() -> Response:
        nonce = secrets.token_urlsafe(18)
        page = render_page(language=get_language(), strings=ui_strings(), nonce=nonce)
        response = Response(page, mimetype="text/html")
        response.headers["Content-Security-Policy"] = _page_csp(nonce)
        return response

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
            return json_error(400, t("web.bad_request"))
        if not ctx.lock.acquire(blocking=False):
            return json_error(409, t("web.busy"))
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
            return json_error(400, t("web.bad_request"))
        message = message.strip()
        name = parse_command(message)
        if name == "sync":
            return stream_under_lock(ctx.lock, partial(_sync_events, ctx, force=False))
        if name is not None:
            return stream_under_lock(ctx.lock, partial(_command_events, ctx, name))
        return stream_under_lock(ctx.lock, partial(_chat_events, ctx, message))

    @app.post("/api/sync")
    def sync() -> Response:
        force = _sync_force()
        if force is None:
            return json_error(400, t("web.bad_request"))
        return stream_under_lock(ctx.lock, partial(_sync_events, ctx, force=force))

    return app


def _ensure_bindable(host: str, port: int) -> None:
    """Fail with ``OSError`` when something already listens on ``host:port``.

    Waitress sets ``SO_REUSEADDR`` before it binds. On Windows that lets a second process share a port that is already
    in use, so a second server would start silently and some requests would reach the wrong one. A probe socket without
    the option is refused there. On other systems ``SO_REUSEADDR`` only tolerates lingering connections, so the probe
    sets it as Waitress does and is refused only by a live listener. The ``OSError`` raised for an address in use, not
    permitted or not resolvable (``socket.gaierror`` is one) is left to propagate.

    Args:
        host: Address to bind.
        port: TCP port.
    """
    family, kind, proto, _name, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    with socket.socket(family, kind, proto) as probe:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(address)


def serve(ctx: WebContext, *, threads: int = DEFAULT_WEB_THREADS, serve_fn: Callable[..., Any] | None = None) -> None:
    """Serve the app with Waitress until interrupted.

    Waitress is imported here, so importing this module never needs it. Access logging stays off (Waitress logs none
    unless wrapped), so the ``/?token=`` URL never reaches a log, and no proxy headers are trusted.
    An address that cannot be bound (in use, no permission, bad host) surfaces as an ``OSError``. With Waitress itself
    the address is probed first (:func:`_ensure_bindable`), because Waitress would not notice a port in use on Windows.

    Args:
        ctx: Shared web context; its host and port are the bind address.
        threads: Worker threads; at least 2 keeps ``/api/state`` answerable while a chat or sync stream holds one.
        serve_fn: Replacement for ``waitress.serve`` (the test seam).
    """
    if serve_fn is None:
        import waitress

        _ensure_bindable(ctx.host, ctx.port)
        serve_fn = waitress.serve
    serve_fn(
        create_app(ctx),
        host=ctx.host,
        port=ctx.port,
        threads=threads,
        max_request_body_size=_WAITRESS_BODY_LIMIT,
    )
