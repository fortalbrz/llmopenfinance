"""Lock-owning NDJSON streaming for the web chat's streaming routes (``POST /api/chat``, ``POST /api/sync``).

A streaming route takes the shared lock before returning its response, and the stream gives it back exactly once:
when it ends, fails, is aborted by a disconnecting client, or is closed without ever being read; or, when a later
response hook fails and the response is dropped unsent, from the app's error handler (:func:`release_pending`).
Each line is one JSON object, and every stream ends with ``{"type": "done", "ok": bool}``.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Generator, Iterator
from contextlib import closing
from dataclasses import dataclass
from typing import Any

from flask import Response, g, jsonify, request, stream_with_context

from llmopenfinance.i18n import t

__all__ = ["StreamEvent", "json_error", "release_pending", "stream_under_lock"]

logger = logging.getLogger(__name__)

_NDJSON = "application/x-ndjson"

StreamEvent = dict[str, Any]


def json_error(status: int, text: str) -> Response:
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


@dataclass
class _Abort:
    """Whether the client went away mid-stream.

    Attributes:
        seen: Set once a ``GeneratorExit`` reached the stream.
    """

    seen: bool = False


def _ndjson_lines(events: Iterator[StreamEvent], abort: _Abort) -> Iterator[str]:
    """Encode events as NDJSON lines, noting in ``abort`` when the stream is closed early (``GeneratorExit``).

    Args:
        events: The events.
        abort: Marked when the client goes away.

    Yields:
        str: NDJSON lines.
    """
    for event in events:
        try:
            yield _ndjson(event)
        except GeneratorExit:
            abort.seen = True
            raise


def _ndjson_stream(
    produce: Callable[[], Generator[StreamEvent]], release: _ReleaseOnce, endpoint: str | None
) -> Iterator[str]:
    """Encode the events of ``produce()`` as NDJSON, turning an unexpected failure into ``error`` + ``done``.

    The lock is released in ``finally`` (end, failure or ``GeneratorExit`` from a disconnecting client), after the
    event generator has been closed, so its own cleanup (the assistant's history rollback, the end of a sync) still
    runs under the lock. Once the client has gone (``GeneratorExit``) nothing more is yielded, even when that cleanup
    fails: the failure is only logged.

    Args:
        produce: Creates the event generator; called on the first read, in the thread that serves the stream.
        release: Releases the request's lock.
        endpoint: Endpoint name for the log (never the request data).

    Yields:
        str: NDJSON lines.
    """
    abort = _Abort()
    try:
        with closing(produce()) as events:
            yield from _ndjson_lines(events, abort)
    except Exception:
        logger.exception("Unexpected error while streaming web endpoint %s", endpoint)
        if abort.seen:
            return
        yield _ndjson({"type": "error", "text": t("web.internal_error")})
        yield _ndjson({"type": "done", "ok": False})
    finally:
        release()


def stream_under_lock(lock: threading.Lock, produce: Callable[[], Generator[StreamEvent]]) -> Response:
    """Take ``lock`` without blocking and stream ``produce()`` as NDJSON while holding it.

    The lock is released exactly once: by the stream's ``finally`` once it has started, or by the response's
    ``close`` (``call_on_close``) when the WSGI server closes it without ever reading it, since a generator that never
    started does not run its ``finally``. The release is also left on :data:`flask.g`, so that when a later response
    hook fails and the response is dropped unsent, the error handler gives the lock back (:func:`release_pending`).

    Args:
        lock: The lock guarding the session, the assistant and DuckDB.
        produce: Creates the event generator, which must end with a ``done`` event.

    Returns:
        Response: The NDJSON stream, or a JSON 409 when the lock is held (nothing is queued).
    """
    if not lock.acquire(blocking=False):
        return json_error(409, t("web.busy"))
    release = _ReleaseOnce(lock)
    try:
        g.pending_release = release
        stream = _ndjson_stream(produce, release, request.endpoint)
        response = Response(stream_with_context(stream), mimetype=_NDJSON)
        response.call_on_close(release)
    except BaseException:
        release()
        raise
    return response


def release_pending() -> None:
    """Give back the lock of a streaming response that will never be sent (an error after the view returned it).

    Call it from the app's error handlers; it does nothing when the request took no lock or already gave it back.
    """
    release = g.pop("pending_release", None)
    if release is not None:
        release()
