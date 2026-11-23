"""Tests for the local web chat: Host check, token cookie, JSON-only POSTs, state, command, chat and sync routes."""

import json
import logging
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import pytest
from flask import request_finished
from flask.testing import FlaskClient
from werkzeug.test import EnvironBuilder
from werkzeug.wsgi import ClosingIterator

from llmopenfinance import web, web_stream
from llmopenfinance.agent import FinanceAssistant, build_system_prompt
from llmopenfinance.chat import LLMError, TextDelta, TurnEnd
from llmopenfinance.constants import ENV_PLUGGY_CLIENT_ID, ENV_PLUGGY_CLIENT_SECRET
from llmopenfinance.i18n import t
from llmopenfinance.session import BuildOptions, ChatSession, Factories, Notice
from llmopenfinance.store import FinanceStore
from llmopenfinance.sync import ItemResult
from llmopenfinance.tools import ToolCall, ToolRegistry
from llmopenfinance.web import COOKIE_NAME, WebContext, create_app, host_allowed, new_token
from llmopenfinance.web_page import render_page
from tests.conftest import BANK_ID, ITEM_ID, FakePluggy
from tests.test_agent import ScriptedChat, answer, tool_turn
from tests.test_session import make_factories, store_opener

UseLanguage = Callable[[str], None]

TOKEN = "s3cr3t-T0ken_value-for-tests"
HOST = "127.0.0.1:8000"
STATE_KEYS = {"provider", "model", "language", "last_sync", "counts", "busy", "strings", "transcript"}
SYNC_THREAD_NAME = "finassist-web-sync"  # the name web.py gives its sync worker thread


def make_ctx(
    db_path: Path, *, turns: list[Any] | None = None, host: str = "127.0.0.1", pluggy: Any = None
) -> WebContext:
    session = ChatSession(store_opener(db_path))
    chat = ScriptedChat(turns or [])
    assistant = FinanceAssistant(chat, ToolRegistry(session.connection), session.snapshot)
    return WebContext(
        session=session,
        assistant=assistant,
        factories=make_factories(db_path, llm=chat, pluggy=pluggy),
        token=TOKEN,
        host=host,
        port=8000,
        options=BuildOptions(think=False, enable_sql=None),
        provider="ollama",
        model="fake-model",
    )


@pytest.fixture
def ctx(db_path: Path) -> Iterator[WebContext]:
    context = make_ctx(db_path, turns=[answer("Resposta")])
    yield context
    context.session.close()


def client_for(ctx: WebContext, *, authenticated: bool = True) -> FlaskClient:
    client = create_app(ctx).test_client()
    if authenticated:
        client.set_cookie(COOKIE_NAME, TOKEN, domain="127.0.0.1")
    return client


def get(client: FlaskClient, path: str, *, host: str = HOST, **kwargs: Any) -> Any:
    return client.get(path, headers={"Host": host, **kwargs.pop("headers", {})}, **kwargs)


def post(client: FlaskClient, path: str, body: Any = None, *, host: str = HOST, **kwargs: Any) -> Any:
    if body is not None:
        kwargs.setdefault("data", json.dumps(body))
        kwargs.setdefault("content_type", "application/json")
    return client.post(path, headers={"Host": host, **kwargs.pop("headers", {})}, **kwargs)


def command(client: FlaskClient, name: Any) -> Any:
    return post(client, "/api/command", {"name": name})


def chat(client: FlaskClient, message: Any, **kwargs: Any) -> Any:
    return post(client, "/api/chat", {"message": message}, **kwargs)


def events(response: Any) -> list[dict[str, Any]]:
    """Parse an NDJSON stream; split on ``\\n`` only (``str.splitlines`` would also split on U+2028)."""
    assert response.status_code == 200
    assert response.mimetype == "application/x-ndjson"
    body = response.get_data(as_text=True)
    assert body.endswith("\n")
    return [json.loads(line) for line in body.split("\n")[:-1]]


def history_length(ctx: WebContext) -> int:
    return len(ctx.assistant.chat.conversation_history)


@pytest.fixture(autouse=True)
def _no_sync_worker_left() -> Iterator[None]:
    """Fail a test that leaves a sync worker thread running (every worker must be joined by its stream)."""
    yield
    for thread in threading.enumerate():
        if thread.name == SYNC_THREAD_NAME:
            thread.join(timeout=10)
            assert not thread.is_alive(), "a web sync worker outlived its test"


# ---------------------------------------------------------------------------
# Host header check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host_header", "bound_host", "port", "expected"),
    [
        ("localhost:8000", "127.0.0.1", 8000, True),
        ("127.0.0.1:8000", "127.0.0.1", 8000, True),
        ("[::1]:8000", "127.0.0.1", 8000, True),
        ("LocalHost:8000", "127.0.0.1", 8000, True),  # case-insensitive
        ("evil.example:8000", "127.0.0.1", 8000, False),  # DNS rebinding
        ("localhost:8001", "127.0.0.1", 8000, False),  # wrong port
        ("localhost", "127.0.0.1", 8000, False),  # port missing
        ("localhost:8000.evil.example", "127.0.0.1", 8000, False),
        ("", "127.0.0.1", 8000, False),
        ("", "0.0.0.0", 8000, False),  # empty Host is rejected even for a wildcard bind
        ("evil.example:8000", "0.0.0.0", 8000, True),  # wildcard bind: the token alone protects
        ("phone.lan:1234", "::", 8000, True),
        ("192.168.1.5:8000", "192.168.1.5", 8000, True),  # the exact --host value
        ("192.168.1.6:8000", "192.168.1.5", 8000, False),
        ("Finance.Lan:8000", "finance.lan", 8000, True),
        ("finance.lan:8000", "FINANCE.LAN", 8000, True),
        ("[fe80::1]:8000", "fe80::1", 8000, True),  # an IPv6 --host is bracketed in Host
        ("[fe80::1]:8000", "[fe80::1]", 8000, True),
        ("localhost", "127.0.0.1", 80, True),  # browsers omit the default port
        ("localhost:80", "127.0.0.1", 80, True),
    ],
)
def test_host_allowed(host_header: str, bound_host: str, port: int, expected: bool):
    assert host_allowed(host_header, bound_host, port) is expected


def test_wrong_host_rejected_before_token(ctx: WebContext):
    for client in (client_for(ctx, authenticated=False), client_for(ctx)):
        response = get(client, "/api/state", host="evil.example:8000")
        assert response.status_code == 400
        assert response.get_json() == {"error": t("web.bad_host")}
    response = get(client_for(ctx, authenticated=False), f"/?token={TOKEN}", host="evil.example:8000")
    assert response.status_code == 400
    assert "Set-Cookie" not in response.headers


def test_wildcard_bind_accepts_any_host_with_the_token(db_path: Path):
    context = make_ctx(db_path, host="0.0.0.0")
    try:
        client = create_app(context).test_client()
        client.set_cookie(COOKIE_NAME, TOKEN, domain="phone.lan")
        assert get(client, "/api/state", host="phone.lan:8000").status_code == 200
        assert get(create_app(context).test_client(), "/api/state", host="phone.lan:8000").status_code == 401
    finally:
        context.session.close()


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


def test_new_token_is_random_and_url_safe():
    first, second = new_token(), new_token()
    assert first != second
    assert len(first) >= 32
    assert all(c.isalnum() or c in "-_" for c in first)


def test_token_cookie_flow(ctx: WebContext):
    client = client_for(ctx, authenticated=False)
    for path in ("/", "/api/state"):
        response = get(client, path)
        assert response.status_code == 401
        assert response.get_json() == {"error": t("web.unauthorized")}
    assert post(client, "/api/command", {"name": "help"}).status_code == 401

    wrong = get(client, "/?token=wrong")
    assert wrong.status_code == 401
    assert "Set-Cookie" not in wrong.headers

    login = get(client, f"/?token={TOKEN}")
    assert login.status_code == 302
    assert login.headers["Location"] == "/"
    assert get(client, "/").status_code == 200
    assert get(client, "/api/state").status_code == 200
    assert command(client, "help").status_code == 200


def test_cookie_flags(ctx: WebContext):
    login = get(client_for(ctx, authenticated=False), f"/?token={TOKEN}")
    (cookie,) = login.headers.getlist("Set-Cookie")
    attributes = [part.strip() for part in cookie.split(";")]
    assert attributes[0] == f"{COOKIE_NAME}={TOKEN}"
    assert "HttpOnly" in attributes
    assert "SameSite=Strict" in attributes
    assert "Path=/" in attributes
    assert "Secure" not in attributes  # plain HTTP: a Secure cookie would never be sent back


def test_wrong_cookie_is_401(ctx: WebContext):
    client = client_for(ctx, authenticated=False)
    client.set_cookie(COOKIE_NAME, TOKEN + "x", domain="127.0.0.1")
    assert get(client, "/api/state").status_code == 401
    client.set_cookie(COOKIE_NAME, "ä-not-ascii", domain="127.0.0.1")
    assert get(client, "/api/state").status_code == 401


def test_query_token_only_logs_in_on_the_page(ctx: WebContext):
    client = client_for(ctx, authenticated=False)
    response = get(client, f"/api/state?token={TOKEN}")
    assert response.status_code == 401
    assert "Set-Cookie" not in response.headers
    assert post(client, f"/api/command?token={TOKEN}", {"name": "help"}).status_code == 401


def test_unknown_route_needs_auth(ctx: WebContext):
    assert get(client_for(ctx, authenticated=False), "/nope").status_code == 401
    response = get(client_for(ctx), "/nope")
    assert response.status_code == 404
    assert response.get_json() == {"error": t("web.not_found")}
    wrong_method = get(client_for(ctx), "/api/command")
    assert wrong_method.status_code == 405
    assert wrong_method.get_json() == {"error": t("web.bad_request")}


@pytest.mark.parametrize(
    ("status", "key"),
    [
        (400, "web.bad_request"),
        (404, "web.not_found"),
        (405, "web.bad_request"),
        (413, "web.too_large"),
        (500, "web.internal_error"),
        (503, "web.internal_error"),
    ],
)
def test_http_error_text(status: int, key: str):
    assert web._http_error_text(status) == t(key)


# ---------------------------------------------------------------------------
# Content type, size, headers
# ---------------------------------------------------------------------------


def test_post_requires_json(ctx: WebContext):
    client = client_for(ctx)
    for content_type in ("text/plain", "application/x-www-form-urlencoded", None):
        kwargs = {} if content_type is None else {"content_type": content_type}
        response = post(client, "/api/command", data='{"name": "help"}', **kwargs)
        assert response.status_code == 415
        assert response.get_json() == {"error": t("web.unsupported_media")}
    ok = post(client, "/api/command", data='{"name": "help"}', content_type="application/json; charset=utf-8")
    assert ok.status_code == 200
    # Order of checks: an unauthenticated non-JSON POST is a 401, not a 415.
    assert post(client_for(ctx, authenticated=False), "/api/command", data="x").status_code == 401


def test_oversized_body_is_413(ctx: WebContext):
    response = post(client_for(ctx), "/api/command", {"name": "help", "pad": "x" * (64 * 1024)})
    assert response.status_code == 413
    assert response.get_json() == {"error": t("web.too_large")}
    assert not ctx.lock.locked()


def test_cache_control_no_store_everywhere(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    client = client_for(ctx)
    responses = [
        get(client, "/api/state"),
        get(client, "/"),
        command(client, "help"),
        get(client_for(ctx, authenticated=False), f"/?token={TOKEN}"),
        get(client_for(ctx, authenticated=False), "/api/state"),
        get(client, "/api/state", host="evil.example:8000"),
        post(client, "/api/command", data="x", content_type="text/plain"),
        command(client, "nope"),
        post(client, "/api/command", {"name": "help", "pad": "x" * (64 * 1024)}),
    ]
    monkeypatch.setattr(ctx.session, "snapshot", _boom)
    responses.append(command(client, "summary"))
    assert [r.status_code for r in responses] == [200, 200, 200, 302, 401, 400, 415, 400, 413, 500]
    for response in responses:
        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["Referrer-Policy"] == "no-referrer"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_state_shape_and_language(ctx: WebContext, use_language: UseLanguage):
    client = client_for(ctx)
    state = get(client, "/api/state").get_json()
    assert set(state) == STATE_KEYS
    assert state["language"] == "pt-BR"
    assert state["strings"]["web.ui.send"] == "Enviar"
    use_language("en")
    state = get(client, "/api/state").get_json()
    assert state["language"] == "en"
    assert state["provider"] == "ollama"
    assert state["model"] == "fake-model"
    assert state["busy"] is False
    assert state["transcript"] == []
    assert state["last_sync"] is None
    assert state["counts"] == ctx.session.store.counts()
    assert state["counts"]["transactions"] > 0
    strings = state["strings"]
    assert strings
    assert all(key.startswith("web.ui.") for key in strings)
    assert strings == {key: t(key) for key in strings}
    assert strings["web.ui.send"] == "Send"
    assert {"web.ui.title", "web.ui.help", "web.ui.summary", "web.ui.sync", "web.ui.reset"} <= set(strings)


def test_state_reports_last_sync_and_transcript(db_path: Path):
    with FinanceStore(db_path) as store:
        store.set_meta("last_sync", "2026-11-22T10:00:00+00:00")
    context = make_ctx(db_path)
    try:
        context.transcript.append({"role": "user", "text": "oi"})
        state = get(client_for(context), "/api/state").get_json()
        assert state["last_sync"] == "2026-11-22T10:00:00+00:00"
        assert state["transcript"] == [{"role": "user", "text": "oi"}]
    finally:
        context.session.close()


class _Untouchable:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"/api/state touched the store ({name})")


def test_state_never_touches_the_store_and_works_while_busy(ctx: WebContext):
    client = client_for(ctx)
    expected = get(client, "/api/state").get_json()["counts"]
    real_store = ctx.session.store
    ctx.session.store = _Untouchable()
    ctx.lock.acquire()
    try:
        state = get(client, "/api/state").get_json()
    finally:
        ctx.lock.release()
        ctx.session.store = real_store
    assert state["busy"] is True
    assert state["counts"] == expected


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_command_routes(ctx: WebContext):
    client = client_for(ctx)
    assert command(client, "help").get_json() == {"notices": [{"level": "info", "text": t("chat.help")}]}
    assert command(client, "summary").get_json() == {"notices": [{"level": "info", "text": ctx.session.snapshot()}]}
    assert command(client, "history").get_json() == {"notices": [{"level": "info", "text": t("chat.history_empty")}]}
    list(ctx.assistant.ask("Quanto gastei?"))
    history = command(client, "history").get_json()["notices"]
    assert [notice["text"] for notice in history] == ["1. [USER] Quanto gastei?", "2. [ASSISTANT] …"]
    ctx.transcript.extend([{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}])
    assert command(client, "reset").get_json() == {"notices": [{"level": "info", "text": t("chat.reset")}]}
    assert ctx.transcript == []
    assert ctx.assistant.chat.get_history() == []
    assert get(client, "/api/state").get_json()["transcript"] == []
    assert not ctx.lock.locked()


@pytest.mark.parametrize(
    "body",
    [
        {"name": "sync"},
        {"name": "exit"},
        {"name": "nope"},
        {"name": "/help"},
        {"name": ["help"]},
        {"name": None},
        {},
        ["help"],
        "help",
    ],
)
def test_unknown_command_is_400(ctx: WebContext, body: Any):
    response = post(client_for(ctx), "/api/command", body)
    assert response.status_code == 400
    assert response.get_json() == {"error": t("web.bad_request")}


def test_malformed_json_is_400(ctx: WebContext):
    response = post(client_for(ctx), "/api/command", data="{not json", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json() == {"error": t("web.bad_request")}


def test_busy_returns_409(ctx: WebContext, use_language: UseLanguage):
    use_language("en")
    client = client_for(ctx)
    ctx.lock.acquire()
    try:
        response = command(client, "help")
    finally:
        ctx.lock.release()
    assert response.status_code == 409
    assert response.get_json() == {"error": t("web.busy")}
    assert response.get_json()["error"].startswith("The assistant is busy")
    assert command(client, "help").status_code == 200


def _boom() -> str:
    raise RuntimeError("database exploded")


def test_exception_in_route_is_json_500_and_frees_lock(
    ctx: WebContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(ctx.session, "snapshot", _boom)
    client = client_for(ctx)
    with caplog.at_level(logging.ERROR, logger="llmopenfinance.web"):
        response = command(client, "summary")
    assert response.status_code == 500
    assert response.get_json() == {"error": t("web.internal_error")}
    assert not ctx.lock.locked()
    assert any(record.exc_info for record in caplog.records)
    assert command(client, "help").status_code == 200


def test_sync_callback_is_never_used_by_commands():
    with pytest.raises(RuntimeError, match="/api/sync"):
        web._no_sync()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def test_chat_streams_text_tool_and_done(db_path: Path):
    call = ToolCall("c1", "detect_subscriptions", {"months": 6})
    context = make_ctx(db_path, turns=[tool_turn(call), [TextDelta("Você "), TextDelta("assina"), TurnEnd("end_turn")]])
    try:
        response = chat(client_for(context), "Quais assinaturas?")
        assert response.headers["Cache-Control"] == "no-store"
        assert events(response) == [
            {"type": "tool", "name": "detect_subscriptions", "arguments": {"months": 6}},
            {"type": "text", "text": "Você "},
            {"type": "text", "text": "assina"},
            {"type": "done", "ok": True},
        ]
        assert not context.lock.locked()
        assert context.assistant.chat.tool_results[0].call_id == "c1"
    finally:
        context.session.close()


def test_chat_error_event_and_history_rolled_back(db_path: Path):
    call = ToolCall("c", "get_accounts", {})
    context = make_ctx(db_path, turns=[tool_turn(call), LLMError("API fora do ar")])
    try:
        before = history_length(context)
        assert events(chat(client_for(context), "saldo?")) == [
            {"type": "tool", "name": "get_accounts", "arguments": {}},
            {"type": "error", "text": "API fora do ar"},
            {"type": "done", "ok": False},
        ]
        assert history_length(context) == before
        assert not context.lock.locked()
    finally:
        context.session.close()


def test_chat_stop_reason_error_is_not_ok(db_path: Path):
    context = make_ctx(db_path, turns=[[TextDelta("parcial"), TurnEnd("max_tokens")]])
    try:
        assert events(chat(client_for(context), "x")) == [
            {"type": "text", "text": "parcial"},
            {"type": "error", "text": t("agent.stop.max_tokens")},
            {"type": "done", "ok": False},
        ]
        assert context.transcript == []
    finally:
        context.session.close()


def test_chat_busy_409_does_not_queue(ctx: WebContext, use_language: UseLanguage):
    use_language("en")
    client = client_for(ctx)
    ctx.lock.acquire()
    try:
        response = chat(client, "Quanto gastei?")
        assert response.status_code == 409
        assert response.get_json() == {"error": t("web.busy")}
        assert ctx.lock.locked()  # the 409 did not release the holder's lock
    finally:
        ctx.lock.release()
    assert history_length(ctx) == 0
    assert len(ctx.assistant.chat.turns) == 1  # the scripted answer was not consumed: nothing queued
    assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}


def test_chat_slash_command_in_box(ctx: WebContext):
    client = client_for(ctx)
    assert events(chat(client, "/ajuda")) == [
        {"type": "notice", "level": "info", "text": t("chat.help")},
        {"type": "done", "ok": True},
    ]
    assert events(chat(client, "/resumo"))[0] == {"type": "notice", "level": "info", "text": ctx.session.snapshot()}
    assert events(chat(client, " /HISTORICO "))[0]["text"] == t("chat.history_empty")
    assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}
    assert len(ctx.transcript) == 2
    assert events(chat(client, "/reset")) == [
        {"type": "notice", "level": "info", "text": t("chat.reset")},
        {"type": "done", "ok": True},
    ]
    assert ctx.transcript == []
    assert ctx.assistant.chat.get_history() == []
    assert not ctx.lock.locked()


def test_exit_in_box_returns_hint_and_keeps_serving(ctx: WebContext, use_language: UseLanguage):
    client = client_for(ctx)
    for message in ("/sair", "sair", "exit"):
        assert events(chat(client, message)) == [
            {"type": "notice", "level": "dim", "text": t("web.exit_hint")},
            {"type": "done", "ok": True},
        ]
    use_language("en")
    hint = events(chat(client, "/quit"))[0]["text"]
    assert hint == t("web.exit_hint")
    assert "Ctrl+C" in hint
    assert get(client, "/api/state").status_code == 200
    assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}
    assert not ctx.lock.locked()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": ""},
        {"message": "   \n\t"},
        {"message": None},
        {"message": 42},
        {"message": ["oi"]},
        {"msg": "oi"},
        ["oi"],
        "oi",
    ],
)
def test_chat_empty_message_400(ctx: WebContext, body: Any):
    response = post(client_for(ctx), "/api/chat", body)
    assert response.status_code == 400
    assert response.get_json() == {"error": t("web.bad_request")}
    assert not ctx.lock.locked()
    assert history_length(ctx) == 0


def test_chat_requires_auth_and_json(ctx: WebContext):
    assert chat(client_for(ctx, authenticated=False), "oi").status_code == 401
    client = client_for(ctx)
    response = post(client, "/api/chat", data='{"message": "oi"}', content_type="text/plain")
    assert response.status_code == 415
    assert chat(client, "oi", host="evil.example:8000").status_code == 400
    assert post(client, "/api/chat", data="{not json", content_type="application/json").status_code == 400
    assert get(client, "/api/chat").status_code == 405
    assert history_length(ctx) == 0
    assert not ctx.lock.locked()


def test_transcript_in_state_after_answer(db_path: Path):
    turns = [[TextDelta("Res"), TextDelta("posta"), TurnEnd("end_turn")], LLMError("caiu")]
    context = make_ctx(db_path, turns=turns)
    try:
        client = client_for(context)
        events(chat(client, "  Quanto gastei?  "))
        expected = [{"role": "user", "text": "Quanto gastei?"}, {"role": "assistant", "text": "Resposta"}]
        assert get(client, "/api/state").get_json()["transcript"] == expected
        assert events(chat(client, "E agora?"))[-1] == {"type": "done", "ok": False}
        assert get(client, "/api/state").get_json()["transcript"] == expected  # nothing added after an error
    finally:
        context.session.close()


def test_chat_non_ascii_round_trips(db_path: Path):
    text = "Você gastou R$ 1.234,56 em Alimentação ✓ 🎉 \N{LINE SEPARATOR} fim"
    context = make_ctx(db_path, turns=[answer(text)])
    try:
        response = chat(client_for(context), "Quanto gastei em alimentação?")
        assert "Alimentação ✓ 🎉".encode() in response.data  # raw UTF-8, not \\u escapes
        assert events(response)[0] == {"type": "text", "text": text}
        assert context.transcript == [
            {"role": "user", "text": "Quanto gastei em alimentação?"},
            {"role": "assistant", "text": text},
        ]
    finally:
        context.session.close()


def test_aborted_stream_releases_lock_and_rolls_back(db_path: Path):
    call = ToolCall("c", "get_accounts", {})
    context = make_ctx(db_path, turns=[tool_turn(call), answer("never")])
    try:
        before = history_length(context)
        response = chat(client_for(context), "saldo?", buffered=False)
        first = json.loads(next(iter(response.response)))
        assert first == {"type": "tool", "name": "get_accounts", "arguments": {}}
        assert context.lock.locked()
        assert history_length(context) > before  # the turn is in progress
        response.close()  # the client went away mid-stream
        assert context.lock.acquire(blocking=False)
        context.lock.release()
        assert history_length(context) == before
        assert context.transcript == []
        assert context.assistant.chat.compacted == 0
    finally:
        context.session.close()


def test_never_iterated_response_releases_lock(ctx: WebContext):
    # Werkzeug's test client reads the first chunk even with ``buffered=False``, so call the WSGI app directly,
    # as a server would, and close its iterable without reading it.
    environ = EnvironBuilder(
        path="/api/chat",
        method="POST",
        json={"message": "Quanto gastei?"},
        headers=[("Host", HOST), ("Cookie", f"{COOKIE_NAME}={TOKEN}")],
    ).get_environ()
    statuses: list[str] = []

    def start_response(status: str, _headers: list[tuple[str, str]], _exc_info: Any = None) -> Callable[[bytes], Any]:
        statuses.append(status)
        return lambda _data: None

    body = create_app(ctx)(environ, start_response)
    assert isinstance(body, ClosingIterator)
    assert statuses == ["200 OK"]
    assert ctx.lock.locked()  # taken by the handler, before the body is read
    body.close()
    assert not ctx.lock.locked()
    assert history_length(ctx) == 0
    assert len(ctx.assistant.chat.turns) == 1  # the assistant was never asked
    body.close()  # closing again neither raises nor releases someone else's lock
    ctx.lock.acquire()
    try:
        body.close()
        assert ctx.lock.locked()
    finally:
        ctx.lock.release()


def test_lock_released_once_after_complete_stream(ctx: WebContext):
    response = chat(client_for(ctx), "Quanto gastei?", buffered=False)
    assert [json.loads(line) for line in response.response][-1] == {"type": "done", "ok": True}
    assert not ctx.lock.locked()  # the generator's finally released it
    ctx.lock.acquire()
    try:
        response.close()  # call_on_close runs afterwards and must not release the new holder's lock
        assert ctx.lock.locked()
    finally:
        ctx.lock.release()


def test_failure_while_building_the_stream_releases_lock(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    def broken(_stream: Any) -> Any:
        raise RuntimeError("cannot build the response")

    monkeypatch.setattr(web_stream, "stream_with_context", broken)
    response = chat(client_for(ctx), "Quanto gastei?")
    assert response.status_code == 500
    assert response.get_json() == {"error": t("web.internal_error")}
    assert not ctx.lock.locked()


class _ExplodingAssistant:
    """Stand-in assistant whose ``ask`` fails with an unexpected exception (before or after a first event)."""

    def __init__(self, *, mid_stream: bool) -> None:
        self.mid_stream = mid_stream

    def ask(self, question: str) -> Iterator[TextDelta]:
        if not self.mid_stream:
            raise RuntimeError("bug before the stream")
        return self._stream()

    def _stream(self) -> Iterator[TextDelta]:
        yield TextDelta("parcial")
        raise RuntimeError("bug mid-stream")


@pytest.mark.parametrize("mid_stream", [False, True])
def test_unexpected_exception_releases_lock(
    ctx: WebContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mid_stream: bool
):
    monkeypatch.setattr(ctx, "assistant", _ExplodingAssistant(mid_stream=mid_stream))
    client = client_for(ctx)
    with caplog.at_level(logging.ERROR, logger="llmopenfinance.web"):
        received = events(chat(client, "Quanto gastei?"))
    assert received[-2:] == [{"type": "error", "text": t("web.internal_error")}, {"type": "done", "ok": False}]
    assert received[:-2] == ([{"type": "text", "text": "parcial"}] if mid_stream else [])
    assert not ctx.lock.locked()
    assert any(record.exc_info for record in caplog.records)
    assert ctx.transcript == []
    assert get(client, "/api/state").status_code == 200


def test_unexpected_exception_in_turn_rolls_back_history(db_path: Path):
    call = ToolCall("c", "get_accounts", {})
    context = make_ctx(db_path, turns=[tool_turn(call), RuntimeError("bug"), answer("depois")])
    try:
        client = client_for(context)
        assert events(chat(client, "saldo?"))[-2:] == [
            {"type": "error", "text": t("web.internal_error")},
            {"type": "done", "ok": False},
        ]
        assert history_length(context) == 0
        assert not context.lock.locked()
        assert events(chat(client, "e agora?")) == [{"type": "text", "text": "depois"}, {"type": "done", "ok": True}]
    finally:
        context.session.close()


class _CleanupRaisingAssistant:
    """Stand-in assistant whose answer fails in its own cleanup when the stream is closed early."""

    def ask(self, question: str) -> Iterator[TextDelta]:
        try:
            yield TextDelta("parcial")
            yield TextDelta("resto")
        finally:
            raise RuntimeError("cleanup failed")


def test_abort_with_raising_cleanup_does_not_yield_after_generator_exit(
    ctx: WebContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(ctx, "assistant", _CleanupRaisingAssistant())
    response = chat(client_for(ctx), "Quanto gastei?", buffered=False)
    assert json.loads(next(iter(response.response))) == {"type": "text", "text": "parcial"}
    with caplog.at_level(logging.ERROR, logger="llmopenfinance.web"):
        response.close()  # must not raise "generator ignored GeneratorExit"
    assert not ctx.lock.locked()
    assert any(record.exc_info for record in caplog.records)


def _raise_in_hook(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("hook failed")


@pytest.mark.parametrize("hook", ["after_request", "request_finished"])
@pytest.mark.parametrize("path", ["/api/chat", "/api/sync"])
def test_raising_response_hook_does_not_leak_the_lock(
    ctx: WebContext, monkeypatch: pytest.MonkeyPatch, hook: str, path: str
):
    fake = RecordingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    app = create_app(ctx)
    client = app.test_client()
    client.set_cookie(COOKIE_NAME, TOKEN, domain="127.0.0.1")
    before = ctx.session.store
    if hook == "after_request":
        app.after_request(_raise_in_hook)
        response = post(client, path, {"message": "Quanto gastei?"})
    else:
        with request_finished.connected_to(_raise_in_hook, app):
            response = post(client, path, {"message": "Quanto gastei?"})
    assert response.status_code == 500
    assert not ctx.lock.locked()
    assert fake.forces == []  # the stream was never read, so no sync started
    assert ctx.session.store is before
    assert history_length(ctx) == 0


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


@pytest.fixture
def pluggy_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PLUGGY_CLIENT_ID, "cid")
    monkeypatch.setenv(ENV_PLUGGY_CLIENT_SECRET, "secret")


def changed_pluggy() -> FakePluggy:
    """A fake Pluggy whose item changed since ``db_path`` was synced: one more bank transaction."""
    pluggy = FakePluggy()
    pluggy.items[ITEM_ID]["lastUpdatedAt"] = "2026-11-22T12:00:00.000Z"
    pluggy.transactions[BANK_ID].append(
        {
            "id": "tv-nova",
            "date": "2026-11-20T00:00:00.000Z",
            "description": "LOJA DE TV",
            "amount": -4321.0,
            "type": "DEBIT",
            "status": "POSTED",
            "currencyCode": "BRL",
        }
    )
    return pluggy


def use_pluggy(ctx: WebContext, pluggy: FakePluggy) -> None:
    ctx.factories = replace(ctx.factories, pluggy_client=lambda: pluggy)


def sync(client: FlaskClient, body: Any = None, **kwargs: Any) -> Any:
    if body is None:
        kwargs.setdefault("content_type", "application/json")
    return post(client, "/api/sync", body, **kwargs)


class RecordingSync:
    """Stand-in for ``session.run_sync``: records ``force``, emits one notice and one item, returns ``code``."""

    def __init__(self, code: int = 0, *, during: Callable[[], None] | None = None) -> None:
        self.code = code
        self.during = during
        self.forces: list[bool] = []

    def __call__(
        self,
        factories: Factories,
        *,
        force: bool,
        notify: Callable[[Notice], None],
        on_item: Callable[[ItemResult], None],
    ) -> int:
        self.forces.append(force)
        notify(Notice("info", "início"))
        if self.during is not None:
            self.during()
        on_item(ItemResult("item-x", ok=True, counts={"transactions": 3}, warnings=("aviso",)))
        return self.code


RECORDED_EVENTS = [
    {"type": "notice", "level": "info", "text": "início"},
    {
        "type": "progress",
        "item_id": "item-x",
        "ok": True,
        "skipped": False,
        "counts": {"transactions": 3},
        "warnings": ["aviso"],
        "error": None,
    },
]


class BlockingSync(RecordingSync):
    """A sync that, after its first notice, waits until the test sets :attr:`proceed`."""

    def __init__(self) -> None:
        super().__init__(during=self._block)
        self.started = threading.Event()
        self.proceed = threading.Event()

    def _block(self) -> None:
        self.started.set()
        assert self.proceed.wait(10), "the test never let the sync finish"


class Streamer(threading.Thread):
    """Reads a whole streaming response in its own thread, as a WSGI server thread would."""

    def __init__(self, send: Callable[[], Any]) -> None:
        super().__init__(daemon=True)
        self.send = send
        self.received: list[dict[str, Any]] = []
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.received = events(self.send())
        except BaseException as exc:
            self.error = exc

    def finish(self) -> list[dict[str, Any]]:
        self.join(timeout=10)
        assert not self.is_alive()
        assert self.error is None
        return self.received


@pytest.mark.usefixtures("pluggy_credentials")
def test_sync_streams_progress_and_done(ctx: WebContext):
    pluggy = changed_pluggy()
    use_pluggy(ctx, pluggy)
    response = sync(client_for(ctx))
    assert response.headers["Cache-Control"] == "no-store"
    received = events(response)
    assert received[0] == {"type": "notice", "level": "info", "text": t("cli.sync.start")}
    progress = received[1]
    assert set(progress) == {"type", "item_id", "ok", "skipped", "counts", "warnings", "error"}
    assert progress["type"] == "progress"
    assert progress["item_id"] == ITEM_ID
    assert progress["ok"] is True
    assert progress["skipped"] is False
    assert progress["counts"]["transactions"] > 0
    assert isinstance(progress["warnings"], list)
    assert progress["error"] is None
    assert received[2:] == [
        {"type": "notice", "level": "info", "text": t("cli.sync.done", ok=1, failed=0)},
        {"type": "done", "ok": True},
    ]
    assert pluggy.closed
    assert not ctx.lock.locked()
    assert ctx.session.store.counts()["transactions"] > 0  # the read-only store is open again


@pytest.mark.usefixtures("pluggy_credentials")
def test_sync_refreshes_assistant_and_status(db_path: Path):
    context = make_ctx(db_path, turns=[answer("Resposta")], pluggy=changed_pluggy())
    try:
        client = client_for(context)
        before = get(client, "/api/state").get_json()
        assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}
        assert history_length(context) > 0
        prompt_before = context.assistant.chat.system_message
        assert events(sync(client))[-1] == {"type": "done", "ok": True}
        after = get(client, "/api/state").get_json()
        assert after["counts"]["transactions"] == before["counts"]["transactions"] + 1
        assert before["last_sync"] is None
        assert after["last_sync"] is not None
        assert after["busy"] is False
        assert history_length(context) == 0  # the conversation was reset
        assert context.assistant.chat.system_message != prompt_before
        assert context.assistant.chat.system_message == build_system_prompt(context.session.snapshot())
    finally:
        context.session.close()


def test_sync_clears_transcript(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    client = client_for(ctx)
    assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}
    assert len(ctx.transcript) == 2
    # A failed sync resets the conversation too (as the CLI's /sync does), so the transcript goes with it.
    monkeypatch.setattr(web, "run_sync", RecordingSync(1))
    assert events(sync(client)) == [*RECORDED_EVENTS, {"type": "done", "ok": False}]
    assert ctx.transcript == []
    assert history_length(ctx) == 0
    ctx.transcript.extend([{"role": "user", "text": "q"}, {"role": "assistant", "text": "a"}])
    monkeypatch.setattr(web, "run_sync", RecordingSync(0))
    assert events(sync(client)) == [*RECORDED_EVENTS, {"type": "done", "ok": True}]
    assert ctx.transcript == []
    assert get(client, "/api/state").get_json()["transcript"] == []


def test_sync_busy_409_when_chat_holds_lock(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    fake = RecordingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    started, proceed = threading.Event(), threading.Event()

    def slow_answer(question: str) -> Iterator[TextDelta]:
        started.set()
        assert proceed.wait(10)
        yield TextDelta("ok")

    monkeypatch.setattr(ctx.assistant, "ask", slow_answer)
    streamer = Streamer(lambda: chat(client_for(ctx), "Quanto gastei?"))
    streamer.start()
    try:
        assert started.wait(10)
        response = sync(client_for(ctx), {"force": True})
        assert response.status_code == 409
        assert response.get_json() == {"error": t("web.busy")}
        assert fake.forces == []  # nothing queued
    finally:
        proceed.set()
    assert streamer.finish()[-1] == {"type": "done", "ok": True}
    assert not ctx.lock.locked()
    assert events(sync(client_for(ctx)))[-1] == {"type": "done", "ok": True}
    assert fake.forces == [False]


def test_chat_and_commands_409_while_sync_runs(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    fake = BlockingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    client = client_for(ctx)
    cached = get(client, "/api/state").get_json()["counts"]
    streamer = Streamer(lambda: sync(client_for(ctx)))
    streamer.start()
    try:
        assert fake.started.wait(10)
        assert any(thread.name == SYNC_THREAD_NAME for thread in threading.enumerate())  # the leak check sees it
        for response in (
            chat(client, "Quanto gastei?"),
            command(client, "summary"),
            sync(client),
            chat(client, "/sync"),
        ):
            assert response.status_code == 409
            assert response.get_json() == {"error": t("web.busy")}
        state = get(client, "/api/state").get_json()  # the store is closed now: state must not touch it
        assert state["busy"] is True
        assert state["counts"] == cached
    finally:
        fake.proceed.set()
    assert streamer.finish() == [*RECORDED_EVENTS, {"type": "done", "ok": True}]
    assert fake.forces == [False]
    assert len(ctx.assistant.chat.turns) == 1  # the chat request was refused, not queued
    assert not ctx.lock.locked()
    assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}


def test_sync_without_credentials_keeps_server_usable(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_PLUGGY_CLIENT_ID, raising=False)
    monkeypatch.delenv(ENV_PLUGGY_CLIENT_SECRET, raising=False)
    client = client_for(ctx)
    before = ctx.session.store
    text = t("cli.sync.no_credentials", id_var=ENV_PLUGGY_CLIENT_ID, secret_var=ENV_PLUGGY_CLIENT_SECRET)
    assert events(sync(client)) == [{"type": "notice", "level": "error", "text": text}, {"type": "done", "ok": False}]
    assert not ctx.lock.locked()
    assert ctx.session.store is not before  # closed for the sync, then reopened read-only
    assert ctx.session.store.counts()["transactions"] > 0
    assert command(client, "summary").get_json() == {"notices": [{"level": "info", "text": ctx.session.snapshot()}]}
    assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}


@pytest.mark.usefixtures("pluggy_credentials")
def test_sync_database_locked_keeps_server_usable(ctx: WebContext, db_path: Path, use_language: UseLanguage):
    use_language("en")
    read_only_opener = store_opener(db_path)
    opened: list[bool] = []

    def open_store(*, read_only: bool = False) -> FinanceStore:
        opened.append(read_only)
        if not read_only:
            raise duckdb.IOException("Could not set lock on file")
        return read_only_opener(read_only=True)

    ctx.factories = replace(ctx.factories, open_store=open_store)
    client = client_for(ctx)
    text = t("cli.sync.db_locked", error="Could not set lock on file")
    assert events(sync(client)) == [{"type": "notice", "level": "warn", "text": text}, {"type": "done", "ok": False}]
    assert opened == [False]
    assert not ctx.lock.locked()
    assert ctx.session.store.counts()["transactions"] > 0
    assert command(client, "summary").status_code == 200
    assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}


def test_sync_read_only_handle_is_closed_during_sync(ctx: WebContext, db_path: Path, monkeypatch: pytest.MonkeyPatch):
    transactions: list[int] = []

    def write_while_syncing() -> None:
        # DuckDB refuses a read-write handle while this process still holds a read-only one on the same file.
        with FinanceStore(db_path) as store:
            transactions.append(store.counts()["transactions"])

    monkeypatch.setattr(web, "run_sync", RecordingSync(during=write_while_syncing))
    assert events(sync(client_for(ctx))) == [*RECORDED_EVENTS, {"type": "done", "ok": True}]
    assert len(transactions) == 1
    assert transactions[0] > 0
    assert ctx.session.store.counts()["transactions"] == transactions[0]


def test_sync_slash_command_in_chat_box(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    fake = RecordingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    client = client_for(ctx)
    ctx.transcript.append({"role": "user", "text": "q"})
    assert events(chat(client, "/sync")) == [*RECORDED_EVENTS, {"type": "done", "ok": True}]
    assert events(chat(client, " /SYNC "))[-1] == {"type": "done", "ok": True}
    assert fake.forces == [False, False]
    assert ctx.transcript == []
    assert not ctx.lock.locked()
    assert ctx.session.store.counts()["transactions"] > 0
    monkeypatch.setattr(web, "run_sync", RecordingSync(1))
    assert events(chat(client, "/sync"))[-1] == {"type": "done", "ok": False}


def test_sync_client_disconnect_waits_for_worker_and_reopens_store(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    fake = BlockingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    ctx.transcript.append({"role": "user", "text": "q"})
    before = ctx.session.store
    first_read = threading.Event()
    outcome: dict[str, Any] = {}

    def disconnecting_client() -> None:
        # Read and close in the same thread, as a WSGI server does (the stream holds that thread's request context).
        try:
            response = sync(client_for(ctx), buffered=False)
            outcome["first"] = json.loads(next(iter(response.response)))
            first_read.set()
            response.close()  # the client went away; this waits for the sync to finish
            outcome["closed"] = True
        except BaseException as exc:
            outcome["error"] = exc
            first_read.set()

    client_thread = threading.Thread(target=disconnecting_client, daemon=True)
    client_thread.start()
    try:
        assert first_read.wait(10)
        assert outcome.get("first") == RECORDED_EVENTS[0]
        assert fake.started.wait(10)
        client_thread.join(timeout=0.5)
        assert client_thread.is_alive()  # close() is still waiting for the worker
        assert ctx.lock.locked()  # the abort did not give the lock back early
        assert get(client_for(ctx), "/api/state").get_json()["busy"] is True
        assert sync(client_for(ctx)).status_code == 409
    finally:
        fake.proceed.set()
    client_thread.join(timeout=10)
    assert not client_thread.is_alive()
    assert "error" not in outcome
    assert outcome["closed"] is True
    assert not ctx.lock.locked()
    assert ctx.session.store is not before
    assert ctx.session.store.counts()["transactions"] > 0  # reopened read-only
    assert ctx.transcript == []  # the post-sync refresh ran even though nobody was listening
    assert events(chat(client_for(ctx), "Quanto gastei?"))[-1] == {"type": "done", "ok": True}


def test_never_iterated_sync_response_never_starts_the_sync(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    fake = RecordingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    environ = EnvironBuilder(
        path="/api/sync",
        method="POST",
        json={"force": True},
        headers=[("Host", HOST), ("Cookie", f"{COOKIE_NAME}={TOKEN}")],
    ).get_environ()
    before = ctx.session.store
    body = create_app(ctx)(environ, lambda _status, _headers, _exc_info=None: lambda _data: None)
    assert isinstance(body, ClosingIterator)
    assert ctx.lock.locked()
    body.close()
    assert not ctx.lock.locked()
    assert fake.forces == []
    assert ctx.session.store is before
    assert before.counts()["transactions"] > 0


def test_sync_unexpected_exception_keeps_server_usable(
    ctx: WebContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    def broken_sync(*_args: Any, **_kwargs: Any) -> int:
        raise RuntimeError("sync bug")

    monkeypatch.setattr(web, "run_sync", broken_sync)
    client = client_for(ctx)
    before = ctx.session.store
    with caplog.at_level(logging.ERROR, logger="llmopenfinance.web"):
        received = events(sync(client))
    assert received == [{"type": "error", "text": t("web.internal_error")}, {"type": "done", "ok": False}]
    assert any(record.exc_info for record in caplog.records)
    assert not ctx.lock.locked()
    assert ctx.session.store is not before
    assert ctx.session.store.counts()["transactions"] > 0
    assert events(chat(client, "Quanto gastei?"))[-1] == {"type": "done", "ok": True}


def test_sync_failure_after_the_sync_reports_error_and_frees_lock(
    ctx: WebContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.setattr(web, "run_sync", RecordingSync())
    monkeypatch.setattr(ctx.assistant, "refresh", _boom)
    with caplog.at_level(logging.ERROR, logger="llmopenfinance.web"):
        received = events(sync(client_for(ctx)))
    assert received == [
        *RECORDED_EVENTS,
        {"type": "error", "text": t("web.internal_error")},
        {"type": "done", "ok": False},
    ]
    assert any(record.exc_info for record in caplog.records)
    assert not ctx.lock.locked()
    assert ctx.session.store.counts()["transactions"] > 0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"data": ""}, False),  # no body at all
        ({"data": "null"}, False),
        ({"data": "{}"}, False),
        ({"data": '{"force": false}'}, False),
        ({"data": '{"force": true}'}, True),
        ({"data": '{"force": true, "other": 1}'}, True),
    ],
)
def test_sync_force_reaches_run_sync(ctx: WebContext, monkeypatch: pytest.MonkeyPatch, kwargs: Any, expected: bool):
    fake = RecordingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    response = post(client_for(ctx), "/api/sync", content_type="application/json", **kwargs)
    assert events(response)[-1] == {"type": "done", "ok": True}
    assert fake.forces == [expected]


@pytest.mark.parametrize("data", ['{"force": "yes"}', '{"force": 1}', '{"force": null}', "[]", '"force"', "true"])
def test_sync_force_validation(ctx: WebContext, monkeypatch: pytest.MonkeyPatch, data: str):
    fake = RecordingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    response = post(client_for(ctx), "/api/sync", data=data, content_type="application/json")
    assert response.status_code == 400
    assert response.get_json() == {"error": t("web.bad_request")}
    assert fake.forces == []
    assert not ctx.lock.locked()


def test_sync_requires_auth_and_json(ctx: WebContext, monkeypatch: pytest.MonkeyPatch):
    fake = RecordingSync()
    monkeypatch.setattr(web, "run_sync", fake)
    before = ctx.session.store
    assert sync(client_for(ctx, authenticated=False)).status_code == 401
    client = client_for(ctx)
    assert post(client, "/api/sync", data="{}", content_type="text/plain").status_code == 415
    assert post(client, "/api/sync", data="{}").status_code == 415  # no content type at all
    assert sync(client, {}, host="evil.example:8000").status_code == 400
    malformed = post(client, "/api/sync", data="{not json", content_type="application/json")
    assert malformed.status_code == 400
    assert malformed.get_json() == {"error": t("web.bad_request")}
    assert sync(client, {"pad": "x" * (64 * 1024)}).status_code == 413
    assert get(client, "/api/sync").status_code == 405
    assert fake.forces == []
    assert ctx.session.store is before
    assert not ctx.lock.locked()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

NONCE_ATTR = re.compile(r'\snonce="([^"]*)"')
STRINGS_BLOCK = re.compile(r'<script type="application/json" id="i18n" nonce="[^"]*">(.*?)</script>', re.S)
EXTERNAL_REFERENCE = re.compile(
    r"""(?:\b(?:src|href)\s*=\s*["']?|url\(\s*["']?|@import\s+(?:url\(\s*)?["']?)\s*(?:https?:)?//""", re.I
)
INLINE_HANDLER = re.compile(r"<[^>]*\son[a-z]+\s*=", re.I)
SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.S | re.I)
PAGE_DIRECTIVES = {
    "default-src",
    "script-src",
    "style-src",
    "connect-src",
    "base-uri",
    "form-action",
    "frame-ancestors",
}


def page_response(client: FlaskClient) -> Any:
    response = get(client, "/")
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    return response


def csp_directives(response: Any) -> dict[str, str]:
    directives: dict[str, str] = {}
    for part in response.headers["Content-Security-Policy"].split(";"):
        name, _, value = part.strip().partition(" ")
        directives[name] = value.strip()
    return directives


def page_nonce(response: Any) -> str:
    source = csp_directives(response)["script-src"]
    assert source.startswith("'nonce-")
    assert source.endswith("'")
    return source.removeprefix("'nonce-").removesuffix("'")


def main_script(body: str) -> str:
    match = re.search(r'<script nonce="[^"]*">(.*?)</script>', body, re.S)
    assert match is not None
    return match.group(1)


def test_page_language_and_strings(ctx: WebContext, use_language: UseLanguage):
    client = client_for(ctx)
    body = page_response(client).get_data(as_text=True)
    assert '<html lang="pt-BR">' in body
    assert ">Enviar</button>" in body
    use_language("en")
    body = page_response(client).get_data(as_text=True)
    assert '<html lang="en">' in body
    assert ">Send</button>" in body
    assert "Enviar" not in body
    block = STRINGS_BLOCK.search(body)
    assert block is not None
    assert json.loads(block.group(1)) == web.ui_strings()


def test_page_has_no_external_resources(ctx: WebContext):
    body = page_response(client_for(ctx)).get_data(as_text=True)
    assert not EXTERNAL_REFERENCE.search(body)
    for needle in ("http://", "https://", "@import", "url("):
        assert needle not in body
    assert not re.search(r"<(?:link|iframe|img|object|embed|base)\b", body, re.I)
    assert not re.search(r"<script\b[^>]*\bsrc\s*=", body, re.I)


def test_page_csp_header(ctx: WebContext):
    response = page_response(client_for(ctx))
    header = response.headers["Content-Security-Policy"]
    assert "unsafe-inline" not in header
    assert "unsafe-eval" not in header
    nonce = page_nonce(response)
    assert len(nonce) >= 16
    assert csp_directives(response) == {
        "default-src": "'none'",
        "script-src": f"'nonce-{nonce}'",
        "style-src": f"'nonce-{nonce}'",
        "connect-src": "'self'",
        "base-uri": "'none'",
        "form-action": "'none'",
        "frame-ancestors": "'none'",
    }
    body = response.get_data(as_text=True)
    assert f'<style nonce="{nonce}">' in body
    assert f'<script nonce="{nonce}">' in body
    assert set(NONCE_ATTR.findall(body)) == {nonce}
    assert body.count("<style") == 1
    assert body.count("<script") == body.count(f'nonce="{nonce}">') - 1  # every script carries it (+ the style)


def test_page_requires_auth(ctx: WebContext):
    response = get(client_for(ctx, authenticated=False), "/")
    assert response.status_code == 401
    assert response.get_json() == {"error": t("web.unauthorized")}
    assert "nonce-" not in response.headers.get("Content-Security-Policy", "")
    assert "<html" not in response.get_data(as_text=True)


def test_nonce_differs_per_request(ctx: WebContext):
    client = client_for(ctx)
    first, second = page_response(client), page_response(client)
    assert page_nonce(first) != page_nonce(second)
    assert set(NONCE_ATTR.findall(second.get_data(as_text=True))) == {page_nonce(second)}


def test_hostile_strings_cannot_break_out():
    hostile = "</script><img src=x onerror=1>\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}&amp;'\""
    strings = {**web.ui_strings(), "web.ui.send": hostile, "web.ui.title": hostile, "web.ui.busy": "</SCRIPT >"}
    body = render_page(language='en"><script>alert(1)</script>', strings=strings, nonce='n"><b>')
    assert "<img" not in body
    assert "<b>" not in body
    assert '<html lang="en&quot;&gt;&lt;script&gt;alert(1)&lt;/script&gt;">' in body
    assert body.lower().count("</script") == body.lower().count("<script")  # every closing tag closes one of ours
    block = STRINGS_BLOCK.search(body)
    assert block is not None
    raw = block.group(1)
    for char in ("<", ">", "&", "\N{LINE SEPARATOR}", "\N{PARAGRAPH SEPARATOR}"):
        assert char not in raw
    assert json.loads(raw) == strings
    assert "&lt;/script&gt;&lt;img src=x onerror=1&gt;" in body  # the visible label is escaped text


def test_page_has_no_inline_event_handlers(ctx: WebContext):
    body = page_response(client_for(ctx)).get_data(as_text=True)
    markup = SCRIPT_OR_STYLE.sub("", body)
    assert "<button" in markup  # not vacuous: the markup is still there
    assert not INLINE_HANDLER.search(markup)
    assert not re.search(r"\sstyle\s*=", markup, re.I)  # the CSP blocks inline style attributes too
    assert "javascript:" not in body.lower()


def test_page_script_has_a_single_html_sink():
    script = main_script(render_page(language="pt-BR", strings=web.ui_strings(), nonce="n"))
    assert script.count("innerHTML") == 1
    (sink,) = [line for line in script.splitlines() if "innerHTML" in line]
    assert "= renderMarkdown(" in sink
    for banned in (
        "eval(",
        "Function(",
        "document.write",
        "insertAdjacentHTML",
        "outerHTML",
        ".onclick",
        'setTimeout("',
    ):
        assert banned not in script


def test_page_catalog_keys_exist():
    body = render_page(language="pt-BR", strings={}, nonce="n")
    keys = set(re.findall(r"web\.ui\.[a-z_]+", body))
    assert len(keys) > 10
    assert sorted(key for key in keys if key not in web.ui_strings()) == []


def test_api_responses_carry_a_strict_csp(ctx: WebContext):
    for response in (get(client_for(ctx), "/api/state"), get(client_for(ctx, authenticated=False), "/api/state")):
        directives = csp_directives(response)
        assert directives["default-src"] == "'none'"
        assert directives["frame-ancestors"] == "'none'"
        assert "script-src" not in directives


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def test_token_never_logged(ctx: WebContext, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)
    client = client_for(ctx, authenticated=False)
    get(client, "/?token=wrong-" + TOKEN)
    get(client, "/api/state")
    get(client, f"/?token={TOKEN}", host="evil.example:8000")
    get(client, f"/?token={TOKEN}")
    get(client, "/")
    get(client, "/api/state")
    command(client, "help")
    command(client, "nope")
    post(client, "/api/command", {"name": "help", "pad": TOKEN * 3000})
    monkeypatch.setattr(ctx.session, "snapshot", _boom)
    command(client, "summary")
    assert caplog.records  # the 500 was logged, so the check below is not vacuous
    assert TOKEN not in caplog.text
    for record in caplog.records:
        assert TOKEN not in record.getMessage()
        assert TOKEN not in str(record.args)


def test_logs_do_not_contain_message_text(db_path: Path, caplog: pytest.LogCaptureFixture):
    question, reply = "pergunta-sigilosa-4711", "resposta-sigilosa-0815"
    call = ToolCall("c", "get_accounts", {})
    turns = [answer(reply), tool_turn(call), LLMError("caiu"), RuntimeError("bug")]
    context = make_ctx(db_path, turns=turns)
    caplog.set_level(logging.DEBUG)
    try:
        client = client_for(context)
        assert events(chat(client, question))[-1] == {"type": "done", "ok": True}
        assert events(chat(client, question))[-2] == {"type": "error", "text": "caiu"}
        assert events(chat(client, question))[-2] == {"type": "error", "text": t("web.internal_error")}
    finally:
        context.session.close()
    assert any(record.exc_info for record in caplog.records)  # the unexpected error was logged: not vacuous
    for secret in (question, reply):
        assert secret not in caplog.text
        for record in caplog.records:
            assert secret not in record.getMessage()
            assert secret not in str(record.args)


def test_serve_passes_waitress_limits(db_path: Path):
    context = make_ctx(db_path, host="127.0.0.1")
    seen: list[tuple[Any, dict[str, Any]]] = []
    try:
        web.serve(context, threads=3, serve_fn=lambda app, **kwargs: seen.append((app, kwargs)))
    finally:
        context.session.close()
    assert len(seen) == 1
    app, kwargs = seen[0]
    assert app.name == "llmopenfinance.web"
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8000
    assert kwargs["threads"] == 3
    # Waitress buffers a whole body before the app can answer 413: keep the limit just above the app's own.
    assert web.MAX_CONTENT_LENGTH < kwargs["max_request_body_size"] <= 2 * web.MAX_CONTENT_LENGTH
    assert set(kwargs) == {"host", "port", "threads", "max_request_body_size"}  # no proxy trust, no access logger


def test_serve_defaults_to_waitress(db_path: Path, monkeypatch: pytest.MonkeyPatch):
    import socket

    import waitress

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(waitress, "serve", lambda _app, **kwargs: seen.append(kwargs))
    context = make_ctx(db_path)
    with socket.socket() as probe:  # a free port: serve() checks that it can bind before it calls Waitress
        probe.bind(("127.0.0.1", 0))
        context.port = probe.getsockname()[1]
    try:
        web.serve(context)
    finally:
        context.session.close()
    assert seen[0]["threads"] == 4


def test_serve_refuses_a_port_that_is_in_use(db_path: Path, monkeypatch: pytest.MonkeyPatch):
    import socket

    import waitress

    started: list[object] = []
    monkeypatch.setattr(waitress, "serve", lambda *_a, **_k: started.append(1))
    context = make_ctx(db_path)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        context.port = listener.getsockname()[1]
        try:
            with pytest.raises(OSError):
                web.serve(context)
        finally:
            context.session.close()
    assert started == []


def test_serve_refuses_an_unresolvable_host(db_path: Path):
    context = make_ctx(db_path, host="no-such-host.invalid")
    try:
        with pytest.raises(OSError):
            web.serve(context)
    finally:
        context.session.close()
