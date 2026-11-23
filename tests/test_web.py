"""Tests for the local web chat: Host check, token cookie, JSON-only POSTs, state, command and chat routes."""

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient
from werkzeug.test import EnvironBuilder
from werkzeug.wsgi import ClosingIterator

from llmopenfinance import web
from llmopenfinance.agent import FinanceAssistant
from llmopenfinance.chat import LLMError, TextDelta, TurnEnd
from llmopenfinance.i18n import t
from llmopenfinance.session import BuildOptions, ChatSession
from llmopenfinance.store import FinanceStore
from llmopenfinance.tools import ToolCall, ToolRegistry
from llmopenfinance.web import COOKIE_NAME, WebContext, create_app, host_allowed, new_token
from tests.test_agent import ScriptedChat, answer, tool_turn
from tests.test_session import make_factories, store_opener

UseLanguage = Callable[[str], None]

TOKEN = "s3cr3t-T0ken_value-for-tests"
HOST = "127.0.0.1:8000"
STATE_KEYS = {"provider", "model", "language", "last_sync", "counts", "busy", "strings", "transcript"}


def make_ctx(db_path: Path, *, turns: list[Any] | None = None, host: str = "127.0.0.1") -> WebContext:
    session = ChatSession(store_opener(db_path))
    chat = ScriptedChat(turns or [])
    assistant = FinanceAssistant(chat, ToolRegistry(session.connection), session.snapshot)
    return WebContext(
        session=session,
        assistant=assistant,
        factories=make_factories(db_path, llm=chat),
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


def test_sync_in_box_warns_to_use_button(ctx: WebContext, use_language: UseLanguage):
    client = client_for(ctx)
    expected = [{"type": "notice", "level": "warn", "text": t("web.sync_via_button")}, {"type": "done", "ok": False}]
    assert events(chat(client, "/sync")) == expected
    assert expected[0]["text"] == "Use o botão Sincronizar."
    use_language("en")
    assert events(chat(client, "/SYNC"))[0]["text"] == "Use the Sync button."
    assert ctx.session.store.counts()["transactions"] > 0  # the read-only store is still open
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

    monkeypatch.setattr(web, "stream_with_context", broken)
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
