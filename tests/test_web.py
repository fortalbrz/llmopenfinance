"""Tests for the local web chat: Host check, token cookie, JSON-only POSTs, state and command routes."""

import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from flask.testing import FlaskClient

from llmopenfinance import web
from llmopenfinance.agent import FinanceAssistant
from llmopenfinance.i18n import t
from llmopenfinance.session import BuildOptions, ChatSession
from llmopenfinance.store import FinanceStore
from llmopenfinance.tools import ToolRegistry
from llmopenfinance.web import COOKIE_NAME, WebContext, create_app, host_allowed, new_token
from tests.test_agent import ScriptedChat, answer
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
