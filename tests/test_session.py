"""Tests for the shared chat/sync core, driven directly (no CLI, no terminal output)."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest

from llmopenfinance.chat import get_llm
from llmopenfinance.constants import ENV_PLUGGY_CLIENT_ID, ENV_PLUGGY_CLIENT_SECRET
from llmopenfinance.i18n import t
from llmopenfinance.pluggy import PluggyError
from llmopenfinance.session import (
    AssistantBuild,
    BuildOptions,
    ChatSession,
    Factories,
    Notice,
    build_assistant,
    parse_command,
    run_command,
    run_sync,
)
from llmopenfinance.store import FinanceStore
from llmopenfinance.sync import ItemResult
from tests.conftest import ITEM_ID, FakePluggy
from tests.test_agent import ScriptedChat, answer

UseLanguage = Callable[[str], None]


class FakeLLM(ScriptedChat):
    def __init__(self, turns, *, provider="claude", reachable=True):
        super().__init__(turns)
        self.provider = provider
        self.reachable = reachable

    def _verify_connection(self) -> bool:
        return self.reachable


def store_opener(path: Path) -> Callable[..., FinanceStore]:
    """Return an ``open_store(*, read_only=False)`` factory over ``path``, like the CLI's."""

    def open_store(*, read_only: bool = False) -> FinanceStore:
        if read_only and not path.exists():
            FinanceStore(path).close()
        return FinanceStore(path, read_only=read_only)

    return open_store


def make_factories(path: Path, *, llm: Any = None, pluggy: Any = None) -> Factories:
    return Factories(
        llm=lambda *_a, **_k: llm,
        open_store=store_opener(path),
        pluggy_client=lambda: pluggy,
    )


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_PLUGGY_CLIENT_ID, "cid")
    monkeypatch.setenv(ENV_PLUGGY_CLIENT_SECRET, "secret")


@pytest.fixture
def session(db_path: Path) -> Iterator[ChatSession]:
    chat_session = ChatSession(store_opener(db_path))
    yield chat_session
    chat_session.close()


class Recorder:
    """Collects the notices and item results a sync emits, in order."""

    def __init__(self) -> None:
        self.events: list[Notice | ItemResult] = []

    def notify(self, notice: Notice) -> None:
        self.events.append(notice)

    def on_item(self, result: ItemResult) -> None:
        self.events.append(result)

    @property
    def notices(self) -> list[Notice]:
        return [event for event in self.events if isinstance(event, Notice)]

    @property
    def items(self) -> list[ItemResult]:
        return [event for event in self.events if isinstance(event, ItemResult)]


def sync_with(factories: Factories, recorder: Recorder, *, force: bool = False) -> int:
    return run_sync(factories, force=force, notify=recorder.notify, on_item=recorder.on_item)


# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------

ALIASES = [
    ("/ajuda", "help"),
    ("ajuda", "help"),
    ("/help", "help"),
    ("help", "help"),
    ("/resumo", "summary"),
    ("/summary", "summary"),
    ("/reset", "reset"),
    ("reset", "reset"),
    ("/historico", "history"),
    ("historico", "history"),
    ("/history", "history"),
    ("history", "history"),
    ("/sync", "sync"),
    ("sair", "exit"),
    ("/sair", "exit"),
    ("exit", "exit"),
    ("/exit", "exit"),
    ("quit", "exit"),
    ("/quit", "exit"),
    ("bye", "exit"),
    ("q", "exit"),
]


@pytest.mark.parametrize(("text", "expected"), ALIASES)
def test_parse_command_aliases(text: str, expected: str):
    assert parse_command(text) == expected
    assert parse_command(text.upper()) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("SAIR", "exit"), ("Exit", "exit"), ("/Ajuda", "help"), ("hello", None), ("", None), ("resumo", None)],
)
def test_parse_command_case_and_unknown(text: str, expected: str | None):
    assert parse_command(text) == expected


# ---------------------------------------------------------------------------
# build_assistant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "hint"),
    [("ollama", "ollama serve"), ("claude", "ANTHROPIC_API_KEY")],
)
def test_build_assistant_unreachable_llm_notices(session: ChatSession, db_path: Path, provider: str, hint: str):
    llm = FakeLLM([], provider=provider, reachable=False)
    build = build_assistant(
        make_factories(db_path, llm=llm),
        session,
        provider=provider,
        model="",
        options=BuildOptions(think=False, enable_sql=None),
    )
    assert build.assistant is None
    assert [notice.level for notice in build.notices] == ["info", "error", "info"]
    verifying, unreachable, hint_notice = build.notices
    assert verifying.text == t("cli.llm.verifying", provider=provider, model="fake-model")
    assert unreachable.text == t("cli.llm.unreachable", provider=provider, model="fake-model")
    assert hint in hint_notice.text  # the hint follows the error and is printed plain, as today


def test_build_assistant_claude_privacy_notice(session: ChatSession, db_path: Path):
    build = build_assistant(
        make_factories(db_path, llm=FakeLLM([], provider="claude")),
        session,
        provider="claude",
        model="",
        options=BuildOptions(think=False, enable_sql=None),
    )
    assert build.assistant is not None
    privacy = [notice for notice in build.notices if "Anthropic" in notice.text]
    assert privacy == [Notice("dim", t("cli.llm.privacy"))]
    assert "run_sql" in build.assistant.registry.names  # SQL defaults on for Claude


def test_build_assistant_ollama_has_no_privacy_notice_and_no_sql(session: ChatSession, db_path: Path):
    build = build_assistant(
        make_factories(db_path, llm=FakeLLM([], provider="ollama")),
        session,
        provider="ollama",
        model="",
        options=BuildOptions(think=True, enable_sql=None),
    )
    assert build.assistant is not None
    assert build.notices == (Notice("info", t("cli.llm.verifying", provider="ollama", model="fake-model")),)
    assert "run_sql" not in build.assistant.registry.names


def test_build_assistant_passes_provider_model_and_think(session: ChatSession, db_path: Path):
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def llm_factory(*args: Any, **kwargs: Any) -> FakeLLM:
        calls.append((args, kwargs))
        return FakeLLM([], provider="ollama")

    factories = Factories(llm=llm_factory, open_store=store_opener(db_path), pluggy_client=FakePluggy)
    build = build_assistant(
        factories, session, provider="ollama", model="m1", options=BuildOptions(think=True, enable_sql=True)
    )
    assert calls == [(("ollama", "m1"), {"think": True})]
    assert build.assistant is not None
    assert "run_sql" in build.assistant.registry.names  # explicit enable_sql wins over the provider default
    build.assistant.refresh()
    assert "NETFLIX" in build.assistant.chat.system_message  # the snapshot comes from the session's store


def test_build_assistant_unknown_provider(session: ChatSession, db_path: Path):
    factories = Factories(llm=get_llm, open_store=store_opener(db_path), pluggy_client=FakePluggy)
    build = build_assistant(
        factories, session, provider="nope", model="", options=BuildOptions(think=False, enable_sql=None)
    )
    assert build == AssistantBuild(
        assistant=None, notices=(Notice("error", t("llm.error.unknown_provider", provider="nope")),)
    )


class RecordingLLM(FakeLLM):
    """Records the notices already delivered when the (possibly slow) connection check starts."""

    def __init__(self, delivered: list[Notice], **kwargs: Any) -> None:
        super().__init__([], **kwargs)
        self.delivered = delivered
        self.seen_at_check: list[Notice] | None = None

    def _verify_connection(self) -> bool:
        self.seen_at_check = list(self.delivered)
        return super()._verify_connection()


@pytest.mark.parametrize(
    ("provider", "reachable", "levels"),
    [
        ("ollama", False, ["info", "error", "info"]),
        ("claude", False, ["info", "error", "info"]),
        ("claude", True, ["info", "dim"]),
        ("ollama", True, ["info"]),
    ],
)
def test_build_assistant_notifies_verifying_before_the_check(
    session: ChatSession, db_path: Path, provider: str, reachable: bool, levels: list[str]
):
    delivered: list[Notice] = []
    llm = RecordingLLM(delivered, provider=provider, reachable=reachable)
    build = build_assistant(
        make_factories(db_path, llm=llm),
        session,
        provider=provider,
        model="",
        options=BuildOptions(think=False, enable_sql=None),
        notify=delivered.append,
    )
    assert llm.seen_at_check == [Notice("info", t("cli.llm.verifying", provider=provider, model="fake-model"))]
    assert tuple(delivered) == build.notices  # every notice streamed exactly once, in the returned order
    assert [notice.level for notice in delivered] == levels
    assert (build.assistant is not None) is reachable


def test_build_assistant_notifies_factory_error(session: ChatSession, db_path: Path):
    delivered: list[Notice] = []
    factories = Factories(llm=get_llm, open_store=store_opener(db_path), pluggy_client=FakePluggy)
    build = build_assistant(
        factories,
        session,
        provider="nope",
        model="",
        options=BuildOptions(think=False, enable_sql=None),
        notify=delivered.append,
    )
    assert build.assistant is None
    assert tuple(delivered) == build.notices == (Notice("error", t("llm.error.unknown_provider", provider="nope")),)


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


def _assistant(session: ChatSession, db_path: Path, turns: list[Any]) -> Any:
    build = build_assistant(
        make_factories(db_path, llm=FakeLLM(turns, provider="ollama")),
        session,
        provider="ollama",
        model="",
        options=BuildOptions(think=False, enable_sql=None),
    )
    return build.assistant


def _no_sync() -> int:
    raise AssertionError("run_sync must only be called by the sync command")


def test_run_command_reset_and_history(session: ChatSession, db_path: Path):
    assistant = _assistant(session, db_path, [answer("Resposta curta")])
    question = "Quanto gasto com delivery? " + "x" * 120
    list(assistant.ask(question))
    history = run_command("history", assistant, session, _no_sync)
    assert history == (
        Notice("info", f"1. [USER] {question[:100]}…"),
        Notice("info", "2. [ASSISTANT] …"),
    )
    assert run_command("reset", assistant, session, _no_sync) == (Notice("info", t("chat.reset")),)
    assert assistant.chat.get_history() == []
    assert run_command("history", assistant, session, _no_sync) == (Notice("info", t("chat.history_empty")),)


def test_run_command_help_and_summary(session: ChatSession, db_path: Path, use_language: UseLanguage):
    assistant = _assistant(session, db_path, [])
    assert run_command("help", assistant, session, _no_sync) == (Notice("info", t("chat.help")),)
    (summary,) = run_command("summary", assistant, session, _no_sync)
    assert summary.level == "info"
    assert summary.text == session.snapshot()
    assert "Resumo financeiro" in summary.text
    use_language("en")
    (help_notice,) = run_command("help", assistant, session, _no_sync)
    assert help_notice.text.startswith("Commands: /help")


@pytest.mark.parametrize(
    ("code", "expected"),
    [(0, Notice("info", "Dados atualizados; conversa reiniciada.")), (1, Notice("warn", t("chat.sync_failed")))],
)
def test_run_command_sync_reopens_the_store_and_resets(
    session: ChatSession, db_path: Path, code: int, expected: Notice
):
    assistant = _assistant(session, db_path, [answer("ok")])
    list(assistant.ask("q"))
    before = session.store
    transactions: list[int] = []

    def fake_sync() -> int:
        # DuckDB allows a single read-write handle, so the read-only one must be closed while syncing.
        with FinanceStore(db_path) as store:
            transactions.append(store.counts()["transactions"])
        return code

    assert run_command("sync", assistant, session, fake_sync) == (expected,)
    assert len(transactions) == 1
    assert transactions[0] > 0
    assert session.store is not before
    assert session.store.counts()["transactions"] > 0  # the reopened read-only store works
    assert assistant.chat.get_history() == []


def test_resync_reopens_even_when_the_sync_raises(session: ChatSession):
    before = session.store

    def boom() -> int:
        raise RuntimeError("sync crashed")

    with pytest.raises(RuntimeError, match="sync crashed"):
        session.resync(boom)
    assert session.store is not before
    assert session.connection() is session.store.con


def test_run_command_rejects_exit_and_unknown(session: ChatSession, db_path: Path):
    assistant = _assistant(session, db_path, [])
    for name in ("exit", "nope"):
        with pytest.raises(ValueError, match=name):
            run_command(name, assistant, session, _no_sync)


# ---------------------------------------------------------------------------
# run_sync
# ---------------------------------------------------------------------------


def test_run_sync_no_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_PLUGGY_CLIENT_SECRET, raising=False)
    monkeypatch.setenv(ENV_PLUGGY_CLIENT_ID, "cid")
    opened: list[bool] = []

    def open_store(**_kwargs: Any) -> Any:
        opened.append(True)
        raise AssertionError("the store must not be opened without credentials")

    recorder = Recorder()
    factories = Factories(llm=get_llm, open_store=open_store, pluggy_client=FakePluggy)
    assert sync_with(factories, recorder) == 1
    assert recorder.events == [
        Notice(
            "error",
            t("cli.sync.no_credentials", id_var=ENV_PLUGGY_CLIENT_ID, secret_var=ENV_PLUGGY_CLIENT_SECRET),
        )
    ]
    assert opened == []


@pytest.mark.usefixtures("credentials")
def test_run_sync_no_items_returns_0(tmp_path: Path):
    recorder = Recorder()
    assert sync_with(make_factories(tmp_path / "empty.duckdb", pluggy=FakePluggy()), recorder) == 0
    assert recorder.events == [Notice("info", t("cli.sync.no_items"))]


@pytest.mark.usefixtures("credentials")
def test_run_sync_reports_items_via_callback(tmp_path: Path):
    path = tmp_path / "sync.duckdb"
    with FinanceStore(path) as store:
        store.add_item(ITEM_ID)
        store.add_item("ghost")
    pluggy = FakePluggy()
    pluggy.fail["list_loans"] = PluggyError("no", 403)
    recorder = Recorder()
    assert sync_with(make_factories(path, pluggy=pluggy), recorder) == 0
    assert [type(event).__name__ for event in recorder.events] == ["Notice", "ItemResult", "ItemResult", "Notice"]
    assert recorder.notices == [
        Notice("info", t("cli.sync.start")),
        Notice("info", t("cli.sync.done", ok=1, failed=1)),
    ]
    by_id = {result.item_id: result for result in recorder.items}
    assert by_id[ITEM_ID].ok
    assert by_id[ITEM_ID].counts["transactions"] > 0
    assert any("empréstimos" in warning for warning in by_id[ITEM_ID].warnings)
    assert not by_id["ghost"].ok
    assert pluggy.closed
    assert "authenticate" in pluggy.calls
    second = Recorder()
    assert sync_with(make_factories(path, pluggy=FakePluggy()), second) == 0
    assert any(result.skipped for result in second.items)


@pytest.mark.usefixtures("credentials")
def test_run_sync_all_failed_returns_1(tmp_path: Path):
    path = tmp_path / "ghost.duckdb"
    with FinanceStore(path) as store:
        store.add_item("ghost")
    recorder = Recorder()
    assert sync_with(make_factories(path, pluggy=FakePluggy()), recorder) == 1
    assert recorder.notices[-1] == Notice("info", t("cli.sync.done", ok=0, failed=1))


@pytest.mark.usefixtures("credentials")
def test_run_sync_auth_failure(tmp_path: Path):
    path = tmp_path / "auth.duckdb"
    with FinanceStore(path) as store:
        store.add_item(ITEM_ID)
    pluggy = FakePluggy()
    pluggy.fail["authenticate"] = PluggyError("unauthorized", 401)
    recorder = Recorder()
    assert sync_with(make_factories(path, pluggy=pluggy), recorder) == 1
    assert recorder.events == [Notice("error", t("cli.sync.auth_failed", error="unauthorized"))]


@pytest.mark.usefixtures("credentials")
def test_run_sync_locked_database(tmp_path: Path, use_language: UseLanguage):
    use_language("en")

    def locked(**_kwargs: Any) -> Any:
        raise duckdb.IOException("Could not set lock on file")

    recorder = Recorder()
    factories = Factories(llm=get_llm, open_store=locked, pluggy_client=FakePluggy)
    assert sync_with(factories, recorder) == 1
    (notice,) = recorder.notices
    assert recorder.items == []
    assert notice == Notice("warn", t("cli.sync.db_locked", error="Could not set lock on file"))
    assert notice.text.startswith("Local database in use by another llmopenfinance session")
