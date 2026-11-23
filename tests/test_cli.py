"""End-to-end CLI tests with Typer's CliRunner, a fake Pluggy client and a scripted LLM."""

import io
import logging
import os
import re
import subprocess  # ruff: ignore[suspicious-subprocess-import] — the startup guard needs a fresh interpreter
import sys
from collections.abc import Callable
from typing import Any, ClassVar

import pytest
from rich.console import Console
from typer.testing import CliRunner

from llmopenfinance import __main__ as cli, __version__, ui
from llmopenfinance.constants import ENV_LANG
from llmopenfinance.i18n import reset_language, t
from llmopenfinance.pluggy import PluggyError
from llmopenfinance.store import FinanceStore
from llmopenfinance.tools import ToolCall
from tests.conftest import ITEM_ID, FakePluggy, populate
from tests.test_agent import ScriptedChat, answer, tool_turn

runner = CliRunner()


class FakeLLM(ScriptedChat):
    def __init__(self, turns, *, provider="claude", reachable=True):
        super().__init__(turns)
        self.provider = provider
        self.reachable = reachable

    def _verify_connection(self) -> bool:
        return self.reachable


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated DB path, Pluggy credentials and fake Pluggy client."""
    db = tmp_path / "cli.duckdb"
    fake = FakePluggy()
    monkeypatch.setenv("PLUGGY_USERNAME", "cid")
    monkeypatch.setenv("PLUGGY_PASSWORD", "secret")
    monkeypatch.delenv("PLUGGY_ITEM_IDS", raising=False)
    monkeypatch.delenv("LLMOPENFINANCE_PROVIDER", raising=False)
    monkeypatch.setattr(cli, "_pluggy_client", lambda: fake)
    return {"db": str(db), "pluggy": fake, "monkeypatch": monkeypatch}


def run(env, *args, input=None):  # ruff: ignore[builtin-argument-shadowing] — mirrors CliRunner.invoke
    return runner.invoke(cli.app, ["--db", env["db"], *args], input=input)


def use_llm(env, llm):
    env["monkeypatch"].setattr(cli, "_llm", lambda *_a, **_k: llm)
    return llm


def seed(env):
    with FinanceStore(env["db"]) as store:
        populate(store)


def test_version():
    assert "llmopenfinance 0.1.0" in runner.invoke(cli.app, ["--version"]).output
    assert "llmopenfinance 0.1.0" in runner.invoke(cli.app, ["ver"]).output


def test_items_add_list_remove_without_verify(env):
    result = run(env, "items", "add", "a1", "a2", "--no-verify")
    assert result.exit_code == 0
    assert "✓ a1" in result.output
    assert "já registrado" in run(env, "items", "add", "a1", "--no-verify").output
    listed = run(env, "items", "list").output
    assert "a1" in listed
    assert "não sincronizado" in listed
    assert run(env, "items", "remove", "a1").exit_code == 0
    assert run(env, "items", "remove", "a1").exit_code == 1


def test_items_list_empty(env):
    assert "Nenhuma conexão" in run(env, "items", "list").output


def test_items_add_with_verify(env):
    result = run(env, "items", "add", ITEM_ID, "missing")
    assert result.exit_code == 1
    assert "MeuPluggy" in result.output
    assert "✗ missing" in result.output
    assert env["pluggy"].closed


def test_items_from_env_var(env):
    env["monkeypatch"].setenv("PLUGGY_ITEM_IDS", "e1; e2")
    run(env, "items", "add", "e3", "--no-verify")  # any read-write command registers the env ids
    listed = run(env, "items", "list").output
    assert "e1" in listed
    assert "e2" in listed


def test_sync_requires_credentials(env):
    env["monkeypatch"].delenv("PLUGGY_PASSWORD")
    result = run(env, "sync")
    assert result.exit_code == 1
    assert "PLUGGY_USERNAME" in result.output


def test_sync_without_items(env):
    result = run(env, "sync")
    assert result.exit_code == 0
    assert "Nenhuma conexão registrada" in result.output


def test_sync_downloads_and_reports(env):
    run(env, "items", "add", ITEM_ID, "--no-verify")
    result = run(env, "sync")
    assert result.exit_code == 0, result.output
    assert "✓ item-1" in result.output
    assert "1 ok, 0 com falha" in result.output
    second = run(env, "sync")
    assert "sem alterações" in second.output


def test_sync_reports_failures_and_warnings(env):
    run(env, "items", "add", ITEM_ID, "ghost", "--no-verify")
    env["pluggy"].fail["list_loans"] = PluggyError("no", 403)
    result = run(env, "sync", "--force")
    assert "✗ ghost" in result.output
    assert "! empréstimos indisponível" in result.output
    assert result.exit_code == 0


def test_sync_all_failed_exit_code(env):
    run(env, "items", "add", "ghost", "--no-verify")
    assert run(env, "sync").exit_code == 1


def test_sync_auth_failure(env):
    run(env, "items", "add", ITEM_ID, "--no-verify")
    env["pluggy"].fail["authenticate"] = PluggyError("unauthorized", 401)
    result = run(env, "sync")
    assert result.exit_code == 1
    assert "Falha ao autenticar" in result.output


def test_status(env):
    seed(env)
    result = run(env, "status")
    assert result.exit_code == 0
    assert "transactions=" in result.output
    assert "consentimento até 01/06/2027" in result.output
    assert "Nenhuma conexão" not in result.output


def test_status_empty(env):
    assert "Nenhuma conexão registrada" in run(env, "status").output


def test_ask_streams_answer_and_tool_status(env):
    seed(env)
    llm = use_llm(
        env, FakeLLM([tool_turn(ToolCall("c", "detect_subscriptions", {"months": 6})), answer("Você assina Netflix.")])
    )
    result = run(env, "ask", "Quais assinaturas eu tenho?")
    assert result.exit_code == 0, result.output
    assert "consultando detect_subscriptions(months=6)" in result.output
    assert "Você assina Netflix." in result.output
    assert "Anthropic" in result.output  # privacy notice for Claude
    assert "NETFLIX" in llm.system_message
    assert "netflix" in llm.tool_results[0].content.lower()


def test_ask_reports_llm_error(env):
    seed(env)
    from llmopenfinance.chat import LLMError

    use_llm(env, FakeLLM([LLMError("sem crédito")], provider="ollama"))
    result = run(env, "ask", "x")
    assert "[erro] sem crédito" in result.output
    assert "Anthropic" not in result.output
    assert result.exit_code == 1  # regression: failed answers used to exit 0


def test_sync_reports_locked_database(env):
    import duckdb

    def locked(**_kwargs):
        raise duckdb.IOException("Could not set lock on file")

    env["monkeypatch"].setattr(cli, "_open_store", locked)
    result = run(env, "sync")
    assert result.exit_code == 1
    assert "em uso por outra sessão" in result.output


def test_chat_sync_failure_is_reported(env):
    seed(env)
    use_llm(env, FakeLLM([]))
    env["monkeypatch"].delenv("PLUGGY_PASSWORD")
    result = run(env, "chat", "--no-sync", input="/sync\nsair\n")
    assert "Sincronização falhou; usando os dados locais" in result.output
    assert "Dados atualizados" not in result.output


def test_status_and_list_work_while_another_reader_is_open(env):
    seed(env)
    with FinanceStore(env["db"], read_only=True):
        assert run(env, "status").exit_code == 0
        assert ITEM_ID in run(env, "items", "list").output


def test_ask_llm_unreachable(env):
    use_llm(env, FakeLLM([], provider="ollama", reachable=False))
    result = run(env, "--provider", "ollama", "ask", "x")
    assert result.exit_code == 1
    assert "ollama serve" in result.output
    use_llm(env, FakeLLM([], provider="claude", reachable=False))
    assert "ANTHROPIC_API_KEY" in run(env, "ask", "x").output


def test_ask_llm_misconfigured(env):
    def boom(*_a, **_k):
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    env["monkeypatch"].setattr(cli, "_llm", boom)
    result = run(env, "ask", "x")
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output


def test_ask_with_sync(env):
    run(env, "items", "add", ITEM_ID, "--no-verify")
    use_llm(env, FakeLLM([answer("ok")]))
    result = run(env, "ask", "x", "--sync")
    assert "Sincronização concluída" in result.output


def test_chat_session_commands(env):
    seed(env)
    llm = use_llm(env, FakeLLM([answer("Resposta 1"), answer("Resposta 2")]))
    lines = ["Quanto gasto com delivery?", "", "/ajuda", "/resumo", "/historico", "/reset", "/historico", "oi", "sair"]
    result = run(env, "chat", "--no-sync", input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    assert "Resposta 1" in result.output
    assert "Comandos:" in result.output
    assert "Resumo financeiro" in result.output
    assert "[USER] Quanto gasto com delivery?" in result.output
    assert "Conversa reiniciada." in result.output
    assert "(vazio)" in result.output
    assert "Resposta 2" in result.output
    assert "Até logo!" in result.output
    assert llm.turns == []


def test_chat_sync_command_and_eof(env):
    run(env, "items", "add", ITEM_ID, "--no-verify")
    use_llm(env, FakeLLM([]))
    result = run(env, "chat", "--no-sync", input="/sync\n")
    assert "Ainda não há transações" in result.output
    assert "Dados atualizados; conversa reiniciada." in result.output
    assert "Até logo!" in result.output  # EOF ends the session
    with FinanceStore(env["db"]) as store:
        assert store.counts()["transactions"] > 0


def test_default_command_is_chat_and_syncs(env):
    run(env, "items", "add", ITEM_ID, "--no-verify")
    use_llm(env, FakeLLM([]))
    result = run(env, input="sair\n")
    assert result.exit_code == 0
    assert "Sincronização concluída" in result.output
    assert "Assistente financeiro pronto" in result.output


def test_chat_without_credentials_uses_local_data(env):
    seed(env)
    env["monkeypatch"].delenv("PLUGGY_USERNAME")
    use_llm(env, FakeLLM([]))
    result = run(env, "chat", input="q\n")
    assert "Credenciais Pluggy ausentes" in result.output


def test_chat_loop_survives_errors_and_interrupts(env, monkeypatch):
    seed(env)
    inputs = iter(["boom", "ctrl-c", "sair"])

    def prompt(_label):
        value = next(inputs)
        if value == "ctrl-c":
            raise KeyboardInterrupt
        return value

    def explode(*_a, **_k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(cli, "_render_answer", explode)
    session = cli._ChatSession.__new__(cli._ChatSession)
    assert cli._chat_loop(object(), session, prompt=prompt) == 0


def test_connect_requires_credentials(env):
    env["monkeypatch"].delenv("PLUGGY_USERNAME")
    assert run(env, "connect").exit_code == 1


def test_connect_flow_registers_items_and_syncs(env):
    import llmopenfinance.connect as connect_mod

    def fake_flow(client, on_item, *, sandbox, on_url):
        assert sandbox is True
        on_url("http://127.0.0.1:1/nonce")
        on_item(ITEM_ID)
        return [ITEM_ID]

    env["monkeypatch"].setattr(connect_mod, "run_connect_flow", fake_flow)
    env["monkeypatch"].setattr(connect_mod, "wait_for_item", lambda client, item_id: client.get_item(item_id))
    result = run(env, "connect", "--sandbox")
    assert result.exit_code == 0, result.output
    assert "Conexão recebida: item-1" in result.output
    assert "status: UPDATED" in result.output
    assert "Sincronização concluída" in result.output


def test_connect_no_items_and_errors(env):
    import llmopenfinance.connect as connect_mod

    env["monkeypatch"].setattr(connect_mod, "run_connect_flow", lambda *_a, **_k: [])
    result = run(env, "connect", "--no-sync")
    assert result.exit_code == 1
    assert "Nenhuma conexão criada" in result.output

    def failing(*_a, **_k):
        raise PluggyError("token refused", 403)

    env["monkeypatch"].setattr(connect_mod, "run_connect_flow", failing)
    result = run(env, "connect")
    assert result.exit_code == 1
    assert "token refused" in result.output


def test_connect_without_sync_returns_zero(env):
    import llmopenfinance.connect as connect_mod

    env["monkeypatch"].setattr(connect_mod, "run_connect_flow", lambda *_a, **_k: ["x"])
    assert run(env, "connect", "--no-sync").exit_code == 0


def test_real_factories(monkeypatch):
    monkeypatch.setenv("PLUGGY_USERNAME", "cid")
    monkeypatch.setenv("PLUGGY_PASSWORD", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert cli._pluggy_client().__class__.__name__ == "PluggyClient"
    assert cli._llm("claude", "").provider == "claude"
    assert cli._llm("ollama", "m", think=True).think is True

llmopenfinance
def test_main_reconfigures_streams(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["llmopenfinance", "ver"])
    with pytest.raises(SystemExit) as info:
        cli.main()
    assert info.value.code == 0


# ---------------------------------------------------------------------------
# Localisation: English output, aliases of both languages, startup guard, pt-BR pins
# ---------------------------------------------------------------------------

UseLanguage = Callable[[str], None]


def test_en_items_list_empty(env, use_language: UseLanguage):
    use_language("en")
    result = run(env, "items", "list")
    assert result.exit_code == 0
    assert "No connections registered." in result.output
    assert "Nenhuma" not in result.output


def test_en_items_add_list_remove(env, use_language: UseLanguage):
    use_language("en")
    assert "✓ a1" in run(env, "items", "add", "a1", "--no-verify").output
    assert "(already registered)" in run(env, "items", "add", "a1", "--no-verify").output
    assert "not synced" in run(env, "items", "list").output
    assert "Removed: a1" in run(env, "items", "remove", "a1").output
    missing = run(env, "items", "remove", "a1")
    assert missing.exit_code == 1
    assert "Item not found: a1" in missing.output


def test_en_sync_downloads_and_reports(env, use_language: UseLanguage):
    use_language("en")
    run(env, "items", "add", ITEM_ID, "--no-verify")
    result = run(env, "sync")
    assert result.exit_code == 0, result.output
    assert "Syncing Pluggy data…" in result.output
    assert "✓ item-1" in result.output
    assert "Sync finished: 1 ok, 0 failed." in result.output
    second = run(env, "sync")
    assert "no changes since the last sync" in second.output
    assert "sem alterações" not in second.output


def test_en_sync_messages_before_the_download(env, use_language: UseLanguage):
    use_language("en")llmopenfinancellmopenfinance
    no_items = run(env, "sync")
    assert "No connections registered. Use 'llmopenfinance connect' or 'llmopenfinance items add <item-id>'." in no_items.output
    run(env, "items", "add", ITEM_ID, "--no-verify")
    env["pluggy"].fail["authenticate"] = PluggyError("unauthorized", 401)
    failed = run(env, "sync")
    assert failed.exit_code == 1
    assert "Failed to authenticate with Pluggy: unauthorized" in failed.output
    env["monkeypatch"].delenv("PLUGGY_PASSWORD")
    no_credentials = run(env, "sync")
    assert no_credentials.exit_code == 1
    assert (
        "Set PLUGGY_USERNAME (clientId) and PLUGGY_PASSWORD (clientSecret) from your application at "
        "dashboard.pluggy.ai." in no_credentials.output
    )


def test_en_sync_reports_locked_database(env, use_language: UseLanguage):
    import duckdb

    use_language("en")

    def locked(**_kwargs):
        raise duckdb.IOException("Could not set lock on file")

    env["monkeypatch"].setattr(cli, "_open_store", locked)
    result = run(env, "sync")llmopenfinance
    assert result.exit_code == 1
    assert "Local database in use by another llmopenfinance session (" in result.output
    assert "sync skipped." in result.output


def test_en_status_labels(env, use_language: UseLanguage):
    use_language("en")
    seed(env)
    result = run(env, "status")
    assert result.exit_code == 0
    assert f"Local database: {env['db']}" in result.output
    assert "Last sync: never" in result.output
    assert "Data: " in result.output
    assert "transactions=" in result.output
    assert "consent until 2027-06-01" in result.output
    assert "Nenhuma" not in result.output


def test_en_status_empty(env, use_language: UseLanguage):
    use_language("en")
    assert "No connections registered." in run(env, "status").output


def test_en_ask_error_prefix(env, use_language: UseLanguage):
    from llmopenfinance.chat import LLMError

    use_language("en")
    seed(env)
    use_llm(env, FakeLLM([LLMError("no credit")], provider="ollama"))
    result = run(env, "ask", "x")
    assert "[error] no credit" in result.output
    assert "[erro]" not in result.output
    assert result.exit_code == 1


def test_en_ask_streams_answer_and_tool_status(env, use_language: UseLanguage):
    use_language("en")
    seed(env)
    use_llm(
        env,
        FakeLLM([tool_turn(ToolCall("c", "detect_subscriptions", {"months": 6})), answer("You subscribe to Netflix.")]),
    )
    result = run(env, "ask", "Which subscriptions do I have?")
    assert result.exit_code == 0, result.output
    assert "Assistant: " in result.output
    assert "↳ querying detect_subscriptions(months=6)" in result.output
    assert "You subscribe to Netflix." in result.output
    assert "consultando" not in result.output


def test_en_privacy_notice_for_claude(env, use_language: UseLanguage):
    use_language("en")
    seed(env)
    use_llm(env, FakeLLM([answer("ok")], provider="claude"))
    result = run(env, "ask", "x")
    assert "Checking connection to claude" in result.output
    assert (
        "Notice: with Claude, the financial summary and the query results are sent to Anthropic's API. "
        "Use --provider ollama to keep everything local." in result.output
    )
    assert "Aviso" not in result.output
    use_llm(env, FakeLLM([answer("ok")], provider="ollama"))
    assert "Anthropic" not in run(env, "--provider", "ollama", "ask", "x").output


def test_en_llm_unreachable_and_misconfigured(env, use_language: UseLanguage):
    use_language("en")
    use_llm(env, FakeLLM([], provider="ollama", reachable=False))
    result = run(env, "--provider", "ollama", "ask", "x")
    assert result.exit_code == 1
    assert "Could not connect to ollama /" in result.output
    assert "Run: ollama serve  and  ollama pull" in result.output
    use_llm(env, FakeLLM([], provider="claude", reachable=False))
    assert "Check ANTHROPIC_API_KEY and the model name." in run(env, "ask", "x").output


def test_en_chat_session_commands(env, use_language: UseLanguage):
    use_language("en")
    seed(env)
    llm = use_llm(env, FakeLLM([answer("Answer 1"), answer("Answer 2")]))
    lines = [
        "How much do I spend on delivery?",
        "",
        "/help",
        "/summary",
        "/history",
        "/reset",
        "/history",
        "hi",
        "exit",
    ]
    result = run(env, "chat", "--no-sync", input="".join(f"{line}\n" for line in lines))
    assert result.exit_code == 0, result.output
    assert "Financial assistant ready. Commands: /help  /summary (data sent to the model)" in result.output
    assert "Answer 1" in result.output
    assert "Commands: /help  /summary" in result.output
    assert "# Financial summary" in result.output
    assert "[USER] How much do I spend on delivery?" in result.output
    assert "Conversation reset." in result.output
    assert "(empty)" in result.output
    assert "Answer 2" in result.output
    assert "Goodbye!" in result.output
    assert "Até logo" not in result.output
    assert llm.turns == []


def test_en_chat_sync_command_and_eof(env, use_language: UseLanguage):
    use_language("en")
    run(env, "items", "add", ITEM_ID, "--no-verify")
    use_llm(env, FakeLLM([]))llmopenfinance
    result = run(env, "chat", "--no-sync", input="/sync\n")
    assert "No local transactions yet. Connect an account with 'llmopenfinance connect'" in result.output
    assert "Data updated; conversation reset." in result.output
    assert "Goodbye!" in result.output  # EOF ends the session


def test_en_chat_sync_failure_is_reported(env, use_language: UseLanguage):
    use_language("en")
    seed(env)
    use_llm(env, FakeLLM([]))
    env["monkeypatch"].delenv("PLUGGY_PASSWORD")
    result = run(env, "chat", "--no-sync", input="/sync\nexit\n")
    assert "Sync failed; using the local data (conversation reset)." in result.output
    assert "Data updated" not in result.output


def test_en_chat_without_credentials_uses_local_data(env, use_language: UseLanguage):
    use_language("en")
    seed(env)
    env["monkeypatch"].delenv("PLUGGY_USERNAME")
    use_llm(env, FakeLLM([]))
    result = run(env, "chat", input="q\n")
    assert "Pluggy credentials missing — using local data only." in result.output


def test_en_connect_messages(env, use_language: UseLanguage):
    import llmopenfinance.connect as connect_mod

    use_language("en")

    def fake_flow(client, on_item, *, sandbox, on_url):
        on_url("http://127.0.0.1:1/nonce")
        on_item(ITEM_ID)
        return [ITEM_ID]

    env["monkeypatch"].setattr(connect_mod, "run_connect_flow", fake_flow)
    env["monkeypatch"].setattr(connect_mod, "wait_for_item", lambda client, item_id: client.get_item(item_id))
    result = run(env, "connect", "--no-sync")
    assert result.exit_code == 0, result.output
    assert "Opening Pluggy Connect in the browser…" in result.output
    assert "If the browser does not open, go to: http://127.0.0.1:1/nonce" in result.output
    assert "Connection received: item-1 — waiting for Pluggy to sync…" in result.output
    env["monkeypatch"].setattr(connect_mod, "run_connect_flow", lambda *_a, **_k: [])
    assert "No connection created." in run(env, "connect", "--no-sync").output

    def failing(*_a, **_k):
        raise PluggyError("token refused", 403)

    env["monkeypatch"].setattr(connect_mod, "run_connect_flow", failing)
    assert "Pluggy error: token refused" in run(env, "connect").output
    env["monkeypatch"].delenv("PLUGGY_PASSWORD")
    assert "Set PLUGGY_USERNAME and PLUGGY_PASSWORD." in run(env, "connect").output


def test_en_chat_loop_interrupt_and_eof(capsys, use_language: UseLanguage):
    use_language("en")
    labels: list[str] = []
    steps = iter([KeyboardInterrupt(), EOFError()])

    def prompt(label):
        labels.append(label)
        raise next(steps)

    session = cli._ChatSession.__new__(cli._ChatSession)
    assert cli._chat_loop(object(), session, prompt=prompt) == 0
    out = capsys.readouterr().out
    assert labels == ["\nYou", "\nYou"]
    assert "(interrupted — type 'exit' to quit)" in out
    assert "Goodbye!" in out


EXIT_ALIASES = ["sair", "exit", "quit", "bye", "q", "/sair", "/exit", "/quit", "SAIR", "Exit"]
COMMAND_ALIASES = [
    # (language, typed command, text the command prints in that language)
    ("pt-BR", "/ajuda", "Comandos: /ajuda"),
    ("pt-BR", "/help", "Comandos: /ajuda"),
    ("pt-BR", "ajuda", "Comandos: /ajuda"),
    ("pt-BR", "help", "Comandos: /ajuda"),
    ("pt-BR", "/resumo", "# Resumo financeiro"),
    ("pt-BR", "/summary", "# Resumo financeiro"),
    ("pt-BR", "/historico", "(vazio)"),
    ("pt-BR", "/history", "(vazio)"),
    ("pt-BR", "historico", "(vazio)"),
    ("pt-BR", "history", "(vazio)"),
    ("pt-BR", "/reset", "Conversa reiniciada."),
    ("en", "/help", "Commands: /help"),
    ("en", "/ajuda", "Commands: /help"),
    ("en", "help", "Commands: /help"),
    ("en", "ajuda", "Commands: /help"),
    ("en", "/summary", "# Financial summary"),
    ("en", "/resumo", "# Financial summary"),
    ("en", "/history", "(empty)"),
    ("en", "/historico", "(empty)"),
    ("en", "historico", "(empty)"),
    ("en", "history", "(empty)"),
    ("en", "/reset", "Conversation reset."),
]


@pytest.mark.parametrize(("language", "command", "expected"), COMMAND_ALIASES)
def test_chat_aliases_accepted_in_any_language(env, use_language: UseLanguage, language, command, expected):
    use_language(language)
    seed(env)
    use_llm(env, FakeLLM([]))  # any line that reached the assistant would fail: there is no scripted turn
    result = run(env, "chat", "--no-sync", input=f"{command}\nsair\n")
    assert result.exit_code == 0, result.output
    assert expected in result.output
    assert "Assistente: " not in result.output
    assert "Assistant: " not in result.output


@pytest.mark.parametrize("language", ["pt-BR", "en"])
@pytest.mark.parametrize("alias", EXIT_ALIASES)
def test_exit_aliases_accepted_in_any_language(env, use_language: UseLanguage, language, alias):
    use_language(language)
    seed(env)
    use_llm(env, FakeLLM([]))
    # The second line must never be read: had the alias not ended the session, it would reach the assistant.
    result = run(env, "chat", "--no-sync", input=f"{alias}\nthis line must not be read\n")
    assert result.exit_code == 0, result.output
    assert "Goodbye!" in result.output or "Até logo!" in result.output
    assert "Assistente: " not in result.output
    assert "Assistant: " not in result.output


def test_help_text_first_line_names_the_current_language_forms(use_language: UseLanguage):
    assert cli._help_text().splitlines()[0] == (
        "Comandos: /ajuda  /resumo (dados enviados ao modelo)  /sync  /reset  /historico  sair"
    )
    use_language("en")
    assert cli._help_text().splitlines()[0] == (
        "Commands: /help  /summary (data sent to the model)  /sync  /reset  /history  exit"
    )


def test_help_option_texts_come_from_the_catalog():
    assert "Assistente financeiro no terminal" in runner.invoke(cli.app, ["--help"]).output
    assert "Provedor de IA: claude | ollama" in runner.invoke(cli.app, ["--help"]).output
    sync_help = runner.invoke(cli.app, ["sync", "--help"]).output
    assert "Sincroniza contas, transações, faturas" in sync_help
    assert "Baixar tudo mesmo sem alterações" in sync_help
    assert "Gerencia as conexões" in runner.invoke(cli.app, ["items", "--help"]).output
    assert "Item ids da Pluggy" in runner.invoke(cli.app, ["items", "add", "--help"]).output


def _cli_subprocess(language: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -I -m llmopenfinance`` in a fresh interpreter with ``LLMOPENFINANCE_LANG`` set."""
    environment = {**os.environ, ENV_LANG: language, "PYTHONUTF8": "1"}
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] — fixed argument list, no shell
        [sys.executable, "-I", "-m", "llmopenfinance", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",  # -I ignores PYTHONUTF8, so a bad value may reach stderr in the console code page
        env=environment,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize("language", ["fr", "français"])
def test_invalid_language_exits_cleanly(language):
    result = _cli_subprocess(language, "ver")
    assert result.returncode == 2
    assert "pt-BR" in result.stderr
    assert "en" in result.stderr
    assert ENV_LANG in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("language", [" EN ", "en_US", "pt_br", ""])
def test_valid_or_blank_language_runs(language):
    result = _cli_subprocess(language, "ver")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "llmopenfinance 0.1.0"
    assert "Traceback" not in result.stderr


def test_help_is_english_when_the_language_is_en():
    result = _cli_subprocess("en", "--help")
    assert result.returncode == 0, result.stderr
    assert "Financial assistant in the terminal" in result.stdout
    assert "Interactive chat with the financial assistant" in result.stdout
    assert "Assistente" not in result.stdout
    assert "Conversa" not in result.stdout


def test_help_is_portuguese_by_default():
    result = _cli_subprocess("", "--help")
    assert result.returncode == 0, result.stderr
    assert "Assistente financeiro no terminal" in result.stdout
    assert "Financial assistant" not in result.stdout


def test_resolve_language_exits_with_status_2(monkeypatch, capsys):
    monkeypatch.setenv(ENV_LANG, "fr")
    reset_language()
    with pytest.raises(SystemExit) as info:
        cli._resolve_language()
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert "pt-BR" in err
    assert "en" in err
    assert "Traceback" not in err


def test_resolve_language_returns_the_normalised_language(monkeypatch):
    monkeypatch.setenv(ENV_LANG, " EN ")
    reset_language()
    assert cli._resolve_language() == "en"


# Every pt-BR string moved to the catalog, pinned to the exact text it had in the source.
PT_GOLDEN: list[tuple[str, dict[str, object], str]] = [
    ("cli.app.help", {}, "Assistente financeiro no terminal — converse com seus dados de Open Finance (Pluggy)."),
    ("cli.items.help", {}, "Gerencia as conexões (itens) Pluggy sincronizadas."),
    ("cli.option.verbose", {}, "Saída detalhada (logs INFO)"),
    ("cli.option.provider", {}, "Provedor de IA: claude | ollama"),
    ("cli.option.model", {}, "Modelo (vazio = padrão do provedor)"),
    ("cli.option.db", {}, "Arquivo DuckDB local"),
    ("cli.option.version", {}, "Mostra a versão"),
    ("cli.chat.help", {}, "Conversa interativa com o assistente financeiro (comando padrão)."),
    ("cli.chat.sync", {}, "Sincronizar antes de conversar"),
    ("cli.chat.think", {}, "Ativa o raciocínio (thinking) no Ollama"),
    ("cli.chat.sql", {}, "Permite a ferramenta run_sql (padrão: só no Claude)"),
    ("cli.ask.help", {}, "Faz uma pergunta única e sai."),
    ("cli.ask.question", {}, "Pergunta"),
    ("cli.ask.sync", {}, "Sincronizar antes"),
    ("cli.ask.sql", {}, "Permite run_sql"),
    ("cli.sync.help", {}, "Sincroniza contas, transações, faturas, investimentos e empréstimos da Pluggy."),
    ("cli.sync.force", {}, "Baixar tudo mesmo sem alterações"),
    ("cli.status.help", {}, "Mostra conexões registradas, consentimentos e volume de dados locais."),
    ("cli.connect.help", {}, "Conecta contas via Pluggy Connect (MeuPluggy) no navegador."),
    ("cli.connect.sandbox", {}, "Mostrar bancos de teste (sandbox)"),
    ("cli.connect.sync", {}, "Sincronizar ao final"),
    ("cli.items.add.help", {}, "Registra item ids da Pluggy para sincronização."),
    ("cli.items.add.ids", {}, "Item ids da Pluggy (dashboard → Demo → Connected Items)"),
    ("cli.items.add.verify", {}, "Validar na Pluggy"),
    ("cli.items.list.help", {}, "Lista as conexões registradas."),
    ("cli.items.remove.help", {}, "Remove uma conexão e seus dados locais."),
    ("cli.items.remove.id", {}, "Item id"),
    ("cli.ver.help", {}, "Mostra a versão."),
    ("cli.no_connections", {}, "Nenhuma conexão registrada."),
    ("cli.sync.unchanged", {}, "sem alterações desde a última sincronização"),
    (
        "cli.sync.no_credentials",
        {"id_var": "PLUGGY_USERNAME", "secret_var": "PLUGGY_PASSWORD"},
        "Defina PLUGGY_USERNAME (clientId) e PLUGGY_PASSWORD (clientSecret) da sua aplicação em dashboard.pluggy.ai.",
    ),
    (
        "cli.sync.db_locked",llmopenfinance
        {"error": "lock"},
        "Banco local em uso por outra sessão do llmopenfinance (lock); sincronização ignorada.",
    ),
    (
        "cli.sync.no_items",llmopenfinancellmopenfinance
        {},
        "Nenhuma conexão registrada. Use 'llmopenfinance connect' ou 'llmopenfinance items add <item-id>'.",
    ),
    ("cli.sync.auth_failed", {"error": "boom"}, "Falha ao autenticar na Pluggy: boom"),
    ("cli.sync.start", {}, "Sincronizando dados da Pluggy…"),
    ("cli.sync.done", {"ok": 1, "failed": 0}, "Sincronização concluída: 1 ok, 0 com falha."),
    ("cli.sync.skipped_no_credentials", {}, "Credenciais Pluggy ausentes — usando apenas os dados locais."),
    ("cli.llm.verifying", {"provider": "claude", "model": "m"}, "Verificando conexão com claude (m)…"),
    ("cli.llm.unreachable", {"provider": "claude", "model": "m"}, "Não foi possível conectar a claude / m."),
    ("cli.llm.ollama_hint", {"model": "m"}, "Rode: ollama serve  e  ollama pull m"),
    ("cli.llm.claude_hint", {}, "Verifique ANTHROPIC_API_KEY e o nome do modelo."),
    (
        "cli.llm.privacy",
        {},
        (
            "Aviso: com Claude, o resumo financeiro e os resultados das consultas são enviados à API da Anthropic. "
            "Use --provider ollama para manter tudo local."
        ),
    ),
    ("cli.status.database", {"path": "x.duckdb"}, "Banco local: x.duckdb"),
    ("cli.status.last_sync", {"when": "nunca"}, "Última sincronização: nunca"),
    ("cli.status.never", {}, "nunca"),
    ("cli.status.data", {"counts": "a=1"}, "Dados: a=1"),
    ("cli.status.consent_until", {"date": "01/06/2027"}, "consentimento até 01/06/2027"),
    ("cli.items.not_synced", {}, "não sincronizado"),
    ("cli.items.already_registered", {}, "(já registrado)"),
    ("cli.items.removed", {"item_id": "a1"}, "Removido: a1"),
    ("cli.items.not_found", {"item_id": "a1"}, "Item não encontrado: a1"),
    ("cli.connect.missing_credentials", {"id_var": "A", "secret_var": "B"}, "Defina A e B."),
    ("cli.connect.pluggy_failed", {"error": "boom"}, "Falha na Pluggy: boom"),
    ("cli.connect.none_created", {}, "Nenhuma conexão criada."),
    ("cli.connect.received", {"item_id": "i1"}, "Conexão recebida: i1 — aguardando a sincronização da Pluggy…"),
    ("cli.connect.opening", {}, "Abrindo o Pluggy Connect no navegador… conclua a conexão e feche o widget."),
    ("cli.connect.browser_hint", {"url": "http://x"}, "Se o navegador não abrir, acesse: http://x"),
    ("chat.assistant_label", {}, "Assistente"),
    ("chat.tool_status", {"name": "f", "args": "a=1"}, "↳ consultando f(a=1)"),
    ("chat.error", {"message": "caiu"}, "[erro] caiu"),
    ("chat.prompt_label", {}, "Você"),
    ("chat.goodbye", {}, "Até logo!"),
    ("chat.reset", {}, "Conversa reiniciada."),
    ("chat.history_empty", {}, "(vazio)"),
    ("chat.sync_ok", {}, "Dados atualizados; conversa reiniciada."),
    ("chat.sync_failed", {}, "Sincronização falhou; usando os dados locais (conversa reiniciada)."),
    ("chat.interrupted", {}, "(interrompido — digite 'sair' para encerrar)"),
    ("chat.ready", {}, "Assistente financeiro pronto."),
    (
        "chat.no_transactions",
        {},llmopenfinance
        (llmopenfinancellmopenfinance
            "Ainda não há transações locais. Conecte uma conta com 'llmopenfinance connect' ou "
            "'llmopenfinance items add <item-id>' e rode 'llmopenfinance sync'."
        ),
    ),
    (
        "chat.help",
        {},
        (
            "Comandos: /ajuda  /resumo (dados enviados ao modelo)  /sync  /reset  /historico  sair\n"
            'Exemplos: "Quais assinaturas eu tenho?"  "Quanto gasto por mês com compras online?"\n'
            '          "Como posso melhorar minhas finanças?"'
        ),
    ),
]


@pytest.mark.parametrize(("key", "values", "expected"), PT_GOLDEN, ids=[row[0] for row in PT_GOLDEN])
def test_pt_cli_strings_are_unchanged(key, values, expected):
    assert t(key, **values) == expected


# ---------------------------------------------------------------------------
# Rich restyle: grouped help, epilog, banner, no ANSI off a terminal, tables on a terminal
# ---------------------------------------------------------------------------

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
ENV_VARS = (
    "LLMOPENFINANCE_PROVIDER",
    "LLMOPENFINANCE_DB",
    "LLMOPENFINANCE_LANG",
    "PLUGGY_USERNAME",
    "PLUGGY_PASSWORD",
    "PLUGGY_ITEM_IDS",
)
LONG_ID = "0a1b2c3d-4e5f-6789-abcd-ef0123456789"  # 36 characters, like a Pluggy item id


def forced_terminal(monkeypatch: pytest.MonkeyPatch, width: int) -> io.StringIO:
    """Replace the ``ui`` console with a forced terminal; everything the CLI prints through ``ui`` lands here."""
    buffer = io.StringIO()
    console = Console(
        file=buffer, theme=ui.THEME, width=width, force_terminal=True, color_system="standard", legacy_windows=False
    )
    monkeypatch.setattr(ui, "console", console)
    return buffer


def help_panels(output: str) -> dict[str, str]:
    """Split ``--help`` output into ``{panel title: panel text}``."""
    parts = re.split(r"^[┌╭]─ (.+?) ─+[┐╮]$", output, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_help_groups_commands():
    output = runner.invoke(cli.app, ["--help"]).output
    panels = help_panels(output)
    groups = {
        t("cli.panel.chat"): ("chat", "ask"),
        t("cli.panel.data"): ("sync", "status"),
        t("cli.panel.connections"): ("connect", "items"),
    }
    for title, commands in groups.items():
        assert title in panels, output
        for command in commands:
            assert re.search(rf"^│ {command}\s", panels[title], flags=re.MULTILINE), (title, command)
        assert not re.search(r"^│ ver\s", panels[title], flags=re.MULTILINE)  # ``ver`` stays ungrouped
    for command in ("chat", "ask", "sync", "status", "connect", "items", "ver"):
        assert command in output


def test_help_panel_titles_are_localised():
    assert (t("cli.panel.chat"), t("cli.panel.data"), t("cli.panel.connections")) == ("Conversa", "Dados", "Conexões")


def test_epilog_lists_env_vars():
    output = runner.invoke(cli.app, ["--help"]).output
    assert "Variáveis de ambiente" in output
    for name in ENV_VARS:
        assert name in output
    assert "Variáveis de ambiente" not in runner.invoke(cli.app, ["sync", "--help"]).output  # only the main help


def test_help_en():
    result = _cli_subprocess("en", "--help")
    assert result.returncode == 0, result.stderr
    assert "Environment variables" in result.stdout
    for name in ENV_VARS:
        assert name in result.stdout
    for title in ("Chat", "Data", "Connections"):
        assert f"─ {title} ─" in result.stdout
    assert "Assistente" not in result.stdout
    assert "Variáveis" not in result.stdout
    assert "Conexões" not in result.stdout


def test_ask_output_has_no_ansi(env):
    seed(env)
    use_llm(
        env, FakeLLM([tool_turn(ToolCall("c", "detect_subscriptions", {"months": 6})), answer("Você assina Netflix.")])
    )
    result = run(env, "ask", "Quais assinaturas eu tenho?")
    assert result.exit_code == 0, result.output
    assert "\x1b" not in result.output
    assert "Você assina Netflix." in result.output
    assert "consultando detect_subscriptions(months=6)" in result.output


def test_output_has_no_ansi_even_when_the_runner_forces_colour(env):
    # CliRunner(color=True) made the old typer.secho emit ANSI; every ``ui`` helper decides from the terminal itself.
    seed(env)
    use_llm(env, FakeLLM([answer("Você assina Netflix.")]))
    argv = ["--db", env["db"], "ask", "x"]
    for args in (argv, ["--db", env["db"], "status"], ["--db", env["db"], "items", "list"]):
        result = runner.invoke(cli.app, args, color=True)
        assert result.exit_code == 0, result.output
        assert "\x1b" not in result.output, args


def test_status_prints_plain_lines_off_a_terminal(env):
    seed(env)
    with FinanceStore(env["db"]) as store:
        store.set_meta("last_sync", "2026-11-21T10:00:00")
    output = run(env, "status").output
    lines = output.splitlines()
    assert lines[0] == f"Banco local: {env['db']}"
    assert lines[1] == "Última sincronização: 2026-11-21T10:00:00"
    assert lines[2].startswith("Dados: items=1, accounts=2, transactions=")
    assert lines[3] == "- item-1  MeuPluggy  status=UPDATED  consentimento até 01/06/2027"
    assert len(lines) == 4
    assert "╭" not in output
    assert "│" not in output


def test_items_list_prints_plain_lines_off_a_terminal(env):
    seed(env)
    run(env, "items", "add", "a1", "--no-verify")
    lines = run(env, "items", "list").output.splitlines()
    assert sorted(lines) == sorted(
        ["item-1  MeuPluggy  UPDATED", "a1  ?  não sincronizado"]
    )  # same added_at: any order


@pytest.mark.parametrize("width", [80, 120])
def test_status_tables_on_a_terminal_keep_the_asserted_substrings(env, monkeypatch, width):
    seed(env)
    run(env, "items", "add", LONG_ID, "--no-verify")
    buffer = forced_terminal(monkeypatch, width)
    result = run(env, "status")
    out = ANSI.sub("", buffer.getvalue())
    assert result.exit_code == 0, result.output
    assert "╭" in out
    assert all(len(line) <= width for line in out.splitlines())
    assert "…" not in out  # no cell was cropped
    assert "transactions=" in out
    assert "consentimento até 01/06/2027" in out
    assert "UPDATED" in out
    assert "MeuPluggy" in out
    assert "Banco local:" in out
    assert "Nenhuma conexão" not in out
    first_column = "".join(line.split("│")[1].strip() for line in out.splitlines() if line.startswith("│"))
    assert LONG_ID in first_column  # the long id may fold at 80 columns, none of it is lost


@pytest.mark.parametrize("width", [80, 120])
def test_items_list_table_on_a_terminal_keeps_the_asserted_substrings(env, monkeypatch, width):
    run(env, "items", "add", LONG_ID, "--no-verify")
    buffer = forced_terminal(monkeypatch, width)
    result = run(env, "items", "list")
    out = ANSI.sub("", buffer.getvalue())
    assert result.exit_code == 0, result.output
    assert "╭" in out
    assert all(len(line) <= width for line in out.splitlines())
    assert "não sincronizado" in out
    first_column = "".join(line.split("│")[1].strip() for line in out.splitlines() if line.startswith("│"))
    assert LONG_ID in first_column


def test_status_without_connections_on_a_terminal_has_no_items_table(env, monkeypatch):
    buffer = forced_terminal(monkeypatch, 80)
    assert run(env, "status").exit_code == 0
    out = ANSI.sub("", buffer.getvalue())
    assert "Nenhuma conexão registrada." in out
    assert "Consentimento" not in out  # no items table, only the counts table


def test_chat_shows_banner_with_language(env):
    seed(env)
    use_llm(ellmopenfinanceLM([], provider="ollama"))
    output = run(env, "chat", "--no-sync", input="sair\n").output
    assert f"llmopenfinance {__versillmopenfinance output
    assert "Provedor: ollama, Modelo: fake-model, Idioma: pt-BR" in output
    banner_at = output.index(f"llmopenfinance {__version__}")
    assert banner_at < output.index("Assistente financeiro pronto.")
    assert output.index("Verificando conexão") < banner_at  # the connection notices come first


def test_chat_banner_follows_the_language(env, use_language: UseLanguage):
    use_language("en")
    seed(env)
    use_llm(env, FakeLLM([], provider="claude"))
    output = run(env, "chat", "--no-sync", input="exit\n").output
    assert "Provider: claude, Model: fake-model, Language: en" in output
    assert "Provedor" not in output


def test_chat_banner_is_a_panel_on_a_terminal(env, monkeypatch):
    seed(env)
    use_llm(env, FakeLLM([], provider="ollama"))
    buffer = forced_terminal(monkeypatch, 80)
    assert run(env, "chat", "--no-sync", input="sair\n").exit_code == 0
    out = ANSllmopenfinance buffer.getvalue())
    assert "╭" in out
    assert f"llmopenfinance {__version__}" in out
    for text in ("Provedor", "ollama", "Modelo", "fake-model", "Idioma", "pt-BR"):
        assert text in out
    assert "Assistente financeiro pronto." in out


@pytest.mark.parametrize(
    "args",
    [("ask", "x"), ("status",), ("sync",), ("items", "list"), ("items", "add", "a1", "--no-verify"), ("connect",)],
    ids=lambda args: args[0] + ("-" + args[1] if args[0] == "items" else ""),
)
def test_banner_not_shown_for_ask_and_status(env, args):
    seed(env)
    use_llm(env, FakeLLM([answer("ok")]))
    env["monkllmopenfinance.delenv("PLUGGY_USERNAME")  # ``sync`` and ``connect`` stop at once, which is all that matters
    output = run(env, *args).output
    assert f"llmopenfinance {__version__}" not in output
    assert "Provedor:" not in output
    assert "Modelo:" not in output


def test_chat_prompt_label_is_styled_on_a_terminal(env, monkeypatch):
    seed(env)
    use_llm(env, FakeLLM([]))
    labels: list[str] = []

    def fake_prompt(label):
        labels.append(label)
        return "sair"

    monkeypatch.setattr(cli.typer, "prompt", fake_prompt)
    forced_terminal(monkeypatch, 80)
    assert run(env, "chat", "--no-sync").exit_code == 0
    assert len(labels) == 1
    assert "\x1b[" in labels[0]
    assert ANSI.sub("", labels[0]) == "\nVocê"


def test_chat_prompt_label_is_plain_off_a_terminal(env, monkeypatch):
    seed(env)
    use_llm(env, FakeLLM([]))
    labels: list[str] = []

    def fake_prompt(label):
        labels.append(label)
        return "sair"

    monkeypatch.setattr(cli.typer, "prompt", fake_prompt)
    assert run(env, "chat", "--no-sync").exit_code == 0
    assert labels == ["\nVocê"]


def test_render_answer_delegates_to_ui(monkeypatch):
    seen: list[object] = []

    class Assistant:
        def ask(self, question):
            return [question]

    monkeypatch.setattr(ui, "render_answer", lambda events: seen.append(events) or False)
    assert cli._render_answer(Assistant(), "q") is False
    assert seen == [["q"]]


# ---------------------------------------------------------------------------
# ``server`` command (nothing binds a socket or opens a browser)
# ---------------------------------------------------------------------------

TOKEN_URL = re.compile(r"http://[\w.\[\]:]+/\?token=([\w-]+)")


class Served:
    """Records the ``waitress.serve`` call; ``error`` is raised instead of serving."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.error: BaseException | None = None

    def __call__(self, app, **kwargs):
        self.calls.append((app, kwargs))
        if self.error is not None:
            raise self.error


@pytest.fixture
def served(monkeypatch) -> Served:
    import waitress

    recorder = Served()
    monkeypatch.setattr(waitress, "serve", recorder)
    return recorder


@pytest.fixture
def browser(monkeypatch) -> list[str]:
    opened: list[str] = []
    monkeypatch.setattr(cli, "_open_browser", lambda url, *_a, **_k: opened.append(url) or True)
    return opened


@pytest.fixture
def server_env(env, served, browser):
    seed(env)
    use_llm(env, FakeLLM([]))
    return env


def server_token(output: str) -> str:
    match = TOKEN_URL.search(output)
    assert match, output
    return match.group(1)


def test_server_starts_and_prints_tokened_url(server_env, served, browser):
    result = run(server_env, "server", "--no-sync", "--port", "8123", "--threads", "3")
    assert result.exit_code == 0, result.output
    token = server_token(result.output)
    assert f"http://127.0.0.1:8123/?token={token}" in result.output
    assert result.output.count(token) == 1  # printed once
    assert len(served.calls) == 1
    app, kwargs = served.calls[0]
    assert app.name == "llmopenfinance.web"
    assert (kwargs["host"], kwargs["port"], kwargs["threads"]) == ("127.0.0.1", 8123, 3)
    assert kwargs["max_request_body_size"] > 64 * 1024
    assert browser == [f"http://127.0.0.1:8123/?token={token}"]
    assert f"{t('ui.banner.provider')}:" not in result.output  # no banner for ``server`` (only ``chat`` shows it)


def test_server_defaults(server_env, served):
    assert run(server_env, "server", "--no-sync").exit_code == 0
    kwargs = served.calls[0][1]
    assert (kwargs["host"], kwargs["port"], kwargs["threads"]) == ("127.0.0.1", 8000, 4)


def test_server_no_browser_flag(server_env, browser):
    assert run(server_env, "server", "--no-sync", "--no-browser").exit_code == 0
    assert browser == []
    assert run(server_env, "server", "--no-sync").exit_code == 0
    assert len(browser) == 1


def test_server_ignores_browser_failures(server_env, served, monkeypatch):
    def broken(*_a, **_k):
        raise OSError("no browser")

    monkeypatch.setattr(cli, "_open_browser", broken)
    assert run(server_env, "server", "--no-sync").exit_code == 0
    assert len(served.calls) == 1
    monkeypatch.setattr(cli, "_open_browser", lambda *_a, **_k: False)
    assert run(server_env, "server", "--no-sync").exit_code == 0


def test_server_missing_extra(server_env, served, monkeypatch):
    run(server_env, "items", "add", ITEM_ID, "--no-verify")
    monkeypatch.setitem(sys.modules, "flask", None)
    monkeypatch.delitem(sys.modules, "llmopenfinance.web", raising=False)
    result = run(server_env, "server")  # sync is on: it must not even start
    assert result.exit_code == 1
    assert "--extra web" in result.output
    assert "Traceback" not in result.output
    assert server_env["pluggy"].calls == []
    assert served.calls == []


def test_server_llm_unreachable_exits_1(env, served, browser):
    seed(env)
    use_llm(env, FakeLLM([], provider="ollama", reachable=False))
    result = run(env, "--provider", "ollama", "server", "--no-sync")
    assert result.exit_code == 1
    assert "ollama serve" in result.output
    assert served.calls == []
    assert browser == []


def test_server_llm_misconfigured_exits_1(env, served):
    seed(env)

    def boom(*_a, **_k):
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

    env["monkeypatch"].setattr(cli, "_llm", boom)
    result = run(env, "server", "--no-sync")
    assert result.exit_code == 1
    assert "ANTHROPIC_API_KEY" in result.output
    assert served.calls == []


def test_server_port_in_use_is_clean(server_env, served):
    served.error = OSError("Address already in use")
    result = run(server_env, "server", "--no-sync", "--port", "8123")
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    lines = [line for line in result.output.splitlines() if "Address already in use" in line]
    assert len(lines) == 1
    assert "127.0.0.1" in lines[0]
    assert "8123" in lines[0]


def test_server_non_loopback_warning(server_env, served):
    result = run(server_env, "server", "--no-sync", "--host", "0.0.0.0")
    assert result.exit_code == 0, result.output
    assert "HTTP" in result.output
    assert "somente o token" in result.output
    assert f"http://127.0.0.1:8000/?token={server_token(result.output)}" in result.output
    assert "0.0.0.0" not in next(line for line in result.output.splitlines() if "?token=" in line)
    assert served.calls[0][1]["host"] == "0.0.0.0"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_server_loopback_host_has_no_warning(server_env, host):
    result = run(server_env, "server", "--no-sync", "--host", host)
    assert result.exit_code == 0, result.output
    assert "HTTP" not in result.output


def test_server_specific_non_loopback_host_warns_without_wildcard_text(server_env):
    result = run(server_env, "server", "--no-sync", "--host", "192.168.1.20")
    assert "http://192.168.1.20:8000/?token=" in result.output
    assert "HTTP" in result.output
    assert "somente o token" not in result.output


def test_server_ipv6_hosts_are_bracketed_in_the_url(server_env):
    assert "http://[::1]:8000/?token=" in run(server_env, "server", "--no-sync", "--host", "::1").output
    wildcard = run(server_env, "server", "--no-sync", "--host", "::").output
    assert "http://127.0.0.1:8000/?token=" in wildcard


def test_server_syncs_before_serving(server_env, served):
    run(server_env, "items", "add", ITEM_ID, "--no-verify")
    result = run(server_env, "server", "--no-browser")
    assert result.exit_code == 0, result.output
    assert "Sincronização concluída" in result.output
    assert "authenticate" in server_env["pluggy"].calls
    assert len(served.calls) == 1
    server_env["pluggy"].calls.clear()
    assert run(server_env, "server", "--no-sync", "--no-browser").exit_code == 0
    assert server_env["pluggy"].calls == []


def test_server_warns_about_empty_store(env, served):
    use_llm(env, FakeLLM([]))
    result = run(env, "server", "--no-sync", "--no-browser")
    assert result.exit_code == 0, result.output
    assert "Ainda não há transações" in result.output


def test_server_en_messages(server_env, served, use_language: UseLanguage):
    use_language("en")
    result = run(server_env, "server", "--no-sync", "--host", "0.0.0.0", "--port", "8123")
    assert result.exit_code == 0, result.output
    assert "plain HTTP" in result.output
    assert "Open in your browser" in result.output
    served.error = OSError("Address already in use")
    failed = run(server_env, "server", "--no-sync", "--port", "8123")
    assert failed.exit_code == 1
    assert "Could not listen on 127.0.0.1:8123" in failed.output


def test_server_en_missing_extra(server_env, monkeypatch, use_language: UseLanguage):
    use_language("en")
    monkeypatch.setitem(sys.modules, "flask", None)
    monkeypatch.delitem(sys.modules, "llmopenfinance.web", raising=False)
    result = run(server_env, "server")
    assert result.exit_code == 1
    assert "uv sync --extra web" in result.output


def test_server_context_failure_is_clean(server_env, served, monkeypatch):
    from llmopenfinance import web

    def boom(**_kwargs):
        raise RuntimeError("store-exploded")

    monkeypatch.setattr(web, "WebContext", boom)
    result = run(server_env, "server", "--no-sync")
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert len([line for line in result.output.splitlines() if "store-exploded" in line]) == 1
    assert served.calls == []


def test_server_token_not_logged(server_env, served, caplog):
    run(server_env, "items", "add", ITEM_ID, "--no-verify")
    caplog.set_level(logging.DEBUG)
    result = run(server_env, "server", "--no-browser")  # the sync logs too
    token = server_token(result.output)
    assert caplog.records  # not vacuous: something was logged at DEBUG
    assert token not in caplog.text
    for record in caplog.records:
        assert token not in record.getMessage()
        assert token not in str(record.args)


def test_server_token_differs_per_launch(server_env):
    first = server_token(run(server_env, "server", "--no-sync").output)
    assert first != server_token(run(server_env, "server", "--no-sync").output)


@pytest.mark.parametrize("args", [["--port", "0"], ["--port", "70000"], ["--threads", "1"], ["--port", "x"]])
def test_server_options_validation(server_env, served, args):
    result = run(server_env, "server", "--no-sync", *args)
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    assert served.calls == []


def test_server_in_help_chat_panel():
    panels = help_panels(runner.invoke(cli.app, ["--help"]).output)
    assert re.search(r"^│ server\s", panels[t("cli.panel.chat")], flags=re.MULTILINE)
    help_text = runner.invoke(cli.app, ["server", "--help"]).output
    for option in ("--host", "--port", "--threads", "--no-browser", "--no-sync", "--think", "--no-sql"):
        assert option in help_text


def test_server_keyboard_interrupt_exits_0(server_env, served):
    served.error = KeyboardInterrupt()
    result = run(server_env, "server", "--no-sync")
    assert result.exit_code == 0
    assert "Até logo!" in result.output
    assert "Traceback" not in result.output


class SpySession(cli._ChatSession):
    instances: ClassVar[list["SpySession"]] = []

    def __init__(self, open_store):
        super().__init__(open_store)
        self.closed = 0
        SpySession.instances.append(self)

    def close(self):
        self.closed += 1
        super().close()


@pytest.mark.parametrize("path", ["ok", "interrupt", "port", "context", "llm"])
def test_server_closes_session(server_env, served, monkeypatch, path):
    from llmopenfinance import web

    def boom(**_kwargs):
        raise RuntimeError("store-exploded")

    SpySession.instances.clear()
    monkeypatch.setattr(cli, "_ChatSession", SpySession)
    if path == "interrupt":
        served.error = KeyboardInterrupt()
    elif path == "port":
        served.error = OSError("in use")
    elif path == "context":
        monkeypatch.setattr(web, "WebContext", boom)
    elif path == "llm":
        use_llm(server_env, FakeLLM([], reachable=False))
    run(server_env, "server", "--no-sync")
    assert [session.closed for session in SpySession.instances] == [1]
