"""End-to-end CLI tests with Typer's CliRunner, a fake Pluggy client and a scripted LLM."""

import os
import subprocess  # ruff: ignore[suspicious-subprocess-import] — the startup guard needs a fresh interpreter
import sys
from collections.abc import Callable

import pytest
from typer.testing import CliRunner

from llmopenfinance import __main__ as cli
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


def test_main_reconfigures_streams(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["finassist", "ver"])
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
    use_language("en")
    no_items = run(env, "sync")
    assert "No connections registered. Use 'finassist connect' or 'finassist items add <item-id>'." in no_items.output
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
    result = run(env, "sync")
    assert result.exit_code == 1
    assert "Local database in use by another finassist session (" in result.output
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
    use_llm(env, FakeLLM([]))
    result = run(env, "chat", "--no-sync", input="/sync\n")
    assert "No local transactions yet. Connect an account with 'finassist connect'" in result.output
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
        "cli.sync.db_locked",
        {"error": "lock"},
        "Banco local em uso por outra sessão do finassist (lock); sincronização ignorada.",
    ),
    (
        "cli.sync.no_items",
        {},
        "Nenhuma conexão registrada. Use 'finassist connect' ou 'finassist items add <item-id>'.",
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
        {},
        (
            "Ainda não há transações locais. Conecte uma conta com 'finassist connect' ou "
            "'finassist items add <item-id>' e rode 'finassist sync'."
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
