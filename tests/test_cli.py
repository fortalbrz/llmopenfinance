"""End-to-end CLI tests with Typer's CliRunner, a fake Pluggy client and a scripted LLM."""

import sys

import pytest
from typer.testing import CliRunner

from llmopenfinance import __main__ as cli
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
