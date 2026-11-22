#!/usr/bin/env python3
"""llmopenfinance — command line interface.

Console chat with an LLM financial assistant (Claude or Ollama) over the user's Open Finance
data (Pluggy), cached in a local DuckDB database.

Commands: ``chat`` (default), ``ask``, ``sync``, ``status``, ``connect``, ``items add|list|remove``.
Structure follows ``apexmusic.__main__``: a global ``_state`` filled by the Typer callback,
``_handle_*`` functions holding the logic (return exit codes) and thin ``@app.command`` wrappers.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer

from llmopenfinance.constants import (
    DEFAULT_DB_PATH,
    DEFAULT_PROVIDER,
    ENV_DB_PATH,
    ENV_PLUGGY_CLIENT_ID,
    ENV_PLUGGY_CLIENT_SECRET,
    ENV_PLUGGY_ITEM_IDS,
    ENV_PROVIDER,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
__all__ = ["app", "main"]


# ---------------------------------------------------------------------------
# Global state — populated by the @app.callback before each command
# ---------------------------------------------------------------------------


@dataclass
class _State:
    verbose: bool = False
    provider: str = DEFAULT_PROVIDER
    model: str = ""
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)


_state = _State()

app = typer.Typer(
    name="finassist",
    help="Assistente financeiro no terminal — converse com seus dados de Open Finance (Pluggy).",
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode="rich",
)
items_app = typer.Typer(help="Gerencia as conexões (itens) Pluggy sincronizadas.", no_args_is_help=True)
app.add_typer(items_app, name="items")


@app.callback()
def _configure(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Saída detalhada (logs INFO)")] = False,
    provider: Annotated[
        str, typer.Option("--provider", "-p", help="Provedor de IA: claude | ollama", envvar=ENV_PROVIDER)
    ] = DEFAULT_PROVIDER,
    model: Annotated[str, typer.Option("--model", "-m", help="Modelo (vazio = padrão do provedor)")] = "",
    db: Annotated[str, typer.Option("--db", help="Arquivo DuckDB local", envvar=ENV_DB_PATH)] = "",
    version: Annotated[bool, typer.Option("--version", help="Mostra a versão")] = False,
) -> None:
    """Assistente financeiro no terminal (Pluggy + Claude/Ollama)."""
    if version:
        from llmopenfinance import __version__

        typer.echo(f"llmopenfinance {__version__}")
        raise typer.Exit(code=0)
    _state.verbose = verbose
    _state.provider = provider
    _state.model = model
    _state.db_path = Path(db) if db else DEFAULT_DB_PATH
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if ctx.invoked_subcommand is None:
        raise typer.Exit(code=_handle_chat())


# ---------------------------------------------------------------------------
# Factories — monkeypatched in tests
# ---------------------------------------------------------------------------


def _pluggy_client() -> Any:
    """Create a Pluggy client from the environment.

    Returns:
        Any: ``PluggyClient`` instance.
    """
    from llmopenfinance.pluggy import PluggyClient

    return PluggyClient.from_env()


def _llm(provider: str, model: str, *, think: bool = False) -> Any:
    """Instantiate the configured LLM chat backend.

    Args:
        provider: ``claude`` or ``ollama``.
        model: Model id (empty = default).
        think: Enable thinking for Ollama models.

    Returns:
        Any: ``AbstractChat`` instance.
    """
    from llmopenfinance.chat import get_llm

    kwargs: dict[str, Any] = {"think": think} if provider == "ollama" else {}
    return get_llm(provider, model, **kwargs)


def _has_pluggy_credentials() -> bool:
    return bool(os.environ.get(ENV_PLUGGY_CLIENT_ID) and os.environ.get(ENV_PLUGGY_CLIENT_SECRET))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_store(*, read_only: bool = False) -> Any:
    """Open the local store; the read-write open also registers ``PLUGGY_ITEM_IDS``.

    Args:
        read_only: Open the sandboxed read-only connection.

    Returns:
        Any: ``FinanceStore`` instance.
    """
    from llmopenfinance.store import FinanceStore

    if read_only:
        if not _state.db_path.exists():
            FinanceStore(_state.db_path).close()
        return FinanceStore(_state.db_path, read_only=True)
    store = FinanceStore(_state.db_path)
    for item_id in os.environ.get(ENV_PLUGGY_ITEM_IDS, "").replace(";", ",").split(","):
        if item_id.strip() and store.add_item(item_id.strip()):
            logger.info("Registered item %s from %s", item_id.strip(), ENV_PLUGGY_ITEM_IDS)
    return store


def _echo_dim(text: str) -> None:
    typer.secho(text, fg=typer.colors.BRIGHT_BLACK)


def _print_item_result(result: Any) -> None:
    if not result.ok:
        typer.secho(f"  ✗ {result.item_id}: {result.error}", fg=typer.colors.RED)
    elif result.skipped:
        _echo_dim(f"  = {result.item_id}: sem alterações desde a última sincronização")
    else:
        counts = ", ".join(f"{k}={v}" for k, v in result.counts.items())
        typer.secho(f"  ✓ {result.item_id}: {counts}", fg=typer.colors.GREEN)
    for warning in result.warnings:
        typer.secho(f"    ! {warning}", fg=typer.colors.YELLOW)


def _run_sync(*, force: bool = False) -> int:
    """Synchronize every registered item into the local store.

    Args:
        force: Re-download unchanged items.

    Returns:
        int: Exit code (0 when at least one item synced or nothing to do).
    """
    from llmopenfinance.pluggy import PluggyError
    from llmopenfinance.sync import sync_all

    if not _has_pluggy_credentials():
        typer.secho(
            f"Defina {ENV_PLUGGY_CLIENT_ID} (clientId) e {ENV_PLUGGY_CLIENT_SECRET} (clientSecret) "
            "da sua aplicação em dashboard.pluggy.ai.",
            fg=typer.colors.RED,
        )
        return 1
    import duckdb

    try:
        store = _open_store()
    except duckdb.IOException as exc:
        typer.secho(
            f"Banco local em uso por outra sessão do finassist ({exc}); sincronização ignorada.",
            fg=typer.colors.YELLOW,
        )
        return 1
    with store:
        if not store.list_items():
            typer.echo("Nenhuma conexão registrada. Use 'finassist connect' ou 'finassist items add <item-id>'.")
            return 0
        try:
            client = _pluggy_client()
            client.authenticate()
        except (PluggyError, ValueError) as exc:
            typer.secho(f"Falha ao autenticar na Pluggy: {exc}", fg=typer.colors.RED)
            return 1
        typer.echo("Sincronizando dados da Pluggy…")
        try:
            report = sync_all(client, store, force=force, on_item=_print_item_result)
        finally:
            client.close()
    typer.echo(f"Sincronização concluída: {report.n_success} ok, {report.n_failed} com falha.")
    return 0 if report.n_success or not report.n_total else 1


class _ChatSession:
    """Holds the read-only store and lets ``/sync`` close/reopen it (DuckDB allows one RW handle)."""

    def __init__(self) -> None:
        self.store = _open_store(read_only=True)

    def connection(self) -> Any:
        return self.store.con

    def snapshot(self) -> str:
        from llmopenfinance.analytics import build_snapshot

        return build_snapshot(self.store.con)

    def resync(self, *, force: bool = False) -> int:
        self.store.close()
        try:
            return _run_sync(force=force)
        finally:
            self.store = _open_store(read_only=True)

    def close(self) -> None:
        self.store.close()


def _build_assistant(session: _ChatSession, *, think: bool, enable_sql: bool | None) -> Any:
    """Create the LLM backend + tools + assistant, verifying the LLM connection.

    Args:
        session: Chat session (read-only store).
        think: Enable Ollama thinking.
        enable_sql: Expose ``run_sql`` (``None`` = on for Claude, off for Ollama).

    Returns:
        Any: ``FinanceAssistant`` or ``None`` when the LLM is unreachable.
    """
    from llmopenfinance.agent import FinanceAssistant
    from llmopenfinance.tools import ToolRegistry

    try:
        llm = _llm(_state.provider, _state.model, think=think)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        return None
    typer.echo(f"Verificando conexão com {llm.provider} ({llm.model})…")
    if not llm._verify_connection():
        typer.secho(f"Não foi possível conectar a {llm.provider} / {llm.model}.", fg=typer.colors.RED)
        if llm.provider == "ollama":
            typer.echo(f"Rode: ollama serve  e  ollama pull {llm.model}")
        else:
            typer.echo("Verifique ANTHROPIC_API_KEY e o nome do modelo.")
        return None
    if llm.provider == "claude":
        _echo_dim(
            "Aviso: com Claude, o resumo financeiro e os resultados das consultas são enviados à API da Anthropic. "
            "Use --provider ollama para manter tudo local."
        )
    sql = enable_sql if enable_sql is not None else llm.provider == "claude"
    registry = ToolRegistry(session.connection, enable_sql=sql)
    return FinanceAssistant(llm, registry, session.snapshot)


def _render_answer(assistant: Any, question: str) -> bool:
    """Stream an answer to the terminal.

    Args:
        assistant: ``FinanceAssistant``.
        question: User question.

    Returns:
        bool: ``False`` if the answer failed (an error was shown).
    """
    from llmopenfinance.agent import AssistantError, ToolStatus
    from llmopenfinance.chat import TextDelta

    ok = True
    typer.echo("\nAssistente: ", nl=False)
    for event in assistant.ask(question):
        if isinstance(event, TextDelta):
            typer.echo(event.text, nl=False)
        elif isinstance(event, ToolStatus):
            args = ", ".join(f"{k}={v!r}" for k, v in event.call.arguments.items())
            _echo_dim(f"\n  ↳ consultando {event.call.name}({args[:120]})")
        elif isinstance(event, AssistantError):
            ok = False
            typer.secho(f"\n[erro] {event.message}", fg=typer.colors.RED)
    typer.echo("\n")
    return ok


HELP_TEXT = """Comandos: /ajuda  /resumo (dados enviados ao modelo)  /sync  /reset  /historico  sair
Exemplos: "Quais assinaturas eu tenho?"  "Quanto gasto por mês com compras online?"
          "Como posso melhorar minhas finanças?\""""


def _process_chat_cmd(user_input: str, assistant: Any, session: _ChatSession) -> bool | None:
    """Dispatch a special chat command.

    Args:
        user_input: Raw user text (already stripped).
        assistant: Active ``FinanceAssistant``.
        session: Chat session.

    Returns:
        ``True`` if a command was handled and the loop should continue, ``False`` if the user wants
        to exit, ``None`` if the input should go to the assistant.
    """
    command = user_input.lower()
    if command in EXIT_COMMANDS:
        typer.echo("\nAté logo!")
        return False
    handler = _CHAT_COMMANDS.get(command)
    if handler is None:
        return None
    handler(assistant, session)
    return True


def _cmd_reset(assistant: Any, _session: _ChatSession) -> None:
    assistant.refresh()
    typer.echo("Conversa reiniciada.")


def _cmd_help(_assistant: Any, _session: _ChatSession) -> None:
    typer.echo(HELP_TEXT)


def _cmd_summary(_assistant: Any, session: _ChatSession) -> None:
    typer.echo(session.snapshot())


def _cmd_sync(assistant: Any, session: _ChatSession) -> None:
    code = session.resync()
    assistant.refresh()
    if code == 0:
        typer.echo("Dados atualizados; conversa reiniciada.")
    else:
        typer.secho("Sincronização falhou; usando os dados locais (conversa reiniciada).", fg=typer.colors.YELLOW)


def _cmd_history(assistant: Any, _session: _ChatSession) -> None:
    history = assistant.chat.get_history()
    if not history:
        typer.echo("(vazio)")
    for i, msg in enumerate(history, 1):
        content = msg.get("content")
        text = content if isinstance(content, str) else str(content)
        snippet = text[:100] + ("…" if len(text) > 100 else "")
        typer.echo(f"{i}. [{msg['role'].upper()}] {snippet}")


EXIT_COMMANDS = frozenset({"sair", "quit", "exit", "bye", "q", "/sair"})
_CHAT_COMMANDS: dict[str, Callable[[Any, _ChatSession], None]] = {
    "/reset": _cmd_reset,
    "reset": _cmd_reset,
    "/ajuda": _cmd_help,
    "/help": _cmd_help,
    "ajuda": _cmd_help,
    "help": _cmd_help,
    "/resumo": _cmd_summary,
    "/sync": _cmd_sync,
    "/historico": _cmd_history,
    "history": _cmd_history,
}


def _handle_input(user_input: str, assistant: Any, session: _ChatSession) -> bool:
    """Handle one line typed by the user.

    Args:
        user_input: Raw input.
        assistant: ``FinanceAssistant``.
        session: Chat session.

    Returns:
        bool: ``False`` when the user asked to exit.
    """
    text = user_input.strip()
    if not text:
        return True
    result = _process_chat_cmd(text, assistant, session)
    if result is None:
        _render_answer(assistant, text)
    return result is not False


def _chat_loop(assistant: Any, session: _ChatSession, prompt: Callable[[str], str] = typer.prompt) -> int:
    """Run the interactive chat loop until the user exits.

    Args:
        assistant: ``FinanceAssistant``.
        session: Chat session.
        prompt: Input function (injectable for tests).

    Returns:
        int: Exit code (always 0 — errors are logged and the loop continues).
    """
    while True:
        try:
            if not _handle_input(prompt("\nVocê"), assistant, session):
                return 0
        except KeyboardInterrupt:
            typer.echo("\n(interrompido — digite 'sair' para encerrar)")
            continue
        except (EOFError, typer.Abort):
            typer.echo("\nAté logo!")
            return 0
        except Exception:
            logger.exception("Chat error")
            continue


# ---------------------------------------------------------------------------
# Handlers (logic; return exit codes)
# ---------------------------------------------------------------------------


def _maybe_sync(*, sync: bool) -> None:
    if not sync:
        return
    if not _has_pluggy_credentials():
        _echo_dim("Credenciais Pluggy ausentes — usando apenas os dados locais.")
        return
    _run_sync()


def _handle_chat(*, sync: bool = True, think: bool = False, enable_sql: bool | None = None) -> int:
    """Run the interactive chat session.

    Args:
        sync: Synchronize Pluggy data before chatting.
        think: Enable Ollama thinking.
        enable_sql: Expose ``run_sql`` (``None`` = provider default).

    Returns:
        int: Exit code.
    """
    _maybe_sync(sync=sync)
    session = _ChatSession()
    try:
        assistant = _build_assistant(session, think=think, enable_sql=enable_sql)
        if assistant is None:
            return 1
        if session.store.counts()["transactions"] == 0:
            typer.secho(
                "Ainda não há transações locais. Conecte uma conta com 'finassist connect' ou "
                "'finassist items add <item-id>' e rode 'finassist sync'.",
                fg=typer.colors.YELLOW,
            )
        typer.echo("\nAssistente financeiro pronto. " + HELP_TEXT.splitlines()[0])
        typer.echo("=" * 70)
        return _chat_loop(assistant, session)
    finally:
        session.close()


def _handle_ask(question: str, *, sync: bool = False, enable_sql: bool | None = None) -> int:
    """Answer one question non-interactively.

    Args:
        question: The question.
        sync: Synchronize first.
        enable_sql: Expose ``run_sql``.

    Returns:
        int: Exit code.
    """
    _maybe_sync(sync=sync)
    session = _ChatSession()
    try:
        assistant = _build_assistant(session, think=False, enable_sql=enable_sql)
        if assistant is None:
            return 1
        return 0 if _render_answer(assistant, question) else 1
    finally:
        session.close()


def _handle_status() -> int:
    """Show registered items, their health and local data counts.

    Returns:
        int: Exit code.
    """
    with _open_store(read_only=True) as store:
        items = store.list_items()
        counts = store.counts()
        last_sync = store.get_meta("last_sync")
    typer.echo(f"Banco local: {_state.db_path}")
    typer.echo(f"Última sincronização: {last_sync or 'nunca'}")
    typer.echo("Dados: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if not items:
        typer.echo("Nenhuma conexão registrada.")
    for item in items:
        line = f"- {item['id']}  {item.get('connector_name') or '?'}  status={item.get('status') or '?'}"
        expires = item.get("consent_expires_at")
        if expires:
            line += f"  consentimento até {expires:%d/%m/%Y}"
        typer.echo(line)
    return 0


def _handle_items_add(item_ids: list[str], *, verify: bool = True) -> int:
    """Register item ids (and optionally validate them against Pluggy).

    Args:
        item_ids: Pluggy item ids.
        verify: Call ``GET /items/{id}`` to validate each id.

    Returns:
        int: Exit code (1 if any id is invalid).
    """
    from llmopenfinance.pluggy import PluggyError

    failed = 0
    client = _pluggy_client() if verify and _has_pluggy_credentials() else None
    with _open_store() as store:
        for item_id in item_ids:
            if client is not None:
                try:
                    item = client.get_item(item_id)
                except PluggyError as exc:
                    typer.secho(f"✗ {item_id}: {exc}", fg=typer.colors.RED)
                    failed += 1
                    continue
                store.add_item(item_id)
                store.update_item(item)
                typer.echo(f"✓ {item_id} ({(item.get('connector') or {}).get('name', '?')}, {item.get('status')})")
            else:
                added = store.add_item(item_id)
                typer.echo(f"{'✓' if added else '='} {item_id}{'' if added else ' (já registrado)'}")
    if client is not None:
        client.close()
    return 1 if failed else 0


def _handle_items_list() -> int:
    """List registered items.

    Returns:
        int: Exit code.
    """
    with _open_store(read_only=True) as store:
        items = store.list_items()
    if not items:
        typer.echo("Nenhuma conexão registrada.")
    for item in items:
        typer.echo(f"{item['id']}  {item.get('connector_name') or '?'}  {item.get('status') or 'não sincronizado'}")
    return 0


def _handle_items_remove(item_id: str) -> int:
    """Forget an item and its local data.

    Args:
        item_id: Pluggy item id.

    Returns:
        int: Exit code.
    """
    with _open_store() as store:
        removed = store.remove_item(item_id)
    typer.echo(f"Removido: {item_id}" if removed else f"Item não encontrado: {item_id}")
    return 0 if removed else 1


def _connect_items(client: Any, *, sandbox: bool) -> list[str]:
    """Run the browser Connect flow, persisting each item id as soon as it arrives.

    Args:
        client: Pluggy client.
        sandbox: Show sandbox connectors.

    Returns:
        list[str]: Item ids created.
    """
    from llmopenfinance.connect import run_connect_flow, wait_for_item

    with _open_store() as store:

        def _on_item(item_id: str) -> None:
            store.add_item(item_id)
            typer.echo(f"Conexão recebida: {item_id} — aguardando a sincronização da Pluggy…")
            item = wait_for_item(client, item_id)
            store.update_item(item)
            typer.echo(f"  status: {item.get('status')} ({item.get('executionStatus')})")

        typer.echo("Abrindo o Pluggy Connect no navegador… conclua a conexão e feche o widget.")
        return run_connect_flow(
            client,
            _on_item,
            sandbox=sandbox,
            on_url=lambda url: _echo_dim(f"Se o navegador não abrir, acesse: {url}"),
        )


def _handle_connect(*, sandbox: bool = False, sync: bool = True) -> int:
    """Open the Pluggy Connect widget in the browser and register the created items.

    Args:
        sandbox: Show sandbox connectors (testing).
        sync: Synchronize after connecting.

    Returns:
        int: Exit code.
    """
    from llmopenfinance.pluggy import PluggyError

    if not _has_pluggy_credentials():
        typer.secho(f"Defina {ENV_PLUGGY_CLIENT_ID} e {ENV_PLUGGY_CLIENT_SECRET}.", fg=typer.colors.RED)
        return 1
    client = _pluggy_client()
    try:
        received = _connect_items(client, sandbox=sandbox)
    except PluggyError as exc:
        typer.secho(f"Falha na Pluggy: {exc}", fg=typer.colors.RED)
        return 1
    finally:
        client.close()
    if not received:
        typer.echo("Nenhuma conexão criada.")
        return 1
    return _run_sync() if sync else 0


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------


@app.command()
def chat(
    sync: Annotated[bool, typer.Option("--sync/--no-sync", help="Sincronizar antes de conversar")] = True,
    think: Annotated[bool, typer.Option("--think", help="Ativa o raciocínio (thinking) no Ollama")] = False,
    sql: Annotated[
        bool | None, typer.Option("--sql/--no-sql", help="Permite a ferramenta run_sql (padrão: só no Claude)")
    ] = None,
) -> None:
    """Conversa interativa com o assistente financeiro (comando padrão)."""
    raise typer.Exit(code=_handle_chat(sync=sync, think=think, enable_sql=sql))


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Pergunta")],
    sync: Annotated[bool, typer.Option("--sync/--no-sync", help="Sincronizar antes")] = False,
    sql: Annotated[bool | None, typer.Option("--sql/--no-sql", help="Permite run_sql")] = None,
) -> None:
    """Faz uma pergunta única e sai."""
    raise typer.Exit(code=_handle_ask(question, sync=sync, enable_sql=sql))


@app.command()
def sync(force: Annotated[bool, typer.Option("--force", help="Baixar tudo mesmo sem alterações")] = False) -> None:
    """Sincroniza contas, transações, faturas, investimentos e empréstimos da Pluggy."""
    raise typer.Exit(code=_run_sync(force=force))


@app.command()
def status() -> None:
    """Mostra conexões registradas, consentimentos e volume de dados locais."""
    raise typer.Exit(code=_handle_status())


@app.command()
def connect(
    sandbox: Annotated[bool, typer.Option("--sandbox", help="Mostrar bancos de teste (sandbox)")] = False,
    sync: Annotated[bool, typer.Option("--sync/--no-sync", help="Sincronizar ao final")] = True,
) -> None:
    """Conecta contas via Pluggy Connect (MeuPluggy) no navegador."""
    raise typer.Exit(code=_handle_connect(sandbox=sandbox, sync=sync))


@items_app.command("add")
def items_add(
    item_ids: Annotated[list[str], typer.Argument(help="Item ids da Pluggy (dashboard → Demo → Connected Items)")],
    verify: Annotated[bool, typer.Option("--verify/--no-verify", help="Validar na Pluggy")] = True,
) -> None:
    """Registra item ids da Pluggy para sincronização."""
    raise typer.Exit(code=_handle_items_add(item_ids, verify=verify))


@items_app.command("list")
def items_list() -> None:
    """Lista as conexões registradas."""
    raise typer.Exit(code=_handle_items_list())


@items_app.command("remove")
def items_remove(item_id: Annotated[str, typer.Argument(help="Item id")]) -> None:
    """Remove uma conexão e seus dados locais."""
    raise typer.Exit(code=_handle_items_remove(item_id))


@app.command("ver")
def version_cmd() -> None:
    """Mostra a versão."""
    from llmopenfinance import __version__

    typer.echo(f"llmopenfinance {__version__}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
