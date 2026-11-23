#!/usr/bin/env python3
"""llmopenfinance — command line interface.

Console chat with an LLM financial assistant (Claude or Ollama) over the user's Open Finance
data (Pluggy), cached in a local DuckDB database.

Commands: ``chat`` (default), ``ask``, ``sync``, ``status``, ``connect``, ``items add|list|remove``.
Structure follows ``apexmusic.__main__``: a global ``_state`` filled by the Typer callback,
``_handle_*`` functions holding the logic (return exit codes) and thin ``@app.command`` wrappers. The chat/sync
core shared with other front ends lives in :mod:`llmopenfinance.session`; this module renders its notices.
Every user-visible text comes from the message catalog (``t("cli.*")`` / ``t("chat.*")``) in the language of
``LLMOPENFINANCE_LANG``, which is validated before the Typer app is built.
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
    ENV_LANG,
    ENV_PLUGGY_CLIENT_ID,
    ENV_PLUGGY_CLIENT_SECRET,
    ENV_PLUGGY_ITEM_IDS,
    ENV_PROVIDER,
)
from llmopenfinance.i18n import format_date, get_language, t
from llmopenfinance.session import (
    BuildOptions,
    ChatSession,
    Factories,
    Notice,
    build_assistant,
    has_pluggy_credentials,
    parse_command,
    run_command,
    run_sync,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
__all__ = ["app", "main"]


def _resolve_language() -> str:
    """Resolve the process language, or exit with status 2 when ``LLMOPENFINANCE_LANG`` is invalid.

    No language is known when the variable is bad, so the message is plain English and goes to stderr.

    Returns:
        str: ``"pt-BR"`` or ``"en"``.

    Raises:
        SystemExit: With status 2 when the language is not supported.
    """
    try:
        return get_language()
    except ValueError as exc:
        typer.echo(f"{ENV_LANG}: {exc}", err=True)
        raise SystemExit(2) from exc


# Must run before the Typer app below is built: its command help texts are localised at import time.
_resolve_language()


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
    help=t("cli.app.help"),
    invoke_without_command=True,
    add_completion=False,
    rich_markup_mode="rich",
)
items_app = typer.Typer(help=t("cli.items.help"), no_args_is_help=True)
app.add_typer(items_app, name="items")


@app.callback()
def _configure(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help=t("cli.option.verbose"))] = False,
    provider: Annotated[
        str, typer.Option("--provider", "-p", help=t("cli.option.provider"), envvar=ENV_PROVIDER)
    ] = DEFAULT_PROVIDER,
    model: Annotated[str, typer.Option("--model", "-m", help=t("cli.option.model"))] = "",
    db: Annotated[str, typer.Option("--db", help=t("cli.option.db"), envvar=ENV_DB_PATH)] = "",
    version: Annotated[bool, typer.Option("--version", help=t("cli.option.version"))] = False,
) -> None:
    """Record the global options and start the chat when no command is given."""
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


def _factories() -> Factories:
    """Bundle the factories, looked up in this module at call time so monkeypatched ones take effect.

    Returns:
        Factories: The current ``_llm``, ``_open_store`` and ``_pluggy_client``.
    """
    return Factories(llm=_llm, open_store=_open_store, pluggy_client=_pluggy_client)


def _echo_dim(text: str) -> None:
    typer.secho(text, fg=typer.colors.BRIGHT_BLACK)


_NOTICE_COLOURS = {
    "ok": typer.colors.GREEN,
    "warn": typer.colors.YELLOW,
    "error": typer.colors.RED,
    "dim": typer.colors.BRIGHT_BLACK,
}


def _render_notice(notice: Notice) -> None:
    colour = _NOTICE_COLOURS.get(notice.level)
    if colour is None:
        typer.echo(notice.text)
    else:
        typer.secho(notice.text, fg=colour)


def _render_notices(notices: tuple[Notice, ...]) -> None:
    for notice in notices:
        _render_notice(notice)


def _print_item_result(result: Any) -> None:
    if not result.ok:
        typer.secho(f"  ✗ {result.item_id}: {result.error}", fg=typer.colors.RED)
    elif result.skipped:
        _echo_dim(f"  = {result.item_id}: {t('cli.sync.unchanged')}")
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
    return run_sync(_factories(), force=force, notify=_render_notice, on_item=_print_item_result)


_ChatSession = ChatSession


def _build_assistant(session: ChatSession, *, think: bool, enable_sql: bool | None) -> Any:
    """Create the LLM backend + tools + assistant, showing the connection check and any error.

    Args:
        session: Chat session (read-only store).
        think: Enable Ollama thinking.
        enable_sql: Expose ``run_sql`` (``None`` = on for Claude, off for Ollama).

    Returns:
        Any: ``FinanceAssistant`` or ``None`` when the LLM is unreachable.
    """
    build = build_assistant(
        _factories(),
        session,
        provider=_state.provider,
        model=_state.model,
        options=BuildOptions(think=think, enable_sql=enable_sql),
        notify=_render_notice,  # each line as it happens, so "verifying" shows before the slow check
    )
    return build.assistant


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
    typer.echo(f"\n{t('chat.assistant_label')}: ", nl=False)
    for event in assistant.ask(question):
        if isinstance(event, TextDelta):
            typer.echo(event.text, nl=False)
        elif isinstance(event, ToolStatus):
            args = ", ".join(f"{k}={v!r}" for k, v in event.call.arguments.items())
            _echo_dim("\n  " + t("chat.tool_status", name=event.call.name, args=args[:120]))
        elif isinstance(event, AssistantError):
            ok = False
            typer.secho("\n" + t("chat.error", message=event.message), fg=typer.colors.RED)
    typer.echo("\n")
    return ok


def _help_text() -> str:
    """Return the in-chat help in the process language.

    Evaluated on every call so it follows the process language; never cache it in a module constant.

    Returns:
        str: The command list on the first line, then example questions.
    """
    return t("chat.help")


def _process_chat_cmd(user_input: str, assistant: Any, session: ChatSession) -> bool | None:
    """Dispatch a special chat command.

    Args:
        user_input: Raw user text (already stripped).
        assistant: Active ``FinanceAssistant``.
        session: Chat session.

    Returns:
        ``True`` if a command was handled and the loop should continue, ``False`` if the user wants
        to exit, ``None`` if the input should go to the assistant.
    """
    command = parse_command(user_input)
    if command is None:
        return None
    if command == "exit":
        typer.echo("\n" + t("chat.goodbye"))
        return False
    _render_notices(run_command(command, assistant, session, _run_sync))
    return True


def _handle_input(user_input: str, assistant: Any, session: ChatSession) -> bool:
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


def _chat_loop(assistant: Any, session: ChatSession, prompt: Callable[[str], str] = typer.prompt) -> int:
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
            if not _handle_input(prompt("\n" + t("chat.prompt_label")), assistant, session):
                return 0
        except KeyboardInterrupt:
            typer.echo("\n" + t("chat.interrupted"))
            continue
        except (EOFError, typer.Abort):
            typer.echo("\n" + t("chat.goodbye"))
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
    if not has_pluggy_credentials():
        _echo_dim(t("cli.sync.skipped_no_credentials"))
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
    session = _ChatSession(_open_store)
    try:
        assistant = _build_assistant(session, think=think, enable_sql=enable_sql)
        if assistant is None:
            return 1
        if session.store.counts()["transactions"] == 0:
            typer.secho(t("chat.no_transactions"), fg=typer.colors.YELLOW)
        typer.echo("\n" + t("chat.ready") + " " + _help_text().splitlines()[0])
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
    session = _ChatSession(_open_store)
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
    typer.echo(t("cli.status.database", path=_state.db_path))
    typer.echo(t("cli.status.last_sync", when=last_sync or t("cli.status.never")))
    typer.echo(t("cli.status.data", counts=", ".join(f"{k}={v}" for k, v in counts.items())))
    if not items:
        typer.echo(t("cli.no_connections"))
    for item in items:
        line = f"- {item['id']}  {item.get('connector_name') or '?'}  status={item.get('status') or '?'}"
        expires = item.get("consent_expires_at")
        if expires:
            line += "  " + t("cli.status.consent_until", date=format_date(expires))
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
    client = _pluggy_client() if verify and has_pluggy_credentials() else None
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
                note = "" if added else " " + t("cli.items.already_registered")
                typer.echo(f"{'✓' if added else '='} {item_id}{note}")
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
        typer.echo(t("cli.no_connections"))
    for item in items:
        status = item.get("status") or t("cli.items.not_synced")
        typer.echo(f"{item['id']}  {item.get('connector_name') or '?'}  {status}")
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
    if removed:
        typer.echo(t("cli.items.removed", item_id=item_id))
    else:
        typer.echo(t("cli.items.not_found", item_id=item_id))
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
            typer.echo(t("cli.connect.received", item_id=item_id))
            item = wait_for_item(client, item_id)
            store.update_item(item)
            typer.echo(f"  status: {item.get('status')} ({item.get('executionStatus')})")

        typer.echo(t("cli.connect.opening"))
        return run_connect_flow(
            client,
            _on_item,
            sandbox=sandbox,
            on_url=lambda url: _echo_dim(t("cli.connect.browser_hint", url=url)),
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

    if not has_pluggy_credentials():
        typer.secho(
            t("cli.connect.missing_credentials", id_var=ENV_PLUGGY_CLIENT_ID, secret_var=ENV_PLUGGY_CLIENT_SECRET),
            fg=typer.colors.RED,
        )
        return 1
    client = _pluggy_client()
    try:
        received = _connect_items(client, sandbox=sandbox)
    except PluggyError as exc:
        typer.secho(t("cli.connect.pluggy_failed", error=exc), fg=typer.colors.RED)
        return 1
    finally:
        client.close()
    if not received:
        typer.echo(t("cli.connect.none_created"))
        return 1
    return _run_sync() if sync else 0


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------


@app.command(help=t("cli.chat.help"))
def chat(
    sync: Annotated[bool, typer.Option("--sync/--no-sync", help=t("cli.chat.sync"))] = True,
    think: Annotated[bool, typer.Option("--think", help=t("cli.chat.think"))] = False,
    sql: Annotated[bool | None, typer.Option("--sql/--no-sql", help=t("cli.chat.sql"))] = None,
) -> None:
    """Run the interactive chat (default command)."""
    raise typer.Exit(code=_handle_chat(sync=sync, think=think, enable_sql=sql))


@app.command(help=t("cli.ask.help"))
def ask(
    question: Annotated[str, typer.Argument(help=t("cli.ask.question"))],
    sync: Annotated[bool, typer.Option("--sync/--no-sync", help=t("cli.ask.sync"))] = False,
    sql: Annotated[bool | None, typer.Option("--sql/--no-sql", help=t("cli.ask.sql"))] = None,
) -> None:
    """Answer one question and exit."""
    raise typer.Exit(code=_handle_ask(question, sync=sync, enable_sql=sql))


@app.command(help=t("cli.sync.help"))
def sync(force: Annotated[bool, typer.Option("--force", help=t("cli.sync.force"))] = False) -> None:
    """Sync every registered item from Pluggy."""
    raise typer.Exit(code=_run_sync(force=force))


@app.command(help=t("cli.status.help"))
def status() -> None:
    """Show registered connections, consents and local data counts."""
    raise typer.Exit(code=_handle_status())


@app.command(help=t("cli.connect.help"))
def connect(
    sandbox: Annotated[bool, typer.Option("--sandbox", help=t("cli.connect.sandbox"))] = False,
    sync: Annotated[bool, typer.Option("--sync/--no-sync", help=t("cli.connect.sync"))] = True,
) -> None:
    """Connect accounts through Pluggy Connect in the browser."""
    raise typer.Exit(code=_handle_connect(sandbox=sandbox, sync=sync))


@items_app.command("add", help=t("cli.items.add.help"))
def items_add(
    item_ids: Annotated[list[str], typer.Argument(help=t("cli.items.add.ids"))],
    verify: Annotated[bool, typer.Option("--verify/--no-verify", help=t("cli.items.add.verify"))] = True,
) -> None:
    """Register Pluggy item ids for syncing."""
    raise typer.Exit(code=_handle_items_add(item_ids, verify=verify))


@items_app.command("list", help=t("cli.items.list.help"))
def items_list() -> None:
    """List the registered connections."""
    raise typer.Exit(code=_handle_items_list())


@items_app.command("remove", help=t("cli.items.remove.help"))
def items_remove(item_id: Annotated[str, typer.Argument(help=t("cli.items.remove.id"))]) -> None:
    """Remove a connection and its local data."""
    raise typer.Exit(code=_handle_items_remove(item_id))


@app.command("ver", help=t("cli.ver.help"))
def version_cmd() -> None:
    """Print the version."""
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
