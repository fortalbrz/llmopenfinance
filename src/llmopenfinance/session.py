"""Shared chat/sync core used by every front end (terminal CLI, web chat).

Nothing here writes to the terminal: the LLM connection check, the chat commands and the Pluggy sync report what
happened as :class:`Notice` values (and per-item :class:`~llmopenfinance.sync.ItemResult` callbacks), and each front
end renders them in its own way. The factories that create the LLM backend, the store and the Pluggy client are
injected through :class:`Factories`, so callers (and tests) choose the implementations.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llmopenfinance.constants import ENV_PLUGGY_CLIENT_ID, ENV_PLUGGY_CLIENT_SECRET
from llmopenfinance.i18n import t

if TYPE_CHECKING:
    from llmopenfinance.sync import ItemResult

__all__ = [
    "AssistantBuild",
    "BuildOptions",
    "ChatSession",
    "Factories",
    "Notice",
    "build_assistant",
    "has_pluggy_credentials",
    "parse_command",
    "run_command",
    "run_sync",
]


@dataclass(frozen=True)
class Notice:
    """A message for the user.

    Attributes:
        level: ``"ok"``, ``"info"``, ``"warn"``, ``"error"`` or ``"dim"``; decides only how a front end styles it.
        text: The complete text to display.
    """

    level: str
    text: str


@dataclass(frozen=True)
class Factories:
    """Injected constructors for the external resources.

    Attributes:
        llm: ``(provider, model, *, think)`` -> ``AbstractChat``.
        open_store: ``(*, read_only=False)`` -> ``FinanceStore``.
        pluggy_client: ``()`` -> ``PluggyClient``.
    """

    llm: Callable[..., Any]
    open_store: Callable[..., Any]
    pluggy_client: Callable[[], Any]


@dataclass(frozen=True)
class BuildOptions:
    """Assistant options chosen by the user.

    Attributes:
        think: Enable Ollama thinking.
        enable_sql: Expose ``run_sql`` (``None`` = on for Claude, off for Ollama).
    """

    think: bool
    enable_sql: bool | None


@dataclass(frozen=True)
class AssistantBuild:
    """Outcome of :func:`build_assistant`.

    Attributes:
        assistant: ``FinanceAssistant``, or ``None`` when the LLM could not be created or reached.
        notices: Messages for the user, in display order.
    """

    assistant: Any | None
    notices: tuple[Notice, ...]


def has_pluggy_credentials() -> bool:
    """Tell whether both Pluggy credentials are set in the environment.

    Returns:
        bool: ``True`` when the client id and secret are both non-empty.
    """
    return bool(os.environ.get(ENV_PLUGGY_CLIENT_ID) and os.environ.get(ENV_PLUGGY_CLIENT_SECRET))


class ChatSession:
    """Holds the read-only store and lets a sync close/reopen it (DuckDB allows one read-write handle)."""

    def __init__(self, open_store: Callable[..., Any]) -> None:
        """Open the read-only store.

        Args:
            open_store: ``(*, read_only=False)`` store factory, kept to reopen the store after a sync.
        """
        self._open_store = open_store
        self.store = open_store(read_only=True)

    def connection(self) -> Any:
        """Return the current DuckDB connection (it changes after :meth:`resync`).

        Returns:
            Any: The read-only connection.
        """
        return self.store.con

    def snapshot(self) -> str:
        """Build the financial snapshot from the current store.

        Returns:
            str: Markdown snapshot.
        """
        from llmopenfinance.analytics import build_snapshot

        return build_snapshot(self.store.con)

    def resync(self, run_sync: Callable[[], int]) -> int:
        """Close the read-only store, run the sync and reopen the store read-only, even if the sync fails.

        Args:
            run_sync: Sync entry point returning an exit code.

        Returns:
            int: The sync exit code.
        """
        self.store.close()
        try:
            return run_sync()
        finally:
            self.store = self._open_store(read_only=True)

    def close(self) -> None:
        """Close the store."""
        self.store.close()


def build_assistant(
    factories: Factories, session: ChatSession, *, provider: str, model: str, options: BuildOptions
) -> AssistantBuild:
    """Create the LLM backend, tools and assistant, verifying the LLM connection.

    Args:
        factories: Resource factories (only ``llm`` is used).
        session: Chat session whose store backs the tools and the snapshot.
        provider: ``claude`` or ``ollama``.
        model: Model id (empty = provider default).
        options: Thinking and ``run_sql`` choices.

    Returns:
        AssistantBuild: The assistant (``None`` on failure) and the notices to show, in order.
    """
    from llmopenfinance.agent import FinanceAssistant
    from llmopenfinance.tools import ToolRegistry

    try:
        llm = factories.llm(provider, model, think=options.think)
    except ValueError as exc:
        return AssistantBuild(assistant=None, notices=(Notice("error", str(exc)),))
    notices = [Notice("info", t("cli.llm.verifying", provider=llm.provider, model=llm.model))]
    if not llm._verify_connection():
        notices.append(Notice("error", t("cli.llm.unreachable", provider=llm.provider, model=llm.model)))
        if llm.provider == "ollama":
            notices.append(Notice("info", t("cli.llm.ollama_hint", model=llm.model)))
        else:
            notices.append(Notice("info", t("cli.llm.claude_hint")))
        return AssistantBuild(assistant=None, notices=tuple(notices))
    if llm.provider == "claude":
        notices.append(Notice("dim", t("cli.llm.privacy")))
    sql = options.enable_sql if options.enable_sql is not None else llm.provider == "claude"
    registry = ToolRegistry(session.connection, enable_sql=sql)
    return AssistantBuild(assistant=FinanceAssistant(llm, registry, session.snapshot), notices=tuple(notices))


# The aliases of both languages are accepted in either; only the help text names the current language's forms.
_EXIT_ALIASES = frozenset({"sair", "quit", "exit", "bye", "q", "/sair", "/exit", "/quit"})
_COMMAND_ALIASES: dict[str, str] = {
    "/reset": "reset",
    "reset": "reset",
    "/ajuda": "help",
    "/help": "help",
    "ajuda": "help",
    "help": "help",
    "/resumo": "summary",
    "/summary": "summary",
    "/sync": "sync",
    "/historico": "history",
    "/history": "history",
    "historico": "history",
    "history": "history",
}


def parse_command(text: str) -> str | None:
    """Recognise a chat command typed by the user (case-insensitive, aliases of both languages).

    Args:
        text: Raw user text.

    Returns:
        str | None: ``"help"``, ``"summary"``, ``"reset"``, ``"history"``, ``"sync"`` or ``"exit"``; ``None`` when
        the text is a question for the assistant.
    """
    command = text.strip().lower()
    if command in _EXIT_ALIASES:
        return "exit"
    return _COMMAND_ALIASES.get(command)


def _history_notices(assistant: Any) -> tuple[Notice, ...]:
    history = assistant.chat.get_history()
    if not history:
        return (Notice("info", t("chat.history_empty")),)
    notices: list[Notice] = []
    for i, msg in enumerate(history, 1):
        content = msg.get("content")
        text = content if isinstance(content, str) else str(content)
        snippet = text[:100] + ("…" if len(text) > 100 else "")
        notices.append(Notice("info", f"{i}. [{msg['role'].upper()}] {snippet}"))
    return tuple(notices)


def run_command(name: str, assistant: Any, session: ChatSession, run_sync: Callable[[], int]) -> tuple[Notice, ...]:
    """Run a chat command (other than ``exit``) and describe the outcome.

    Args:
        name: ``"help"``, ``"summary"``, ``"reset"``, ``"history"`` or ``"sync"`` (see :func:`parse_command`).
        assistant: Active ``FinanceAssistant``.
        session: Chat session.
        run_sync: Sync entry point used by ``sync`` (called while the read-only store is closed).

    Returns:
        tuple[Notice, ...]: Messages for the user, in display order.

    Raises:
        ValueError: For ``exit`` or an unknown command name.
    """
    match name:
        case "help":
            return (Notice("info", t("chat.help")),)
        case "summary":
            return (Notice("info", session.snapshot()),)
        case "reset":
            assistant.refresh()
            return (Notice("info", t("chat.reset")),)
        case "history":
            return _history_notices(assistant)
        case "sync":
            code = session.resync(run_sync)
            assistant.refresh()
            if code == 0:
                return (Notice("info", t("chat.sync_ok")),)
            return (Notice("warn", t("chat.sync_failed")),)
        case _:
            msg = f"not a runnable chat command: {name!r}"
            raise ValueError(msg)


def run_sync(
    factories: Factories,
    *,
    force: bool,
    notify: Callable[[Notice], None],
    on_item: Callable[[ItemResult], None],
) -> int:
    """Synchronize every registered item into the local store.

    Args:
        factories: Resource factories (``open_store`` and ``pluggy_client`` are used).
        force: Re-download unchanged items.
        notify: Receives each message for the user, in order.
        on_item: Receives each item's result as soon as it is synced.

    Returns:
        int: Exit code (0 when at least one item synced or nothing to do).
    """
    from llmopenfinance.pluggy import PluggyError
    from llmopenfinance.sync import sync_all

    if not has_pluggy_credentials():
        notify(
            Notice(
                "error",
                t("cli.sync.no_credentials", id_var=ENV_PLUGGY_CLIENT_ID, secret_var=ENV_PLUGGY_CLIENT_SECRET),
            )
        )
        return 1
    import duckdb

    try:
        store = factories.open_store()
    except duckdb.IOException as exc:
        notify(Notice("warn", t("cli.sync.db_locked", error=exc)))
        return 1
    with store:
        if not store.list_items():
            notify(Notice("info", t("cli.sync.no_items")))
            return 0
        try:
            client = factories.pluggy_client()
            client.authenticate()
        except (PluggyError, ValueError) as exc:
            notify(Notice("error", t("cli.sync.auth_failed", error=exc)))
            return 1
        notify(Notice("info", t("cli.sync.start")))
        try:
            report = sync_all(client, store, force=force, on_item=on_item)
        finally:
            client.close()
    notify(Notice("info", t("cli.sync.done", ok=report.n_success, failed=report.n_failed)))
    return 0 if report.n_success or not report.n_total else 1
