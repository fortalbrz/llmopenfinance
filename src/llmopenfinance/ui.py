"""Rich console helpers: the one place that decides how the terminal front end looks.

Every helper chooses between two renderings **at call time**: styled output (colours, panels, a live answer) on an
interactive terminal when ``NO_COLOR`` is unset, and plain text, with no style codes at all, in every other case
(pipes, ``CliRunner``, ``NO_COLOR``). The plain output is the wording and the bytes the CLI always printed, so
scripts and tests that capture it keep working.

Text handed to a helper (model answers, merchant names, item ids, ...) is data. It is never parsed as Rich markup
or emoji: styled output wraps it in :class:`rich.text.Text`, plain output writes it as raw segments.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment, Segments
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from llmopenfinance.i18n import t

if TYPE_CHECKING:
    from llmopenfinance.agent import AssistantEvent
    from llmopenfinance.session import Notice
    from llmopenfinance.sync import ItemResult

__all__ = [
    "THEME",
    "banner",
    "console",
    "interactive",
    "item_result",
    "key_value_line",
    "render_answer",
    "show_notice",
]

_APP_NAME = "finassist"
_MAX_TOOL_ARGS = 120
_LIVE_REFRESHES_PER_SECOND = 10

THEME = Theme(
    {
        "ok": "green",
        "warn": "yellow",
        "err": "red",
        "dim": "dim",
        "accent": "bold cyan",
        "user": "bold blue",
        "assistant": "bold magenta",
    }
)

# Notice level -> theme style. ``info`` (and any unknown level) has no style: plain text on every console.
_NOTICE_STYLES = {"ok": "ok", "warn": "warn", "error": "err", "dim": "dim"}

console: Console = Console(theme=THEME, highlight=False)
"""The shared console. A plain module attribute: tests replace it to capture output."""

_Kind = Literal["text", "tool", "error"]


def interactive() -> bool:
    """Tell whether the console is attached to an interactive terminal.

    Returns:
        bool: ``True`` when the console writes to a terminal.
    """
    return console.is_terminal


def _styled() -> bool:
    """Tell whether to emit style codes right now.

    Decided per call from the console's current output and the environment, never from state captured when the
    console was built: output redirected after start-up stays free of escape codes, and ``NO_COLOR`` (when set to
    a non-empty value, as on no-color.org) switches styling off for the whole call.

    Returns:
        bool: ``True`` on a terminal with ``NO_COLOR`` unset.
    """
    return interactive() and not os.environ.get("NO_COLOR")


def _say(text: str, *, style: str = "", end: str = "\n") -> None:
    """Print ``text`` verbatim, styled only when :func:`_styled` allows it.

    Plain output goes out as raw segments: Rich's text pipeline would strip carriage returns, expand tabs and
    trim trailing blanks, and the plain stream must stay byte-identical to the old ``echo`` output. It is written
    unwrapped (``soft_wrap=True``) so long answers are never cut at the console width.

    Args:
        text: Text to print; never parsed as markup or emoji.
        style: Theme style name for the styled rendering.
        end: What to print after ``text``.
    """
    if _styled():
        console.print(Text(text, style=style), end=end)
    else:
        console.print(Segments([Segment(text + end)]), soft_wrap=True)


def key_value_line(pairs: Mapping[str, object]) -> str:
    """Format a mapping as ``k=v, k=v`` (the data counts of ``status`` and the sync lines).

    Args:
        pairs: Names and values, in display order.

    Returns:
        str: The joined line; empty for an empty mapping.
    """
    return ", ".join(f"{key}={value}" for key, value in pairs.items())


def show_notice(notice: Notice) -> None:
    """Print a notice; its level only picks a theme style, the text is already complete.

    Args:
        notice: The message; ``ok``, ``warn``, ``error`` and ``dim`` are styled, ``info`` is plain.
    """
    _say(notice.text, style=_NOTICE_STYLES.get(notice.level, ""))


def item_result(result: ItemResult) -> None:
    """Print the outcome of syncing one item: ``✓`` done, ``=`` unchanged, ``✗`` failed, ``!`` for warnings.

    Args:
        result: The per-item sync result.
    """
    if not result.ok:
        _say(f"  ✗ {result.item_id}: {result.error}", style="err")
    elif result.skipped:
        unchanged = t("cli.sync.unchanged")
        _say(f"  = {result.item_id}: {unchanged}", style="dim")
    else:
        _say(f"  ✓ {result.item_id}: {key_value_line(result.counts)}", style="ok")
    for warning in result.warnings:
        _say(f"    ! {warning}", style="warn")


def banner(provider: str, model: str, language: str, version: str) -> None:
    """Print the start-up banner: a titled panel on a terminal, two plain lines otherwise.

    Args:
        provider: LLM provider name.
        model: Model id.
        language: Process language code.
        version: Application version.
    """
    title = f"{_APP_NAME} {version}"
    details = (
        (t("ui.banner.provider"), provider),
        (t("ui.banner.model"), model),
        (t("ui.banner.language"), language),
    )
    if not _styled():
        _say(title)
        _say(", ".join(f"{label}: {value}" for label, value in details))
        return
    grid = Table.grid(padding=(0, 2))
    for label, value in details:
        grid.add_row(Text(label, style="dim"), Text(value))
    console.print(Panel.fit(grid, title=Text(title, style="accent"), title_align="left", border_style="accent"))


def _parse(events: Iterable[AssistantEvent]) -> Iterator[tuple[_Kind, str]]:
    """Turn assistant events into ``(kind, display text)`` pairs shared by both renderers.

    ``agent`` and ``chat`` are imported here, on first use, so commands that never answer a question do not pay
    for their imports.

    Args:
        events: Events from ``FinanceAssistant.ask``.

    Yields:
        tuple[_Kind, str]: ``("text", delta)``, ``("tool", status line)`` or ``("error", error line)``.
    """
    from llmopenfinance.agent import AssistantError, ToolStatus
    from llmopenfinance.chat import TextDelta

    for event in events:
        if isinstance(event, TextDelta):
            yield "text", event.text
        elif isinstance(event, ToolStatus):
            args = ", ".join(f"{key}={value!r}" for key, value in event.call.arguments.items())
            yield "tool", t("chat.tool_status", name=event.call.name, args=args[:_MAX_TOOL_ARGS])
        elif isinstance(event, AssistantError):
            yield "error", t("chat.error", message=event.message)


@dataclass
class _LiveView:
    """State behind the live answer display; Rich renders it again on every refresh, from a timer thread.

    The main thread only ever replaces a field with a new immutable value, and a refresh reads each field once,
    so a refresh never sees a half-updated state.

    Attributes:
        answer: Answer text received so far (Markdown).
        tools: Status lines of the tool calls running now.
        errors: Error lines to show below the answer.
        waiting: ``True`` until the first event arrives (shows the assistant label with a spinner).
        spinner: The one spinner, kept across refreshes so that it animates.
    """

    answer: str = ""
    tools: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    waiting: bool = True
    spinner: Spinner = field(default_factory=lambda: Spinner("dots", style="accent"))

    def __rich__(self) -> RenderableType:  # ruff: ignore[bad-dunder-method-name] — Rich's renderable protocol
        """Build the display: the answer panel, then the running tools (spinner on the last), then the errors.

        Returns:
            RenderableType: What Rich draws at this refresh.
        """
        answer, tools, errors, waiting = self.answer, self.tools, self.errors, self.waiting
        rows: list[RenderableType] = []
        if answer:
            label = Text(t("chat.assistant_label"))
            rows.append(Panel(Markdown(answer), title=label, title_align="left", border_style="assistant"))
        rows.extend(Text(line, style="dim") for line in tools[:-1])
        status = tools[-1] if tools else (t("chat.assistant_label") if waiting else "")
        if status:
            self.spinner.update(text=Text(status, style="dim"))
            rows.append(self.spinner)
        rows.extend(Text(line, style="err") for line in errors)
        return Group(*rows)


def _render_plain(events: Iterable[AssistantEvent]) -> bool:
    """Stream the answer as plain text, exactly as the CLI always printed it.

    Args:
        events: Events from ``FinanceAssistant.ask``.

    Returns:
        bool: ``False`` when an error was shown.
    """
    ok = True
    _say("\n" + t("chat.assistant_label") + ": ", end="")
    for kind, text in _parse(events):
        if kind == "text":
            _say(text, end="")
        elif kind == "tool":
            _say("\n  " + text)
        else:
            ok = False
            _say("\n" + text)
    _say("\n")  # the text ends without a newline: this closes it and leaves one blank line
    return ok


def _render_live(events: Iterable[AssistantEvent]) -> bool:
    """Show the answer as Markdown in a titled panel that grows while it streams, with a status line per tool.

    The ``with`` block stops the display (cursor back, stdout restored) on any exit, ``KeyboardInterrupt``
    included; the final frame has no spinner.

    Args:
        events: Events from ``FinanceAssistant.ask``.

    Returns:
        bool: ``False`` when an error was shown.
    """
    ok = True
    view = _LiveView()
    with Live(view, console=console, refresh_per_second=_LIVE_REFRESHES_PER_SECOND) as live:
        try:
            for kind, text in _parse(events):
                view.waiting = False
                if kind == "tool":
                    view.tools = (*view.tools, text)
                    live.refresh()  # draw the spinner now: the next event only arrives once the tools have run
                    continue
                view.tools = ()
                if kind == "text":
                    view.answer += text
                else:
                    ok = False
                    view.errors = (*view.errors, text)
        finally:
            view.tools, view.waiting = (), False
    return ok


def render_answer(events: Iterable[AssistantEvent]) -> bool:
    """Show an assistant answer as it streams.

    On an interactive terminal the answer is a live Markdown panel with a spinner line per tool call. Anywhere
    else (pipes, ``NO_COLOR``) the text, the tool status lines and the errors are streamed as plain text.

    ``events`` is always consumed or closed, also when the display fails or the user presses Ctrl+C, so the
    ``finally`` of ``FinanceAssistant.ask`` (history rollback) runs.

    Args:
        events: Events from ``FinanceAssistant.ask``.

    Returns:
        bool: ``False`` when an assistant error was shown, ``True`` otherwise.
    """
    try:
        return _render_live(events) if _styled() else _render_plain(events)
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()
