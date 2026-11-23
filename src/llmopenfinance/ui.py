"""Rich console helpers: the one place that decides how the terminal front end looks.

Every helper chooses between two renderings **at call time**: styled output (colours, panels, a live answer) on an
interactive terminal when ``NO_COLOR`` is unset, and plain text, with no style codes at all, in every other case
(pipes, ``CliRunner``, ``NO_COLOR``). The plain output is the wording and the bytes the CLI always printed, so
scripts and tests that capture it keep working.

Text handed to a helper (model answers, merchant names, item ids, ...) is data. It is never parsed as Rich markup
or emoji: styled output wraps it in :class:`rich.text.Text`, plain output writes it as raw segments. It is also
stripped of terminal escape sequences (:func:`_clean`), so third-party text cannot drive the user's terminal or
corrupt piped output.

Tables (the data counts and the connections of ``status`` and ``items list``) exist only in the styled rendering.
Their plain counterpart is the one-line-per-item text the CLI always printed, so scripts keep a stable format.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import typer
from rich import box
from rich.cells import cell_len
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
    "ItemRow",
    "banner",
    "console",
    "data_counts",
    "interactive",
    "item_list",
    "item_result",
    "key_value_line",
    "prompt_label",
    "render_answer",
    "show_notice",
    "status_items",
]

_APP_NAME = "finassist"
_MAX_TOOL_ARGS = 120
# CSI sequences: the very pattern ``typer.echo`` / ``click.unstyle`` strip on a non-terminal, which keeps plain
# output byte-identical to the old echo. (click is vendored inside typer, so it is not importable by itself.)
_CSI = re.compile(r"\x1b\[[;?0-9]*[a-zA-Z]")
# OSC sequences (window title, hyperlinks, ...): ``ESC ]``, a payload, then BEL or ``ESC \``.
_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_LIVE_REFRESHES_PER_SECOND = 10
_MIN_ID_WIDTH = 12  # below this the id column would fold into slivers: show one line per item instead
_DETAIL_STYLES = ("", "", "dim")  # connector, status, consent

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

    This is only the terminal check. It is NOT "output is styled": styling also needs ``NO_COLOR`` unset, and
    that is decided per call by :func:`_styled`, which every helper uses.

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


def _clean(text: str) -> str:
    """Remove terminal escape sequences from data before it is printed or handed to a Rich renderable.

    Model answers, merchant names or item ids are third-party text. Left alone, an escape sequence in them would
    reach the user's terminal (clear the screen, retitle the window, ...) or corrupt piped output. CSI sequences
    go first, exactly as ``typer.echo`` strips them off a terminal, then OSC sequences, then any ESC left over
    (a lone or truncated sequence). Text without an ESC character is returned unchanged.

    Each call sees one piece of text, so an escape sequence split across two streamed deltas is not recognised as
    one sequence: only its ESC characters are removed and the rest of it shows as plain text.

    Args:
        text: Data to sanitise.

    Returns:
        str: ``text`` without escape sequences and without the ESC character.
    """
    if "\x1b" not in text:
        return text
    return _OSC.sub("", _CSI.sub("", text)).replace("\x1b", "")


def _say(text: str, *, style: str = "", end: str = "\n") -> None:
    """Print ``text`` verbatim, styled only when :func:`_styled` allows it.

    Plain output goes out as raw segments: Rich's text pipeline would strip carriage returns, expand tabs and
    trim trailing blanks, and the plain stream must stay byte-identical to the old ``echo`` output. It is written
    unwrapped (``soft_wrap=True``) so long answers are never cut at the console width.

    Args:
        text: Text to print; never parsed as markup or emoji, and cleaned of escape sequences.
        style: Theme style name for the styled rendering.
        end: What to print after ``text``.
    """
    text = _clean(text)
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
        grid.add_row(Text(label, style="dim"), Text(_clean(value)))
    panel_title = Text(_clean(title), style="accent")
    console.print(Panel.fit(grid, title=panel_title, title_align="left", border_style="accent"))


def prompt_label(label: str) -> str:
    """Style the label of an input prompt (bold cyan), or return it unchanged when styling is off.

    ``typer.prompt`` writes the label itself, through Typer's own echo, so the terminal and ``NO_COLOR`` decision
    has to be taken here: off a terminal the label is plain text and the prompt output is byte-identical to the
    old one.

    Args:
        label: The prompt text, such as ``"\\nVocê"``.

    Returns:
        str: ``label`` wrapped in style codes on a terminal with ``NO_COLOR`` unset, ``label`` otherwise.
    """
    return typer.style(label, fg=typer.colors.CYAN, bold=True) if _styled() else label


@dataclass(frozen=True)
class ItemRow:
    """One registered connection as ``status`` and ``items list`` show it.

    Attributes:
        item_id: Pluggy item id.
        connector: Connector name (``?`` when unknown).
        status: Item status, or the "not synced" text.
        consent: Consent line such as ``consentimento até 01/06/2027``; empty when there is no expiry date.
    """

    item_id: str
    connector: str
    status: str
    consent: str = ""


def data_counts(counts: Mapping[str, int]) -> None:
    """Print the local data counts: a titled table of ``key=value`` cells on a terminal, one line otherwise.

    The plain line is the ``Dados: k=v, k=v`` line of ``status``, unchanged.

    Args:
        counts: Row count per table, in display order.
    """
    if not _styled():
        _say(t("cli.status.data", counts=key_value_line(counts)))
        return
    table = Table(
        title=Text(t("ui.table.data"), style="accent"),
        title_justify="left",
        box=box.ROUNDED,
        show_header=False,
        border_style="dim",
    )
    table.add_column(no_wrap=True)
    for key, value in counts.items():
        table.add_row(Text(_clean(f"{key}={value}")))
    console.print(table, crop=False)


def status_items(rows: Sequence[ItemRow]) -> None:
    """Print the connections for ``status``: a table (id, connector, status, consent) on a terminal, else lines.

    The plain lines are ``- id  connector  status=X  consent`` exactly as ``status`` always printed them.

    Args:
        rows: One row per registered item; nothing is printed for an empty sequence.
    """
    _show_items(rows, consent=True)


def item_list(rows: Sequence[ItemRow]) -> None:
    """Print the connections for ``items list``: a table (id, connector, status) on a terminal, else lines.

    The plain lines are ``id  connector  status`` exactly as ``items list`` always printed them.

    Args:
        rows: One row per registered item; nothing is printed for an empty sequence.
    """
    _show_items(rows, consent=False)


def _show_items(rows: Sequence[ItemRow], *, consent: bool) -> None:
    """Print the rows as a table when styling is on and the table fits, as one plain line per row otherwise.

    Args:
        rows: The items to show.
        consent: ``True`` for the ``status`` layout (with consent), ``False`` for the ``items list`` layout.
    """
    if rows and _styled() and _items_fit(rows, consent=consent):
        console.print(_items_table(rows, consent=consent), crop=False)
        return
    for row in rows:
        _say(_item_line(row, consent=consent))


def _item_line(row: ItemRow, *, consent: bool) -> str:
    """Format one item as the plain line the CLI always printed.

    Args:
        row: The item.
        consent: ``True`` for the ``status`` layout, ``False`` for the ``items list`` layout.

    Returns:
        str: The line, without a newline.
    """
    if not consent:
        return f"{row.item_id}  {row.connector}  {row.status}"
    line = f"- {row.item_id}  {row.connector}  status={row.status}"
    return f"{line}  {row.consent}" if row.consent else line


def _detail_headers(*, consent: bool) -> tuple[str, ...]:
    """Name the columns that come after the id.

    Args:
        consent: Include the consent column.

    Returns:
        tuple[str, ...]: Localised headers: connector, status and, with ``consent``, consent.
    """
    headers = (t("ui.table.connector"), t("ui.table.status"))
    return (*headers, t("ui.table.consent")) if consent else headers


def _detail_cells(row: ItemRow, *, consent: bool) -> tuple[str, ...]:
    """Return the cells that come after the id, matching :func:`_detail_headers`.

    Args:
        row: The item.
        consent: Include the consent cell.

    Returns:
        tuple[str, ...]: Connector, status and, with ``consent``, consent.
    """
    return (row.connector, row.status, row.consent) if consent else (row.connector, row.status)


def _items_fit(rows: Sequence[ItemRow], *, consent: bool) -> bool:
    """Tell whether every cell but the id fits whole, leaving the id column room to fold.

    Every column except the id is ``no_wrap``, so Rich would cut its text with an ellipsis once the table is wider
    than the terminal. Measuring first lets the caller fall back to plain lines instead of cropping anything.

    Args:
        rows: The items to show.
        consent: Whether the consent column is shown.

    Returns:
        bool: ``True`` when the id column keeps at least ``_MIN_ID_WIDTH`` columns.
    """
    matrix = [_detail_headers(consent=consent), *(_detail_cells(row, consent=consent) for row in rows)]
    widths = [max(cell_len(_clean(cell)) for cell in column) for column in zip(*matrix, strict=True)]
    chrome = 3 * (len(widths) + 1) + 1  # per column a border and two padding spaces, plus the closing border
    return console.width - sum(widths) - chrome >= _MIN_ID_WIDTH


def _items_table(rows: Sequence[ItemRow], *, consent: bool) -> Table:
    """Build the connections table: the id column folds, every other cell stays on one line.

    Cells are :class:`~rich.text.Text` objects holding cleaned text, never markup.

    Args:
        rows: The items to show.
        consent: Add the consent column.

    Returns:
        Table: The table, ready to print.
    """
    table = Table(box=box.ROUNDED, header_style="accent", border_style="dim")
    table.add_column(t("ui.table.id"), overflow="fold")
    for header, style in zip(_detail_headers(consent=consent), _DETAIL_STYLES, strict=False):
        table.add_column(header, style=style, no_wrap=True)
    for row in rows:
        table.add_row(Text(_clean(row.item_id)), *(Text(_clean(cell)) for cell in _detail_cells(row, consent=consent)))
    return table


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
            # hyperlinks=False: the model controls link targets, which must not become OSC 8 terminal hyperlinks
            label = Text(t("chat.assistant_label"))
            rows.append(
                Panel(Markdown(answer, hyperlinks=False), title=label, title_align="left", border_style="assistant")
            )
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
            for kind, raw in _parse(events):
                text = _clean(raw)  # the Markdown and Text renderables below would pass escape sequences through
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
