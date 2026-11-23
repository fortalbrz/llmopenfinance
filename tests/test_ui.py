"""Tests for the Rich console helpers: plain text off a terminal, styles on one, verbatim dynamic text."""

import io
import re
import sys
import threading
from collections.abc import Callable, Iterator

import pytest
import typer
from rich.console import Console
from rich.live import Live

from llmopenfinance import ui
from llmopenfinance.agent import AssistantError, AssistantEvent, ToolStatus
from llmopenfinance.chat import TextDelta
from llmopenfinance.i18n import t
from llmopenfinance.session import Notice
from llmopenfinance.sync import ItemResult
from llmopenfinance.tools import ToolCall

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LEVELS = ("ok", "info", "warn", "error", "dim")
TRICKY = "[erro] [bold]x[/bold] :tada: [PROMO]"


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def make_console(buffer: io.StringIO, *, terminal: bool, height: int | None = None) -> Console:
    """Build the console the way the tests of the brief do: 80 columns, the app theme, a fixed colour system."""
    return Console(
        file=buffer,
        theme=ui.THEME,
        width=80,
        height=height,
        force_terminal=terminal,
        color_system="standard",
        legacy_windows=False,
    )


@pytest.fixture(autouse=True)
def _colour_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with ``NO_COLOR`` and ``TERM`` unset, whatever the caller's shell exports."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)  # TERM=dumb would stop Rich from drawing intermediate live frames


@pytest.fixture
def plain(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Replace the module console with one that is not a terminal (but still has a colour system)."""
    buffer = io.StringIO()
    monkeypatch.setattr(ui, "console", make_console(buffer, terminal=False))
    return buffer


@pytest.fixture
def tty(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Replace the module console with a forced terminal writing standard ANSI colours."""
    buffer = io.StringIO()
    monkeypatch.setattr(ui, "console", make_console(buffer, terminal=True))
    return buffer


def tool_event(name: str = "spending_by_category", **arguments: object) -> ToolStatus:
    return ToolStatus(ToolCall(id="call-1", name=name, arguments=arguments))


def tracked(events: list[AssistantEvent], closed: list[str]) -> Iterator[AssistantEvent]:
    """Yield ``events`` and record in ``closed`` when the generator's ``finally`` runs (like ``ask``)."""
    try:
        yield from events
    finally:
        closed.append("closed")


# ---------------------------------------------------------------------------
# Theme, console, small helpers
# ---------------------------------------------------------------------------


def test_theme_has_the_styles_callers_use():
    assert {"ok", "warn", "err", "dim", "accent", "user", "assistant"} <= set(ui.THEME.styles)


def test_module_console_is_a_plain_replaceable_attribute():
    assert isinstance(ui.console, Console)
    assert ui.console.get_style("ok") == ui.THEME.styles["ok"]


def test_interactive_follows_the_console(plain, monkeypatch):
    assert ui.interactive() is False
    monkeypatch.setattr(ui, "console", make_console(io.StringIO(), terminal=True))
    assert ui.interactive() is True


def test_key_value_line_matches_the_status_counts_line():
    assert ui.key_value_line({"accounts": 2, "transactions": 10}) == "accounts=2, transactions=10"
    assert not ui.key_value_line({})


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", LEVELS)
def test_notice_is_printed_unchanged_off_a_terminal(plain, level):
    ui.show_notice(Notice(level, "Conexão verificada."))
    assert plain.getvalue() == "Conexão verificada.\n"


@pytest.mark.parametrize(
    ("level", "sgr"),
    [("ok", "\x1b[32m"), ("warn", "\x1b[33m"), ("error", "\x1b[31m"), ("dim", "\x1b[2m")],
)
def test_notice_levels_and_styles(tty, level, sgr):
    ui.show_notice(Notice(level, "Conexão verificada."))
    out = tty.getvalue()
    assert out.startswith(sgr)
    assert strip_ansi(out) == "Conexão verificada.\n"


def test_notice_levels_get_distinct_styles_and_info_has_none(tty):
    outputs = {}
    for level in LEVELS:
        tty.seek(0)
        tty.truncate()
        ui.show_notice(Notice(level, "x"))
        outputs[level] = tty.getvalue()
    assert len(set(outputs.values())) == len(LEVELS)
    assert outputs["info"] == "x\n"


def test_notice_adds_no_symbol_prefix(tty):
    ui.show_notice(Notice("error", "falhou"))
    assert strip_ansi(tty.getvalue()) == "falhou\n"


def test_unknown_notice_level_prints_plain_text(tty):
    ui.show_notice(Notice("mystery", "texto"))
    assert tty.getvalue() == "texto\n"


# ---------------------------------------------------------------------------
# Per-item sync results
# ---------------------------------------------------------------------------


def test_item_result_markers_match_cli(plain):
    ui.item_result(ItemResult("i1", ok=True, counts={"accounts": 2, "transactions": 10}))
    ui.item_result(ItemResult("i2", ok=True, skipped=True))
    ui.item_result(ItemResult("i3", ok=False, error="boom"))
    ui.item_result(ItemResult("i4", ok=True, counts={"accounts": 1}, warnings=("w1", "w2")))
    assert plain.getvalue() == (
        "  ✓ i1: accounts=2, transactions=10\n"
        "  = i2: sem alterações desde a última sincronização\n"
        "  ✗ i3: boom\n"
        "  ✓ i4: accounts=1\n"
        "    ! w1\n"
        "    ! w2\n"
    )


def test_item_result_follows_the_language(plain, use_language):
    use_language("en")
    ui.item_result(ItemResult("i2", ok=True, skipped=True))
    assert plain.getvalue() == f"  = i2: {t('cli.sync.unchanged')}\n"
    assert "sem alterações" not in plain.getvalue()


def test_item_result_styles_on_a_terminal(tty):
    for result in (
        ItemResult("a", ok=True, counts={"n": 1}, warnings=("careful",)),
        ItemResult("b", ok=True, skipped=True),
        ItemResult("c", ok=False, error="boom"),
    ):
        ui.item_result(result)
    lines = tty.getvalue().splitlines()
    assert lines[0].startswith("\x1b[32m  ✓ a: n=1")
    assert lines[1].startswith("\x1b[33m    ! careful")
    assert lines[2].startswith("\x1b[2m  = b:")
    assert lines[3].startswith("\x1b[31m  ✗ c: boom")


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def test_banner_is_two_plain_lines_off_a_terminal(plain):
    ui.banner("ollama", "llama3", "pt-BR", "1.2.3")
    assert plain.getvalue() == "finassist 1.2.3\nProvedor: ollama, Modelo: llama3, Idioma: pt-BR\n"


def test_banner_labels_follow_the_language(plain, use_language):
    use_language("en")
    ui.banner("claude", "sonnet", "en", "1.2.3")
    assert plain.getvalue() == "finassist 1.2.3\nProvider: claude, Model: sonnet, Language: en\n"


def test_banner_is_a_titled_panel_on_a_terminal(tty):
    ui.banner("ollama", "llama3", "pt-BR", "1.2.3")
    out = strip_ansi(tty.getvalue())
    assert "╭" in out
    assert "finassist 1.2.3" in out.splitlines()[0]
    for text in ("Provedor", "ollama", "Modelo", "llama3", "Idioma", "pt-BR"):
        assert text in out


# ---------------------------------------------------------------------------
# Dynamic text is never interpreted as markup or emoji
# ---------------------------------------------------------------------------


def _notice() -> None:
    ui.show_notice(Notice("warn", TRICKY))


def _failed_item() -> None:
    ui.item_result(ItemResult(TRICKY, ok=False, error=TRICKY))


def _synced_item() -> None:
    ui.item_result(ItemResult(TRICKY, ok=True, counts={TRICKY: 1}, warnings=(TRICKY,)))


def _banner() -> None:
    ui.banner(TRICKY, TRICKY, TRICKY, TRICKY)


def _answer() -> None:
    ui.render_answer([TextDelta(TRICKY), tool_event("t", q=TRICKY), AssistantError(TRICKY)])


@pytest.mark.parametrize("terminal", [False, True], ids=["plain", "terminal"])
@pytest.mark.parametrize(
    ("show", "fragments"),
    [
        (_notice, [TRICKY]),
        (_failed_item, [f"✗ {TRICKY}: {TRICKY}"]),
        (_synced_item, [f"✓ {TRICKY}: {TRICKY}=1", f"! {TRICKY}"]),
        (_banner, [f"finassist {TRICKY}"]),
        (_answer, [TRICKY, f"t(q={TRICKY!r})", f"[erro] {TRICKY}"]),
    ],
    ids=["notice", "failed_item", "synced_item", "banner", "answer"],
)
def test_dynamic_text_is_verbatim(monkeypatch, terminal, show: Callable[[], None], fragments: list[str]):
    buffer = io.StringIO()
    monkeypatch.setattr(ui, "console", make_console(buffer, terminal=terminal))
    show()  # must not raise MarkupError, and must not eat [bold] / :tada: / [PROMO]
    out = strip_ansi(buffer.getvalue())
    for fragment in fragments:
        assert fragment in out


@pytest.mark.parametrize("terminal", [False, True], ids=["plain", "terminal"])
def test_banner_shows_every_value_verbatim(monkeypatch, terminal):
    buffer = io.StringIO()
    monkeypatch.setattr(ui, "console", make_console(buffer, terminal=terminal))
    ui.banner(TRICKY, TRICKY, TRICKY, TRICKY)
    assert strip_ansi(buffer.getvalue()).count(TRICKY) == 4  # the title and the provider, model and language rows


def test_dynamic_text_keeps_square_brackets_off_a_terminal_exactly(plain):
    ui.show_notice(Notice("info", TRICKY))
    assert plain.getvalue() == TRICKY + "\n"


# ---------------------------------------------------------------------------
# No ANSI unless the console is a terminal and NO_COLOR is unset
# ---------------------------------------------------------------------------


def test_no_ansi_when_not_a_terminal(plain):
    # The console still has a colour system (it was "created on a terminal"): only the call-time check keeps
    # the redirected output clean.
    for level in LEVELS:
        ui.show_notice(Notice(level, "texto"))
    ui.item_result(ItemResult("i", ok=False, error="boom", warnings=("w",)))
    ui.banner("ollama", "llama3", "pt-BR", "1.2.3")
    ui.render_answer([TextDelta("Olá"), tool_event(month="2026-11"), AssistantError("boom")])
    assert "\x1b" not in plain.getvalue()
    assert plain.getvalue()


def test_no_color_env_disables_styles(tty, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")  # set after the console was built: detection is per call
    ui.show_notice(Notice("error", "falhou"))
    ui.item_result(ItemResult("i", ok=True, counts={"n": 1}, warnings=("w",)))
    ui.banner("ollama", "llama3", "pt-BR", "1.2.3")
    assert ui.render_answer([TextDelta("Olá"), tool_event(), AssistantError("boom")]) is False
    out = tty.getvalue()
    assert "\x1b" not in out
    assert out.startswith("falhou\n  ✓ i: n=1\n    ! w\nfinassist 1.2.3\n")
    assert "╭" not in out
    assert out.endswith("[erro] boom\n\n\n")


def test_empty_no_color_does_not_disable_styles(tty, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "")
    ui.show_notice(Notice("error", "falhou"))
    assert tty.getvalue().startswith("\x1b[31m")


# ---------------------------------------------------------------------------
# render_answer: plain streaming
# ---------------------------------------------------------------------------


def test_render_answer_plain_streams_text_tools_and_errors(plain):
    events = [
        TextDelta("Olá "),
        tool_event(month="2026-11", limit=5),
        TextDelta("resultado"),
        AssistantError("boom"),
    ]
    ok = ui.render_answer(events)
    assert ok is False
    assert plain.getvalue() == (
        "\nAssistente: Olá \n  ↳ consultando spending_by_category(month='2026-11', limit=5)\n"
        "resultado\n[erro] boom\n\n\n"
    )


def test_render_answer_returns_true_without_errors(plain):
    assert ui.render_answer([TextDelta("Tudo certo.")]) is True
    assert plain.getvalue() == "\nAssistente: Tudo certo.\n\n"


def test_render_answer_with_no_events_prints_the_label_and_succeeds(plain):
    assert ui.render_answer([]) is True
    assert plain.getvalue() == "\nAssistente: \n\n"


def test_render_answer_plain_follows_the_language(plain, use_language):
    use_language("en")
    ui.render_answer([TextDelta("Hi"), tool_event("net_worth"), AssistantError("boom")])
    assert plain.getvalue() == "\nAssistant: Hi\n  ↳ querying net_worth()\n\n[error] boom\n\n\n"


def test_render_answer_truncates_tool_arguments_to_120_characters(plain):
    ui.render_answer([tool_event("run_sql", query="x" * 300)])
    args = "query='" + "x" * 113  # the first 120 characters of ``query='xxx...xxx'``
    assert len(args) == 120
    assert plain.getvalue() == f"\nAssistente: \n  ↳ consultando run_sql({args})\n\n\n"


def test_render_answer_does_not_wrap_or_alter_long_text_off_a_terminal(plain):
    chunks = ["palavra " * 40, "x" * 300, "a\tb\r\n  recuo  ", "fim   "]
    ui.render_answer([TextDelta(chunk) for chunk in chunks])
    # The label, the raw stream (no wrapping at 80 columns, tabs, \r and trailing blanks kept), the closing blank line.
    assert plain.getvalue() == "\nAssistente: " + "".join(chunks) + "\n\n"


# ---------------------------------------------------------------------------
# Escape sequences in data (model text, merchant names, item ids) are removed on every path
# ---------------------------------------------------------------------------

ESCAPES = [
    pytest.param("a\x1b[31mred\x1b[0m\x1b[2Jb", "aredb", id="csi"),
    pytest.param("a\x1b]0;title\x07b", "ab", id="osc-bel"),
    pytest.param("a\x1b]8;;http://example.com\x1b\\link\x1b]8;;\x1b\\b", "alinkb", id="osc-st"),
    pytest.param("a\x1bb", "ab", id="lone-esc"),
    pytest.param("a\x1b[2", "a[2", id="truncated-csi"),
    pytest.param("a\x1b]0;ti\x1b[5mtle\x07b", "ab", id="csi-inside-osc"),
]


@pytest.mark.parametrize(("raw", "clean"), ESCAPES)
def test_escape_sequences_in_data_are_removed_off_a_terminal(plain, raw, clean):
    ui.show_notice(Notice("info", raw))
    ui.item_result(ItemResult(raw, ok=False, error=raw, warnings=(raw,)))
    ui.banner(raw, raw, raw, raw)
    assert plain.getvalue() == (
        f"{clean}\n"
        f"  ✗ {clean}: {clean}\n"
        f"    ! {clean}\n"
        f"finassist {clean}\n"
        f"Provedor: {clean}, Modelo: {clean}, Idioma: {clean}\n"
    )


@pytest.mark.parametrize(("raw", "clean"), ESCAPES)
def test_escape_sequences_in_an_answer_are_removed_off_a_terminal(plain, raw, clean):
    ok = ui.render_answer([TextDelta(raw), tool_event(raw, q=raw), AssistantError(raw)])
    # The tool arguments go through ``repr``, which already writes an ESC as the four characters ``\x1b``.
    assert plain.getvalue() == (f"\nAssistente: {clean}\n  ↳ consultando {clean}(q={raw!r})\n\n[erro] {clean}\n\n\n")
    assert ok is False


def test_an_escape_sequence_split_across_deltas_leaves_no_escape(plain):
    ui.render_answer([TextDelta("a\x1b[3"), TextDelta("1mb")])
    assert "\x1b" not in plain.getvalue()


@pytest.mark.parametrize(
    "data",
    [
        "plain text",
        TRICKY,
        "tab\tcr\r\nend  ",
        "a\x1b[31mred\x1b[0m\x1b[2Jb",
        "\x1b[1;5;38mtext\x1b[K",
    ],
)
def test_plain_output_matches_typer_echo_for_csi_and_escape_free_data(plain, data):
    expected = io.StringIO()
    typer.echo(data, file=expected)  # not a terminal: click strips CSI sequences and nothing else
    ui.show_notice(Notice("info", data))
    assert plain.getvalue() == expected.getvalue()


def test_escape_sequences_in_data_never_reach_a_terminal(tty):
    raw = "a\x1b[2Jb\x1b]0;title\x07c\x1b[5md\x1b]8;;http://example.com\x1b\\e\x1bf"
    ui.show_notice(Notice("warn", raw))
    ui.item_result(ItemResult(raw, ok=False, error=raw, warnings=(raw,)))
    ui.banner(raw, raw, raw, raw)
    ui.render_answer([TextDelta(raw), tool_event(raw), AssistantError(raw)])
    out = tty.getvalue()
    for sequence in ("\x1b[2J", "\x1b[5m", "\x1b]0;", "\x1b]8;", "\x1bf"):
        assert sequence not in out
    assert "\x1b" not in strip_ansi(out)  # whatever is left is the CSI style codes Rich emits itself
    assert strip_ansi(out).count("abcdef") >= 8  # the cleaned text is still shown (notice, item, banner, answer, ...)


# ---------------------------------------------------------------------------
# render_answer: live display on a terminal
# ---------------------------------------------------------------------------


def test_render_answer_live_on_forced_terminal(tty):
    events = [
        TextDelta("Você gastou "),
        tool_event(month="2026-11"),
        TextDelta("**R$ 1.234,56** em novembro."),
    ]
    ok = ui.render_answer(events)
    out = strip_ansi(tty.getvalue())
    assert ok is True
    assert "Você gastou R$ 1.234,56 em novembro." in out  # Markdown rendered: the ** markers are gone
    assert "**" not in out
    assert "Assistente" in out
    assert "╭" in out  # the titled panel
    assert "consultando spending_by_category(month='2026-11')" in out  # the status line shown while the tool ran
    assert "\x1b[" in tty.getvalue()  # styled


def test_render_answer_live_shows_errors_and_returns_false(tty):
    ok = ui.render_answer([TextDelta("Parcial"), AssistantError("falhou")])
    out = strip_ansi(tty.getvalue())
    assert ok is False
    assert "Parcial" in out
    assert "[erro] falhou" in out
    assert tty.getvalue().count("\x1b[31m") >= 1


def test_render_answer_live_shows_the_whole_answer_when_taller_than_the_terminal(monkeypatch):
    buffer = io.StringIO()
    monkeypatch.setattr(ui, "console", make_console(buffer, terminal=True, height=10))
    lines = [f"linha {number}" for number in range(1, 31)]
    ui.render_answer([TextDelta(f"{line}\n\n") for line in lines])
    final_frame = strip_ansi(buffer.getvalue()).rsplit("╭", 1)[-1]  # while streaming, Live crops to the height
    assert all(line in final_frame for line in lines)


def test_render_answer_live_without_events_succeeds(tty):
    assert ui.render_answer([]) is True


def test_render_answer_live_follows_the_language(tty, use_language):
    use_language("en")
    ui.render_answer([TextDelta("Hi")])
    assert "Assistant" in strip_ansi(tty.getvalue())
    assert "Assistente" not in tty.getvalue()


# ---------------------------------------------------------------------------
# The event generator is always consumed or closed, and Ctrl+C leaves the terminal clean
# ---------------------------------------------------------------------------


def test_events_are_consumed_to_the_end(plain):
    closed: list[str] = []
    ui.render_answer(tracked([TextDelta("a"), TextDelta("b")], closed))
    assert closed == ["closed"]
    assert plain.getvalue() == "\nAssistente: ab\n\n"


def test_interrupt_while_streaming_off_a_terminal_closes_the_generator(monkeypatch):
    class ExplodingFile(io.StringIO):
        def write(self, text: str) -> int:
            if "boom" in text:
                raise KeyboardInterrupt
            return super().write(text)

    monkeypatch.setattr(ui, "console", make_console(ExplodingFile(), terminal=False))
    closed: list[str] = []
    events = tracked(
        [TextDelta("a"), TextDelta("boom"), TextDelta("never")], closed
    )  # we keep a reference: no GC close
    with pytest.raises(KeyboardInterrupt):
        ui.render_answer(events)
    assert closed == ["closed"]


def test_interrupt_inside_the_live_block_stops_the_display_and_closes_the_generator(tty, monkeypatch):
    original_refresh = Live.refresh
    main_thread_refreshes: list[str] = []

    def interrupt_on_second_refresh(self: Live) -> None:
        # Refreshes of the timer thread are left alone. The main thread's first refresh is Live.start()'s own;
        # its second is the one the tool status asks for, which is where the simulated Ctrl+C lands.
        if threading.current_thread() is threading.main_thread():
            main_thread_refreshes.append("refresh")
            if len(main_thread_refreshes) == 2:
                raise KeyboardInterrupt
        original_refresh(self)

    monkeypatch.setattr(Live, "refresh", interrupt_on_second_refresh)
    monkeypatch.setattr(ui, "_LIVE_REFRESHES_PER_SECOND", 0.01)  # no timer-thread frame during the test
    stdout_before = sys.stdout
    closed: list[str] = []
    events = tracked([TextDelta("Resposta parcial"), tool_event(), TextDelta("never")], closed)  # kept: no GC close
    with pytest.raises(KeyboardInterrupt):
        # The tool status triggers the immediate refresh that raises, with the generator suspended at its yield.
        ui.render_answer(events)
    out = tty.getvalue()
    assert len(main_thread_refreshes) >= 2
    assert closed == ["closed"]
    assert "consultando" not in out  # the final frame, drawn by Live's stop, has no stale tool spinner
    assert "Resposta parcial" in strip_ansi(out)  # ... but still shows the answer received so far
    assert "never" not in out
    assert out.rindex("\x1b[?25h") > out.index("\x1b[?25l")  # cursor hidden by Live, shown again by its stop
    assert sys.stdout is stdout_before  # Live's stdout redirection is undone


def test_interrupt_raised_by_the_generator_leaves_the_terminal_clean(tty):
    closed: list[str] = []

    def events() -> Iterator[AssistantEvent]:
        try:
            yield TextDelta("a")
            raise KeyboardInterrupt
        finally:
            closed.append("closed")

    stdout_before = sys.stdout
    with pytest.raises(KeyboardInterrupt):
        ui.render_answer(events())
    out = tty.getvalue()
    assert closed == ["closed"]
    assert out.rindex("\x1b[?25h") > out.index("\x1b[?25l")
    assert sys.stdout is stdout_before


# ---------------------------------------------------------------------------
# Markdown links never become terminal hyperlinks (the model controls the target)
# ---------------------------------------------------------------------------


def test_live_answer_does_not_emit_terminal_hyperlinks(tty):
    ui.render_answer([TextDelta("Veja [o site](https://example.com/x) agora.")])
    out = tty.getvalue()
    assert "\x1b]8;" not in out  # no OSC 8 sequence, whatever the link target
    assert "o site" in strip_ansi(out)


# ---------------------------------------------------------------------------
# Counts and connection tables: terminal only; plain lines everywhere else
# ---------------------------------------------------------------------------

LONG_ID = "0a1b2c3d-4e5f-6789-abcd-ef0123456789"  # 36 characters, like a Pluggy item id
CONSENT = "consentimento até 01/06/2027"
COUNTS = {"items": 1, "accounts": 2, "transactions": 91, "bills": 2, "investments": 2, "loans": 1, "categories": 3}


def rows():
    return [ui.ItemRow(LONG_ID, "MeuPluggy", "UPDATED", CONSENT), ui.ItemRow("item-2", "Banco do Brasil", "?")]


def terminal(monkeypatch: pytest.MonkeyPatch, width: int) -> io.StringIO:
    """Replace the module console with a forced terminal ``width`` columns wide."""
    buffer = io.StringIO()
    console = Console(
        file=buffer, theme=ui.THEME, width=width, force_terminal=True, color_system="standard", legacy_windows=False
    )
    monkeypatch.setattr(ui, "console", console)
    return buffer


def first_column(table_output: str) -> str:
    """Join the first cell of every table row: a folded id comes back in one piece."""
    return "".join(line.split("│")[1].strip() for line in table_output.splitlines() if line.startswith("│"))


def test_data_counts_plain_line_is_the_old_status_line(plain):
    ui.data_counts(COUNTS)
    assert plain.getvalue() == (
        "Dados: items=1, accounts=2, transactions=91, bills=2, investments=2, loans=1, categories=3\n"
    )


def test_data_counts_plain_line_follows_the_language(plain, use_language):
    use_language("en")
    ui.data_counts({"accounts": 2})
    assert plain.getvalue() == "Data: accounts=2\n"


@pytest.mark.parametrize("width", [80, 120])
def test_data_counts_table_on_a_terminal(monkeypatch, width):
    buffer = terminal(monkeypatch, width)
    ui.data_counts(COUNTS)
    out = strip_ansi(buffer.getvalue())
    assert "╭" in out
    assert "Dados" in out
    for key, value in COUNTS.items():
        assert f"{key}={value}" in out
    assert "Dados:" not in out
    assert all(len(line) <= width for line in out.splitlines())


def test_status_items_plain_lines_are_the_old_status_lines(plain):
    ui.status_items(rows())
    assert plain.getvalue() == (
        f"- {LONG_ID}  MeuPluggy  status=UPDATED  {CONSENT}\n- item-2  Banco do Brasil  status=?\n"
    )


def test_item_list_plain_lines_are_the_old_list_lines(plain):
    ui.item_list(rows())
    assert plain.getvalue() == f"{LONG_ID}  MeuPluggy  UPDATED\nitem-2  Banco do Brasil  ?\n"


@pytest.mark.parametrize("name", ["status_items", "item_list"])
def test_no_rows_print_nothing(plain, name):
    getattr(ui, name)([])
    assert not plain.getvalue()


@pytest.mark.parametrize("name", ["status_items", "item_list"])
def test_no_rows_print_nothing_on_a_terminal_either(tty, name):
    getattr(ui, name)([])
    assert not tty.getvalue()


@pytest.mark.parametrize("width", [80, 120])
def test_status_items_table_keeps_every_cell_intact(monkeypatch, width):
    buffer = terminal(monkeypatch, width)
    ui.status_items(rows())
    out = strip_ansi(buffer.getvalue())
    assert "╭" in out
    assert all(len(line) <= width for line in out.splitlines())  # nothing is wider than the terminal
    assert "…" not in out  # nothing was cropped with an ellipsis
    for text in (CONSENT, "UPDATED", "MeuPluggy", "Banco do Brasil", "item-2", "Consentimento"):
        assert text in out
    assert LONG_ID in first_column(out)  # at 80 columns the id folds onto two lines, but all of it is there
    if width == 120:
        assert LONG_ID in out  # wide enough for one line


@pytest.mark.parametrize("width", [80, 120])
def test_item_list_table_keeps_every_cell_intact(monkeypatch, width):
    buffer = terminal(monkeypatch, width)
    ui.item_list([ui.ItemRow(LONG_ID, "MeuPluggy", "não sincronizado")])
    out = strip_ansi(buffer.getvalue())
    assert "╭" in out
    assert all(len(line) <= width for line in out.splitlines())
    assert "não sincronizado" in out
    assert "MeuPluggy" in out
    assert "Consentimento" not in out  # ``items list`` has no consent column
    assert LONG_ID in first_column(out)


def test_table_headers_follow_the_language(monkeypatch, use_language):
    use_language("en")
    buffer = terminal(monkeypatch, 120)
    ui.status_items(rows())
    out = strip_ansi(buffer.getvalue())
    for header in ("ID", "Connector", "Status", "Consent"):
        assert header in out
    assert "Conector" not in out


def test_items_fall_back_to_one_line_per_item_when_the_columns_cannot_fit(monkeypatch):
    buffer = terminal(monkeypatch, 50)  # the connector, status and consent columns alone need 44 + 13 columns
    ui.status_items(rows())
    out = strip_ansi(buffer.getvalue())
    assert "╭" not in out
    assert "│" not in out
    expected = f"- {LONG_ID}  MeuPluggy  status=UPDATED  {CONSENT}\n- item-2  Banco do Brasil  status=?\n"
    assert out.split() == expected.split()  # the lines may wrap at spaces, but every word is there


@pytest.mark.parametrize("name", ["status_items", "item_list"])
def test_no_color_prints_the_plain_lines_on_a_terminal(monkeypatch, name):
    buffer = terminal(monkeypatch, 120)
    monkeypatch.setenv("NO_COLOR", "1")
    getattr(ui, name)(rows())
    ui.data_counts(COUNTS)
    out = buffer.getvalue()
    assert "\x1b" not in out
    assert "╭" not in out
    assert "│" not in out
    assert "Dados: items=1, accounts=2" in out


def test_table_cells_are_verbatim_and_free_of_escape_sequences(monkeypatch):
    buffer = terminal(monkeypatch, 160)
    ui.status_items([ui.ItemRow(TRICKY, TRICKY + "\x1b[2J", "a\x1b]0;title\x07b", "c\x1bd")])
    raw = buffer.getvalue()
    out = strip_ansi(raw)
    assert out.count(TRICKY) == 2  # the id and the connector: brackets and emoji codes untouched
    assert "ab" in out
    assert "cd" in out
    for sequence in ("\x1b[2J", "\x1b]0;", "\x1bd"):
        assert sequence not in raw
    assert "\x1b" not in out


def test_counts_table_is_free_of_escape_sequences(monkeypatch):
    buffer = terminal(monkeypatch, 80)
    ui.data_counts({"a\x1b[2Jb": 1})
    assert "\x1b[2J" not in buffer.getvalue()
    assert "ab=1" in strip_ansi(buffer.getvalue())


# ---------------------------------------------------------------------------
# Prompt label
# ---------------------------------------------------------------------------


def test_prompt_label_is_unchanged_off_a_terminal(plain):
    assert ui.prompt_label("\nVocê") == "\nVocê"


def test_prompt_label_is_styled_on_a_terminal(tty):
    label = ui.prompt_label("\nVocê")
    assert "\x1b[" in label
    assert strip_ansi(label) == "\nVocê"


def test_prompt_label_respects_no_color(tty, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.prompt_label("\nVocê") == "\nVocê"
