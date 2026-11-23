"""Tests for the page's pure JavaScript (escaping, Markdown, NDJSON line splitting), run in Node without a browser.

The page keeps those functions between ``// <pure>`` and ``// </pure>``; each test runs that block plus a short
snippet with ``node`` and compares what the snippet printed (one JSON value). Skipped when Node.js is missing.
"""

import json
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] — runs the Node binary on a script the test wrote
from pathlib import Path
from typing import Any

import pytest

from llmopenfinance import session
from llmopenfinance.web_page import render_page

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

PURE_BLOCK = re.compile(r"// <pure>\n(.*?)// </pure>", re.S)
TAG = re.compile(r"<(/?)([a-zA-Z0-9]+)([^>]*)>")
ALLOWED_TAGS = {"p", "br", "strong", "em", "code", "pre", "h2", "h3", "h4", "ul", "ol", "li", "hr", "div"}
ALLOWED_TAGS |= {"table", "thead", "tbody", "tr", "th", "td"}
ALLOWED_ATTRIBUTES = re.compile(r'(?: class="(?:r|c|tbl)"| start="\d+")?')

HOSTILE = [
    "<img src=x onerror=alert(1)>",
    "<script>alert(1)</script>",
    "**<svg onload=alert(1)>**",
    "*<iframe src=javascript:alert(1)>*",
    "`<b onclick=x>`",
    "# <h1 onmouseover=x>",
    "- <a href=javascript:alert(1)>x</a>\n- ok",
    "1. <img src=x onerror=1>",
    '| <b>a</b> | <i>b</i> |\n|---|:-:|\n| <script>c</script> | d" onclick="x |',
    "```\n</code></pre><img src=x onerror=1>\n```",
    "[x](javascript:alert(1)) ![y](https://example.invalid/y.png)",
    '"><img src=x onerror=1> & &amp; &lt;',
    "**ok** <img src=x onerror=alert(1)> | a | b |",
]


@pytest.fixture(scope="module")
def pure() -> str:
    page = render_page(language="pt-BR", strings={}, nonce="test")
    match = PURE_BLOCK.search(page)
    assert match is not None
    return match.group(1)


@pytest.fixture
def run_js(pure: str, tmp_path: Path) -> Any:
    def run(snippet: str) -> Any:
        assert NODE is not None
        script = tmp_path / "case.js"
        helper = "const out = (value) => process.stdout.write(JSON.stringify(value));\n"
        script.write_text(pure + "\n" + helper + snippet, encoding="utf-8")
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] — fixed argv: the Node binary and a script this test wrote
            [NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return run


def js(value: Any) -> str:
    """Encode a Python value as a JavaScript literal."""
    return json.dumps(value)


def markdown(run_js: Any, text: str) -> str:
    return run_js(f"out(renderMarkdown({js(text)}));")


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


def test_escape_html(run_js: Any):
    assert run_js("""out(escapeHtml(`<a href="x">'&`));""") == "&lt;a href=&quot;x&quot;&gt;&#39;&amp;"


def test_markdown_escapes_html_first(run_js: Any):
    html = markdown(run_js, "<img src=x onerror=alert(1)>\n\n<script>alert(1)</script>")
    assert html == "<p>&lt;img src=x onerror=alert(1)&gt;</p><p>&lt;script&gt;alert(1)&lt;/script&gt;</p>"


def test_markdown_links_and_images_stay_plain_text(run_js: Any):
    text = "[clique](javascript:alert(1)) ![x](https://example.invalid/x.png) <javascript:alert(1)>"
    expected = "<p>[clique](javascript:alert(1)) ![x](https://example.invalid/x.png) &lt;javascript:alert(1)&gt;</p>"
    assert markdown(run_js, text) == expected


def test_markdown_table_cells_and_list_items_are_escaped(run_js: Any):
    html = markdown(run_js, "| <b>a</b> | b |\n|---|---|\n| <img src=x> | ok |\n\n- <i>y</i>\n1. <u>z</u>")
    assert "<th>&lt;b&gt;a&lt;/b&gt;</th>" in html
    assert "<td>&lt;img src=x&gt;</td>" in html
    assert "<li>&lt;i&gt;y&lt;/i&gt;</li>" in html
    assert "<li>&lt;u&gt;z&lt;/u&gt;</li>" in html


def test_markdown_only_emits_known_tags_without_free_attributes(run_js: Any):
    outputs = run_js(f"out({js(HOSTILE)}.map(renderMarkdown));")
    assert len(outputs) == len(HOSTILE)
    for html in outputs:
        tags = TAG.findall(html)
        assert html.count("<") == len(tags), html  # every "<" starts one of the renderer's own tags
        for closing, name, attributes in tags:
            assert name in ALLOWED_TAGS, html
            assert ALLOWED_ATTRIBUTES.fullmatch(attributes), html
            assert not (closing and attributes), html
        assert "<img" not in html


# ---------------------------------------------------------------------------
# Markdown features
# ---------------------------------------------------------------------------


def test_markdown_inline_formatting(run_js: Any):
    html = markdown(run_js, "**negrito** e *itálico* e _também_ e `a<b` e **`x`**")
    assert html == (
        "<p><strong>negrito</strong> e <em>itálico</em> e <em>também</em> e <code>a&lt;b</code> e "
        "<strong><code>x</code></strong></p>"
    )


def test_markdown_does_not_emphasise_snake_case_or_spaced_stars(run_js: Any):
    assert markdown(run_js, "nome_de_coluna e 2 * 3 * 4") == "<p>nome_de_coluna e 2 * 3 * 4</p>"


def test_markdown_headings(run_js: Any):
    html = markdown(run_js, "# Um\n## Dois **b**\n### Três\n#### quatro")
    assert html == "<h2>Um</h2><h3>Dois <strong>b</strong></h3><h4>Três</h4><p>#### quatro</p>"


def test_markdown_lists(run_js: Any):
    html = markdown(run_js, "- a\n* **b**\n\n3. c\n4. d\n\n1) e")
    assert html == (
        '<ul><li>a</li><li><strong>b</strong></li></ul><ol start="3"><li>c</li><li>d</li></ol><ol><li>e</li></ol>'
    )


def test_markdown_table(run_js: Any):
    html = markdown(run_js, "Total:\n\n| Categoria | Valor | Nota |\n|:---|---:|:-:|\n| Mercado | R$ 1,00 |\n")
    assert html == (
        '<p>Total:</p><div class="tbl"><table><thead><tr><th>Categoria</th><th class="r">Valor</th>'
        '<th class="c">Nota</th></tr></thead><tbody><tr><td>Mercado</td><td class="r">R$ 1,00</td>'
        '<td class="c"></td></tr></tbody></table></div>'
    )


def test_markdown_pipe_line_without_separator_is_a_paragraph(run_js: Any):
    html = markdown(run_js, "**ok** <img src=x onerror=alert(1)> | a | b |")
    assert html == "<p><strong>ok</strong> &lt;img src=x onerror=alert(1)&gt; | a | b |</p>"


def test_markdown_fenced_code_is_escaped_and_verbatim(run_js: Any):
    html = markdown(run_js, "```sql\nselect '<x>' * 2 -- **não**\n```\ndepois")
    assert html == "<pre><code>select &#39;&lt;x&gt;&#39; * 2 -- **não**</code></pre><p>depois</p>"


def test_markdown_unclosed_fence_while_streaming(run_js: Any):
    assert markdown(run_js, "```\nparcial") == "<pre><code>parcial</code></pre>"


def test_markdown_paragraph_lines_and_rule(run_js: Any):
    html = markdown(run_js, "linha 1\r\nlinha 2\n\n---\nfim")
    assert html == "<p>linha 1<br>linha 2</p><hr><p>fim</p>"


# ---------------------------------------------------------------------------
# NDJSON line splitter
# ---------------------------------------------------------------------------


def chunks_snippet(chunks: list[bytes]) -> str:
    arrays = ", ".join(f"new Uint8Array({js(list(chunk))})" for chunk in chunks)
    return f"""
const splitter = createLineSplitter();
const seen = [{arrays}].map((chunk) => splitter.push(chunk));
out({{seen, end: splitter.end()}});
"""


def test_line_splitter_line_split_across_chunks(run_js: Any):
    data = b'{"type":"text","text":"ol\xc3\xa1"}\n{"type":"done","ok":true}\n'
    result = run_js(chunks_snippet([data[:7], data[7:20], data[20:]]))
    assert result == {"seen": [[], [], ['{"type":"text","text":"olá"}', '{"type":"done","ok":true}']], "end": []}


def test_line_splitter_several_lines_in_one_chunk(run_js: Any):
    data = b'{"a":1}\n\n{"b":2}\n{"c":3}'
    result = run_js(chunks_snippet([data]))
    assert result == {"seen": [['{"a":1}', '{"b":2}']], "end": ['{"c":3}']}


def test_line_splitter_split_multibyte_character(run_js: Any):
    data = '{"type":"text","text":"ação \N{LINE SEPARATOR} €"}\n'.encode()
    cut_cedilla = data.index("ç".encode()) + 1
    cut_euro = data.index("€".encode()) + 2
    result = run_js(chunks_snippet([data[:cut_cedilla], data[cut_cedilla:cut_euro], data[cut_euro:]]))
    assert result["seen"][:2] == [[], []]
    (line,) = result["seen"][2]
    assert json.loads(line) == {"type": "text", "text": "ação \N{LINE SEPARATOR} €"}
    assert result["end"] == []


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_chat_command_matches_the_server_aliases(run_js: Any):
    aliases = sorted(session._COMMAND_ALIASES)
    texts = [*aliases, "  /SYNC ", "Reset", "sincronizar", "/sync agora", "como reiniciar?"]
    results = run_js(f"out({js(texts)}.map(chatCommand));")
    expected = [name if name in {"sync", "reset"} else None for name in map(session.parse_command, texts)]
    assert results == expected
    assert "sync" in results
    assert "reset" in results


def test_tool_text(run_js: Any):
    result = run_js("""out([toolText("↳ consultando", "query", {sql: "select 1", n: 2}), toolText("p", "x", null)]);""")
    assert result == ['↳ consultando query(sql="select 1", n=2)', "p x()"]
    long = run_js("""out(toolText("p", "x", {a: "y".repeat(500)}).length);""")
    assert long == len("p x()") + 120


def test_progress_lines(run_js: Any):
    events = [
        {
            "item_id": "i1",
            "ok": True,
            "skipped": False,
            "counts": {"accounts": 2, "transactions": 9},
            "warnings": ["w"],
        },
        {"item_id": "i2", "ok": True, "skipped": True, "counts": {}, "warnings": []},
        {"item_id": "i3", "ok": False, "skipped": False, "counts": {}, "warnings": [], "error": "boom"},
    ]
    result = run_js(f"out({js(events)}.map((event) => progressLines(event, 'sem alterações')));")
    assert result == [
        [{"level": "ok", "text": "✓ i1: accounts=2, transactions=9"}, {"level": "warn", "text": "! w"}],
        [{"level": "dim", "text": "= i2: sem alterações"}],
        [{"level": "error", "text": "✗ i3: boom"}],
    ]


def test_format_when(run_js: Any):
    result = run_js(
        'const iso = "2026-11-22T10:05:00+00:00";'
        'out([formatWhen(iso, "pt-BR"), formatWhen(iso, "en"), formatWhen("ontem", "en"), formatCounts({a: 1, b: 2})]);'
    )
    assert re.fullmatch(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", result[0])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", result[1])
    assert result[2:] == ["ontem", "a=1, b=2"]
