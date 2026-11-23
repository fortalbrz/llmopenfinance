"""The web chat's single page: one self-contained HTML document with inline CSS and JavaScript.

The page loads nothing from anywhere: no external scripts, styles, fonts or images. Its ``<style>`` and ``<script>``
carry a per-request nonce, and ``web.py`` serves it with a ``Content-Security-Policy`` that allows only those two
inline blocks and ``fetch`` to the same origin, so injected markup could not run even if escaping failed.

Model output and transaction descriptions are third-party text. The script inserts text with ``textContent``; the
only HTML sink is the assistant's Markdown, and ``renderMarkdown`` escapes the whole text **first** and then adds a
fixed set of tags (headings, bold, italics, code, lists, tables) without attributes taken from the text. It has no
links or images, so there is no URL to abuse. The localised strings reach the script as a JSON data block whose
``<``, ``>``, ``&``, U+2028 and U+2029 are ``\\uXXXX`` escapes, so no string can close the ``<script>`` element.

The pure helpers (``escapeHtml``, ``renderMarkdown``, the NDJSON line splitter and the line formatters) sit between
``// <pure>`` and ``// </pure>`` at the top of the script, so the tests can run them in Node.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["render_page"]

# Placeholder in _BODY -> catalog key of the text it shows.
_LABELS = {
    "title": "web.ui.title",
    "busy": "web.ui.busy",
    "help": "web.ui.help",
    "summary": "web.ui.summary",
    "history": "web.ui.history",
    "sync": "web.ui.sync",
    "reset": "web.ui.reset",
    "send": "web.ui.send",
    "placeholder": "web.ui.input_placeholder",
    "message_label": "web.ui.message_label",
    "messages_label": "web.ui.messages_label",
    "commands_label": "web.ui.commands_label",
    "empty": "web.ui.empty",
    "hint": "web.ui.hint",
}

# Characters that must not appear literally inside the JSON data block (``</script>`` and the JS line separators).
_SCRIPT_ESCAPES = str.maketrans(
    {
        "<": "\\u003c",
        ">": "\\u003e",
        "&": "\\u0026",
        "\N{LINE SEPARATOR}": "\\u2028",
        "\N{PARAGRAPH SEPARATOR}": "\\u2029",
    }
)

_HEAD = """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<style nonce="{nonce}">
"""

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f5f4ef; --surface: #ffffff; --surface-2: #eeece5; --ink: #1c2220; --muted: #5b6561;
  --line: #dddad1; --accent: #1b6a56; --accent-ink: #ffffff; --accent-soft: #e1eee8;
  --ok: #1b7a4c; --warn: #8a5800; --err: #b42318;
  --radius: 14px; --shadow: 0 1px 2px rgb(20 30 25 / 6%), 0 6px 20px rgb(20 30 25 / 5%);
  --font: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "Cascadia Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  --gutter: max(16px, calc((100% - 50rem) / 2));
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #101413; --surface: #171c1a; --surface-2: #1f2623; --ink: #e5ebe8; --muted: #98a59f;
    --line: #2b3330; --accent: #5ec4a4; --accent-ink: #0a1511; --accent-soft: #19302a;
    --ok: #72d4a2; --warn: #e9b64c; --err: #ff8f84; --shadow: none;
  }
}
*, *::before, *::after { box-sizing: border-box; }
[hidden] { display: none !important; }
html, body { height: 100%; }
body {
  margin: 0; display: flex; flex-direction: column; height: 100dvh; overflow: hidden;
  background: var(--bg); color: var(--ink); font: 16px/1.55 var(--font);
  -webkit-text-size-adjust: 100%; text-size-adjust: 100%;
}
button, textarea { font: inherit; color: inherit; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.vh {
  position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}
.top {
  padding: 14px var(--gutter) 12px; background: var(--surface); border-bottom: 1px solid var(--line);
}
.bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 28px; }
h1 {
  margin: 0; font-size: 1.05rem; font-weight: 650; letter-spacing: -0.01em;
  display: flex; align-items: center; gap: 10px;
}
h1::before {
  content: ""; width: 10px; height: 10px; border-radius: 3px; background: var(--accent);
  box-shadow: 4px 4px 0 var(--accent-soft);
}
.status {
  margin: 6px 0 10px; display: flex; flex-wrap: wrap; gap: 2px 14px;
  font-size: 0.78rem; color: var(--muted); font-variant-numeric: tabular-nums;
}
.status span { overflow-wrap: anywhere; }
.status b { font-weight: 600; color: var(--ink); margin-right: 4px; }
.tools { display: flex; flex-wrap: wrap; gap: 8px; }
.tools button {
  border: 1px solid var(--line); background: var(--bg); border-radius: 999px;
  padding: 5px 13px; font-size: 0.85rem; cursor: pointer;
  transition: border-color 0.15s, background-color 0.15s;
}
.tools button:hover:not(:disabled) { border-color: var(--accent); background: var(--accent-soft); }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.busy { display: inline-flex; align-items: center; gap: 8px; font-size: 0.8rem; color: var(--accent); }
.dots { display: inline-flex; gap: 4px; }
.dots i {
  width: 6px; height: 6px; border-radius: 50%; background: currentColor;
  animation: pulse 1.2s ease-in-out infinite;
}
.dots i:nth-child(2) { animation-delay: 0.15s; }
.dots i:nth-child(3) { animation-delay: 0.3s; }
@keyframes pulse { 0%, 80%, 100% { opacity: 0.25; transform: scale(0.75); } 40% { opacity: 1; transform: none; } }
main { flex: 1; overflow-y: auto; overflow-x: hidden; padding: 20px var(--gutter) 24px; }
.messages { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.msg { overflow-wrap: anywhere; min-width: 0; }
.msg.user {
  align-self: flex-end; max-width: min(85%, 36rem); padding: 9px 14px; white-space: pre-wrap;
  background: var(--accent-soft); border-radius: var(--radius) var(--radius) 4px var(--radius);
}
.msg.assistant {
  max-width: 100%; padding: 12px 16px; background: var(--surface); border: 1px solid var(--line);
  border-radius: 4px var(--radius) var(--radius) var(--radius); box-shadow: var(--shadow);
}
.who {
  display: block; margin-bottom: 4px; font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
}
.line {
  padding: 1px 0 1px 12px; border-left: 2px solid var(--line); font-size: 0.875rem;
  white-space: pre-wrap; overflow-wrap: anywhere; color: var(--ink);
}
.line.tool, .line.dim { color: var(--muted); }
.line.tool { font-family: var(--mono); font-size: 0.78rem; border-left-style: dotted; }
.line.ok { border-left-color: var(--ok); }
.line.warn { border-left-color: var(--warn); }
.line.error { border-left-color: var(--err); color: var(--err); }
.md > :first-child { margin-top: 0; }
.md > :last-child { margin-bottom: 0; }
.md p, .md ul, .md ol, .md pre, .md .tbl { margin: 0.55em 0; }
.md h2, .md h3, .md h4 { margin: 0.9em 0 0.35em; line-height: 1.3; letter-spacing: -0.01em; }
.md h2 { font-size: 1.15rem; }
.md h3 { font-size: 1.05rem; }
.md h4 { font-size: 0.95rem; }
.md ul, .md ol { padding-left: 1.4em; }
.md li + li { margin-top: 0.2em; }
.md hr { border: 0; border-top: 1px solid var(--line); margin: 1em 0; }
.md code {
  font-family: var(--mono); font-size: 0.86em; background: var(--surface-2);
  padding: 0.1em 0.35em; border-radius: 5px;
}
.md pre { background: var(--surface-2); padding: 12px 14px; border-radius: 10px; overflow-x: auto; }
.md pre code { background: none; padding: 0; font-size: 0.82rem; white-space: pre; }
.tbl { overflow-x: auto; }
.tbl table { border-collapse: collapse; font-size: 0.88rem; font-variant-numeric: tabular-nums; }
.tbl th, .tbl td {
  padding: 6px 12px; text-align: left; border-bottom: 1px solid var(--line);
  overflow-wrap: normal; vertical-align: top;
}
.tbl th { font-weight: 600; border-bottom-width: 2px; }
.tbl .r { text-align: right; }
.tbl .c { text-align: center; }
.empty { margin: 18vh auto 0; max-width: 26rem; text-align: center; color: var(--muted); }
.composer {
  display: grid; grid-template-columns: 1fr auto; gap: 6px 8px; align-items: end;
  padding: 12px var(--gutter) max(12px, env(safe-area-inset-bottom));
  background: var(--surface); border-top: 1px solid var(--line);
}
textarea {
  width: 100%; min-height: 46px; max-height: 40vh; resize: none; field-sizing: content;
  padding: 11px 14px; background: var(--bg); border: 1px solid var(--line); border-radius: 12px;
}
textarea:focus-visible { outline-offset: 0; border-color: var(--accent); }
.send {
  min-height: 46px; padding: 0 20px; border: 0; border-radius: 12px; cursor: pointer;
  background: var(--accent); color: var(--accent-ink); font-weight: 600;
}
.hint { grid-column: 1 / -1; margin: 0; font-size: 0.72rem; color: var(--muted); }
@media (max-width: 480px) {
  .tools button { padding: 5px 11px; font-size: 0.8rem; }
  .msg.user { max-width: 92%; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; scroll-behavior: auto !important; }
}
"""

_BODY = """</style>
</head>
<body>
<header class="top">
  <div class="bar">
    <h1>{title}</h1>
    <div id="busy" class="busy" role="status" hidden><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>
      <span>{busy}</span></div>
  </div>
  <p id="status" class="status"></p>
  <nav class="tools" aria-label="{commands_label}">
    <button type="button" data-command="help">{help}</button>
    <button type="button" data-command="summary">{summary}</button>
    <button type="button" data-command="history">{history}</button>
    <button type="button" id="sync">{sync}</button>
    <button type="button" data-command="reset">{reset}</button>
  </nav>
</header>
<main id="main">
  <p id="empty" class="empty" hidden>{empty}</p>
  <ol id="messages" class="messages" aria-live="polite" aria-label="{messages_label}"></ol>
</main>
<form id="composer" class="composer" novalidate>
  <label for="input" class="vh">{message_label}</label>
  <textarea id="input" rows="1" placeholder="{placeholder}" aria-describedby="hint"></textarea>
  <button type="submit" id="send" class="send">{send}</button>
  <p id="hint" class="hint">{hint}</p>
</form>
"""

_SCRIPT = r"""
"use strict";
// <pure>
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Bold and italics over text that is ALREADY escaped; only fixed tags are added around escaped content.
function emphasis(escaped) {
  return escaped
    .replace(/\*\*(?=\S)(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^\p{L}\p{N}_])__(?=\S)(.+?)__(?![\p{L}\p{N}_])/gu, "$1<strong>$2</strong>")
    .replace(/(^|[^\p{L}\p{N}_*])\*(?=[^\s*])([^*]*?[^\s*])?\*(?![\p{L}\p{N}_*])/gu, "$1<em>$2</em>")
    .replace(/(^|[^\p{L}\p{N}_])_(?=[^\s_])([^_]*?[^\s_])?_(?![\p{L}\p{N}_])/gu, "$1<em>$2</em>");
}

// Inline Markdown for one escaped line: `code` spans are set aside (as \u0000n\u0000 marks) so emphasis never
// touches them, then put back verbatim.
function renderInline(escaped) {
  const codes = [];
  const marked = escaped.replace(/\u0000/g, "").replace(/`([^`]+)`/g, (match, code) => {
    codes.push(code);
    return "\u0000" + (codes.length - 1) + "\u0000";
  });
  return emphasis(marked).replace(/\u0000(\d+)\u0000/g, (match, n) => "<code>" + codes[Number(n)] + "</code>");
}

function inline(raw) {
  return renderInline(escapeHtml(raw));
}

const FENCE = /^\s*```/;
const HEADING = /^\s{0,3}(#{1,3})\s+(.*?)\s*#*\s*$/;
const RULE = /^\s{0,3}([-*_])(\s*\1){2,}\s*$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const NUMBERED = /^\s*(\d{1,9})[.)]\s+(.*)$/;
const TABLE_RULE = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

function tableRow(line) {
  let row = line.trim();
  if (row.startsWith("|")) row = row.slice(1);
  if (row.endsWith("|")) row = row.slice(0, -1);
  return row.split("|").map((cell) => cell.trim());
}

function isTableStart(lines, i) {
  return lines[i].includes("|") && i + 1 < lines.length && lines[i + 1].includes("|") && TABLE_RULE.test(lines[i + 1]);
}

function renderTable(rows, aligns) {
  const cell = (tag, text, j) =>
    "<" + tag + (aligns[j] ? ' class="' + aligns[j] + '"' : "") + ">" + inline(text) + "</" + tag + ">";
  const head = rows[0];
  const body = rows.slice(1).map((row) => "<tr>" + head.map((_, j) => cell("td", row[j] || "", j)).join("") + "</tr>");
  return (
    '<div class="tbl"><table><thead><tr>' + head.map((text, j) => cell("th", text, j)).join("") + "</tr></thead>" +
    (body.length ? "<tbody>" + body.join("") + "</tbody>" : "") + "</table></div>"
  );
}

function isBlockStart(lines, i) {
  const line = lines[i];
  return (
    FENCE.test(line) || HEADING.test(line) || RULE.test(line) || BULLET.test(line) || NUMBERED.test(line) ||
    isTableStart(lines, i)
  );
}

// Minimal Markdown -> HTML. The text is escaped FIRST (every piece goes through escapeHtml before any tag is
// added); links and images are not supported and stay plain text, so the output never carries a URL.
function renderMarkdown(text) {
  const lines = String(text).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (FENCE.test(line)) {
      const code = [];
      for (i += 1; i < lines.length && !FENCE.test(lines[i]); i += 1) code.push(lines[i]);
      i += 1;
      out.push("<pre><code>" + escapeHtml(code.join("\n")) + "</code></pre>");
    } else if (isTableStart(lines, i)) {
      const aligns = tableRow(lines[i + 1]).map((rule) =>
        rule.endsWith(":") ? (rule.startsWith(":") ? "c" : "r") : "");
      const rows = [tableRow(line)];
      for (i += 2; i < lines.length && lines[i].includes("|") && lines[i].trim(); i += 1) rows.push(tableRow(lines[i]));
      out.push(renderTable(rows, aligns));
    } else if (HEADING.test(line)) {
      const [, marks, title] = HEADING.exec(line);
      const tag = "h" + (marks.length + 1);
      out.push("<" + tag + ">" + inline(title) + "</" + tag + ">");
      i += 1;
    } else if (RULE.test(line)) {
      out.push("<hr>");
      i += 1;
    } else if (BULLET.test(line)) {
      const items = [];
      for (; i < lines.length && BULLET.test(lines[i]); i += 1) {
        items.push("<li>" + inline(BULLET.exec(lines[i])[1]) + "</li>");
      }
      out.push("<ul>" + items.join("") + "</ul>");
    } else if (NUMBERED.test(line)) {
      const start = parseInt(NUMBERED.exec(line)[1], 10);
      const items = [];
      for (; i < lines.length && NUMBERED.test(lines[i]); i += 1) {
        items.push("<li>" + inline(NUMBERED.exec(lines[i])[2]) + "</li>");
      }
      out.push((start === 1 ? "<ol>" : '<ol start="' + start + '">') + items.join("") + "</ol>");
    } else if (!line.trim()) {
      i += 1;
    } else {
      const para = [];
      for (; i < lines.length && lines[i].trim() && (para.length === 0 || !isBlockStart(lines, i)); i += 1) {
        para.push(inline(lines[i].trim()));
      }
      out.push("<p>" + para.join("<br>") + "</p>");
    }
  }
  return out.join("");
}

// NDJSON: decode byte chunks (a multi-byte character may be split across chunks) and return complete lines.
function createLineSplitter() {
  const decoder = new TextDecoder();
  let buffer = "";
  const complete = (lines) => lines.filter((line) => line.trim() !== "");
  return {
    push(bytes) {
      buffer += decoder.decode(bytes, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      return complete(lines);
    },
    end() {
      const lines = complete([buffer + decoder.decode()]);
      buffer = "";
      return lines;
    },
  };
}

// The chat-box commands whose effect the page redraws (mirrors the "sync" and "reset" aliases of the server).
function chatCommand(text) {
  const command = String(text).trim().toLowerCase();
  if (command === "/sync") return "sync";
  if (command === "/reset" || command === "reset") return "reset";
  return null;
}

function formatCounts(counts) {
  return Object.entries(counts || {}).map(([key, value]) => key + "=" + value).join(", ");
}

function toolText(prefix, name, args) {
  const shown = Object.entries(args || {}).map(([key, value]) => key + "=" + JSON.stringify(value)).join(", ");
  return prefix + " " + name + "(" + shown.slice(0, 120) + ")";
}

function progressLines(event, unchanged) {
  const lines = [];
  if (!event.ok) lines.push({ level: "error", text: "✗ " + event.item_id + ": " + (event.error || "") });
  else if (event.skipped) lines.push({ level: "dim", text: "= " + event.item_id + ": " + unchanged });
  else lines.push({ level: "ok", text: "✓ " + event.item_id + ": " + formatCounts(event.counts) });
  for (const warning of event.warnings || []) lines.push({ level: "warn", text: "! " + warning });
  return lines;
}

function formatWhen(value, language) {
  const date = new Date(value);
  if (typeof value !== "string" || Number.isNaN(date.getTime())) return String(value);
  const two = (n) => String(n).padStart(2, "0");
  const day = language === "en"
    ? date.getFullYear() + "-" + two(date.getMonth() + 1) + "-" + two(date.getDate())
    : two(date.getDate()) + "/" + two(date.getMonth() + 1) + "/" + date.getFullYear();
  return day + " " + two(date.getHours()) + ":" + two(date.getMinutes());
}
// </pure>

(() => {
  const strings = JSON.parse(document.getElementById("i18n").textContent);
  const S = (key) => (Object.prototype.hasOwnProperty.call(strings, key) ? strings[key] : key);
  const $ = (id) => document.getElementById(id);
  const list = $("messages");
  const main = $("main");
  const input = $("input");
  const form = $("composer");
  const busyIndicator = $("busy");
  const statusLine = $("status");
  const empty = $("empty");
  const controls = Array.from(document.querySelectorAll("button, textarea"));
  const LEVELS = ["info", "ok", "warn", "error", "dim"];
  let language = document.documentElement.lang;
  let localBusy = false;
  let serverBusy = false;
  let pollTimer = 0;
  let captured = null; // nodes appended during a sync or reset, kept across the redraw

  const atBottom = () => main.scrollHeight - main.scrollTop - main.clientHeight < 96;
  const toBottom = () => { main.scrollTop = main.scrollHeight; };

  function append(node) {
    const stick = atBottom();
    list.appendChild(node);
    if (captured) captured.push(node);
    empty.hidden = true;
    if (stick) toBottom();
  }

  function line(level, text) {
    const item = document.createElement("li");
    item.className = "line " + (LEVELS.includes(level) || level === "tool" ? level : "info");
    item.textContent = text;
    return item;
  }

  function message(role, label) {
    const item = document.createElement("li");
    item.className = "msg " + role;
    const who = document.createElement("span");
    who.className = role === "user" ? "vh" : "who";
    who.textContent = label;
    const body = document.createElement("div");
    item.append(who, body);
    return { item, body };
  }

  function userMessage(text) {
    const { item, body } = message("user", S("web.ui.you"));
    body.textContent = text;
    return item;
  }

  function setMarkdown(body, text) {
    body.innerHTML = renderMarkdown(text); // the only HTML sink: renderMarkdown escapes everything first
  }

  function assistantMessage(text) {
    const parts = message("assistant", S("web.ui.assistant"));
    parts.body.className = "md";
    if (text) setMarkdown(parts.body, text);
    return parts;
  }

  const errorLine = (text) => line("error", S("web.ui.error_prefix") + " " + text);
  const networkError = () => append(line("error", S("web.ui.network_error")));

  function renderBusy() {
    const busy = localBusy || serverBusy;
    busyIndicator.hidden = !busy;
    list.setAttribute("aria-busy", busy ? "true" : "false");
    for (const control of controls) control.disabled = busy;
    if (serverBusy && !localBusy && !pollTimer) pollTimer = setTimeout(poll, 2000);
  }

  function renderStatus(state) {
    const items = [
      ["web.ui.status_provider", state.provider],
      ["web.ui.status_model", state.model],
      ["web.ui.status_language", state.language],
      ["web.ui.status_last_sync", state.last_sync ? formatWhen(state.last_sync, language) : S("web.ui.status_never")],
      ["web.ui.status_data", formatCounts(state.counts)],
    ];
    statusLine.replaceChildren(...items.map(([key, value]) => {
      const span = document.createElement("span");
      const label = document.createElement("b");
      label.textContent = S(key);
      span.append(label, String(value ?? ""));
      return span;
    }));
  }

  function renderTranscript(turns) {
    list.replaceChildren();
    for (const turn of Array.isArray(turns) ? turns : []) {
      if (turn.role === "user") list.appendChild(userMessage(turn.text));
      else list.appendChild(assistantMessage(turn.text).item);
    }
    empty.hidden = list.children.length > 0;
    toBottom();
  }

  async function errorText(response) {
    try {
      const data = await response.json();
      if (data && typeof data.error === "string") return data.error;
    } catch {
      // not JSON: fall through to the generic message
    }
    return S("web.ui.network_error");
  }

  async function fetchState() {
    const response = await fetch("/api/state", { credentials: "same-origin", headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(await errorText(response));
    return response.json();
  }

  function applyState(state, withTranscript) {
    if (state.strings && typeof state.strings === "object") Object.assign(strings, state.strings);
    if (typeof state.language === "string") language = state.language;
    renderStatus(state);
    if (withTranscript) renderTranscript(state.transcript);
    serverBusy = state.busy === true;
    renderBusy();
  }

  async function poll() {
    pollTimer = 0;
    if (localBusy) return;
    try {
      const state = await fetchState();
      if (!localBusy) applyState(state, !state.busy);
    } catch {
      serverBusy = false;
      renderBusy();
    }
  }

  function post(url, payload) {
    return fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json, application/x-ndjson" },
      body: JSON.stringify(payload),
    });
  }

  // Renders the events of one stream; text deltas are re-rendered at most once per animation frame.
  function streamView() {
    let current = null;
    let text = "";
    let frame = 0;
    const flush = () => {
      frame = 0;
      if (!current) return;
      const stick = atBottom();
      setMarkdown(current.body, text);
      if (stick) toBottom();
    };
    const close = () => {
      if (frame) cancelAnimationFrame(frame);
      flush();
      current = null;
      text = "";
    };
    const show = (event) => {
      switch (event.type) {
        case "text":
          if (typeof event.text !== "string") return;
          if (!current) {
            current = assistantMessage("");
            append(current.item);
          }
          text += event.text;
          if (!frame) frame = requestAnimationFrame(flush);
          break;
        case "tool":
          close();
          append(line("tool", toolText(S("web.ui.tool_prefix"), String(event.name), event.arguments)));
          break;
        case "notice":
          close();
          append(line(event.level, String(event.text)));
          break;
        case "progress":
          close();
          for (const item of progressLines(event, S("web.ui.sync_unchanged"))) append(line(item.level, item.text));
          break;
        case "error":
          close();
          append(errorLine(String(event.text)));
          break;
        default:
          break;
      }
    };
    return { show, close };
  }

  // POSTs and reads the NDJSON reply. Returns the "done" flag, or null when the request was refused (409, ...).
  // onRefused runs when the server never took the request (refused, or unreachable before any reply).
  async function stream(url, payload, onRefused) {
    const view = streamView();
    let response;
    try {
      response = await post(url, payload);
    } catch (error) {
      if (onRefused) onRefused();
      throw error;
    }
    if (!response.ok || !response.body) {
      if (onRefused) onRefused();
      append(line(response.status === 409 ? "warn" : "error", await errorText(response)));
      return null;
    }
    const reader = response.body.getReader();
    const splitter = createLineSplitter();
    let ok = false;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        for (const raw of done ? splitter.end() : splitter.push(value)) {
          const event = JSON.parse(raw);
          if (event && event.type === "done") ok = event.ok === true;
          else if (event) view.show(event);
        }
        if (done) break;
      }
    } catch (error) {
      reader.cancel().catch(() => {});
      throw error;
    } finally {
      view.close();
    }
    return ok;
  }

  // After a sync or a reset the server starts a new conversation: redraw it, keep this action's lines.
  async function redraw(command, ok) {
    const kept = captured || [];
    captured = null;
    try {
      applyState(await fetchState(), true);
      for (const node of kept) list.appendChild(node);
      empty.hidden = list.children.length > 0;
    } catch {
      networkError();
    }
    if (command === "sync") {
      const key = ok ? "web.ui.conversation_reset" : "web.ui.conversation_reset_failed";
      append(line(ok ? "ok" : "warn", S(key)));
    }
    toBottom();
  }

  async function run(task, focusInput) {
    if (localBusy) return;
    localBusy = true;
    renderBusy();
    try {
      await task();
    } catch {
      networkError();
    } finally {
      captured = null;
      localBusy = false;
      renderBusy();
      if (focusInput) input.focus({ preventScroll: true });
    }
  }

  function sendMessage() {
    const text = input.value.trim();
    if (!text || localBusy || serverBusy) return;
    const command = chatCommand(text);
    input.value = "";
    run(async () => {
      captured = command ? [] : null;
      const bubble = userMessage(text);
      append(bubble);
      const ok = await stream("/api/chat", { message: text }, () => {
        bubble.remove();
        if (!input.value) input.value = text;
      });
      if (command && ok !== null) await redraw(command, ok);
    }, true);
  }

  function runSync() {
    run(async () => {
      captured = [];
      const ok = await stream("/api/sync", { force: false }, null);
      if (ok !== null) await redraw("sync", ok);
    }, false);
  }

  function runCommand(name) {
    run(async () => {
      const response = await post("/api/command", { name });
      if (!response.ok) {
        append(line(response.status === 409 ? "warn" : "error", await errorText(response)));
        return;
      }
      const data = await response.json();
      if (name === "reset") {
        captured = [];
        await redraw("reset", true);
      }
      for (const notice of Array.isArray(data.notices) ? data.notices : []) {
        append(line(notice.level, String(notice.text)));
      }
    }, false);
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
  for (const button of document.querySelectorAll("[data-command]")) {
    button.addEventListener("click", () => runCommand(button.dataset.command));
  }
  $("sync").addEventListener("click", runSync);

  (async () => {
    try {
      applyState(await fetchState(), true);
    } catch {
      empty.hidden = true;
      networkError();
    }
    if (!input.disabled) input.focus({ preventScroll: true });
  })();
})();
"""


def _script_json(strings: Mapping[str, str]) -> str:
    """Encode the strings as JSON that is safe inside a ``<script>`` element.

    Args:
        strings: Catalog key -> localised text.

    Returns:
        str: JSON with ``<``, ``>``, ``&``, U+2028 and U+2029 written as ``\\uXXXX`` escapes (still the same JSON).
    """
    return json.dumps(dict(strings), ensure_ascii=False, sort_keys=True).translate(_SCRIPT_ESCAPES)


def render_page(*, language: str, strings: Mapping[str, str], nonce: str) -> str:
    """Render the chat page.

    Args:
        language: Language code for ``<html lang>`` (``pt-BR`` or ``en``).
        strings: The ``web.ui.*`` catalog texts in that language (a missing key shows the key itself).
        nonce: The request's CSP nonce, put on the inline ``<style>`` and ``<script>`` elements.

    Returns:
        str: One self-contained HTML document.
    """
    labels = {name: html.escape(strings.get(key, key)) for name, key in _LABELS.items()}
    nonce_attr = html.escape(nonce)
    return "".join(
        (
            _HEAD.format(lang=html.escape(language), title=labels["title"], nonce=nonce_attr),
            _CSS,
            _BODY.format(**labels),
            f'<script type="application/json" id="i18n" nonce="{nonce_attr}">{_script_json(strings)}</script>\n',
            f'<script nonce="{nonce_attr}">{_SCRIPT}</script>\n</body>\n</html>\n',
        )
    )
