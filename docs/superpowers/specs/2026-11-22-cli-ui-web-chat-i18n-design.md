# Design: Rich CLI, web chat server, and pt-BR / en internationalisation

Date: 2026-11-22 · Branch: `feat/finance-assistant` · Status: draft for approval

## 1. Intent

`finassist` is a console chat with an LLM financial assistant over the user's Open Finance (Pluggy) data, cached in
local DuckDB. Three changes were requested together:

1. A colourful, elegant Typer CLI.
2. A `server` command that serves the same chat as a Flask app on the Waitress web server.
3. The whole application usable in **pt-BR** and **en**, one language per process.

The user gave no business reason beyond these asks, so this document does not invent one. The implied purpose is a
nicer daily-use CLI, a second front end for the same assistant, and a tool usable by English speakers.

## 2. Goals, non-goals, anti-goals

**Goals**

- G1. `LLMOPENFINANCE_LANG` selects the language. Unset means `DEFAULT_LANGUAGE`, a new constant in `constants.py`
  whose value is `"pt-BR"`. `pt` and `pt-BR` select Portuguese. `en` selects English.
- G2. In `en`, every application-generated string is English: CLI text, LLM system prompt, tool descriptions, snapshot
  labels, error and warning messages, the Pluggy Connect page, and the web UI. The assistant is told
  `ALWAYS respond in English`.
- G3. In `pt-BR`, behaviour is unchanged, including the prompt line "Responda SEMPRE em português do Brasil". Existing
  tests keep passing with **no change to any expected output string**. Test patch targets and imports may change
  only where moving code makes that unavoidable (see 5.3 for the seams that are kept so that it is rarely needed).
- G4. The CLI is restyled with Rich: banner, themed status lines, tables, grouped help, spinner during tool calls,
  and the assistant's answer rendered as Markdown.
- G5. `finassist server` serves a chat page with the same chat functionality as the CLI chat: ask questions with
  streamed answers and visible tool calls, plus the `help`, `summary`, `sync`, `reset` and `history` commands.
- G6. All of it stays inside the project's rules: `uv` only, ruff and pyrefly clean, tests offline, coverage at
  or above 90%.

**Non-goals**

- N1. Translating stored data: Pluggy values, transaction descriptions, merchant names, and the local category
  labels written at sync time (`Alimentação` and the like). The database must not depend on the language.
- N2. Translating developer-facing log messages, docstrings, command names (`ver`, `connect`, `items`), or the docs.
- N3. Per-user or per-request language switching, or runtime language switching inside one process.
- N4. Web equivalents of `connect`, `items add|remove` and `status`. They stay CLI-only. The web page shows a
  read-only status line (provider, model, language, last sync).
- N5. Multi-user web access, accounts, or HTTPS.
- N6. Languages beyond pt-BR and en. The catalog design must make a third language an additive change.

**Anti-goals** (failure even if it technically works)

- A1. Colour or Live rendering corrupting piped output. `finassist ask "..." > out.txt` must yield clean text with
  no ANSI codes.
- A2. The web server reachable by another page in the user's browser or by another machine without the secret
  token.
- A3. Any language-dependent value stored in DuckDB.
- A4. Model-authored arithmetic. All figures still come from SQL (AGENTS.md principle 1).
- A5. A mixed-language session: English chrome around a Portuguese prompt, or the reverse.

## 3. Constraints

- AGENTS.md: `uv` only. Non-stop operating model: a failing request is logged and returned as a sentinel or error
  event, never a crash. DuckDB allows one read-write handle or many read-only handles per file. Only the read-only,
  sandboxed connection ever reaches the LLM. Raw transactions are not logged.
- `ruff.toml` and `pyproject.toml` are authoritative for style (Google docstrings, lazy `%` logging) and pytest
  (`--cov-fail-under=90`, strict markers).
- `[tool.setuptools] packages = ["llmopenfinance"]` lists only the top-level package. New code is therefore **flat
  modules**, and the web page is an inline string (as `connect.py` already does), so no package-data is needed.
- Windows 11 is the primary platform. Gunicorn is out. Waitress is pure-Python WSGI and runs there.
- `typer.testing.CliRunner` tests assert `result.output`. Rich output to a non-TTY must be plain text.
- Typer `help=` strings and docstrings are bound at import time, before any callback runs. The language must be
  known at import.

## 4. Approaches considered

### 4.1 CLI look

- **Picked: Rich through a new `ui.py`.** Themed `Console`, helper functions, tables, Markdown answers, spinner.
  Rich is already installed as a Typer dependency, so it only needs listing explicitly in `pyproject.toml`.
- Rejected: plain `typer.secho` colours. Cheap, but gives no Markdown, no tables, no spinner.
- Note: the reference `apexmusic/__main__.py` has the same structure the project already follows (state dataclass,
  `@app.callback`, `_handle_*` functions, `ver` command). It enables `rich_markup_mode="rich"` but prints only with
  `typer.echo`, so it supplies no look to copy. The visual design here is original.

### 4.2 Translation mechanism

- **Picked: a hand-rolled catalog.** `i18n.py` exposes `t(key, **vars)`. The messages live in `messages_pt.py` and
  `messages_en.py`, plain `dict[str, str]` using `str.format` placeholders.
- Rejected: gettext/Babel `.po` files. They need a compile step, package-data and extraction tooling for roughly
  150 strings in two languages, which conflicts with AGENTS.md "Simplicity".
- Guards that make a hand-rolled catalog safe: a parity test (identical keys and identical placeholders in both
  catalogs), and a source scan test (every literal key passed to `t()` exists in the catalog).

### 4.3 Web chat

- **Picked:** Flask built by `create_app()` and served by Waitress. The chat logic is extracted from `__main__.py`
  into a shared `session.py`, so the CLI and the web render from one core. Answers stream to the browser. There is
  **one conversation per server process**, serialised by a lock.
- Rejected: copying the CLI logic into the web module (two code paths to keep in sync). Also rejected: one
  conversation per browser tab, because it needs per-tab LLM chat objects, cursors per thread, and a way to quiesce
  all tabs before `/sync` closes the read-only handle. That is more code than a single-user local tool warrants.
- Rejected: Uvicorn. Flask is WSGI, and the user chose Waitress.

## 5. Design

### 5.1 Language handling (`constants.py`, `i18n.py`, `messages_*.py`)

- `constants.py` adds `DEFAULT_LANGUAGE = "pt-BR"`, `SUPPORTED_LANGUAGES = ("pt-BR", "en")` and
  `ENV_LANG = "LLMOPENFINANCE_LANG"`.
- Resolution happens **once per process**, on the first `t()` or `get_language()` call, and is then frozen:
  1. Read `os.environ[ENV_LANG]`. If unset or empty, use `DEFAULT_LANGUAGE`.
  2. Normalise case-insensitively, treating `_` as `-`. A primary subtag of `pt` (`pt`, `pt-br`, `pt_BR`) gives
     `pt-BR`. A primary subtag of `en` (`en`, `en-US`, `en-GB`) gives `en`.
  3. Any other value raises `ValueError` listing the supported values. This is a configuration error, raised
     early like an unknown provider. The CLI shows it as a red message and exits non-zero.
- Because Typer help text is evaluated at import, the language is resolved when `__main__` is imported. There is no
  `--lang` option: a flag is parsed after import, so help text could not follow it, which would break
  "one language per process". The environment variable is the only switch, as requested.
- `t(key, **vars)` returns `catalog[language][key].format(**vars)`. A missing key falls back to the pt-BR entry and
  logs a warning once, never raises. A missing placeholder returns the unformatted template and logs a warning.
- Tests need a reset hook to switch the frozen language (a pytest fixture using it is the only consumer).
- Date and number helpers become language-aware:
  - `brl()`: pt-BR `R$ 1.234,56`, en `R$ 1,234.56` (the currency stays BRL).
  - Dates shown to people: pt-BR `DD/MM/AAAA`, en `YYYY-MM-DD` (unambiguous for English readers).
- **What gets localised.** The rule is: application-generated text is localised. Stored or third-party data is
  shown verbatim. Concretely:

  | Area | Source | Notes |
  |---|---|---|
  | System prompt | `agent.py` | Two full prompts. pt-BR is the current text, byte-for-byte. en is a faithful English translation that opens with `ALWAYS respond in English`. |
  | Tool descriptions, parameter descriptions, `SQL_SCHEMA_DOC` | `tools.py` | Descriptions are localised; tool names, parameter names and SQL identifiers are not. |
  | Tool errors and notes | `tools.py`, `analytics.py` | Missing-argument, invalid-argument, "already called" note, SQL guard errors. |
  | Snapshot | `analytics.py` `build_snapshot` and helpers | Headings, labels, `—` placeholders. |
  | Computed enums | `analytics.py` | `_frequency` and `_confidence` return canonical internal codes. Display labels are mapped through the catalog at the output edge (tool results and snapshot). The internal logic does not change. |
  | LLM and sync messages | `chat.py`, `sync.py`, `agent.py` | `FINAL_ANSWER_NOTE`, communication errors, stop-reason messages, sync warnings, unknown-provider error. |
  | CLI | `__main__.py` | Every user-visible string, Typer `help=` strings and command docstrings. |
  | Connect page | `connect.py` | Page text, `<html lang>`, and the Pluggy Connect widget `language` option (`pt` or `en`). |
  | Web UI | `web.py` | Page text delivered to the browser from the catalog. |
  | Chat commands | `__main__.py`, `session.py` | The input aliases of **both** languages stay accepted (`/ajuda` and `/help`, `sair` and `exit`). Only the displayed help names the current language's forms. |

- **Category labels** (`Alimentação`, ...) are stored data. The English prompt states that stored category labels
  are in Brazilian Portuguese, that the model may filter with them or with Pluggy's English names (as the tool
  already accepts), and that it must translate labels naturally when answering.
- Catalog guard: the `en` catalog contains no Portuguese accented characters, with an explicit allow-list for
  deliberate exceptions such as quoted Portuguese labels.

### 5.2 Rich UI (`ui.py`, `__main__.py`)

- `ui.py` owns one `rich.console.Console` with a theme (names, not literal colours, are used by callers): `ok`
  (green), `warn` (yellow), `err` (red), `dim`, `accent`, `user`, `assistant`. Helpers: `banner()`, `ok()`,
  `warn()`, `error()`, `dim()`, `item_result()`, `table()`, `answer_panel()`, `tool_status()`.
- Colour and live updates are enabled only on an interactive terminal, and respect `NO_COLOR`. On a non-TTY
  (pipes, `CliRunner`) the helpers print plain text with the same wording. The `ask` command streams plain text and
  prints the final answer once, with no ANSI.
- Chat flow (TTY): banner panel (app, version, provider, model, language), then connection and privacy notices,
  then per turn a `Você ▸` / `You ▸` prompt, a spinner line `↳ consultando <tool>(…)` for each tool call, and the
  answer in a titled panel with Markdown rendered through `rich.live.Live`. On a non-TTY the answer is streamed as
  plain text.
- `status` and `items list` render as tables. Sync per-item results keep the existing `✓ / = / ✗ / !` markers, now
  coloured. The tables must keep these substrings intact at 80 columns, because `CliRunner` renders at that width
  and the tests assert them: `transactions=`, `consentimento até 01/06/2027` and `não sincronizado`. Cells are
  therefore no-wrap (or the layout is a key-value grid), and the data counts stay in `key=value` form.
- Typer: keep `rich_markup_mode="rich"`, group commands and options with `rich_help_panel` (Chat, Data, Connections),
  add an epilog with the environment variables, set `pretty_exceptions_show_locals=False`. Keep `--version` and the
  `ver` command. No emoji; the existing symbols are enough and render on the Windows console.
- `main()` keeps its UTF-8 `reconfigure` of stdout and stderr.
- The chat prompt stays injectable (`_chat_loop(..., prompt=...)`) so tests drive it unchanged.

### 5.3 Shared session core (`session.py`)

Extracted from `__main__.py` with no behaviour change in `pt-BR`:

- `ChatSession` (read-only store, snapshot, `resync()` that closes and reopens the read-only handle).
- `build_assistant(...)`: creates the LLM, verifies the connection, applies the Claude privacy notice, builds the
  `ToolRegistry` and `FinanceAssistant`. It **returns a result object** (assistant or error, plus notices) instead of
  printing, so both front ends can render it.
- Chat commands (`help`, `summary`, `sync`, `reset`, `history`) as functions returning structured results.
- The sync entry point reports progress through callbacks (`sync_all` already takes `on_item`), so the CLI prints
  Rich lines and the web page shows progress without either duplicating logic.

`__main__.py` remains the Typer layer: state, callback, commands, rendering.

**Seams kept so the existing tests need no patch-target edits.** `tests/test_cli.py` patches or calls these names on
the `__main__` module: `_pluggy_client`, `_llm`, `_open_store`, `_render_answer`, `_ChatSession`,
`_chat_loop(..., prompt=)`, `main` and `app`. They stay defined or importable in `__main__`. The factories
(`_pluggy_client`, `_llm`, `_open_store`) stay in `__main__` and are **injected** into `session.py` (as parameters
or a small context object), so a test that patches `cli._open_store` still affects sync and chat. Any function that
looks those factories up (for example `_run_sync`) must do it through the `__main__` module globals at call time.
If one test truly cannot be kept as is, editing its patch target is allowed, but never an expected string.

### 5.4 Web chat (`web.py`, `server` command)

**Command.** `finassist server [--host 127.0.0.1] [--port 8000] [--no-browser] [--threads 4] [--sync/--no-sync]
[--think] [--sql/--no-sql]`. The last three mirror `chat` exactly (same defaults: sync on, thinking off, `run_sql`
on for Claude only). It reuses the global `--provider`, `--model`, `--db` options. Startup mirrors the CLI chat:
optional sync (read-write handle, closed before the read-only one opens), build the assistant, verify the LLM
connection, and fail early with the same localised message. It then prints the URL (with token) and opens the
browser unless `--no-browser`.

**Optional dependency.** `pyproject.toml` gets `[project.optional-dependencies] web = ["flask", "waitress"]`, and the
dev group includes them so tests run. `rich` is added to the core dependencies explicitly. The `web.py` imports are
lazy. If they are missing, `server` prints a localised message with the install command and exits 1.

**Routes** (all JSON except the page and the stream):

| Route | Purpose |
|---|---|
| `GET /` | The inline chat page. With a valid `?token=` it sets the auth cookie and redirects to `/`. |
| `GET /api/state` | Provider, model, language, last sync, counts, the localised UI strings, and the transcript so far (see below). |
| `POST /api/chat` | Body `{"message": "..."}`. Streams events: `text` (delta), `tool` (name and arguments), `error`, `done`. Slash commands typed in the box are routed to the command logic instead. |
| `POST /api/command` | `{"name": "help|summary|reset|history"}` returns JSON (they are quick). |
| `POST /api/sync` | `{"force": false}`. Streams events: `progress` (one per item, same data the CLI prints), `error`, `done` (with the final counts). Sync can take minutes, so it streams like chat does. Typing `/sync` in the box calls this route. |

**Transcript on reload.** The server holds one conversation, so a reloaded or second tab must show it. `/api/state`
returns the user and assistant text turns so far (no tool rows). They are derived from the existing chat history
that `history` already exposes, or, if history compaction drops text turns, from a small in-memory list kept beside
the assistant. It is never written to disk or to logs.

**Concurrency.** One `FinanceAssistant` and one read-only store per process, guarded by a `threading.Lock`. A request
that finds the lock held gets `409` with a localised "assistant is busy" message and does not queue. `sync` takes
the same lock, closes the read-only handle, syncs, reopens it, and refreshes the assistant, as `/sync` does in the
CLI. A failing request becomes an `error` event or a JSON error and never takes the server down.

**Security.**

- Default bind is loopback. A non-loopback `--host` prints a prominent warning that financial data is being served
  over plain HTTP.
- A per-launch secret token (`secrets.token_urlsafe`, the same idea as the nonce URL in `connect.py`) is required on
  every route. It travels once in the URL and then in an `HttpOnly`, `SameSite=Strict` cookie.
- `Host` header check (DNS-rebinding defence). Accepted: `localhost`, `127.0.0.1` and `[::1]` with the bound port,
  plus the exact `--host` value with the port when it is a specific address or name. For a wildcard bind
  (`0.0.0.0` or `::`) the client's `Host` cannot be predicted, so the name check is skipped and the token alone
  protects access; the non-loopback warning says so. LAN use (for example from a phone) is therefore possible but
  deliberate. `POST` routes require `Content-Type: application/json`.
- The page loads **no external scripts, styles or fonts**. All model and data text is HTML-escaped before display,
  because transaction descriptions are third-party input. Responses carry `Cache-Control: no-store` and a restrictive
  `Content-Security-Policy`.
- The token and API keys never appear in logs.

**Streaming risk.** Answers can be long, so the design streams them over a chunked `POST` response. Waitress is
expected to flush each chunk as it is yielded, but that is **verified first, before the rest of the web work**:
a throwaway script served by real Waitress, read with `curl --no-buffer`. Fallback if chunks are buffered: return the
complete answer in one response, still listing tool calls, and record the limitation in the docs.

**Page.** One inline HTML/CSS/JS module-level string (same pattern as `connect.py`): message list, input box,
buttons for the commands, a status line, a "busy" indicator, light and dark via `prefers-color-scheme`, usable on a
narrow viewport. A minimal Markdown renderer (headings, bold, italics, code, lists, tables) runs after escaping.

### 5.5 Packaging, docs, scripts

- New flat modules: `i18n.py`, `messages_pt.py`, `messages_en.py`, `ui.py`, `session.py`, `web.py`.
- `pyproject.toml`: `rich` in dependencies, `web` extra, dev group updated, `uv.lock` committed with the change.
- Docs: `README.md`, `docs/usage.md` (the new env var and the `server` command and options), `docs/architecture.md`
  (new modules), `docs/development.md` (the manual Waitress streaming smoke test).
- `Makefile`, `run.bat` and `build.bat` are untracked and out of scope.

### 5.6 Tests

- Existing tests keep passing unmodified in the default language.
- New: catalog parity and literal-key scan; language resolution (unset, `pt`, `pt_BR`, `en-US`, invalid);
  English prompt contains `ALWAYS respond in English` and pt-BR prompt is unchanged; `brl()` and date formats per
  language; English CLI runs of `status`, `sync`, `items`, `ask` and `chat` with no Portuguese chrome; ANSI-free
  output on non-TTY; `ui.py` helpers.
- Web: Flask test client for token, cookie, Host check, content-type check, busy `409`, streamed events, command
  routes, sync lock behaviour; the Waitress `serve` call injected so no socket is opened.
- All offline, using the synthetic dataset and fake clients already in `tests/conftest.py`.

## 6. Delivery

One design, three shippable milestones, each ending with the full check green (`ruff format`, `ruff check --fix`,
`pyrefly check`, `pytest -x`):

1. **i18n foundation.** Constants, `i18n.py`, catalogs, localise every module, English prompt, tests.
2. **Rich CLI.** Begin with the pure `session.py` extraction (no behaviour change), then `ui.py` and the restyle on
   top of the catalog.
3. **Web chat.** Waitress streaming check first, then `web.py`, the `server` command, the extra, docs.

## 7. Calls made by the author (not yet agreed)

- C1. No `--lang` option; the environment variable is the only switch (reason in 5.1).
- C2. Web scope is chat and its commands only. `connect`, `items` and `status` management stay CLI-only (N4).
- C3. `en` formats: `R$ 1,234.56` and `YYYY-MM-DD`.
- C4. An unsupported `LLMOPENFINANCE_LANG` is a hard error, not a silent fallback.
- C5. Both languages' chat command aliases are accepted in either language.
- C6. Flask and Waitress are an optional `web` extra; `rich` is promoted to an explicit dependency.
- C7. One conversation per server process, busy requests rejected with `409`, not queued.
- C8. Default port 8000, default 4 Waitress threads.
- C9. Categories and other stored data are never translated; the English prompt asks the model to translate labels
  when answering.
- C10. G3 forbids changes to expected strings only; `__main__` keeps the test seams and injects factories into
  `session.py`; tables are built so the asserted substrings survive at 80 columns (5.2, 5.3).
- C11. Wildcard binds skip the `Host` name check and rely on the token; specific non-loopback hosts are accepted by
  exact match (5.4).
- C12. Web `sync` streams per-item progress through its own route and takes the same lock as chat (5.4).
- C13. `server` mirrors `chat`'s `--sync/--no-sync`, `--think` and `--sql/--no-sql` with the same defaults (5.4).
- C14. A reloaded or second tab shows the current conversation's text turns, held only in memory (5.4).

## 8. Left to the builder

Exact colours, glyphs and banner artwork; message key naming and grouping; the page's visual styling; the minimal
Markdown renderer's exact feature set; whether Rich `Live` or `Status` suits each streaming moment better;
function signatures inside `session.py`; thread-safety details of DuckDB cursors inside the lock; the exact CSP
string.
