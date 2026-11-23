# Architecture

```text
src/llmopenfinance/
├── __main__.py    Typer CLI. _state callback, _handle_* logic, thin commands (pattern from apexmusicmachine); prints only through ui.py
├── ui.py          Rich console helpers: themed notices, banner, tables, live Markdown answer; plain text off a terminal
├── session.py     Chat/sync core shared by every front end, with no printing: ChatSession, build_assistant, chat commands, run_sync
├── web.py         Optional local web chat (Flask app + Waitress serve): Host check, token cookie, JSON/NDJSON routes, one lock
├── web_stream.py  NDJSON streaming helpers: lock held for the whole stream, released exactly once, on disconnect too
├── web_page.py    The chat page: one inline HTML/CSS/JS string, nonce'd for the Content-Security-Policy
├── i18n.py        Process language (LLMOPENFINANCE_LANG), t(key, **vars), language-aware money and dates
├── messages_pt.py pt-BR message catalog (default; fallback for a key missing elsewhere)
├── messages_en.py English message catalog (same keys and placeholders as pt-BR)
├── constants.py   Defaults: models, URLs, limits, environment variable names
├── pluggy.py      Thin httpx client: API-key cache, 429/5xx retries, page/cursor pagination
├── connect.py     Pluggy Connect widget on a loopback page; polls items until they settle
├── store.py       DuckDB schema, pure row mappers, bulk UNNEST inserts, sandboxed read-only connection
├── categorize.py  Local pt-BR taxonomy, keyword rules, sign and kind classification
├── sync.py        Resilient per-item sync with an immutable SyncReport
├── analytics.py   Deterministic SQL analytics, recurring-charge detection, guarded SELECT, snapshot
├── tools.py       Provider-agnostic tool registry: validation, memoization, output caps
├── chat.py        AbstractChat, OllamaChat, ClaudeChat (raw HTTP + SSE) with tool calling
└── agent.py       System prompt (pt-BR or en, by language) and the tool loop (FinanceAssistant)
```

## Data flow

1. **Sync**:
   - `PluggyClient` calls `GET /items/{id}`, then `/accounts`, `/v2/transactions` (full history),
     `/bills`, `/investments` and `/loans`.
   - The `*_row` mappers turn the payloads into rows, and `FinanceStore.replace_*` writes them to DuckDB
     (opened read-write).
2. **Chat**:
   - DuckDB is reopened **read-only** with external access disabled and its configuration locked.
   - `build_snapshot` turns the aggregates into roughly 1–2k tokens of text (pt-BR or en), appended to the system prompt.
   - `FinanceAssistant.ask` runs a loop:
     1. stream a model turn;
     2. if the model requests tools, `ToolRegistry.run` executes them and feeds the results back;
     3. repeat. After `MAX_TOOL_ROUNDS` (6) rounds the model is forced to answer.

## Terminal front end and language

- **One language per process.** `i18n` resolves `LLMOPENFINANCE_LANG` once, on first use, and freezes it. Every
  application-generated text is a catalog key (`t("area.name", **vars)`) in `messages_pt` and `messages_en`; guard
  tests require the same keys and placeholders in both catalogs and every literal key used in the source to exist.
  Stored data (Pluggy values, category labels, descriptions) is never translated.
- **`session` renders nothing.** It returns `Notice(level, text)` values and calls per-item callbacks, so the CLI (and
  later other front ends) decide how to show them. The factories for the LLM, the store and the Pluggy client are
  injected, which keeps `__main__._llm`, `_open_store` and `_pluggy_client` the patch points of the tests.
- **`ui` is the only place that decides how output looks, and it decides per call.** Styled output (colours, panels,
  tables, the live Markdown answer) needs an interactive terminal and an unset `NO_COLOR`. Everywhere else (pipes,
  `CliRunner`) it prints the exact plain text the CLI always printed, written as raw segments so no byte changes.
  Dynamic text is wrapped in `rich.text.Text` (never parsed as markup) and stripped of terminal escape sequences.
- **Tables exist only on a terminal.** Scripts keep the one-line-per-item format. A table never crops a cell: the
  ID column folds, the others stay on one line, and `ui` falls back to plain lines when they would not fit.
- **Typer help** is built at import (`help=`, panel titles and epilog come from the catalog), which is why the language
  must be known before the app is created and why there is no `--lang` option.

## Web front end

`finassist server` (the `web` extra: Flask and Waitress) is a second front end over the same `session` core, so it adds
no chat or sync logic of its own.

- **Optional and lazy.** `__main__` imports `web` only inside `server`, after the language guard; when Flask or
  Waitress is missing, `server` prints the install line and returns 1 before any sync or LLM check.
  `web.serve` imports Waitress lazily too, and `llmopenfinance.web` is the only module that imports Flask.
- **One assistant, one read-only store, one lock.** `WebContext` bundles them with the token, host, port and a
  `threading.Lock`. Everything that touches the session, the assistant or DuckDB runs while holding the lock; a request
  that finds it held gets 409 and is never queued. `GET /api/state` serves cached status only, so it answers while busy
  (which is why `--threads` is at least 2).
- **Streaming as NDJSON.** `POST /api/chat` and `POST /api/sync` answer with one JSON object per line
  (`text`, `tool`, `notice`, `progress`, `error`, always ending in `done`). `web_stream` holds the lock for the whole
  stream and gives it back once, when the stream ends, fails, is aborted by a client that disconnects or is closed without
  being read. A sync cannot be cancelled, so a disconnected sync keeps the lock until it has finished and the read-only
  store is reopened; the history of an aborted answer is rolled back by the assistant.
- **Request guards, in order:** `Host` header (400), token cookie compared in constant time (401), JSON content type
  (415) and a 64 KiB body limit (413). Waitress is also capped just above that body limit, because it buffers a whole
  body before the app can answer. The token is printed once by `server`, never logged; access logging is off.
- **The page** (`web_page`) is a single inline string with no external resources. It takes its texts from the `web.ui.*`
  catalog keys (served in `/api/state`), escapes all model and data text before display, and re-reads `/api/state` after a
  sync to redraw the conversation.
- **The conversation is in memory only.** `WebContext.transcript` keeps the text turns so a reload or a second tab can show
  them; it is never written to disk or to logs.

## Key decisions

| Decision | Why |
|---|---|
| Hand-written Pluggy client instead of `pluggy-sdk` | Pluggy has no official Python SDK. The PyPI package is a community OpenAPI build with no `PluggyClient`, strict enums that break when Pluggy adds values, and no key refresh. We need about 10 endpoints. |
| DuckDB, no vector DB | The questions are analytical (sums, months, recurrence) over a few thousand structured rows. SQL is exact; embeddings would add nothing. |
| Full history re-pull on every changed item | Pluggy re-creates card installments and pending rows with new ids, and date-filtered queries can miss old installments. A full replace is correct and cheap at personal volume. |
| Midnight-UTC dates kept as calendar dates | Most connectors send dates as `T00:00:00Z`. Converting those to UTC-3 would move them to the previous day or month. Only real timestamps are converted. |
| `type` decides the sign | Pluggy normalizes `type`: purchases are always `DEBIT`, even on cards whose raw amounts are positive. `flow = -abs(amount)` for DEBIT. |
| Card bill payments and own-account transfers are kinds of `transfer` | Otherwise the bank-side bill payment and the card purchases would both count as spending. |
| Local categorization | Pluggy's categorization is paid after the trial. Local categories keep monthly aggregates consistent over time. |
| Snapshot + tools + optional `run_sql` | The snapshot answers broad questions in zero tool rounds. Curated tools give exact, testable numbers. `run_sql` covers the long tail (on by default only for Claude). |
| Native transcript per provider | Claude needs its content blocks (thinking signatures) replayed byte for byte; Ollama uses `tool_calls`/`tool_name`. Only events and tool specs are normalized. |
| Claude: top-level `system`, no `temperature`, `tool_choice: none` for the last round | A `{"role": "system"}` message at `messages[0]` is rejected, and Claude 5.5 rejects non-default sampling parameters. Keeping the tool list unchanged preserves the prompt-cache prefix. |
| Ollama: explicit `think=False`, `num_ctx=16384`, JSON-text tool-call fallback | qwen3.5 thinks by default, and Ollama's default context is 4k on GPUs under 24 GB. Small models sometimes write tool calls as text. |
| `/sync` resets the conversation | Changing the system prompt mid-conversation would invalidate Claude's cache and its preserved thinking. |

## Safety of `run_sql`

The tool has three layers of protection:

1. **Read-only connection**: `enable_external_access=false` and `lock_configuration=true` together block
   writes, file reads, `COPY` and `ATTACH`.
2. **Statement checks**: `duckdb.extract_statements` must return exactly one statement, and it must be a `SELECT`.
3. **Limits**: the query is wrapped in `LIMIT n+1` to detect truncation, and a `threading.Timer` calls
   `con.interrupt()` after 5 s.

Transaction descriptions are third-party text, so the system prompt tells the model to treat them as data, not
as instructions.

## Error handling

The project follows the non-stop batch policy from AGENTS.md:

- **Fatal, raised before work starts**: configuration errors such as missing credentials or an unknown provider.
- **Recoverable**: logged and returned as a sentinel or an error result. Examples:
  - a single Pluggy item failing during sync;
  - an optional product that is unavailable (for example, loans on MeuPluggy);
  - a tool error, which goes back to the model as `is_error` so it can correct itself;
  - an LLM or HTTP failure, which becomes an `AssistantError` and rolls the history back.
- **The chat loop never dies** on an exception. It logs the error and continues.

## Testing

| File | Covers |
|---|---|
| `tests/conftest.py` | Synthetic Pluggy dataset (11 months, two accounts, subscriptions, refunds, installments, card payments, investments, loans) and `FakePluggy` |
| `test_pluggy.py` | `httpx.MockTransport`: auth caching and refresh, 401 re-auth, 429 `RateLimit-Reset`, cursor pagination |
| `test_store.py`, `test_categorize.py` | Mappers, date and sign rules, sandbox enforcement |
| `test_analytics.py`, `test_tools.py` | Exact aggregates, subscription detection, SQL guard and timeout, argument validation |
| `test_chat.py` | Ollama with a fake client; Claude with recorded SSE (thinking, tool use, `max_tokens`, error events, retries) |
| `test_agent.py`, `test_sync.py`, `test_connect.py`, `test_cli.py` | Tool loop and rollback, resilient sync, loopback server, CLI end to end (plain output off a terminal; tables, banner and styled prompt on a forced terminal) |
| `test_i18n.py`, `test_catalog.py` | Language resolution, `t()` fallbacks, date and money formats; catalog parity, placeholders and source-scan guards |
| `test_web.py` | The Flask app with a test client: Host, token and content-type guards, state, commands, streamed chat and sync, lock release on disconnect, `web.serve` limits |
| `test_session.py`, `test_ui.py` | Notices and commands without a terminal; `ui` in plain, styled and `NO_COLOR` modes, verbatim dynamic text, escape stripping, tables at 80 and 120 columns |
