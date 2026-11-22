# Architecture

```text
src/llmopenfinance/
├── __main__.py    Typer CLI. _state callback, _handle_* logic, thin commands (pattern from apexmusicmachine)
├── constants.py   Defaults: models, URLs, limits, environment variable names
├── pluggy.py      Thin httpx client: API-key cache, 429/5xx retries, page/cursor pagination
├── connect.py     Pluggy Connect widget on a loopback page; polls items until they settle
├── store.py       DuckDB schema, pure row mappers, bulk UNNEST inserts, sandboxed read-only connection
├── categorize.py  Local pt-BR taxonomy, keyword rules, sign and kind classification
├── sync.py        Resilient per-item sync with an immutable SyncReport
├── analytics.py   Deterministic SQL analytics, recurring-charge detection, guarded SELECT, snapshot
├── tools.py       Provider-agnostic tool registry: validation, memoization, output caps
├── chat.py        AbstractChat, OllamaChat, ClaudeChat (raw HTTP + SSE) with tool calling
└── agent.py       pt-BR system prompt and the tool loop (FinanceAssistant)
```

## Data flow

1. **Sync**:
   - `PluggyClient` calls `GET /items/{id}`, then `/accounts`, `/v2/transactions` (full history),
     `/bills`, `/investments` and `/loans`.
   - The `*_row` mappers turn the payloads into rows, and `FinanceStore.replace_*` writes them to DuckDB
     (opened read-write).
2. **Chat**:
   - DuckDB is reopened **read-only** with external access disabled and its configuration locked.
   - `build_snapshot` turns the aggregates into roughly 1–2k tokens of pt-BR text, appended to the system prompt.
   - `FinanceAssistant.ask` runs a loop:
     1. stream a model turn;
     2. if the model requests tools, `ToolRegistry.run` executes them and feeds the results back;
     3. repeat. After `MAX_TOOL_ROUNDS` (6) rounds the model is forced to answer.

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
| `test_agent.py`, `test_sync.py`, `test_connect.py`, `test_cli.py` | Tool loop and rollback, resilient sync, loopback server, CLI end to end |
