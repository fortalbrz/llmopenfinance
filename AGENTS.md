# AGENTS.md

Guidelines for any agent, human or AI, working on this codebase.

> **Operating model.** Treat sync and chat as a **non-stop process**. One failing Pluggy item, product, tool call
> or LLM request must **never** abort the run. Recoverable failures are **logged and returned as a sentinel or
> error result**, not raised.

The configs own the tool-enforced conventions, so do not restate them here:

- [`ruff.toml`](ruff.toml) owns style, naming, Google docstrings, lazy `%`-logging and the security rules.
- [`pyproject.toml`](pyproject.toml) owns pytest, the 90% coverage gate and pyrefly.

## 1. Principles

| Principle | Meaning here |
|---|---|
| Correct numbers | Every figure the assistant states must come from SQL (snapshot or tool), never from the model. Add a tool rather than trusting the model's arithmetic. |
| Resilience | Two tiers. Configuration errors (missing credentials, unknown provider) are raised early. Everything else is logged and handled with a sentinel. |
| Privacy | Keep raw transactions local. Send only aggregates and capped tool rows to remote LLMs. Never log credentials or API keys. |
| Safety | The LLM only ever gets the **read-only, sandboxed** DuckDB connection. Tools have no network or filesystem access. |
| Simplicity | Few files, plain dicts at API boundaries, no frameworks the task doesn't need. |

## 2. Package management

Use **`uv` only**: `uv add pkg`, `uv add --dev pkg`, `uv run tool`. Commit `uv.lock`. Never use bare `pip`.

## 3. Error handling

- **Fatal (raise):** contract or configuration violations detected before work starts.
- **Recoverable (log + sentinel):**
  - one item failing in `sync_item` produces an `ItemResult(ok=False)`;
  - an unavailable product becomes a warning;
  - a tool failure produces a `ToolResult(is_error=True)`;
  - an LLM failure produces an `AssistantError` plus a history rollback.
- One logger per module (`logging.getLogger(__name__)`). Call `logging.basicConfig` only in `__main__._configure`.

## 4. Concurrency

- Functions are pure where possible. Results are frozen dataclasses (`ItemResult`, `SyncReport`, `ToolCall`, `ToolResult`).
- `PluggyClient` is thread-safe: its API-key cache sits behind a lock.
- DuckDB allows **one read-write or many read-only** handles per file. Sync opens, writes and closes; chat then
  reopens read-only. Never hold both in one process.

## 5. External APIs (verified facts; re-verify before changing)

**Pluggy**

- Use `/v2/transactions` (cursor). The legacy `/transactions` returns 410 for new apps.
- Card purchases are `type=DEBIT` with a positive `amount`.
- There is no list-items endpoint for MeuPluggy accounts.
- The API key lasts 2 h.

**Claude**

- `system` goes in the top-level parameter, never as a `messages[0]` system role.
- Send no `temperature` (Claude 5.5 models reject non-default sampling parameters).
- Keep the history append-only.

**Ollama**

- Set `think` and `num_ctx` explicitly.
- Tool results are sent as `role: "tool"` with `tool_name`.

## 6. Testing

- Run `uv run pytest`. Coverage must stay at or above 90%.
- Tests never hit the network:
  - use `httpx.MockTransport` for Pluggy and Claude;
  - use fake clients for Ollama;
  - use the synthetic dataset in `tests/conftest.py`.
- Every bug fix needs a regression test, for example the "aggregates only" rule in the system prompt.
- Manual live smoke tests (synthetic DB, real LLM) are described in `docs/development.md`.

## 7. Git workflow

- Never commit directly to `main`. Branch as `feat/`, `fix/` or `chore/`. Use conventional commits.
- CI failure fix order: `uv run ruff format .` → `uv run ruff check . --fix` → `uv run pyrefly check` → `uv run pytest -x`.
