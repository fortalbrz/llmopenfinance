# Rich CLI, Web Chat and pt-BR / en i18n Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `finassist` run in pt-BR or en (one language per process), restyle its CLI with Rich, and add a `server` command that serves the same chat as a Flask app on Waitress.

**Architecture:** A hand-rolled message catalog (`i18n.py` + two message modules) feeds every user-visible and LLM-facing string. The chat logic moves out of `__main__.py` into `session.py` so the Rich CLI (`ui.py`) and the web app (`web.py`) render from one core. The web app holds one conversation per process behind a lock and streams NDJSON.

**Tech Stack:** Python 3.12, Typer + Rich, DuckDB, Flask + Waitress (optional `web` extra), pytest, ruff, pyrefly, `uv` only.

**Spec:** `docs/superpowers/specs/2026-11-22-cli-ui-web-chat-i18n-design.md` (read it first; section numbers below refer to it).

## Global Constraints

- Baseline before this plan: 270 tests pass, coverage 98.7%. Coverage must stay at or above 90%.
- **CHECK** (run at the end of every task, from the repo root): `uv run ruff format . ; uv run ruff check . --fix ; uv run pyrefly check ; uv run pytest -x -m "not live"`. Partial runs use `uv run pytest <path> -q --no-cov`.
- `uv` only. Commit `uv.lock`. Branch is `feat/finance-assistant`; conventional commits; each commit message ends with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Commit steps run only once the user has OK'd committing.
- ruff is strict (`ruff.toml`): Google docstrings on every function (D, DOC), booleans keyword-only (FBT), at most 5 parameters (PLR0913; use a frozen dataclass beyond that), McCabe complexity, no `print` (T20), `# ruff: ignore[rule-name] — reason` for justified exceptions, lazy `%` logging.
- `DEFAULT_LANGUAGE = "pt-BR"`; env var `LLMOPENFINANCE_LANG`; `pt*` means pt-BR, `en*` means en, anything else is a `ValueError`. There is no `--lang` option.
- pt-BR output is unchanged: **no existing expected string in any test may change**. The pt-BR system prompt keeps "Responda SEMPRE em português do Brasil"; the en prompt contains `ALWAYS respond in English`.
- Formats: pt-BR `R$ 1.234,56` and `DD/MM/AAAA`; en `R$ 1,234.56` and `YYYY-MM-DD`.
- Stored data (category labels such as `Alimentação`, Pluggy values, descriptions) is never translated. Log messages, docstrings, command names and library exception text stay as they are.
- New code is **flat modules** under `src/llmopenfinance/` (`[tool.setuptools] packages` lists only the top package). No package-data.
- Rich output is plain text with no ANSI on a non-TTY and under `NO_COLOR`. Dynamic text is never parsed as Rich markup.
- Web: default `--host 127.0.0.1`, `--port 8000`, `--threads 4`; per-launch secret token; no external scripts, styles or fonts; all model and data text HTML-escaped.
- Never log credentials, API keys or the web token.

## Review Focus

Failure modes the spec implies but a straight reading would not test. Each has a pinning test in the task named.

1. **Odd `LLMOPENFINANCE_LANG` values** (`" EN "`, `en_US`, empty string, `fr`): normalised or rejected with a clean stderr message and non-zero exit, never a traceback at import. Tasks 1 and 7.
2. **Square brackets and emoji codes in dynamic text** (`[erro]`, `[USER]`, a merchant called `[PROMO] :tada:`) print verbatim and never raise Rich's `MarkupError`. Task 9.
3. **A client that disconnects mid-stream** (or an exception inside the stream) must release the lock and roll back the history; otherwise the server stays busy forever. Task 13.
4. **A failed web sync** (no credentials, DB locked by another process, Pluggy error) leaves the read-only store reopened and chat still working. Task 14.
5. **Port already in use** on `server`: a localised one-line error and exit 1, not a traceback. Task 16.

---

## Milestone 1: i18n

### Task 1: Language core

**Files:**
- Modify: `src/llmopenfinance/constants.py`, `tests/conftest.py`
- Create: `src/llmopenfinance/i18n.py`, `src/llmopenfinance/messages_pt.py`, `src/llmopenfinance/messages_en.py`, `tests/test_i18n.py`

**Interfaces:**
- Produces (`constants.py`): `DEFAULT_LANGUAGE = "pt-BR"`, `SUPPORTED_LANGUAGES = ("pt-BR", "en")`, `ENV_LANG = "LLMOPENFINANCE_LANG"`.
- Produces (`i18n.py`):
  - `normalize_language(value: str) -> str` (strips whitespace, case-insensitive, `_` equals `-`, primary subtag `pt` gives `"pt-BR"`, `en` gives `"en"`, else `ValueError` naming the supported values).
  - `get_language() -> str` (resolves once from the environment, empty or unset means default, then frozen).
  - `t(key: str, **values: object) -> str` (catalog lookup plus `str.format`; a key missing in the active catalog falls back to pt-BR and logs one warning per key; missing everywhere returns the key; a missing placeholder returns the unformatted template with one warning).
  - `format_date(value: date) -> str` (pt-BR `%d/%m/%Y`, en ISO).
  - `reset_language(code: str | None = None) -> None` (tests only: clears the frozen value, or freezes `code` after normalising it).
- Produces (message modules): `MESSAGES: dict[str, str]` in each; keys are dotted `area.name`, placeholders `{name}`.
- Produces (`conftest.py`): autouse fixture `_pin_language` that deletes `LLMOPENFINANCE_LANG` and calls `reset_language("pt-BR")` before and `reset_language()` after each test; fixture `use_language` returning a callable `use(code: str) -> None` that calls `reset_language(code)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_i18n.py`: `test_normalize_language` (parametrised: `"pt"`, `"PT_br"`, `" pt-BR "` give `"pt-BR"`; `"en"`, `"en-US"`, `"EN_gb"` give `"en"`); `test_normalize_rejects_unknown` (`"fr"` raises `ValueError` whose message contains `pt-BR` and `en`); `test_get_language_default_and_env` (unset and `""` give `"pt-BR"`; `"en"` gives `"en"`); `test_language_is_frozen_after_first_use` (change the env after `get_language()`, still the first value); `test_t_formats_and_falls_back` (key only in pt catalog under `en` returns the pt text and logs once, via `caplog`); `test_t_missing_placeholder_returns_template`; `test_format_date_per_language` (`date(2026, 11, 22)` gives `"22/11/2026"` and `"2026-11-22"`).
- [ ] **Step 2: Run** `uv run pytest tests/test_i18n.py -q --no-cov`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the interfaces above. The catalogs start with two sample keys each (`i18n.test.hello` with a `{name}` placeholder, and one pt-only key for the fallback test, added by the test through `monkeypatch.setitem`).
- [ ] **Step 4: Run** `uv run pytest tests/test_i18n.py -q --no-cov`. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(i18n): language resolution and message catalog core`.

### Task 2: Catalog guards and language-aware money

**Files:**
- Modify: `src/llmopenfinance/analytics.py` (`brl` only, line 52)
- Create: `tests/test_catalog.py`
- Test: `tests/test_analytics.py` (append)

**Interfaces:**
- Consumes: `get_language`, `MESSAGES` from Task 1.
- Produces: `analytics.brl(value: float | None) -> str` now follows the language (en `R$ 1,234.56`, negative `-R$ 1,234.50`, `None` gives `"—"`).

- [ ] **Step 1: Write the failing tests.** `tests/test_catalog.py`: `test_catalogs_have_same_keys`; `test_catalogs_have_same_placeholders` (compare `string.Formatter().parse` field names per key); `test_every_literal_t_key_exists` (AST-walk every `src/llmopenfinance/*.py`; for each call to `t` or `i18n.t` whose first argument is a string constant, assert the key is in both catalogs); `test_en_catalog_has_no_portuguese_accents` (no character from `áàâãéêíóôõúç` in any en value, with a module-level allow-list set `ALLOWED_PT_KEYS` that starts empty). In `test_analytics.py`: `test_brl_en` using `use_language("en")` (`brl(1234.5) == "R$ 1,234.50"`, `brl(-1234.5) == "-R$ 1,234.50"`, `brl(None) == "—"`).
- [ ] **Step 2: Run** both files with `--no-cov`. Expected: `test_brl_en` FAILS; the guard tests pass vacuously.
- [ ] **Step 3: Implement** `brl` using `get_language()` (same rounding; swap separators only for pt-BR).
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `test(i18n): catalog guards; feat: language-aware brl`.

### Task 3: Agent prompts

**Files:**
- Modify: `src/llmopenfinance/agent.py`, `messages_pt.py`, `messages_en.py`, `docs/development.md` (the `agent.SYSTEM_PROMPT` mention)
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: `t`, `format_date`.
- Produces: `agent.system_prompt() -> str` (replaces the `SYSTEM_PROMPT` constant, which nothing else uses); `agent.stop_reason_message(reason: str) -> str | None` (replaces `STOP_REASON_MESSAGES`); `build_system_prompt(snapshot: str, today: date | None = None) -> str` keeps its signature. Catalog keys: `prompt.system`, `prompt.today` (`{date}`), `agent.stop.max_tokens`, `agent.stop.refusal`.

- [ ] **Step 1: Write the failing tests:**
  - `test_system_prompt_pt_is_byte_identical`: `hashlib.sha256(system_prompt().encode()).hexdigest() == "75aac0f1c8c6815c7d3bae9038b3719ee1996699a020466909acf387e71edb53"` and `len(system_prompt()) == 1856`.
  - `test_system_prompt_en`: with `en`, the prompt contains `ALWAYS respond in English`, the seven numbered rules, the names `detect_subscriptions` and `run_sql`, and a sentence that stored category labels are in Brazilian Portuguese and must be translated when answering.
  - `test_build_system_prompt_date_line`: pt-BR ends the preamble with `Data de hoje: 22/11/2026.`; en with `Today's date: 2026-11-22.`.
  - `test_stop_reason_messages_en`: an `AssistantError` from a `max_tokens` turn reads `Response cut off by the token limit.`.
- [ ] **Step 2: Run** `uv run pytest tests/test_agent.py -q --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement.** Move the pt text verbatim into `messages_pt.py` (copy from the current `SYSTEM_PROMPT`; the hash test guards it). The en prompt opens with: `You are a personal financial assistant in the terminal. ALWAYS respond in English, clearly and concisely, formatting amounts as R$ 1,234.56 and dates as YYYY-MM-DD.`, then a faithful translation of rules 1 to 7, plus rule 8: `Stored category labels (for example 'Alimentação') are in Brazilian Portuguese. You may filter by them or by Pluggy's English names, and you must translate them naturally when answering.`. Update the docs mention to `agent.system_prompt`.
- [ ] **Step 4: Run** `uv run pytest tests/test_agent.py -q --no-cov`. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(i18n): localised assistant system prompt`.

### Task 4: Tools

**Files:**
- Modify: `src/llmopenfinance/tools.py`, `src/llmopenfinance/analytics.py` (add `localize_subscriptions`), both message modules
- Test: `tests/test_tools.py`, `tests/test_analytics.py` (append)

**Interfaces:**
- Produces (`tools.py`): `build_base_tools() -> tuple[ToolSpec, ...]`, `build_sql_tool() -> ToolSpec`, `sql_schema_doc() -> str`. `BASE_TOOLS` and `SQL_TOOL` stay as import-time module attributes (`BASE_TOOLS = build_base_tools()`); `SQL_SCHEMA_DOC` and `_MONTHS` become functions or are inlined. `ToolRegistry.__init__` calls the build functions so a language set after import still applies. Tool names, parameter names, enums and SQL identifiers never change.
- Produces (`analytics.py`): `localize_subscriptions(detected: dict[str, Any]) -> dict[str, Any]` returns a copy where each `subscriptions[*].frequency` and `.confidence` is replaced by `t("enum.frequency.<code>")` / `t("enum.confidence.<code>")`. Internal codes stay `mensal|anual|semanal|irregular` and `alta|média|baixa`; catalog keys are `monthly|annual|weekly|irregular` and `high|medium|low` (pt-BR values are the current words, en values the English words). `find_recurring` and `detect_subscriptions` are not changed.

- [ ] **Step 1: Write the failing tests** (all with `use_language("en")` unless noted): `test_tool_specs_same_shape_in_both_languages` (names and parameter property names equal between `build_base_tools()` under pt-BR and en); `test_tool_descriptions_en_have_no_portuguese_accents`; `test_registry_uses_current_language` (`ToolRegistry(...).specs` descriptions are English under en even though the module was imported under pt-BR); `test_tool_error_messages_en` (unknown tool starts `Unknown tool`, invalid arguments `Invalid arguments:`, missing argument `missing required arguments:`, repeated call note starts `[You already called this tool`); `test_to_json_truncation_marker_en` (`…"[result truncated — refine the filters]"`); `test_localize_subscriptions` (pt-BR is the identity; en maps `mensal` to `monthly` and `média` to `medium`; input not mutated); `test_subscriptions_tool_localised` (the `detect_subscriptions` tool result under en contains `"monthly"`, not `"mensal"`).
- [ ] **Step 2: Run** the two test files with `--no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement.** Replace every Portuguese literal in `tools.py` (descriptions, error texts, truncation marker, the SQL schema doc) with catalog keys. In the SQL doc keep `category (local pt-BR label)` wording in en. Wire `localize_subscriptions` into `_subscriptions`.
- [ ] **Step 4: Run** the same command. Expected: PASS; existing tests unchanged.
- [ ] **Step 5: Run CHECK**, then commit `feat(i18n): localised tool specs and enum labels`.

### Task 5: Analytics snapshot and SQL guard

**Files:**
- Modify: `src/llmopenfinance/analytics.py` (lines 576-578, 598-700), both message modules
- Test: `tests/test_analytics.py` (append)

**Interfaces:**
- Consumes: `localize_subscriptions` (Task 4), `format_date`, `brl`.
- Produces: `build_snapshot(con, *, today=None) -> str` unchanged signature; every heading and label comes from the catalog (`snapshot.*`); `run_select` error keys `error.sql.one_statement`, `error.sql.select_only`.

- [ ] **Step 1: Write the failing tests:** `test_build_snapshot_en` (fixture `con`, `use_language("en")`, `today=TODAY`): contains `# Financial summary`, `## Accounts`, `## Monthly cash flow`, `## Largest spending categories`, `## Active recurring charges detected`, and `R$ 1,` style amounts; contains none of `Resumo`, `Contas`, `Fluxo`; `test_build_snapshot_empty_db_en` (`(no accounts synced)` line and the attention heading `## Connections needing attention`); `test_run_select_errors_en` (`Send exactly one SQL statement.`, `Only SELECT queries are allowed.`); `test_snapshot_pt_unchanged` is the existing `test_build_snapshot` (must stay green).
- [ ] **Step 2: Run** `uv run pytest tests/test_analytics.py -q --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement** by replacing each f-string label with `t(...)` and the date in the header via `format_date`; subscriptions in the snapshot go through `localize_subscriptions`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(i18n): localised financial snapshot`.

### Task 6: LLM, sync and Connect messages

**Files:**
- Modify: `src/llmopenfinance/chat.py` (lines 52-54, 434, 696, 741, 756, 799), `src/llmopenfinance/sync.py` (lines 103-105, 139, 144, 207, 232-233), `src/llmopenfinance/connect.py`, both message modules
- Test: `tests/test_chat.py`, `tests/test_sync.py`, `tests/test_connect.py` (append)

**Interfaces:**
- Produces: `chat.final_answer_note() -> str` (replaces the `FINAL_ANSWER_NOTE` constant); `connect.build_page(connect_token, base_path, *, sandbox=False, script_url=PLUGGY_CONNECT_JS) -> str` keeps its signature but renders the page text and `<html lang>` from the catalog, and sets the widget option `"language"` to `"pt"` or `"en"`. Sync product labels use keys `sync.product.investments` and `sync.product.loans` (the `transient` list still holds the localised label).

- [ ] **Step 1: Write the failing tests** (`use_language("en")`): `test_get_llm_unknown_provider_en` (`Unknown provider 'x'. Use: claude | ollama`); `test_final_answer_note_en`; `test_consent_warning_en` (expired: `consent expired on 2026-05-01 — reconnect at meu.pluggy.ai`, expiring: `consent expires on 2026-12-01 — renew at meu.pluggy.ai`); `test_sync_product_warnings_en` (`investments unavailable (403 )` style and `loans failed (see the log)`); `test_build_page_en` (`lang="en"` in the HTML and `"language": "en"` in the options JSON; pt-BR keeps `lang="pt-BR"` and `"language": "pt"`); `test_synced_categories_do_not_depend_on_language` (populate a store under pt-BR and under en; the set of `(id, category)` from `v_transactions` is identical).
- [ ] **Step 2: Run** the three files with `--no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement** with catalog keys under `llm.*`, `sync.*`, `connect.*`; date formatting in `_consent_warning` through `format_date`. `pluggy.py` messages are untouched.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(i18n): localised LLM, sync and connect messages`.

### Task 7: CLI strings and startup guard

**Files:**
- Modify: `src/llmopenfinance/__main__.py`, both message modules, `docs/usage.md`, `README.md`
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: every user-visible string in `__main__.py` is `t("cli.*")` or `t("chat.*")`, including all Typer `help=` strings (pass `help=` explicitly; keep short English docstrings for developers) and `HELP_TEXT` (becomes `t("chat.help")`). At the top of the module, before the Typer app is built, resolve the language: on `ValueError`, write the message (plain English, since no language is known) to stderr and `raise SystemExit(2)`. Chat command aliases of both languages stay accepted in either (`/ajuda` and `/help`, `/resumo` and `/summary`, `/historico` and `/history`, `sair` and `exit`).

Required pt-BR / en pairs asserted by tests (the rest are faithful translations):

| key meaning | pt-BR (unchanged) | en |
|---|---|---|
| no connections | `Nenhuma conexão registrada.` | `No connections registered.` |
| sync summary | `Sincronização concluída: 1 ok, 0 com falha.` | `Sync finished: 1 ok, 0 failed.` |
| chat reset | `Conversa reiniciada.` | `Conversation reset.` |
| empty history | `(vazio)` | `(empty)` |
| goodbye | `Até logo!` | `Goodbye!` |
| chat help first line | `Comandos: /ajuda  /resumo ...` | `Commands: /help  /summary ...` |

- [ ] **Step 1: Write the failing tests** (`use_language("en")`, same `run(env, ...)` helper as the existing tests): `test_en_items_list_empty`; `test_en_sync_downloads_and_reports`; `test_en_status_labels` (`Local database:`, `Last sync:`, `never`); `test_en_chat_session_commands` (`/help`, `/summary`, `/history`, `/reset`, `exit`; output contains the table's en strings; the summary shows the English snapshot heading); `test_en_ask_error_prefix` (`[error] ...`); `test_en_privacy_notice_for_claude`; `test_chat_aliases_accepted_in_any_language`; `test_invalid_language_exits_cleanly` (run `python -I -m llmopenfinance ver` through `subprocess` with `LLMOPENFINANCE_LANG=fr`: exit code 2, stderr names `pt-BR` and `en`, no `Traceback`; with `" EN "` it exits 0).
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -q --no-cov`. Expected: new tests FAIL, existing PASS.
- [ ] **Step 3: Implement.** Replace each literal; keep `_handle_*` structure for now (Task 8 moves it). Document `LLMOPENFINANCE_LANG` in `docs/usage.md` (env table) and `README.md`.
- [ ] **Step 4: Run** `uv run pytest tests/test_cli.py -q --no-cov`, then the whole suite. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(i18n): localise CLI; pt-BR and en`. **Milestone 1 ends here: the app is fully usable in both languages.**

---

## Milestone 2: Rich CLI

### Task 8: Shared session core (pure refactor)

**Files:**
- Create: `src/llmopenfinance/session.py`, `tests/test_session.py`
- Modify: `src/llmopenfinance/__main__.py`

**Interfaces** (all in `session.py`):
- `@dataclass(frozen=True) class Notice: level: str; text: str` with `level` one of `"ok" | "info" | "warn" | "error" | "dim"`.
- `@dataclass(frozen=True) class Factories: llm: Callable[..., Any]; open_store: Callable[..., Any]; pluggy_client: Callable[[], Any]`.
- `class ChatSession(open_store: Callable[..., Any])` with attribute `store`, methods `connection() -> Any`, `snapshot() -> str`, `resync(run_sync: Callable[[], int]) -> int` (closes the store, calls `run_sync`, reopens read-only in a `finally`), `close() -> None`.
- `@dataclass(frozen=True) class AssistantBuild: assistant: Any | None; notices: tuple[Notice, ...]`; `build_assistant(factories: Factories, session: ChatSession, *, provider: str, model: str, options: BuildOptions) -> AssistantBuild` where `BuildOptions(think: bool, enable_sql: bool | None)` is a frozen dataclass. Notices carry the connection check, the Claude privacy warning, and errors (unknown provider, unreachable LLM with the `ollama serve` hint). It prints nothing.
- `parse_command(text: str) -> str | None` returns `"help" | "summary" | "reset" | "history" | "sync" | "exit"` or `None` (case-insensitive, both languages' aliases from Task 7).
- `run_command(name: str, assistant: Any, session: ChatSession, run_sync: Callable[[], int]) -> tuple[Notice, ...]` for `help`, `summary`, `reset`, `history`, `sync` (sync: `session.resync`, then `assistant.refresh()`, notice `ok` or `warn` as the CLI shows today).
- `run_sync(factories: Factories, *, force: bool, notify: Callable[[Notice], None], on_item: Callable[[ItemResult], None]) -> int`: the logic of today's `_run_sync` (credentials, locked DB, no items, auth failure, per-item callbacks, summary), returning the same exit codes and emitting `Notice`s instead of printing.

`__main__.py` keeps these names defined or importable, because `tests/test_cli.py` patches them: `_pluggy_client`, `_llm`, `_open_store`, `_render_answer`, `_ChatSession` (alias of `ChatSession`), `_chat_loop(assistant, session, prompt=...)`, `main`, `app`. It builds `Factories(llm=_llm, open_store=_open_store, pluggy_client=_pluggy_client)` **at call time** (a small `_factories()`), so monkeypatching still takes effect.

- [ ] **Step 1: Write the failing tests** in `tests/test_session.py` (fakes from `tests/test_agent.py` `ScriptedChat` and `tests/conftest.py`): `test_parse_command_aliases` (parametrised over every alias in both languages, `"hello"` gives `None`, `"SAIR"` gives `"exit"`); `test_build_assistant_unreachable_llm_notices` (error notice contains `ollama serve` for Ollama and `ANTHROPIC_API_KEY` for Claude); `test_build_assistant_claude_privacy_notice` (`dim` or `warn` level, mentions Anthropic); `test_build_assistant_unknown_provider` (assistant is `None`, one `error` notice); `test_run_command_reset_and_history`; `test_run_sync_no_credentials` (returns 1, one `error` notice); `test_run_sync_no_items_returns_0`; `test_run_sync_reports_items_via_callback` (fake Pluggy from `tests/test_cli.py`'s `FakePluggy` pattern).
- [ ] **Step 2: Run** `uv run pytest tests/test_session.py -q --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement `session.py` and rewire `__main__.py`** to render `Notice`s with the current `typer.secho` colours (no visual change yet).
- [ ] **Step 4: Run the whole suite.** Expected: PASS with `tests/test_cli.py` unmodified (if a single test truly cannot be kept as is, change only its patch target, never an expected string).
- [ ] **Step 5: Run CHECK**, then commit `refactor: extract shared chat session core`.

### Task 9: `ui.py`

**Files:**
- Create: `src/llmopenfinance/ui.py`, `tests/test_ui.py`
- Modify: `pyproject.toml` (`uv add rich`)

**Interfaces** (all in `ui.py`; `console` is a module attribute tests may replace):
- `THEME: rich.theme.Theme` with styles `ok`, `warn`, `err`, `dim`, `accent`, `user`, `assistant`.
- `console: Console = Console(theme=THEME, highlight=False)`; colour follows `NO_COLOR` and TTY detection (Rich default).
- `interactive() -> bool` (`console.is_terminal`).
- `show_notice(notice: Notice) -> None` (`ok` prefixed `✓`, `warn` `!`, `error` `✗`, `dim` styled dim).
- `item_result(result: ItemResult) -> None` keeps today's markers: `  ✓ id: k=v, ...`, `  = id: ...`, `  ✗ id: error`, and `    ! warning` lines.
- `banner(provider: str, model: str, language: str, version: str) -> None` (a titled panel; plain two-line text when not interactive).
- `render_answer(events: Iterable[AssistantEvent]) -> bool` (TTY: `Live` with a `Markdown` panel and a spinner status line per `ToolStatus`; non-TTY: stream `TextDelta` text with `soft_wrap=True`, tool statuses as dim `  ↳ consultando ...` lines, errors as `[erro] ...`; returns `False` if an `AssistantError` was shown).
- `key_value_line(pairs: Mapping[str, object]) -> str` returning `k=v, k=v` (the `status` data counts line).

**Rule for every helper:** dynamic text is printed with `markup=False, emoji=False` (or wrapped in `rich.text.Text`), never interpreted.

- [ ] **Step 1: Write the failing tests** (capture with `Console(file=io.StringIO(), theme=THEME, width=80)` swapped in through `monkeypatch.setattr(ui, "console", ...)`): `test_notice_levels_and_symbols`; `test_item_result_markers_match_cli` (the three markers and warning indent); `test_dynamic_text_is_verbatim` (a notice with `[erro] [bold]x[/bold] :tada: [PROMO]` prints that exact text, no `MarkupError`); `test_no_ansi_when_not_a_terminal` (`"\x1b" not in output` for notice, banner and `render_answer`); `test_render_answer_plain_streams_text_tools_and_errors` (a scripted event list; `ok` is `False` after an error); `test_render_answer_live_on_forced_terminal` (`Console(force_terminal=True, file=StringIO())`: no exception, the answer text appears in the output); `test_no_color_env_disables_styles` (`NO_COLOR=1`).
- [ ] **Step 2: Run** `uv run pytest tests/test_ui.py -q --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement.** Add `rich` to `[project].dependencies` with `uv add rich`.
- [ ] **Step 4: Run** `uv run pytest tests/test_ui.py -q --no-cov`. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(ui): Rich console helpers`.

### Task 10: Restyle the CLI

**Files:**
- Modify: `src/llmopenfinance/__main__.py`, `docs/usage.md`, `docs/architecture.md`
- Test: `tests/test_cli.py` (append)

**Interfaces:** `__main__.py` renders only through `ui`. `typer.Typer(...)` gets `rich_markup_mode="rich"`, `pretty_exceptions_show_locals=False`, an epilog listing the environment variables, and `rich_help_panel` groups (`Chat`: `chat`, `ask`; `Data`: `sync`, `status`; `Connections`: `connect`, `items`; panel titles come from the catalog). `--version` and `ver` stay. The chat prompt label is styled with `typer.style` and stays injectable through `_chat_loop(..., prompt=...)`. `status` and `items list` use Rich tables or styled one-line rows, whichever keeps these substrings intact at 80 columns in the existing tests: `transactions=`, `consentimento até 01/06/2027`, `não sincronizado`, `✓ item-1`. The first resort is `Table` cells with `no_wrap=True` printed with `crop=False`; otherwise fall back to one styled line per item.

- [ ] **Step 1: Write the failing tests:** `test_help_groups_commands` (`--help` output contains the three panel titles and every command name); `test_help_en` (`use_language` cannot re-bind import-time help, so run `python -I -m llmopenfinance --help` through `subprocess` with `LLMOPENFINANCE_LANG=en` and assert an English line and no `Assistente`); `test_ask_output_has_no_ansi` (`"\x1b" not in result.output`, answer text intact); `test_status_table_keeps_asserted_substrings` (the existing `test_status` and `test_items_list_*` already assert them, so this task must keep them green); `test_chat_shows_banner_with_language` (banner line contains the provider, model and `pt-BR`).
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -q --no-cov`. Expected: new tests FAIL, existing PASS.
- [ ] **Step 3: Implement the restyle.** Document the new look in `docs/usage.md` and the new modules (`i18n`, `session`, `ui`) in `docs/architecture.md`.
- [ ] **Step 4: Run the whole suite.** Expected: PASS with no expected-string edits.
- [ ] **Step 5: Run CHECK**, then commit `feat(cli): Rich-styled interface`. **Milestone 2 ends here.**

---

## Milestone 3: Web chat

### Task 11: Dependencies and the Waitress streaming check

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `docs/development.md`
- Scratch (not committed): `D:\Users\jorge\AppData\Local\Temp\claude\16e434c877bc9786\d1a642b0-cc69-4c06-888e-f5bde997ea7f\scratchpad\waitress_spike.py`

- [ ] **Step 1:** `uv add --optional web flask waitress`, then `uv add --dev flask waitress`. Check `[project.optional-dependencies] web` lists both and the dev group lists both.
- [ ] **Step 2: Write the throwaway spike:** a Flask route `POST /slow` yielding five `application/x-ndjson` lines one second apart, served with `waitress.serve(app, host="127.0.0.1", port=8765)`.
- [ ] **Step 3: Run it and observe:** `curl -sN -X POST http://127.0.0.1:8765/slow` piped through a loop that prints a timestamp per line. Expected: lines arrive about one second apart. If they arrive together at the end, Waitress buffers: continue anyway (the NDJSON client works either way) and write that limitation in `docs/development.md`.
- [ ] **Step 4:** Record the result and the exact curl command under a "Web server notes" heading in `docs/development.md`.
- [ ] **Step 5: Run CHECK**, then commit `chore(web): add optional web extra and rich dependency`.

### Task 12: Web app skeleton, security, state and commands

**Files:**
- Create: `src/llmopenfinance/web.py`, `tests/test_web.py`

**Interfaces** (`web.py`, Flask imported at module top):
- `new_token() -> str` (`secrets.token_urlsafe(24)`).
- `host_allowed(host_header: str, bound_host: str, port: int) -> bool`: accepts `localhost`, `127.0.0.1`, `[::1]` with the bound port; the exact `bound_host:port` for a specific address; **any** Host when `bound_host` is `0.0.0.0` or `::`.
- `@dataclass class WebContext: session: ChatSession; assistant: Any; factories: Factories; token: str; host: str; port: int; options: BuildOptions; provider: str; model: str; lock: threading.Lock; transcript: list[dict[str, str]]; status: dict[str, Any]` plus `refresh_status() -> None` (reads `last_sync` and `counts()` from the store; call only while holding the lock or at startup). `/api/state` serves the cached `status`, never the database.
- `create_app(ctx: WebContext) -> Flask` with `MAX_CONTENT_LENGTH = 64 * 1024`.
- Wire format for every streaming route: `application/x-ndjson`, one JSON object per line, `{"type": "text", "text": str}`, `{"type": "tool", "name": str, "arguments": dict}`, `{"type": "notice", "level": str, "text": str}`, `{"type": "progress", "item_id": str, "ok": bool, "skipped": bool, "counts": dict, "warnings": list, "error": str | None}`, `{"type": "error", "text": str}`, `{"type": "done", "ok": bool}`.
- Routes in this task: `GET /` (placeholder page until Task 15; a valid `?token=` sets cookie `finassist_token`, `HttpOnly`, `SameSite=Strict`, and redirects to `/`), `GET /api/state` (`provider`, `model`, `language`, `last_sync`, `counts`, `busy`, `strings`, `transcript`), `POST /api/command` (`{"name": "help|summary|reset|history"}` returning `{"notices": [{"level", "text"}]}`; unknown name gives 400; busy gives 409 with a localised message).
- Rules: every request checks Host (400), then token via cookie compared with `secrets.compare_digest` (401, JSON `{"error": ...}` localised); `POST` requires `Content-Type: application/json` (415); every response carries `Cache-Control: no-store`.

- [ ] **Step 1: Write the failing tests** (Flask test client; build a `WebContext` from `ScriptedChat` in `tests/test_agent.py` and the `con`/`db_path` fixtures): `test_host_allowed` (parametrised, including `evil.example:8000` False, wrong port False, wildcard bind True); `test_token_cookie_flow` (no cookie gives 401; `/?token=ok` sets cookie and redirects; wrong token gives 401); `test_wrong_host_rejected_before_token` (400); `test_post_requires_json` (415); `test_state_shape_and_language` (`language == "en"` under `use_language("en")`; `strings` non-empty); `test_command_routes` (help, summary, history, reset; `summary` returns the snapshot text); `test_busy_returns_409` (hold `ctx.lock`, call `/api/command`); `test_oversized_body_is_413`; `test_token_never_logged` (`caplog` has no record containing the token).
- [ ] **Step 2: Run** `uv run pytest tests/test_web.py -q --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement** with `session.run_command` for commands. Catalog keys under `web.*` (`web.busy`, `web.unauthorized`, `web.bad_request`, `web.too_large`, plus the page strings added in Task 15).
- [ ] **Step 4: Run** `uv run pytest tests/test_web.py -q --no-cov`. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(web): Flask app with token, host check and command routes`.

### Task 13: Streaming chat route and transcript

**Files:**
- Modify: `src/llmopenfinance/web.py`, `tests/test_web.py`

**Interfaces:** `POST /api/chat` with `{"message": str}`: empty or non-string message gives 400. If the typed text is a slash command (`session.parse_command`), the route streams `notice` events and `done`. Otherwise it streams `text`, `tool` and `error` events from `ctx.assistant.ask(message)` and ends with `done`. The route acquires `ctx.lock` non-blocking before returning the response (409 if held) and releases it in the generator's `finally`. After a completed answer, append `{"role": "user", "text": ...}` and `{"role": "assistant", "text": ...}` to `ctx.transcript` (never persisted, never logged); if the history is the better source, derive it, otherwise keep this list. An unexpected exception in the stream becomes an `error` event, never a 500 or a crash.

- [ ] **Step 1: Write the failing tests:** `test_chat_streams_text_tool_and_done` (scripted tool turn then answer; line-by-line JSON); `test_chat_error_event_and_history_rolled_back` (`LLMError` gives an `error` event; `chat.truncate_history` effect: history length unchanged); `test_chat_busy_409_does_not_queue`; `test_chat_slash_command_in_box` (`/reset` streams a `notice`); `test_chat_empty_message_400`; `test_transcript_in_state_after_answer`; `test_aborted_stream_releases_lock_and_rolls_back` (**Review Focus 3**: open with `buffered=False`, read one line, `response.close()`, then `ctx.lock.acquire(blocking=False)` succeeds and the history length equals its starting value); `test_unexpected_exception_releases_lock` (an assistant whose `ask` raises `RuntimeError`: an `error` event and the lock is free).
- [ ] **Step 2: Run** `uv run pytest tests/test_web.py -q --no-cov`. Expected: new tests FAIL.
- [ ] **Step 3: Implement** with `flask.stream_with_context` and a generator wrapper that owns the lock.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(web): streaming chat route with transcript`.

### Task 14: Streaming sync route

**Files:**
- Modify: `src/llmopenfinance/web.py`, `tests/test_web.py`

**Interfaces:** `POST /api/sync` with `{"force": bool}` (default false), same lock and wire format. It runs `session.run_sync` through `ctx.session.resync(...)` so the read-only handle is closed, the sync runs read-write, and the read-only handle is reopened in a `finally`; it emits `notice` events for session notices and one `progress` event per item (`on_item`); on success it calls `ctx.assistant.refresh()` and `ctx.refresh_status()`; it ends with `done` (`ok` is the exit code being 0). Typing `/sync` in the chat box routes here. Sync failures become `error`/`notice` events and leave the server usable.

- [ ] **Step 1: Write the failing tests** (fake Pluggy as in `tests/test_cli.py`, DB from `db_path`): `test_sync_streams_progress_and_done`; `test_sync_refreshes_assistant_and_status` (counts in `/api/state` change after sync); `test_sync_busy_409_when_chat_holds_lock`; `test_sync_without_credentials_keeps_server_usable` (**Review Focus 4**: an `error` notice, `done` with `ok: false`, then `/api/command` `summary` still works and `ctx.session.store` is usable); `test_sync_database_locked_keeps_server_usable` (patch `open_store` to raise `duckdb.IOException` for the read-write open only; afterwards the read-only store works); `test_sync_slash_command_in_chat_box`.
- [ ] **Step 2: Run** `uv run pytest tests/test_web.py -q --no-cov`. Expected: new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Run CHECK**, then commit `feat(web): streaming sync route`.

### Task 15: The page

**Files:**
- Create: `src/llmopenfinance/web_page.py`
- Modify: `src/llmopenfinance/web.py`, `tests/test_web.py`, both message modules

**Interfaces:** `web_page.render_page(*, language: str, strings: Mapping[str, str], nonce: str) -> str` (one self-contained HTML document: inline CSS and JS carrying the nonce; `<html lang>` is the language; light and dark via `prefers-color-scheme`; usable at phone width). `GET /` returns it after authentication. Behaviour: load `/api/state`, render transcript and a status line (provider, model, language, last sync), message list with tool-call lines, input box, buttons for help, summary, sync and reset, a busy indicator, and the localised busy text on 409. Client reads the NDJSON stream with `response.body.getReader()` and works whether chunks arrive progressively or all at once. A small `renderMarkdown(text)` function escapes HTML **first**, then supports headings, bold, italics, inline code, lists and tables. The response sets a restrictive `Content-Security-Policy` that allows only the nonce'd inline script and style and `connect-src 'self'`. Catalog keys `web.ui.*` for every visible label.

- [ ] **Step 1: Write the failing tests:** `test_page_language_and_strings` (`lang="en"` and the English send-button label under en; `lang="pt-BR"` under pt-BR); `test_page_has_no_external_resources` (no `src=` or `href=` or `url(` pointing at `http://` or `https://`); `test_page_csp_header` (nonce present in header and in both inline tags, no `unsafe-inline`); `test_page_requires_auth`.
- [ ] **Step 2: Run** `uv run pytest tests/test_web.py -q --no-cov`. Expected: FAIL.
- [ ] **Step 3: Implement** the page and wire it in.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Manual browser check** (the JS cannot be unit-tested here): write a scratchpad script that builds a `WebContext` with a scripted assistant whose answer is `**ok** <img src=x onerror=alert(1)> | a | b |` and serves it with Waitress, then use the Chrome DevTools tools to open the printed URL, send a message, and confirm: the answer renders, no dialog appears, the image tag shows as literal text, the busy indicator shows during streaming, and the reset and sync buttons work. Note the result in the commit message.
- [ ] **Step 6: Run CHECK**, then commit `feat(web): inline chat page`.

### Task 16: The `server` command, docs and close-out

**Files:**
- Modify: `src/llmopenfinance/__main__.py`, `src/llmopenfinance/web.py`, `tests/test_cli.py`, `README.md`, `docs/usage.md`, `docs/architecture.md`, `docs/development.md`, both message modules

**Interfaces:**
- `web.serve(ctx: WebContext, *, threads: int = 4, serve_fn: Callable[..., Any] | None = None) -> None` (calls `waitress.serve(app, host=ctx.host, port=ctx.port, threads=threads)`; `serve_fn` is the test seam).
- `_handle_server` constructs `WebContext(...)` directly (no separate builder function).
- `@dataclass(frozen=True) class ServerOptions: host: str; port: int; threads: int; open_browser: bool; sync: bool; think: bool; enable_sql: bool | None` in `__main__.py`; `_handle_server(options: ServerOptions) -> int`.
- Typer command `server` with `--host` (default `127.0.0.1`), `--port` (`8000`), `--threads` (`4`), `--no-browser`, `--sync/--no-sync` (default on), `--think`, `--sql/--no-sql`, grouped under the `Chat` help panel. Startup order: optional sync, open the read-only store, `build_assistant`, render notices, import `web` lazily, build the context, print the URL `http://<host>:<port>/?token=<token>` (loopback name for wildcard hosts), open the browser through a module-level seam `_open_browser = webbrowser.open` unless `--no-browser`, then `web.serve`. A non-loopback `--host` prints a prominent localised warning (plain HTTP, wildcard binds rely on the token alone). `KeyboardInterrupt` exits 0 with the goodbye text; the session is always closed.
- Errors, all exit 1 with one localised line: the `web` extra is missing (message names `uv sync --extra web`); the LLM is unreachable (same notices as chat); the port is in use (`OSError` from `serve`).

- [ ] **Step 1: Write the failing tests** (`serve_fn` and `_open_browser` monkeypatched, nothing binds a socket): `test_server_starts_and_prints_tokened_url` (URL in output, `serve_fn` called with the host, port and threads); `test_server_no_browser_flag` (`_open_browser` not called; called without the flag); `test_server_missing_extra` (`monkeypatch.setitem(sys.modules, "flask", None)` and `monkeypatch.delitem(sys.modules, "llmopenfinance.web", raising=False)`; exit 1; message mentions `--extra web`); `test_server_llm_unreachable_exits_1`; `test_server_port_in_use_is_clean` (**Review Focus 5**: `serve_fn` raising `OSError`; exit 1, one line, no `Traceback`); `test_server_non_loopback_warning` (`--host 0.0.0.0` prints the warning); `test_server_syncs_before_serving` (`--sync` default calls the fake Pluggy; `--no-sync` does not); `test_server_en_messages` (`use_language("en")`).
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -q --no-cov`. Expected: new tests FAIL.
- [ ] **Step 3: Implement.** Docs: `README.md` and `docs/usage.md` get the `server` command, options, security notes and the `uv sync --extra web` install line; `docs/architecture.md` gets `session`, `web`, `web_page`; `docs/development.md` gets the manual smoke test for the web page.
- [ ] **Step 4: Run the whole suite.** Expected: PASS, coverage at or above 90%.
- [ ] **Step 5: Run CHECK**, then commit `feat(web): finassist server command and docs`. **Milestone 3 ends here.**

---

## Self-review

- **Spec coverage:** G1 Task 1; G2 Tasks 3 to 7; G3 hash test (Task 3) and unmodified existing tests (every task); G4 Tasks 9 and 10; G5 Tasks 12 to 16; G6 CHECK. 5.1 formats Tasks 2 and 5; English-prompt category sentence Task 3; connect page and widget language Task 6; guards Task 2. 5.2 Tasks 9, 10. 5.3 Task 8 including the kept seams. 5.4 routes Tasks 12 to 15; Host rule, token, content-type and no-store Task 12; lock, 409 and transcript Task 13; sync stream Task 14; flags, extra, warning, port Task 16; Waitress check Task 11. 5.5 Tasks 9, 11, 16. A1 Tasks 9 and 10; A2 Task 12; A3 Task 6; A5 guards (Task 2) and Task 7 tests. N1 to N6 are respected: no task translates stored data or adds a language switch or web `connect`.
- **Type consistency:** `Notice`, `Factories`, `ChatSession`, `BuildOptions`, `AssistantBuild`, `parse_command`, `run_command`, `run_sync` are defined once in Task 8 and consumed by name in Tasks 9, 12, 14 and 16; `WebContext` fields are defined in Task 12 and reused as written; the NDJSON event shapes are defined once in Task 12.
- **Plan decisions beyond the spec:** NDJSON (one JSON object per line) is the stream format; the page lives in its own `web_page.py` module so `web.py` stays focused; `localize_subscriptions` maps enum labels at the output edge.
