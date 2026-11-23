# llmopenfinance

Talk to your own bank data from the terminal. `llmopenfinance` is a console chat with an LLM financial assistant
(**Claude** or a local **Ollama** model). It answers questions about your accounts, credit cards, investments and
loans, using data pulled through Brazilian **Open Finance** via [Pluggy](https://pluggy.ai) and cached in a local
**DuckDB** database.

```text
$ llmopenfinance
Assistente financeiro pronto. Comandos: /ajuda  /resumo (dados enviados ao modelo)  /sync  /reset  /historico  sair
======================================================================

Você: Quais assinaturas eu tenho?

Assistente:
  ↳ consultando detect_subscriptions(months=12)
| Serviço | Valor | Última cobrança | Custo anual |
|---|---|---|---|
| Netflix | R$ 55,90 | 08/11/2026 | R$ 670,80 |
| Spotify | R$ 21,90 | 12/11/2026 | R$ 262,80 |
...
```

Example questions (answers are in Brazilian Portuguese by default; set `LLMOPENFINANCE_LANG=en` for English,
see [Language](#language)):

- *Quais serviços eu assino?* (what services do I subscribe to?)
- *Quanto gasto por mês com compras online?* (how much do I spend on online shopping each month?)
- *Como posso melhorar minhas finanças?* (how can I improve my finances?)
- *Quais foram minhas 3 maiores despesas em novembro?* (what were my 3 largest expenses in November?)
- *Quanto ainda devo no empréstimo e qual é o CET?* (how much is left on my loan, and what is its total effective cost?)

## Quick start

```powershell
# 1. Install (uv only)
uv sync

# 2. Credentials (Pluggy application clientId/clientSecret from dashboard.pluggy.ai)
$env:PLUGGY_USERNAME = "<clientId>"
$env:PLUGGY_PASSWORD = "<clientSecret>"
$env:ANTHROPIC_API_KEY = "<key>"          # only for the default Claude provider

# 3. Connect your banks (MeuPluggy, free for personal use) — see docs/pluggy-setup.md
uv run llmopenfinance items add <item-id>       # item ids copied from the Pluggy dashboard
#   or: uv run llmopenfinance connect           # opens the Pluggy Connect widget in your browser

# 4. Chat (syncs first)
uv run llmopenfinance                           # Claude (claude-sonnet-5-5)
uv run llmopenfinance --provider ollama         # 100% local (qwen3.5:9b)
```

## Commands

| Command | What it does |
|---|---|
| `llmopenfinance` / `llmopenfinance chat` | Interactive chat. Syncs first; skip with `--no-sync` |
| `llmopenfinance server [--host H] [--port N]` | Serves the same chat as a local web page (needs `uv sync --extra web`) |
| `llmopenfinance ask "pergunta"` | Asks one question and exits |
| `llmopenfinance sync [--force]` | Downloads accounts, transactions, card bills, investments and loans |
| `llmopenfinance status` | Shows connections, consent expiry dates and local row counts |
| `llmopenfinance connect [--sandbox]` | Connects accounts through the Pluggy Connect widget on a local page |
| `finassist items add/list/remove` | Manages the registered Pluggy item ids |

Global options: `--provider claude|ollama`, `--model <id>`, `--db <file>`, `-v`.
See [docs/usage.md](docs/usage.md) for the full reference.

## Language

The application speaks Brazilian Portuguese (`pt-BR`, the default) or English (`en`): the command help, every message,
the assistant's system prompt and its answers. Pick one with the `LLMOPENFINANCE_LANG` environment variable. There is
no command-line option for it.

```powershell
$env:LLllmopenfinanceNCE_LANG = "en"    # or "pt-BR"
uv run finassist
```

- English shows money as `R$ 1,234.56` and dates as `YYYY-MM-DD`. Portuguese keeps `R$ 1.234,56` and `DD/MM/AAAA`.
- An unsupported value stops the program with a short message on stderr and exit status 2.
- Stored data such as category labels (`Alimentação`) is never translated, and the chat commands of both languages
  (`/ajuda` and `/help`, `/resumo` and `/summary`, `sair` and `exit`) work in either.

## Web chat

`finassist server` serves the same assistant as a chat page in your browser, with buttons for `/help`, `/summary`,
`/sync`, `/reset` and `/history` and a language that follows `LLMOPENFINANCE_LANG`. It needs the optional `web` extra
(Flask and Waitress):

```powershell
uv sync --extra web
uv run finassist server                    # syncs, prints a URL with a token and opens the browser
uv run finassist server --no-browser --no-sync --port 8123
```

Options: `--host` (default `127.0.0.1`), `--port` (`8000`), `--threads` (`4`, at least 2), `--browser/--no-browser`,
`--sync/--no-sync`, `--think` and `--sql/--no-sql` (the last three work as in `chat`). Without the extra the command
prints the install line and exits with status 1.

Security, in short (details in [docs/usage.md](docs/usage.md#server)):

- It listens on the loopback address only, unless you pass another `--host`; the page then travels over **plain HTTP**
  and `server` prints a warning. With a wildcard host (`0.0.0.0`, `::`) the token alone protects access.
- Every launch creates a new secret token. The URL with the token (`http://127.0.0.1:8000/?token=...`) is printed once
  and never logged; the browser keeps it in an `HttpOnly` cookie and the address bar drops it.
- Requests whose `Host` header is not this server are refused (DNS-rebinding defence).
- One server process holds one conversation, shared by every tab. A request that arrives while the assistant is busy is
  refused with "busy"; requests are not queued.

## How it works

```text
Pluggy API ──sync──▶ DuckDB (local) ──read-only──▶ tools ◀──tool calls── LLM (Claude / Ollama)
                         │                                                    ▲
                         └──── financial snapshot (aggregates) ───────────────┘ system prompt
```

- **Sync** (`sync.py`) pulls each item's full history. It is incremental per item and skips unchanged items.
  A failing item or product is reported, and the rest of the sync continues.
- **Local enrichment** (`categorize.py`):
  - Every transaction gets a pt-BR category and a *kind*: expense, refund, income, transfer or investment.
  - Card bill payments and transfers between your own accounts are never counted as spending.
  - This works even after Pluggy's paid categorization stops at the end of the trial.
- **Assistant** (`agent.py`, `tools.py`, `analytics.py`):
  - The system prompt carries a compact snapshot of your finances.
  - The model calls deterministic SQL tools to get the details, so the numbers it gives come from your data.
  - For Claude there is also a guarded `run_sql` escape hatch.

Details: [docs/architecture.md](docs/architecture.md).

## Privacy

With `--provider ollama` nothing leaves your machine except the Pluggy sync itself. With Claude, the snapshot and
the rows returned by tools are sent to Anthropic's API. Each tool result is capped at 100 rows, but `run_sql` (on by
default for Claude) can select any column, including the full raw Pluggy record of a transaction. Use `--no-sql`
to limit Claude to the curated tools. The local database
lives at `~/.llmopenfinance/finance.duckdb` (change it with `--db` or `LLMOPENFINANCE_DB`).

## Development

```powershell
uv run ruff format . ; uv run ruff check . ; uv run pyrefly check ; uv run pytest
```

The tests use synthetic Pluggy payloads, `httpx.MockTransport` and fake LLMs, so they never touch the network.
The coverage gate is 90%. See [AGENTS.md](AGENTS.md).

## License

MIT
