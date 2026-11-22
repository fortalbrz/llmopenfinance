# llmopenfinance

Talk to your own bank data from the terminal. `finassist` is a console chat with an LLM financial assistant
(**Claude** or a local **Ollama** model). It answers questions about your accounts, credit cards, investments and
loans, using data pulled through Brazilian **Open Finance** via [Pluggy](https://pluggy.ai) and cached in a local
**DuckDB** database.

```text
$ finassist
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

Example questions (answers are always in Brazilian Portuguese):

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
uv run finassist items add <item-id>       # item ids copied from the Pluggy dashboard
#   or: uv run finassist connect           # opens the Pluggy Connect widget in your browser

# 4. Chat (syncs first)
uv run finassist                           # Claude (claude-sonnet-5-5)
uv run finassist --provider ollama         # 100% local (qwen3.5:9b)
```

## Commands

| Command | What it does |
|---|---|
| `finassist` / `finassist chat` | Interactive chat. Syncs first; skip with `--no-sync` |
| `finassist ask "pergunta"` | Asks one question and exits |
| `finassist sync [--force]` | Downloads accounts, transactions, card bills, investments and loans |
| `finassist status` | Shows connections, consent expiry dates and local row counts |
| `finassist connect [--sandbox]` | Connects accounts through the Pluggy Connect widget on a local page |
| `finassist items add/list/remove` | Manages the registered Pluggy item ids |

Global options: `--provider claude|ollama`, `--model <id>`, `--db <file>`, `-v`.
See [docs/usage.md](docs/usage.md) for the full reference.

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
