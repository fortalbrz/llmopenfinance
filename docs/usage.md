# Usage reference

## Global options

These go **before** the command, for example `finassist --provider ollama chat`.

| Option | Env var | Default | Meaning |
|---|---|---|---|
| `--provider`, `-p` | `LLMOPENFINANCE_PROVIDER` | `claude` | LLM backend: `claude` or `ollama` |
| `--model`, `-m` | — | `claude-sonnet-5-5` / `qwen3.5:9b` | Model id for the chosen provider |
| `--db` | `LLMOPENFINANCE_DB` | `~/.llmopenfinance/finance.duckdb` | Local DuckDB file |
| `--verbose`, `-v` | — | off | INFO logs |
| `--version` | — | — | Print the version |

## Environment variables

| Variable | Used for |
|---|---|
| `PLUGGY_USERNAME` | Pluggy application **clientId** |
| `PLUGGY_PASSWORD` | Pluggy application **clientSecret** |
| `PLUGGY_ITEM_IDS` | Optional comma-separated item ids, registered automatically |
| `ANTHROPIC_API_KEY` (or `CLAUDE_TOKEN`) | Claude API key |
| `OLLAMA_SERVER` / `OLLAMA_HOST` | Ollama URL (default `http://localhost:11434`) |

## Commands

### `chat` (default)

```text
finassist [GLOBAL OPTIONS] chat [--sync/--no-sync] [--think] [--sql/--no-sql]
```

- `--no-sync` skips the Pluggy sync at start-up. It is skipped automatically when the Pluggy credentials are missing.
- `--think` turns on Ollama's thinking mode. It is slower, but it uses tools more reliably on small models.
- `--sql/--no-sql` exposes the guarded `run_sql` tool. By default it is on for Claude and off for Ollama.

In-chat commands:

| Command | Effect |
|---|---|
| `/ajuda` | Help |
| `/resumo` | Prints the financial snapshot that goes into the system prompt |
| `/sync` | Syncs, reloads the data and starts a new conversation |
| `/reset` | Starts a new conversation |
| `/historico` | Shows the conversation history |
| `sair` / `quit` / `exit` / `q` | Exit. Ctrl+C at the prompt also exits, as does end of input (Ctrl+Z then Enter on Windows, Ctrl+D elsewhere). Ctrl+C while an answer is streaming only interrupts that answer. |

### `ask`

```text
finassist ask "Quanto gastei com delivery em outubro?" [--sync] [--sql/--no-sql]
```

Answers one question and exits. Useful in scripts.

### `sync`

```text
finassist sync [--force]
```

For every registered item, `sync` checks `GET /items/{id}`. It skips the item when `lastUpdatedAt` hasn't changed,
unless you pass `--force`. Otherwise it replaces the item's local data:

- accounts and their **full** transaction history;
- credit-card bills;
- investments and loans.

Failures are reported per item or product, and the sync never stops part-way. If a product fails with a temporary
error (network, 429 or 5xx), the item is not marked as synced, so the next `sync` downloads it again without
needing `--force`.

`sync`, `connect` and `items add/remove` need write access to the database. If another `finassist` session has
it open, `sync` prints a warning and the chat continues with the local data. `status` and `items list` open the
database read-only.

### `status`

Shows the database path, the last sync, row counts and every connection with its status and consent expiry date.

### `connect`

```text
finassist connect [--sandbox] [--sync/--no-sync]
```

Opens the Pluggy Connect widget on a loopback page. See [pluggy-setup.md](pluggy-setup.md).

### `items`

```text
finassist items add <id>... [--verify/--no-verify]
finassist items list
finassist items remove <id>
```

`add` validates each id with Pluggy when the credentials are available. `remove` also deletes that item's local data.

## Choosing a model

| Model | When |
|---|---|
| `claude-sonnet-5-5` (default) | Best answers and tool use. Data leaves the machine (aggregates and tool rows only). |
| `claude-haiku-5-5` | Cheaper and faster Claude. |
| `qwen3.5:9b` (Ollama default) | Private. Fits an 8 GB GPU with `num_ctx=16384`. |
| `qwen3.8:27b` | Private and stronger, but needs about 17 GB, so it runs mostly on the CPU of an 8 GB GPU machine. |

Small local models sometimes answer from the snapshot instead of calling tools. The system prompt forbids this. If
you still see invented numbers, try `--think` or a larger model.

## Example questions

- *Quais serviços eu assino? Quanto isso custa por ano?* (what do I subscribe to, and what does it cost per year?)
- *Quanto gasto por mês com compras online nos últimos 6 meses?* (monthly online shopping over the last 6 months?)
- *Quais foram as maiores despesas da semana passada?* (what were last week's largest expenses?)
- *Minha taxa de poupança está melhorando?* (is my savings rate improving?)
- *Como posso melhorar minhas finanças?* (how can I improve my finances?)
- *Quanto pago de juros e tarifas bancárias por mês?* (how much do I pay in interest and bank fees each month?)
- *Qual o total investido e onde?* (how much do I have invested, and where?)
