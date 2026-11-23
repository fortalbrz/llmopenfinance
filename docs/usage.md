# Usage reference

## Global options

These go **before** the command, for example `llmopenfinance --provider ollama chat`.

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
| `LLMOPENFINANCE_LANG` | Language of the whole application: `pt-BR` (default) or `en`. See [Language](#language) |
| `NO_COLOR` | Any non-empty value turns off colours, panels and tables, even on a terminal. See [Look and feel](#look-and-feel) |

`llmopenfinance --help` lists the commands in three groups (Chat: `chat`, `server`, `ask`; Data: `sync`, `status`; Connections: `connect`,
`items`) and ends with a list of the main variables: `LLMOPENFINANCE_PROVIDER`, `LLMOPENFINANCE_DB`,
`LLMOPENFINANCE_LANG`, `PLUGGY_USERNAME`, `PLUGGY_PASSWORD` and `PLUGGY_ITEM_IDS`.

## Look and feel

The terminal front end is styled with [Rich](https://rich.readthedocs.io), but only where styling helps:

| | On an interactive terminal | Piped, redirected or `NO_COLOR` set |
|---|---|---|
| Messages | Coloured by meaning: green for success, yellow for warnings, red for errors, dim for side notes | Plain text, no escape codes |
| `chat` start | A banner panel with the app version, provider, model and language | Two plain lines: `llmopenfinance <version>` and `Provider: …, Model: …, Language: …` |
| Answers | Markdown in a titled panel that grows while it streams, with a spinner line for each tool call | The answer streamed as plain text, one status line per tool call |
| `status` | A table of the local data counts (`transactions=…`) and a table of the connections (ID, connector, status, consent) | Lines: `Dados: k=v, …` and `- id  connector  status=…  consentimento até …` |
| `items list` | A table (ID, connector, status) | Lines: `id  connector  status` |
| `--help` | Grouped panels and the environment-variable epilog | The same text, without colours |

- The plain output is the wording and the bytes `llmopenfinance` always printed, so `llmopenfinance ask "…" > out.txt` and
  `llmopenfinance status | grep …` keep working. The banner is shown only by `chat`: `ask`, `sync`, `status`, `items` and
  `connect` print nothing extra.
- Tables never cut a cell. The ID column folds onto several lines when the terminal is narrow; if the other columns leave
  it fewer than 12 columns (a terminal narrower than roughly 70 columns for `status`), each item is printed as one plain
  line instead.
- Text that comes from outside (model answers, merchant names, item ids) is shown as it is: it is never read as Rich
  markup, and terminal escape sequences inside it are removed.
- Model answers are Markdown, but links are shown as text only, never as clickable terminal hyperlinks.
- The prompt label (`Você` / `You`) is bold cyan on a terminal.

## Language

`LLMOPENFINANCE_LANG` picks the language of the command help, every message, the chat commands' output, the
system prompt (so the model answers in that language) and the tool descriptions. There is no `--lang` option.

| Value | Language | Money | Dates |
|---|---|---|---|
| unset, empty, `pt`, `pt-BR`, `pt_br` | Brazilian Portuguese (default) | `R$ 1.234,56` | `DD/MM/AAAA` |
| `en`, `en-US`, `en_GB` | English | `R$ 1,234.56` | `YYYY-MM-DD` |

- Case and surrounding spaces are ignored, and `_` counts as `-`. Only the first part (`pt` or `en`) matters.
- Any other value is an error: `llmopenfinance` prints a one-line message that names the supported values on stderr and exits
  with status 2.
- The language is fixed for the whole run. Stored data is never translated: category labels such as `Alimentação`,
  Pluggy values and transaction descriptions stay as they are.
- The chat commands of both languages work in either language (for example `/help` and `/ajuda`). Only the in-chat help
  names the forms of the current language.

```powershell
$env:LLMOPENFINANCE_LANG = "en"
uv run llmopenfinance --help
```

## Commands

### `chat` (default)

```text
llmopenfinance [GLOBAL OPTIONS] chat [--sync/--no-sync] [--think] [--sql/--no-sql]
```

- `--no-sync` skips the Pluggy sync at start-up. It is skipped automatically when the Pluggy credentials are missing.
- `--think` turns on Ollama's thinking mode. It is slower, but it uses tools more reliably on small models.
- `--sql/--no-sql` exposes the guarded `run_sql` tool. By default it is on for Claude and off for Ollama.

When the chat starts it shows the banner (version, provider, model and language), then the usual ready line. See
[Look and feel](#look-and-feel).

In-chat commands:

| Command | Effect |
|---|---|
| `/ajuda` or `/help` | Help |
| `/resumo` or `/summary` | Prints the financial snapshot that goes into the system prompt |
| `/sync` | Syncs, reloads the data and starts a new conversation |
| `/reset` | Starts a new conversation |
| `/historico` or `/history` | Shows the conversation history |
| `sair` / `exit` / `quit` / `bye` / `q` (also `/sair`, `/exit`, `/quit`) | Exit. Ctrl+C at the prompt also exits, as does end of input (Ctrl+Z then Enter on Windows, Ctrl+D elsewhere). Ctrl+C while an answer is streaming only interrupts that answer. |

### `server`

```text
llmopenfinance [GLOBAL OPTIONS] server [--host HOST] [--port PORT] [--threads N] [--browser/--no-browser]
                                  [--sync/--no-sync] [--think] [--sql/--no-sql]
```

Serves the chat as a local web page. It needs the optional `web` extra (Flask and Waitress), installed with:

```powershell
uv sync --extra web
llmopenfinance

Without it, `server` prints the install line and exits with status 1 before doing anything else. The global
`--provider`, `--model` and `--db` options apply as in `chat`.

| Option | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | Address to listen on. Anything other than `127.0.0.1`, `localhost` or `::1` prints a warning |
| `--port` | `8000` | TCP port, 1 to 65535 |
| `--threads` | `4` | Waitress worker threads, at least 2, so the page can read its state while an answer or a sync is streaming |
| `--browser/--no-browser` | on | Open the page in the default browser once the server is ready |
| `--sync/--no-sync` | on | Sync Pluggy data first; skipped automatically when the credentials are missing (as in `chat`) |
| `--think` | off | Ollama thinking mode (as in `chat`) |
| `--sql/--no-sql` | on for Claude, off for Ollama | Expose the guarded `run_sql` tool (as in `chat`) |
llmopenfinance
Start-up goes from the cheapest check to the most expensive: the `web` extra, the optional sync, the LLM connection
check (the same messages as `chat`), then the server. It prints the address and, once, the secret token inside it:

```text
Abra no navegador: http://127.0.0.1:8000/?token=<random token>  (Ctrl+C encerra o servidor)
```

For a wildcard host (`0.0.0.0`, `::`) the printed URL uses `127.0.0.1`; IPv6 literals are shown in brackets. There is
no banner (that belongs to `chat`). Ctrl+C stops the server and exits with status 0. A port that is already in use (or
any other bind error) prints one line naming the host and port, and exits with status 1.

llmopenfinancehas the same commands as the chat as buttons (`/help`, `/summary`, `/sync`, `/reset`, `/history`), a sync that
shows per-item progress, and a status line. Slash commands typed in the box work too; `exit` only explains how to leave.

**Security**

- **Loopback by default.** The server is reachable only from this computer. A `--host` outside loopback serves your
  financial data over **plain HTTP**, unencrypted, and `server` prints a prominent warning. For a wildcard bind
  (`0.0.0.0` or `::`), the client's `Host` header cannot be predicted, so the check below is skipped and the token alone
llmopenfinances access; the warning says so. Use such a bind only on a network you trust.
llmopenfinanceaunch token.** A random secret is created on every start and printed once, in the URL. It is never written to
llmopenfinanceOpening the URL sets an `HttpOnly`, `SameSite=Strict` cookie and redirects to `/`, so the token leaves the
  address bar; every other route needs the cookie. Restarting the server invalidates old links.
- **`Host` check.** Requests whose `Host` header is not `localhost`, `127.0.0.1`, `[::1]` or the bound `--host` (with the
  port) are refused, which blocks DNS-rebinding attacks. `POST` routes accept only `application/json` bodies of up to
  64 KiB.
- **Nothing external.** The page loads no external script, style or font. Answers and data are HTML-escaped, and every
  response has `Cache-Control: no-store` and a restrictive `Content-Security-Policy`. Waitress access logging is off.
- **One conversation per server process.** Every tab shows and shares the same conversation, which is kept in memory
  only (a reload shows its text turns again; restarting the server starts over). Requests are **not queued**: while an
  answer or a sync is running, another chat, command or sync gets a "busy" error instead of waiting. The page's status
  line keeps working.

### `ask`

```text
llmopenfinance ask "Quanto gastei com delivery em outubro?" [--sync] [--sql/--no-sql]
```

Answers one question and exits. Useful in scripts.

### `sync`

```text
llmopenfinance sync [--force]
```

For every registered item, `sync` checks `GET /items/{id}`. It skips the item when `lastUpdatedAt` hasn't changed,
unless you pass `--force`. Otherwise it replaces the item's local data:

- accounts and their **full** transaction history;
- credit-card bills;
- investments and loans.

Failures are reported per item or product, and the sync never stops part-way. If a product fails with a temporary
error (network, 429 or 5xx), the item is not marked as synced, so the next `sync` downloads it again without
needing `--force`.

`sync`, `connect` and `items add/remove` need write access to the database. If another `llmopenfinance` session has
it open, `sync` prints a warning and the chat continues with the local data. `status` and `items list` open the
database read-only.

### `status`

Shows the database path, the last sync, row counts and every connection with its status and consent expiry date. On a
terminal the counts and the connections are tables; piped, they are plain lines (see [Look and feel](#look-and-feel)).

### `connect`

```text
llmopenfinance connect [--sandbox] [--sync/--no-sync]
```

Opens the Pluggy Connect widget on a loopback page. See [pluggy-setup.md](pluggy-setup.md).

### `items`

```text
llmopenfinance items add <id>... [--verify/--no-verify]
llmopenfinance items list
llmopenfinance items remove <id>
```

`add` validates each id with Pluggy when the credentials are available. `remove` also deletes that item's local data.
`list` prints a table on a terminal and one plain line per connection otherwise.

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
