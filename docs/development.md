# Development

## Setup and checks

```powershell
uv sync
uv run ruff format . ; uv run ruff check . ; uv run pyrefly check ; uv run pytest
```

## Live smoke test (synthetic data, real LLM)

The unit tests mock every network call. To check that a real model drives the tool loop correctly, use a database
filled with the synthetic test dataset. This never touches your real bank data:

```powershell
uv run python -c "from tests.conftest import populate; from llmopenfinance.store import FinanceStore; s = FinanceStore('smoke.duckdb'); populate(s); s.close()"

uv run llmopenfinance --db smoke.duckdb ask "Quais assinaturas eu tenho?"
uv run llmopenfinance --provider ollama --db smoke.duckdb ask "Liste as 3 maiores despesas de novembro de 2026."
```

Expected results for the second question: R$ 300,00 (Loja de Móveis), R$ 250,00 (Mercado Livre) and R$ 210,00
(Amazon), found through a `query_transactions` call. If the model answers without the
`↳ consultando ...` status line, it skipped the tools. That is a regression; see the rule about aggregates in
`agent.system_prompt`.

## Pluggy sandbox (real API, fake bank)

```powershell
uv run llmopenfinance --db sandbox.duckdb connect --sandbox   # choose "Pluggy Bank", user-ok / password-ok
```

Sandbox items include a checking account (17 transactions, recurring bills) and a credit card (subscriptions and
installments). They don't count toward item limits and are deleted after 30 days without updates.

## Adding a tool

1. Write a pure query function in `analytics.py` that takes a DuckDB connection and returns JSON-able data.
2. Add a `ToolSpec` in `tools.build_base_tools()`; `ToolRegistry` builds its specs from that function.
   `tools.BASE_TOOLS` is only a lazy compatibility attribute, so don't register anything there. Take the description
   and the parameter descriptions from catalog keys (`tool.<name>.description` and so on) and add those keys to both
   `messages_pt.py` and `messages_en.py`. The description must say *when* to call it. Use a JSON Schema with
   `enum`, `format` and min/max wherever they apply.
3. Add tests in `test_analytics.py` (exact values on the synthetic dataset) and add the tool to
   `test_tools.py::test_every_tool_runs`.
4. Keep tool lists and descriptions static. Claude's prompt cache depends on a stable prefix.


## Web page smoke test (synthetic data)

A manual check of the whole web chat, with the synthetic dataset and the `web` extra installed (`uv sync --extra web`):

```powershell
uv run python -c "from tests.conftest import populate; from llmopenfinance.store import FinanceStore; s = FinanceStore('smoke.duckdb'); populate(s); s.close()"
uv run finassist --db smoke.duckdb server --no-sync          # add --provider ollama to stay local
```

The command prints `http://127.0.0.1:8000/?token=...` and opens it. Then check:

1. The page loads, the address bar shows `/` without the token, and the status line names the provider, model, language, last sync and data counts.
2. Type *Quais assinaturas eu tenho?*: the answer streams in, with a `consultando ...` row for each tool call.
3. Click **Help**, **Summary**, **History** and **Reset conversation**: each shows its text, and Reset empties the conversation.
4. Reload the tab: the text turns reappear. Open a second tab on the printed URL: it shows the same conversation.
5. While an answer is streaming, send another message from the second tab: it is refused as busy, and the status line still updates.
6. Open `http://127.0.0.1:8000/` in a private window (no cookie): the reply is a JSON "access denied", not the page. `curl -H "Host: evil.example" http://127.0.0.1:8000/` gets 400.
7. Press Ctrl+C in the terminal: the server stops with a goodbye line. Start a second server on the same port while the first runs: one error line naming host and port, exit status 1.
8. `LLMOPENFINANCE_LANG=en` shows the page and the messages in English. `--host 0.0.0.0` prints the plain-HTTP warning.

Delete `smoke.duckdb` afterwards.


## Web server notes

The `web` extra (`uv sync --extra web`, or `pip install "llmopenfinance[web]"`) installs Flask 3.1.3 and
Waitress 3.0.2 (versions pinned in `uv.lock`). Both are also in the dev group so the tests can use them.

Streaming check: Waitress does flush chunked NDJSON lines as the application yields them. A throwaway Flask app
(`POST /slow`, `application/x-ndjson`, five JSON lines yielded one second apart with `time.sleep(1)` inside a
`flask.stream_with_context` generator) was served with `waitress.serve(app, host="127.0.0.1", port=8765)` and read with:

```bash
curl -sN -X POST http://127.0.0.1:8765/slow | while IFS= read -r l; do date +%T.%N | cut -c1-12; echo "$l"; done
```

Observed arrival times: 00:17:01.680, 00:17:02.648, 00:17:03.645, 00:17:04.650 and 00:17:05.645, so the five lines
arrived about one second apart, not all at the end. The response headers were `HTTP/1.1 200 OK`,
`Transfer-Encoding: chunked` and `Server: waitress`. The chat page can therefore show answers progressively.
The page also works if a proxy or client buffers the stream; the answer then appears all at once.

Client disconnect: `curl --max-time 2.5` was stopped after three lines. The server logged nothing and the next
request, sent immediately, was served normally with all five lines, so an abandoned stream does not block the
server. (The spike's generator has no lock; releasing the chat lock and rolling back the history on disconnect
is the job of the web code and is tested separately.)
