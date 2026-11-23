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

uv run finassist --db smoke.duckdb ask "Quais assinaturas eu tenho?"
uv run finassist --provider ollama --db smoke.duckdb ask "Liste as 3 maiores despesas de novembro de 2026."
```

Expected results for the second question: R$ 300,00 (Loja de Móveis), R$ 250,00 (Mercado Livre) and R$ 210,00
(Amazon), found through a `query_transactions` call. If the model answers without the
`↳ consultando ...` status line, it skipped the tools. That is a regression; see the rule about aggregates in
`agent.system_prompt`.

## Pluggy sandbox (real API, fake bank)

```powershell
uv run finassist --db sandbox.duckdb connect --sandbox   # choose "Pluggy Bank", user-ok / password-ok
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
