# CLAUDE.md

All collaboration guidelines for this project live in **[AGENTS.md](AGENTS.md)**.

The configs own the tool-enforced conventions. Defer to them and do not restate their rules in prose:

- [`ruff.toml`](ruff.toml) owns formatting, lint, import order, naming, Google-style docstrings, lazy `%`-logging
  and the security rules.
- [`pyproject.toml`](pyproject.toml) owns pytest discovery, the coverage gate (`--cov-fail-under=90`) and pyrefly.

## CI failure fix order

`uv run ruff format .` → `uv run ruff check . --fix` → `uv run pyrefly check` → `uv run pytest -x`.
