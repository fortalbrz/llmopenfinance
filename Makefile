# Common dev tasks. Every recipe goes through `uv` (AGENTS.md: uv only), and the
# recipes avoid shell-specific syntax so they work with GNU make under both
# cmd.exe and sh (Git Bash / WSL / macOS / Linux).
#
# Pass CLI arguments with ARGS, e.g.:
#   make run ARGS="--provider ollama"
#   make run ARGS="items add <item-id>"

UV   ?= uv
ARGS ?=

# Removed by `make clean` (paths are globbed relative to the repo root).
CLEAN_DIRS  = build dist htmlcov .pytest_cache .ruff_cache src/*.egg-info src/**/__pycache__ tests/**/__pycache__
CLEAN_FILES = .coverage .coverage.* coverage.xml

.DEFAULT_GOAL := help

.PHONY: help venv build tests test lint format typecheck check run clean distclean

help: ## Show this help
	@echo Targets:
	@echo   venv       Create .venv and install runtime + dev dependencies (uv sync)
	@echo   build      Build sdist and wheel into dist/
	@echo   tests      Run the test suite with the coverage gate (live tests deselected)
	@echo   lint       ruff format check + ruff lint
	@echo   format     Auto-format and auto-fix lint (ruff format, ruff check --fix)
	@echo   typecheck  pyrefly check
	@echo   check      Full CI order: format, lint fix, typecheck, tests
	@echo   run        Run the finassist CLI; pass arguments with ARGS="..."
	@echo   clean      Remove build artifacts and caches (keeps .venv)
	@echo   distclean  clean + remove .venv

venv: ## Create .venv and install all dependency groups
	$(UV) sync

build: ## Build sdist + wheel into dist/
	$(UV) build

tests: ## Run pytest (coverage gate from pyproject.toml)
	$(UV) run pytest -m "not live"

test: tests

lint: ## Check formatting and lint without modifying files
	$(UV) run ruff format --check .
	$(UV) run ruff check .

format: ## Auto-format and auto-fix lint
	$(UV) run ruff format .
	$(UV) run ruff check . --fix

typecheck: ## Static type check
	$(UV) run pyrefly check

check: format typecheck ## CI failure fix order: format -> lint fix -> pyrefly -> pytest -x
	$(UV) run pytest -x -m "not live"

run: ## Run the main CLI, e.g. make run ARGS="--provider ollama"
	$(UV) run finassist $(ARGS)

clean: ## Remove build artifacts and caches
	$(UV) run --no-project python -c "import pathlib, shutil; r = pathlib.Path('.'); [shutil.rmtree(p, ignore_errors=True) for g in '$(CLEAN_DIRS)'.split() for p in r.glob(g) if p.is_dir()]; [p.unlink() for g in '$(CLEAN_FILES)'.split() for p in r.glob(g) if p.is_file()]"

distclean: clean ## clean + remove the virtualenv
	$(UV) run --no-project --isolated python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
