"""Default configuration constants for llmopenfinance.

Every value can be overridden by a CLI option or an environment variable;
see ``docs/usage.md`` for the full list.
"""

from datetime import timedelta, timezone
from pathlib import Path

# --- LLM providers -----------------------------------------------------------
DEFAULT_PROVIDER = "claude"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5-5"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_SERVER = "http://localhost:11434"
DEFAULT_OLLAMA_NUM_CTX = 16384
DEFAULT_OLLAMA_KEEP_ALIVE = "30m"
CLAUDE_API_URL = "https://api.anthropic.com"
CLAUDE_API_VERSION = "2023-06-01"
CLAUDE_MAX_TOKENS = 64000

# --- Pluggy ------------------------------------------------------------------
PLUGGY_API_URL = "https://api.pluggy.ai"
# The URL in Pluggy's dashboard sample (cdn.pluggy.ai/connect/v2/...) returns 404; pin a published build.
PLUGGY_CONNECT_JS = "https://cdn.pluggy.ai/pluggy-connect/v2.11.0/pluggy-connect.js"
MEU_PLUGGY_CONNECTOR_ID = 200
PLUGGY_API_KEY_TTL_SECONDS = 110 * 60  # API keys live 2 h; refresh with a 10 min margin
PLUGGY_PAGE_SIZE = 500

# --- Environment variable names ----------------------------------------------
ENV_PLUGGY_CLIENT_ID = "PLUGGY_USERNAME"
ENV_PLUGGY_CLIENT_SECRET = "PLUGGY_PASSWORD"
ENV_PLUGGY_ITEM_IDS = "PLUGGY_ITEM_IDS"
ENV_DB_PATH = "LLMOPENFINANCE_DB"
ENV_PROVIDER = "LLMOPENFINANCE_PROVIDER"
ENV_LANG = "LLMOPENFINANCE_LANG"
ENV_ANTHROPIC_KEYS = ("ANTHROPIC_API_KEY", "CLAUDE_TOKEN")

# --- Language ----------------------------------------------------------------
DEFAULT_LANGUAGE = "pt-BR"
SUPPORTED_LANGUAGES = ("pt-BR", "en")

# --- Web chat (``finassist server``) -----------------------------------------
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8000
DEFAULT_WEB_THREADS = 4

# --- Local storage -----------------------------------------------------------
DEFAULT_DB_PATH = Path.home() / ".llmopenfinance" / "finance.duckdb"

# --- Assistant behaviour -----------------------------------------------------
MAX_TOOL_ROUNDS = 6
TOOL_ROW_LIMIT = 100
TOOL_RESULT_MAX_CHARS = 12000
SQL_TIMEOUT_SECONDS = 5.0

# Brazil has had no daylight-saving time since 2019 — a fixed offset avoids needing tzdata on Windows.
BRT = timezone(timedelta(hours=-3), "BRT")
