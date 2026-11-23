"""Tools the LLM can call to query the user's financial data.

One registry renders the same specs for both providers (Anthropic ``input_schema`` and Ollama
``function.parameters``). :meth:`ToolRegistry.run` never raises: invalid arguments and handler
errors come back as ``is_error`` results with an actionable message so the model can self-correct.
Tool descriptions are static (stable prompt-cache prefix); results are compact JSON, row-capped.
Descriptions follow the process language: :func:`build_base_tools` and :func:`build_sql_tool` build them on
demand, and ``BASE_TOOLS`` / ``SQL_TOOL`` are module attributes resolved on access, never at import.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import duckdb

from llmopenfinance import analytics
from llmopenfinance.constants import TOOL_RESULT_MAX_CHARS, TOOL_ROW_LIMIT
from llmopenfinance.i18n import t

logger = logging.getLogger(__name__)

Handler = Callable[[duckdb.DuckDBPyConnection, dict[str, Any]], Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model.

    Attributes:
        id: Provider call id (synthesized for Ollama).
        name: Tool name.
        arguments: Parsed JSON arguments.
    """

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a :class:`ToolCall`.

    Attributes:
        call_id: Id of the originating call.
        name: Tool name.
        content: JSON (or error text) returned to the model.
        is_error: Whether the call failed.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """Provider-agnostic tool definition.

    Attributes:
        name: Tool name.
        description: When/why to call it (shown to the model).
        parameters: JSON Schema of the arguments (``type: object``).
        handler: Pure function ``(connection, args) -> JSON-able result``.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: Handler

    def anthropic(self) -> dict[str, Any]:
        """Render for the Anthropic Messages API.

        Returns:
            dict[str, Any]: ``{"name", "description", "input_schema"}``.
        """
        return {"name": self.name, "description": self.description, "input_schema": dict(self.parameters)}

    def ollama(self) -> dict[str, Any]:
        """Render for Ollama ``/api/chat``.

        Returns:
            dict[str, Any]: ``{"type": "function", "function": {...}}``.
        """
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": dict(self.parameters)},
        }


# ---------------------------------------------------------------------------
# Argument validation (small subset of JSON Schema — no extra dependency)
# ---------------------------------------------------------------------------

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
}


def _coerce(name: str, value: Any, schema: Mapping[str, Any]) -> Any:
    """Validate (and leniently coerce) one argument.

    Small local models often send numbers as strings; those are coerced when unambiguous.

    Args:
        name: Argument name (for messages).
        value: Raw value.
        schema: Property schema.

    Returns:
        Any: Validated value.

    Raises:
        ValueError: When the value does not match the schema.
    """
    expected = schema.get("type")
    if expected in {"integer", "number"} and isinstance(value, str):
        try:
            value = int(value) if expected == "integer" else float(value)
        except ValueError as exc:
            raise ValueError(t("tool.error.must_be_type", name=name, expected=expected)) from exc
    if expected == "boolean" and isinstance(value, str) and value.lower() in {"true", "false"}:
        value = value.lower() == "true"
    if expected in _JSON_TYPES:
        allowed = _JSON_TYPES[expected]
        if not isinstance(value, allowed) or (expected != "boolean" and isinstance(value, bool)):
            raise ValueError(t("tool.error.must_be_type", name=name, expected=expected))
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(t("tool.error.must_be_one_of", name=name, options=list(schema["enum"])))
    if schema.get("format") == "date":
        try:
            date.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(t("tool.error.bad_date", name=name)) from exc
    for bound, check in (("minimum", lambda v, b: v >= b), ("maximum", lambda v, b: v <= b)):
        if bound in schema and isinstance(value, int | float) and not check(value, schema[bound]):
            raise ValueError(t("tool.error.out_of_range", name=name, bound=bound, limit=schema[bound]))
    return value


def validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    """Validate tool arguments against a JSON Schema subset.

    Unknown arguments are dropped; ``None`` values are treated as absent.

    Args:
        arguments: Arguments sent by the model.
        schema: Tool parameter schema.

    Returns:
        dict[str, Any]: Clean arguments.

    Raises:
        ValueError: On a missing required argument or an invalid value.
    """
    properties: Mapping[str, Any] = schema.get("properties", {})
    clean: dict[str, Any] = {}
    for name, value in arguments.items():
        if name in properties and value is not None and not (isinstance(value, str) and not value):
            clean[name] = _coerce(name, value, properties[name])
    missing = [name for name in schema.get("required", []) if name not in clean]
    if missing:
        raise ValueError(t("tool.error.missing_arguments", missing=missing))
    return clean


def to_json(value: Any, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    """Serialize a tool result compactly, truncating very large payloads.

    Args:
        value: JSON-able value (dates/decimals are stringified).
        max_chars: Maximum characters.

    Returns:
        str: JSON text (with a truncation marker when cut).
    """
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + t("tool.truncated")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def _months() -> dict[str, Any]:
    """Build the shared ``months`` parameter schema with its description in the process language.

    Returns:
        dict[str, Any]: A fresh ``{"type": "integer", ...}`` property schema.
    """
    return {"type": "integer", "minimum": 1, "maximum": 24, "description": t("tool.param.months")}


def sql_schema_doc() -> str:
    """Describe the queryable views and tables for the ``run_sql`` tool, in the process language.

    Identifiers (tables, columns, enum values) are the same in every language; only the prose changes.

    Returns:
        str: The multi-line schema documentation.
    """
    return t("tool.sql_schema")


def _spending(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.spending_by_category(con, months=args.get("months", 3), category=args.get("category"))


def _transactions(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.query_transactions(con, **args)


def _subscriptions(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    detected = analytics.detect_subscriptions(
        con, months=args.get("months", 13), include_inactive=args.get("include_inactive", False)
    )
    return analytics.localize_subscriptions(detected)


def _cashflow(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.cashflow(con, months=args.get("months", 6))


def _merchants(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.top_merchants(con, months=args.get("months", 3), limit=args.get("limit", 10))


def _sql(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.run_select(con, args["sql"], limit=TOOL_ROW_LIMIT)


def build_base_tools() -> tuple[ToolSpec, ...]:
    """Build the tool specs with descriptions in the process language.

    Tool names, parameter names, enums and JSON-schema structure never depend on the language.

    Returns:
        tuple[ToolSpec, ...]: The base tools (everything except ``run_sql``).
    """
    return (
        ToolSpec(
            "get_accounts",
            t("tool.get_accounts.description"),
            {"type": "object", "properties": {}},
            lambda con, _args: analytics.accounts_overview(con),
        ),
        ToolSpec(
            "spending_by_category",
            t("tool.spending_by_category.description"),
            {
                "type": "object",
                "properties": {
                    "months": _months(),
                    "category": {"type": "string", "description": t("tool.spending_by_category.category")},
                },
            },
            _spending,
        ),
        ToolSpec(
            "query_transactions",
            t("tool.query_transactions.description"),
            {
                "type": "object",
                "properties": {
                    "date_from": {"type": "string", "format": "date", "description": t("tool.query_transactions.date")},
                    "date_to": {"type": "string", "format": "date", "description": t("tool.query_transactions.date")},
                    "category": {"type": "string", "description": t("tool.query_transactions.category")},
                    "search": {"type": "string", "description": t("tool.query_transactions.search")},
                    "kind": {"type": "string", "enum": list(analytics.KINDS)},
                    "account_type": {"type": "string", "enum": ["BANK", "CREDIT"]},
                    "min_amount": {"type": "number", "minimum": 0},
                    "max_amount": {"type": "number", "minimum": 0},
                    "order": {"type": "string", "enum": ["date_desc", "date_asc", "amount_desc"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": TOOL_ROW_LIMIT},
                },
            },
            _transactions,
        ),
        ToolSpec(
            "detect_subscriptions",
            t("tool.detect_subscriptions.description"),
            {
                "type": "object",
                "properties": {
                    "months": _months(),
                    "include_inactive": {"type": "boolean", "description": t("tool.detect_subscriptions.inactive")},
                },
            },
            _subscriptions,
        ),
        ToolSpec(
            "cashflow_summary",
            t("tool.cashflow_summary.description"),
            {"type": "object", "properties": {"months": _months()}},
            _cashflow,
        ),
        ToolSpec(
            "top_merchants",
            t("tool.top_merchants.description"),
            {
                "type": "object",
                "properties": {"months": _months(), "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            },
            _merchants,
        ),
        ToolSpec(
            "credit_card_bills",
            t("tool.credit_card_bills.description"),
            {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 24}}},
            lambda con, args: analytics.bills_overview(con, limit=args.get("limit", 6)),
        ),
        ToolSpec(
            "get_investments",
            t("tool.get_investments.description"),
            {"type": "object", "properties": {}},
            lambda con, _args: analytics.investments_overview(con),
        ),
        ToolSpec(
            "get_loans",
            t("tool.get_loans.description"),
            {"type": "object", "properties": {}},
            lambda con, _args: analytics.loans_overview(con),
        ),
    )


def build_sql_tool() -> ToolSpec:
    """Build the ``run_sql`` escape-hatch spec with its description (and schema doc) in the process language.

    Returns:
        ToolSpec: The ``run_sql`` tool.
    """
    return ToolSpec(
        "run_sql",
        t("tool.run_sql.description", limit=TOOL_ROW_LIMIT, schema=sql_schema_doc()),
        {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": t("tool.run_sql.sql")}},
            "required": ["sql"],
        },
        _sql,
    )


def __getattr__(name: str) -> Any:
    """Serve ``BASE_TOOLS`` and ``SQL_TOOL`` (PEP 562), built fresh on every access.

    They are not assigned at import: building them calls :func:`~llmopenfinance.i18n.t`, which resolves the
    language, so an invalid ``LLMOPENFINANCE_LANG`` would raise when ``tools`` (and ``agent``, ``chat``) is
    imported. Building on access also means they can never be stale after a language change.

    Args:
        name: Attribute looked up on the module.

    Returns:
        Any: ``tuple[ToolSpec, ...]`` for ``BASE_TOOLS``, a :class:`ToolSpec` for ``SQL_TOOL``.

    Raises:
        AttributeError: For any other name.
    """
    if name == "BASE_TOOLS":
        return build_base_tools()
    if name == "SQL_TOOL":
        return build_sql_tool()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ToolRegistry:
    """Executes tool calls against a (read-only) DuckDB connection.

    Args:
        connection: Callable returning the current connection (it changes after ``/sync``).
        enable_sql: Expose the ``run_sql`` escape hatch.
    """

    def __init__(self, connection: Callable[[], duckdb.DuckDBPyConnection], *, enable_sql: bool = True) -> None:
        self._connection = connection
        specs = [*build_base_tools(), build_sql_tool()] if enable_sql else list(build_base_tools())
        self.specs: tuple[ToolSpec, ...] = tuple(sorted(specs, key=lambda s: s.name))
        self._by_name = {spec.name: spec for spec in self.specs}
        self._cache: dict[str, ToolResult] = {}

    @property
    def names(self) -> list[str]:
        """Registered tool names."""
        return [spec.name for spec in self.specs]

    def anthropic_tools(self) -> list[dict[str, Any]]:
        """Tool list for the Anthropic API.

        Returns:
            list[dict[str, Any]]: Tool definitions.
        """
        return [spec.anthropic() for spec in self.specs]

    def ollama_tools(self) -> list[dict[str, Any]]:
        """Tool list for Ollama.

        Returns:
            list[dict[str, Any]]: Tool definitions.
        """
        return [spec.ollama() for spec in self.specs]

    def reset_cache(self) -> None:
        """Forget memoized results (call once per user question or after a sync)."""
        self._cache.clear()

    def run(self, call: ToolCall) -> ToolResult:
        """Execute one tool call. Never raises.

        Repeated identical calls in the same question return the memoized result plus a note,
        which stops small models from looping on the same tool.

        Args:
            call: The call requested by the model.

        Returns:
            ToolResult: JSON result or error message.
        """
        spec = self._by_name.get(call.name)
        if spec is None:
            return ToolResult(call.id, call.name, t("tool.error.unknown", name=call.name, names=self.names), True)
        key = f"{call.name}:{json.dumps(dict(call.arguments), sort_keys=True, default=str)}"
        if key in self._cache:
            cached = self._cache[key]
            return ToolResult(call.id, call.name, t("tool.repeated_note") + cached.content, cached.is_error)
        try:
            args = validate_arguments(call.arguments, spec.parameters)
        except (ValueError, TypeError) as exc:
            return ToolResult(call.id, call.name, t("tool.error.invalid_arguments", error=exc), True)
        try:
            content = to_json(spec.handler(self._connection(), args))
            result = ToolResult(call.id, call.name, content)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", call.name, exc, exc_info=True)
            result = ToolResult(call.id, call.name, t("tool.error.run_failed", name=call.name, error=exc), True)
        self._cache[key] = result
        return result

    def run_all(self, calls: Sequence[ToolCall]) -> list[ToolResult]:
        """Execute several calls in order.

        Args:
            calls: Calls from one model turn.

        Returns:
            list[ToolResult]: Results in the same order.
        """
        return [self.run(call) for call in calls]
