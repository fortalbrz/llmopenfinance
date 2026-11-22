"""Tools the LLM can call to query the user's financial data.

One registry renders the same specs for both providers (Anthropic ``input_schema`` and Ollama
``function.parameters``). :meth:`ToolRegistry.run` never raises: invalid arguments and handler
errors come back as ``is_error`` results with an actionable message so the model can self-correct.
Tool descriptions are static (stable prompt-cache prefix); results are compact JSON, row-capped.
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
            raise ValueError(f"'{name}' deve ser {expected}") from exc
    if expected == "boolean" and isinstance(value, str) and value.lower() in {"true", "false"}:
        value = value.lower() == "true"
    if expected in _JSON_TYPES:
        allowed = _JSON_TYPES[expected]
        if not isinstance(value, allowed) or (expected != "boolean" and isinstance(value, bool)):
            raise ValueError(f"'{name}' deve ser {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"'{name}' deve ser um de {list(schema['enum'])}")
    if schema.get("format") == "date":
        try:
            date.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(f"'{name}' deve estar no formato AAAA-MM-DD") from exc
    for bound, check in (("minimum", lambda v, b: v >= b), ("maximum", lambda v, b: v <= b)):
        if bound in schema and isinstance(value, int | float) and not check(value, schema[bound]):
            raise ValueError(f"'{name}' fora do intervalo ({bound} {schema[bound]})")
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
        raise ValueError(f"argumentos obrigatórios ausentes: {missing}")
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
    return text[:max_chars] + '…"[resultado truncado — refine os filtros]"'


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_MONTHS = {"type": "integer", "minimum": 1, "maximum": 24, "description": "Quantidade de meses (inclui o atual)."}

SQL_SCHEMA_DOC = """Views/tabelas (DuckDB, somente leitura):
- v_transactions(id, date DATE, month 'AAAA-MM', description, merchant, merchant_key, category (pt-BR local),
  category_pluggy, kind ['expense','refund','income','transfer','investment'], flow DOUBLE (negativo = saída),
  spend DOUBLE (gasto líquido: despesas positivas, estornos negativos, 0 para outros), currency_code, type
  ['DEBIT','CREDIT'], status, payment_method, installment_number, total_installments, account_id, account_name,
  account_type ['BANK','CREDIT'], account_subtype, item_id)
- accounts(id, item_id, type, subtype, name, marketing_name, number, balance, currency_code, credit_limit,
  available_credit_limit, balance_close_date, balance_due_date, minimum_payment)
- bills(id, account_id, due_date, closing_date, total_amount, currency_code, minimum_payment_amount)
- investments(id, item_id, name, type, subtype, balance, amount_profit, currency_code, annual_rate,
  last_twelve_months_rate, due_date, issuer, status)
- loans(id, item_id, product_name, type, contract_amount, outstanding_balance, currency_code, total_installments,
  paid_installments, due_installments, installment_amount, due_date, cet)
- items(id, connector_name, status, execution_status, last_updated_at, consent_expires_at, synced_at)"""


def _spending(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.spending_by_category(con, months=args.get("months", 3), category=args.get("category"))


def _transactions(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.query_transactions(con, **args)


def _subscriptions(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.detect_subscriptions(
        con, months=args.get("months", 13), include_inactive=args.get("include_inactive", False)
    )


def _cashflow(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.cashflow(con, months=args.get("months", 6))


def _merchants(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.top_merchants(con, months=args.get("months", 3), limit=args.get("limit", 10))


def _sql(con: duckdb.DuckDBPyConnection, args: dict[str, Any]) -> Any:
    return analytics.run_select(con, args["sql"], limit=TOOL_ROW_LIMIT)


BASE_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "get_accounts",
        "Lista contas bancárias e cartões com saldo, limite, vencimento e status da conexão. "
        "Use para perguntas sobre saldo, limite disponível ou quais contas existem.",
        {"type": "object", "properties": {}},
        lambda con, _args: analytics.accounts_overview(con),
    ),
    ToolSpec(
        "spending_by_category",
        "Gasto líquido mensal (despesas menos estornos) por categoria. Use para 'quanto gasto com X por mês', "
        "ex.: categoria 'Compras online', 'Delivery de comida', 'Streaming'. Sem categoria retorna todas.",
        {
            "type": "object",
            "properties": {
                "months": _MONTHS,
                "category": {"type": "string", "description": "Trecho do nome da categoria (pt-BR ou inglês)."},
            },
        },
        _spending,
    ),
    ToolSpec(
        "query_transactions",
        "Busca lançamentos individuais com filtros e devolve também contagem e somas de TODOS os que casam. "
        "Use para detalhar gastos, procurar um estabelecimento (search) ou verificar valores.",
        {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "format": "date", "description": "AAAA-MM-DD inclusive."},
                "date_to": {"type": "string", "format": "date", "description": "AAAA-MM-DD inclusive."},
                "category": {"type": "string", "description": "Trecho do nome da categoria."},
                "search": {"type": "string", "description": "Trecho da descrição/estabelecimento, ex.: 'amazon'."},
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
        "Detecta assinaturas e cobranças recorrentes (streaming, apps, telefonia, academia, seguros...) com valor "
        "médio, frequência, última cobrança e custo mensal/anual estimado. Use para 'quais assinaturas eu tenho'.",
        {
            "type": "object",
            "properties": {
                "months": _MONTHS,
                "include_inactive": {"type": "boolean", "description": "Incluir as que parecem canceladas."},
            },
        },
        _subscriptions,
    ),
    ToolSpec(
        "cashflow_summary",
        "Receitas, gastos, investimento líquido, saldo e taxa de poupança por mês. "
        "Use para avaliar a saúde financeira e sugerir melhorias.",
        {"type": "object", "properties": {"months": _MONTHS}},
        _cashflow,
    ),
    ToolSpec(
        "top_merchants",
        "Estabelecimentos/favorecidos com maior gasto no período.",
        {
            "type": "object",
            "properties": {"months": _MONTHS, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
        },
        _merchants,
    ),
    ToolSpec(
        "credit_card_bills",
        "Faturas de cartão de crédito recentes (valor total, vencimento, pagamento mínimo).",
        {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 24}}},
        lambda con, args: analytics.bills_overview(con, limit=args.get("limit", 6)),
    ),
    ToolSpec(
        "get_investments",
        "Posições de investimento, total por tipo, rentabilidade e vencimentos.",
        {"type": "object", "properties": {}},
        lambda con, _args: analytics.investments_overview(con),
    ),
    ToolSpec(
        "get_loans",
        "Empréstimos e financiamentos: saldo devedor, parcelas pagas/restantes, valor da parcela e CET.",
        {"type": "object", "properties": {}},
        lambda con, _args: analytics.loans_overview(con),
    ),
)

SQL_TOOL = ToolSpec(
    "run_sql",
    "Executa UMA consulta SELECT (DuckDB) somente leitura quando as outras ferramentas não bastam. "
    f"Máximo {TOOL_ROW_LIMIT} linhas.\n{SQL_SCHEMA_DOC}",
    {
        "type": "object",
        "properties": {"sql": {"type": "string", "description": "Uma única instrução SELECT."}},
        "required": ["sql"],
    },
    _sql,
)


class ToolRegistry:
    """Executes tool calls against a (read-only) DuckDB connection.

    Args:
        connection: Callable returning the current connection (it changes after ``/sync``).
        enable_sql: Expose the ``run_sql`` escape hatch.
    """

    def __init__(self, connection: Callable[[], duckdb.DuckDBPyConnection], *, enable_sql: bool = True) -> None:
        self._connection = connection
        specs = [*BASE_TOOLS, SQL_TOOL] if enable_sql else list(BASE_TOOLS)
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
            return ToolResult(call.id, call.name, f"Ferramenta desconhecida '{call.name}'. Use: {self.names}", True)
        key = f"{call.name}:{json.dumps(dict(call.arguments), sort_keys=True, default=str)}"
        if key in self._cache:
            cached = self._cache[key]
            note = "[Você já chamou esta ferramenta com os mesmos argumentos; use o resultado abaixo.] "
            return ToolResult(call.id, call.name, note + cached.content, cached.is_error)
        try:
            args = validate_arguments(call.arguments, spec.parameters)
        except (ValueError, TypeError) as exc:
            return ToolResult(call.id, call.name, f"Argumentos inválidos: {exc}", True)
        try:
            content = to_json(spec.handler(self._connection(), args))
            result = ToolResult(call.id, call.name, content)
        except Exception as exc:
            logger.warning("Tool %s failed: %s", call.name, exc, exc_info=True)
            result = ToolResult(call.id, call.name, f"Erro ao executar {call.name}: {exc}", True)
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
