"""Tests for the tool registry (validation, execution, error handling, rendering)."""

import json

import pytest

from llmopenfinance.tools import BASE_TOOLS, ToolCall, ToolRegistry, to_json, validate_arguments

SCHEMA = {
    "type": "object",
    "properties": {
        "months": {"type": "integer", "minimum": 1, "maximum": 24},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "kind": {"type": "string", "enum": ["a", "b"]},
        "day": {"type": "string", "format": "date"},
        "text": {"type": "string"},
    },
    "required": ["text"],
}


def test_validate_arguments_coerces_and_drops_unknown():
    args = validate_arguments(
        {
            "months": "3",
            "ratio": "1.5",
            "flag": "true",
            "kind": "a",
            "day": "2026-01-01",
            "text": "x",
            "zzz": 1,
            "kind2": None,
        },
        SCHEMA,
    )
    assert args == {"months": 3, "ratio": 1.5, "flag": True, "kind": "a", "day": "2026-01-01", "text": "x"}


def test_validate_arguments_ignores_none_and_empty():
    assert validate_arguments({"text": "x", "months": None, "kind": ""}, SCHEMA) == {"text": "x"}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({}, "obrigatórios"),
        ({"text": "x", "months": "abc"}, "integer"),
        ({"text": "x", "months": 0}, "minimum"),
        ({"text": "x", "months": 99}, "maximum"),
        ({"text": "x", "months": True}, "integer"),
        ({"text": "x", "kind": "c"}, "um de"),
        ({"text": "x", "day": "01/02/2026"}, "AAAA-MM-DD"),
        ({"text": 5}, "string"),
    ],
)
def test_validate_arguments_errors(args, message):
    with pytest.raises(ValueError, match=message):
        validate_arguments(args, SCHEMA)


def test_to_json_truncates():
    assert to_json({"a": 1}) == '{"a":1}'
    long = to_json(["x" * 50] * 10, max_chars=40)
    assert long.endswith('refine os filtros]"')


@pytest.fixture
def registry(con):
    return ToolRegistry(lambda: con)


def test_registry_specs_render_for_both_providers(registry):
    assert "run_sql" in registry.names
    assert registry.names == sorted(registry.names)
    anthropic = registry.anthropic_tools()
    ollama = registry.ollama_tools()
    assert {t["name"] for t in anthropic} == set(registry.names)
    assert all("input_schema" in t for t in anthropic)
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in ollama)


def test_registry_without_sql(con):
    names = ToolRegistry(lambda: con, enable_sql=False).names
    assert "run_sql" not in names
    assert len(names) == len(BASE_TOOLS)


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_accounts", {}),
        ("spending_by_category", {"months": 2, "category": "compras"}),
        ("query_transactions", {"search": "netflix", "limit": 2}),
        ("detect_subscriptions", {"months": 6}),
        ("cashflow_summary", {"months": 2}),
        ("top_merchants", {"months": 2, "limit": 2}),
        ("credit_card_bills", {"limit": 1}),
        ("get_investments", {}),
        ("get_loans", {}),
        ("run_sql", {"sql": "SELECT count(*) AS n FROM v_transactions"}),
    ],
)
def test_every_tool_runs(registry, name, args):
    result = registry.run(ToolCall("c1", name, args))
    assert not result.is_error, result.content
    json.loads(result.content)


def test_unknown_tool(registry):
    result = registry.run(ToolCall("c", "nope", {}))
    assert result.is_error
    assert "desconhecida" in result.content


def test_invalid_arguments_return_error(registry):
    result = registry.run(ToolCall("c", "query_transactions", {"kind": "lucro"}))
    assert result.is_error
    assert "inválidos" in result.content


def test_handler_exception_returns_error(registry):
    result = registry.run(ToolCall("c", "run_sql", {"sql": "DROP TABLE accounts"}))
    assert result.is_error
    assert "SELECT" in result.content


def test_repeated_call_is_memoized_with_note(registry):
    first = registry.run(ToolCall("c1", "get_accounts", {}))
    second = registry.run(ToolCall("c2", "get_accounts", {}))
    assert second.call_id == "c2"
    assert second.content.startswith("[Você já chamou")
    assert second.content.endswith(first.content)
    registry.reset_cache()
    assert registry.run(ToolCall("c3", "get_accounts", {})).content == first.content


def test_run_all_preserves_order(registry):
    results = registry.run_all([ToolCall("a", "get_loans", {}), ToolCall("b", "get_investments", {})])
    assert [r.call_id for r in results] == ["a", "b"]


def test_list_argument_returns_error_instead_of_raising(registry):
    """Regression: an unhashable argument value raised TypeError out of ToolRegistry.run."""
    result = registry.run(ToolCall("c", "spending_by_category", {"months": [3], "category": {"x": 1}}))
    assert result.is_error
    assert "inválidos" in result.content


def test_validate_arguments_rejects_container_for_scalar():
    with pytest.raises(ValueError, match="string"):
        validate_arguments({"text": ["a"]}, SCHEMA)
