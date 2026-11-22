"""Tests for the tool registry (validation, execution, error handling, rendering)."""

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from llmopenfinance import analytics
from llmopenfinance.constants import TOOL_ROW_LIMIT
from llmopenfinance.tools import (
    BASE_TOOLS,
    SQL_TOOL,
    ToolCall,
    ToolRegistry,
    ToolSpec,
    build_base_tools,
    build_sql_tool,
    sql_schema_doc,
    to_json,
    validate_arguments,
)
from tests.conftest import TODAY
from tests.test_catalog import accented_values

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


# ---------------------------------------------------------------------------
# Localisation (descriptions, error texts, truncation marker)
# ---------------------------------------------------------------------------


def _all_specs() -> list[ToolSpec]:
    return [*build_base_tools(), build_sql_tool()]


def _texts(specs: Sequence[ToolSpec]) -> dict[str, str]:
    """Collect every description a spec shows the model: the tool's own and each parameter's."""
    texts: dict[str, str] = {}
    for spec in specs:
        texts[spec.name] = spec.description
        for prop, schema in spec.parameters.get("properties", {}).items():
            if "description" in schema:
                texts[f"{spec.name}.{prop}"] = schema["description"]
    return texts


def _shape(spec: ToolSpec) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Name, top-level schema keys and every property schema, with all description text removed."""
    properties = {
        prop: {key: value for key, value in schema.items() if key != "description"}
        for prop, schema in spec.parameters.get("properties", {}).items()
    }
    top = {key: value for key, value in spec.parameters.items() if key != "properties"}
    return spec.name, top, properties


def test_tool_specs_same_shape_in_both_languages(use_language: Callable[[str], None]):
    pt = _all_specs()
    use_language("en")
    en = _all_specs()
    assert [spec.name for spec in en] == [spec.name for spec in pt]
    assert [_shape(spec) for spec in en] == [_shape(spec) for spec in pt]
    # The comparison above must not pass vacuously: the text really is different in every tool.
    assert all(a.description != b.description for a, b in zip(en, pt, strict=True))
    assert set(_texts(en)) == set(_texts(pt))


def test_tool_descriptions_en_have_no_portuguese_accents(use_language: Callable[[str], None]):
    use_language("en")
    texts = _texts(_all_specs())
    assert len(texts) > 15  # every tool description and every parameter description was scanned
    assert accented_values(texts, frozenset()) == []


def test_tool_texts_pt_keep_original_wording():
    texts = _texts(_all_specs())
    assert texts["get_accounts"].startswith("Lista contas bancárias e cartões com saldo, limite, vencimento")
    assert texts["spending_by_category.months"] == "Quantidade de meses (inclui o atual)."
    assert texts["query_transactions.date_from"] == "AAAA-MM-DD inclusive."
    assert texts["run_sql"].startswith("Executa UMA consulta SELECT (DuckDB) somente leitura")
    assert f"Máximo {TOOL_ROW_LIMIT} linhas.\nViews/tabelas (DuckDB, somente leitura):" in texts["run_sql"]
    assert "month 'AAAA-MM'" in sql_schema_doc()
    assert "category (pt-BR local)" in sql_schema_doc()


def test_months_schema_is_localised_in_every_tool_that_has_it(use_language: Callable[[str], None]):
    use_language("en")
    months = [
        spec.parameters["properties"]["months"]
        for spec in build_base_tools()
        if "months" in spec.parameters["properties"]
    ]
    assert len(months) == 4
    assert all(m["description"] == "Number of months (includes the current one)." for m in months)
    assert all(m["type"] == "integer" and m["minimum"] == 1 and m["maximum"] == 24 for m in months)


def test_sql_schema_doc_en_keeps_identifiers(use_language: Callable[[str], None]):
    pt = sql_schema_doc()
    use_language("en")
    en = sql_schema_doc()
    assert en != pt
    assert re.findall(r"^- (\w+)\(", en, re.MULTILINE) == re.findall(r"^- (\w+)\(", pt, re.MULTILINE)
    assert re.findall(r"\b[a-z]+_[a-z_]+\b", en) == re.findall(r"\b[a-z]+_[a-z_]+\b", pt)
    assert re.findall(r"'([A-Za-z]+)'", en) == re.findall(r"'([A-Za-z]+)'", pt)
    assert "month 'YYYY-MM'" in en
    assert "category (local pt-BR label)" in en
    assert "AAAA" not in en


def test_run_sql_description_en_embeds_row_limit_and_schema(use_language: Callable[[str], None]):
    use_language("en")
    description = build_sql_tool().description
    assert description.startswith("Runs ONE read-only SELECT query (DuckDB)")
    assert f"Maximum {TOOL_ROW_LIMIT} rows.\n{sql_schema_doc()}" in description


def test_module_level_tools_match_the_builders():
    assert [spec.name for spec in BASE_TOOLS] == [spec.name for spec in build_base_tools()]
    assert SQL_TOOL.name == "run_sql"


def test_registry_uses_current_language(con, use_language: Callable[[str], None]):
    pt_descriptions = {spec.name: spec.description for spec in ToolRegistry(lambda: con).specs}
    use_language("en")
    registry = ToolRegistry(lambda: con)  # the module itself was imported under pt-BR
    en_descriptions = {spec.name: spec.description for spec in registry.specs}
    assert en_descriptions == {spec.name: spec.description for spec in _all_specs()}
    assert en_descriptions["get_accounts"].startswith("Lists bank accounts and cards")
    assert pt_descriptions["get_accounts"].startswith("Lista contas bancárias")
    assert en_descriptions.keys() == pt_descriptions.keys()
    assert [t["function"]["description"] for t in registry.ollama_tools()] == [
        spec.description for spec in registry.specs
    ]


def test_registry_without_sql_uses_current_language(con, use_language: Callable[[str], None]):
    use_language("en")
    specs = ToolRegistry(lambda: con, enable_sql=False).specs
    assert "run_sql" not in {spec.name for spec in specs}
    assert accented_values(_texts(specs), frozenset()) == []


def test_tool_error_messages_en(registry, use_language: Callable[[str], None]):
    use_language("en")
    unknown = registry.run(ToolCall("c", "nope", {}))
    assert unknown.is_error
    assert unknown.content == f"Unknown tool 'nope'. Use: {registry.names}"

    invalid = registry.run(ToolCall("c", "query_transactions", {"kind": "profit"}))
    assert invalid.is_error
    assert invalid.content.startswith("Invalid arguments: 'kind' must be one of ['expense'")

    missing = registry.run(ToolCall("c", "run_sql", {}))
    assert missing.is_error
    assert missing.content == "Invalid arguments: missing required arguments: ['sql']"

    failed = registry.run(ToolCall("c", "run_sql", {"sql": "DROP TABLE accounts"}))
    assert failed.is_error
    assert failed.content.startswith("Error running run_sql: ")

    first = registry.run(ToolCall("c1", "get_accounts", {}))
    again = registry.run(ToolCall("c2", "get_accounts", {}))
    assert again.content.startswith("[You already called this tool")
    note = "[You already called this tool with the same arguments; use the result below.] "
    assert again.content == note + first.content


def test_tool_error_messages_pt_unchanged(registry):
    assert registry.run(ToolCall("c", "nope", {})).content == f"Ferramenta desconhecida 'nope'. Use: {registry.names}"
    missing = registry.run(ToolCall("c", "run_sql", {}))
    assert missing.content == "Argumentos inválidos: argumentos obrigatórios ausentes: ['sql']"
    failed = registry.run(ToolCall("c", "run_sql", {"sql": "DROP TABLE accounts"}))
    assert failed.content.startswith("Erro ao executar run_sql: ")
    registry.run(ToolCall("c1", "get_accounts", {}))
    note = "[Você já chamou esta ferramenta com os mesmos argumentos; use o resultado abaixo.] "
    assert registry.run(ToolCall("c2", "get_accounts", {})).content.startswith(note)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({}, "missing required arguments: ['text']"),
        ({"text": "x", "months": "abc"}, "'months' must be integer"),
        ({"text": "x", "months": True}, "'months' must be integer"),
        ({"text": "x", "kind": "c"}, "'kind' must be one of ['a', 'b']"),
        ({"text": "x", "day": "01/02/2026"}, "'day' must be in the format YYYY-MM-DD"),
        ({"text": "x", "months": 0}, "'months' out of range (minimum 1)"),
        ({"text": "x", "months": 99}, "'months' out of range (maximum 24)"),
    ],
)
def test_validate_arguments_errors_en(use_language: Callable[[str], None], args, message):
    use_language("en")
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_arguments(args, SCHEMA)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({}, "argumentos obrigatórios ausentes: ['text']"),
        ({"text": "x", "months": "abc"}, "'months' deve ser integer"),
        ({"text": "x", "kind": "c"}, "'kind' deve ser um de ['a', 'b']"),
        ({"text": "x", "day": "01/02/2026"}, "'day' deve estar no formato AAAA-MM-DD"),
        ({"text": "x", "months": 0}, "'months' fora do intervalo (minimum 1)"),
    ],
)
def test_validate_arguments_errors_pt_unchanged(args, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        validate_arguments(args, SCHEMA)


def test_to_json_truncation_marker_en(use_language: Callable[[str], None]):
    use_language("en")
    assert to_json(["x" * 50] * 10, max_chars=40).endswith('…"[result truncated — refine the filters]"')


def test_to_json_truncation_marker_pt():
    assert to_json(["x" * 50] * 10, max_chars=40).endswith('…"[resultado truncado — refine os filtros]"')


@pytest.mark.parametrize(
    ("language", "present", "absent"),
    [
        ("pt-BR", '"frequency":"mensal"', '"frequency":"monthly"'),
        ("en", '"frequency":"monthly"', '"frequency":"mensal"'),
    ],
)
def test_subscriptions_tool_localised(
    registry, monkeypatch, use_language: Callable[[str], None], language: str, present: str, absent: str
):
    monkeypatch.setattr(analytics, "today_brt", lambda: TODAY)
    use_language(language)
    result = registry.run(ToolCall("c", "detect_subscriptions", {"months": 13}))
    assert not result.is_error, result.content
    assert present in result.content
    assert absent not in result.content
