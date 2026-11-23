"""Safety-net guards for the message catalogs, plus self-tests proving the guards can fail.

The guard logic lives in small pure helpers. Each helper is tested on synthetic inputs (toy catalogs, a source
snippet), so a guard cannot pass vacuously while the real catalogs are empty or no module calls ``t()`` yet.
The ``test_*`` guards then apply the same helpers to the real catalogs and the real ``src/llmopenfinance`` modules.
"""

import ast
import string
from collections.abc import Mapping
from pathlib import Path

from llmopenfinance import messages_en, messages_pt

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "llmopenfinance"

PT_ACCENTS = "áàâãéêíóôõúç"

# Catalog keys whose English value may legitimately contain Portuguese (for example a stored category label).
ALLOWED_PT_KEYS: frozenset[str] = frozenset(
    {
        # Rule 8 quotes the stored Portuguese category label 'Alimentação' inside the English prompt.
        "prompt.system",
    }
)


def catalog_keys_diff(pt: Mapping[str, str], en: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """Compare the key sets of the two catalogs.

    Args:
        pt: The pt-BR catalog.
        en: The English catalog.

    Returns:
        ``(only_in_pt, only_in_en)``, each sorted.
    """
    return sorted(pt.keys() - en.keys()), sorted(en.keys() - pt.keys())


def placeholder_names(template: str) -> frozenset[str]:
    """Return the replacement-field names used by a ``str.format`` template.

    Args:
        template: Message template such as ``"{n} itens"``.

    Returns:
        The set of field names (empty when the template has none).
    """
    return frozenset(name for _, name, _, _ in string.Formatter().parse(template) if name is not None)


def placeholder_mismatches(pt: Mapping[str, str], en: Mapping[str, str]) -> list[str]:
    """List the keys present in both catalogs whose placeholder sets differ.

    Args:
        pt: The pt-BR catalog.
        en: The English catalog.

    Returns:
        Sorted keys whose two templates do not use exactly the same field names.
    """
    return sorted(key for key in pt.keys() & en.keys() if placeholder_names(pt[key]) != placeholder_names(en[key]))


def _is_t_call(node: ast.Call) -> bool:
    """Tell whether a call node is ``t(...)`` or ``i18n.t(...)``.

    Args:
        node: A call expression.

    Returns:
        ``True`` for the bare name ``t`` or the attribute ``t`` on the name ``i18n``.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "t"
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "t"
        and isinstance(func.value, ast.Name)
        and func.value.id == "i18n"
    )


def _first_key_argument(node: ast.Call) -> ast.expr | None:
    """Return the expression passed as the message key, positionally or as ``key=``.

    Args:
        node: A call to ``t``.

    Returns:
        The key expression, or ``None`` when the call passes none.
    """
    if node.args:
        return node.args[0]
    return next((kw.value for kw in node.keywords if kw.arg == "key"), None)


def literal_t_keys(source: str) -> list[str]:
    """Collect the string-literal keys of every ``t(...)`` / ``i18n.t(...)`` call in a module.

    Calls whose key is not a string constant (``t(var)``) are skipped: they cannot be checked statically.

    Args:
        source: Python source text.

    Returns:
        The literal keys in source order (duplicates kept).
    """
    found: list[tuple[int, int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and _is_t_call(node)):
            continue
        key = _first_key_argument(node)
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            found.append((node.lineno, node.col_offset, key.value))
    return [key for _, _, key in sorted(found)]


def accented_values(catalog: Mapping[str, str], allowed: frozenset[str]) -> list[str]:
    """List the keys whose value contains a Portuguese accented letter, except allowed keys.

    Args:
        catalog: Catalog to scan (normally the English one).
        allowed: Keys exempt from the check.

    Returns:
        Sorted offending keys.
    """
    return sorted(
        key for key, value in catalog.items() if key not in allowed and any(c in PT_ACCENTS for c in value.lower())
    )


# ---------------------------------------------------------------------------
# Self-tests: each helper fails on synthetic bad input and passes on synthetic good input.
# ---------------------------------------------------------------------------


def test_catalog_keys_diff_reports_each_side():
    pt = {"a.one": "x", "a.two": "y", "a.pt_only": "z"}
    en = {"a.one": "x", "a.two": "y", "a.en_only": "z"}
    assert catalog_keys_diff(pt, en) == (["a.pt_only"], ["a.en_only"])


def test_catalog_keys_diff_empty_when_equal():
    assert catalog_keys_diff({"a": "x", "b": "y"}, {"b": "Y", "a": "X"}) == ([], [])


def test_placeholder_names_ignores_literals_and_format_specs():
    assert placeholder_names("sem campos") == frozenset()
    assert placeholder_names("{n:.2f} de {total} {{literal}}") == frozenset({"n", "total"})


def test_placeholder_mismatches_flags_renamed_missing_and_extra_fields():
    pt = {"ok": "{n} itens", "renamed": "{n} itens", "missing": "{n} de {total}", "extra": "{n}", "plain": "oi"}
    en = {"ok": "{n} items", "renamed": "{count} items", "missing": "{n} of", "extra": "{n} {m}", "plain": "hi"}
    assert placeholder_mismatches(pt, en) == ["extra", "missing", "renamed"]


def test_placeholder_mismatches_ignores_keys_present_on_one_side_only():
    assert placeholder_mismatches({"a": "{x}"}, {"b": "{y}"}) == []


def test_literal_t_keys_finds_bare_and_attribute_calls_and_skips_dynamic_keys():
    source = (
        "from llmopenfinance import i18n\n"
        "from llmopenfinance.i18n import t\n"
        "def f(var, n):\n"
        '    a = t("a.b")\n'
        '    b = i18n.t("c.d", n=1)\n'
        "    c = t(var)\n"
        '    d = t(key="e.f")\n'
        "    e = t(f'g.{n}')\n"
        '    g = other.t("not.ours")\n'
        '    h = tt("not.ours.either")\n'
        "    return t()\n"
    )
    assert literal_t_keys(source) == ["a.b", "c.d", "e.f"]


def test_literal_t_keys_empty_without_calls():
    assert literal_t_keys("x = 1\n") == []


def test_accented_values_flags_pt_letters_and_honours_the_allow_list():
    catalog = {"clean": "Spending by category", "ptword": "Alimentação", "upper": "É", "kept": "Saúde"}
    assert accented_values(catalog, frozenset()) == ["kept", "ptword", "upper"]
    assert accented_values(catalog, frozenset({"kept"})) == ["ptword", "upper"]


def test_accented_values_clean_catalog_is_empty():
    assert accented_values({"a": "Hello, world", "b": "Total: R$ 1,234.50"}, frozenset()) == []


# ---------------------------------------------------------------------------
# Guards applied to the real catalogs and the real source tree.
# ---------------------------------------------------------------------------


def test_source_scan_finds_the_modules():
    names = {path.name for path in SOURCE_DIR.glob("*.py")}
    assert {"analytics.py", "i18n.py", "messages_pt.py", "messages_en.py"} <= names


def test_catalogs_have_same_keys():
    only_pt, only_en = catalog_keys_diff(messages_pt.MESSAGES, messages_en.MESSAGES)
    assert only_pt == [], f"keys only in pt-BR: {only_pt}"
    assert only_en == [], f"keys only in en: {only_en}"


def test_catalogs_have_same_placeholders():
    assert placeholder_mismatches(messages_pt.MESSAGES, messages_en.MESSAGES) == []


def test_every_literal_t_key_exists():
    missing: list[str] = []
    for path in sorted(SOURCE_DIR.glob("*.py")):
        for key in literal_t_keys(path.read_text(encoding="utf-8")):
            for name, catalog in (("pt-BR", messages_pt.MESSAGES), ("en", messages_en.MESSAGES)):
                if key not in catalog:
                    missing.append(f"{path.name}: {key!r} missing from {name}")
    assert missing == []


def test_en_catalog_has_no_portuguese_accents():
    assert accented_values(messages_en.MESSAGES, ALLOWED_PT_KEYS) == []
