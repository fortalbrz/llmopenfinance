"""Tests for the language core: normalisation, one-shot resolution, catalog lookup, fallback and dates."""

import logging
from datetime import date

import pytest

from llmopenfinance import messages_en, messages_pt
from llmopenfinance.constants import DEFAULT_LANGUAGE, ENV_LANG, SUPPORTED_LANGUAGES
from llmopenfinance.i18n import format_date, get_language, normalize_language, reset_language, t

I18N_LOGGER = "llmopenfinance.i18n"


def _warnings(caplog, needle):
    return [r for r in caplog.records if r.levelno == logging.WARNING and needle in r.getMessage()]


def test_constants():
    assert DEFAULT_LANGUAGE == "pt-BR"
    assert SUPPORTED_LANGUAGES == ("pt-BR", "en")
    assert ENV_LANG == "LLMOPENFINANCE_LANG"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pt", "pt-BR"),
        ("PT_br", "pt-BR"),
        (" pt-BR ", "pt-BR"),
        ("pt-PT", "pt-BR"),
        ("en", "en"),
        ("en-US", "en"),
        ("EN_gb", "en"),
        (" EN ", "en"),
    ],
)
def test_normalize_language(raw, expected):
    assert normalize_language(raw) == expected


@pytest.mark.parametrize("raw", ["fr", "", "   ", "ptx", "english", "-en"])
def test_normalize_rejects_unknown(raw):
    with pytest.raises(ValueError, match="supported") as excinfo:
        normalize_language(raw)
    message = str(excinfo.value)
    assert "pt-BR" in message
    assert "pt-BR, en" in message


def test_catalogs_start_empty():
    assert messages_pt.MESSAGES == {}
    assert messages_en.MESSAGES == {}


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        (None, "pt-BR"),
        ("", "pt-BR"),
        ("   ", "pt-BR"),
        ("en", "en"),
        (" EN ", "en"),
        ("en_US", "en"),
        ("pt_BR", "pt-BR"),
    ],
)
def test_get_language_default_and_env(monkeypatch, env, expected):
    reset_language()
    if env is None:
        monkeypatch.delenv(ENV_LANG, raising=False)
    else:
        monkeypatch.setenv(ENV_LANG, env)
    assert get_language() == expected


def test_get_language_rejects_invalid_env_without_freezing(monkeypatch):
    reset_language()
    monkeypatch.setenv(ENV_LANG, "fr")
    with pytest.raises(ValueError, match="supported"):
        get_language()
    monkeypatch.setenv(ENV_LANG, "en")
    assert get_language() == "en"


def test_language_is_frozen_after_first_use(monkeypatch):
    reset_language()
    monkeypatch.setenv(ENV_LANG, "en")
    assert get_language() == "en"
    monkeypatch.setenv(ENV_LANG, "pt-BR")
    assert get_language() == "en"
    monkeypatch.delenv(ENV_LANG)
    assert get_language() == "en"


def test_reset_language_freezes_normalised_code(monkeypatch):
    monkeypatch.setenv(ENV_LANG, "pt-BR")
    reset_language("EN_us")
    assert get_language() == "en"
    reset_language()
    assert get_language() == "pt-BR"


def test_reset_language_rejects_unknown_and_keeps_state():
    reset_language("en")
    with pytest.raises(ValueError, match="supported"):
        reset_language("fr")
    assert get_language() == "en"


def test_use_language_fixture(use_language):
    assert get_language() == "pt-BR"
    use_language("en")
    assert get_language() == "en"
    use_language("pt")
    assert get_language() == "pt-BR"


def test_t_formats_active_catalog(monkeypatch, use_language):
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.hello", "Olá, {name}!")
    monkeypatch.setitem(messages_en.MESSAGES, "i18n.test.hello", "Hello, {name}!")
    assert t("i18n.test.hello", name="Ana") == "Olá, Ana!"
    use_language("en")
    assert t("i18n.test.hello", name="Ana") == "Hello, Ana!"


def test_t_formats_and_falls_back(monkeypatch, caplog, use_language):
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.only_pt", "Só em português, {name}")
    use_language("en")
    with caplog.at_level(logging.WARNING, logger=I18N_LOGGER):
        first = t("i18n.test.only_pt", name="Ana")
        second = t("i18n.test.only_pt", name="Bia")
    assert first == "Só em português, Ana"
    assert second == "Só em português, Bia"
    assert len(_warnings(caplog, "i18n.test.only_pt")) == 1


def test_t_fallback_warning_is_per_key(monkeypatch, caplog, use_language):
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.a", "A")
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.b", "B")
    use_language("en")
    with caplog.at_level(logging.WARNING, logger=I18N_LOGGER):
        t("i18n.test.a")
        t("i18n.test.b")
        t("i18n.test.a")
    assert len(_warnings(caplog, "i18n.test.a")) == 1
    assert len(_warnings(caplog, "i18n.test.b")) == 1


@pytest.mark.parametrize("code", ["pt-BR", "en"])
def test_t_missing_everywhere_returns_key(caplog, use_language, code):
    use_language(code)
    with caplog.at_level(logging.WARNING, logger=I18N_LOGGER):
        assert t("i18n.test.nowhere") == "i18n.test.nowhere"
        assert t("i18n.test.nowhere", name="x") == "i18n.test.nowhere"
    assert len(_warnings(caplog, "i18n.test.nowhere")) == 1


def test_t_missing_placeholder_returns_template(monkeypatch, caplog):
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.hello", "Olá, {name}!")
    with caplog.at_level(logging.WARNING, logger=I18N_LOGGER):
        assert t("i18n.test.hello") == "Olá, {name}!"
        assert t("i18n.test.hello", other="x") == "Olá, {name}!"
    assert len(_warnings(caplog, "i18n.test.hello")) == 1


def test_t_malformed_template_returns_template(monkeypatch, caplog):
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.bad", "Olá, {name")
    with caplog.at_level(logging.WARNING, logger=I18N_LOGGER):
        assert t("i18n.test.bad", name="Ana") == "Olá, {name"
    assert len(_warnings(caplog, "i18n.test.bad")) == 1


def test_t_escaped_braces_are_formatted(monkeypatch):
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.braces", "use {{name}} em {tool}")
    assert t("i18n.test.braces", tool="sql") == "use {name} em sql"


def test_t_reads_catalogs_live(monkeypatch):
    assert t("i18n.test.late") == "i18n.test.late"
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.late", "tarde")
    assert t("i18n.test.late") == "tarde"


def test_t_uses_language_resolved_from_env(monkeypatch):
    monkeypatch.setitem(messages_pt.MESSAGES, "i18n.test.env", "pt")
    monkeypatch.setitem(messages_en.MESSAGES, "i18n.test.env", "en")
    reset_language()
    monkeypatch.setenv(ENV_LANG, "en")
    assert t("i18n.test.env") == "en"


def test_format_date_per_language(use_language):
    day = date(2026, 11, 22)
    assert format_date(day) == "22/11/2026"
    use_language("en")
    assert format_date(day) == "2026-11-22"
