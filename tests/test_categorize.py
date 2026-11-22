"""Tests for local categorization and flow classification."""

import pytest

from llmopenfinance.categorize import (
    CARD_PAYMENT,
    INVESTMENTS,
    OTHER,
    OWN_TRANSFER,
    TRANSFERS,
    classify_kind,
    keyword_category,
    local_category,
    map_pluggy_category,
    merchant_key,
    signed_flow,
    strip_accents,
)


def test_strip_accents():
    assert strip_accents("Ação ÇÉ") == "acao ce"


@pytest.mark.parametrize(
    ("description", "merchant", "expected"),
    [
        ("NETFLIX.COM 12/03", None, "netflix com"),
        ("LOJA DE MOVEIS PARC 02/10", None, "loja de moveis"),
        ("PIX ENVIADO JOAO DA SILVA", None, "joao da silva"),
        ("qualquer", "Spotify Brasil", "spotify brasil"),
        (None, None, ""),
        ("PAG", None, "pag"),
    ],
)
def test_merchant_key(description, merchant, expected):
    assert merchant_key(description, merchant) == expected


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("Online shopping", "Compras online"),
        ("ONLINE SHOPPING", "Compras online"),
        ("Transfer - PIX", TRANSFERS),
        ("Same person transfer - TED", OWN_TRANSFER),
        ("Transfers - Boleto", TRANSFERS),
        ("Credit card payment", CARD_PAYMENT),
        ("Video streaming", "Streaming de vídeo"),
        ("Something brand new", None),
        (None, None),
        ("", None),
    ],
)
def test_map_pluggy_category(category, expected):
    assert map_pluggy_category(category) == expected


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("PAGAMENTO DE FATURA", CARD_PAYMENT),
        ("Pagamento recebido", CARD_PAYMENT),
        ("AMAZON PRIME CANAIS", "Streaming de vídeo"),
        ("AMAZON MARKETPLACE", "Compras online"),
        ("MERCADOLIVRE*LOJA", "Compras online"),
        ("Spotify P1234", "Streaming de música"),
        ("UBER *TRIP", "Transporte"),
        ("UBER EATS", "Delivery de comida"),
        ("iFood", "Delivery de comida"),
        ("Drogaria São Paulo", "Saúde"),
        ("IOF COMPRA INTERNACIONAL", "Impostos e taxas"),
        ("PIX ENVIADO", TRANSFERS),
        ("APLICACAO CDB", INVESTMENTS),
        ("xyz unknown", None),
        ("", None),
        (None, None),
    ],
)
def test_keyword_category(description, expected):
    assert keyword_category(description) == expected


def test_local_category_prefers_specific_pluggy_category():
    assert local_category("Video streaming", "AMAZON MARKETPLACE") == "Streaming de vídeo"


def test_local_category_refines_generic_pluggy_category_with_keywords():
    assert local_category("Shopping", "AMAZON MARKETPLACE") == "Compras online"


def test_local_category_generic_transfer_not_overridden_by_generic_keyword():
    assert local_category("Shopping", "PIX LOJA") == "Compras"


def test_local_category_transfer_refined_to_card_payment():
    assert local_category("Transfer - PIX", "PAGAMENTO FATURA") == CARD_PAYMENT


def test_local_category_unknown_pluggy_uses_translation_then_raw():
    assert local_category("Brand new", "xyz", translated="Novidade") == "Novidade"
    assert local_category("Brand new", "xyz") == "Brand new"


def test_local_category_fallbacks():
    assert local_category(None, "NETFLIX") == "Streaming de vídeo"
    assert local_category(None, "nada conhecido") == OTHER


@pytest.mark.parametrize(
    ("flow", "category", "account_type", "expected"),
    [
        (-100.0, CARD_PAYMENT, "BANK", "transfer"),
        (100.0, OWN_TRANSFER, "BANK", "transfer"),
        (-100.0, INVESTMENTS, "BANK", "investment"),
        (-50.0, "Compras online", "CREDIT", "expense"),
        (50.0, "Compras online", "CREDIT", "refund"),
        (5000.0, "Salário", "BANK", "income"),
        (10.0, "Outros", None, "income"),
    ],
)
def test_classify_kind(flow, category, account_type, expected):
    assert classify_kind(flow, category, account_type) == expected


@pytest.mark.parametrize(
    ("amount", "tx_type", "account_type", "expected"),
    [
        (-100.0, "DEBIT", "BANK", -100.0),
        (100.0, "DEBIT", "CREDIT", -100.0),  # card charge is positive at Pluggy
        (-1500.0, "CREDIT", "CREDIT", 1500.0),  # card payment is negative at Pluggy
        (200.0, "CREDIT", "BANK", 200.0),
        (-30.0, None, "BANK", -30.0),
        (30.0, None, "CREDIT", -30.0),
        (None, "DEBIT", "BANK", -0.0),
    ],
)
def test_signed_flow(amount, tx_type, account_type, expected):
    assert signed_flow(amount, tx_type, account_type) == expected


def test_jcp_and_word_boundaries():
    """Regression: 'juros sobre capital' (dividend income) was filed as interest charges."""
    assert keyword_category("JUROS SOBRE CAPITAL PROPRIO ITSA4") == "Rendimentos"
    assert keyword_category("JUROS ROTATIVO") == "Juros e encargos"
    assert keyword_category("MORA CONTA") == "Juros e encargos"
    assert keyword_category("AMORA DOCES") is None
