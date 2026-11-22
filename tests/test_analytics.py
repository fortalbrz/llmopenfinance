"""Tests for deterministic analytics over the synthetic dataset."""

import copy
from datetime import date

import duckdb
import pytest

from llmopenfinance import analytics
from llmopenfinance.analytics import brl, find_recurring, month_start
from tests.conftest import TODAY


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1234.56, "R$ 1.234,56"), (-0.5, "-R$ 0,50"), (0, "R$ 0,00"), (1_000_000, "R$ 1.000.000,00"), (None, "—")],
)
def test_brl(value, expected):
    assert brl(value) == expected


def test_brl_en(use_language):
    use_language("en")
    assert brl(1234.5) == "R$ 1,234.50"
    assert brl(-1234.5) == "-R$ 1,234.50"
    assert brl(None) == "—"


@pytest.mark.parametrize(
    ("months_back", "expected"),
    [(0, date(2026, 11, 1)), (1, date(2026, 10, 1)), (11, date(2025, 12, 1)), (23, date(2024, 12, 1))],
)
def test_month_start(months_back, expected):
    assert month_start(TODAY, months_back) == expected


def test_today_brt_is_a_date():
    assert isinstance(analytics.today_brt(), date)


def test_accounts_overview(con):
    accounts = {a["name"]: a for a in analytics.accounts_overview(con)}
    assert accounts["Nubank Conta"]["balance"] == pytest.approx(3500.75)
    card = accounts["Nubank Ultravioleta"]
    assert card["type"] == "CREDIT"
    assert card["available_credit_limit"] == pytest.approx(8109.6)
    assert card["institution"] == "MeuPluggy"


def test_spending_by_category_online_shopping(con):
    rows = analytics.spending_by_category(con, months=2, category="compras online", today=TODAY)
    by_month = {r["month"]: r["total"] for r in rows}
    # Nov: amazon 210 + mercadolivre 250 - refund 50 ; Oct: amazon 200
    assert by_month == {"2026-11": 410.0, "2026-10": 200.0}


def test_spending_by_category_matches_pluggy_name_and_accents(con):
    # Pluggy's English name only matches rows Pluggy categorized (amazon 210 - refund 50);
    # Mercado Livre has no Pluggy category and is matched via the local pt-BR category instead.
    rows = analytics.spending_by_category(con, months=1, category="ONLINE shopping", today=TODAY)
    assert rows[0]["total"] == pytest.approx(160.0)
    rows = analytics.spending_by_category(con, months=1, category="vídeo", today=TODAY)
    assert rows[0]["category"] == "Streaming de vídeo"


def test_spending_excludes_transfers_and_investments(con):
    categories = {r["category"] for r in analytics.spending_by_category(con, months=12, today=TODAY)}
    assert "Pagamento de fatura" not in categories
    assert "Investimentos" not in categories
    assert "Transferência entre contas próprias" not in categories


def test_query_transactions_filters_and_summary(con):
    result = analytics.query_transactions(con, search="amazon", date_from="2026-11-01", date_to="2026-11-30")
    assert result["count"] == 2  # purchase + refund
    assert result["total_spend"] == pytest.approx(160.0)
    assert {r["kind"] for r in result["rows"]} == {"expense", "refund"}


def test_query_transactions_all_filters(con):
    result = analytics.query_transactions(
        con,
        category="streaming",
        kind="expense",
        account_type="credit",
        min_amount=20,
        max_amount=60,
        order="amount_desc",
        limit=3,
    )
    assert result["returned"] == 3
    assert result["count"] > 3
    assert result["rows"][0]["amount"] == pytest.approx(-55.9)
    asc = analytics.query_transactions(con, order="date_asc", limit=1)
    assert asc["rows"][0]["date"] == date(2026, 1, 5)


def test_query_transactions_limit_is_capped(con):
    assert analytics.query_transactions(con, limit=10_000)["returned"] <= 100


def test_cashflow(con):
    months = {r["month"]: r for r in analytics.cashflow(con, months=3, today=TODAY)}
    assert set(months) == {"2026-09", "2026-10", "2026-11"}
    oct_ = months["2026-10"]
    assert oct_["income"] == pytest.approx(8000.0)
    assert oct_["invested_net"] == pytest.approx(1000.0)
    assert oct_["net"] == round(8000.0 - oct_["spending"], 2)
    assert 0 < oct_["savings_rate"] < 1


def test_top_merchants(con):
    top = analytics.top_merchants(con, months=1, limit=3, today=TODAY)
    assert len(top) == 3
    assert top[0]["total"] >= top[1]["total"] >= top[2]["total"]


def test_bills_investments_loans(con):
    bills = analytics.bills_overview(con, limit=1)
    assert bills[0]["due_date"] == date(2026, 12, 5)
    inv = analytics.investments_overview(con)
    assert inv["total"] == pytest.approx(20000.0)
    assert {t["type"] for t in inv["by_type"]} == {"FIXED_INCOME", "MUTUAL_FUND"}
    loans = analytics.loans_overview(con)
    assert loans["total_outstanding"] == pytest.approx(12000.0)
    assert loans["loans"][0]["paid_installments"] == 10


def test_detect_subscriptions(con):
    result = analytics.detect_subscriptions(con, today=TODAY)
    names = {s["merchant_key"]: s for s in result["subscriptions"]}
    assert "netflix com" in names
    assert "spotify" in names
    assert names["netflix com"]["frequency"] == "mensal"
    assert names["netflix com"]["confidence"] == "alta"
    assert names["netflix com"]["annual_estimate"] == round(55.9 * 12, 2)
    assert "loja de moveis" not in names  # installments are not subscriptions
    assert "amazon marketplace" not in names  # irregular amounts
    assert result["monthly_total"] == round(sum(s["monthly_estimate"] for s in result["subscriptions"]), 2)


def _charge(key, day, spend, category="Outros"):
    return {
        "merchant_key": key,
        "description": key.upper(),
        "category": category,
        "date": day,
        "spend": spend,
        "account_name": "Card",
        "total_installments": None,
    }


def test_find_recurring_annual_and_inactive():
    rows = [
        _charge("seguro auto", date(2024, 12, 1), 1200.0, "Seguros"),
        _charge("seguro auto", date(2025, 12, 1), 1250.0, "Seguros"),
        _charge("academia", date(2026, 1, 10), 99.0),
        _charge("academia", date(2026, 2, 10), 99.0),
        _charge("academia", date(2026, 3, 10), 99.0),
    ]
    found = {r["merchant_key"]: r for r in find_recurring(rows, TODAY)}
    assert found["seguro auto"]["frequency"] == "anual"
    assert found["seguro auto"]["monthly_estimate"] == round(1225.0 / 12, 2)
    assert found["seguro auto"]["active"] is True
    assert found["academia"]["active"] is False  # last charge in March


def test_find_recurring_weekly_and_single_subscription():
    rows = [_charge("feira", date(2026, 11, d), 50.0) for d in (1, 8, 15)]
    rows.append(_charge("novo streaming", date(2026, 11, 10), 29.9, "Streaming de vídeo"))
    rows.append(_charge("streaming antigo", date(2026, 1, 10), 29.9, "Streaming de vídeo"))
    rows.append(_charge("", date(2026, 11, 10), 10.0))
    rows.append({**_charge("parcelado", date(2026, 11, 10), 10.0), "total_installments": 5})
    rows.append(_charge("refund", date(2026, 11, 10), -10.0))
    found = {r["merchant_key"]: r for r in find_recurring(rows, TODAY)}
    assert found["feira"]["frequency"] == "semanal"
    assert found["novo streaming"]["confidence"] == "baixa"
    assert "streaming antigo" not in found
    assert set(found) == {"feira", "novo streaming"}


def test_find_recurring_subscription_like_irregular_interval():
    rows = [
        _charge("app", date(2026, 9, 1), 10.0, "Serviços digitais"),
        _charge("app", date(2026, 11, 15), 11.0, "Serviços digitais"),
    ]
    found = find_recurring(rows, TODAY)
    assert found[0]["confidence"] == "média"
    assert found[0]["frequency"] == "irregular"
    # 21.00 spent over 75 days + one 75-day interval (~4.93 months)
    assert found[0]["monthly_estimate"] == pytest.approx(21.0 / (150 / 30.44), abs=0.01)


def test_find_recurring_twice_a_month_uses_observed_rate():
    """Regression: charges on the 5th and 20th were estimated as one charge per month."""
    rows = [
        _charge("cloud", date(2026, month, day), 9.9, "Serviços digitais") for month in range(1, 12) for day in (5, 20)
    ]
    found = find_recurring(rows, TODAY)[0]
    assert found["frequency"] == "irregular"
    assert found["monthly_estimate"] == pytest.approx(19.8, rel=0.06)


def test_run_select_trailing_comment_after_semicolon(con):
    assert analytics.run_select(con, "SELECT 1 AS a; -- fim")["rows"] == [[1]]
    assert analytics.run_select(con, "SELECT ';' AS s; /* x */\n")["rows"] == [[";"]]


def test_detect_subscriptions_include_inactive(con):
    result = analytics.detect_subscriptions(con, today=date(2027, 6, 1), include_inactive=True)
    assert result["subscriptions"]
    assert all(not s["active"] for s in result["subscriptions"])
    assert result["monthly_total"] == 0


def test_run_select_happy_path(con):
    result = analytics.run_select(
        con, "SELECT category, count(*) n FROM v_transactions GROUP BY 1 ORDER BY 1;", limit=3
    )
    assert result["columns"] == ["category", "n"]
    assert result["row_count"] == 3
    assert result["truncated"] is True


def test_run_select_trailing_comment(con):
    assert analytics.run_select(con, "SELECT 1 AS a -- comentário\n;")["rows"] == [[1]]


@pytest.mark.parametrize(
    "sql",
    ["DELETE FROM transactions", "SELECT 1; SELECT 2", "PRAGMA version", "ATTACH 'x.db'", "COPY accounts TO 'x.csv'"],
)
def test_run_select_rejects_non_select(con, sql):
    with pytest.raises(ValueError, match=r"SQL|SELECT"):
        analytics.run_select(con, sql)


def test_run_select_blocks_file_access_even_inside_select(con):
    with pytest.raises(duckdb.Error):
        analytics.run_select(con, "SELECT * FROM read_csv('C:/Windows/win.ini')")


def test_run_select_timeout_interrupts(con):
    with pytest.raises(duckdb.Error):
        analytics.run_select(
            con, "SELECT count(*) FROM range(100000000000) a, range(10) b WHERE a.range % 7 = b.range", timeout=0.2
        )


def test_build_snapshot(con):
    text = analytics.build_snapshot(con, today=TODAY)
    assert "# Resumo financeiro (hoje: 2026-11-22" in text
    assert "Nubank Conta" in text
    assert "R$ 3.500,75" in text
    assert "NETFLIX.COM" in text
    assert "Investimentos: total R$ 20.000,00" in text
    assert "Empréstimos/financiamentos" in text
    assert "Faturas de cartão" in text
    assert len(text) < 8000


def test_build_snapshot_empty_db_and_attention():
    from llmopenfinance.store import FinanceStore

    store = FinanceStore(":memory:")
    store.update_item({"id": "x", "status": "LOGIN_ERROR", "connector": {"name": "Banco"}})
    text = analytics.build_snapshot(store.con, today=TODAY)
    assert "nenhuma conta sincronizada" in text
    assert "Banco: LOGIN_ERROR" in text
    store.close()


def _detected() -> dict:
    """A ``detect_subscriptions``-shaped payload covering every internal frequency and confidence code."""
    rows = [
        {"name": "Netflix", "frequency": "mensal", "confidence": "alta", "monthly_estimate": 55.9},
        {"name": "Seguro", "frequency": "anual", "confidence": "média", "monthly_estimate": 100.0},
        {"name": "Feira", "frequency": "semanal", "confidence": "baixa", "monthly_estimate": 216.67},
        {"name": "Nuvem", "frequency": "irregular", "confidence": "média", "monthly_estimate": 19.8},
    ]
    return {"subscriptions": rows, "monthly_total": 392.37, "annual_total": 4708.44}


def test_localize_subscriptions_pt_is_identity():
    detected = _detected()
    result = analytics.localize_subscriptions(detected)
    assert result == detected
    assert result is not detected
    assert result["subscriptions"] is not detected["subscriptions"]


def test_localize_subscriptions_en_maps_codes_without_mutating_input(use_language):
    use_language("en")
    detected = _detected()
    before = copy.deepcopy(detected)
    result = analytics.localize_subscriptions(detected)
    assert [(r["frequency"], r["confidence"]) for r in result["subscriptions"]] == [
        ("monthly", "high"),
        ("annual", "medium"),
        ("weekly", "low"),
        ("irregular", "medium"),
    ]
    # Everything else is carried over untouched; only the two label fields change.
    assert result["monthly_total"] == pytest.approx(392.37)
    assert result["annual_total"] == pytest.approx(4708.44)
    assert [(r["name"], r["monthly_estimate"]) for r in result["subscriptions"]] == [
        (r["name"], r["monthly_estimate"]) for r in before["subscriptions"]
    ]
    assert detected == before
    assert detected["subscriptions"][0]["frequency"] == "mensal"


def test_localize_subscriptions_unknown_codes_and_missing_fields_pass_through(use_language):
    use_language("en")
    detected = {"subscriptions": [{"name": "X", "frequency": "quinzenal", "confidence": "?"}, {"name": "Y"}]}
    result = analytics.localize_subscriptions(detected)
    assert result["subscriptions"] == [{"name": "X", "frequency": "quinzenal", "confidence": "?"}, {"name": "Y"}]


def test_localize_subscriptions_empty(use_language):
    use_language("en")
    assert analytics.localize_subscriptions({"subscriptions": []}) == {"subscriptions": []}


def test_detect_subscriptions_internal_codes_stay_portuguese_in_en(con, use_language):
    """Localising is an output-edge step: the detector keeps returning the canonical codes."""
    use_language("en")
    result = analytics.detect_subscriptions(con, today=TODAY)
    names = {s["merchant_key"]: s for s in result["subscriptions"]}
    assert names["netflix com"]["frequency"] == "mensal"
    assert names["netflix com"]["confidence"] == "alta"
    localized = analytics.localize_subscriptions(result)
    shown = {s["merchant_key"]: s for s in localized["subscriptions"]}
    assert (shown["netflix com"]["frequency"], shown["netflix com"]["confidence"]) == ("monthly", "high")


def test_build_snapshot_en(con, use_language):
    use_language("en")
    text = analytics.build_snapshot(con, today=TODAY)
    lines = text.splitlines()
    assert lines[0] == "# Financial summary (today: 2026-11-22; transactions from 2026-01-05 to 2026-11-20, 91 entries)"
    for heading in (
        "## Accounts",
        "## Monthly cash flow (last 6 months; current month partial)",
        "month | income | spending | balance",
        "## Largest spending categories (monthly average over the last 3 complete months)",
        "## Active recurring charges detected (total ≈ R$ 329.80/month)",
        "## Recent credit card bills",
        "## Investments: total R$ 20,000.00",
        "## Loans/financing: outstanding balance R$ 12,000.00",
    ):
        assert heading in lines
    assert "R$ 1," in text
    assert "R$ 3,500.75" in text
    # Amounts, ISO dates and stored data keep their value; only the sentence frames follow the language.
    assert "- Nubank Conta (CHECKING_ACCOUNT): balance R$ 3,500.75" in lines
    assert (
        "- Nubank Ultravioleta (credit card): current bill R$ 1,890.40, "
        "available limit R$ 8,109.60 of R$ 10,000.00, due 2026-12-05"
    ) in lines
    assert "- Nubank Ultravioleta: R$ 1,450.00, due 2026-11-05, minimum R$ 217.50" in lines
    assert "- NETFLIX.COM [Streaming de vídeo]: R$ 55.90 monthly, last 2026-11-08 (confidence high)" in lines
    assert "- FIXED_INCOME: R$ 15,000.00 (1 positions)" in lines
    assert "- Crédito pessoal: balance R$ 12,000.00, installment R$ 980.00, CET 35.2" in lines
    # The Portuguese frame words are gone. "Contas de consumo" is a stored category label and stays as it is.
    for word in ("Resumo", "Fluxo", "mensal", "confiança", "vencimento", "saldo", "posições"):
        assert word not in text
    assert "Contas" not in text.replace("Contas de consumo", "")
    assert "- Contas de consumo: R$ 189.00" in lines


def test_build_snapshot_empty_db_en(use_language):
    from llmopenfinance.store import FinanceStore

    use_language("en")
    store = FinanceStore(":memory:")
    store.update_item({"id": "x", "status": "LOGIN_ERROR", "connector": {"name": "Banco"}})
    text = analytics.build_snapshot(store.con, today=TODAY)
    store.close()
    lines = text.splitlines()
    assert lines[0] == "# Financial summary (today: 2026-11-22; transactions from — to —, 0 entries)"
    assert "- (no accounts synced)" in lines
    assert "## Connections needing attention" in lines
    assert "- Banco: LOGIN_ERROR" in lines
    assert "nenhuma" not in text
    assert "atenção" not in text


@pytest.mark.parametrize(
    ("language", "card", "loan"),
    [
        (
            "pt-BR",
            "- Cartao X (cartão de crédito): fatura atual R$ 10,00, limite disponível — de —, vencimento —",
            "- FINANCING: saldo R$ 500,00, parcela R$ 50,00, CET —",
        ),
        (
            "en",
            "- Cartao X (credit card): current bill R$ 10.00, available limit — of —, due —",
            "- FINANCING: balance R$ 500.00, installment R$ 50.00, CET —",
        ),
    ],
)
def test_build_snapshot_missing_values_show_a_dash(language, card, loan, use_language):
    from llmopenfinance.store import FinanceStore
    from tests.conftest import populate

    use_language(language)
    store = FinanceStore(":memory:")
    populate(store)
    store.add_item("item-2")
    store.replace_accounts(
        "item-2",
        [{"id": "acc-x", "itemId": "item-2", "type": "CREDIT", "name": "Cartao X", "balance": 10.0}],
    )
    store.replace_loans(
        "item-2",
        [
            {
                "id": "loan-x",
                "type": "FINANCING",
                "payments": {"contractOutstandingBalance": 500.0},
                "installmentAmount": 50,
            }
        ],
    )
    lines = analytics.build_snapshot(store.con, today=TODAY).splitlines()
    store.close()
    assert card in lines
    assert loan in lines


def test_build_snapshot_pt_frames_unchanged(con):
    """Pins the pt-BR sentence frames, so the catalog move cannot drift from the original f-strings."""
    lines = analytics.build_snapshot(con, today=TODAY).splitlines()
    assert lines[0] == "# Resumo financeiro (hoje: 2026-11-22; transações de 2026-01-05 a 2026-11-20, 91 lançamentos)"
    for expected in (
        "## Contas",
        "- Nubank Conta (CHECKING_ACCOUNT): saldo R$ 3.500,75",
        (
            "- Nubank Ultravioleta (cartão de crédito): fatura atual R$ 1.890,40, "
            "limite disponível R$ 8.109,60 de R$ 10.000,00, vencimento 2026-12-05"
        ),
        "## Fluxo de caixa mensal (últimos 6 meses; mês atual parcial)",
        "mês | receitas | gastos | saldo",
        "## Maiores categorias de gasto (média mensal dos últimos 3 meses completos)",
        "## Cobranças recorrentes ativas detectadas (total ≈ R$ 329,80/mês)",
        "- NETFLIX.COM [Streaming de vídeo]: R$ 55,90 mensal, último 2026-11-08 (confiança alta)",
        "## Faturas de cartão recentes",
        "- Nubank Ultravioleta: R$ 1.450,00, vencimento 2026-11-05, mínimo R$ 217,50",
        "## Investimentos: total R$ 20.000,00",
        "- FIXED_INCOME: R$ 15.000,00 (1 posições)",
        "## Empréstimos/financiamentos: saldo devedor R$ 12.000,00",
        "- Crédito pessoal: saldo R$ 12.000,00, parcela R$ 980,00, CET 35.2",
    ):
        assert expected in lines


def test_run_select_errors_en(con, use_language):
    use_language("en")
    with pytest.raises(ValueError, match=r"^Send exactly one SQL statement\.$"):
        analytics.run_select(con, "SELECT 1; SELECT 2")
    with pytest.raises(ValueError, match=r"^Only SELECT queries are allowed\.$"):
        analytics.run_select(con, "DELETE FROM transactions")


def test_run_select_errors_pt(con):
    with pytest.raises(ValueError, match=r"^Envie exatamente uma instrução SQL\.$"):
        analytics.run_select(con, "SELECT 1; SELECT 2")
    with pytest.raises(ValueError, match=r"^Somente consultas SELECT são permitidas\.$"):
        analytics.run_select(con, "DELETE FROM transactions")
