"""Tests for deterministic analytics over the synthetic dataset."""

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
