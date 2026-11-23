"""Tests for the DuckDB store and the pure row mappers."""

import json
from datetime import date, datetime

import duckdb
import pytest

from llmopenfinance.store import (
    FinanceStore,
    account_row,
    bill_row,
    dig,
    investment_row,
    item_row,
    loan_row,
    local_date,
    parse_timestamp,
    to_float,
    to_int,
    transaction_row,
)
from tests.conftest import BANK_ID, CARD_ID, ITEM, ITEM_ID, populate


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-03-01T00:00:00.000Z", date(2026, 3, 1)),  # midnight UTC = calendar date (no shift)
        ("2026-03-01T02:30:00.000Z", date(2026, 2, 28)),  # real timestamp -> UTC-3
        ("2026-03-01T15:00:00Z", date(2026, 3, 1)),
        ("2026-03-01", date(2026, 3, 1)),
        ("2026-03-01T10:00:00", date(2026, 3, 1)),
        ("not a date", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_local_date(value, expected):
    assert local_date(value) == expected


def test_parse_timestamp():
    assert parse_timestamp("2026-01-02T03:04:05Z") == datetime(2026, 1, 2, 3, 4, 5)
    assert parse_timestamp("2026-01-02T03:04:05") == datetime(2026, 1, 2, 3, 4, 5)
    assert parse_timestamp("bad") is None
    assert parse_timestamp(None) is None


def test_scalar_helpers():
    assert dig({"a": {"b": 1}}, "a", "b") == 1
    assert dig({"a": None}, "a", "b") is None
    assert to_float("1.5") == pytest.approx(1.5)
    assert to_float("x") is None
    assert to_float(True) is None
    assert to_float(None) is None
    assert to_int("3.0") == 3
    assert to_int(None) is None


def test_item_row_with_error():
    row = item_row({"id": "i", "error": {"code": "LOGIN_ERROR", "message": "bad"}})
    assert row[0] == "i"
    assert row[5] == "LOGIN_ERROR: bad"


def test_account_row_without_credit_data():
    row = account_row({"id": "a", "type": "BANK"}, "item")
    assert row[1] == "item"
    assert row[8] == "BRL"
    assert row[9] is None


def test_transaction_row_enrichment():
    tx = {
        "id": "t1",
        "date": "2026-05-10T00:00:00.000Z",
        "description": "AMAZON",
        "amount": 99.9,
        "type": "DEBIT",
        "categoryId": "08010000",
        "category": "Online shopping",
        "merchant": {"businessName": "Amazon Servicos"},
        "creditCardMetadata": {"installmentNumber": 1, "totalInstallments": 3, "billId": "b1"},
        "paymentData": {"paymentMethod": "PIX"},
    }
    row = transaction_row(tx, CARD_ID, ITEM_ID, "CREDIT", {"08010000": "Compras online"})
    assert row[3] == date(2026, 5, 10)
    assert row[7] == pytest.approx(-99.9)  # flow
    assert row[13] == "Compras online"
    assert row[14] == "expense"
    assert row[15] == "Amazon Servicos"
    assert row[18] == "PIX"
    assert (row[19], row[20], row[21]) == (1, 3, "b1")
    assert json.loads(row[22])["id"] == "t1"


def test_transaction_row_uses_raw_description_when_missing():
    row = transaction_row({"id": "t", "descriptionRaw": "RAW", "amount": 1, "type": "CREDIT"}, "a", "i", "BANK")
    assert row[5] == "RAW"


def test_other_row_mappers():
    assert bill_row({"id": "b", "dueDate": "2026-01-01", "closingDate": "2025-12-20"}, "a")[3] == date(2025, 12, 20)
    assert investment_row({"id": "i", "fixedAnnualRate": 10}, "it")[8] == pytest.approx(10.0)
    loan = loan_row({"id": "l", "outstandingBalance": 5, "cet": 2}, "it")
    assert loan[5] == pytest.approx(5.0)
    assert loan[12] == pytest.approx(2.0)


def test_store_roundtrip(tmp_path):
    path = tmp_path / "x" / "db.duckdb"
    with FinanceStore(path) as store:
        populate(store)
        counts = store.counts()
        assert counts["accounts"] == 2
        assert counts["transactions"] > 50
        assert counts["bills"] == 2
        assert counts["investments"] == 2
        assert counts["loans"] == 1
        assert counts["categories"] == 2
        assert store.category_translations()["08010000"] == "Compras online"
    assert path.exists()


def test_items_lifecycle(tmp_path):
    with FinanceStore(tmp_path / "db.duckdb") as store:
        assert store.add_item(" item-a ") is True
        assert store.add_item("item-a") is False
        assert [i["id"] for i in store.list_items()] == ["item-a"]
        assert store.item_watermark("item-a") == (None, None)
        assert store.item_watermark("missing") == (None, None)
        store.update_item({**ITEM, "id": "item-a"})
        # metadata is stored, but only a completed sync may advance the watermark
        assert store.list_items()[0]["connector_name"] == "MeuPluggy"
        assert store.item_watermark("item-a") == (None, None)
        store.update_item({**ITEM, "id": "item-a"}, synced=True)
        last_updated, synced = store.item_watermark("item-a")
        assert last_updated == datetime(2026, 11, 21, 10, 0)
        assert synced is not None
        # a later non-synced update keeps the previous synced_at
        store.update_item({**ITEM, "id": "item-a"})
        assert store.item_watermark("item-a")[1] is not None
        assert store.remove_item("item-a") is True
        assert store.remove_item("item-a") is False


def test_remove_item_deletes_derived_rows(tmp_path):
    with FinanceStore(tmp_path / "db.duckdb") as store:
        populate(store)
        store.remove_item(ITEM_ID)
        counts = store.counts()
        assert counts["transactions"] == counts["accounts"] == counts["bills"] == counts["loans"] == 0


def test_replace_transactions_replaces_account_rows(tmp_path):
    with FinanceStore(tmp_path / "db.duckdb") as store:
        populate(store)
        before = store.counts()["transactions"]
        row = transaction_row({"id": "only", "amount": -1, "type": "DEBIT"}, BANK_ID, ITEM_ID, "BANK")
        store.replace_transactions(BANK_ID, [row, row])  # duplicates are de-duplicated
        after = store.con.execute("SELECT count(*) FROM transactions WHERE account_id = ?", [BANK_ID]).fetchone()
        assert after == (1,)
        assert store.counts()["transactions"] < before


def test_replace_transactions_rolls_back_on_error(tmp_path, monkeypatch):
    with FinanceStore(tmp_path / "db.duckdb") as store:
        populate(store)
        before = store.counts()["transactions"]

        def boom(*_args, **_kwargs):
            raise RuntimeError("insert failed")

        monkeypatch.setattr(store, "_insert", boom)
        with pytest.raises(RuntimeError):
            store.replace_transactions(BANK_ID, [("x",)])
        monkeypatch.undo()
        assert store.counts()["transactions"] == before


def test_meta(tmp_path):
    with FinanceStore(tmp_path / "db.duckdb") as store:
        assert store.get_meta("k") is None
        store.set_meta("k", "v1")
        store.set_meta("k", "v2")
        assert store.get_meta("k") == "v2"


def test_in_memory_store():
    store = FinanceStore(":memory:")
    assert store.counts()["items"] == 0
    store.close()


def test_read_only_store_is_sandboxed(ro_store):
    con = ro_store.con
    assert con.execute("SELECT count(*) FROM v_transactions").fetchone()[0] > 0
    with pytest.raises(duckdb.Error):
        con.execute("DELETE FROM transactions")
    with pytest.raises(duckdb.Error):
        con.execute("SELECT * FROM read_csv('secret.csv')")
    with pytest.raises(duckdb.Error):
        con.execute("SET enable_external_access = true")


def test_view_spend_semantics(con):
    rows = dict(
        con.execute("SELECT id, spend FROM v_transactions WHERE id IN ('fat-1','est-1','nf-1','cdb-1')").fetchall()
    )
    assert rows == {"fat-1": 0.0, "est-1": -50.0, "nf-1": 55.9, "cdb-1": 0.0}


def test_unsynced_update_keeps_watermark(tmp_path):
    """Regression: `items add` / `connect` advanced last_updated_at, so the next sync skipped new data."""
    with FinanceStore(tmp_path / "db.duckdb") as store:
        store.update_item({**ITEM, "lastUpdatedAt": "2026-11-01T00:00:00Z"}, synced=True)
        store.update_item({**ITEM, "lastUpdatedAt": "2026-11-21T00:00:00Z", "status": "UPDATING"})
        last_updated, synced = store.item_watermark(ITEM_ID)
        assert last_updated == datetime(2026, 11, 1)
        assert synced is not None
        assert store.list_items()[0]["status"] == "UPDATING"


def test_dropped_account_rows_are_removed(tmp_path):
    """Regression: transactions/bills of accounts that disappeared kept counting as spending."""
    from tests.conftest import ACCOUNTS

    with FinanceStore(tmp_path / "db.duckdb") as store:
        populate(store)
        store.replace_accounts(ITEM_ID, [ACCOUNTS[0]])  # the card account is gone
        card_rows = store.con.execute(
            "SELECT (SELECT count(*) FROM transactions WHERE account_id = ?), "
            "(SELECT count(*) FROM bills WHERE account_id = ?)",
            [CARD_ID, CARD_ID],
        ).fetchone()
        assert card_rows == (0, 0)
        bank_rows = store.con.execute("SELECT count(*) FROM transactions WHERE account_id = ?", [BANK_ID]).fetchone()
        assert bank_rows is not None
        assert bank_rows[0] > 0


def test_remove_item_deletes_bills_of_orphan_accounts(tmp_path):
    with FinanceStore(tmp_path / "db.duckdb") as store:
        populate(store)
        store.con.execute("DELETE FROM accounts WHERE id = ?", [CARD_ID])  # simulate an orphaned card account
        store.remove_item(ITEM_ID)
        assert store.counts()["bills"] == 0
