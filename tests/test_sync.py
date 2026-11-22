"""Tests for the resilient Pluggy -> DuckDB sync."""

from datetime import UTC, datetime, timedelta

import pytest

from llmopenfinance.pluggy import PluggyError
from llmopenfinance.store import FinanceStore
from llmopenfinance.sync import ItemResult, SyncReport, _consent_warning, sync_all, sync_item
from tests.conftest import CARD_ID, ITEM, ITEM_ID


@pytest.fixture
def store(tmp_path):
    with FinanceStore(tmp_path / "db.duckdb") as s:
        s.add_item(ITEM_ID)
        yield s


def test_sync_all_downloads_everything(store, fake_pluggy):
    progress = []
    report = sync_all(fake_pluggy, store, on_item=progress.append)
    assert report.n_total == report.n_success == 1
    assert report.n_failed == 0
    totals = report.totals()
    assert totals["accounts"] == 2
    assert totals["transactions"] > 50
    assert totals["bills"] == 2
    assert totals["investments"] == 2
    assert totals["loans"] == 1
    assert progress[0].ok
    counts = store.counts()
    assert counts["categories"] == 2
    assert store.get_meta("last_sync") is not None
    assert store.get_meta("categories_synced") is not None


def test_sync_skips_unchanged_item(store, fake_pluggy):
    sync_all(fake_pluggy, store)
    fake_pluggy.calls.clear()
    result = sync_item(fake_pluggy, store, ITEM_ID)
    assert result.ok
    assert result.skipped
    assert "list_accounts" not in fake_pluggy.calls
    forced = sync_item(fake_pluggy, store, ITEM_ID, force=True)
    assert not forced.skipped
    assert "list_accounts" in fake_pluggy.calls


def test_sync_refetches_when_item_updated(store, fake_pluggy):
    sync_all(fake_pluggy, store)
    fake_pluggy.items[ITEM_ID]["lastUpdatedAt"] = "2026-11-22T10:00:00.000Z"
    assert not sync_item(fake_pluggy, store, ITEM_ID).skipped


def test_sync_replaces_deleted_transactions(store, fake_pluggy):
    sync_all(fake_pluggy, store)
    fake_pluggy.transactions[CARD_ID] = fake_pluggy.transactions[CARD_ID][:3]
    sync_item(fake_pluggy, store, ITEM_ID, force=True)
    remaining = store.con.execute("SELECT count(*) FROM transactions WHERE account_id = ?", [CARD_ID]).fetchone()
    assert remaining == (3,)


def test_item_fetch_failure_is_reported_not_raised(store, fake_pluggy):
    fake_pluggy.fail["get_item"] = PluggyError("down", 500)
    report = sync_all(fake_pluggy, store)
    assert report.n_failed == 1
    assert report.results[0].error == "down"


def test_transactions_failure_marks_item_failed(store, fake_pluggy):
    fake_pluggy.fail["iter_transactions"] = RuntimeError("boom")
    result = sync_item(fake_pluggy, store, ITEM_ID)
    assert not result.ok
    assert result.counts["accounts"] == 2
    assert "boom" in (result.error or "")


def test_optional_products_failures_become_warnings(store, fake_pluggy):
    fake_pluggy.fail["list_loans"] = PluggyError("not supported", 403, "FORBIDDEN")
    fake_pluggy.fail["list_investments"] = RuntimeError("weird")
    fake_pluggy.fail["list_bills"] = PluggyError("nope", 404)
    result = sync_item(fake_pluggy, store, ITEM_ID)
    assert result.ok
    assert result.counts["loans"] == 0
    assert any("empréstimos indisponível (403 FORBIDDEN)" in w for w in result.warnings)
    assert any("investimentos falhou" in w for w in result.warnings)
    assert any("faturas" in w for w in result.warnings)


def test_attention_status_and_consent_warnings(store, fake_pluggy):
    soon = (datetime.now(UTC) + timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    fake_pluggy.items[ITEM_ID] = {**ITEM, "status": "LOGIN_ERROR", "consentExpiresAt": soon}
    result = sync_item(fake_pluggy, store, ITEM_ID)
    assert any("LOGIN_ERROR" in w for w in result.warnings)
    assert any("expira em" in w for w in result.warnings)


def test_consent_warning_variants():
    now = datetime(2026, 11, 22)
    assert _consent_warning({}, now) is None
    assert _consent_warning({"consentExpiresAt": "2027-11-22T00:00:00Z"}, now) is None
    expired = _consent_warning({"consentExpiresAt": "2026-01-01T00:00:00Z"}, now)
    assert expired is not None
    assert "expirou" in expired


def test_categories_unavailable_does_not_block(store, fake_pluggy):
    fake_pluggy.fail["list_categories"] = PluggyError("premium", 403)
    report = sync_all(fake_pluggy, store)
    assert report.n_success == 1
    assert store.get_meta("categories_synced") is None


def test_report_properties():
    report = SyncReport((ItemResult("a", ok=True, counts={"x": 1}), ItemResult("b", ok=False, counts={"x": 2})))
    assert (report.n_total, report.n_success, report.n_failed) == (2, 1, 1)
    assert report.totals() == {"x": 3}


def test_transient_product_failure_is_retried_next_sync(store, fake_pluggy):
    """Regression: a 503 on bills stamped the watermark, so the following syncs skipped the item."""
    fake_pluggy.fail["list_bills"] = PluggyError("svc", 503)
    first = sync_item(fake_pluggy, store, ITEM_ID)
    assert first.ok
    assert first.counts["bills"] == 0
    del fake_pluggy.fail["list_bills"]
    second = sync_item(fake_pluggy, store, ITEM_ID)
    assert not second.skipped
    assert second.counts["bills"] == 2
    assert sync_item(fake_pluggy, store, ITEM_ID).skipped


def test_definitive_product_failure_still_marks_synced(store, fake_pluggy):
    fake_pluggy.fail["list_loans"] = PluggyError("not offered", 403)
    sync_item(fake_pluggy, store, ITEM_ID)
    assert sync_item(fake_pluggy, store, ITEM_ID).skipped
