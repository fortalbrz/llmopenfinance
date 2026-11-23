"""Synchronize Pluggy data into the local DuckDB store.

Non-stop batch semantics: one failing item (expired consent, 404, rate limit…) or one failing
product (loans not supported by the connector…) is logged and reported, never aborts the sync.

Incremental strategy: ``GET /items/{id}`` is cheap; when its ``lastUpdatedAt`` matches the stored
watermark and the item was already synced, the data download is skipped (``force`` overrides).
Otherwise each account's **full** history is re-pulled and replaced: Pluggy re-creates card
installments and pending rows with new ids, and older installments can be missed by date-filtered
queries, so a full replace is the only correct (and, at personal volumes, cheap) option.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from llmopenfinance.i18n import format_date, t
from llmopenfinance.pluggy import PluggyClient, PluggyError
from llmopenfinance.store import FinanceStore, parse_timestamp, transaction_row

logger = logging.getLogger(__name__)

#: Item statuses meaning the data cannot be refreshed until the user re-authorizes.
ATTENTION_STATUSES = frozenset({"LOGIN_ERROR", "OUTDATED", "WAITING_USER_INPUT", "WAITING_USER_ACTION"})


@dataclass(frozen=True)
class ItemResult:
    """Outcome of syncing one item.

    Attributes:
        item_id: Pluggy item id.
        ok: ``True`` when the item's core data (accounts + transactions) was refreshed or up to date.
        skipped: ``True`` when nothing changed since the last sync.
        counts: Rows written per table.
        warnings: Non-fatal problems (product unavailable, consent expiring…).
        error: Fatal error message for this item, if any.
    """

    item_id: str
    ok: bool
    skipped: bool = False
    counts: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class SyncReport:
    """Immutable summary of a sync run.

    Attributes:
        results: Per-item results.
    """

    results: tuple[ItemResult, ...]

    @property
    def n_total(self) -> int:
        """Number of items processed."""
        return len(self.results)

    @property
    def n_failed(self) -> int:
        """Number of items that failed."""
        return sum(1 for r in self.results if not r.ok)

    @property
    def n_success(self) -> int:
        """Number of items refreshed or up to date."""
        return self.n_total - self.n_failed

    def totals(self) -> dict[str, int]:
        """Sum rows written per table across items.

        Returns:
            dict[str, int]: Table -> rows.
        """
        out: dict[str, int] = {}
        for result in self.results:
            for table, value in result.counts.items():
                out[table] = out.get(table, 0) + value
        return out


def _consent_warning(item: Mapping[str, Any], now: datetime) -> str | None:
    """Warn when the item's consent is expired or expires within 30 days.

    Args:
        item: Pluggy item JSON.
        now: Current naive-UTC time.

    Returns:
        str | None: Warning text, if any.
    """
    expires = parse_timestamp(item.get("consentExpiresAt"))
    if expires is None:
        return None
    if expires <= now:
        return t("sync.warning.consent_expired", date=format_date(expires.date()))
    if expires - now <= timedelta(days=30):
        return t("sync.warning.consent_expiring", date=format_date(expires.date()))
    return None


def _is_transient(exc: Exception) -> bool:
    """Whether a failure may succeed on retry (network, 429, 5xx, unexpected errors).

    Args:
        exc: The exception raised by a product fetch.

    Returns:
        bool: ``False`` only for definitive API answers such as 403/404 (product not offered).
    """
    if not isinstance(exc, PluggyError):
        return True
    return exc.status_code in {0, 408, 429} or exc.status_code >= 500


def _try(label: str, func: Callable[[], int], warnings: list[str], transient: list[str]) -> int:
    """Run an optional product fetch; failures become warnings.

    Args:
        label: Product name for messages.
        func: Callable returning rows written.
        warnings: List collecting warnings.
        transient: List collecting labels of failures worth retrying on the next sync.

    Returns:
        int: Rows written (0 on failure).
    """
    try:
        return func()
    except PluggyError as exc:
        logger.warning("%s unavailable: %s", label, exc)
        warnings.append(t("sync.warning.unavailable", label=label, status=exc.status_code, code=exc.code or "").strip())
        if _is_transient(exc):
            transient.append(label)
    except Exception:
        logger.exception("%s failed", label)
        warnings.append(t("sync.warning.failed", label=label))
        transient.append(label)
    return 0


def _sync_accounts(
    client: PluggyClient,
    store: FinanceStore,
    item_id: str,
    counts: dict[str, int],
    warnings: list[str],
    transient: list[str],
) -> None:
    """Replace accounts, their full transaction history and card bills of one item.

    Args:
        client: Pluggy client.
        store: Read-write finance store.
        item_id: Pluggy item id.
        counts: Mutable row counters (updated in place, so partial progress is reported on failure).
        warnings: Mutable list collecting non-fatal warnings.
        transient: Mutable list collecting retryable product failures.
    """
    accounts = client.list_accounts(item_id)
    counts["accounts"] = store.replace_accounts(item_id, accounts)
    translations = store.category_translations()
    counts["transactions"] = 0
    for account in accounts:
        account_id = str(account.get("id"))
        account_type = account.get("type")
        rows = [
            transaction_row(tx, account_id, item_id, account_type, translations)
            for tx in client.iter_transactions(account_id)
        ]
        counts["transactions"] += store.replace_transactions(account_id, rows)
        if str(account_type).upper() == "CREDIT":
            counts["bills"] = counts.get("bills", 0) + _try(
                t("sync.product.bills"),
                lambda a=account_id: store.replace_bills(a, client.list_bills(a)),
                warnings,
                transient,
            )


def sync_item(client: PluggyClient, store: FinanceStore, item_id: str, *, force: bool = False) -> ItemResult:
    """Synchronize one item. Never raises: failures are returned in the result.

    Args:
        client: Pluggy client.
        store: Read-write finance store.
        item_id: Pluggy item id.
        force: Download data even when the item did not change since the last sync.

    Returns:
        ItemResult: What happened.
    """
    warnings: list[str] = []
    now = datetime.now(UTC).replace(tzinfo=None)
    try:
        item = client.get_item(item_id)
    except Exception as exc:
        logger.exception("Item %s: cannot fetch item", item_id)
        return ItemResult(item_id, ok=False, error=str(exc))

    status = str(item.get("status") or "")
    if status in ATTENTION_STATUSES:
        warnings.append(t("sync.warning.attention", status=status))
    consent = _consent_warning(item, now)
    if consent:
        warnings.append(consent)

    last_updated, synced_at = store.item_watermark(item_id)
    remote_updated = parse_timestamp(item.get("lastUpdatedAt"))
    if not force and synced_at is not None and remote_updated is not None and remote_updated == last_updated:
        store.update_item(item, synced=True)
        return ItemResult(item_id, ok=True, skipped=True, warnings=tuple(warnings))

    counts: dict[str, int] = {}
    transient: list[str] = []
    try:
        _sync_accounts(client, store, item_id, counts, warnings, transient)
    except Exception as exc:
        logger.exception("Item %s: accounts/transactions sync failed", item_id)
        return ItemResult(item_id, ok=False, counts=counts, warnings=tuple(warnings), error=str(exc))

    counts["investments"] = _try(
        t("sync.product.investments"),
        lambda: store.replace_investments(item_id, client.list_investments(item_id)),
        warnings,
        transient,
    )
    counts["loans"] = _try(
        t("sync.product.loans"), lambda: store.replace_loans(item_id, client.list_loans(item_id)), warnings, transient
    )
    # A transient product failure keeps the old watermark, so the next sync retries instead of skipping.
    store.update_item(item, synced=not transient)
    return ItemResult(item_id, ok=True, counts=counts, warnings=tuple(warnings))


def sync_categories(client: PluggyClient, store: FinanceStore) -> int:
    """Cache Pluggy's category tree (optional; free plans may not expose it).

    Args:
        client: Pluggy client.
        store: Read-write finance store.

    Returns:
        int: Categories written (0 when unavailable).
    """
    try:
        return store.upsert_categories(client.list_categories())
    except Exception as exc:
        logger.warning("Categories unavailable: %s", exc)
        return 0


def sync_all(
    client: PluggyClient,
    store: FinanceStore,
    *,
    force: bool = False,
    on_item: Callable[[ItemResult], None] | None = None,
) -> SyncReport:
    """Synchronize every registered item.

    Args:
        client: Pluggy client.
        store: Read-write finance store.
        force: Re-download even unchanged items.
        on_item: Optional progress callback invoked after each item.

    Returns:
        SyncReport: Immutable per-item report.
    """
    if store.get_meta("categories_synced") is None or force:
        written = sync_categories(client, store)
        if written:
            store.set_meta("categories_synced", datetime.now(UTC).isoformat())

    results: list[ItemResult] = []
    for item in store.list_items():
        result = sync_item(client, store, str(item["id"]), force=force)
        results.append(result)
        if on_item is not None:
            on_item(result)

    report = SyncReport(tuple(results))
    store.set_meta("last_sync", datetime.now(UTC).isoformat())
    logger.info(
        "Sync complete: %d ok, %d failed (%.2f%%).",
        report.n_success,
        report.n_failed,
        100.0 * report.n_failed / max(report.n_total, 1),
    )
    return report
