"""Local DuckDB cache of the user's financial data.

DuckDB (columnar, embedded, SQL) fits this workload: a few thousand rows, analytical queries
(monthly aggregates, recurrence detection) and a safe read-only SQL surface for the LLM. No vector
DB is needed — questions are answered by SQL over structured data, not by semantic retrieval.

Connection model (DuckDB allows one read-write *or* many read-only handles per file):

* sync opens the file **read-write**, replaces data, then closes;
* chat opens it **read-only** with external access disabled and configuration locked, so even a
  hostile LLM-generated query cannot write, read files or attach databases.

Row mappers (``*_row``) are pure functions from Pluggy JSON to tuples; they never raise on
missing fields.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Self

import duckdb

from llmopenfinance.categorize import classify_kind, local_category, merchant_key, signed_flow
from llmopenfinance.constants import BRT

logger = logging.getLogger(__name__)

Row = tuple[Any, ...]

# table -> ordered (column, DuckDB type). The first column is the primary key.
SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    "items": (
        ("id", "VARCHAR"),
        ("connector_id", "INTEGER"),
        ("connector_name", "VARCHAR"),
        ("status", "VARCHAR"),
        ("execution_status", "VARCHAR"),
        ("error", "VARCHAR"),
        ("last_updated_at", "TIMESTAMP"),
        ("consent_expires_at", "TIMESTAMP"),
        ("added_at", "TIMESTAMP"),
        ("synced_at", "TIMESTAMP"),
    ),
    "accounts": (
        ("id", "VARCHAR"),
        ("item_id", "VARCHAR"),
        ("type", "VARCHAR"),
        ("subtype", "VARCHAR"),
        ("name", "VARCHAR"),
        ("marketing_name", "VARCHAR"),
        ("number", "VARCHAR"),
        ("balance", "DOUBLE"),
        ("currency_code", "VARCHAR"),
        ("credit_limit", "DOUBLE"),
        ("available_credit_limit", "DOUBLE"),
        ("balance_close_date", "DATE"),
        ("balance_due_date", "DATE"),
        ("minimum_payment", "DOUBLE"),
        ("raw", "VARCHAR"),
    ),
    "transactions": (
        ("id", "VARCHAR"),
        ("account_id", "VARCHAR"),
        ("item_id", "VARCHAR"),
        ("date_local", "DATE"),
        ("date_utc", "VARCHAR"),
        ("description", "VARCHAR"),
        ("amount", "DOUBLE"),
        ("flow", "DOUBLE"),
        ("currency_code", "VARCHAR"),
        ("type", "VARCHAR"),
        ("status", "VARCHAR"),
        ("category", "VARCHAR"),
        ("category_id", "VARCHAR"),
        ("category_local", "VARCHAR"),
        ("kind", "VARCHAR"),
        ("merchant", "VARCHAR"),
        ("merchant_key", "VARCHAR"),
        ("operation_type", "VARCHAR"),
        ("payment_method", "VARCHAR"),
        ("installment_number", "INTEGER"),
        ("total_installments", "INTEGER"),
        ("bill_id", "VARCHAR"),
        ("raw", "VARCHAR"),
    ),
    "categories": (
        ("id", "VARCHAR"),
        ("description", "VARCHAR"),
        ("description_translated", "VARCHAR"),
        ("parent_id", "VARCHAR"),
        ("parent_description", "VARCHAR"),
    ),
    "bills": (
        ("id", "VARCHAR"),
        ("account_id", "VARCHAR"),
        ("due_date", "DATE"),
        ("closing_date", "DATE"),
        ("total_amount", "DOUBLE"),
        ("currency_code", "VARCHAR"),
        ("minimum_payment_amount", "DOUBLE"),
        ("raw", "VARCHAR"),
    ),
    "investments": (
        ("id", "VARCHAR"),
        ("item_id", "VARCHAR"),
        ("name", "VARCHAR"),
        ("type", "VARCHAR"),
        ("subtype", "VARCHAR"),
        ("balance", "DOUBLE"),
        ("amount_profit", "DOUBLE"),
        ("currency_code", "VARCHAR"),
        ("annual_rate", "DOUBLE"),
        ("last_twelve_months_rate", "DOUBLE"),
        ("due_date", "DATE"),
        ("issuer", "VARCHAR"),
        ("status", "VARCHAR"),
        ("raw", "VARCHAR"),
    ),
    "loans": (
        ("id", "VARCHAR"),
        ("item_id", "VARCHAR"),
        ("product_name", "VARCHAR"),
        ("type", "VARCHAR"),
        ("contract_amount", "DOUBLE"),
        ("outstanding_balance", "DOUBLE"),
        ("currency_code", "VARCHAR"),
        ("total_installments", "INTEGER"),
        ("paid_installments", "INTEGER"),
        ("due_installments", "INTEGER"),
        ("installment_amount", "DOUBLE"),
        ("due_date", "DATE"),
        ("cet", "DOUBLE"),
        ("raw", "VARCHAR"),
    ),
    "meta": (("key", "VARCHAR"), ("value", "VARCHAR")),
}

VIEWS_SQL = """
CREATE OR REPLACE VIEW v_transactions AS
SELECT
    t.id,
    t.date_local AS date,
    strftime(t.date_local, '%Y-%m') AS month,
    t.description,
    t.merchant,
    t.merchant_key,
    t.category_local AS category,
    t.category AS category_pluggy,
    t.kind,
    t.flow,
    CASE WHEN t.kind IN ('expense', 'refund') THEN -t.flow ELSE 0 END AS spend,
    t.currency_code,
    t.type,
    t.status,
    t.payment_method,
    t.installment_number,
    t.total_installments,
    t.account_id,
    coalesce(a.marketing_name, a.name) AS account_name,
    a.type AS account_type,
    a.subtype AS account_subtype,
    t.item_id
FROM transactions t
LEFT JOIN accounts a ON a.id = t.account_id;
"""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def dig(obj: Any, *path: str) -> Any:
    """Safely read a nested key path from dicts.

    Args:
        obj: Root object.
        *path: Keys to follow.

    Returns:
        Any: The value, or ``None`` when any step is missing.
    """
    for key in path:
        if not isinstance(obj, Mapping):
            return None
        obj = obj.get(key)
    return obj


def to_float(value: Any) -> float | None:
    """Convert to float, returning ``None`` for missing/invalid values.

    Args:
        value: Input value.

    Returns:
        float | None: Parsed number.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    """Convert to int, returning ``None`` for missing/invalid values.

    Args:
        value: Input value.

    Returns:
        int | None: Parsed integer.
    """
    number = to_float(value)
    return None if number is None else int(number)


def parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp into a naive UTC ``datetime``.

    Args:
        value: ISO string (``Z`` suffix accepted) or ``None``.

    Returns:
        datetime | None: Naive UTC timestamp, or ``None`` if unparsable.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def local_date(value: Any) -> date | None:
    """Convert a Pluggy date to the Brazilian calendar date.

    Pluggy documents ``date`` as UTC ("convert to GMT-3"), but most connectors send calendar dates
    as midnight UTC (``...T00:00:00.000Z``); converting those would shift them to the previous
    day (and month). Midnight-UTC values are therefore treated as plain dates.

    Args:
        value: ISO date or timestamp string.

    Returns:
        date | None: Local (UTC-3) calendar date.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.date()
    utc = parsed.astimezone(UTC)
    return utc.date() if utc.time() == time(0) else utc.astimezone(BRT).date()


def _raw(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def item_row(item: Mapping[str, Any], added_at: datetime | None = None) -> Row:
    """Map a Pluggy item to an ``items`` row.

    Args:
        item: Pluggy item JSON.
        added_at: When the item was registered locally.

    Returns:
        Row: Tuple matching ``SCHEMA["items"]``.
    """
    error = item.get("error")
    error_text = None
    if isinstance(error, Mapping):
        error_text = f"{error.get('code', '')}: {error.get('message', '')}".strip(": ")
    return (
        str(item.get("id")),
        to_int(dig(item, "connector", "id")),
        dig(item, "connector", "name"),
        item.get("status"),
        item.get("executionStatus"),
        error_text,
        parse_timestamp(item.get("lastUpdatedAt")),
        parse_timestamp(item.get("consentExpiresAt")),
        added_at or datetime.now(UTC).replace(tzinfo=None),
        None,
    )


def account_row(account: Mapping[str, Any], item_id: str) -> Row:
    """Map a Pluggy account to an ``accounts`` row.

    Args:
        account: Pluggy account JSON.
        item_id: Owning item id.

    Returns:
        Row: Tuple matching ``SCHEMA["accounts"]``.
    """
    credit = account.get("creditData") or {}
    return (
        str(account.get("id")),
        str(account.get("itemId") or item_id),
        account.get("type"),
        account.get("subtype"),
        account.get("name"),
        account.get("marketingName"),
        account.get("number"),
        to_float(account.get("balance")),
        account.get("currencyCode") or "BRL",
        to_float(credit.get("creditLimit")),
        to_float(credit.get("availableCreditLimit")),
        local_date(credit.get("balanceCloseDate")),
        local_date(credit.get("balanceDueDate")),
        to_float(credit.get("minimumPayment")),
        _raw(account),
    )


def transaction_row(
    tx: Mapping[str, Any],
    account_id: str,
    item_id: str,
    account_type: str | None,
    translations: Mapping[str, str] | None = None,
) -> Row:
    """Map a Pluggy transaction to a ``transactions`` row, enriching it locally.

    Args:
        tx: Pluggy transaction JSON.
        account_id: Owning account id.
        item_id: Owning item id.
        account_type: ``BANK`` or ``CREDIT`` (needed to interpret the sign).
        translations: Category id -> pt-BR description (from ``GET /categories``).

    Returns:
        Row: Tuple matching ``SCHEMA["transactions"]``.
    """
    description = tx.get("description") or tx.get("descriptionRaw") or ""
    merchant = dig(tx, "merchant", "businessName") or dig(tx, "merchant", "name")
    category = tx.get("category")
    category_id = tx.get("categoryId")
    translated = (translations or {}).get(str(category_id)) if category_id else None
    category_loc = local_category(category, description, translated)
    flow = signed_flow(to_float(tx.get("amount")), tx.get("type"), account_type)
    card = tx.get("creditCardMetadata") or {}
    return (
        str(tx.get("id")),
        account_id,
        item_id,
        local_date(tx.get("date")),
        tx.get("date"),
        description,
        to_float(tx.get("amount")),
        flow,
        tx.get("currencyCode") or "BRL",
        tx.get("type"),
        tx.get("status"),
        category,
        str(category_id) if category_id else None,
        category_loc,
        classify_kind(flow, category_loc, account_type),
        merchant,
        merchant_key(description, merchant),
        tx.get("operationType"),
        dig(tx, "paymentData", "paymentMethod"),
        to_int(card.get("installmentNumber")),
        to_int(card.get("totalInstallments")),
        card.get("billId"),
        _raw(tx),
    )


def category_row(category: Mapping[str, Any]) -> Row:
    """Map a Pluggy category to a ``categories`` row.

    Args:
        category: Pluggy category JSON.

    Returns:
        Row: Tuple matching ``SCHEMA["categories"]``.
    """
    return (
        str(category.get("id")),
        category.get("description"),
        category.get("descriptionTranslated"),
        category.get("parentId"),
        category.get("parentDescription"),
    )


def bill_row(bill: Mapping[str, Any], account_id: str) -> Row:
    """Map a Pluggy credit-card bill to a ``bills`` row.

    Args:
        bill: Pluggy bill JSON.
        account_id: Owning credit-card account id.

    Returns:
        Row: Tuple matching ``SCHEMA["bills"]``.
    """
    return (
        str(bill.get("id")),
        account_id,
        local_date(bill.get("dueDate")),
        local_date(bill.get("billClosingDate") or bill.get("closingDate")),
        to_float(bill.get("totalAmount")),
        bill.get("totalAmountCurrencyCode") or "BRL",
        to_float(bill.get("minimumPaymentAmount")),
        _raw(bill),
    )


def investment_row(investment: Mapping[str, Any], item_id: str) -> Row:
    """Map a Pluggy investment to an ``investments`` row.

    Args:
        investment: Pluggy investment JSON.
        item_id: Owning item id.

    Returns:
        Row: Tuple matching ``SCHEMA["investments"]``.
    """
    return (
        str(investment.get("id")),
        item_id,
        investment.get("name"),
        investment.get("type"),
        investment.get("subtype"),
        to_float(investment.get("balance")),
        to_float(investment.get("amountProfit")),
        investment.get("currencyCode") or "BRL",
        to_float(investment.get("annualRate") or investment.get("fixedAnnualRate")),
        to_float(investment.get("lastTwelveMonthsRate")),
        local_date(investment.get("dueDate")),
        investment.get("issuer"),
        investment.get("status"),
        _raw(investment),
    )


def loan_row(loan: Mapping[str, Any], item_id: str) -> Row:
    """Map a Pluggy (Open Finance) loan to a ``loans`` row.

    Args:
        loan: Pluggy loan JSON.
        item_id: Owning item id.

    Returns:
        Row: Tuple matching ``SCHEMA["loans"]``.
    """
    installments = loan.get("installments") or {}
    payments = loan.get("payments") or {}
    return (
        str(loan.get("id")),
        item_id,
        loan.get("productName") or loan.get("productSubType"),
        loan.get("type"),
        to_float(loan.get("contractAmount")),
        to_float(payments.get("contractOutstandingBalance") or loan.get("outstandingBalance")),
        loan.get("currencyCode") or "BRL",
        to_int(installments.get("totalNumberOfInstallments")),
        to_int(installments.get("paidInstallments")),
        to_int(installments.get("dueInstallments")),
        to_float(loan.get("installmentAmount") or dig(installments, "installmentAmount")),
        local_date(loan.get("dueDate")),
        to_float(loan.get("CET") or loan.get("cet")),
        _raw(loan),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class FinanceStore:
    """DuckDB-backed cache of items, accounts, transactions, bills, investments and loans.

    Args:
        path: Database file (``":memory:"`` for an in-memory database).
        read_only: Open read-only and sandboxed (used by the chat tools).

    Note:
        A store instance is not safe for concurrent use from several threads; open one per thread
        (DuckDB allows many read-only handles).
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = str(path)
        self.read_only = read_only
        if read_only:
            self.con = duckdb.connect(
                self.path,
                read_only=True,
                config={
                    "enable_external_access": False,
                    "autoload_known_extensions": False,
                    "autoinstall_known_extensions": False,
                    "threads": 2,
                    "memory_limit": "1GB",
                },
            )
            try:
                self.con.execute("SET lock_configuration = true")
            except duckdb.InvalidInputException:
                # Another read-only handle in this process shares the instance and already locked it.
                logger.debug("DuckDB configuration already locked")
        else:
            if self.path != ":memory:":
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(self.path)
            self.init_schema()

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.con.close()

    def __enter__(self) -> Self:
        """Enter a context manager.

        Returns:
            FinanceStore: ``self``.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the connection when leaving a ``with`` block.

        Args:
            *exc_info: Exception information (ignored).
        """
        self.close()

    # ------------------------------------------------------------------ schema

    def init_schema(self) -> None:
        """Create tables and views if they do not exist (idempotent)."""
        for table, columns in SCHEMA.items():
            cols = ", ".join(f"{name} {kind}" for name, kind in columns)
            self.con.execute(f"CREATE TABLE IF NOT EXISTS {table} ({cols}, PRIMARY KEY ({columns[0][0]}))")
        self.con.execute(VIEWS_SQL)

    def _insert(self, table: str, rows: Sequence[Row], *, replace: bool = True) -> int:
        """Bulk insert rows using columnar ``UNNEST`` (much faster than ``executemany``).

        Args:
            table: Target table (must be in ``SCHEMA``).
            rows: Tuples ordered like ``SCHEMA[table]``.
            replace: Use ``INSERT OR REPLACE`` (upsert by primary key).

        Returns:
            int: Number of rows written.
        """
        if not rows:
            return 0
        columns = SCHEMA[table]
        deduped = list({row[0]: row for row in rows}.values())
        names = ", ".join(name for name, _ in columns)
        selects = ", ".join(f"unnest(?::{kind}[])" for _, kind in columns)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        params = [list(col) for col in zip(*deduped, strict=True)]
        self.con.execute(f"{verb} INTO {table} ({names}) SELECT {selects}", params)
        return len(deduped)

    # ------------------------------------------------------------------ items

    def add_item(self, item_id: str) -> bool:
        """Register an item id to be synced (no network call).

        Args:
            item_id: Pluggy item id.

        Returns:
            bool: ``True`` if newly added, ``False`` if already present.
        """
        item_id = item_id.strip()
        exists = self.con.execute("SELECT 1 FROM items WHERE id = ?", [item_id]).fetchone()
        if exists:
            return False
        self._insert("items", [item_row({"id": item_id})])
        return True

    def remove_item(self, item_id: str) -> bool:
        """Forget an item and every row derived from it.

        Args:
            item_id: Pluggy item id.

        Returns:
            bool: ``True`` if the item existed.
        """
        existed = self.con.execute("SELECT 1 FROM items WHERE id = ?", [item_id]).fetchone() is not None
        self.con.execute(
            "DELETE FROM bills WHERE account_id IN "
            "(SELECT id FROM accounts WHERE item_id = ? UNION SELECT account_id FROM transactions WHERE item_id = ?)",
            [item_id, item_id],
        )
        for table in ("transactions", "accounts", "investments", "loans"):
            self.con.execute(f"DELETE FROM {table} WHERE item_id = ?", [item_id])  # ruff: ignore[hardcoded-sql-expression] — fixed names
        self.con.execute("DELETE FROM items WHERE id = ?", [item_id])
        return existed

    def list_items(self) -> list[dict[str, Any]]:
        """List registered items with their last known status.

        Returns:
            list[dict[str, Any]]: One dict per item (columns of ``items``).
        """
        cursor = self.con.execute("SELECT * FROM items ORDER BY added_at, id")
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def update_item(self, item: Mapping[str, Any], *, synced: bool = False) -> None:
        """Store the latest item metadata from ``GET /items/{id}``.

        Args:
            item: Pluggy item JSON.
            synced: Also stamp ``synced_at`` and advance the ``last_updated_at`` watermark. Without it the
                stored watermark is kept, so the next sync still downloads the data.
        """
        previous = self.con.execute(
            "SELECT added_at, synced_at, last_updated_at FROM items WHERE id = ?", [str(item.get("id"))]
        ).fetchone()
        row = list(item_row(item, added_at=previous[0] if previous else None))
        if synced:
            row[-1] = datetime.now(UTC).replace(tzinfo=None)
        elif previous:
            # Only a completed sync may advance the watermark; otherwise the next sync would skip new data.
            row[-1] = previous[1]
            row[6] = previous[2]
        self._insert("items", [tuple(row)])

    def item_watermark(self, item_id: str) -> tuple[datetime | None, datetime | None]:
        """Return ``(last_updated_at, synced_at)`` stored for an item.

        Args:
            item_id: Pluggy item id.

        Returns:
            tuple[datetime | None, datetime | None]: Stored timestamps (``None`` when unknown).
        """
        row = self.con.execute("SELECT last_updated_at, synced_at FROM items WHERE id = ?", [item_id]).fetchone()
        return (row[0], row[1]) if row else (None, None)

    # ------------------------------------------------------------------ data

    def upsert_categories(self, categories: Iterable[Mapping[str, Any]]) -> int:
        """Insert or update Pluggy categories.

        Args:
            categories: Pluggy category objects.

        Returns:
            int: Rows written.
        """
        return self._insert("categories", [category_row(c) for c in categories])

    def category_translations(self) -> dict[str, str]:
        """Return category id -> pt-BR description from the cached category tree.

        Returns:
            dict[str, str]: Translations (empty when categories were never synced).
        """
        rows = self.con.execute(
            "SELECT id, description_translated FROM categories WHERE description_translated IS NOT NULL"
        ).fetchall()
        return {str(cid): str(text) for cid, text in rows}

    def replace_accounts(self, item_id: str, accounts: Sequence[Mapping[str, Any]]) -> int:
        """Replace all accounts of an item.

        Args:
            item_id: Pluggy item id.
            accounts: Pluggy account objects.

        Returns:
            int: Rows written.
        """
        keep = [str(a.get("id")) for a in accounts]
        for table in ("transactions", "bills"):
            # Drop rows of accounts that disappeared from the item, so they stop counting as spending.
            self.con.execute(
                f"DELETE FROM {table} WHERE account_id IN "  # ruff: ignore[hardcoded-sql-expression] — fixed names
                "(SELECT id FROM accounts WHERE item_id = ? "
                "UNION SELECT account_id FROM transactions WHERE item_id = ?) "
                "AND NOT list_contains(?::VARCHAR[], account_id)",
                [item_id, item_id, keep],
            )
        self.con.execute("DELETE FROM accounts WHERE item_id = ?", [item_id])
        return self._insert("accounts", [account_row(a, item_id) for a in accounts])

    def replace_transactions(self, account_id: str, rows: Sequence[Row]) -> int:
        """Replace all transactions of an account (handles rows Pluggy deleted/re-created).

        Args:
            account_id: Pluggy account id.
            rows: Rows from :func:`transaction_row`.

        Returns:
            int: Rows written.
        """
        self.con.execute("BEGIN TRANSACTION")
        try:
            self.con.execute("DELETE FROM transactions WHERE account_id = ?", [account_id])
            written = self._insert("transactions", rows)
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        return written

    def replace_bills(self, account_id: str, bills: Sequence[Mapping[str, Any]]) -> int:
        """Replace all bills of a credit-card account.

        Args:
            account_id: Pluggy account id.
            bills: Pluggy bill objects.

        Returns:
            int: Rows written.
        """
        self.con.execute("DELETE FROM bills WHERE account_id = ?", [account_id])
        return self._insert("bills", [bill_row(b, account_id) for b in bills])

    def replace_investments(self, item_id: str, investments: Sequence[Mapping[str, Any]]) -> int:
        """Replace all investments of an item.

        Args:
            item_id: Pluggy item id.
            investments: Pluggy investment objects.

        Returns:
            int: Rows written.
        """
        self.con.execute("DELETE FROM investments WHERE item_id = ?", [item_id])
        return self._insert("investments", [investment_row(i, item_id) for i in investments])

    def replace_loans(self, item_id: str, loans: Sequence[Mapping[str, Any]]) -> int:
        """Replace all loans of an item.

        Args:
            item_id: Pluggy item id.
            loans: Pluggy loan objects.

        Returns:
            int: Rows written.
        """
        self.con.execute("DELETE FROM loans WHERE item_id = ?", [item_id])
        return self._insert("loans", [loan_row(loan, item_id) for loan in loans])

    # ------------------------------------------------------------------ meta / stats

    def set_meta(self, key: str, value: str) -> None:
        """Store a key/value pair (e.g. ``last_sync``).

        Args:
            key: Key.
            value: Value.
        """
        self._insert("meta", [(key, value)])

    def get_meta(self, key: str) -> str | None:
        """Read a key/value pair.

        Args:
            key: Key.

        Returns:
            str | None: Stored value.
        """
        row = self.con.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        return None if row is None else str(row[0])

    def counts(self) -> dict[str, int]:
        """Count rows per data table.

        Returns:
            dict[str, int]: Table name -> row count.
        """
        tables = ("items", "accounts", "transactions", "bills", "investments", "loans", "categories")
        return {
            table: int((self.con.execute(f"SELECT count(*) FROM {table}").fetchone() or (0,))[0])  # ruff: ignore[hardcoded-sql-expression] — fixed names
            for table in tables
        }
