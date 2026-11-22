"""Deterministic financial analytics over the local DuckDB store.

Every function takes a DuckDB connection (normally the sandboxed read-only one), never mutates
state and returns JSON-friendly Python structures. The LLM reaches these through tools
(:mod:`llmopenfinance.tools`); numbers in answers therefore come from SQL, not from the model.

Conventions (see ``v_transactions`` in :mod:`llmopenfinance.store`):

* ``flow`` < 0 is money out; ``spend`` = expenses minus refunds (card bill payments, own-account
  transfers and investments are excluded to avoid double counting);
* months are ``YYYY-MM`` strings in Brazilian local time; amounts are BRL floats.
"""

from __future__ import annotations

import itertools
import re
import statistics
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import duckdb

from llmopenfinance.categorize import SUBSCRIPTION_CATEGORIES, strip_accents
from llmopenfinance.constants import BRT, SQL_TIMEOUT_SECONDS, TOOL_ROW_LIMIT

Record = dict[str, Any]

KINDS = ("expense", "refund", "income", "transfer", "investment")

# A final ';' followed only by whitespace/comments (``SELECT 1; -- note``) — removed before wrapping the query.
_TRAILING_SQL_RE = re.compile(r";(?:\s|--[^\n]*|/\*.*?\*/)*\Z", re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def today_brt() -> date:
    """Return today's date in Brazilian time (UTC-3).

    Returns:
        date: Local date.
    """
    return datetime.now(BRT).date()


def brl(value: float | None) -> str:
    """Format a number as Brazilian Real, e.g. ``R$ 1.234,56``.

    Args:
        value: Amount (``None`` renders as ``"—"``).

    Returns:
        str: Formatted amount.
    """
    if value is None:
        return "—"
    text = f"{abs(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"-R$ {text}" if value < 0 else f"R$ {text}"


def month_start(today: date, months_back: int) -> date:
    """First day of the month ``months_back`` months before ``today``'s month.

    Args:
        today: Reference date.
        months_back: 0 = current month, 1 = previous month, ...

    Returns:
        date: First day of that month.
    """
    index = today.year * 12 + (today.month - 1) - months_back
    return date(index // 12, index % 12 + 1, 1)


def _records(con: duckdb.DuckDBPyConnection, sql: str, params: Sequence[Any] = ()) -> list[Record]:
    """Run a query and return rows as dicts.

    Args:
        con: DuckDB connection.
        sql: Query.
        params: Positional parameters.

    Returns:
        list[Record]: One dict per row.
    """
    cursor = con.execute(sql, list(params))
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _like(text: str) -> str:
    """Build an accent/case-insensitive ``LIKE`` pattern.

    Args:
        text: User/LLM supplied text.

    Returns:
        str: ``%text%`` folded for comparison against ``strip_accents(lower(col))``.
    """
    return f"%{strip_accents(text.strip())}%"


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def accounts_overview(con: duckdb.DuckDBPyConnection) -> list[Record]:
    """List accounts with balances and credit-card data.

    Args:
        con: DuckDB connection.

    Returns:
        list[Record]: Accounts (``name``, ``type``, ``subtype``, ``balance``, credit fields, item status).
    """
    return _records(
        con,
        """
        SELECT a.id, coalesce(a.marketing_name, a.name) AS name, a.type, a.subtype, a.number,
               round(a.balance, 2) AS balance, a.currency_code,
               a.credit_limit, a.available_credit_limit, a.balance_due_date, a.balance_close_date,
               a.minimum_payment, i.connector_name AS institution, i.status AS connection_status
        FROM accounts a LEFT JOIN items i ON i.id = a.item_id
        ORDER BY a.type, name
        """,
    )


def spending_by_category(
    con: duckdb.DuckDBPyConnection,
    *,
    months: int = 3,
    category: str | None = None,
    today: date | None = None,
) -> list[Record]:
    """Monthly spending (expenses net of refunds) per local category.

    Args:
        con: DuckDB connection.
        months: Number of calendar months including the current one.
        category: Optional case/accent-insensitive filter (matches local or Pluggy category).
        today: Reference date (defaults to today in BRT).

    Returns:
        list[Record]: Rows ``month``, ``category``, ``total``, ``transactions`` ordered by month, total desc.
    """
    start = month_start(today or today_brt(), max(months, 1) - 1)
    params: list[Any] = [start]
    where = ""
    if category:
        where = (
            "AND (strip_accents(lower(category)) LIKE ? OR strip_accents(lower(coalesce(category_pluggy, ''))) LIKE ?)"
        )
        params += [_like(category), _like(category)]
    return _records(
        con,
        f"""
        SELECT month, category, round(sum(spend), 2) AS total, count(*) AS transactions
        FROM v_transactions
        WHERE kind IN ('expense', 'refund') AND date >= ? {where}
        GROUP BY month, category
        HAVING round(sum(spend), 2) <> 0
        ORDER BY month DESC, total DESC
        """,  # ruff: ignore[hardcoded-sql-expression] — only a fixed fragment is interpolated
        params,
    )


def query_transactions(
    con: duckdb.DuckDBPyConnection,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    category: str | None = None,
    search: str | None = None,
    kind: str | None = None,
    account_type: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    order: str = "date_desc",
    limit: int = 50,
) -> dict[str, Any]:
    """Filter individual transactions.

    Args:
        con: DuckDB connection.
        date_from: Inclusive ``YYYY-MM-DD``.
        date_to: Inclusive ``YYYY-MM-DD``.
        category: Category substring (local or Pluggy).
        search: Substring of the description / merchant.
        kind: One of :data:`KINDS`.
        account_type: ``BANK`` or ``CREDIT``.
        min_amount: Minimum absolute amount.
        max_amount: Maximum absolute amount.
        order: ``date_desc``, ``date_asc`` or ``amount_desc``.
        limit: Maximum rows (capped at ``TOOL_ROW_LIMIT``).

    Returns:
        dict[str, Any]: ``rows`` plus aggregate ``count``/``total_flow``/``total_spend`` over *all* matches.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if date_from:
        clauses.append("date >= CAST(? AS DATE)")
        params.append(date_from)
    if date_to:
        clauses.append("date <= CAST(? AS DATE)")
        params.append(date_to)
    if category:
        clauses.append(
            "(strip_accents(lower(category)) LIKE ? OR strip_accents(lower(coalesce(category_pluggy, ''))) LIKE ?)"
        )
        params += [_like(category), _like(category)]
    if search:
        clauses.append(
            "(strip_accents(lower(description)) LIKE ? OR strip_accents(lower(coalesce(merchant, ''))) LIKE ?)"
        )
        params += [_like(search), _like(search)]
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if account_type:
        clauses.append("account_type = ?")
        params.append(account_type.upper())
    if min_amount is not None:
        clauses.append("abs(flow) >= ?")
        params.append(min_amount)
    if max_amount is not None:
        clauses.append("abs(flow) <= ?")
        params.append(max_amount)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    order_sql = {"date_asc": "date ASC, id", "amount_desc": "abs(flow) DESC, date DESC"}.get(order, "date DESC, id")
    capped = max(1, min(int(limit), TOOL_ROW_LIMIT))
    summary = _records(
        con,
        f"SELECT count(*) AS count, round(coalesce(sum(flow), 0), 2) AS total_flow, "  # ruff: ignore[hardcoded-sql-expression]
        f"round(coalesce(sum(spend), 0), 2) AS total_spend FROM v_transactions {where}",
        params,
    )[0]
    rows = _records(
        con,
        f"""
        SELECT date, description, merchant, category, kind, round(flow, 2) AS amount, account_name, account_type,
               installment_number, total_installments, status
        FROM v_transactions {where}
        ORDER BY {order_sql}
        LIMIT {capped}
        """,  # ruff: ignore[hardcoded-sql-expression] — clauses are fixed fragments; values are bound parameters
        params,
    )
    return {**summary, "returned": len(rows), "rows": rows}


def cashflow(con: duckdb.DuckDBPyConnection, *, months: int = 6, today: date | None = None) -> list[Record]:
    """Monthly income, spending, net result and savings rate.

    Args:
        con: DuckDB connection.
        months: Number of calendar months including the current one.
        today: Reference date.

    Returns:
        list[Record]: ``month``, ``income``, ``spending``, ``invested_net``, ``net``, ``savings_rate``.
    """
    start = month_start(today or today_brt(), max(months, 1) - 1)
    return _records(
        con,
        """
        SELECT month,
               round(sum(CASE WHEN kind = 'income' THEN flow ELSE 0 END), 2) AS income,
               round(sum(spend), 2) AS spending,
               round(-sum(CASE WHEN kind = 'investment' THEN flow ELSE 0 END), 2) AS invested_net,
               round(sum(CASE WHEN kind = 'income' THEN flow ELSE 0 END) - sum(spend), 2) AS net,
               CASE WHEN sum(CASE WHEN kind = 'income' THEN flow ELSE 0 END) > 0
                    THEN round(1 - sum(spend) / sum(CASE WHEN kind = 'income' THEN flow ELSE 0 END), 3)
               END AS savings_rate
        FROM v_transactions
        WHERE date >= ?
        GROUP BY month
        ORDER BY month DESC
        """,
        [start],
    )


def top_merchants(
    con: duckdb.DuckDBPyConnection, *, months: int = 3, limit: int = 10, today: date | None = None
) -> list[Record]:
    """Counterparties with the highest spending.

    Args:
        con: DuckDB connection.
        months: Number of calendar months including the current one.
        limit: Maximum rows.
        today: Reference date.

    Returns:
        list[Record]: ``merchant``, ``category``, ``total``, ``transactions``.
    """
    start = month_start(today or today_brt(), max(months, 1) - 1)
    capped = max(1, min(int(limit), TOOL_ROW_LIMIT))
    return _records(
        con,
        f"""
        SELECT coalesce(any_value(merchant), any_value(description)) AS merchant, mode(category) AS category,
               round(sum(spend), 2) AS total, count(*) AS transactions
        FROM v_transactions
        WHERE kind IN ('expense', 'refund') AND date >= ? AND merchant_key <> ''
        GROUP BY merchant_key
        HAVING sum(spend) > 0
        ORDER BY total DESC
        LIMIT {capped}
        """,  # ruff: ignore[hardcoded-sql-expression] — integer limit
        [start],
    )


def bills_overview(con: duckdb.DuckDBPyConnection, *, limit: int = 6) -> list[Record]:
    """Most recent credit-card bills.

    Args:
        con: DuckDB connection.
        limit: Maximum bills.

    Returns:
        list[Record]: ``account``, ``due_date``, ``closing_date``, ``total_amount``, ``minimum_payment_amount``.
    """
    capped = max(1, min(int(limit), TOOL_ROW_LIMIT))
    return _records(
        con,
        f"""
        SELECT coalesce(a.marketing_name, a.name) AS account, b.due_date, b.closing_date,
               round(b.total_amount, 2) AS total_amount, round(b.minimum_payment_amount, 2) AS minimum_payment_amount
        FROM bills b LEFT JOIN accounts a ON a.id = b.account_id
        ORDER BY b.due_date DESC NULLS LAST
        LIMIT {capped}
        """,  # ruff: ignore[hardcoded-sql-expression] — integer limit
    )


def investments_overview(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Investment positions and totals per type.

    Args:
        con: DuckDB connection.

    Returns:
        dict[str, Any]: ``total``, ``by_type`` and ``positions``.
    """
    positions = _records(
        con,
        """
        SELECT name, type, subtype, round(balance, 2) AS balance, round(amount_profit, 2) AS profit,
               annual_rate, last_twelve_months_rate, due_date, issuer, status
        FROM investments ORDER BY balance DESC NULLS LAST
        """,
    )
    by_type = _records(
        con,
        "SELECT type, round(sum(balance), 2) AS total, count(*) AS positions FROM investments "
        "GROUP BY type ORDER BY total DESC",
    )
    total = round(sum(float(p["balance"] or 0) for p in positions), 2)
    return {"total": total, "by_type": by_type, "positions": positions[:TOOL_ROW_LIMIT]}


def loans_overview(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Loans and financing contracts.

    Args:
        con: DuckDB connection.

    Returns:
        dict[str, Any]: ``total_outstanding`` and ``loans``.
    """
    loans = _records(
        con,
        """
        SELECT product_name, type, round(contract_amount, 2) AS contract_amount,
               round(outstanding_balance, 2) AS outstanding_balance, total_installments, paid_installments,
               due_installments, round(installment_amount, 2) AS installment_amount, due_date, cet
        FROM loans ORDER BY outstanding_balance DESC NULLS LAST
        """,
    )
    total = round(sum(float(loan["outstanding_balance"] or 0) for loan in loans), 2)
    return {"total_outstanding": total, "loans": loans}


# ---------------------------------------------------------------------------
# Recurring charges / subscriptions
# ---------------------------------------------------------------------------


def _frequency(median_interval: float) -> str | None:
    """Name the periodicity implied by the median interval between charges.

    Args:
        median_interval: Median days between consecutive charges.

    Returns:
        str | None: ``mensal``, ``anual``, ``semanal`` or ``None``.
    """
    if 25 <= median_interval <= 35:
        return "mensal"
    if 330 <= median_interval <= 400:
        return "anual"
    if 6 <= median_interval <= 8:
        return "semanal"
    return None


def _confidence(n: int, frequency: str | None, cv: float, subscription_like: bool) -> str | None:
    """Score how likely a group of charges is a recurring service.

    Args:
        n: Number of charges.
        frequency: Periodicity from :func:`_frequency`.
        cv: Coefficient of variation of the amounts.
        subscription_like: Category typical of subscriptions.

    Returns:
        str | None: ``alta``, ``média``, ``baixa`` or ``None`` (not recurring).
    """
    if frequency and n >= 3 and cv <= 0.15:
        return "alta"
    if frequency and n >= 2 and (cv <= 0.10 or subscription_like):
        return "média"
    if subscription_like and n >= 2 and cv <= 0.25:
        return "média"
    if subscription_like and n == 1:
        return "baixa"
    return None


def find_recurring(rows: Sequence[Record], today: date) -> list[Record]:
    """Detect recurring charges (subscriptions, bills, memberships) from expense rows.

    A group (same normalized merchant) is recurring when charges repeat at a regular cadence
    (monthly 25–35 days, annual, weekly) with stable amounts (coefficient of variation <= 15 %),
    or belongs to a subscription-typical category (streaming, digital services, telecom, gym…).
    Installment purchases (``total_installments > 1``) are excluded.

    Args:
        rows: Dicts with ``merchant_key``, ``description``, ``category``, ``date``, ``spend``,
            ``account_name``, ``total_installments``.
        today: Reference date (to flag possibly cancelled subscriptions).

    Returns:
        list[Record]: Detected recurring charges, active and most expensive first.
    """
    groups: dict[str, list[Record]] = defaultdict(list)
    for row in rows:
        if (row.get("spend") or 0) <= 0 or (row.get("total_installments") or 0) > 1 or not row.get("merchant_key"):
            continue
        groups[str(row["merchant_key"])].append(row)

    found: list[Record] = []
    for key, charges in groups.items():
        charges.sort(key=lambda r: r["date"])
        amounts = [float(c["spend"]) for c in charges]
        dates: list[date] = [c["date"] for c in charges]
        intervals = [(b - a).days for a, b in itertools.pairwise(dates) if (b - a).days > 0]
        median_interval = statistics.median(intervals) if intervals else 0.0
        mean = statistics.fmean(amounts)
        cv = statistics.pstdev(amounts) / mean if mean > 0 and len(amounts) > 1 else 0.0
        category = Counter(str(c["category"]) for c in charges).most_common(1)[0][0]
        frequency = _frequency(median_interval)
        confidence = _confidence(len(charges), frequency, cv, category in SUBSCRIPTION_CATEGORIES)
        if confidence is None:
            continue
        if confidence == "baixa" and (today - dates[-1]).days > 40:
            continue
        if frequency is None and len(charges) >= 2:
            # No regular cadence (e.g. charged twice a month): estimate from the observed spend rate.
            frequency = "irregular"
            span_days = (dates[-1] - dates[0]).days + (median_interval or 30)
            per_month = sum(amounts) / max(1.0, span_days / 30.44)
        else:
            frequency = frequency or "mensal"
            per_month = {"mensal": mean, "anual": mean / 12, "semanal": mean * 52 / 12}[frequency]
        max_gap = {"mensal": 45, "irregular": 45, "anual": 400, "semanal": 15}[frequency]
        found.append(
            {
                "name": Counter(str(c["description"]) for c in charges).most_common(1)[0][0],
                "merchant_key": key,
                "category": category,
                "frequency": frequency,
                "average_amount": round(mean, 2),
                "last_amount": round(amounts[-1], 2),
                "last_date": dates[-1].isoformat(),
                "occurrences": len(charges),
                "monthly_estimate": round(per_month, 2),
                "annual_estimate": round(per_month * 12, 2),
                "confidence": confidence,
                "active": (today - dates[-1]).days <= max_gap,
                "account": charges[-1].get("account_name"),
            }
        )
    found.sort(key=lambda r: (not r["active"], -r["monthly_estimate"]))
    return found


def detect_subscriptions(
    con: duckdb.DuckDBPyConnection, *, months: int = 13, today: date | None = None, include_inactive: bool = False
) -> dict[str, Any]:
    """Detect subscriptions and other recurring charges.

    Args:
        con: DuckDB connection.
        months: History window in calendar months.
        today: Reference date.
        include_inactive: Also return charges that seem cancelled.

    Returns:
        dict[str, Any]: ``subscriptions`` plus ``monthly_total``/``annual_total`` of the active ones.
    """
    ref = today or today_brt()
    rows = _records(
        con,
        """
        SELECT merchant_key, description, category, date, spend, account_name, total_installments
        FROM v_transactions
        WHERE kind = 'expense' AND date >= ?
        """,
        [month_start(ref, max(months, 1) - 1)],
    )
    found = find_recurring(rows, ref)
    if not include_inactive:
        found = [r for r in found if r["active"]]
    active = [r for r in found if r["active"]]
    return {
        "subscriptions": found[:TOOL_ROW_LIMIT],
        "monthly_total": round(sum(r["monthly_estimate"] for r in active), 2),
        "annual_total": round(sum(r["annual_estimate"] for r in active), 2),
    }


# ---------------------------------------------------------------------------
# Guarded SQL
# ---------------------------------------------------------------------------


def run_select(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    limit: int = TOOL_ROW_LIMIT,
    timeout: float = SQL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run one read-only ``SELECT`` with a row cap and a timeout.

    Defence in depth: the connection itself is read-only with external access disabled; this
    function additionally accepts exactly one ``SELECT`` statement and interrupts slow queries.

    Args:
        con: DuckDB connection (should be the read-only sandbox).
        sql: A single SELECT (``WITH`` allowed).
        limit: Maximum rows returned.
        timeout: Seconds before the query is interrupted.

    Returns:
        dict[str, Any]: ``columns``, ``rows``, ``row_count`` and ``truncated``.

    Raises:
        ValueError: If the SQL is not exactly one SELECT statement.
    """
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1:
        raise ValueError("Envie exatamente uma instrução SQL.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError("Somente consultas SELECT são permitidas.")
    query = _TRAILING_SQL_RE.sub("", statements[0].query).strip().rstrip(";").strip()
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        cursor = con.execute(f"SELECT * FROM (\n{query}\n) AS q LIMIT {int(limit) + 1}")  # ruff: ignore[hardcoded-sql-expression] — guarded above
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
    finally:
        timer.cancel()
    truncated = len(rows) > limit
    rows = rows[:limit]
    return {"columns": columns, "rows": [list(r) for r in rows], "row_count": len(rows), "truncated": truncated}


# ---------------------------------------------------------------------------
# Snapshot for the system prompt
# ---------------------------------------------------------------------------


def _snapshot_accounts(con: duckdb.DuckDBPyConnection) -> list[str]:
    lines = ["## Contas"]
    for acc in accounts_overview(con):
        if acc["type"] == "CREDIT":
            lines.append(
                f"- {acc['name']} (cartão de crédito): fatura atual {brl(acc['balance'])}, "
                f"limite disponível {brl(acc['available_credit_limit'])} de {brl(acc['credit_limit'])}, "
                f"vencimento {acc['balance_due_date'] or '—'}"
            )
        else:
            lines.append(f"- {acc['name']} ({acc['subtype'] or acc['type']}): saldo {brl(acc['balance'])}")
    return lines if len(lines) > 1 else [*lines, "- (nenhuma conta sincronizada)"]


def _snapshot_cashflow(con: duckdb.DuckDBPyConnection, today: date) -> list[str]:
    lines = ["## Fluxo de caixa mensal (últimos 6 meses; mês atual parcial)", "mês | receitas | gastos | saldo"]
    lines += [
        f"{r['month']} | {brl(r['income'])} | {brl(r['spending'])} | {brl(r['net'])}"
        for r in cashflow(con, months=6, today=today)
    ]
    return lines


def _snapshot_categories(con: duckdb.DuckDBPyConnection, today: date) -> list[str]:
    totals: dict[str, float] = defaultdict(float)
    start, end = month_start(today, 3), month_start(today, 0)
    for row in spending_by_category(con, months=4, today=today):
        if start.strftime("%Y-%m") <= row["month"] < end.strftime("%Y-%m"):
            totals[row["category"]] += float(row["total"])
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:8]
    lines = ["## Maiores categorias de gasto (média mensal dos últimos 3 meses completos)"]
    lines += [f"- {cat}: {brl(total / 3)}" for cat, total in ranked]
    return lines


def _snapshot_recurring(con: duckdb.DuckDBPyConnection, today: date) -> list[str]:
    detected = detect_subscriptions(con, today=today)
    lines = [f"## Cobranças recorrentes ativas detectadas (total ≈ {brl(detected['monthly_total'])}/mês)"]
    lines += [
        f"- {s['name']} [{s['category']}]: {brl(s['average_amount'])} {s['frequency']}, último {s['last_date']}"
        f" (confiança {s['confidence']})"
        for s in detected["subscriptions"][:15]
    ]
    return lines


def _snapshot_other(con: duckdb.DuckDBPyConnection) -> list[str]:
    lines: list[str] = []
    bills = bills_overview(con, limit=3)
    if bills:
        lines.append("## Faturas de cartão recentes")
        lines += [
            f"- {b['account']}: {brl(b['total_amount'])}, vencimento {b['due_date']}, "
            f"mínimo {brl(b['minimum_payment_amount'])}"
            for b in bills
        ]
    inv = investments_overview(con)
    if inv["positions"]:
        lines.append(f"## Investimentos: total {brl(inv['total'])}")
        lines += [f"- {t['type']}: {brl(t['total'])} ({t['positions']} posições)" for t in inv["by_type"]]
    loans = loans_overview(con)
    if loans["loans"]:
        lines.append(f"## Empréstimos/financiamentos: saldo devedor {brl(loans['total_outstanding'])}")
        lines += [
            f"- {loan['product_name'] or loan['type']}: saldo {brl(loan['outstanding_balance'])}, "
            f"parcela {brl(loan['installment_amount'])}, CET {loan['cet'] if loan['cet'] is not None else '—'}"
            for loan in loans["loans"][:5]
        ]
    return lines


def build_snapshot(con: duckdb.DuckDBPyConnection, *, today: date | None = None) -> str:
    """Build a compact pt-BR financial summary injected into the system prompt.

    Args:
        con: DuckDB connection.
        today: Reference date.

    Returns:
        str: Markdown-ish snapshot (~1–2k tokens).
    """
    ref = today or today_brt()
    span = con.execute("SELECT min(date_local), max(date_local), count(*) FROM transactions").fetchone() or (
        None,
        None,
        0,
    )
    header = (
        f"# Resumo financeiro (hoje: {ref.isoformat()}; transações de {span[0] or '—'} a {span[1] or '—'}, "
        f"{span[2]} lançamentos)"
    )
    parts = [header, *_snapshot_accounts(con), *_snapshot_cashflow(con, ref), *_snapshot_categories(con, ref)]
    parts += [*_snapshot_recurring(con, ref), *_snapshot_other(con)]
    attention = _records(
        con,
        "SELECT connector_name, status, consent_expires_at FROM items WHERE status IS NOT NULL AND status <> 'UPDATED'",
    )
    if attention:
        parts.append("## Conexões que precisam de atenção")
        parts += [f"- {a['connector_name']}: {a['status']}" for a in attention]
    return "\n".join(parts)
