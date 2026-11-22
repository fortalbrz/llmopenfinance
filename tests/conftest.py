"""Shared fixtures: realistic synthetic Pluggy payloads, a fake Pluggy client and populated stores."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from llmopenfinance.constants import ENV_LANG
from llmopenfinance.i18n import reset_language
from llmopenfinance.store import FinanceStore, transaction_row

TODAY = date(2026, 11, 22)
ITEM_ID = "item-1"
BANK_ID = "acc-bank"
CARD_ID = "acc-card"

ITEM = {
    "id": ITEM_ID,
    "connector": {"id": 200, "name": "MeuPluggy"},
    "status": "UPDATED",
    "executionStatus": "SUCCESS",
    "lastUpdatedAt": "2026-11-21T10:00:00.000Z",
    "consentExpiresAt": "2027-06-01T00:00:00.000Z",
}

ACCOUNTS = [
    {
        "id": BANK_ID,
        "itemId": ITEM_ID,
        "type": "BANK",
        "subtype": "CHECKING_ACCOUNT",
        "name": "Conta Corrente",
        "marketingName": "Nubank Conta",
        "number": "1234-5",
        "balance": 3500.75,
        "currencyCode": "BRL",
    },
    {
        "id": CARD_ID,
        "itemId": ITEM_ID,
        "type": "CREDIT",
        "subtype": "CREDIT_CARD",
        "name": "Cartão",
        "marketingName": "Nubank Ultravioleta",
        "number": "4321",
        "balance": 1890.4,
        "currencyCode": "BRL",
        "creditData": {
            "creditLimit": 10000,
            "availableCreditLimit": 8109.6,
            "balanceCloseDate": "2026-11-28",
            "balanceDueDate": "2026-12-05T00:00:00.000Z",
            "minimumPayment": 283.56,
        },
    },
]


def _tx(tx_id: str, day: str, description: str, amount: float, tx_type: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": tx_id,
        "date": f"{day}T00:00:00.000Z",
        "description": description,
        "amount": amount,
        "type": tx_type,
        "status": "POSTED",
        "currencyCode": "BRL",
        **extra,
    }


def bank_transactions() -> list[dict[str, Any]]:
    """Checking-account transactions (outflows are negative)."""
    txs: list[dict[str, Any]] = []
    for month in range(1, 12):
        m = f"2026-{month:02d}"
        txs.append(_tx(f"sal-{month}", f"{m}-05", "SALARIO EMPRESA X", 8000.0, "CREDIT", category="Salary"))
        txs.append(
            _tx(
                f"fat-{month}",
                f"{m}-10",
                "PAGAMENTO FATURA NUBANK",
                -1500.0,
                "DEBIT",
                category="Credit card payment",
            )
        )
        txs.append(_tx(f"luz-{month}", f"{m}-15", "ENEL DISTRIBUICAO", -180.0 - month, "DEBIT", category="Utilities"))
    txs.append(_tx("cdb-1", "2026-10-20", "APLICACAO CDB", -1000.0, "DEBIT", category="Fixed income"))
    txs.append(
        _tx("pix-own", "2026-10-21", "PIX MESMA TITULARIDADE", -300.0, "DEBIT", category="Same person transfer - PIX")
    )
    txs.append(
        _tx(
            "pix-joao",
            "2026-11-03",
            "PIX ENVIADO JOAO",
            -120.0,
            "DEBIT",
            category="Transfer - PIX",
            paymentData={"paymentMethod": "PIX"},
        )
    )
    # Real timestamp (not midnight UTC) -> converted to UTC-3: 2026-11-01 02:30Z is 2026-10-31 in Brazil.
    txs.append({**_tx("late", "2026-11-01", "PADARIA PAO QUENTE", -25.0, "DEBIT"), "date": "2026-11-01T02:30:00.000Z"})
    return txs


def card_transactions() -> list[dict[str, Any]]:
    """Credit-card transactions (charges positive, payments/refunds negative; type normalized)."""
    txs: list[dict[str, Any]] = []
    for month in range(1, 12):
        m = f"2026-{month:02d}"
        txs.append(_tx(f"nf-{month}", f"{m}-08", "NETFLIX.COM", 55.9, "DEBIT", category="Video streaming"))
        if month >= 6:
            txs.append(_tx(f"sp-{month}", f"{m}-12", "SPOTIFY", 21.9, "DEBIT"))  # no Pluggy category
        txs.append(
            _tx(
                f"amz-{month}", f"{m}-18", "AMAZON MARKETPLACE", 100.0 + 10 * month, "DEBIT", category="Online shopping"
            )
        )
        txs.append(_tx(f"if-{month}", f"{m}-20", "IFOOD *RESTAURANTE", 60.0 + month, "DEBIT"))
        txs.append(
            _tx(f"pay-{month}", f"{m}-11", "Pagamento recebido", -1500.0, "CREDIT", operationType="PAGAMENTO_FATURA")
        )
    txs.append(_tx("ml-1", "2026-11-02", "MERCADOLIVRE*VENDEDOR", 250.0, "DEBIT"))
    txs.append(_tx("est-1", "2026-11-05", "ESTORNO AMAZON MARKETPLACE", -50.0, "CREDIT", category="Online shopping"))
    txs.append(
        _tx(
            "parc-2",
            "2026-11-09",
            "LOJA DE MOVEIS PARC 02/10",
            300.0,
            "DEBIT",
            category="Shopping",
            creditCardMetadata={"installmentNumber": 2, "totalInstallments": 10, "billId": "bill-11"},
        )
    )
    txs.append(
        _tx(
            "parc-1",
            "2026-10-09",
            "LOJA DE MOVEIS PARC 01/10",
            300.0,
            "DEBIT",
            category="Shopping",
            creditCardMetadata={"installmentNumber": 1, "totalInstallments": 10},
        )
    )
    return txs


BILLS = [
    {
        "id": "bill-10",
        "dueDate": "2026-11-05T00:00:00.000Z",
        "billClosingDate": "2026-10-28",
        "totalAmount": 1450.0,
        "minimumPaymentAmount": 217.5,
    },
    {"id": "bill-11", "dueDate": "2026-12-05", "billClosingDate": "2026-11-28", "totalAmount": 1890.4},
]

INVESTMENTS = [
    {
        "id": "inv-1",
        "name": "CDB Banco X 110% CDI",
        "type": "FIXED_INCOME",
        "subtype": "CDB",
        "balance": 15000.0,
        "amountProfit": 1200.0,
        "annualRate": 12.1,
        "dueDate": "2028-01-10",
        "issuer": "Banco X",
        "status": "ACTIVE",
    },
    {"id": "inv-2", "name": "Fundo Y", "type": "MUTUAL_FUND", "balance": 5000.0, "lastTwelveMonthsRate": 9.5},
]

LOANS = [
    {
        "id": "loan-1",
        "productName": "Crédito pessoal",
        "type": "PERSONAL_LOAN",
        "contractAmount": 20000.0,
        "payments": {"contractOutstandingBalance": 12000.0},
        "installments": {"totalNumberOfInstallments": 24, "paidInstallments": 10, "dueInstallments": 14},
        "installmentAmount": 980.0,
        "dueDate": "2027-12-10",
        "CET": 35.2,
    }
]

CATEGORIES = [
    {
        "id": "08010000",
        "description": "Online shopping",
        "descriptionTranslated": "Compras online",
        "parentId": "08000000",
    },
    {"id": "08000000", "description": "Shopping", "descriptionTranslated": "Compras"},
]


class FakePluggy:
    """In-memory stand-in for :class:`llmopenfinance.pluggy.PluggyClient`."""

    def __init__(self) -> None:
        self.items = {ITEM_ID: dict(ITEM)}
        self.accounts = {ITEM_ID: [dict(a) for a in ACCOUNTS]}
        self.transactions = {BANK_ID: bank_transactions(), CARD_ID: card_transactions()}
        self.bills = {CARD_ID: list(BILLS)}
        self.investments = {ITEM_ID: list(INVESTMENTS)}
        self.loans = {ITEM_ID: list(LOANS)}
        self.categories = list(CATEGORIES)
        self.calls: list[str] = []
        self.fail: dict[str, Exception] = {}
        self.closed = False

    def _maybe_fail(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            raise self.fail[name]

    def authenticate(self) -> None:
        self._maybe_fail("authenticate")

    def get_item(self, item_id: str) -> dict[str, Any]:
        self._maybe_fail("get_item")
        if item_id not in self.items:
            from llmopenfinance.pluggy import PluggyError

            raise PluggyError("not found", 404, "ITEM_NOT_FOUND")
        return self.items[item_id]

    def list_accounts(self, item_id: str) -> list[dict[str, Any]]:
        self._maybe_fail("list_accounts")
        return self.accounts.get(item_id, [])

    def iter_transactions(self, account_id: str, **_: Any) -> Iterator[dict[str, Any]]:
        self._maybe_fail("iter_transactions")
        yield from self.transactions.get(account_id, [])

    def list_bills(self, account_id: str) -> list[dict[str, Any]]:
        self._maybe_fail("list_bills")
        return self.bills.get(account_id, [])

    def list_investments(self, item_id: str) -> list[dict[str, Any]]:
        self._maybe_fail("list_investments")
        return self.investments.get(item_id, [])

    def list_loans(self, item_id: str) -> list[dict[str, Any]]:
        self._maybe_fail("list_loans")
        return self.loans.get(item_id, [])

    def list_categories(self) -> list[dict[str, Any]]:
        self._maybe_fail("list_categories")
        return self.categories

    def create_connect_token(self, **_: Any) -> str:
        self._maybe_fail("create_connect_token")
        return "connect-token-123"

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_pluggy() -> FakePluggy:
    return FakePluggy()


def populate(store: FinanceStore) -> None:
    """Fill a read-write store with the synthetic dataset (no network)."""
    store.add_item(ITEM_ID)
    store.update_item(ITEM, synced=True)
    store.upsert_categories(CATEGORIES)
    store.replace_accounts(ITEM_ID, ACCOUNTS)
    translations = store.category_translations()
    store.replace_transactions(
        BANK_ID, [transaction_row(t, BANK_ID, ITEM_ID, "BANK", translations) for t in bank_transactions()]
    )
    store.replace_transactions(
        CARD_ID, [transaction_row(t, CARD_ID, ITEM_ID, "CREDIT", translations) for t in card_transactions()]
    )
    store.replace_bills(CARD_ID, BILLS)
    store.replace_investments(ITEM_ID, INVESTMENTS)
    store.replace_loans(ITEM_ID, LOANS)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "finance.duckdb"
    with FinanceStore(path) as store:
        populate(store)
    return path


@pytest.fixture
def ro_store(db_path: Path) -> Iterator[FinanceStore]:
    store = FinanceStore(db_path, read_only=True)
    yield store
    store.close()


@pytest.fixture
def con(ro_store: FinanceStore) -> Any:
    return ro_store.con


@pytest.fixture(autouse=True)
def _pin_language(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Freeze pt-BR for every test, ignoring the caller's environment, and unfreeze afterwards."""
    monkeypatch.delenv(ENV_LANG, raising=False)
    reset_language("pt-BR")
    yield
    reset_language()


@pytest.fixture
def use_language() -> Callable[[str], None]:
    """Return ``use(code)``, which switches the frozen language for the current test."""

    def use(code: str) -> None:
        reset_language(code)

    return use
