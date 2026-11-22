"""Thin, tolerant HTTP client for the Pluggy Open Finance API.

Why not the ``pluggy-sdk`` package? There is no official Python SDK: the PyPI package is a
community OpenAPI build (no ``PluggyClient`` class), regenerated weekly, with strict enum
validation that breaks when Pluggy adds values, and no API-key refresh. We need ~10 endpoints,
so a small ``httpx`` client is simpler, typed loosely (plain dicts) and trivially testable with
``httpx.MockTransport``.

API facts encoded here (https://docs.pluggy.ai):

* ``POST /auth`` exchanges clientId/clientSecret for an API key valid for 2 h, sent as ``X-API-KEY``.
* Page-based lists return ``{"results", "page", "totalPages"}`` with ``pageSize`` <= 500.
* ``GET /v2/transactions`` is cursor-based: ``{"results", "next"}`` where ``next`` is a query
  string carrying ``after``. The legacy ``GET /transactions`` returns 410 for new applications.
* Rate limits answer 429 with ``RateLimit-Reset`` (seconds) and ``Retry-After``.

The client is thread-safe: the cached API key is guarded by a lock; everything else is stateless.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, Self
from urllib.parse import parse_qs

import httpx

from llmopenfinance.constants import (
    ENV_PLUGGY_CLIENT_ID,
    ENV_PLUGGY_CLIENT_SECRET,
    PLUGGY_API_KEY_TTL_SECONDS,
    PLUGGY_API_URL,
    PLUGGY_PAGE_SIZE,
)

logger = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRY_WAIT_SECONDS = 60.0

JsonDict = dict[str, Any]


class PluggyError(RuntimeError):
    """Raised when the Pluggy API answers with a non-success status.

    Attributes:
        status_code: HTTP status code (0 for transport errors).
        code: Pluggy error code from the body (e.g. ``ITEM_NOT_FOUND``), if any.
    """

    def __init__(self, message: str, status_code: int = 0, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def _error_from_response(response: httpx.Response) -> PluggyError:
    """Build a :class:`PluggyError` from an error response.

    Args:
        response: The failed HTTP response.

    Returns:
        PluggyError: Exception carrying status, Pluggy code and message.
    """
    code: str | None = None
    message = response.text[:300]
    try:
        body = response.json()
        if isinstance(body, dict):
            raw_code = body.get("codeDescription") or body.get("code")
            code = str(raw_code) if raw_code is not None else None
            message = str(body.get("message") or message)
    except ValueError:
        pass
    return PluggyError(
        f"Pluggy {response.request.method} {response.request.url.path} -> {response.status_code}: {message}",
        status_code=response.status_code,
        code=code,
    )


def _retry_wait_seconds(response: httpx.Response, attempt: int) -> float:
    """Compute how long to wait before retrying a throttled/failed request.

    Args:
        response: The 429/5xx response.
        attempt: Zero-based retry attempt number.

    Returns:
        float: Seconds to sleep (capped at 60 s).
    """
    for header in ("RateLimit-Reset", "Retry-After"):
        value = response.headers.get(header)
        if value:
            try:
                return min(max(float(value), 0.0), _MAX_RETRY_WAIT_SECONDS)
            except ValueError:
                continue
    return min(2.0**attempt, _MAX_RETRY_WAIT_SECONDS)


class PluggyClient:
    """Minimal Pluggy API client with API-key caching, retries and pagination.

    Args:
        client_id: Pluggy application clientId (env ``PLUGGY_USERNAME``).
        client_secret: Pluggy application clientSecret (env ``PLUGGY_PASSWORD``).
        base_url: API base URL.
        http: Optional pre-built ``httpx.Client`` (tests inject a ``MockTransport``).
        max_retries: Retries for 429/5xx responses.
        sleep: Sleep function (injectable for tests).
        clock: Monotonic clock (injectable for tests).
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        base_url: str = PLUGGY_API_URL,
        http: httpx.Client | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not client_id or not client_secret:
            raise ValueError(
                f"Pluggy credentials missing: set {ENV_PLUGGY_CLIENT_ID} (clientId) and "
                f"{ENV_PLUGGY_CLIENT_SECRET} (clientSecret) from dashboard.pluggy.ai"
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http or httpx.Client(base_url=base_url, timeout=60.0)
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._api_key: str | None = None
        self._api_key_expiry = 0.0

    @classmethod
    def from_env(cls, **kwargs: Any) -> PluggyClient:
        """Create a client from ``PLUGGY_USERNAME`` / ``PLUGGY_PASSWORD``.

        Args:
            **kwargs: Extra keyword arguments forwarded to the constructor.

        Returns:
            PluggyClient: Configured client.
        """
        return cls(
            os.environ.get(ENV_PLUGGY_CLIENT_ID, ""),
            os.environ.get(ENV_PLUGGY_CLIENT_SECRET, ""),
            **kwargs,
        )

    # ------------------------------------------------------------------ plumbing

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> Self:
        """Enter a context manager.

        Returns:
            PluggyClient: ``self``.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the client when leaving a ``with`` block.

        Args:
            *exc_info: Exception information (ignored).
        """
        self.close()

    def _api_key_value(self, *, force: bool = False) -> str:
        """Return a valid API key, calling ``POST /auth`` when missing or expired.

        Args:
            force: Discard the cached key and authenticate again.

        Returns:
            str: API key for the ``X-API-KEY`` header.
        """
        with self._lock:
            if force or self._api_key is None or self._clock() >= self._api_key_expiry:
                try:
                    response = self._http.post(
                        "/auth",
                        json={"clientId": self._client_id, "clientSecret": self._client_secret},
                    )
                except httpx.TransportError as exc:
                    raise PluggyError(f"Pluggy POST /auth transport error: {exc}") from exc
                if response.status_code != 200:
                    raise _error_from_response(response)
                self._api_key = str(response.json()["apiKey"])
                self._api_key_expiry = self._clock() + PLUGGY_API_KEY_TTL_SECONDS
                logger.debug("Pluggy API key refreshed")
            return self._api_key

    def authenticate(self) -> None:
        """Force a fresh ``POST /auth`` — useful to validate credentials early."""
        self._api_key_value(force=True)

    def request(self, method: str, path: str, *, params: JsonDict | None = None, json: JsonDict | None = None) -> Any:
        """Send an authenticated request, retrying 429/5xx and refreshing an expired key once.

        Args:
            method: HTTP method.
            path: Path relative to the base URL (e.g. ``/accounts``).
            params: Query parameters (``None`` values are dropped).
            json: JSON body.

        Returns:
            Any: Decoded JSON body.
        """
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        refreshed = False
        attempt = 0
        while True:
            headers = {"X-API-KEY": self._api_key_value()}
            try:
                response = self._http.request(method, path, params=clean_params, json=json, headers=headers)
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    raise PluggyError(f"Pluggy {method} {path} transport error: {exc}") from exc
                self._sleep(min(2.0**attempt, _MAX_RETRY_WAIT_SECONDS))
                attempt += 1
                continue
            if response.status_code == 401 and not refreshed:
                self._api_key_value(force=True)
                refreshed = True
                continue
            if response.status_code in _RETRY_STATUS and attempt < self._max_retries:
                wait = _retry_wait_seconds(response, attempt)
                logger.warning("Pluggy %s %s -> %d; retrying in %.1fs", method, path, response.status_code, wait)
                self._sleep(wait)
                attempt += 1
                continue
            if response.status_code >= 400:
                raise _error_from_response(response)
            return response.json() if response.content else {}

    def _paged(self, path: str, params: JsonDict) -> Iterator[JsonDict]:
        """Iterate every result of a page-based list endpoint.

        Args:
            path: Endpoint path.
            params: Base query parameters.

        Yields:
            JsonDict: One result object at a time.
        """
        page = 1
        while True:
            body = self.request("GET", path, params={**params, "page": page, "pageSize": PLUGGY_PAGE_SIZE})
            yield from body.get("results") or []
            total_pages = int(body.get("totalPages") or 1)
            if page >= total_pages:
                return
            page += 1

    # ------------------------------------------------------------------ endpoints

    def get_item(self, item_id: str) -> JsonDict:
        """Fetch one item (a bank connection).

        Args:
            item_id: Pluggy item id.

        Returns:
            JsonDict: Item object (``status``, ``executionStatus``, ``connector``, ...).
        """
        return self.request("GET", f"/items/{item_id}")

    def list_accounts(self, item_id: str) -> list[JsonDict]:
        """List the accounts (checking, savings, credit cards) of an item.

        Args:
            item_id: Pluggy item id.

        Returns:
            list[JsonDict]: Account objects.
        """
        return list(self._paged("/accounts", {"itemId": item_id}))

    def iter_transactions(
        self, account_id: str, *, date_from: str | None = None, date_to: str | None = None
    ) -> Iterator[JsonDict]:
        """Iterate all transactions of an account via the cursor-based ``/v2/transactions``.

        Args:
            account_id: Pluggy account id.
            date_from: Optional ``YYYY-MM-DD`` lower bound.
            date_to: Optional ``YYYY-MM-DD`` upper bound.

        Yields:
            JsonDict: Transaction objects, newest first.
        """
        params: JsonDict = {"accountId": account_id, "dateFrom": date_from, "dateTo": date_to}
        while True:
            body = self.request("GET", "/v2/transactions", params=params)
            yield from body.get("results") or []
            next_query = body.get("next")
            if not next_query:
                return
            after = parse_qs(str(next_query).lstrip("?")).get("after")
            if not after:
                return
            params = {**params, "after": after[0]}

    def list_bills(self, account_id: str) -> list[JsonDict]:
        """List credit-card bills (statements) of a credit account.

        Args:
            account_id: Pluggy credit-card account id.

        Returns:
            list[JsonDict]: Bill objects.
        """
        return list(self._paged("/bills", {"accountId": account_id}))

    def list_investments(self, item_id: str) -> list[JsonDict]:
        """List investment positions of an item.

        Args:
            item_id: Pluggy item id.

        Returns:
            list[JsonDict]: Investment objects.
        """
        return list(self._paged("/investments", {"itemId": item_id}))

    def list_loans(self, item_id: str) -> list[JsonDict]:
        """List loans/financing contracts of an item.

        Args:
            item_id: Pluggy item id.

        Returns:
            list[JsonDict]: Loan objects.
        """
        return list(self._paged("/loans", {"itemId": item_id}))

    def list_categories(self) -> list[JsonDict]:
        """List Pluggy's transaction category tree.

        Returns:
            list[JsonDict]: Category objects (``id``, ``description``, ``descriptionTranslated``, ``parentId``...).
        """
        return list(self._paged("/categories", {}))

    def create_connect_token(self, *, client_user_id: str | None = None, item_id: str | None = None) -> str:
        """Create a 30-minute token for the Pluggy Connect widget.

        Args:
            client_user_id: Optional identifier of the end user (sent inside ``options``).
            item_id: Optional item to update/re-authorize.

        Returns:
            str: Connect token (``accessToken``).
        """
        body: JsonDict = {}
        if item_id:
            body["itemId"] = item_id
        if client_user_id:
            body["options"] = {"clientUserId": client_user_id}
        return str(self.request("POST", "/connect_token", json=body)["accessToken"])
