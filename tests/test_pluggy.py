"""Tests for the Pluggy HTTP client using httpx.MockTransport (no network)."""

import json
from urllib.parse import parse_qs

import httpx
import pytest

from llmopenfinance.pluggy import PluggyClient, PluggyError, _error_from_response, _retry_wait_seconds


class Api:
    """Scriptable fake Pluggy API."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.auth_calls = 0
        self.routes = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/auth":
            self.auth_calls += 1
            body = json.loads(request.content)
            if body["clientSecret"] == "bad":
                return httpx.Response(401, json={"code": 401, "codeDescription": "CLIENT_KEYS_UNAUTHORIZED"})
            return httpx.Response(200, json={"apiKey": f"key-{self.auth_calls}"})
        handler = self.routes.get((request.method, request.url.path))
        if handler is None:
            return httpx.Response(404, json={"code": 404, "message": "not found", "codeDescription": "NOT_FOUND"})
        return handler(request)


def make_client(api: Api, **kwargs) -> PluggyClient:
    http = httpx.Client(transport=httpx.MockTransport(api), base_url="https://api.pluggy.ai")
    return PluggyClient("id", kwargs.pop("secret", "secret"), http=http, sleep=lambda _s: None, **kwargs)


def test_requires_credentials():
    with pytest.raises(ValueError, match="PLUGGY_USERNAME"):
        PluggyClient("", "")


def test_from_env(monkeypatch):
    monkeypatch.setenv("PLUGGY_USERNAME", "cid")
    monkeypatch.setenv("PLUGGY_PASSWORD", "csecret")
    client = PluggyClient.from_env(http=httpx.Client(transport=httpx.MockTransport(Api())))
    assert client._client_id == "cid"
    client.close()


def test_api_key_is_cached_and_sent():
    api = Api()
    api.routes["GET", "/items/i1"] = lambda r: httpx.Response(200, json={"id": "i1", "key": r.headers["X-API-KEY"]})
    with make_client(api) as client:
        assert client.get_item("i1")["key"] == "key-1"
        client.get_item("i1")
    assert api.auth_calls == 1


def test_api_key_refreshed_after_expiry():
    api = Api()
    api.routes["GET", "/items/i1"] = lambda r: httpx.Response(200, json={"key": r.headers["X-API-KEY"]})
    now = [0.0]
    client = make_client(api, clock=lambda: now[0])
    assert client.get_item("i1")["key"] == "key-1"
    now[0] = 10_000.0
    assert client.get_item("i1")["key"] == "key-2"


def test_401_triggers_single_reauth():
    api = Api()
    seen = []

    def item(request):
        seen.append(request.headers["X-API-KEY"])
        if request.headers["X-API-KEY"] == "key-1":
            return httpx.Response(401, json={"message": "expired"})
        return httpx.Response(200, json={"ok": True})

    api.routes["GET", "/items/i1"] = item
    assert make_client(api).get_item("i1") == {"ok": True}
    assert seen == ["key-1", "key-2"]


def test_auth_failure_raises():
    with pytest.raises(PluggyError) as info:
        make_client(Api(), secret="bad").authenticate()
    assert info.value.status_code == 401
    assert info.value.code == "CLIENT_KEYS_UNAUTHORIZED"


def test_429_is_retried_with_rate_limit_reset():
    api = Api()
    attempts = []

    def accounts(_request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"RateLimit-Reset": "2", "Retry-After": "60"}, json={})
        return httpx.Response(200, json={"results": [{"id": "a"}], "totalPages": 1})

    api.routes["GET", "/accounts"] = accounts
    slept = []
    http = httpx.Client(transport=httpx.MockTransport(api), base_url="https://api.pluggy.ai")
    client = PluggyClient("id", "secret", http=http, sleep=slept.append)
    assert client.list_accounts("item") == [{"id": "a"}]
    assert slept == [2.0, 2.0]


def test_retries_exhausted_raises():
    api = Api()
    api.routes["GET", "/items/x"] = lambda _r: httpx.Response(503, text="down")
    with pytest.raises(PluggyError) as info:
        make_client(api, max_retries=1).get_item("x")
    assert info.value.status_code == 503


def test_transport_error_retried_then_raised():
    calls = []

    def handler(request):
        if request.url.path == "/auth":
            return httpx.Response(200, json={"apiKey": "k"})
        calls.append(1)
        raise httpx.ConnectError("boom", request=request)

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.pluggy.ai")
    client = PluggyClient("id", "secret", http=http, sleep=lambda _s: None, max_retries=2)
    with pytest.raises(PluggyError, match="transport error"):
        client.get_item("x")
    assert len(calls) == 3


def test_404_error_carries_code():
    with pytest.raises(PluggyError) as info:
        make_client(Api()).get_item("missing")
    assert info.value.status_code == 404
    assert info.value.code == "NOT_FOUND"


def test_paged_endpoints_follow_total_pages():
    api = Api()

    def bills(request):
        page = int(request.url.params["page"])
        assert request.url.params["pageSize"] == "500"
        return httpx.Response(200, json={"results": [{"id": f"b{page}"}], "totalPages": 3, "page": page})

    api.routes["GET", "/bills"] = bills
    api.routes["GET", "/investments"] = lambda _r: httpx.Response(200, json={"results": [{"id": "i"}]})
    api.routes["GET", "/loans"] = lambda _r: httpx.Response(200, json={"results": []})
    api.routes["GET", "/categories"] = lambda _r: httpx.Response(200, json={"results": [{"id": "c"}], "totalPages": 1})
    client = make_client(api)
    assert [b["id"] for b in client.list_bills("acc")] == ["b1", "b2", "b3"]
    assert client.list_investments("item") == [{"id": "i"}]
    assert client.list_loans("item") == []
    assert client.list_categories() == [{"id": "c"}]


def test_transactions_cursor_pagination():
    api = Api()
    seen_params = []

    def transactions(request):
        params = dict(request.url.params)
        seen_params.append(params)
        if "after" not in params:
            return httpx.Response(200, json={"results": [{"id": "t1"}], "next": "?accountId=acc&after=abc%3D%3D"})
        if params["after"] == "abc==":
            return httpx.Response(200, json={"results": [{"id": "t2"}], "next": "?accountId=acc"})
        return httpx.Response(500)

    api.routes["GET", "/v2/transactions"] = transactions
    ids = [t["id"] for t in make_client(api).iter_transactions("acc", date_from="2026-01-01")]
    assert ids == ["t1", "t2"]
    assert seen_params[0] == {"accountId": "acc", "dateFrom": "2026-01-01"}
    assert seen_params[1]["after"] == "abc=="


def test_transactions_stop_when_next_is_null():
    api = Api()
    api.routes["GET", "/v2/transactions"] = lambda _r: httpx.Response(
        200, json={"results": [{"id": "t"}], "next": None}
    )
    assert len(list(make_client(api).iter_transactions("acc"))) == 1


def test_connect_token_body():
    api = Api()
    bodies = []

    def token(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"accessToken": "tok"})

    api.routes["POST", "/connect_token"] = token
    client = make_client(api)
    assert client.create_connect_token(client_user_id="me", item_id="it") == "tok"
    assert client.create_connect_token() == "tok"
    assert bodies == [{"itemId": "it", "options": {"clientUserId": "me"}}, {}]


def test_empty_body_returns_empty_dict():
    api = Api()
    api.routes["GET", "/items/e"] = lambda _r: httpx.Response(204)
    assert make_client(api).get_item("e") == {}


def test_error_from_non_json_response():
    request = httpx.Request("GET", "https://api.pluggy.ai/x")
    error = _error_from_response(httpx.Response(500, text="<html>oops</html>", request=request))
    assert error.code is None
    assert "oops" in str(error)


def test_retry_wait_seconds():
    request = httpx.Request("GET", "https://x")
    assert _retry_wait_seconds(httpx.Response(429, headers={"Retry-After": "7"}, request=request), 0) == pytest.approx(
        7.0
    )
    assert _retry_wait_seconds(
        httpx.Response(429, headers={"RateLimit-Reset": "abc"}, request=request), 2
    ) == pytest.approx(4.0)
    assert _retry_wait_seconds(
        httpx.Response(429, headers={"RateLimit-Reset": "999"}, request=request), 0
    ) == pytest.approx(60.0)


def test_none_params_are_dropped():
    api = Api()
    api.routes["GET", "/v2/transactions"] = lambda _r: httpx.Response(200, json={"results": [], "next": None})
    list(make_client(api).iter_transactions("acc"))
    query = parse_qs(api.requests[-1].url.query.decode())
    assert set(query) == {"accountId"}


def test_auth_transport_error_is_wrapped():
    """Regression: an offline /auth raised a raw httpx error and crashed the CLI."""

    def offline(request):
        raise httpx.ConnectError("no network", request=request)

    http = httpx.Client(transport=httpx.MockTransport(offline), base_url="https://api.pluggy.ai")
    with pytest.raises(PluggyError, match="/auth transport error"):
        PluggyClient("id", "secret", http=http).authenticate()
