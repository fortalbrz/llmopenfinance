"""Tests for the loopback Pluggy Connect flow (real local HTTP server, fake browser)."""

import json
import threading

import httpx
import pytest

from llmopenfinance import messages_pt
from llmopenfinance.connect import build_page, run_connect_flow, wait_for_item


def test_build_page_meu_pluggy():
    html = build_page("tok-123", "/nonce")
    assert '"connectToken": "tok-123"' in html
    assert '"connectorIds": [200]' in html
    assert "includeSandbox" not in html
    assert 'const base = "/nonce";' in html
    assert "pluggy-connect/v2.11.0/pluggy-connect.js" in html
    assert "onSuccess" in html
    assert "onClose: done" in html


def test_build_page_sandbox():
    html = build_page("t", "/n", sandbox=True)
    assert '"includeSandbox": true' in html
    assert "connectorIds" not in html


def _browser(actions):
    """Fake browser: performs HTTP calls against the served page in a background thread."""

    def open_browser(url):
        def run():
            with httpx.Client(timeout=5) as http:
                for action in actions:
                    action(http, url)

        threading.Thread(target=run, daemon=True).start()
        return True

    return open_browser


def test_run_connect_flow_collects_items(fake_pluggy):
    seen = {}

    def get_page(http, url):
        seen["page"] = http.get(url)
        seen["404"] = http.get(url.rsplit("/", 1)[0] + "/wrong")
        seen["bad_post"] = http.post(url.rsplit("/", 1)[0] + "/wrong/item")

    def post_items(http, url):
        http.post(url + "/item", json={"itemId": "item-a", "event": "ITEM_RESPONSE"})
        http.post(url + "/item", json={"itemId": "item-a", "event": "SUCCESS"})  # duplicate
        http.post(url + "/item", content=b"not json")
        http.post(url + "/item", json={"itemId": "item-b"})

    def close(http, url):
        http.post(url + "/done")

    received_cb = []
    items = run_connect_flow(
        fake_pluggy, received_cb.append, open_browser=_browser([get_page, post_items, close]), timeout=10
    )
    assert items == ["item-a", "item-b"]
    assert received_cb == ["item-a", "item-b"]
    assert seen["page"].status_code == 200
    assert "connect-token-123" in seen["page"].text
    assert seen["404"].status_code == 404
    assert seen["bad_post"].status_code == 404


def test_run_connect_flow_timeout_without_items(fake_pluggy):
    assert run_connect_flow(fake_pluggy, lambda _i: None, open_browser=lambda _u: None, timeout=0.6) == []


def test_run_connect_flow_server_is_closed(fake_pluggy):
    urls = []
    run_connect_flow(fake_pluggy, lambda _i: None, open_browser=urls.append, timeout=0.1)
    with pytest.raises(httpx.HTTPError):
        httpx.get(urls[0], timeout=1)


def test_wait_for_item_polls_until_final(fake_pluggy):
    statuses = iter(["UPDATING", "UPDATING", "UPDATED"])

    def get_item(item_id):
        return {"id": item_id, "status": next(statuses)}

    fake_pluggy.get_item = get_item
    slept = []
    item = wait_for_item(fake_pluggy, "x", sleep=slept.append, interval=1)
    assert item["status"] == "UPDATED"
    assert slept == [1, 1]


def test_wait_for_item_times_out(fake_pluggy):
    fake_pluggy.get_item = lambda item_id: {"id": item_id, "status": "UPDATING"}
    clock = iter([0.0, 5.0, 11.0, 20.0])
    item = wait_for_item(fake_pluggy, "x", timeout=10, sleep=lambda _s: None, clock=lambda: next(clock))
    assert item["status"] == "UPDATING"


def test_page_options_are_valid_json_prefix():
    html = build_page('tok"quote', "/n")
    start = html.index("new PluggyConnect(") + len("new PluggyConnect(")
    json_part = html[start : html.index(", onEvent")] + "}"
    assert json.loads(json_part)["connectToken"] == 'tok"quote'


def test_run_connect_flow_reports_url(fake_pluggy):
    urls = []
    run_connect_flow(fake_pluggy, lambda _i: None, open_browser=lambda _u: False, timeout=0.1, on_url=urls.append)
    assert urls
    assert urls[0].startswith("http://127.0.0.1:")


def test_build_page_pt_keeps_language_markers():
    html = build_page("t", "/n")
    assert '<html lang="pt-BR">' in html
    assert '"language": "pt"' in html
    assert "<title>Conectar conta — llmopenfinance</title>" in html
    assert "<h1>Conectar conta</h1>" in html
    assert "Abrindo o Pluggy Connect…" in html
    assert "status('Pronto! Volte ao terminal.')" in html


def test_build_page_en(use_language):
    use_language("en")
    html = build_page("t", "/n")
    assert '<html lang="en">' in html
    assert '"language": "en"' in html
    assert "<title>Connect account — llmopenfinance</title>" in html
    assert "<h1>Connect account</h1>" in html
    assert "Opening Pluggy Connect…" in html
    assert "status('Account connected: ' + item.id + '. You can connect another one or close this tab.')" in html
    assert "status('Done! Go back to the terminal.')" in html
    assert "status('Error: ' + (e && e.message ? e.message : 'unknown'))" in html
    for portuguese in ("Conectar", "Abrindo", "Você", "Pronto", "desconhecido"):
        assert portuguese not in html


def test_build_page_escapes_catalog_text(monkeypatch):
    """A future catalog edit with quotes or markup must not break the HTML or the inline script."""
    nasty = 'O\'Brien "quoted" </script><b>& back\\slash'
    for key in (
        "connect.page.title",
        "connect.page.heading",
        "connect.page.opening",
        "connect.page.connected_prefix",
        "connect.page.connected_suffix",
        "connect.page.done",
        "connect.page.error_prefix",
        "connect.page.error_unknown",
    ):
        monkeypatch.setitem(messages_pt.MESSAGES, key, nasty)
    html = build_page("t", "/n")
    head, _, script = html.partition("<script>\n")
    escaped = "O&#x27;Brien &quot;quoted&quot; &lt;/script&gt;&lt;b&gt;&amp; back\\slash"
    assert f"<title>{escaped}</title>" in head
    assert f"<h1>{escaped}</h1>" in html
    # Exactly one closing script tag in the body: the nasty text never ends the inline script early.
    assert script.count("</script>") == 1
    assert "'O\\'Brien \"quoted\" <\\/script><b>& back\\\\slash'" in script
