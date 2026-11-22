"""Connect bank accounts from the terminal with the Pluggy Connect widget on a loopback page.

The CLI plays the role of the "backend" from Pluggy's samples: it creates a connect token,
serves a tiny HTML page on ``http://127.0.0.1:<random port>/<nonce>`` that loads the Connect
widget, opens it in the browser and receives the created item ids via same-origin ``fetch``.
No webhook or public URL is needed (OAuth logins redirect to Pluggy's own callback).

By default the widget is restricted to the **MeuPluggy** connector (free personal-use path);
``sandbox=True`` shows Pluggy's sandbox banks for testing (user ``user-ok`` / ``password-ok``).

If this flow does not work for your account, use the documented manual path instead: connect in
dashboard.pluggy.ai → Demo → MeuPluggy, copy the item id and run ``finassist items add <id>``.
"""

from __future__ import annotations

import html
import json
import logging
import queue
import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from llmopenfinance.constants import MEU_PLUGGY_CONNECTOR_ID, PLUGGY_CONNECT_JS
from llmopenfinance.i18n import get_language, t
from llmopenfinance.pluggy import PluggyClient

logger = logging.getLogger(__name__)

FINAL_ITEM_STATUSES = frozenset({"UPDATED", "LOGIN_ERROR", "OUTDATED", "WAITING_USER_INPUT", "WAITING_USER_ACTION"})

PAGE_TEMPLATE = """<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;max-width:40rem;margin:3rem auto;padding:0 1rem;line-height:1.5}}
code{{background:#eee;padding:.1rem .3rem;border-radius:.2rem}}</style>
<script src="{script_url}"></script></head>
<body><h1>{heading}</h1>
<p id="status">{opening}</p>
<script>
const base = {base_path};
const seen = new Set();
const status = (t) => {{ document.getElementById('status').textContent = t; }};
const report = (item, event) => {{
  if (!item || !item.id || seen.has(item.id)) return;
  seen.add(item.id);
  fetch(base + '/item', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{itemId: item.id, event: event}})}});
  status({connected_prefix} + item.id + {connected_suffix});
}};
const done = () => fetch(base + '/done', {{method: 'POST'}}).finally(() => status({done_text}));
const connect = new PluggyConnect({options});
connect.init();
</script></body></html>
"""


def _js_string(text: str) -> str:
    """Render ``text`` as a single-quoted JavaScript string literal for the inline page script.

    ``json.dumps`` does the escaping (``ensure_ascii=False`` keeps accents readable); the double quotes
    are then swapped for single ones and ``</`` is split so no text can close the ``<script>`` element.

    Args:
        text: Catalog text to embed.

    Returns:
        str: The quoted literal, including the surrounding single quotes.
    """
    body = json.dumps(text, ensure_ascii=False)[1:-1]
    body = body.replace('\\"', '"').replace("'", "\\'").replace("</", "<\\/")
    return f"'{body}'"


def build_page(
    connect_token: str, base_path: str, *, sandbox: bool = False, script_url: str = PLUGGY_CONNECT_JS
) -> str:
    """Render the loopback HTML page in the process language (text, ``<html lang>`` and widget language).

    Args:
        connect_token: Token from ``POST /connect_token``.
        base_path: Secret path prefix (``/<nonce>``) for callbacks.
        sandbox: Show sandbox connectors instead of restricting to MeuPluggy.
        script_url: Pluggy Connect script URL.

    Returns:
        str: HTML document.
    """
    language = get_language()
    options: dict[str, Any] = {"connectToken": connect_token, "language": language.split("-")[0]}
    if sandbox:
        options["includeSandbox"] = True
    else:
        options["connectorIds"] = [MEU_PLUGGY_CONNECTOR_ID]
    error_prefix = _js_string(t("connect.page.error_prefix"))
    error_unknown = _js_string(t("connect.page.error_unknown"))
    # Callbacks are JS functions, so splice them into the JSON object literal.
    options_js = json.dumps(options)[:-1] + (
        ", onEvent: (e) => { if (e && e.item) report(e.item, e.event); }"
        ", onSuccess: (d) => report(d && d.item, 'SUCCESS')"
        ", onError: (e) => { report(e && e.data && e.data.item, 'ERROR');"
        f" status({error_prefix} + (e && e.message ? e.message : {error_unknown})); }}"
        ", onClose: done }"
    )
    return PAGE_TEMPLATE.format(
        script_url=script_url,
        base_path=json.dumps(base_path),
        options=options_js,
        lang=html.escape(language),
        title=html.escape(t("connect.page.title")),
        heading=html.escape(t("connect.page.heading")),
        opening=html.escape(t("connect.page.opening")),
        connected_prefix=_js_string(t("connect.page.connected_prefix")),
        connected_suffix=_js_string(t("connect.page.connected_suffix")),
        done_text=_js_string(t("connect.page.done")),
    )


@dataclass
class _Session:
    """Shared state between the HTTP handler and the waiting CLI thread."""

    nonce: str
    page: str
    items: queue.Queue[str] = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)


def _make_handler(session: _Session) -> type[BaseHTTPRequestHandler]:
    """Create a request handler class bound to ``session``.

    Args:
        session: Connect session state.

    Returns:
        type[BaseHTTPRequestHandler]: Handler class.
    """
    base = f"/{session.nonce}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # ruff: ignore[builtin-argument-shadowing] — stdlib signature
            logger.debug("connect server: %s", format % args)

        def _reply(self, status: int, body: bytes = b"", content_type: str = "text/plain; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == base:
                self._reply(200, session.page.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._reply(404)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            if self.path == f"{base}/item":
                try:
                    item_id = str(json.loads(body or b"{}").get("itemId") or "").strip()
                except (ValueError, AttributeError):
                    item_id = ""
                if item_id:
                    session.items.put(item_id)
                self._reply(204)
            elif self.path == f"{base}/done":
                session.done.set()
                self._reply(204)
            else:
                self._reply(404)

    return Handler


def run_connect_flow(
    client: PluggyClient,
    on_item: Callable[[str], None],
    *,
    sandbox: bool = False,
    timeout: float = 30 * 60,
    open_browser: Callable[[str], Any] = webbrowser.open,
    client_user_id: str = "llmopenfinance-cli",
    on_url: Callable[[str], None] | None = None,
) -> list[str]:
    """Serve the Connect widget locally and collect the item ids the user creates.

    ``on_item`` is invoked as soon as each id arrives (persist it immediately — Pluggy offers no
    endpoint to list items later).

    Args:
        client: Pluggy client (used to create the connect token).
        on_item: Callback receiving each new item id.
        sandbox: Show sandbox connectors.
        timeout: Seconds to wait for the user to finish.
        open_browser: Function opening a URL (injectable for tests).
        client_user_id: Identifier attached to created items.
        on_url: Optional callback receiving the local page URL (shown in case the browser does not open).

    Returns:
        list[str]: Item ids received, in arrival order.
    """
    token = client.create_connect_token(client_user_id=client_user_id)
    session = _Session(nonce=secrets.token_urlsafe(16), page="")
    session.page = build_page(token, f"/{session.nonce}", sandbox=sandbox)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(session))
    thread = threading.Thread(target=server.serve_forever, name="pluggy-connect", daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/{session.nonce}"
    received: list[str] = []
    try:
        if on_url is not None:
            on_url(url)
        open_browser(url)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                item_id = session.items.get(timeout=0.5)
            except queue.Empty:
                if session.done.is_set():
                    break
                continue
            if item_id not in received:
                received.append(item_id)
                on_item(item_id)
        while not session.items.empty():
            item_id = session.items.get_nowait()
            if item_id not in received:
                received.append(item_id)
                on_item(item_id)
    finally:
        server.shutdown()
        server.server_close()
    return received


def wait_for_item(
    client: PluggyClient,
    item_id: str,
    *,
    timeout: float = 10 * 60,
    interval: float = 3.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll ``GET /items/{id}`` until the connection settles (UPDATED, LOGIN_ERROR...).

    Args:
        client: Pluggy client.
        item_id: Item id.
        timeout: Maximum seconds to wait.
        interval: Seconds between polls.
        sleep: Sleep function (injectable).
        clock: Monotonic clock (injectable).

    Returns:
        dict[str, Any]: Last item JSON seen (check its ``status``).
    """
    deadline = clock() + timeout
    item = client.get_item(item_id)
    while str(item.get("status")) not in FINAL_ITEM_STATUSES and clock() < deadline:
        sleep(interval)
        item = client.get_item(item_id)
    return item
