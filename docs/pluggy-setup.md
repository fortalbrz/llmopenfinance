# Connecting your bank accounts (Pluggy)

`llmopenfinance` reads your data through the Pluggy API. For personal use the recommended path is **MeuPluggy**:

- it is free, needs no company due diligence and keeps working after the 15-day trial;
- it is for personal, non-commercial use only, with up to 5 connections.

## 1. Create the Pluggy application (once)

1. Sign up at <https://dashboard.pluggy.ai>.
2. Create an application. Copy its **Client ID** and **Client Secret**. The secret is shown only once.
3. Export them. They are the *application's* credentials, not your MeuPluggy login:

   ```powershell
   [Environment]::SetEnvironmentVariable("PLUGGY_USERNAME", "<clientId>", "User")
   [Environment]::SetEnvironmentVariable("PLUGGY_PASSWORD", "<clientSecret>", "User")
   ```

## 2. Connect your banks in MeuPluggy

1. Sign up at <https://meu.pluggy.ai> **with the same e-mail as the dashboard**.
2. Connect each bank. Each one needs your CPF, and you approve the access in the bank's app.

## 3. Give the CLI the item ids

Pluggy has **no API to list your items**, so the CLI must be told each item id once. Choose one way:

**A. Dashboard (documented path, always works)**

1. In the dashboard, go to **Applications → your app → Demo → Connect Account**. Choose **MeuPluggy**, not your
   bank and not Sandbox, and sign in.
2. In **Connected Items**, copy each item id.
3. Register the ids:

   ```powershell
   uv run llmopenfinance items add <item-id> [<item-id> ...]
   ```

   Alternatively, set `PLUGGY_ITEM_IDS=<id1>,<id2>`. The ids are registered automatically on the next run.

**B. `llmopenfinance connect` (experimental)**

```powershell
uv run llmopenfinance connect            # MeuPluggy only
uv run llmopenfinance connect --sandbox  # Pluggy test banks: user-ok / password-ok
```

1. The CLI creates a connect token and serves a small page on `http://127.0.0.1:<random-port>/<secret>`.
2. It opens the page in your browser. The page runs the Pluggy Connect widget.
3. Each new item id is saved as soon as the widget reports it. Close the widget to finish.
4. The CLI then waits for Pluggy to finish collecting the data and runs a sync.

No webhook or public URL is involved: OAuth logins return to Pluggy's own callback.

## 4. Sync and check

```powershell
uv run llmopenfinance sync
uv run llmopenfinance status
```

## Good to know

| Topic | Behaviour |
|---|---|
| Refresh | MeuPluggy items mirror your MeuPluggy connections, which Pluggy refreshes every 24 h. They **cannot** be refreshed on demand, so `sync` just downloads what Pluggy already has and skips unchanged items. |
| Consent | Authorizations last **1 year**. `sync` warns 30 days before `consentExpiresAt`. After it expires, reads come back empty. Reconnect in meu.pluggy.ai, register the new item id and `items remove` the old one. |
| Categories | Pluggy's categorization, merchant data and recurring-payment insights are **paid** features (available during the trial). `llmopenfinance` categorizes everything locally, so answers stay consistent without them. |
| Loans | The MeuPluggy connector does not cover loans. The sync shows "empréstimos indisponível" and continues. |
| Transactions endpoint | The CLI uses `GET /v2/transactions` (cursor-based). The legacy `GET /transactions` returns `410 Gone` for apps created after 2026-06-02. |
| Rate limits | 429 responses are retried, honouring `RateLimit-Reset` and `Retry-After`. The API key (valid for 2 h) is cached and renewed automatically. |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401 CLIENT_KEYS_UNAUTHORIZED` | `PLUGGY_USERNAME`/`PLUGGY_PASSWORD` must be the application's clientId/clientSecret. |
| `404 ITEM_NOT_FOUND` on `items add` | Wrong id, or an id from another application. Copy it again from Demo → Connected Items. |
| `400 CLIENT_PLAN_REAL_CONNECTORS_NOT_ALLOWED` | You picked your bank's connector directly. On free/trial plans only MeuPluggy and Sandbox work. |
| `connect` page shows an error or never reports an item | Use path A (dashboard). The widget flow outside the dashboard Demo page is not officially documented for MeuPluggy. |
| Status `LOGIN_ERROR` / `OUTDATED` | Re-authorize the bank in meu.pluggy.ai. |
