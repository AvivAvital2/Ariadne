# Slack Bridge — Production Deployment

`ariadne-slack` is a **read-only Slack bot** that bridges **Slack → Claude (Agent SDK) → Ariadne's MCP tools**: a user @mentions it, DMs it, or runs `/ariadne`, and Claude answers from Ariadne's knowledge base. It runs in **Socket Mode** — it dials *out* over a WebSocket, so there's no public URL and no inbound ports.

This is the production runbook. The deployment model is a **serve/build split**: build `ariadne.db` on a beefy box (or CI), ship it to a small always-on host, and run the bridge there *serve-only*. The serving box never generates docs or runs the SCIP indexers.

## Credential model (read this first)

The bridge is strict about credentials, and one mistake stops it at startup:

| Token | Where it comes from | Env var |
|---|---|---|
| Bot token `xoxb-…` | installing the app | `SLACK_BOT_TOKEN` |
| App-level token `xapp-…` | App-Level Tokens, scope `connections:write` | `SLACK_APP_TOKEN` |
| Claude OAuth token | `claude setup-token` | `CLAUDE_CODE_OAUTH_TOKEN` |
| OpenAI key | your account | `OPENAI_API_KEY` |
| Anthropic key | your account | `ARIADNE_ANTHROPIC_API_KEY` |

- **Never set `ANTHROPIC_API_KEY` in the bridge process.** It would switch the agent off your Claude subscription onto metered API billing, so the bridge **aborts at startup** if it sees it. The Ariadne-scoped key rides as `ARIADNE_ANTHROPIC_API_KEY` and is re-exported as `ANTHROPIC_API_KEY` *only inside* the Ariadne MCP subprocess (for `ariadne_ask`).
- `OPENAI_API_KEY` is scoped into that subprocess too (embeddings / search).
- The bridge reads the **process environment directly — it does not load `.env`.** Use `export`, or a systemd `EnvironmentFile`.
- `claude setup-token` mints a **portable** token: run it wherever you can complete the browser login (your laptop is easiest) and copy the value to the box. For a shared bot, mint it from a **dedicated account**, not a personal one.

## 1. Create the Slack app

From the manifest at [`slack-app-manifest.yaml`](../slack-app-manifest.yaml):

1. **api.slack.com/apps → Create New App → From an app manifest** → pick the workspace → paste the manifest → **Create**. This pre-fills the bot scopes, events, the `/ariadne` slash command, and Socket Mode.
2. **Basic Information → App-Level Tokens → Generate Token and Scopes** → add `connections:write` → copy the `xapp-…` value → `SLACK_APP_TOKEN`.
3. **Install App → Install to Workspace** → then **OAuth & Permissions → Bot User OAuth Token** → copy the `xoxb-…` value → `SLACK_BOT_TOKEN`.
4. **Basic Information → Display Information → App icon** → upload [`assets/Ariadne-icon.png`](../assets/Ariadne-icon.png). (The icon is not part of the manifest.)

## 2. Provision the box & install Ariadne

A small always-on host (≈2 GB RAM) is enough — the bridge serves queries, it doesn't generate. Outbound internet only (Slack + OpenAI/Anthropic); no inbound ports.

```bash
sudo useradd -m ariadne
sudo -iu ariadne
git clone <repo-url> /opt/ariadne && cd /opt/ariadne
uv sync
```

**Optional — diagram rendering.** To have the bot post diagrams as *images*, install Graphviz so the `dot` binary is on `PATH` (e.g. `sudo apt install graphviz`). It's used **only here, at the bridge**, to render stored DOT diagrams to PNG. Without it the bridge degrades gracefully — it posts the DOT source plus a one-line note instead of an image (and logs a warning); generation and search never need it.

**Do not build the database here.** Ship the prebuilt one from your build box:

```bash
rsync build-box:/path/to/ariadne.db /opt/ariadne/ariadne.db
```

You do **not** need the `.scip` index files or `ariadne_staleness.db` — both are build-time artifacts. The cross-source graph is already persisted inside `ariadne.db` (the `scip_symbols`/`scip_edges` tables) by `ariadne index`; the staleness DB only tracks per-file SHAs for incremental regeneration and is read solely by the generation path, which the serve-only bridge never runs.

## 3. Write the serving `ariadne.yaml`

Write a **minimal** config — *not* the build box's `ariadne.yaml`. Its `exclude`/`index_kinds`/`scip` blocks and absolute source paths are build-time and irrelevant to serving. Only these fields are read when the bot answers a query:

```yaml
default_source: ariadne
sources:
  ariadne: .          # path is vestigial for DB-only tools (see below)
  projecta: .
  projectb: .
defaults:
  db_path: ariadne.db
  provider: anthropic       # so ariadne_ask synthesizes with Claude
  model: claude-opus-4-8
```

Source `path:` values are **not** validated at config load — they're only dereferenced by the live-file-reading tools. That drives one decision:

- **DB-only (simplest):** drop `ariadne_read` / `ariadne_body` / `ariadne_source_path` from `READ_ONLY_TOOLS` in `slack_bridge/allowed_tools.py`. The paths above are then never used (placeholders like `.` are fine), and you need no source checkouts on the box. `search` / `symbol` / `explain` / `themes` / `impact_radius` / `coverage` all answer straight from `ariadne.db`.
- **Full source access:** keep those tools, check out each source repo on the box, and set its `path:` to the real location so the bot can quote live source.

## 4. Operational config (`slack_bridge.yaml`)

```bash
cp slack_bridge.yaml.example slack_bridge.yaml
```

Edit it — it holds **no secrets**:

- `allowed_users` / `allowed_channels` — **the access boundary.** Both empty = deny everyone (fail-closed); add the Slack user IDs (`U…`) / channel IDs (`C…`) allowed to use the bot.
- `pool.max_size` — each warm session holds its own Ariadne MCP subprocess; on a small box start at **5–10** (the default 50 can exhaust 2 GB).
- `source_descriptions` / `source_aliases` — one-liners that help the agent route a question to the right source.
- `enable_feedback` — opt-in; lets the bot record `ariadne_log_hit`/`miss` into `usage_events`.

## 5. Run it under systemd

`/etc/ariadne/slack.env` (`chmod 600`, owned by `ariadne`):

```ini
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
CLAUDE_CODE_OAUTH_TOKEN=...
OPENAI_API_KEY=sk-...
ARIADNE_ANTHROPIC_API_KEY=sk-ant-...
ARIADNE_SLACK_CONFIG=/opt/ariadne/slack_bridge.yaml
# Do NOT add ANTHROPIC_API_KEY — the bridge aborts if it sees it.
```

`/etc/systemd/system/ariadne-slack.service`:

```ini
[Unit]
Description=Ariadne Slack bridge (Socket Mode)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ariadne
WorkingDirectory=/opt/ariadne
EnvironmentFile=/etc/ariadne/slack.env
ExecStart=/opt/ariadne/.venv/bin/ariadne-slack
Restart=always
RestartSec=5
# the bridge spawns `uv run --directory /opt/ariadne ariadne mcp` per session,
# so uv must be on PATH with a writable cache:
Environment=PATH=/home/ariadne/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=HOME=/home/ariadne

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ariadne-slack
journalctl -u ariadne-slack -f      # wait for "Ariadne Slack bridge starting (Socket Mode)…"
```

## 6. Verify

- **Channel:** `/invite @Ariadne` (it must be in the channel to receive mentions), then `@Ariadne how does X work?`
- **DM:** message the bot directly.
- **Slash:** `/ariadne how does X work?`

The asking user/channel must be on the allowlist, or the bot replies that they're not set up.

### Asking with an image

`/ariadne` is **text-only** — Slack can't attach a file to a slash-command invocation (its payload carries no `files`), so an image can't ride along with it. To ask about an image, **@mention the bot or DM it** with the file attached. Accepted: **JPEG / PNG / GIF / WebP, ≤ 5 MB** (other formats — HEIC, SVG, PDF — are skipped); the bot token needs the `files:read` scope.

## Refreshing the knowledge base

Rebuild on the build box, then re-ship and restart:

```bash
# build box — regenerate, then:
rsync /path/to/ariadne.db serving-box:/opt/ariadne/ariadne.db
# serving box:
sudo systemctl restart ariadne-slack
```

Replacing `ariadne.db` resets `usage_events` on the box. If you want the hit/miss/score telemetry preserved across a refresh, export it before the swap with `ariadne usage --export-report <file>` (see the analytics-report flow in the README).

If you ship the optional embedding matrix (see below), rebuild and re-ship it in the same step — a matrix that no longer matches the DB is ignored (the bot falls back to SQLite ranking until you refresh it).

## Embedding matrix (optional — faster semantic ranking)

Ariadne can rank search candidates against a memory-mapped embedding matrix
instead of loading each candidate's embedding from SQLite per query. On a large
knowledge base this cuts the dominant cost of a cold query from **seconds to
~100 ms**. It is **entirely optional** — without the matrix, ranking uses the
SQLite path and the bot behaves exactly as before.

The matrix is a derived artifact, `.ariadne/doc_embeddings.npy` (~1 GB for ~80k
docs), built from the embeddings already in `ariadne.db` — no re-embedding, no
API calls. Follow the serve/build split: **build it on the build box and ship
it** (don't build it on a small serving box — the build briefly needs ~2× the
matrix size in RAM):

```bash
# build box — after generating the DB:
ariadne build-matrix                  # writes .ariadne/doc_embeddings.npy (+ .meta.json)
rsync /path/to/.ariadne/doc_embeddings.* serving-box:/opt/ariadne/.ariadne/
# serving box:
sudo systemctl restart ariadne-slack
```

- The server **loads** the matrix; by default it never builds one. It
  freshness-checks the file against the DB, so a matrix that doesn't match the
  shipped `ariadne.db` is ignored and ranking falls back to SQLite — never
  wrong, just not accelerated. **Rebuild and re-ship the matrix whenever you
  ship a new `ariadne.db`** (or delete the stale file).
- **Add or remove it any time:** copy the file in and restart to turn the
  speedup on; delete it and restart to turn it off.
- **Building on the serving box is opt-in and off by default.** Only sensible on
  a roomy, non-pooled box (the build can spike ~2 GB RAM). To enable it, set
  `ARIADNE_BUILD_MATRIX_ON_STARTUP=1` in `slack.env`; leave it unset to keep
  startup load-only.

## Security notes

- **Outbound-only** (Socket Mode) — no inbound ports to open.
- Secrets live in `slack.env` (`chmod 600`), never in `slack_bridge.yaml` or git.
- The **allowlist is the access control**: allow-listed users can query the knowledge base — and, if `read`/`body` are enabled, real source. Set it deliberately.
- Keep the box in a private subnet with NAT for outbound where you can; lock down SSH.
