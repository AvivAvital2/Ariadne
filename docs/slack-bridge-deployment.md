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
- `allow_all` — **org-wide override** (default `false`). Set `true` to let *anyone who can reach the bot* use it (any user, channel, or DM), ignoring the two lists above — convenient for a whole-workspace rollout. The bot runs on your Claude subscription, so this opens that cost to the entire org; leave it `false` unless that's intended.
- `allowed_orgs` — **hard org boundary**, checked *before* everything above and **independent of `allow_all`**. List your workspace's team id (`T…`) and/or Enterprise Grid org id (`E…`); a request is then served only if it comes from a listed org **and** isn't an externally-shared (Slack Connect) channel — anything else is silently ignored. This is the defense-in-depth layer so Slack's own config isn't your only boundary: e.g. `allow_all: true` + `allowed_orgs: [T0YOURTEAM]` = "anyone in *your* org, internal channels/DMs only; every outsider ignored." Empty ⇒ off. The bridge logs its own `team`/`enterprise` id at startup so you know what to set here. For surfaces whose payload carries no shared flag — **slash commands** and **Slack Connect DMs** — it confirms the channel isn't externally shared via `conversations.info` (cached), which needs the `channels:read`/`groups:read`/`im:read`/`mpim:read` scopes (all in the manifest — **reinstall** the app if you're upgrading); if that lookup fails it **fails closed** (treats the channel as shared and ignores it). And if `allow_all` is on with no `allowed_orgs`, the bridge logs a startup warning — you'd be relying on Slack config alone.
- `pool.max_size` — each warm session holds its own Ariadne MCP subprocess; on a small box start at **5–10** (the default 50 can exhaust 2 GB).
- `source_descriptions` / `source_aliases` — one-liners that help the agent route a question to the right source.
- `source_titles` — friendly display labels for the **`/ariadne greet`** announcement's *Covers* list (e.g. `dp: 'Discovery Platform (DP)'`). A source with no title shows its bare key; leave the whole map out and `greet` simply omits the Covers block.
- `enable_feedback` — opt-in; lets the bot record `ariadne_log_hit`/`miss` into `usage_events` and makes the agent **score each answer** (`score:N`). Required for the testimonials best-of store to populate live (see below).

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

**Announce it:** run **`/ariadne greet`** in the launch channel — the bot posts a public "Meet Ariadne" intro (how to ask, naming a project, diagrams/altitude tips, and a *Covers* list built from `source_titles`). It's a canned render — no agent turn, no cost — and a bare **`/ariadne`** shows the same usage tips any time.

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

## Testimonials (best-of showcase)

The bridge keeps an all-time **top-20 of the highest-scored Q&A** in `.ariadne/local/` — a store that is **swap-proof** (it's gitignored and lives outside `ariadne.db`, so a knowledge-base refresh can't wipe it). Scoring requires **`enable_feedback: true`**: the agent is then instructed to rate each answer (`score:N` via `ariadne_log_hit`), so good answers are captured **live** as they happen. You can also **backfill from existing history** with a one-off run on the serving box, using the same env as the service:

```bash
sudo -iu ariadne && cd /opt/ariadne
set -a; . /etc/ariadne/slack.env; set +a        # same env the systemd unit loads

.venv/bin/ariadne-slack scan                    # all public channels the bot is in
.venv/bin/ariadne-slack scan --channel C0123    # pin to one channel (repeatable; reaches private)
.venv/bin/ariadne-slack scan --channel C0123 --generate-scores   # also LLM-score un-scored history
.venv/bin/ariadne-slack scan --limit 200        # cap how many past pairs (newest first)
.venv/bin/ariadne testimonials                  # read the store (ranked by richness, not bare score)
.venv/bin/ariadne testimonials --export-html best-of.html   # self-contained HTML showcase page
```

The store is ranked by **richness** — the quality score plus feature-rich signals (a diagram attached, distinct source files cited, thoroughness) — so the most detailed, diagram-backed answers surface first, not just the highest bare score. The backfill also **downloads the bot's rendered diagrams** from each thread (needs the `files:read` scope), so historical answers carry their diagrams too. `--export-html` writes a single self-contained page (inline CSS, diagrams embedded as base64) you can open, share, or drop into a deck.

**Scope.** With no `--channel`, `scan` reads **only the public channels the bot is a member of** (`/invite @Ariadne` first); it *lists* them via `conversations.list`, which needs the **`channels:read`** scope (the manifest grants it — an app created from an older manifest must add it under **OAuth & Permissions** and reinstall), and it never touches private channels or DMs. Pass **`--channel C…`** (repeatable) to target channels **by id, read directly** — no listing, so this reaches a **private** channel the bot is in, using the bot's existing `groups:history` scope (no extra scope needed). Scope is the bot's **channel membership, not `allow_all`**: opening the bot org-wide changes *who can ask*; it does **not** trigger or widen a backfill, and `scan` runs only when you run it.

**Cost & timing.** By default `scan` reads the scores already in `usage_events` (joined to each answer by time) — **no agent turn, no cost**. Add **`--generate-scores`** to also have Claude rate answers the DB never scored (so chatter from before scoring was enabled is still captured); that **runs the agent on the subscription** — it needs `CLAUDE_CODE_OAUTH_TOKEN`, and the no-`ANTHROPIC_API_KEY` cost gate applies, one LLM call per un-scored pair.

The scan is a **delta**: a pair already in the store is skipped *before* scoring, so re-runs cost nothing for history already captured (and never duplicate). To re-judge entries you've already stored — e.g. to apply a changed scoring rubric — add **`--rescore`**, which re-scores and replaces them in place (no need to delete the store by hand):

```bash
.venv/bin/ariadne-slack scan --channel C0123 --generate-scores --rescore   # apply a new rubric to existing entries
```

Because DB scores live in `usage_events` (wiped on a swap), **run `scan` before you refresh `ariadne.db`** if you want that history captured.

**If `scan` records 0**, read its log line — `scan: N channel(s) read, M Q&A pair(s), K DB-scored + G generated, …`:
- **0 channels** — the bot isn't a member/listed for the target. Invite it (`/invite @Ariadne`); for a private channel, name it with `--channel C…`.
- **0 pairs** — no threaded Q&A found (the bot answers in-thread; a channel of plain chatter with no bot answers yields nothing).
- **pairs but 0 DB-scored** — those answers were never scored (scoring needs `enable_feedback: true` *and* the agent emitting `score:N`; chatter from before that can't be DB-matched). Either confirm scores exist — `sqlite3 ariadne.db "SELECT count(*) total, count(quality_score) scored FROM usage_events"` — or re-run with **`--generate-scores`** to have Claude rate the un-scored ones now.

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
