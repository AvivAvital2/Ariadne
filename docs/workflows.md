# Workflow

### Initial Setup

```bash
# 1. Generate docs from source
uv run ariadne generate

# 2. Export to markdown
uv run ariadne export

# 3. Configure target project's CLAUDE.md and hooks
```

### Keeping Docs Fresh

```bash
# Check what's stale
uv run ariadne check --verbose

# Regenerate stale docs
uv run ariadne generate

# Re-export
uv run ariadne export
```

### After Code Changes

```bash
# Quick check
uv run ariadne check

# If stale, regenerate
uv run ariadne generate --force
uv run ariadne export
```

### Git-Aware Sync

The `sync` command tracks git changes and regenerates only affected docs:

```bash
# Check what needs updating
uv run ariadne sync --status

# Sync docs with git changes
uv run ariadne sync

# Preview without making changes
uv run ariadne sync --dry-run
```

### Automatic Sync via Git Hook

You can configure a git `post-commit` hook in your target project to automatically sync Ariadne docs after every commit. This keeps documentation up-to-date without manual intervention.

#### Prerequisites

1. **API key**: Create a `.env` file in your Ariadne directory:
   ```bash
   # /path/to/ariadne/.env
   OPENAI_API_KEY=sk-...
   ```

2. **Protect the key**: keep `.env` out of version control — Ariadne's `.gitignore` already excludes it, so the key is never committed or pulled into agent context.

#### Setup

Create `.git/hooks/post-commit` in your target project (the repo whose code Ariadne documents):

```bash
#!/bin/sh
# Auto-sync Ariadne docs after commits
ARIADNE_DIR="/path/to/ariadne"
ENV_FILE="$ARIADNE_DIR/.env"
LOG_FILE="$HOME/.ariadne-sync.log"

if [ -f "$ENV_FILE" ]; then
    (
        set -a
        . "$ENV_FILE"
        set +a

        OUTPUT=$(cd "$ARIADNE_DIR" && uv run ariadne sync -s <source_name> 2>&1)
        STATUS=$?
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

        echo "[$TIMESTAMP] exit=$STATUS" >> "$LOG_FILE"
        echo "$OUTPUT" >> "$LOG_FILE"
        echo "---" >> "$LOG_FILE"

        if [ $STATUS -eq 0 ]; then
            TITLE="Ariadne Sync Complete"
            MSG="Documentation synced successfully."
        else
            TITLE="Ariadne Sync Failed"
            MSG="Check ~/.ariadne-sync.log for details."
        fi

        # Cross-platform notification
        case "$(uname)" in
            Darwin)
                osascript -e "display notification \"$MSG\" with title \"$TITLE\"" 2>/dev/null
                ;;
            Linux)
                notify-send "$TITLE" "$MSG" 2>/dev/null
                ;;
        esac
    ) &
fi
```

Make it executable:
```bash
chmod +x .git/hooks/post-commit
```

Replace `/path/to/ariadne` with your Ariadne directory and `<source_name>` with the source defined in `ariadne.yaml`.

#### How it works

- Sync runs **in the background** so it doesn't slow down commits
- Output is logged to `~/.ariadne-sync.log`
- A native OS notification appears on completion (macOS via `osascript`, Linux via `notify-send`)

#### Verification

```bash
# Check the log after a commit
tail -20 ~/.ariadne-sync.log
```

### Branch-Specific Documentation

When working on feature branches, code changes may make existing documentation stale.
Ariadne can detect affected documents and create branch-specific versions.

```bash
# See which documents are affected by your branch changes
uv run ariadne sync --vs-main --dry-run

# Regenerate affected docs for this branch (tagged as experimental)
uv run ariadne sync --vs-main --branch
```

Branch-specific docs:
- Tagged with `status: experimental` and `branches: [your-branch]`
- Automatically shown when on that branch
- Expire after 180 days (configurable via `branch_doc_ttl_days`)

#### Cleanup Expired Docs

```bash
# Remove expired branch-specific documents
uv run ariadne cleanup --expired

# Preview what would be removed
uv run ariadne cleanup --expired --dry-run
```

#### Configuration

In `ariadne.yaml`:
```yaml
main_branch: main          # Branch to compare against (default: main)
branch_doc_ttl_days: 180   # Expiration for branch docs (default: 180)
```

### Migration

If existing documents are missing `source_files` metadata:

```bash
# Check migration status
uv run ariadne migrate --check

# Attempt to backfill source_files
uv run ariadne migrate --source-files
```

### Capturing Findings

Save quick insights during a session:

```bash
# Save a finding (opens editor or reads from stdin)
uv run ariadne finding -t "Why caching uses MD5"

# Associate with source files
uv run ariadne finding -s src/cache.py -t "Cache invalidation logic"
```

### Replacing the Database & Fetching Analytics

When you run Ariadne in a **serve/build split** — build `ariadne.db` on a beefy box (or CI), ship it to a small always-on serving box, and *replace* the file there — the swap wipes `usage_events` (it lives inside `ariadne.db`). That reset is usually what you want: each generation gets a clean slate of hit/miss/score telemetry. The catch is that the swap also discards the signal `improve`/`gaps` feed on. The fix is to **fetch the analytics into a portable report before the swap**, so the insight survives the replacement and accumulates across generations.

#### Fetch analytics into a portable report

On the serving box, *before* replacing the database:

```bash
uv run ariadne usage --export-report analytics-2026-06.json
```

This writes a self-contained JSON `AnalyticsReport` distilled from `usage_events` — independent of the database, so it outlives the swap:

| Field | Source | Carries |
|-------|--------|---------|
| `usage_summary` | `get_usage_stats` | calls / hits / misses, hit-rate, per-tool breakdown, **and the response-score aggregates** (`avg_quality_score`, `score_distribution`) |
| `missed_queries` | `get_gap_report` | miss feedback grouped by theme, with counts and example queries |
| `recent_misses` | `get_gap_report` | raw miss events (the input to LLM gap analysis) |
| `doc_signals` | `usage_by_document` + `find_low_value_documents` | top-served docs and served-but-never-hit docs |
| `gaps` *(optional)* | `analyze_gaps` | the LLM `GapReport` insights, when you pass them in |

Response **scores** are captured the way Ariadne already records them: a `score:N` (1–10) embedded in hit/miss feedback (e.g. `ariadne_log_hit(event_id, "score:8 — found it")`) lands in `usage_events.quality_score` and is aggregated into `usage_summary`.

#### The lifecycle

1. **Gather** — the serving box records `usage_events` as it answers. (For the Slack bridge, set `enable_feedback: true` so hit/miss/score get written.)
2. **Fetch** — run `uv run ariadne usage --export-report <file>` before each rebuild; keep the JSON in a `reports/` history for a longitudinal view.
3. **Rebuild** — on the build box, regenerate `ariadne.db`, informed by the report's gaps and doc-signals.
4. **Replace** — ship the new `ariadne.db` to the serving box and swap it in. `usage_events` resets to empty, measuring the *new* generation; the exported report has already preserved the old signal.

The raw events are disposable per generation; the **report series is the durable record**.

#### Consuming the report

A saved report hands its signal back in the exact shape `improve`/`gaps` read from a live database, so it can drive a rebuild after the source DB has been wiped:

```python
from pathlib import Path
from analytics_report import AnalyticsReport

report = AnalyticsReport.from_json(Path("analytics-2026-06.json").read_text())
gap_report = report.as_gap_report()   # {total_misses, top_gaps, recent_misses}
# feed gap_report wherever a live get_gap_report() result is expected
```

> A `--from-report PATH` flag on `improve`/`gaps` (making the swap-in a single command) is the planned next step; `as_gap_report()` is the building block it will use.

---

_Part of the [Ariadne documentation](../README.md)._
