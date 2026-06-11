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


---

_Part of the [Ariadne documentation](../README.md)._
