# Ariadne - Documentation Library for Claude

Ariadne generates and manages documentation that helps Claude understand codebases.

## Documentation Layout

`docs/` holds **hand-written guides** at the top level — e.g. `new-project-onboarding.md`, `claude-code-integration.md`, `directory-scoping.md` — which are authoritative and linked from the README (fine to read). Exported per-source docs under `docs/{source}/` are **generated** and duplicate README/source content, so skip those to save context.

## Quick Reference

### Commands
```bash
uv run ariadne onboard           # Full pipeline in one run: free phases + cost preview, then PROMPTS to continue into paid phases (no re-indexing)
uv run ariadne onboard --approve # Skip the proceed-prompt (non-interactive/CI); --live/--batch also skip the LLM-mode prompt
uv run ariadne dry-run           # Estimate LLM cost of the pipeline; runs only the free phases, no API calls
uv run ariadne generate          # Generate docs from source code
uv run ariadne export            # Export docs to ./docs/{source}/
uv run ariadne import            # Import docs from ./docs/{source}/
uv run ariadne rebuild           # Rebuild embeddings
uv run ariadne check             # Check for stale documentation
uv run ariadne config            # Show current configuration
uv run ariadne source add NAME --path DIR   # Add/update a source in ariadne.yaml (no manual editing)
uv run ariadne source list       # List configured sources
uv run ariadne source remove NAME           # Remove a source from ariadne.yaml
uv run ariadne finding "text"    # Save a session finding/conclusion
uv run ariadne sync              # Catch up with git changes since last sync
uv run ariadne merge --dry-run   # Preview post-merge doc regeneration
uv run ariadne merge             # Regenerate stable docs after branch merge
uv run ariadne usage             # Show usage statistics (last 30 days)
uv run ariadne gaps              # Show documentation gap analysis
uv run ariadne gaps --analyze    # LLM-powered gap recommendations
uv run ariadne testimonials      # Show top-scored Q&A from the local best-of store
uv run ariadne mcp               # Start MCP server (stdio transport)
```

### Branch Documentation
```bash
uv run ariadne sync --vs-main --dry-run   # See affected docs on branch
uv run ariadne sync --vs-main --branch    # Generate branch-specific docs
uv run ariadne cleanup --expired          # Remove expired branch docs
uv run ariadne migrate --check            # Check source_files status
uv run ariadne migrate --source-files     # Backfill missing source_files
uv run ariadne migrate --fix-paths        # Normalize staleness DB paths
uv run ariadne generate --path sub/dir    # Generate docs for subdirectory only
```

### SCIP Cross-Source Intelligence
```bash
uv run ariadne discover --source X        # Walk tree, write manifest, AND auto-fill index_kinds + scip block in ariadne.yaml
uv run ariadne discover --source X --config-only   # Same but skip the future auto-index trigger
uv run ariadne index --source X           # Run scip-X per declared kind, then 10 persist_* steps fill library_scip
uv run ariadne callers <symbol>           # Cross-source caller tree
uv run ariadne callees <symbol>           # Cross-source callee tree
uv run ariadne impact_radius <symbol> --depth 3   # Files affected by changing a symbol
uv run ariadne improve --dead-code        # Symbols with zero references in any indexed source
uv run ariadne trace-flow <symbol> --depth 3      # Cross-language flow trace (SCIP + HTTP tiers)
```

`ariadne sync` auto-detects when changed files include a SCIP-routable language not yet declared in `index_kinds`, runs `discover --config-only`, and prints a hint to re-run `index` for the new language. Manual config of `index_kinds` / `scip:` blocks isn't required.

### Configuration
See `ariadne.yaml` for:
- `default_source`: Default source name (e.g., "mylib")
- `sources`: Named paths to source code directories
- `docs_base`: Where to export documentation (default: ./docs)

Set `ARIADNE_CONFIG=/path/to/ariadne.yaml` to use a config file outside cwd.

**User-authored fields** under `sources.<name>:`: `path`, `depends_on`, `parent`, `branches`, `ref`, `exclude`, `exclude_dirs`, `exempt_dirs`, `swagger_paths`, `env_hints`, `ignore_staleness`. **Ariadne-managed** (written by `discover`): `index_kinds`, `scip:` block. Manual edits to the managed fields get regenerated on next `discover` run.

**`ignore_staleness`** (opt-in, default off) exempts a source from staleness checks — for repos that update rarely (e.g. only on releases), where the constant "stale, regenerate" nag is noise. Set `ignore_staleness: true` to exempt the whole source (this also disables the SCIP index-age gate, so an old `.scip` index is reused instead of forcing a re-index), or give a list of globs (`["vendor/**", "legacy/*.py"]`) to exempt only matching files. It suppresses the *content-changed → stale* signal only; never-documented files still surface as coverage gaps. Set it via `ariadne source add NAME --ignore-staleness`, in `ariadne.yaml`, or when `ariadne onboard` prompts.

Rather than hand-editing `ariadne.yaml`, you can manage the user-authored fields from the CLI with `ariadne source`:
```bash
ariadne source add mylib --path /path/to/mylib          # create entry (bootstraps ariadne.yaml if absent)
ariadne source add mylib --depends-on otherlib --exclude "**/.env" --exclude-dirs build,dist
ariadne source add mylib --branches "feature/*,main" --ref main   # branch-scoped / ref-pinned
ariadne source list                                     # show configured sources
ariadne source remove mylib                             # delete an entry (--yes to skip confirm)
```
`source add` is idempotent — re-running with only the flags you want changes updates just those fields and leaves the rest intact. The first source added to a fresh project becomes `default_source`. Next steps after adding: `ariadne discover --source NAME` then `ariadne onboard --source NAME`.

> **Note:** commands that rewrite `ariadne.yaml` (`source add`/`remove`, and `discover`/`onboard` which auto-write the managed `index_kinds`/`scip` block) serialize via PyYAML, which does **not** preserve comments or custom formatting. If you hand-author a commented `ariadne.yaml`, expect comments to be dropped the first time any of these commands rewrites it. Keep notes elsewhere, or prefer the CLI as the source of truth.

### Behavioral Directive & Usage Tracking
- `mention_ariadne`: Config option (enabled by default) that injects a directive telling Claude to mention when Ariadne helped
- Usage tracking: MCP tools auto-log calls; Claude reports hits/misses via `ariadne_log_hit`/`ariadne_log_miss`
- Gap analysis: `ariadne gaps` shows missed topics; `--analyze` runs LLM-powered recommendations

### Role-aware responses (optional)
`ariadne_search` / `ariadne_ask` accept an optional `role` kwarg. Default `'developer'` returns the existing technical docs unchanged. When the user's phrasing implies a different audience ("from a product perspective", "for the stakeholder review"), pass `role='product_manager'` — Ariadne caches audience-adapted responses on demand and reuses them for repeat questions. Dev content stays accessible via separate `role='developer'` queries ("explain further"). See `designs/role-aware-responses.md` for the data model + cascade-invalidation semantics.

### Source Configuration Schema
```yaml
sources:
  mylib:
    path: /path/to/mylib          # Required: source directory
    depends_on: [otherlib]        # Dependencies to load as context
    parent: parentlib             # For subdirectory sources
    branches: ["feature/*"]       # Branch patterns where active
    ref: main                     # Pin to specific git ref
    ignore_staleness: true        # Opt out of staleness checks (or a glob list)
```

### Scope-Aware Commands
```bash
ariadne manifest --auto-scope    # Auto-detect scope from cwd + branch
ariadne search --include-all     # Search all docs regardless of scope
```

### Conflict Resolution
When documents from multiple sources conflict:
1. Subdirectory source wins over parent
2. Branch-specific source wins over base
3. Most recently updated wins (tiebreaker)

## For Target Codebases

To integrate Ariadne docs into a codebase, add to that project's CLAUDE.md:

```markdown
## Knowledge Base
Before exploring code, check Ariadne docs for pre-researched documentation:
- `/path/to/Ariadne/docs/{source}/manifest.yaml` - Index of all documents
- `/path/to/Ariadne/docs/{source}/explanations/` - How systems work
- `/path/to/Ariadne/docs/{source}/architecture/` - Design decisions
- `/path/to/Ariadne/docs/{source}/findings/` - Session discoveries and conclusions

When asked about the codebase:
1. First check if Ariadne docs have relevant documentation
2. Read those docs before grepping/globbing source code
3. Only explore source if the library doesn't cover the topic

## Saving Findings
When you discover something noteworthy during a session, **proactively suggest** saving it to Ariadne. Examples of findings worth saving:
- Unexpected behavior or design patterns (e.g., "X never uses Y despite the interface")
- Key architectural decisions or constraints
- Important relationships between components
- Corrections to previous assumptions

To save a finding:
\`\`\`bash
cd /path/to/Ariadne && uv run ariadne finding "Your finding here" --topic "Topic Name" --source-files "path/to/file.py" --no-embed
\`\`\`

Then export to persist: `uv run ariadne export`
```

And add a SessionStart hook in `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|resume|clear",
        "hooks": [
          {
            "type": "command",
            "command": "echo '## Ariadne Knowledge Base Active' && cat /path/to/Ariadne/docs/{source}/manifest.yaml 2>/dev/null | head -40 || echo 'Ariadne docs not found'"
          }
        ]
      }
    ]
  }
}
```

Note: The `matcher` must specify when the hook runs: `startup` (new session), `resume` (continued session), or `clear` (after /clear).

## Documentation Structure

Exported docs follow this structure:
```
docs/{source}/
  manifest.yaml        # Index with metadata
  README.md
  explanations/        # How things work
    core-data-abstractions.md
    feature-system.md
  architecture/        # Design decisions
    llm-caching-architecture.md
  findings/            # Session discoveries and conclusions
    sqlbackedfeature-vs-join_strategies.md
```
