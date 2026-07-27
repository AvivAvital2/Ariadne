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
uv run ariadne export            # Export ONE source-scoped zip artifact: ./docs/{source}.zip — only that source's docs travel (--no-archive writes the ./docs/{source}/ tree)
uv run ariadne import            # Import from ./docs/{source}.zip when present, else ./docs/{source}/ — delta: identical docs are skipped; large embed runs PROMPT live-vs-batch + cost (--live/--batch pick the mode, --yes approves; batch = half price via OpenAI Batch API)
uv run ariadne rebuild           # Rebuild embeddings (same live-vs-batch prompt on large runs; --live/--batch skip it, --yes skips the cost prompt)
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
uv run ariadne symbol_impact_radius <symbol> --depth 3   # Files affected by changing a symbol
uv run ariadne file_impact_radius <file>          # Files/tests/docs affected by changing a file (SCIP call-site graph)
uv run ariadne improve --dead-code        # Zero-reference symbols + stale rst (autodoc targets that no longer resolve)
uv run ariadne trace-flow <symbol> --depth 3      # Cross-language flow trace (SCIP + HTTP tiers)
```

`ariadne sync` auto-detects when changed files include a SCIP-routable language not yet declared in `index_kinds`, runs `discover --config-only`, and prints a hint to re-run `index` for the new language. Manual config of `index_kinds` / `scip:` blocks isn't required.

### Environment Spools
An opt-in **environment knowledge plugin**: a declarative, versioned pack of prebuilt docs/embeddings for a runtime (Databricks is Phase 1). Enable per project with a `spools:` mapping in `ariadne.yaml` (e.g. `spools: {databricks: {runtime: databricks-dbr17.3-lts}}` — the `runtime:` pin is **required**; an unpinned enable fails closed with a `runtime-unpinned` gap so a spool can't silently accept any version); registered spools join the query scope under a reserved `spool:<name>` source id.
```bash
uv run ariadne spools                 # Status: each enabled spool registered (target runtime) or a structured gap; exit 1 iff gaps
uv run ariadne spools create [ENV]    # ONE command — interactively set up ./spools.yaml (which spool + each version; repo set from a built-in recipe OR discovered on GitHub by name) then build: consent → fetch (clone at pins) → source add → index → onboard (cost prompt) → pack
uv run ariadne spools create --batch  # Pre-select batched onboard embeddings AND batched theme summaries (~half price); --live picks live; no flag → ONE live-vs-batch prompt at create scope drives both. Batch (flag or prompt) + missing theme-model API key fails BEFORE consent/fetch
uv run ariadne spools theme NAME --batch  # Build/refresh a spool's OWN themes post-hoc: free clustering, then PAID summaries (cost disclosed from the real prompts first; --batch ~half price). Interactive create gates the same spend with a y/N prompt (--yes skips)
uv run ariadne spools create --yes --batch        # Non-interactive: skip setup + all prompts, build an existing complete spools.yaml (CI)
uv run ariadne spools create --allow-ungrounded   # Build even when the corpus language has no SCIP indexer (docs-only, no code-tier grounding)
uv run ariadne spools install PACK.zip            # Verify checksum + install a pack into the store + cache
```
**Generation model:** spool builds use the global `model` from `ariadne.yaml`. Set `spools_model:` there to override it **for spool builds only** — e.g. keep day-to-day generation on `claude-opus-4-8` and set `spools_model: claude-sonnet-5` so packs build on a cheaper model. Unset → inherit `model`.
**Leaner doc types:** a spool build defaults `architecture`, `qa`, and `diagram` **off** (a reference pack wants `explanation` + `gotcha`, plus `catalog` from its own phase) to cut cost — they stay opt-in in the interactive doc-type picker, and are excluded on a non-TTY build.
**Recipe (`spools.yaml`):** `runtime:` pins the runtime edition (enable fails closed if unpinned or on a mismatch); `languages:` lists the corpus languages; `corpus:` maps each repo to a pinned tag/sha — **you specify the versions** (`create`'s interactive setup prompts you for each; the repo set comes from the environment automatically), and `create` resolves any missing shas from the tags, shows them at consent, and pins them back (TOFU). **Where the repo set comes from:** a shipped recipe (`spool_content/recipes/<env>.yaml`) if one exists; otherwise `create` **searches GitHub for repositories named `<env>`** (exact-name, most-starred first) — one match is used, several are shown to pick from (or paste a URL), none prompts for a URL. A discovered single-repo environment has no separate runtime edition, so its chosen corpus version becomes the `runtime`. (Real GitHub search results — never a guessed/hallucinated repo URL.) **Grounding gate:** a Spool must be SCIP-indexable, so `create` refuses a corpus whose language has no registered SCIP indexer (e.g. Ruby) — the declared `languages:` are checked *before any fetch*, and the fetched corpus *before the paid `onboard`* — instead of silently falling back to the raw-file/ast-grep path. `--allow-ungrounded` (default off) overrides. Step-by-step build guide: `docs/building-a-databricks-spool.md`. Full design: `designs/spool-environment-plugin.md`.

### Configuration
See `ariadne.yaml` for:
- `default_source`: Default source name (e.g., "mylib")
- `sources`: Named paths to source code directories
- `docs_base`: Where to export documentation (default: ./docs)

Set `ARIADNE_CONFIG=/path/to/ariadne.yaml` to use a config file outside cwd.

**User-authored fields** under `sources.<name>:`: `path`, `depends_on`, `parent`, `branches`, `ref`, `exclude`, `exclude_dirs`, `exempt_dirs`, `swagger_paths`, `env_hints`, `ignore_staleness`, `low_confidence_doc_languages`. **Ariadne-managed** (written by `discover`): `index_kinds`, `scip:` block. Manual edits to the managed fields get regenerated on next `discover` run.

**`ignore_staleness`** (opt-in, default off) exempts a source from staleness checks — for repos that update rarely (e.g. only on releases), where the constant "stale, regenerate" nag is noise. Set `ignore_staleness: true` to exempt the whole source (this also disables the SCIP index-age gate, so an old `.scip` index is reused instead of forcing a re-index), or give a list of globs (`["vendor/**", "legacy/*.py"]`) to exempt only matching files. It suppresses the *content-changed → stale* signal only; never-documented files still surface as coverage gaps. Set it via `ariadne source add NAME --ignore-staleness`, in `ariadne.yaml`, or when `ariadne onboard` prompts.

**`low_confidence_doc_languages`** (default `[rst, markdown]`) marks which source languages are *human-authored prose* rather than code-derived ground truth. Docs generated from these languages are tagged `provenance: human-doc` and rank **below** code-derived docs for the same query, so stale or aspirational prose can't outrank what the code actually does. Give an explicit list to add languages (e.g. `[rst, markdown, html]`) or `[]` to opt a source out entirely.

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
    low_confidence_doc_languages: [rst, markdown]  # Human-prose langs ranked below code (default)
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
