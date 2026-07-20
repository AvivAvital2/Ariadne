# Commands

| Command | Description |
|---------|-------------|
| `ariadne init` | Initialize Ariadne integration for a project |
| `ariadne source add/list/remove` | Manage `ariadne.yaml` source entries from the CLI (bootstraps the file; no hand-editing) |
| **Onboarding & cost** | |
| `ariadne onboard` | Full pipeline in one run: free phases + cost preview, then prompts to continue into the paid phases (no re-indexing). `--approve` skips the prompt; `--live`/`--batch` skip the LLM-mode prompt |
| `ariadne dry-run` | Run the free phases (discover, index, catalog-sync) and estimate the cost of the LLM-paid phases — no API calls |
| **Generation** | |
| `ariadne generate` | Generate LLM documentation from source code |
| `ariadne improve` | Run improvement cycle: analyze gaps, regenerate weakest docs |
| `ariadne topic` | Generate a cross-cutting topic doc spanning multiple files |
| `ariadne docs` | Generate user-facing documentation site (MkDocs) |
| **Catalog (structural)** | |
| `ariadne catalog-sync` | Build/refresh structural catalog via ast-grep + SCIP |
| `ariadne catalog-describe` | Generate LLM descriptions for catalog elements |
| `ariadne notify-changed` | Incremental catalog update for specific changed files |
| `ariadne symbol --name <qname>` | Look up a catalog element by qualified name |
| **Themes** | |
| `ariadne themes build` | Run Leiden clustering + theme summarization |
| `ariadne themes list` | List discovered themes |
| `ariadne themes show <id>` | Show a theme's content and members |
| **Search & retrieval** | |
| `ariadne search "query"` | Semantic search across documentation |
| `ariadne list` | List all documents |
| `ariadne get <id>` | Retrieve a document by ID |
| `ariadne stats` | Show library statistics |
| `ariadne usage` | Show MCP usage statistics and feedback; `--export-report PATH` writes a portable analytics report |
| `ariadne gaps` | Generate miss report; `--analyze` for LLM recommendations |
| `ariadne testimonials` | Show the best-of Q&A from the local store, ranked by richness (score + diagram + source-file citations + detail); `--limit`, `--export DIR` (images), `--export-html FILE` (self-contained showcase page) |
| **Maintenance** | |
| `ariadne check` | Check for stale or missing documentation |
| `ariadne sync` | Sync docs with git changes since last sync (delta — only changed files) |
| `ariadne export` | Export the database to a single zip (`docs/<source>.zip`); `--no-archive` writes a markdown tree |
| `ariadne import` | Import a zip or markdown tree into the database (delta — unchanged docs skipped); `--batch`/`--live`/`--yes`/`--skip-embeddings` control the embedding rebuild |
| `ariadne rebuild` | Rebuild embeddings for all documents |
| `ariadne add` | Add a new document manually |
| `ariadne delete <id>` | Delete a document |
| `ariadne finding` | Save a session finding/insight |
| `ariadne tag <id>` | Tag a document with metadata (status, branch) |
| `ariadne manifest` | Output filtered manifest for session hooks |
| `ariadne cleanup` | Clean up expired or orphaned documents |
| `ariadne migrate --doc-ids` | Migrate legacy UUID4 docs → deterministic UUID5 IDs |
| `ariadne migrate --infer-source-name` | Backfill missing `source_name` on legacy docs |
| `ariadne migrate --check` | Show migration status without changes |
| `ariadne migrate --source-files` | Backfill missing `source_files` |
| `ariadne migrate --fix-paths` | Normalize staleness DB paths |
| `ariadne migrate --fix-catalog-language` | Re-detect catalog `file_index` languages |
| `ariadne vacuum` | Optimize SQLite file size |
| `ariadne sync-claude-md` | Sync edited CLAUDE.md back to Ariadne |
| `ariadne edit-instructions` | Edit Ariadne CLAUDE.md in $EDITOR |
| **SCIP cross-source intelligence** | |
| `ariadne discover` | Walk a source tree, write `.ariadne/manifest.json` AND auto-author `index_kinds` + `scip:` blocks in `ariadne.yaml` |
| `ariadne index` | Run SCIP indexers per the manifest, merge to `.ariadne/index.scip`, then run all 10 `persist_*` steps to fill `library_scip` tables |
| `ariadne callers <symbol>` | Show what calls a symbol (cross-source, compiler-precise) |
| `ariadne callees <symbol>` | Show what a symbol calls |
| `ariadne impact_radius <symbol>` | Show files affected by a change to a symbol |
| `ariadne improve --dead-code` | Surface zero-reference symbols (requires SCIP indexes) |
| **MCP** | |
| `ariadne mcp` | Start the MCP server (stdio transport) |

### Common Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `--types` | generate | Doc types to generate (explanation,architecture,qa,diagram) |
| `--concurrency` | generate | Max concurrent LLM requests (default: 3) |
| `--verbose/-v` | generate, check | Show detailed output (validation reports for failures) |
| `--chunks` | search | Search at chunk level instead of document level |
| `-k` | search | Number of results (default: 5) |
| `--auto-scope` | manifest | Auto-detect source based on cwd and branch |
| `--include-all` | search | Include all docs regardless of scope |

### Common Options

```bash
# Use a specific source
uv run ariadne generate --source myproject
uv run ariadne export --source myproject

# Force regeneration
uv run ariadne generate --force

# Dry run (don't save)
uv run ariadne generate --dry-run

# Custom database path
uv run ariadne --db /path/to/library.db list
```


---

_Part of the [Ariadne documentation](../README.md)._
