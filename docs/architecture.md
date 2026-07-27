# How It Works

1. **Discovery**: Walks the source tree, honoring `exclude` / `exclude_dirs` and per-language type filters (e.g., JSON files only get `explanation`, never `architecture`).

2. **Catalog extraction**: SCIP (for Python, JS/TS, Scala, Java, Go) or ast-grep (for HTML and the config/doc formats) extracts every public element. One catalog doc per element, plus a `file_index` doc per file linking them.

3. **Per-file LLM generation**: For each requested doc-type (`explanation`, `architecture`, `qa`, `gotcha`, `diagram`), the LLM produces a markdown doc. Validation runs structural checks (closed code blocks, required sections); on failure, retries up to twice with temperature variance. ~5–10% of generations fail-after-retries on large or unusual files; those file/type combos remain stale until re-run picks them up.

4. **Themes**: After generation, Leiden community detection over the hybrid graph (structural + semantic) discovers cross-cutting clusters; each is LLM-summarized into a topic doc.

5. **Crossrefs**: Graph BFS injects a "Related Documents" section into every doc, linking related members within scope.

6. **Storage**: All docs (LLM-written, catalog, theme, finding) live in a SQLite database with embeddings for semantic search. Deterministic UUID5 IDs make re-runs idempotent.

7. **Staleness tracking**: A separate SQLite tracks per-file SHA + which doc-types succeeded. Type-aware: a file with `explanation` but no `architecture` is stale when `--types architecture` is requested. Language-filtered: JSON files aren't considered stale for unsupported types.

8. **Cost & telemetry**: Per-run logs at `INFO` level land in `ariadne_runs/generate-<timestamp>.log` (no `--verbose` required). `--dry-run` projects cost upfront with caching/batch discounts. Quota exhaustion triggers a graceful abort with a resume summary; the next run picks up where the previous left off.

9. **Integration**: Agents query the database via the MCP server; CLAUDE.md and SessionStart hooks make Ariadne the first thing a new Claude session reads.

## Subsystems

### Catalog (structural index)

Beyond LLM-written docs, Ariadne maintains a **structural catalog** of every public class, method, function, and module-level value across the codebase. Built via:

- **SCIP** for Python, JavaScript/TypeScript (incl. Vue), Scala, Java, and Go — via `scip-python`, `scip-typescript`, `scip-java`, and `scip-go`; precise compiler-derived symbols and call graphs, not heuristics
- **ast-grep** for HTML, JSON, YAML, Markdown, HOCON, CSS, and Dockerfiles

The catalog powers `ariadne_symbol "module.ClassName"` lookups, fine-grained search, and cross-file relationship graphs that the crossref injector uses to link related docs.

```bash
ariadne catalog-sync --source mylib       # build/refresh structural index
ariadne catalog-describe --source mylib   # generate one-line LLM descriptions per element
ariadne notify-changed --files a.py b.py  # incremental refresh
ariadne symbol --name mypkg.Foo.bar       # look up an element by qualified name
```

Catalog data is stored alongside generated docs in the same SQLite library and uses deterministic IDs (`UUID5` of source + content_type + element key), so re-running is idempotent — same element produces the same row.

### Themes (cross-cutting concerns)

Code rarely organizes itself by concern: authentication might span 30 files across 5 modules. Ariadne's **themes pipeline** discovers these clusters automatically:

1. **Hybrid graph** — combines structural edges (imports, file containment) with semantic edges (k-nearest-neighbor by doc embedding cosine similarity)
2. **Leiden community detection** — clusters tightly-connected nodes
3. **LLM summarization** — each cluster is summarized into a topic doc; the LLM may declare a cluster `INCOHERENT` (algorithmic noise) and skip it
4. **Stable cluster IDs** — clusters track across runs via Jaccard overlap, so a "retry logic" theme keeps the same ID even as members shift

Themes appear in `ariadne_search` results alongside per-file docs, and `ariadne_themes show <id>` reveals the cluster's members.

```bash
ariadne themes build       # cluster + summarize (auto-runs at end of `generate`)
ariadne themes list        # show discovered themes
ariadne themes show <id>   # show one theme with members
```

### Cross-references (graph-based)

After generation, Ariadne walks the doc graph (imports + topic membership + semantic neighbors) via BFS to inject a "Related Documents" section into each doc. Scoped to current source + dependencies — search results don't bleed across unrelated codebases. Uses precomputed edges (O(N×K) where K is the BFS frontier ~10–50), not O(N²) regex scanning.

### Environment spools (opt-in)

A **spool** is a prebuilt, version-pinned knowledge pack for a runtime (Databricks is Phase 1): its corpus is cloned at pinned SHAs, SCIP-indexed, documented, and themed **once** on a build machine, then distributed as a checksum-verified zip. Installing costs no cloning, indexing, or LLM spend; enabled spools join the query scope under the reserved `spool:<name>` source id.

At query time a deterministic **bidirectional lens** routes each question: repo-subject questions rank your code first with the spool as a capped, labeled lens; environment-subject questions flip the sides. The environment's *names* only mark the seam — they never decide the subject or select documents. Answers carry the runtime pin + component versions, pinned `@Since`/deprecation **version facts** extracted from the corpus, and a provenance line citing each corpus SHA + detected license.

Build/install walkthrough: [building-a-databricks-spool.md](building-a-databricks-spool.md).

### Staleness model

Files are checked at the start of every `generate` run and processed only if needed:

- **File hash changed** (SHA-256 of bytes differs from last sync) → regenerate all requested doc-types for this file
- **File hash unchanged** AND **all requested doc-types already exist** (filtered by language) → skip the file entirely
- **File hash unchanged** but a doc-type is missing AND the language supports it → regenerate just the missing type
- **Validation failed-after-retries** for a doc-type → that doc isn't recorded; next run picks it up; persistent failures (typically 5–10% of large files) stay stale until source changes

The language filter is critical: a JSON file with `explanation` only is not stale just because `architecture` was requested — JSON doesn't support architecture docs per `LANGUAGE_DOC_TYPES`. Without this, JSON/YAML/MD files would be eternally stale.

The staleness DB lives at `ariadne_staleness.db` next to `ariadne.db`. It tracks `(path, sha, documented_at, doc_ids)` per file.

### Workflow: catalog-sync vs generate

`catalog-sync` and `generate` are independent commands that build different layers:

```bash
ariadne catalog-sync --source mylib   # cheap (~$5-10), structural only, no LLM generation
ariadne generate --source mylib       # expensive ($X to $$$), LLM commentary on top
```

For a fresh codebase: run **both** in order. Catalog gives you symbol lookup and structural search; generate gives you explanations/architecture/etc. on top.

For incremental work after edits: `generate` alone is usually sufficient — it's idempotent (deterministic IDs) and only touches files whose hash changed or whose doc-types are missing. Run `catalog-sync` again only when files were added/deleted (catalog won't auto-discover new files via generate).

`generate` does NOT auto-trigger `catalog-sync`. They share the same SQLite library but run on different signals.

## Usage Tracking

Ariadne tracks how its MCP tools are used and whether results are helpful. This data drives documentation quality improvement.

### What's Tracked

Each MCP tool call automatically logs:
- **Timestamp** — when the tool was called
- **Tool name** — which tool (search, list, etc.)
- **Query** — what was searched for
- **Result count** — how many results returned
- **Outcome** — `call` (default), `hit` (useful), or `miss` (not useful)
- **Feedback** — optional text describing what was helpful or missing

### How Data is Obtained

1. **Auto-logged**: Every MCP tool call is automatically recorded with an event ID
2. **Claude reports back**: After using results, Claude calls `ariadne_log_hit` or `ariadne_log_miss` with the event ID and optional feedback
3. **Event ID**: Appears at the end of each tool result as `[Usage event: <id>]`

### Querying Statistics

```bash
# CLI
ariadne usage                    # Last 30 days
ariadne usage --days 7           # Last 7 days
ariadne usage --tool ariadne_search  # Filter by tool

# Gap analysis
ariadne gaps                     # SQL-based quick report
ariadne gaps --analyze           # LLM-powered recommendations
```

Claude can also query stats via MCP tools: `ariadne_usage_stats` and `ariadne_gaps`.

### Improving Documentation

The gap analysis identifies patterns in miss feedback:
- Which topics users search for but don't find
- Which queries return results that aren't useful
- Recommendations for new documentation to create

Run `ariadne gaps --analyze` for LLM-powered recommendations, or ask Claude: "What documentation gaps does Ariadne have?"


---

_Part of the [Ariadne documentation](../README.md)._
