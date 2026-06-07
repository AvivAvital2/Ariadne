<p align="center">
  <img src="assets/Ariadne.png" alt="Ariadne logo" width="220">
</p>

# Ariadne

A source-code knowledge base for LLM agents. Ariadne generates, indexes, and serves documentation about your codebase — code explanations, structural catalogs, cross-cutting themes — so any MCP-enabled agent (Claude Code, custom agents, anything speaking the Model Context Protocol) can answer questions about your code without rediscovering it.

> **License: [PolyForm Noncommercial 1.0.0](LICENSE).** Free to use, modify, and redistribute for any noncommercial purpose — personal projects, academic research, teaching, and use by charitable, educational, public research, public safety/health, environmental, or governmental organizations. For commercial use, contact Aviv Avital at ubthor@gmail.com for a commercial license.

## Why Ariadne?

When an LLM agent works with a codebase, it typically greps and reads files to understand the code. This works, but:
- It's slow and uses lots of context
- The agent rediscovers the same patterns each session
- Complex systems need explanation, not just code
- Cross-cutting concerns (auth, retries, error handling) span many files and aren't visible from any single one

Ariadne solves this by:
- **Generating documentation once** with an LLM (per-file `explanation`, `architecture`, `qa`, `gotcha`, `diagram`)
- **Building a structural catalog** of every class/method/function via ast-grep (multi-language) and SCIP (Scala/Java)
- **Discovering themes** — clusters of related code elements that share a cross-cutting concern, summarized as topic docs
- **Exposing everything via MCP** — agents query it on demand, scoped to the source they're working on plus its declared dependencies

## How it works

Ariadne walks the source tree, extracts a structural catalog of classes, functions, and signatures across all supported languages, then uses an LLM — Claude or OpenAI — to produce six per-file documents (`explanation`, `architecture`, `qa`, `gotcha`, `diagram`, `catalog`). On top of those it builds a hybrid graph from imports, call sites, and embeddings, over which Leiden community detection discovers cross-cutting themes — auth, caching, logging, anything that spans many files — and the LLM summarizes each one as its own theme document. The whole library lives in SQL and is exposed through an MCP server, so any agent can search, follow cross-references, and ask narrative questions.

To stay current, Ariadne hooks into git: a `notify-changed` call or a post-merge sync re-extracts and re-documents only the files whose content actually changed, so a small diff costs only the touched files in LLM work, not a full re-index. That's the key cost advantage over GraphRAG, which typically rebuilds entity graphs and community summaries wholesale on each refresh; Ariadne reuses everything stable and caches static prompt prefixes to pay the LLM only on real deltas.

The net effect: when an agent opens a file, it already knows what the file does, what depends on it, and which theme it belongs to.

## Installation

```bash
# Clone the repository
git clone https://github.com/AvivAvital2/ariadne.git
cd ariadne

# Install with uv
uv sync
```

## Prerequisites

- Python 3.12+
- **API keys.** Ariadne uses keys for two separate jobs:
  - **Embeddings — always OpenAI.** Every run embeds docs for semantic search via OpenAI (`text-embedding-3-large`), so `OPENAI_API_KEY` is **always** required, whichever generation provider you pick.
  - **Generation — Anthropic *or* OpenAI.** Chosen by `provider:` in `ariadne.yaml`, or inferred from the model name (`claude-*` → anthropic, `gpt-*` → openai).

  So generating with **Claude needs both keys** (`ANTHROPIC_API_KEY` for generation + `OPENAI_API_KEY` for embeddings); generating with **OpenAI needs only `OPENAI_API_KEY`** (one key covers both). Place them as environment variables, or in a `.env` file in the Ariadne directory — Ariadne loads it automatically (`python-dotenv`):

  ```bash
  # .env in the Ariadne directory — or `export` these in your shell
  OPENAI_API_KEY=sk-...           # always required (embeddings)
  ANTHROPIC_API_KEY=sk-ant-...    # only when generating with Claude (e.g. claude-opus-4-8)
  ```

### Supported Languages

Generation and catalog extraction work across:

| Language | Catalog | Explanation | Architecture | QA | Gotcha | Diagram |
|---|---|---|---|---|---|---|
| Python | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| JavaScript / TypeScript | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Scala (via SCIP) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Java (via SCIP) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HTML | ✅ | ✅ | ✅ | — | — | — |
| JSON / YAML / Markdown | ✅ | ✅ | — | — | — | — |

Scala/Java sources require a SCIP index (built once via `scip-java` / `scip-scala`); other languages use ast-grep directly.

### `scip merge` requirement for multi-language sources

`ariadne index` runs each language's indexer separately and then calls `scip merge` to fold the per-language `.scip` files into one `<source>/.ariadne/index.scip`. **Single-language sources skip merge entirely**, so the requirement only bites when a source has more than one configured `index_kind` (Python + TypeScript, Scala + Python, etc.).

The community `scip` CLI ([scip-code/scip](https://github.com/scip-code/scip)) does not yet ship `merge` in its released binaries — see [PR #420](https://github.com/scip-code/scip/pull/420) for the upstream work in flight. Until that lands in a release, multi-language indexing requires building `scip` from that PR (or any branch that includes the `merge` subcommand) and putting the resulting binary on `PATH`. Single-language indexing works with any `scip` build.

## ty Type Checker

Ariadne uses [ty](https://github.com/astral-sh/ty) for type checking. Install the Claude Code LSP plugin for real-time type intelligence:

```
/plugin marketplace add astral-sh/claude-code-plugins
/plugin install astral@astral-sh
```

Run type checks: `uvx ty check`

## Quick Start

### 1. Configure your source

Add a source from the CLI — this creates `ariadne.yaml` if it doesn't exist and sets the first source as `default_source`:

```bash
uv run ariadne source add myproject --path /path/to/myproject/src
```

Useful options: `--depends-on a,b`, `--parent X`, `--exclude '**/*.min.js'`, `--exclude-dirs build,dist`, `--branches 'feature/*'`, `--ref main`. `source add` is idempotent — re-running with only the flags you want to change leaves the rest intact. Inspect with `ariadne source list`; drop one with `ariadne source remove <name>`.

You can also hand-edit `ariadne.yaml` (in the Ariadne directory where you cloned it):

```yaml
default_source: myproject

sources:
  myproject: /path/to/myproject/src

docs_base: ./docs
```

> **Note:** `source add`/`remove` (and `discover`/`onboard`, which auto-write the managed `index_kinds`/`scip` blocks) rewrite `ariadne.yaml` via PyYAML, which does not preserve comments or custom formatting. Keep notes elsewhere, or treat the CLI as the source of truth.

### 2. Onboard end-to-end (recommended), or run phases individually

`uv run ariadne onboard` runs the whole pipeline for a source in one go — discover → index → catalog-sync → catalog-describe → generate → themes. It runs the **free phases and shows a cost preview first**, then **prompts to continue** into the paid phases; the free phases are *not* re-run when you proceed (no re-indexing). Pass `--approve` to skip the prompt (for non-interactive/CI), and `--live`/`--batch` to skip the LLM-mode prompt (`--batch` uses Anthropic's Message Batches API, ~50% off, up to 24h SLA). To see only the cost estimate and stop, use `uv run ariadne dry-run`.

For finer control, run the phases individually:

```bash
# Generate docs from source code using LLM
uv run ariadne generate

# Export to markdown files
uv run ariadne export

# Verify it worked
uv run ariadne list
```

### 3. Integrate with Claude Code

Use the automated setup:

```bash
cd /path/to/your-project
uv run --directory /path/to/ariadne ariadne init --source myproject
```

This creates:
- `.claude/settings.json` with session start hook
- `CLAUDE.md` with instructions for Claude to check docs first

**Verify integration:**

1. Start a new Claude Code session in your target project
2. You should see "Ariadne Knowledge Base Active" message
3. Ask Claude about your codebase — it should reference the docs

See [docs/new-project-onboarding.md](docs/new-project-onboarding.md) for the full end-to-end guide, or [docs/claude-code-integration.md](docs/claude-code-integration.md) for advanced options.

**MCP Server (Recommended)**

The MCP server enables on-demand documentation search during sessions. Use it alongside session hooks — hooks provide immediate context at startup, MCP enables deeper exploration when needed.

```bash
# Register MCP globally (one-time setup, works across all projects)
ariadne init --source myproject --target /path/to/your-project --global
```

Or register MCP manually: `claude mcp add -s user ariadne -- uv run --directory /path/to/ariadne ariadne mcp`

### MCP Tools (for agents)

The MCP server exposes 40+ tools (some multi-action) to LLM agents.

**Pick the right tool:**

| Question type | Tool |
|---|---|
| "How does X work?" / "What pattern is used for Y?" | `ariadne_search` (conceptual, semantic) |
| "What is this specific class/function?" | `ariadne_symbol` (qualified-name lookup) |
| "What does this whole file do?" | `ariadne_explain` (per-file explanation) |
| "Find code related to this concept" | `ariadne_themes` (cross-cutting clusters) |
| "What docs exist for X?" | `ariadne_list_all` |
| "What changed in this diff?" | `ariadne_diff_explain` |
| "What might break if I change X?" | `ariadne_impact_radius` |
| "What tests cover this?" | `ariadne_find_tests` |
| "Why is this error happening?" | `ariadne_diagnose` |
| "Show me this file's source" | `ariadne_read` (file) or `ariadne_body` (one element) |

**Disambiguating the three doc-shape tools:**

| Tool | Use when |
|---|---|
| `ariadne_explain` | You want the canonical per-file explanation doc (generate if missing) |
| `ariadne_expand` | An existing doc is too brief and you want it elaborated on a specific topic |
| `ariadne_summarize` | An existing doc is too long and you want a tighter version |

**Full tool reference, by category:**

**Retrieval & search**
| Tool | Purpose |
|------|---------|
| `ariadne_search` | Semantic search; auto-scopes to current source + dependencies |
| `ariadne_ask` | Ask a question and get a synthesized answer |
| `ariadne_symbol` | Look up a catalog element by qualified name |
| `ariadne_read` | Read a file via Ariadne's path-aware reader |
| `ariadne_body` | Get the source body of a catalog element |
| `ariadne_source_path` | Resolve a logical source name to its filesystem path |
| `ariadne_explain` | Generate/retrieve an explanation for a file |
| `ariadne_expand` | Expand an existing doc on a specific topic |
| `ariadne_summarize` | Summarize an existing doc more tightly |
| `ariadne_list_all` | List all documents including branch-specific/experimental |
| `ariadne_themes` | List / get / show cross-cutting themes |
| `ariadne_docs_read` | Read a generated doc by ID |

**Code understanding**
| Tool | Purpose |
|------|---------|
| `ariadne_diagnose` | Diagnose a failure mode (error message → likely causes) |
| `ariadne_diff_explain` | Summarize what changed in a diff |
| `ariadne_impact_radius` | Find code affected by a proposed change |
| `ariadne_graph` | Query the doc relationship graph |
| `ariadne_find_tests` | Locate tests covering a given file/symbol |
| `ariadne_test_suggestions` | Suggest tests to add for uncovered behavior |
| `ariadne_refactor_plan` | Plan a refactor across the codebase |
| `ariadne_review` / `ariadne_review_checklist` | Pre-merge review of changed files |
| `ariadne_debug_context` | Assemble debugging context for an issue |
| `ariadne_analyze_issue` | Walk through an issue with code context |

**Branch & sync state**
| Tool | Purpose |
|------|---------|
| `ariadne_branch_status` | Which docs are affected by current-branch changes |
| `ariadne_branch_sync` | Regenerate stale docs for the current branch |
| `ariadne_sync_status` | Show staleness state across the source |

**Telemetry & feedback**
| Tool | Purpose |
|------|---------|
| `ariadne_log_hit` / `ariadne_log_miss` | Report whether a result was useful |
| `ariadne_usage_stats` | Last-N-day usage statistics |
| `ariadne_gaps` | Documentation gaps with optional LLM analysis |
| `ariadne_coverage` | Per-file documentation coverage report |
| `ariadne_project_stats` | High-level project health snapshot |
| `ariadne_document_usage` | Per-document read-frequency stats |
| `ariadne_self_improve` | Suggest improvements for Ariadne itself |
| `ariadne_list_issues` | List known issues / open questions |

**Generation & contribution** (write paths)
| Tool | Purpose |
|------|---------|
| `ariadne_generate` / `ariadne_generate_docs` | Generate docs for a specific file |
| `ariadne_improve` | Run the improvement cycle on a target |
| `ariadne_contribute` | Save a finding back to the knowledge base |
| `ariadne_notify_changed` | Trigger incremental catalog refresh for changed files |
| `ariadne_merge` | Post-merge doc reconciliation |

Retrieval tools are read-only. Tracking tools record feedback. Administrative actions (full generation, cleanup, migration) require the CLI.

### Role-aware responses (optional)

`ariadne_search` and `ariadne_ask` accept an optional `role` parameter (default `'developer'`). The default behavior is unchanged — developer-level docs are what `ariadne generate` produces and what queries return today.

When the calling LLM reads phrasing like "from a product perspective" / "for the stakeholder review" / "in plain English" from the user's natural-language query, it can pass `role='product_manager'`. Ariadne then:

1. Looks for a cached audience-adapted response for that exact (audience, question) pair. Cache hit → return directly, no LLM call.
2. Cache miss → calls an LLM to translate the developer-level docs into a PM-friendly response (no code, focus on user-facing behavior and business impact), persists it as a new `audience_response` document, returns it.
3. Subsequent identical PM questions hit the cache.

The "explain further" property: dev-level content is always retrievable via a separate query (default `role='developer'`), so a PM who wants to dig into the technical detail can follow up without re-running anything.

Cache invalidation is lazy — at the next cache-lookup, if any parent dev doc has been updated since the audience_response was created, the stale row is dropped and the adapter re-runs. Dev doc updates never block on this; the freshness cost lives entirely on the PM-query path.

Role taxonomy is currently `developer` (default) + `product_manager`. Extension to other audiences is a config change, not a code change — see `designs/role-aware-responses.md`.

> **Tip:** Search returns full document content, which may trigger Claude Code's "Large MCP response" warning. Set `MAX_MCP_OUTPUT_TOKENS=50000` in your `.claude/settings.local.json` `env` section to suppress it. See [docs/claude-code-integration.md](docs/claude-code-integration.md) for details.

See [docs/claude-code-integration.md](docs/claude-code-integration.md) for setup details.

## Commands

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
| `ariadne usage` | Show MCP usage statistics and feedback |
| `ariadne gaps` | Generate miss report; `--analyze` for LLM recommendations |
| **Maintenance** | |
| `ariadne check` | Check for stale or missing documentation |
| `ariadne sync` | Sync docs with git changes since last sync |
| `ariadne export` | Export database to markdown files |
| `ariadne import` | Import markdown files into database |
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

## SCIP Cross-Source Intelligence

When sources have SCIP indexes published by their build tooling, Ariadne joins them into a cross-source code graph and unlocks:

- **Compiler-precise call hierarchy** — `ariadne callers/callees` walks resolved bindings, not heuristic name matches
- **Cross-source callers in every architecture doc** — at the end of `ariadne index` the graph is materialized into `library_scip` tables. `ariadne generate` then looks each element up and renders external callers into the architecture prompt's `Dependents` section, replacing the legacy "(Analysis not performed)" placeholder.
- **Reverse-augmented docs** — when a hub source like scalaproject is regenerated, its docs auto-include consumer-context describing how indexed downstream sources use scalaproject's symbols
- **Impact analysis** — `ariadne impact_radius` reports the file set affected by a proposed change
- **Dead-code surfacing** — `ariadne improve --dead-code` lists symbols nothing references in any indexed source

Cadence: `ariadne index` is the expensive step (runs scip-X, materializes + persists the graph). `ariadne sync` reuses the already-persisted graph and only regenerates docs whose files changed — fast, frequent. Re-run `ariadne index` when source has changed meaningfully enough to warrant a fresh cross-source layer.

Cross-source SCIP edges are SCIP-only (no fallback heuristics). A source either has a current `.scip` and participates fully, or doesn't and contributes nothing — predictable, never partial.

### Setup workflow

You author only the source's basic facts — `path`, `depends_on`, `exclude`, `exclude_dirs`. Ariadne fills in the SCIP-related fields:

```yaml
# What the user authors:
sources:
  scalaproject:
    path: /path/to/scalaproject
    depends_on: [shared]
```

```bash
# 1. discover walks the tree, writes manifest, AND fills index_kinds + scip block in ariadne.yaml
uv run ariadne discover --source scalaproject

# 2. index runs scip-X for each detected language AND populates library_scip tables
uv run ariadne index --source scalaproject
```

After `discover`, `ariadne.yaml` looks like:

```yaml
sources:
  scalaproject:
    path: /path/to/scalaproject
    depends_on: [shared]
    # Ariadne-managed below this line:
    index_kinds:
      java: scip
      javascript: scip
      scala: scip
    scip:
      artifact_path: /path/to/scalaproject/.ariadne/index.scip
      max_staleness_days: 7
```

Re-running `discover` is idempotent — the YAML mtime stays untouched when nothing detected has changed. If you remove an auto-managed entry by hand, `discover` re-adds it on the next run.

`ariadne discover` walks marker files (`__init__.py` for Python, `package.json` for JS/TS, `build.sbt`/`pom.xml`/`build.gradle*` for JVM) and writes `<source>/.ariadne/manifest.json` with the indexer plan. It also adds `.ariadne/` to the source's `.gitignore`. The artifact lives per-machine — never committed.

`ariadne index` reads the manifest, invokes the appropriate SCIP indexer in each declared cwd (multiple indexers per source supported — polyglot repos can have Scala + JS + Python side by side), merges intermediates via `scip merge` into `<source>/.ariadne/index.scip`, then runs ten `persist_*` steps that fill `library_scip` tables (cross-source graph, API endpoints from Swagger + Akka HTTP + Flask/FastAPI + Express, HTTP client calls in Python / JS / Scala, plus a final URL→endpoint resolver). Every consumer of those tables (`ariadne callers`, `impact_radius`, `improve --dead-code`, the architecture-doc Dependents section, `ariadne_trace_flow`'s cross-language hops) gets fresh data at the end of every successful index.

`ariadne sync` runs frequently (per commit / per pull) and reuses the persisted graph — no scip-X invocation. If sync detects a changed file in a language not yet declared in `index_kinds`, it triggers `discover --config-only` automatically to keep the YAML current and prints a hint to re-run `index` for the new language.

### Per-language indexer prerequisites

Install the indexer binaries once, system-wide:

```bash
# Python
npm install -g @sourcegraph/scip-python

# TypeScript / JavaScript (also handles JS via tsc's allowJs)
npm install -g @sourcegraph/scip-typescript

# Scala / Java (via Coursier)
brew install coursier/formulas/coursier
cs install scip-java

# Merge tool (always required)
# Download from https://github.com/sourcegraph/scip/releases
```

For Python projects with non-standard environments (conda, pipenv with hashed venv names), declare the interpreter in `ariadne.yaml`:

```yaml
sources:
  scalaproject:
    path: /path/to/scalaproject
    env_hints:
      python_path: /path/to/conda/envs/scalaproject/bin/python
```

### Vue-aware TypeScript indexing

`scip-typescript` doesn't natively parse `.vue` Single File Components. Ariadne ships a Vue extractor (`scripts/scip/extract-vue-scripts.js`) that pulls `<script>` blocks into `*.vue.script.{js,ts}` companions before indexing. The line-mapping is preserved so SCIP positions resolve back to the original `.vue` files when consumed.

Trigger automatically: if `ariadne index` detects any `.vue` files in cwd, the extractor runs first.

The extractor **borrows `@vue/compiler-sfc` from the indexed project's own `node_modules`** (resolved from the scope's cwd) so it matches the project's Vue version — both Vue 2.7's `parse({source, filename}) → descriptor` and Vue 3's `parse(source, {filename}) → {descriptor, errors}` shapes are handled. This means the project's dependencies must be **installed**: if `node_modules` is empty you'll see `Vue extractor: @vue/compiler-sfc not found` (and scip-typescript can't resolve types either) — run `npm ci`/`npm install` in that scope first.

`discover` does **not** create a TypeScript scope for a directory whose only JS is vendored/minified (`*.min.js`), so bundled third-party libraries (jQuery, Highcharts, …) don't become spurious scopes that scip-typescript would reject with "no files got indexed".

### Querying the cross-source graph

```bash
# Who calls this method?
uv run ariadne callers LicenseService.validate_token

# What does this method call?
uv run ariadne callees auth.login

# What files would be affected by a change to this symbol?
uv run ariadne impact_radius LicenseService.validate_token --depth 3

# Surface dead code in the source
uv run ariadne improve --dead-code --source scalaproject
```

Symbol resolution is permissive: type a qualified name, a tail (e.g., `validate_token`), or a substring. If multiple candidates match, Ariadne returns a disambiguation list and exits non-zero so you can re-run with a more specific input.

### Reverse-augment phase

`ariadne generate --source scalaproject` automatically runs reverse-augment after the main generation phase: for each file in scalaproject whose symbols are consumed by other indexed sources, the docs get regenerated with consumer-context injected into the LLM prompt. The consumer context lists exact call sites — scalaproject's `LicenseService.validate_token` doc will explicitly mention "called from `biggerproject.SessionManager.refresh()` (SessionManager.scala:128)".

This requires that the consuming sources also have current `.scip` indexes. Run `ariadne index --source <consumer>` for each.

### Swagger / OpenAPI ingestion

Declare API specs in `ariadne.yaml`:

```yaml
sources:
  scalaproject:
    path: /path/to/scalaproject
    swagger_paths:
      - api/openapi.yaml
      - docs/legacy-swagger.json
```

Ariadne parses each spec at the end of `ariadne index`, extracts `(method, path, operationId)` triples, binds operationIds back to SCIP symbols via convention (`operationId` matches a symbol's `display_name`, with camelCase ↔ snake_case fallback), and persists each endpoint into `library_scip.api_endpoints` with `resolution_source='swagger'`. These coexist with pattern-detected endpoints from Akka HTTP / Flask-FastAPI / Express — re-running the route extractors clears only `resolution_source='pattern'` rows for the source.

### Cross-language flow tracing

When both servers (route extractors / Swagger) and clients (`fetch` / `axios` / `requests` / `httpx` / `urllib` / Akka HTTP / sttp) are indexed, the URL resolver joins them by URL pattern matching and writes the resolved edges to `library_scip.api_calls`. `ariadne_trace_flow` then walks SCIP edges + api_calls + api_endpoints to return cross-language hop chains:

> "Vue component `TokenButton.handleClick` (webapp/src/TokenButton.vue:42)
>  → calls `fetch('/api/token')` (HTTP boundary)
>  → matches POST `/api/token` in scalaproject (api/routes/Token.scala:18)
>  → which dispatches to `LicenseService.validateToken` (services/LicenseService.scala:88)"

Tracing requires every participating source to have been through `ariadne discover && ariadne index` so its `.scip` is current and persisted. Sources that aren't indexed don't appear in traces — predictable, never partial.

### Verifying the tracing stack

After `ariadne discover && ariadne index --source X`, you can probe the tables that hold each tier of the data:

```bash
# Cross-source graph (Tier 1) — symbols + edges per indexed source
sqlite3 ariadne.db "SELECT source_name, COUNT(*) FROM scip_symbols GROUP BY source_name"
sqlite3 ariadne.db "SELECT COUNT(*) FROM scip_edges"

# When was each source last indexed?
sqlite3 ariadne.db "SELECT source_name, indexed_at, substr(file_sha256, 1, 8) AS sha8 FROM scip_index_state ORDER BY indexed_at DESC"

# HTTP API surface (Tier 2) — endpoints from Swagger + route patterns
sqlite3 ariadne.db "SELECT source_name, http_method, path_template, resolution_source FROM api_endpoints"

# HTTP client calls (Tier 4) — outbound URLs per language
sqlite3 ariadne.db "SELECT source_name, sink_name, raw_url, http_method FROM http_client_calls"

# Resolved cross-language edges (Phase 8c) — clients joined to endpoints
sqlite3 ariadne.db "SELECT consumer_symbol_id, endpoint_id FROM api_calls"
```

If a tier's table is empty, the chain stops there. The most common gaps:
- `api_endpoints` empty → no `swagger_paths` declared AND no route-extractor pattern matched in your `.scip`s
- `http_client_calls` empty → no recognized HTTP-client library in your code (wrappers around `httpx`/`fetch`/etc. need a configurable extension)
- `api_calls` empty though both above have rows → URL templates don't match (e.g., framework-specific `<id>` vs `:id` vs `{id}` syntax not normalized)

Then walk a real trace:

```bash
uv run ariadne trace-flow <symbol> --depth 3
# Or via MCP from a Claude session:
# ariadne_trace_flow(start_symbol="<symbol>", depth=3)
```

### Limitations

- **`.scip` artifacts must be current.** Stale indexes raise `ScipTooStaleError`. Re-run `ariadne index --source X` after meaningful source changes (or just push a new commit and let the next `ariadne sync` print the hint).
- **Indexer environment matters.** scip-python needs the project's Python venv to resolve third-party imports. scip-typescript needs `node_modules`. scip-java needs the project's build tool (sbt/Maven/Gradle). If these aren't available where Ariadne runs, the indexer fails and `ariadne index` exits non-zero.
- **HTTP-tier resolution is pattern-based.** `axios.post('/api/login')` matches `POST /api/login` perfectly. Dynamic URLs like `axios.post(`/api/${kind}/login`)` bind with `confidence='ambiguous'`. Constants resolve via SCIP-known string definitions; non-constant interpolation doesn't.
- **Layer C (config / process invocations) not yet wired.** `string_literals` is populated as a prereq for the route extractors, but `config_values` (HOCON / dotenv) and `process_invocations` (subprocess.run / Popen edge synthesis) extractors exist with tests but aren't called by any production CLI command. Lower immediate value than the HTTP path.

### Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `ariadne sync` raises `ScipUnavailableError` | `.scip` file missing — run `ariadne index --source X` |
| `ariadne sync` raises `ScipTooStaleError` | `.scip` older than `max_staleness_days` — re-run `ariadne index` |
| `ariadne callers` says "No symbol matches" | Symbol not in graph — verify `ariadne sync` ran after `ariadne index` |
| `ariadne callers` returns disambiguation list | Multiple matches; re-run with a more specific qualified name |
| `scip-python` errors with `PathDistribution.name` | Python 3.9 vs 3.10+ compat — update scip-python or use a separate Python 3.11+ env (see `env_hints.python_path`) |
| Reverse-augment didn't fire on `ariadne generate` | Other sources don't have current `.scip` — index them first |
| `typescript adapter failed … @vue/compiler-sfc not found` | The indexed project's deps aren't installed — run `npm ci`/`npm install` in that scope (Ariadne borrows the Vue compiler from the project's own `node_modules`) |
| `typescript adapter failed … no files got indexed` | The scope has no indexable TS/JS input (e.g. a config-only or vendored dir). Vendored `*.min.js`-only dirs are auto-skipped; for others, exclude the dir or check its contents |

## Configuration

Create `ariadne.yaml` in your working directory:

```yaml
# Default source when --source is not specified
default_source: myproject

# Named source paths
sources:
  # Simple form: just a path
  myproject: /path/to/myproject/src

  # Full form: with all options
  mylib:
    path: /path/to/mylib/src
    depends_on: [myproject]   # Explicit dependencies
    parent: myproject         # For subdirectory sources
    branches: ["*"]           # Active on which branches (glob patterns)
    ref: main                 # Pin to specific branch/tag

# Base directory for exported docs (organized by source name)
docs_base: ./docs

# Behavioral directive — tells Claude to mention when Ariadne helped
mention_ariadne:
  enabled: true
  # message: "custom message..."  # Optional override

# LLM defaults — provider is inferred from the model name when omitted (gpt-* → openai, claude-* → anthropic)
defaults:
  db_path: ariadne.db
  provider: anthropic        # 'anthropic' | 'openai'
  model: claude-opus-4-8     # claude-opus-4-8 (Anthropic) or gpt-5.5 (OpenAI)
```

### Source Configuration Fields

You author the **user fields** below. The **Ariadne-managed fields** are filled in by `ariadne discover` — manual edits to those get regenerated on the next `discover` run.

**User fields:**

| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Path to the source code directory (required) |
| `depends_on` | list | Source names to load as context. Search auto-includes their docs. For Python sources, Ariadne can auto-detect these — see [Automatic dependency detection](#automatic-dependency-detection). |
| `parent` | string | Parent source name (child path must be subdirectory of parent) |
| `branches` | list | Git branch patterns where source is active (e.g., `["feature/*"]`) |
| `ref` | string | Pin dependency to specific git ref (branch/tag) |
| `exclude` | list | Glob patterns matched against `Path.match` to skip individual **files** (e.g., `"**/.env"`, `"**/credentials.json"`). For *secret-bearing* files inside otherwise-walkable directories. |
| `exclude_dirs` | list | Directory **names** added to the global exclusion policy for this source — pruned at every depth (e.g., `[legacy_archive, snapshots]`). |
| `exempt_dirs` | list | Directory **names** removed from the global exclusion policy for this source — opt-in to walking a directory the policy would normally skip (e.g., `[dist]` if your project's `dist/` legitimately holds source). |
| `swagger_paths` | list | OpenAPI / Swagger spec files to ingest into `api_endpoints` (relative to source root). |
| `env_hints` | dict | Per-source indexer hints, e.g., `python_path: /path/to/conda/envs/X/bin/python` for non-standard Python environments. |

**Ariadne-managed fields** (written by `ariadne discover`):

| Field | Type | Description |
|-------|------|-------------|
| `index_kinds` | dict | Per-language SCIP routing — `{javascript: scip, scala: scip, java: scip}` for languages whose catalog extraction should go through scip-X. Auto-derived from detected file extensions. |
| `scip` | dict | `artifact_path` (the merged `.scip` file at `<source>/.ariadne/index.scip`) and `max_staleness_days` (default 7). |

### Automatic dependency detection

You don't have to hand-write every `depends_on`. During `ariadne onboard` / `ariadne generate`, when a source has **no `depends_on` set yet** (and at least one other source exists), Ariadne scans that source's Python imports and matches them against your other configured sources. On a match it shows a **Dependency Detection** panel with the evidence (file, line, import statement) and asks whether to save the relationship to `depends_on`:

```text
┌─ Dependency Detection ──────────────────────────────┐
│ No dependency was explicitly configured between      │
│ web-app and auth-app, but Ariadne detected a         │
│ relationship based on import analysis.               │
│                                                      │
│ Evidence:                                            │
│   be/app/services/auth.py:3                          │
│     from auth_app import AuthAdmin                   │
└──────────────────────────────────────────────────────┘
Save dependency web-app -> auth-app to config? [Y/n]
```

How the scan behaves:

- **Python imports only.** It's a local, offline AST scan (`*.py` parsed with `ast`) — nothing is sent to the LLM, so it adds **no generation cost**. Other languages are **not** auto-detected; set their `depends_on` by hand. (The multi-language SCIP graph still captures cross-source relationships for callers/callees/flow tracing — see [SCIP Cross-Source Intelligence](#scip-cross-source-intelligence) — but it doesn't drive this prompt.)
- **Installed packages count.** The scan intentionally includes imports that resolve into directories excluded from *documentation* (e.g. `.venv/`, `site-packages/`), so it can catch dependencies that surface only through an installed package. When evidence comes from such a directory, the panel says so.
- **Ariadne's own source is never proposed** — a documented project doesn't depend on the documentation tool, and the name `ariadne` collides with common packages (e.g. the GraphQL library), so a source pointing at the Ariadne repo is excluded from detection.
- **Only prompts while unset** — once `depends_on` is configured for a source (whether you saved a detected one or wrote it by hand), detection stops prompting for it.

### Directory Exclusion Policy

Three knobs control which directories Ariadne walks during catalog extraction, generation, and sync:

| Knob | Where | Effect |
|------|-------|--------|
| `exclude_policy` | top-level in `ariadne.yaml` | **Replaces** the built-in default list of pruned directory names (build artifacts, vendored deps, IDE metadata, JS framework caches, JVM build dirs, etc.). |
| `exclude_dirs` | per-source under `sources:` | **Adds** directory names to the policy for that source. |
| `exempt_dirs` | per-source under `sources:` | **Removes** directory names from the policy for that source — i.e., walk a directory the policy would normally skip. |

The resolved per-source set is `(exclude_policy ∪ exclude_dirs) − exempt_dirs`. If a directory name appears in both `exclude_dirs` and `exempt_dirs` for the same source, configuration loading fails with `ConfigError` — that's contradictory.

The built-in default policy covers the obvious "never useful to index" cases: `.git`, `.venv`, `__pycache__`, `node_modules`, `target`, `build`, `dist`, `.idea`, `vendor`, plus framework-specific caches (`.next`, `.nuxt`, `.svelte-kit`, `.gradle`, `.metals`, `.bloop`, ...). See `DEFAULT_EXCLUDE_POLICY` in `config.py` for the full list.

Example covering all three knobs:

```yaml
# Replace the default list — full walk plus three project-specific skips
exclude_policy:
  - .git
  - .venv
  - node_modules
  - legacy_archive

sources:
  myapp:
    path: /path/to/myapp
    exclude_dirs:
      - snapshots         # Add: don't index test snapshot files
    exempt_dirs:
      - dist              # Remove: this project's dist/ is hand-maintained source
    exclude:
      - "**/.env*"        # Skip secret-bearing files (different from exclude_dirs)
      - "**/credentials.json"
```

`exclude` vs `exclude_dirs`: `exclude` matches **individual files** via glob (file-by-file decision, useful for secrets in otherwise-walkable directories); `exclude_dirs` prunes **whole directory subtrees** at walk time (much faster than glob filtering — Ariadne never enters the directory at all).

Set `exclude_policy: []` for a full walk with no defaults. Most users won't need to touch `exclude_policy` — the built-in defaults are conservative.

### Directory-Scoped Dependencies

For monorepos or projects with subdirectories that need their own documentation, use `parent` and `branches`:

```yaml
sources:
  pythonproject:
    path: /path/to/myproject

  benchmark:
    path: /path/to/myproject/benchmark
    parent: pythonproject              # benchmark is inside pythonproject
    depends_on: [pythonproject]        # needs pythonproject docs for context

  experimental:
    path: /path/to/myproject/experimental
    parent: pythonproject
    branches: ["feature/*"]       # only active on feature branches
    depends_on: [pythonproject]
```

Use `--auto-scope` to let Ariadne detect the appropriate source:

```bash
cd /path/to/myproject/benchmark
ariadne manifest --auto-scope    # Detects benchmark source + dependencies
```

See [docs/directory-scoping.md](docs/directory-scoping.md) for the complete guide.

## Documentation Structure

Ariadne exports docs in this structure:

```
docs/{source}/
├── manifest.yaml          # Index of all documents with metadata
├── README.md              # Overview
├── explanations/          # How things work
│   ├── core-concepts.md
│   └── feature-system.md
└── architecture/          # Design decisions
    ├── design-patterns.md
    └── caching-layer.md
```

### Document Types

| Type | Purpose |
|------|---------|
| `explanation` | How a system or feature works |
| `architecture` | Design decisions and component relationships |
| `qa` | Common questions and answers |
| `diagram` | Visual diagrams (Graphviz DOT) |
| `catalog` | Inventory of modules, classes, or endpoints |
| `finding` | Quick insights and conclusions captured during sessions |
| `gotcha` | Structured pitfall docs with trigger/fix/category and encounter counting |

## Workflow

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

## How It Works

1. **Discovery**: Walks the source tree, honoring `exclude` / `exclude_dirs` and per-language type filters (e.g., JSON files only get `explanation`, never `architecture`).

2. **Catalog extraction**: ast-grep (or SCIP for Scala/Java) extracts every public element. One catalog doc per element, plus a `file_index` doc per file linking them.

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

- **ast-grep** for Python, JavaScript, TypeScript, HTML, JSON, YAML, Markdown
- **SCIP** (`scip-scala` / `scip-java`) for Scala and Java — uses precise compiler-derived semantic data, not heuristics

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

## Slack Bridge

`ariadne-slack` is an optional **read-only Slack bot** — Slack → Claude (Agent SDK) → Ariadne's MCP tools. A user @mentions it, DMs it, or runs `/ariadne`, and Claude answers from the knowledge base. It runs in Socket Mode (an outbound WebSocket; no public URL).

See **[docs/slack-bridge-deployment.md](docs/slack-bridge-deployment.md)** for the full production runbook: the Slack app manifest, the token/credential model, the serve/build split (ship a prebuilt `ariadne.db` — no `.scip`, staleness DB, or re-indexing on the box), the minimal serving `ariadne.yaml`, and the systemd setup.

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for any noncommercial purpose. Commercial use requires a separate written license from the copyright holder; contact ubthor@gmail.com.
