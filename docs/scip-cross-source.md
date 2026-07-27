# SCIP Cross-Source Intelligence

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

# Go (needs a working Go toolchain on PATH)
go install github.com/scip-code/scip-go/cmd/scip-go@latest

# Merge tool (always required)
# Download from https://github.com/sourcegraph/scip/releases
```

`scip-go` type-checks each Go module (rooted at a `go.mod`) via `go/packages` — no build-tool orchestration, so it indexes in a single fast pass per module. Unlike `scip-java` it needs no `pom.xml`/`build.sbt`; it does need the Go toolchain (`go`) installed.

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
- **Indexer environment matters.** scip-python needs the project's Python venv to resolve third-party imports. scip-typescript needs `node_modules`. scip-java needs the project's build tool (sbt/Maven/Gradle). scip-go needs a working Go toolchain (`go`) on PATH. If these aren't available where Ariadne runs, the indexer fails and `ariadne index` exits non-zero.
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


---

_Part of the [Ariadne documentation](../README.md)._
