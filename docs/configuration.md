# Configuration

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

# Environment spools (opt-in) — installed knowledge packs join the query
# scope under the reserved source id spool:<name>. The runtime pin is
# REQUIRED: an unpinned or mismatched enable fails closed with a structured
# gap (`ariadne spools` shows it) rather than silently accepting any pack.
spools:
  databricks:
    runtime: dbr17.3-lts

# Behavioral directive — tells Claude to mention when Ariadne helped
mention_ariadne:
  enabled: true
  # message: "custom message..."  # Optional override

# LLM defaults — provider is inferred from the model name when omitted (gpt-* → openai, claude-* → anthropic)
defaults:
  db_path: ariadne.db
  provider: anthropic        # 'anthropic' | 'openai'
  model: claude-opus-4-8     # claude-opus-4-8 (Anthropic) or gpt-5.5 (OpenAI)
  spools_model: claude-sonnet-5  # optional: model for SPOOL BUILDS only (unset → inherit model)
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
| `ignore_staleness` | bool \| list | Exempt this source from staleness checks — `true` for the whole source, or a list of globs (`["vendor/**"]`) for specific files. For repos that update rarely. |
| `low_confidence_doc_languages` | list | Source languages treated as human-authored prose (default `[rst, markdown]`). Docs from these are tagged `provenance: human-doc` and rank **below** code-derived docs for the same query; `[]` opts out. |

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

- **Python imports only.** It's a local, offline AST scan (`*.py` parsed with `ast`) — nothing is sent to the LLM, so it adds **no generation cost**. Other languages are **not** auto-detected; set their `depends_on` by hand. (The multi-language SCIP graph still captures cross-source relationships for callers/callees/flow tracing — see [SCIP Cross-Source Intelligence](scip-cross-source.md) — but it doesn't drive this prompt.)
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

See [directory-scoping.md](directory-scoping.md) for the complete guide.


---

_Part of the [Ariadne documentation](../README.md)._
