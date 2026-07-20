<p align="center">
  <img src="assets/Ariadne.png" alt="Ariadne logo" width="220">
</p>

# Ariadne

A source-code knowledge base for LLM agents. Ariadne generates, indexes, and serves documentation about your codebase — code explanations, structural catalogs, cross-cutting themes — so any MCP-enabled agent (Claude Code, custom agents, anything speaking the Model Context Protocol) can answer questions about your code without rediscovering it.

> **License: [Apache 2.0](LICENSE).** Free to use, modify, and redistribute — including commercially — under the terms of the Apache License 2.0.

## Why Ariadne?

When an LLM agent works with a codebase it greps and reads files to understand it — slowly, burning context, rediscovering the same patterns every session, and never seeing the cross-cutting concerns (auth, retries, error handling) that span many files. Ariadne documents your codebase **once**, with an LLM, into a queryable knowledge base — so when an agent opens a file it already knows what the file does, what depends on it, and which theme it belongs to. Keeping that current costs only the files that actually changed.

## Features

**What your agents get**

- **Five complementary doc types per file.** An `explanation` (what the code does), an `architecture` note (how it's built and who depends on it), `qa` pairs, `gotcha`s (the traps that bite you), and a `diagram` — curated per language, so a JSON file never gets an architecture essay it can't support.
- **Automatic theme discovery.** Leiden community detection over a hybrid structural-plus-semantic graph finds clusters of code that share a concern — authentication, caching, retries — even when they're scattered across dozens of files, and writes each up as its own theme doc. This is what raw `grep` can never give an agent.
- **Compiler-precise cross-source intelligence.** For Python, JS/TS, and Scala/Java, Ariadne builds a real [SCIP](docs/scip-cross-source.md) call graph — not regex heuristics — and joins it *across repos and languages*. Ask for a symbol's `callers`/`callees`, compute the `impact_radius` of a change *before* you make it, `trace-flow` a request from an HTTP route through several services, or surface dead code with zero references anywhere.
- **Served over MCP.** The whole library lives in one queryable SQLite store exposed to any MCP agent (Claude Code, custom agents), automatically scoped to the source it's working in plus that source's dependencies — so results never bleed across unrelated codebases.
- **Portable — author once, consume anywhere.** `export` packs the whole library into a single git-committable zip (or a markdown tree with `--no-archive`); `import` rebuilds a fully searchable database from it on any machine, re-embedding locally — so the model that *authors* the docs and the one that *serves* them can differ (a local model works fine). Re-import is a **delta**: documents whose content hasn't changed are skipped, so syncing a team's knowledge base only costs the docs that actually moved. See [docs/import-export.md](docs/import-export.md).
- **Ask from Slack (optional).** A read-only Slack bot puts the knowledge base in your team's chat — @mention it, DM it, or run `/ariadne`, and Claude (via the Agent SDK) answers from Ariadne's docs. It runs in Socket Mode (an outbound WebSocket, no public URL to expose). See [docs/slack-bridge-deployment.md](docs/slack-bridge-deployment.md).

**Effortless setup**

- **Zero-config language detection.** Point `discover` at a repo and it walks the tree, identifies every language, and writes the indexer plan straight into `ariadne.yaml` for you. Add a new language later and `sync` notices it and updates the config itself — no manual wiring.
- **Automatic dependency detection.** Ariadne reads your Python imports with an offline AST scan — **no LLM, no cost** — and proposes which other sources this one depends on, so searches automatically pull in the right neighboring docs.

**Spend with your eyes open**

- **A cost evaluator before you spend a cent.** `dry-run` projects the exact LLM cost of documenting your codebase — cache and batch discounts already factored in — by running only the *free* phases, with zero API calls. No surprise bills.
- **An interactive cost explorer.** `dry-run -i` opens a full-screen, ncdu-style file browser ranked by generation cost: drill in, see the dollars and a bar on every file, and exclude vendored or generated noise with a single keystroke while the grand total **re-prices live**. Toggle which doc types to generate and watch the price move. Apply, and your excludes persist to `ariadne.yaml` for every future run.
- **Batch or live generation.** Generate live for immediate results, or pass `--batch` to route through Anthropic's Message Batches API for roughly **half the cost** when you're not in a hurry.
- **Cheap to keep fresh.** A git-aware sync re-documents only the files whose content actually changed — a small diff costs only the touched files in LLM work, not the GraphRAG-style wholesale rebuild other tools force on every change.

## How it works

Ariadne runs a one-time pipeline over your source tree, then keeps the result current as the code changes. Each stage adds a layer that agents can query:

1. **Catalog the structure.** It walks the tree and extracts a structural index of every public class, function, method, and module-level value — using **SCIP** (compiler-precise symbols and call graphs, not heuristics) for Python, JS/TS (and Vue), Scala, and Java, and **ast-grep** for HTML and the common config and documentation formats (JSON, YAML, Markdown, HOCON, CSS). This layer alone gives exact symbol lookup and cross-file relationships, and it uses no LLM, so it's cheap to build and refresh.

2. **Document each file.** For every file, an LLM (Claude or OpenAI) writes the doc types you ask for — an `explanation` of what the code does, an `architecture` note on how it's put together, `qa` pairs, `gotcha`s, and a `diagram`. Every document is validated (closed code blocks, required sections) and retried on failure, so the library stays well-formed.

3. **Connect the dots.** Ariadne builds a hybrid graph from imports, call sites, and embedding similarity, then runs Leiden community detection to find clusters of code that share a concern — authentication, retries, error handling — even when they're scattered across many files, and summarizes each cluster as its own *theme* document. It also walks that graph to inject a "Related Documents" section into every doc.

4. **Store and serve.** Everything — the catalog, the per-file docs, the themes, and any findings you save — lives in a single SQLite library with embeddings for semantic search, exposed to agents over an MCP server. Deterministic IDs make every step idempotent, so re-runs never duplicate work.

5. **Keep it fresh.** A git-aware sync re-documents only the files whose content actually changed (tracked per file by content hash plus which doc types already succeeded), so ongoing upkeep costs just the touched files instead of a full re-index.

See [docs/architecture.md](docs/architecture.md) for the internals.

## Installation

```bash
git clone https://github.com/AvivAvital2/ariadne.git
cd ariadne
uv sync
```

**Prerequisites:** Python 3.12+, and API keys for two jobs —

- **Embeddings — always OpenAI** (`text-embedding-3-large`): `OPENAI_API_KEY` is **always** required.
- **Generation — Anthropic *or* OpenAI**, chosen by `provider:` in `ariadne.yaml` (inferred from the model: `claude-*` → anthropic, `gpt-*` → openai).

So generating with Claude needs both keys; generating with OpenAI needs only `OPENAI_API_KEY`. Put them in your shell or a `.env` in the Ariadne directory (auto-loaded).

| Language | Catalog | Explanation | Architecture | QA | Gotcha | Diagram |
|---|---|---|---|---|---|---|
| Python · JS/TS · Scala · Java | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HTML | ✅ | ✅ | ✅ | — | — | — |
| JSON / YAML / Markdown | ✅ | ✅ | — | — | — | — |

Python, JS/TS, and Scala/Java are indexed with SCIP — a one-time step that just needs the matching indexer installed (`scip-python`, `scip-typescript`, `scip-java`); multi-language sources also need a `scip merge`-capable binary. `discover` wires the config up for you — see [docs/scip-cross-source.md](docs/scip-cross-source.md).

## Getting started

> **One-time setup** — register your code so the commands below have a source to point at:
> ```bash
> uv run ariadne source add myproject --path /path/to/myproject/src
> ```
> This bootstraps `ariadne.yaml` and is idempotent (re-run to change flags like `--depends-on a,b` or `--exclude-dirs build,dist`). See [docs/configuration.md](docs/configuration.md).

### 1. See what it'll cost — and trim it — interactively

```bash
uv run ariadne dry-run -i --source myproject
```

`dry-run` runs only the **free** phases and projects the LLM cost; `-i` opens a full-screen **explorer** over that estimate so you can shape it *before* spending anything:

- A tree of every directory and file, ranked by generation cost, with bars and a per-file `$`.
- **Navigate:** `↑/↓` move · `→` open a dir / `←` back · `Enter`/`Space` expand.
- **`x`** excludes (or re-keeps) the highlighted file or directory — drop expensive noise like vendored or generated dirs. On apply this writes `exclude_dirs`/`exclude` to `ariadne.yaml`, so it sticks for every future run.
- **Doc-type panel (left):** check/uncheck the doc types to generate — the whole tree **re-prices live**, so you see exactly what each type adds.
- **`t`** switches color theme (remembered), **`a`** applies & writes the excludes, **`q`** cancels.

No TTY (CI/pipes)? It prints a static, ranked per-directory cost table instead. Plain `ariadne dry-run` (no `-i`) just prints the estimate.

### 2. Onboard — generate the whole library

```bash
uv run ariadne onboard --source myproject
```

This is the one command that does everything: discover → index → catalog-sync → catalog-describe → generate → themes. It runs the **free phases and shows a cost preview first** — and offers the same interactive explorer from step 1 so you can trim excludes inline — then **prompts before it spends anything**, continuing into the paid phases without re-running the free work. So you can even skip step 1 and let `onboard` walk you through it.

Flags: `--approve` (skip the prompt, for CI), `--live`/`--batch` (skip the dispatch-mode prompt; `--batch` uses Anthropic's Message Batches API, ~50% off). Prefer driving the phases yourself? `uv run ariadne generate` → `uv run ariadne export` → `uv run ariadne list`.

### 3. Integrate with Claude Code

```bash
cd /path/to/your-project
uv run --directory /path/to/ariadne ariadne init --source myproject
```

This writes a `.claude/settings.json` session hook and a `CLAUDE.md` telling Claude to check Ariadne first, and can register the MCP server (`--global`). Full walkthrough: [docs/new-project-onboarding.md](docs/new-project-onboarding.md) · advanced options & MCP: [docs/claude-code-integration.md](docs/claude-code-integration.md).

## Commands

Common ones (full reference: [docs/commands.md](docs/commands.md)):

| Command | What it does |
|---|---|
| `ariadne source add/list/remove` | Manage sources in `ariadne.yaml` |
| `ariadne dry-run [-i]` | Estimate cost; `-i` opens the interactive explorer |
| `ariadne onboard` | Full pipeline with a cost preview + prompt |
| `ariadne generate` / `export` / `import` | Generate docs · export to a zip (or markdown tree) · rebuild the DB elsewhere (delta) |
| `ariadne search "query"` | Semantic search across the docs |
| `ariadne sync` / `check` | Re-document only the changed files (delta) · find stale docs |
| `ariadne mcp` | Start the MCP server (stdio) |

## Configuration

Minimal `ariadne.yaml`:

```yaml
default_source: myproject
sources:
  myproject: /path/to/myproject/src
docs_base: ./docs
defaults:
  provider: anthropic        # or 'openai' (inferred from the model if omitted)
  model: claude-opus-4-8
```

Full reference — source fields, dependency detection, the exclusion policy — in [docs/configuration.md](docs/configuration.md). Exported docs land under `docs/{source}/` (`manifest.yaml`, `explanations/`, `architecture/`, `findings/`, …).

## Documentation

**Using Ariadne from an LLM agent?** It's primarily an MCP server — point your agent at the Ariadne MCP tools and have it search the knowledge base before grepping. Start with [docs/claude-code-integration.md](docs/claude-code-integration.md) and the tool reference in [docs/mcp-tools.md](docs/mcp-tools.md).

| Guide | Covers |
|---|---|
| [new-project-onboarding.md](docs/new-project-onboarding.md) | End-to-end first-run walkthrough |
| [claude-code-integration.md](docs/claude-code-integration.md) | Hooks, MCP setup, branch filtering, usage feedback |
| [mcp-tools.md](docs/mcp-tools.md) | Full MCP tool catalog for agents |
| [commands.md](docs/commands.md) | Complete CLI command & flag reference |
| [configuration.md](docs/configuration.md) | `ariadne.yaml` fields, dependency detection, exclusion policy |
| [directory-scoping.md](docs/directory-scoping.md) | Subdirectory sources & directory-scoped dependencies |
| [scip-cross-source.md](docs/scip-cross-source.md) | SCIP indexing & cross-source / cross-language intelligence |
| [workflows.md](docs/workflows.md) | Keeping docs fresh, git-sync, hooks, branch docs, findings |
| [import-export.md](docs/import-export.md) | Export/import round-trip; author once, consume anywhere (incl. local LLMs) |
| [architecture.md](docs/architecture.md) | How it works, subsystems, usage tracking |
| [slack-bridge-deployment.md](docs/slack-bridge-deployment.md) | Read-only Slack → Ariadne bridge |

## License

[Apache License 2.0](LICENSE) — free to use, modify, and redistribute, including for commercial use.
