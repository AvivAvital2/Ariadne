# Import & export — moving the knowledge base around

Ariadne keeps its documentation in a SQLite database (`ariadne.db`). `export` writes that database out to markdown files; `import` reads markdown files back into a database. They're inverses, and the markdown is the portable, git-committable copy of your knowledge base.

> The database is the source of truth. There is no separate backup command — exporting **is** the backup.

## The two commands

```bash
ariadne export [path]        # database → a single zip (or a markdown tree with --no-archive)
ariadne import [path]        # zip or markdown tree → database (delta: unchanged docs skipped)
```

Both default to your configured source under `docs/`, so you normally run them with no arguments. Pass a path to override (e.g. `ariadne export /tmp/kb.zip`), and `--source NAME` to pick a non-default source.

- **`export`** packs the whole library into a single source-scoped zip — `docs/<source>.zip` — by default: one markdown file per document (organized into `explanations/`, `architecture/`, `findings/`, …) plus a `manifest.yaml` index. Pass `--no-archive` to write the loose `docs/<source>/` tree instead.
- **`import`** reads a zip (or a markdown tree) back in, then by default regenerates embeddings and rebuilds the per-file catalog index. It auto-detects the input: `docs/<source>.zip` when present, else the `docs/<source>/` tree.

Re-running either is safe: re-export overwrites the artifact, and re-import is a **delta** — documents whose content, source files, and metadata are unchanged are skipped; only new or changed docs are written (matched by a deterministic ID), never duplicated.

## Embeddings are rebuilt, not carried

Embeddings — the vectors that power search — are **not** written to the markdown; they live only in the database. So:

- `export` drops them; the markdown is pure text.
- `import` **rebuilds** them on the importing machine, which therefore needs access to an embedding model.

By default the embedding model is OpenAI's `text-embedding-3-large` (`OPENAI_API_KEY`). The supported local suite uses `Qwen/Qwen3-Embedding-8B` with its fixed embedding recipe and the Qwen chat model; configure both through [supported models](supported-models.md).

Pass `--skip-embeddings` to import the text without embedding (fast, no API calls) — but search and `ask` won't work until you build them:

```bash
ariadne import --skip-embeddings   # text only; search disabled
ariadne rebuild                    # build embeddings later
```

On a large cloud import the embedding rebuild shows the projected cost and **prompts** you to choose live vs batched embeddings first. `--live` / `--batch` skip that prompt (`--batch` routes through OpenAI's Batch API for roughly **half the price** when you're not in a hurry), and `--yes` approves the cost without prompting — the combination you want in CI. Because import is a delta, only the docs that actually changed get re-embedded.

## Team / multi-machine workflow

```bash
# Machine A — has the populated database
ariadne export
git add docs/ && git commit -m "update knowledge base"

# Machine B — fresh clone, no database yet
ariadne import       # rebuilds the database + embeddings from the committed markdown
```

After `import`, machine B has a fully working, searchable knowledge base — no source tree or generation step required.

## Author with one LLM, consume with another (including local models)

A useful consequence of the round-trip: the LLM that **writes** the docs and the LLM that **uses** them are completely decoupled.

- **Authoring LLM** (`ariadne generate`) does the expensive work of writing the documentation; then you `export`. It is **never called again** at consumption time.
- **Consuming LLM** is an agent that reasons over the knowledge base — via the MCP server's `ariadne_search`, or `ariadne ask`. Because the exported docs are plain markdown and retrieval is embedding similarity (not generation), **the consuming LLM can be anything — including a locally-run model** (Ollama, LM Studio, llama.cpp, …).

An external MCP agent can use a smaller or local model to read a cloud-generated library. Moving generated documentation to Ariadne's bound local suite requires regeneration with that pair.

To make consumption **fully local** (no cloud calls), configure the fixed Qwen chat/embedding pair and fresh library paths as described in [local models](local-models.md). Query vectors and document vectors must use the same bound embedding recipe.

Bound exports carry model provenance. Importing generated documentation into a different model pair is rejected; changing providers requires regenerating it in a fresh library. Authored text can be added through explicit authored ingestion under the target pair. An external MCP agent can still use its own model independently; Ariadne's built-in `ask` uses the library's configured chat model.
