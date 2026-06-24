# Import & export — moving the knowledge base around

Ariadne keeps its documentation in a SQLite database (`ariadne.db`). `export` writes that database out to markdown files; `import` reads markdown files back into a database. They're inverses, and the markdown is the portable, git-committable copy of your knowledge base.

> The database is the source of truth. There is no separate backup command — exporting **is** the backup.

## The two commands

```bash
ariadne export [dir]         # database → markdown files
ariadne import [dir]         # markdown files → database
```

Both default to `docs/<source>/` for your configured source, so you normally run them with no arguments. Pass a directory to override (e.g. `ariadne export /tmp/kb`), and `--source NAME` to pick a non-default source.

- **`export`** writes one markdown file per document (organized into `explanations/`, `architecture/`, `findings/`, …) plus a `manifest.yaml` index.
- **`import`** reads those files back in, then by default regenerates embeddings and rebuilds the per-file catalog index.

Re-running either is safe: re-export overwrites the files, and re-import **updates** existing documents (matched by a deterministic ID) rather than creating duplicates.

## Embeddings are rebuilt, not carried

Embeddings — the vectors that power search — are **not** written to the markdown; they live only in the database. So:

- `export` drops them; the markdown is pure text.
- `import` **rebuilds** them on the importing machine, which therefore needs access to an embedding model.

By default the embedding model is OpenAI's `text-embedding-3-large` (`OPENAI_API_KEY`). To embed against a local or self-hosted model instead, point `OPENAI_BASE_URL` at any OpenAI-compatible embeddings endpoint.

Pass `--skip-embeddings` to import the text without embedding (fast, no API calls) — but search and `ask` won't work until you build them:

```bash
ariadne import --skip-embeddings   # text only; search disabled
ariadne rebuild                    # build embeddings later
```

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

So a common deployment is: author the KB once with a strong cloud model, commit it, then `import` it onto a remote or local box where a smaller or local LLM serves it.

To make consumption **fully local** (no cloud calls):

1. `import` on the local machine with `OPENAI_BASE_URL` pointed at a local OpenAI-compatible **embeddings** endpoint, so both the embedding rebuild and later query-embedding run locally. (`import` re-embeds everything fresh, so the database is self-consistent regardless of what the authoring machine used.)
2. Point your agent at the local Ariadne MCP server and have it call `ariadne_search`. A plain `search` runs **no generation LLM** — it's pure vector retrieval — so the only model it needs is the local embedder; your agent's own local LLM does all the reasoning.
3. Only if you use Ariadne's built-in `ariadne ask` (which *does* synthesize an answer) set Ariadne's `provider` / `model` in `ariadne.yaml` to your local model too. The agent-driven pattern (agent retrieves, then reasons itself) doesn't need this.

The one invariant: within a given database, query vectors and document vectors must come from the **same** embedding model. Since `import` rebuilds all embeddings locally, that's automatic.
