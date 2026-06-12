# Sharing a database slice for one source

Ariadne can export a single source (e.g. an SDK) as a **standalone slice
database** — a normal `ariadne.db` containing just that source's documents,
their embeddings, chunks, and sections. Hand it to someone and they can ask
questions about that source over MCP or Slack, with no re-generation.

## Produce a slice

```bash
# Slice the source 'mylib' out of this database into mylib.db
ariadne --db ariadne.db export-db --source mylib --out mylib.db
```

- `--source` defaults to `default_source` from `ariadne.yaml` if omitted.
- `--db` is the full database to slice from (defaults to config / `ariadne.db`).
- Embeddings are copied **verbatim** — the recipient pays nothing to stand the
  slice up and can search immediately.
- `--no-embeddings` omits the vectors to shrink the file; the recipient then
  runs `ariadne rebuild` once (a few dollars of embedding API) to restore search.
- `--with-scip` also carries the source's SCIP call graph — its symbols plus the
  cross-source symbols it calls into — so `callers` / `impact_radius` work on the
  slice. Omitted by default to keep slices lean.

The command prints what it wrote:

```
Exported slice of 'mylib' -> mylib.db
  documents=468 chunks=120 sections=540 embeddings_included=True
```

## Consume a slice

**Option A — it's the only database (simplest).** A slice *is* a valid
`ariadne.db`. Point Ariadne at it and run the Slack bridge:

```bash
export OPENAI_API_KEY=sk-...        # embeds incoming questions for retrieval
ariadne --db mylib.db mcp           # or: ariadne-slack  (reads the same DB)
```

The consuming agent (Claude, via the Slack bridge / MCP) writes the answers, so
there is no server-side synthesis cost. The only recurring cost is embedding
each question (fractions of a cent).

**Option B — merge into an existing database.** If you already run Ariadne over
other sources, fold the slice in:

```bash
ariadne --db ariadne.db import-db mylib.db
```

- The merge is additive: unrelated sources are untouched, and re-importing the
  same slice is idempotent.
- `--on-conflict {replace,skip,fail}` controls what happens if a document id
  already exists (`replace` is the default — the slice wins).
- The slice is stamped with the embedding model it was built with; import is
  refused (`--embedding-model` mismatch) if your database embeds differently,
  since mixed vector spaces would corrupt ranking.

## What the slice contains (v1)

Carried: **every document Ariadne generated for the source** — all content types
(explanation, architecture, qa, gotcha, diagram, catalog, finding) — with their
chunks, sections, and embeddings, plus the source's **themes** and **"Related
Documents" graph** and its relational / sync metadata.

Optional, with `--with-scip`: the **SCIP call graph** — the source's symbols plus
the cross-source symbols they reference — so `callers` / `impact_radius` resolve on
the slice. `local N` symbol ids are namespaced by source, so merging a slice into a
populated database can't collide them onto unrelated symbols.
