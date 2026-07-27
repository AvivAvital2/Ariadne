# Building a Databricks Spool

A **spool** is an opt-in *environment knowledge plugin* — a declarative, versioned pack of
prebuilt docs and embeddings for a runtime, so Ariadne can answer cross-environment
questions (e.g. *"how would my code run on Databricks?"*). This guide builds the Phase-1
Databricks spool.

Spools are **build-once, distribute-many** (design §10): you pay the clone + index +
generate cost **once** per runtime edition when you cut the pack. Everyone who then
`install`s it just copies the prebuilt docs/embeddings into their store — no cloning, no
SCIP indexing, no compilation, no LLM spend (and, on a matching embedding model, not even
a re-embed).

## Prerequisites (build machine only)

All commands run from the Ariadne repo root. Consumers who only `install` a finished pack
need none of this.

**Ariadne + git** — Ariadne installed (so `uv run ariadne …` works) and `git` with network
access to `github.com`.

**`scip-python`** (indexes pyspark + the SDK):
```bash
npm install -g @sourcegraph/scip-python     # needs Node.js >= 18
which scip-python                           # confirm it's on PATH
```

**`scip-java` + JDK 17** (indexes Spark's and Delta's Scala source; it *compiles* each
project, so the build tool must resolve — Spark → Maven, Delta → SBT — and Spark 4.0 needs
JDK 17):
```bash
# install Coursier first: https://get-coursier.io/docs/cli-installation
cs install --contrib scip-java
which scip-java                             # confirm it's on PATH
```
This compile is the slow step — but a **one-time** cost per runtime edition, amortized
across every install and query, never paid again.

**API keys** (environment variables — never written to any file):
```bash
export OPENAI_API_KEY=sk-...                # embeddings, and generation (provider defaults to openai)
# export ANTHROPIC_API_KEY=...              # only if ariadne.yaml sets `defaults: {provider: anthropic}`
```

## Steps

### 1. Create the spool (set up + build)
```bash
uv run ariadne spools create        # or name it: `create databricks`
```
One command sets up the recipe, then builds it — in order:
1. **Recipe setup (interactive)** — pick the spool (`databricks`, or pass it as above) and
   the runtime edition, then **specify each version**. The repo set comes from the
   environment automatically; you supply the pins:

   | repo | tag |
   |------|-----|
   | `spark` | `v4.0.0` |
   | `delta` | `v4.0.0` (Delta 4.0 tracks Spark 4.0.0) |
   | `databricks-sdk-py` | the `databricks-sdk` version bundled with DBR 17.3 |

   Your answers are written to `./spools.yaml`. **You specify the versions; the mechanism
   fetches them** — it never scrapes or guesses. (Re-running `create` seeds the prompts from
   the existing file, so you confirm/edit rather than start over.)
2. **Consent prompt** — lists each repo at its resolved SHA. Type `y`. The resolved SHAs
   are pinned back into `spools.yaml` (trust-on-first-use).
3. **Grounding gate** — verifies the corpus languages are SCIP-indexable. `[python, scala]`
   pass; a corpus in an unsupported language (e.g. Ruby) aborts here with a clear message
   rather than building a hollow, ungrounded pack.
4. **Fetch → source add → discover/index** — clones the repos and runs `scip-python` +
   `scip-java`.
5. **onboard** — prints a **cost preview and prompts** before any spend. Approve it; choose
   **batch** for half-price embeddings (or pre-select up front: `create --batch`).
6. Produces **`databricks-dbr17.3-lts.zip`**.

Once `spools.yaml` is pinned, `uv run ariadne spools create --yes --batch` rebuilds it with
no prompts at all (CI).

### 2. Install the pack
```bash
uv run ariadne spools install databricks-dbr17.3-lts.zip
```
Verifies the checksum, then lands the docs under the reserved `spool:databricks` source id.

### 3. Enable it
Add to `ariadne.yaml`. The `runtime:` pin is **required** — an unpinned spool
fails closed (it would otherwise silently accept any signed version):
```yaml
spools:
  databricks:
    runtime: dbr17.3-lts
```

### 3b. (Optional) Cross-source themes
Step 3 surfaces the spool's docs in your query scope. To also get **cross-source
themes** — clusters that span *your project* and the environment — cross-check a
project. Onboard the project first (so its catalog + base themes exist), then:
```bash
uv run ariadne spools enable databricks --project yourproject
uv run ariadne themes build     # summarize the new themes (shows cost, prompts)
```
`uv run ariadne spools reconcile` refreshes them after your project's code
changes; `spools disable databricks --project yourproject` removes them. Enabling
against a not-yet-installed spool is refused loudly (it names the gap), not a
silent no-op.

### 4. Verify
```bash
uv run ariadne spools      # expect: registered  databricks  (runtime dbr17.3-lts, version 1.0.0)
```
Then ask a cross-environment question via `ariadne_ask` / `ariadne_search` (or
`uv run ariadne ask "…"`). Spool docs join the query scope under `spool:databricks`, ranked
**below** your own code (the trust ladder: code first, official docs fill gaps).

## Good to know
- **Build the spool with the same embedding model your projects use.** `install`
  verifies the pack's embedding model + dimension against your config and
  **refuses** a mismatch (a pack's vectors are meaningless under a different
  model) — keep the same `OPENAI_API_KEY`/embedding model on both sides.
- **Nothing paid or destructive happens before the prompts.** Interactive `create` always
  asks (setup → consent → cost) before any fetch or spend. `--yes` is the non-interactive
  path: it skips setup + all prompts and builds an existing complete `spools.yaml`; pair it
  with `--batch`/`--live` (else the embedding-mode toggle would have nothing to answer it).
- **A wrong tag fails loud** at SHA resolution — nothing partial is left behind.
- **`--allow-ungrounded`** (default off) forces a docs-only build when the corpus language
  has no SCIP indexer. Use it only as a deliberate escape hatch — the resulting pack has no
  code-tier grounding.
- **Faster first pass (reduced fidelity):** if the scip-java/Scala compile is a hurdle, trim
  `spools.yaml` to the Python repos (pyspark + sdk) for an initial pack and add the Scala
  corpora later. That pack is "behavior-blind" (interface only, no Spark internals) — a
  smoke test, not the real thing.

See `designs/spool-environment-plugin.md` for the full design.
