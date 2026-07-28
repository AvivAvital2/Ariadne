# Retrieval evals

Measures the one thing a knowledge base must get right and usually can't
prove: **is the document that holds the answer actually in the context —
and does the environment speak only at the seam?**

## Metrics

- **Participation** — for each battery row, the environment's speak/silent
  verdict vs the row's `want`. Half the battery is rows where the
  environment MUST stay silent, including *saturated controls*: questions
  about the consumer's own code phrased in vocabulary the environment also
  owns. Most retrieval evals never test silence; it's where fused context
  quietly rots.
- **Ground-truth-in-context** — for speak rows, whether a doc matching the
  row's auditable `ground_truth` title criteria ranks inside the admitted
  window. When it doesn't, the LLM fills the gap from training data, which
  cannot know your code, your pinned versions, or the seam between them.
- **Junk admissions** — environment docs admitted on a must-stay-silent row.

The verdicts come from the **production** participation criterion
(`library.lens_retrieval.breadth_speaks`) over the **production** candidate
pool — the battery measures the shipped pipeline, it does not reimplement
a scoring proxy. That also makes it a regression harness: any ranking
change that moves these numbers shows up as a diff.

## Layout

- `battery.yaml` — the question battery: a consumer-archetype matrix
  (adopter / peripheral / integrated — see `RESULTS.md`) against the
  **Databricks environment** (Spark 4.0.0 + Delta 4.0.0 + the SDK, built
  from the shipped `spool_content/recipes/databricks.yaml` at pinned
  SHAs), with per-row ground-truth criteria. A large, versioned,
  gotcha-rich environment is the point — serialization traps, concurrent
  writers, deprecations — a toy corpus has nothing for a spool to shine on.
- `fixtures/` — three small synthetic consumer repos, one per archetype
  (adopter: zero Spark calls; peripheral: one Spark ingest edge +
  name-saturated prose; integrated: Spark/Delta surfaces woven through).
- `build_eval_store.sh` — one-time store build. With a prebuilt pack zip
  (`PACK=…`, or one in the repo root): minutes, free. Without: builds the
  pack from the shipped recipe — reproducible from pinned sources, but
  scip-java compiles Spark (hours) and generation costs real money, every
  cost shown first.
- `run_battery.py` — the runner. First run embeds the questions and caches
  vectors in `query_vectors.npz`; every later run is offline and free.
  Exit code 0 iff every participation verdict is correct.

## Run

```bash
evals/build_eval_store.sh          # once (needs OPENAI_API_KEY + scip-python;
                                   #  PACK=…zip skips the from-source build)
uv run python evals/run_battery.py # repeatable, offline after first run
```
