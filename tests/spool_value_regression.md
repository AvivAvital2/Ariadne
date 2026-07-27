# Databricks-spool value regression set

The questions Ariadne answers **only WITH the databricks spool enabled** — the
proven value. Re-run WITH and WITHOUT the spool (toggle the `spools:` mapping in
`ariadne.yaml`, reconnect the MCP) after any change to the ask/search/anchored
ranking. A regression = a WIN question stops being answered WITH the spool, or
starts being answered WITHOUT it (i.e. the delta collapses).

Ask via `ariadne_ask(source='ao-core', question=...)`. Baseline captured
2026-07-24 on the clean, anchored, matrix-rebuilt store.

| # | Question | WITHOUT spool | WITH spool (must hold) | Spool evidence expected |
|---|----------|---------------|------------------------|-------------------------|
| R1 | How do concurrent writers to a Delta table behave — optimistic concurrency, conflict detection, retries? | cannot answer | **answered (HIGH)** | `OptimisticTransaction` / `ConflictChecker` / `ConcurrentWriteException` |
| R2 | How does Spark's shuffle affect a groupBy/aggregation used to compute per-slice statistics? | cannot answer | **answered** | `SkewedGroupByTest` / `SimpleSkewedGroupByTest` / `GroupByTest` |
| R3 | What's the reliable way to broadcast a config object to all Spark executors? | cannot answer | **answered** | `HadoopRDD.broadcastedConf` / `NewHadoopRDD.confBroadcast` / `SerializableConfiguration` |
| R4 | How does Spark serialize and ship Python objects (closures, config) to executors, and what commonly fails to pickle? | cannot answer | **answered (HIGH)** | `Serializers Module` (CloudPickle) / `SerDeUtil` / `Broadcast Gotchas` |

**Pass criteria (per question, WITH spool):** answer is not a "cannot answer",
confidence ≥ medium, and at least one cited source is a `spool:databricks` doc.
**No-harm criterion (any CONTROL question, e.g. `split_duckdb_table` hashing,
`FeatureComputer`):** answer unchanged vs WITHOUT, no spool citation.

**Baseline delta:** WITHOUT = 0/4 answered · WITH = 4/4 answered → +4. This is
the number to defend.

## Excluded (not wins — do NOT use as spool-health signals)
- MLflow `sklearn.autolog` slowdown — an **ao-core dependency**, not Databricks;
  correctly absent from the spool corpus.
- DuckDB-file placement / concurrent readers on Databricks — genuine corpus gap
  (the specific fact isn't in spark/delta/sdk).
- `split_duckdb_table` → Hive-partitioned output — premise mismatch (the function
  adds boolean marker columns, it does not emit file slices) + the fence
  discounts the (tangential) spool docs.

## Deterministic mechanism guards (pytest, no LLM)
The retrieval mechanism these depend on is guarded by
`tests/test_anchored_retrieval.py` (anchor floor / relevance gate / diversity)
and `tests/test_spool_gate.py::TestAnchoredGroundSearchIntegration` (repo not
crowded out + relevant ground surfaces, incl. the built-matrix path).
