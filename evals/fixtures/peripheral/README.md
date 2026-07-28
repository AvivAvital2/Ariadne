# harvest

Builds regional yield reports. Raw field readings are ingested nightly by
a **Spark** job on **Databricks** (see `harvest/edge_job.py`); everything
downstream of the Spark ingest is plain Python.

- Spark does the heavy lifting at ingest: schema casting, null filtering,
  and partitioned parquet output all run on the Databricks cluster.
- The report step reads the Spark job's parquet output; Spark is not
  involved in summarizing or rendering.
- If the Spark ingest fails, that night's Databricks run is retried once
  before the region is skipped.

(This repo is a synthetic eval fixture: its prose deliberately saturates
on the environment's names — Spark, Databricks — the peripheral-archetype
trap where the consumer's own catalog resolves environment vocabulary.)
