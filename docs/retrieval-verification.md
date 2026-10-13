# Hosted retrieval verification

The [frozen corpus](../tests/fixtures/model_support/retrieval-corpus-v1.json) compares hosted `Qwen/Qwen3-Embedding-8B` with OpenAI `text-embedding-3-large` at 3072 dimensions. It uses 50 source-grounded questions and the same 22 repository excerpts in separate, freshly bound libraries. No documentation model is called. Labels and corpus bytes are fixed before inference; results cannot change them.

```sh
uv run python -m tests.live_retrieval_benchmark \
  --output /path/to/new-retrieval-run --budget-usd 0.50

# Offline regression checks; no paid requests.
uv run pytest tests/test_retrieval_benchmark.py tests/test_qwen_recorded_workflows.py
```

The default runs both embedding profiles with one shared cap. `--profile qwen` or `--profile openai` selects one. Credentials come from `DEEPINFRA_API_KEY` and `OPENAI_API_KEY`, or the ignored repository `.env`. Only frozen excerpts and questions are sent. Existing evidence directories are rejected. Budget accounting, route/model restrictions, credential redaction and bounded retries reuse the shared live-model runner.

The writer, document search, chunk search and MCP section selection are production code. Corpus round trips and stored-vector correspondence are checked independently against raw provider response indices and normalization. A separate numeric oracle checks document/chunk ordering and scores for all 50 queries. Query vectors use the exact production input recipes, batched for cost and rate limits, then reused across retrieval paths. The Qwen query instruction and native-4096-to-normalized-3072 reduction remain unchanged.

The report separates implementation checks from quality measurements:

- Document and chunk Recall@5, Hit@5 and MRR@5 measure recovery of the frozen known-relevant IDs. Four short-document questions have no stored chunks; chunk metrics use 46 eligible questions.
- Section Recall@5 is a **diagnostic global ranking of heading vectors**, not an application API. It measures the unchanged section embedding recipe separately.
- MCP evidence delivery measures whether the actual document-then-section retrieval path returns the supporting quotation among its first five documents. It retains the existing threshold and full-content fallback.

Library rankings are measured before MCP records usage, preventing unanswered benchmark searches from down-weighting later questions. No quality score is silently converted to a pass by changing thresholds, labels or ordering. The comparison reports Qwen minus OpenAI against the design's proposed five-percentage-point tolerance; a completed measurement does not establish an accepted production threshold. Exit status is nonzero for an incomplete experiment or failed implementation assertion, rather than for an observed model-quality difference.

Judgments were authored from repository excerpts by the coding agent, before calls. They are not independent human assessments or exhaustive relevance labels. The corpus is small and curated. Results measure these hosted embedding endpoints and existing retrieval recipes; they do not qualify pinned local weights, tokenization, GPU capacity, latency or every generated factual claim.

Every run saves its frozen protocol/corpus, runner snapshots, requests, raw responses, per-question rankings, MCP selections, database pair, usage and errors. Failed attempts remain evidence. The maintained runner is [live_retrieval_benchmark.py](../tests/live_retrieval_benchmark.py).

## Recorded result and fix

The 2026-10-11 evidence measured 96% document Recall@5 for both providers and chunk Recall@5 of 97.8% for Qwen versus 95.7% for OpenAI. It also exposed a pre-existing application defect: below-threshold section selection could return only the document title.

The fallback now skips heading-only sections while retaining the threshold and first-body-section policy. A search-cache version change prevents old responses from surviving the upgrade. A complete replay using the identical saved embeddings improved MCP evidence delivery from 78% to 96% for Qwen and 36% to 96% for OpenAI. Document/chunk/heading rankings and every stored vector remained unchanged. The two remaining misses per profile were outside the first five retrieved documents. This replay is a controlled before/after check on the discovery corpus, not an independent held-out evaluation.

The paid run cost an accounted $0.00195283. The fix, 105-test regression suite, Docker rebuild/full distribution verification and saved-vector replays required no additional model calls. Replay the durable archive without credentials:

```sh
uv run python -m tests.replay_retrieval_benchmark \
  --source /path/to/saved-retrieval-run \
  --output /path/to/new-offline-replay
```
