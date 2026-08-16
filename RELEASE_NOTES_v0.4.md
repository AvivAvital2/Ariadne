# Ariadne v0.4

## New Features

### Compiler-grounded answers — evidence that travels all the way to the response

- Rebuilt the Ask path around compiler-derived evidence rather than an unstructured document bundle. Ariadne now turns selected SCIP relationships into a bounded evidence graph, materializes the exact definition bodies and relation sites it needs, and asks the model to explain only that graph — with `file:line` evidence for each code claim.
- Added an evidence IR and source ledger to the formulation path. The model can refer to stable nodes, transitions, and exact-source chunks; Ariadne expands them deterministically, appends selected proof the narration omitted, and rejects or filters claims the assembled evidence does not support.
- Made answer quality legible instead of collapsing it into a similarity score. `AskResponse` now carries additive machine-readable citations, chain files, selected and hydrated symbols, supported claims, evidence gaps, route-selection diagnostics, per-phase timing, and separate chain, formulation, and scope completeness/confidence signals.
- Added **clews**: pre-generated, embeddable compiler routes such as `fit → withTransformEvent → listenerBus → getOrCreate`. Clews position a question on a path before traversal; they are not themselves evidence. In the Databricks measurement, pooling local clew-generation strategies covered 92.8% of required answer-key symbols, versus 66.0% for a single live walk.
- Made selection bounded and obligation-aware. Questions are split into facets and fixed obligations; Ariadne builds compact route-family cards per obligation, preserves a reserve for each one, caps retrieval without silently dropping proof, and falls back deterministically when a selector reply is empty, truncated, or malformed.
- Added the opt-in compact answer path (`ask_pipeline: compact`): a fixed three-model-call budget that plans obligations, selects route families, expands only those families into source evidence, and fails cheaply and explicitly when it cannot assemble complete compiler evidence.
- Made wide dispatches honest and usable. Instead of quietly trimming a large implementation set, Ariadne reports the indexed implementation count and package shape, explains that the result is an index-bounded floor rather than a census, and asks the caller to narrow the route (or explicitly request the full list).

### SCIP indexing and structural traversal

- Rebuilt the compiler-index ingest around a single load → graph projection → store path. The new loader and graph builder make SCIP identity, extents, relationships, edge typing, persistence, and path resolution explicit rather than letting individual consumers rediscover them inconsistently.
- Corrected document-local SCIP identities: `local N` bindings are now namespaced per document, preventing unrelated files or sources from sharing one global symbol. The rebuild also preserves definition bodies through `enclosing_range` (and reconstructs missing Java extents), so a compiler hop can be quoted rather than pointing only at an identifier.
- Added first-class `contains` and `implements` relationships; calls are attributed to the enclosing callable rather than a same-line local binding, and non-definition occurrences are typed as calls or type references according to their target. This makes polymorphic dispatch, ownership, and source-directed traversal available without treating every reference as an invocation.
- Added the SCIP wiring gate to `ariadne index`. It reports four ingest invariants — no unscoped local IDs, no cross-source edges through locals, quotable definition extents, and retained implementation edges — and makes a structurally broken store visibly untrustworthy instead of silently reporting a successful index.
- Added a source-scoped persistence path and `ariadne index --persist-only`, so an upgraded ingest can rebuild `library_scip` from existing `.scip` artifacts without re-running indexers, re-generating docs, or spending on embeddings. The extraction/ingest coverage stamp is now v2 and prompts a refresh when the stored graph predates these semantics.
- Improved source-to-SCIP path handling and freshness reporting: unresolved paths are reported rather than guessed, stale artifacts cannot silently re-enter the graph, and installed packs backfill exact canonical ownership edges without needing a checkout or an LLM.

### Evaluation, reproducibility, and experiment discipline

- Published a compiler-aware comparison against a real bare-LLM source-reading arm on twelve DBR 17.3 questions and 25 reviewed claims. Ariadne completed 8/12 questions and 19/25 claims; the bare arm completed 2/12. The report also publishes strict evidence recall — Ariadne: 112/121 symbols, 109/122 definitions, 88/97 relation sites, 159/187 witnesses — including the bare arm's format limitation rather than presenting its zero canonical-symbol score as a capability claim.
- Shipped the comparison as a verifiable public panel: a proof manifest, sanitized recorded replay, charts, and an offline verifier. Its minimal source-root builder fetches only the 46 referenced Spark and Delta files (1.36 MB), checks every SHA-256, and requires neither a spool nor a database, embedding, model call, raw answer, or full checkout.
- Added a clean-room Claude arm that runs fresh, read-only, and isolated from Ariadne data: only a stripped raw corpus and plain questions are mounted, with MCP and network-capable tools disabled. The documentation states the remaining egress caveat explicitly.
- Added a sealed experiment framework for the chain benchmark: gold-blind runs, tamper-checked artifacts, replay, grade, first-loss classification, Pareto comparison, calibration backtests, preflight/cost reports, and certificates. A paid canary is only prepared — never run — until an explicit `--allow-paid` budget, fresh nonce, valid certificate, unchanged database fingerprint, and explicit question IDs are supplied.
- Added live-path diagnostics for answer lifecycle, request/response and token-usage capture, graph inspection, source-of-truth audits, and the ability to identify the earliest retrieval or formulation loss rather than treating a final score as an explanation.

### Slack

- Added explicit Markdown-file delivery for Slack answers. Ask Ariadne to “attach the answer as Markdown” (or use `--attach-markdown`) and it uploads `ariadne-answer.md`; ordinary answers remain inline at every length.
- Preserved clean Slack prose around a requested document: a single `markdown` fence becomes the file body while the surrounding reply becomes the thread message. If upload fails, Ariadne posts a lossless, bounded thread-chunk fallback instead of leaving a partial answer; diagram attachments still render and upload normally.

## Bugfixes

- Recovered the public Ask synthesis path for Anthropic-configured installations: the key check now follows the configured provider instead of assuming `OPENAI_API_KEY` and incorrectly downgrading a valid Anthropic setup to document-only output.
- Scoped enabled spools to the project they serve, preventing an environment pack from ranking against unrelated codebases. Existing undeclared scopes remain compatible but `ariadne spools status` now warns when a spool is globally serving every configured source.
- Kept selected compiler evidence properly scoped during route hydration: context references no longer silently materialize unrelated bodies, while exact nested execution bridges are retained only when SCIP proves the complete enclosing handoff.
- Avoided redundant question embedding and bounded vector reads to lexical and structurally positioned clew candidates, with a lexical fallback whenever embeddings are unavailable.

## Other Changes

- Added `ariadne spools embed <spool>` to fill missing installed-pack embeddings in both the pack and local store. It reports the gap and estimated cost before work, supports `--dry-run`, batch submission, and resumable batch collection, then re-stamps the pack manifest and checksum.
- Made `migrate --infer-source-name --dry-run` genuinely non-mutating and report its proposed source assignments.
- Documented the rebuilt SCIP pipeline, its invariants, diagrams, measured gaps, and reproduction commands; added public compiler-aware comparison links to the README.
