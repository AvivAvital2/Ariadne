# Local model qualification

Date: 2026-10-11. Status: **software implementation tested; hosted workflow evidence recorded, including a Qwen factual failure; pinned local deployment qualification deferred**.

Implementation and automated verification can proceed without a running Qwen deployment. The current handoff includes the code, configuration and deployment examples, deterministic HTTP integration tests, and saved verification results. Pinned local deployment qualification is deferred because neither model is running locally. This deferral does not waive the production acceptance criteria below or count any unperformed measurement as a pass.

The implementation follows [the implementation plan](../designs/local-model-support-implementation-plan.md). The [deployment manifest](../deployments/local-v1/manifest.yaml) records real artifact, tokenizer, vLLM source and container pins verified on 2026-10-10. These metadata checks do not establish model compatibility, quality, capacity or latency. No model weights were downloaded and no GPU deployment was started in this workspace.

## Software evidence

Environment: macOS x86_64, CPython 3.14.2, dependencies installed with `uv sync --group dev`. The original revision was `fad4b84faf09d4074ff95e7c1b99bc9b960ea7d2`. The plan's 17-file hosted baseline passed **140 tests in 9.61 seconds** before implementation; its command is preserved in baseline.json.

The local fixtures use separate loopback HTTP servers for chat and embeddings. They exercise production CLI/MCP dispatch, request construction, tokenization, response parsing and SQLite application. A transport trap rejects public inference hosts and provider batch/file routes. This is application-level routing evidence, not an operating-system firewall test against real serving processes.

Coverage includes:

- Immutable pair binding, sidecar ownership, pure previews, explicit credentials, model and vector contract failures, and busy writer rejection.
- All five generated document types in live and durable modes, language filtering, source/scope staleness, atomic document/chunk/section replacement and same-model rebuild.
- Catalog additions/removals/descriptions, coherent and incoherent themes, scoped query/section retrieval, audience response caching, auxiliary trace/gap calls, and generated export/import provenance.
- Onboarding and fixed phase checkpoints, incremental sync success/failure, persisted retry exhaustion, cancellation and terminal cleanup.
- Process termination at request/response/embedding/commit boundaries, lock release after death, saved-response reuse and idempotent replay.
- Legacy hosted regression anchors and newly bound OpenAI/Anthropic unary/batch/embedding transport contracts. The latter use deterministic HTTP transports; no paid hosted API calls were made.

The final command results, baseline differences and test output are recorded in software-checks.json and software-tests.txt. `docker compose -f deployments/local-v1/compose.yaml config --quiet` passed; this validates configuration syntax, not a running deployment.

The full suite finished with **2,062 passed, 6 skipped and 2 baseline failures in 85.63 seconds**. The focused design-audit run passed **28 tests in 13.98 seconds**. That audit exposed missing source forwarding in the issue-analysis and design-review MCP paths; both now pass explicit source scope to retrieval, and the review cache includes it. Those tests call the public MCP functions and the real local HTTP transports with cloud keys both absent and present. Merge regeneration is tested through the CLI, including retention of experimental documents and the previous sync point when required generation fails.

The repository's existing `ty.toml` sets all type rules to `ignore`; a passing `ty check` therefore provides limited evidence. Repository-wide Ruff checks have pre-existing failures. New Python modules and changed-file formatting are checked separately, and the report compares repository diagnostics to the original checkout. Ruff's existing single-multiline-quote setting conflicts with its formatter; narrow `Q001` suppressions preserve formatted SQL literals without changing repository-wide rules.

Two existing `test_catalog_body.py` integration tests require a populated operator catalog containing `embedding.EMBEDDING_DIM`. That catalog is absent in this environment. Both failures reproduce in an isolated checkout of the original revision with the same execution permissions and a provisioned environment: **2 failed, 14 passed in 2.19 seconds**. The failures are `TestCmdBodyCli::test_known_symbol_returns_body` and `TestRealLibraryIntegration::test_real_library_known_symbol`. Tests do not fabricate that user catalog to make the gate pass.

## Hosted live smoke

A separate paid DeepInfra smoke used the named Qwen chat and embedding models with three synthetic Python inputs. The evidence bundle contains the input code, unedited generated documents, prompts, both attempt reports and an offline review. Its archived harness used a test-only hosted HTTP adapter with the shared request builder/parser, embedding reduction, writer and search code. The entry point now delegates to the shared runner below.

The completed run ranked the expected document first for all three document and chunk queries, validated 49 embedding vectors, and passed idempotent replay and injected transaction rollback checks. The first attempt stopped on HTTP 429; the second recovered from another 429 using a bounded retry. Both attempts together cost an estimated $0.005 based on returned usage and embedding pricing.

The generated documentation contains verified semantic mistakes, preserved and described in the evidence bundle. Structural validation and infrastructure success do not establish output quality. This smoke does not qualify local vLLM transport, `/tokenize`, spool execution, pinned artifacts, BF16 serving, GPU capacity or offline operation; the real-deployment gate below remains open.

## Shared hosted workflows

The [shared runner](../tests/live_model_smoke.py) executes the same application workflows with `--profile qwen`, `--profile claude` or `--profile openai`. It uses production prompts for all five document types, bound storage and vectors, scoped document/chunk/section retrieval, MCP ask, source-edit replacement, idempotent replay and rollback. The [run guide](live-model-verification.md) records exact model IDs, credentials, scope and spending limits.

Paid runs used Qwen3.8-27B with Qwen3-Embedding-8B, Claude Opus 5.5 with OpenAI embeddings, and GPT-6 Astra with OpenAI embeddings. The reports and generated outputs preserve both successful and failed checks. Qwen passed the infrastructure checks but generated a false claim about negative cache capacity that propagated into its MCP answer; the factual check remains failed. The OpenAI run exposed an incorrect test assumption about sectionless diagrams, corrected with a regression scenario and a separate live recheck.

These workflows also exposed and fixed a usage-statistics tuple access crash and source filtering that discarded semantic ranking. GPT-6 now uses `max_completion_tokens` through the existing shared token-limit helper. Deterministic tests exercise every profile, native protocols, spending enforcement, invalid vectors, truncated output and full-document fallback without paid calls. None of these hosted results close the local deployment gate.

A controlled source-verification experiment subsequently tested Qwen self-review and native Opus 5.5 review of identical Qwen drafts. Qwen accepted known incorrect documents. Opus detected the tested errors but admitted only one of twelve fresh drafts, leaving two of three downstream questions unanswered. Both complete workflows failed the frozen quality/coverage gates. The experiment preserves executed source oracles, raw decisions, annotation reasons and costs; no production gate or prompt change was enabled. The stronger-reviewer result is hosted hybrid evidence and does not establish a local remedy.

## Design traceability

This table maps the design's software requirements to implementation and behavior tests. The real-model acceptance criteria remain unmeasured, as detailed below. The original design and implementation plan are unchanged.

| Design requirement | Implementation | Automated evidence |
| --- | --- | --- |
| One immutable pair, independent endpoints and credentials, canonical identity, safe database/sidecar selection | `model_config.py`, `inference.py`, `library/model_binding.py`, `cli/runtime.py` | `test_model_config.py`, `test_model_binding.py`, `test_local_contract_fixtures.py` |
| Whole-operation POSIX writer exclusion; offline inspection and pure local previews | `library/locking.py`, `inference.py`, `library/__init__.py`, `cli/runtime.py` | `test_model_binding.py`, `test_local_workflows.py`, `test_local_boundaries.py`, process-death tests |
| Exact token preflight, non-thinking chat, bounded retries, no redirects/proxy inheritance/cloud fallback | `docgen/llm/local.py`, `embedding.py`, `llm.py` | `test_local_provider.py`, `test_local_embeddings.py`, exact wire fixtures and two-server CLI/MCP tests |
| Native 4096 embeddings reduced to a normalized 3072 prefix, query-only instruction, all three vector stores | `embedding.py`, `model_config.py`, `writer.py`, `library/core.py`, `library/search.py` | `test_local_embeddings.py`, `test_local_atomic_application.py`, corrupt-vector boundary tests |
| Atomic document/chunk/section and derived metadata replacement; same-model rebuild; corpus revision and cache freshness | `library/prepared.py`, `library/core.py`, `writer.py`, MCP service/search/analysis | Atomic rollback/replay/rebuild tests, external-writer cache test, role-cache tests |
| Existing content types, catalog maintenance, themes/topics, authored contributions, auxiliary synthesis | `docgen/generator.py`, catalog/theme modules, `docgen/trace_flow_llm_bridge.py`, CLI/MCP entry points | `test_local_maintenance.py`: all five types, descriptions, theme coherence, topic/contribution, ask/role/gap/trace, issue and design-review HTTP calls |
| Two spool tables, four closed job kinds, exact persisted requests, stage budgets, durable responses, atomic application markers | `docgen/spool.py`, `docgen/llm/local_batch.py` | `test_local_spool.py`, frozen payloads in `test_local_contract_fixtures.py`, retry/obsolescence/cancellation tests |
| Foreground recovery through existing batch CLI; no provider batches/uploads; interrupted work preserves committed responses | `cli/batch.py`, `docgen/batch_resolution.py`, `docgen/local_pipeline.py` | Batch lifecycle test and `test_local_spool_recovery.py`: SIGKILL at all five required boundaries, lock release and exact chat-call counts |
| Fixed phase order and one owner; source/scope/hash-aware completeness; onboarding, sync and merge | `docgen/local_pipeline.py`, `docgen/orchestrator.py`, `docgen/staleness.py`, `cli/generation.py`, `docgen/merge.py` | One-run onboarding, live/spooled type completeness, colliding-source tests, incremental sync and merge success/failure tests |
| Additive export provenance, whole-import preflight and authored-text preservation | `export.py`, `library/imports.py`, `writer.py` | Nested metadata round trip, checked-in generated/authored/incompatible/unidentified examples, rejection before any import writes even with embeddings disabled |
| Legacy hosted behavior and newly bound hosted profiles | Existing hosted transports with shared context/factory routing | Original focused baseline, full regression suite, `test_bound_hosted_models.py` for OpenAI/Anthropic unary/batch/embedding HTTP contracts |
| Honest local estimates, endpoint doctor, pinned deployment and operator instructions | `docgen/pricing.py`, CLI health/integration, YAML examples, `deployments/local-v1/` | Pure preview and doctor HTTP tests; Compose configuration check; real serving qualification remains pending |

The versioned [contract fixture](../tests/fixtures/model_support/contract-v1.json) freezes canonical JSON and expected identity hashes, exact request bodies, source/scope sidecar records, a payload and parsed result for each job kind, and the expected process-crash transcript. Its artifact hashes are deliberately synthetic test identities. The [export examples](../tests/fixtures/model_support/exports-v1/) cover both accepted provenance kinds and both required rejection cases. Production artifact pins are only in the deployment manifest.

## Fresh implementation and container verification

The blind hosted Qwen run passed 41/41 frozen implementation checks across six new source fixtures and all five document types. It validated 114 persisted vectors against their exact embedding inputs and provider responses. There were 49 HTTP attempts, including five embedding rate-limit responses recovered by bounded retries; accounted cost was $0.01992720. Model prose was not scored as an implementation defect and no production changes were needed.

The application image checks passed on linux/amd64 with the installed wheel, UID 1000 and Docker `--network none`. Across runs with dummy cloud credentials absent/present, 22 CLI invocations and two real stdio MCP sessions exercised independent loopback HTTP fixtures, including local tokenization, generation, durable jobs, retrieval, rebuild and persisted model binding. Neither fixture received authorization. This verifies application packaging and execution without external networking; it does not qualify actual model servers.

The subsequent local distribution verification pushed the unchanged image to an isolated registry and pulled it into an empty Docker daemon. Filesystem layers and runtime configuration matched the original build. The installed CLI/MCP checks passed again; an additional MCP session and completed-batch replay passed after container recreation, with both database dumps unchanged and no inference during completed-job replay. Both shipped Compose configurations worked with only configuration and synthetic source files present. No external publication or paid API calls were needed. GPU execution remains unverified.

The later uninterrupted Docker report passed every distribution, runtime, persistence and Compose stage together, including all cleanup commands. The recorded-response regression also passed 40 focused tests, including 14 new cases using unedited Qwen chat responses through local CLI/MCP and interrupted durable work. No application-code change was needed for either result.

The subsequent frozen 50-question hosted retrieval benchmark reached 96% document Recall@5 for both Qwen and OpenAI embeddings. It found a pre-existing MCP defect: section fallback could return just a Markdown title. Skipping heading-only fallback sections raised supporting-passage delivery from 78%/36% to 96%/96% on a complete replay of the same Qwen/OpenAI vectors, without changing embeddings, thresholds or labels. All 105 relevant regressions passed, and the rebuilt image passed the complete Docker distribution run. The corpus has agent-authored relevance labels and is not an independent human-reviewed production benchmark; local artifact qualification remains deferred.

A fresh Claude Opus 5.5 live recheck then passed 24/24 hosted workflow checks with OpenAI embeddings against the fixed code. It generated new documents and an MCP answer, exercised the corrected section fallback in all three leading retrieval results, and passed source replacement and rollback. Its additional accounted cost was $0.49045852; no further application changes were needed.

## Required real-deployment qualification

The release gate remains open; it does not block the software handoff. Resume this qualification when both model endpoints are available. Before production binding, select the hardware and acceptable p50/p95 latency, provision the manifest's exact BF16 artifacts, and run the [operator recipe](../deployments/local-v1/README.md) on a fresh database pair. A changed artifact, tokenizer, representation or embedding recipe requires a separate pair.

Retain the effective server command lines, artifact hashes, GPU/driver/runtime information, vLLM startup logs and readiness output. Verify that `/tokenize` renders exactly the chat/embedding input consumed by inference, that thinking is disabled, that complete 8192-token output reservations are honored, and that embedding pooling returns the expected native 4096-component vectors. Confirm the reduction recipe independently on the pinned artifacts.

Run onboarding, every generated document type, topics, themes, authored findings/contributions, search/ask/sections, rebuild, incremental sync/merge and killed/resumed work with public inference egress denied. Repeat with cloud credentials absent and present. Preserve failed examples as well as successful outputs. Process-crash tests do not qualify machine power-loss durability.

| Acceptance area | Required evidence | Measured result |
| --- | --- | --- |
| Structural generation validity | At least 95% within the allowed live/batch repair budget, reported by task and mode | Not measured |
| Source/API correctness | Reviewed examples with no known critical false claims | Not reviewed on real-model outputs |
| Retrieval | At least 50 judged repository questions; Recall@5 within five percentage points of an agreed baseline, or an agreed absolute target | Hosted 50-question corpus and OpenAI baseline recorded with agent-authored labels; independent human review and pinned local measurements remain open |
| Retrieval levels | Document, chunk and section results evaluated separately | Hosted document/chunk/heading rankings and actual MCP passage delivery measured; pinned local results remain unmeasured |
| Semantic organization | Human judgments of cluster and theme usefulness | Not reviewed |
| Performance | Cold/warm p50/p95 latency, throughput, time per accepted document and peak memory at concurrency one | Not measured |
| Egress | Firewall-enforced end-to-end execution with both cloud-key conditions | Fixture routing trap only |

Store reviewed examples, per-question relevance judgments and raw timing/memory measurements beside the chosen deployment report. Compare result quality between libraries; vector coordinates from different embedding spaces are not comparable. Promote the candidate only after these measurements satisfy the stated targets.
