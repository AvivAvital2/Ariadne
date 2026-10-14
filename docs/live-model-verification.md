# Live model verification

One runner exercises three hosted model pairs. No local model server or weights are needed.

For application configuration, cloud/local switching and the distinction between hosted verification and local deployment support, see [supported models](supported-models.md).

| Profile | Documentation | Embeddings | Credential variables |
| --- | --- | --- | --- |
| `qwen` | DeepInfra `Qwen/Qwen3.8-27B` | DeepInfra `Qwen/Qwen3-Embedding-8B` | `DEEPINFRA_API_KEY` |
| `claude` | Anthropic `claude-opus-5-5` | OpenAI `text-embedding-3-large`, 3072 dimensions | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` |
| `openai` | OpenAI `gpt-6-astra` | OpenAI `text-embedding-3-large`, 3072 dimensions | `OPENAI_API_KEY` |

The runner loads selected credentials from the environment or ignored repository `.env`. It sends only synthetic source fixtures and content generated from them. Each profile gets a fresh library and staleness database; the operator's library is never opened.

## Run

```sh
# Same scenarios; change only the profile.
uv run python -m tests.live_model_smoke --profile qwen --budget-usd 1
uv run python -m tests.live_model_smoke --profile claude --budget-usd 3
uv run python -m tests.live_model_smoke --profile openai --budget-usd 5

# One spending cap shared by all three profiles, executed sequentially.
uv run python -m tests.live_model_smoke --profile all --budget-usd 9
```

These commands make paid requests. Defaults are profile `qwen` and a $1 total budget. Add `--output /path/to/new-directory` to choose the evidence destination; existing destinations are rejected. Otherwise, a temporary directory is created and printed. The previous `python -m tests.live_qwen_smoke` entry point delegates to this runner.

Any failed assertion, provider error or exhausted budget produces a nonzero exit status. Failed profiles retain their evidence; other selected profiles still run within the remaining shared budget. Reruns need a new directory and a budget accounting for earlier spending.

## Shared scenarios

1. Analyze three Python files and generate explanations using production `DocGenerator` prompts. Generate architecture, Q&A, gotcha and diagram documents for the cache fixture: seven initial documents across all five types, using the normal 8192-token generation allowance.
2. Run `ContentValidator`, write through the bound `LibraryWriter`, and validate stored document, chunk and section vectors. Short documents can legitimately have no chunks or sections.
3. Replay an identical write, asserting unchanged content, timestamps, child vectors and inference request count.
4. Ask the same three retrieval questions through document/chunk search and MCP section selection; assert the expected source ranks first. When the selected document has no stored sections (for example, a short diagram), verify the full-document fallback.
5. Use MCP ask for a known source fact: negative cache capacity eventually raises `KeyError`. Save the answer and references, checking the exception and a cache-document reference.
6. Change the average function's empty-input behavior and regenerate through the same generator/writer. Check that the source snapshot, content and document vector change.
7. Inject a SQLite error during section insertion. Check that rollback preserves document content, timestamp and every child content/vector byte.

All profiles use the real provider factories, generator, validators, writer, database and MCP service. An HTTP wrapper records requests and enforces endpoint/model/spending limits. For hosted Qwen, it adds `reasoning_effort=none`; a small embedding adapter reuses the production query instruction and native-4096-to-normalized-3072 reduction. OpenAI and Anthropic retain their production request formats and OpenAI embedding client.

The source-edit scenario exercises generation and writing directly, not CLI sync/staleness orchestration. This runner does not qualify local `/tokenize`, vLLM transport, local durable spool, pinned weights, GPU capacity, offline egress, provider batches or full CLI onboarding. Existing deterministic local tests cover their software paths; hosted results do not qualify a pinned local deployment.

## Clew workflow

Run the focused clew scenario with the same Qwen, Claude or OpenAI profiles:

```sh
uv run python -m tests.live_model_smoke --profile qwen --scenario clews \
  --budget-usd 0.25 --output /path/to/new-clew-run
```

It builds payment and password-reset routes from a small source-backed synthetic compiler graph, generates a theme question and embeds routes through the configured model pair, checks question-driven invalidation, resumability and semantic ranking, and asks MCP a payment question. Checks require a selected clew, its intact route in answer evidence, source bodies in model prompts and successful completion calls. Exact prompts, replies and diagnostics are saved as `clew-completions.json` and `clew-answer.json`. This verifies the clew mechanism independently of the factual quality of generated prose; it does not qualify a real indexer or GPU deployment. Ordinary pytest runs the same scenario against HTTP fixtures for all three profiles.

The 2026-10-14 hosted Qwen run passed all eight clew checks, including generated-question invalidation and re-embedding, using `Qwen/Qwen3.8-27B` and `Qwen/Qwen3-Embedding-8B`. Its 14 requests had an accounted upper cost of about $0.003.

## Evidence

Each profile directory contains configuration with credential names only, exact source files, generated Markdown, request JSON bodies, response bodies, section results, MCP answer, the database pair and `report.json`. Request headers are not saved; credentials are redacted from response/error text. `average-before.py` preserves the source before editing; `documents/average-updated.md` contains its replacement documentation. Failure tracebacks are saved in `error.txt`. Top-level `summary.json` collects all profiles and shared spending accounting.

Structural failures, retrieval failures and the grounded-answer assertion fail the run. Validator warnings are retained. These checks do not prove every generated claim or satisfy the design's 50-question retrieval target. Review the saved outputs, including failed examples.

The 2026-10-11 evidence bundle preserves the three-profile run, the OpenAI recheck after correcting the section assertion, and the Qwen factual failure. It includes exact prompts, chat responses and generated documents; manifests identify the larger raw embedding responses and databases retained in the original local output directories.

A fresh Claude Opus 5.5 recheck after the MCP section-fallback fix passed **24/24 live checks** with OpenAI embeddings, including generation, retrieval, answering, replacement and rollback. All three leading section selections exercised the fixed fallback. The additional accounted cost was **$0.49045852**; this bundle also retains compressed embedding responses and databases.

The separate Qwen prompt investigation found inconsistent factual claims across repeated generations and counterexamples. The experimental prompt changes were not retained. Structural validity and the narrow answer assertion do not substitute for checking factual claims against source.

## Blind implementation checks

Use the separate [blind Qwen runner](../tests/live_qwen_blind.py) to check implementation behavior without using model factuality as a pass/fail condition:

```sh
uv run python -m tests.live_qwen_blind --output /path/to/new-blind-run --budget-usd 1
```

Six fresh synthetic modules and their checks are frozen before requests begin. The runner saves their hashes and protocol before inference. It makes no prompt adjustments or selective reruns after observing outputs. This is a frozen-input experiment, not an independent human-blinded study.

Its 41 checks cover all five generated document types, exact content/source round trips, replay without inference, independently calculated document/chunk ranking, source isolation with colliding filenames, section/full-content delivery, answer transport and source metadata, deterministic replacement, atomic rollback, exact embedding input/vector correspondence, close/reopen persistence and incompatible model rejection. Validator observations are saved separately; no expected prose, factual oracle or model judge determines success.

The 2026-10-11 blind run passed **41/41 checks**, validating **114 stored vectors**, over 49 HTTP attempts in 602 seconds. Five embedding rate-limit responses recovered through bounded retries. Its accounted upper cost was **$0.01992720**. No new implementation failure was observed. Hosted transport does not qualify local vLLM, tokenization, spool or GPU serving; the [installed-container checks](docker.md) separately exercise local application paths against HTTP fixtures.

## Frozen retrieval benchmark

For embedding retrieval quality, use the separate [50-question repository benchmark](retrieval-verification.md). It compares Qwen and OpenAI embeddings on one frozen corpus, with no chat-generation calls, and reports relevance measurements separately from implementation assertions. The Claude and GPT profiles share the OpenAI embedding backend; those retrieval scores do not measure either chat model's answer quality.

The recorded result found and fixed a title-only MCP section fallback. Using the same saved vectors, supporting-passage delivery improved to 96% for both providers. Unedited Qwen generation/answer responses also pass 14 additional local workflow cases, without paid requests.

## Experimental source verification

The separate [source-verification runner](../tests/live_source_verification.py) compares baseline storage with a source-review gate. It uses executed synthetic source oracles, saved failures, balanced controls, repeated held-out reviews, identical fresh drafts and actual stored-document retrieval/answers. It does not enable a production gate or alter generation prompts.

```sh
# Generate with Qwen, review with Qwen, compare baseline/gated workflows.
uv run python -m tests.live_source_verification run \
  --budget-usd 1 --output /path/to/new-qwen-study

# Review the exact same drafts with native Opus; reuse baseline answers.
uv run python -m tests.live_source_verification run \
  --reviewer claude --reuse-drafts /path/to/new-qwen-study \
  --budget-usd 1.50 --output /path/to/new-opus-study

# After source-based annotation of every draft and downstream answer:
uv run python -m tests.live_source_verification score /path/to/completed-study \
  --annotations /path/to/annotations.json
```

Each paid run has its own budget. Generation, embedding and answering remain Qwen; `--reviewer` selects only the review model. The Opus comparison is an explicit cloud hybrid and needs `DEEPINFRA_API_KEY` plus the Claude profile credentials listed above. `--reuse-drafts` requires unchanged prompt, fixtures, samples and twelve draft hashes, plus a completed baseline. No automatic repairs are attempted. Run/score exit nonzero until every qualification gate passes; a completed live run initially awaits annotations.

The recorded experiment cost about $0.88 including its incomplete first attempt. Qwen accepted six of eleven saved incorrect documents. Opus correctly classified the controls, but accepted only one of twelve fresh drafts, leaving two of three downstream questions unanswered. Both workflows failed qualification. Exact prompts, responses, source probes, annotations, runner snapshots and reproduction details are preserved.

## Spending

Every attempt, including retries, reserves a conservative cost from UTF-8 request bytes, token overhead and the entire requested output allowance. Complete successful usage replaces that reservation with an upper estimate from returned token counts. Failed requests and missing usage retain their reservation. Selected profiles share a maximum of 120 attempts and the explicit dollar cap.

Qwen embeddings are paced to accommodate provider limits. Explicit Qwen 429 responses receive at most two adapter retries. Native provider retries also pass through the spending gate. This accounting is an estimate, not an invoice; raw usage and any provider-reported cost are retained.

Standard short-context rates checked on 2026-10-11 are recorded in the runner. Reservations include possible cache-write pricing and use undiscounted Qwen rates. Update these values when provider prices change:

- [GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra): $10 input / $50 output per million tokens; cache writes $12.50.
- [Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/overview): $4 input / $20 output; five-minute cache writes $5.
- [OpenAI embeddings](https://developers.openai.com/api/docs/models/text-embedding-3-large): $0.13 per million tokens.
- [Qwen chat](https://deepinfra.com/Qwen/Qwen3.8-27B): undiscounted $0.20 input / $2.50 output; [Qwen embeddings](https://deepinfra.com/Qwen/Qwen3-Embedding-8B): $0.01 per million tokens.

## Offline tests

```sh
uv run pytest tests/test_live_model_smoke.py
uv run pytest tests/test_live_qwen_blind.py
uv run pytest tests/test_source_verification.py
```

This executes the same scenarios for all three profiles with simulated HTTP. It verifies native provider routing and request fields, malformed-vector rejection before writes, spending enforcement and credential redaction. It needs no real credentials or network access and provides no live model-quality evidence.
