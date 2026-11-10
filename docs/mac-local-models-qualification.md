# Mac local model work: evidence and open gates

Updated: 2026-10-26. Branch: `codex/mac-local-models-tdd`.

**Acceptance scope update:** the [user-managed server path](user-managed-models.md)
assigns model installation, server operation and hardware capacity to the operator.
Ariadne owns compatibility validation and correct application behavior. M5
measurements do not block that path's acceptance; no M5 capacity or latency claim
is made. The optional managed Mac launcher and its historical qualification record
below remain separate. Generation quality and live integration still require
successful evidence.

The user-managed UI implementation and its 2026-10-24 software/live checks are
recorded in the [earlier follow-up report](../evaluation/user-managed-models-2026-10-24/README.md).
The numbered evidence below preserves the earlier managed-profile checkpoints;
the new CPU checkpoint is separate from those historical results.

## Intel CPU checkpoint — 2026-10-25

The [native CPU report](../evaluation/intel-local-2026-10-25/README.md) records a
completed Q4 workflow on an Intel Core i9-9980HK with 8 physical cores and 64 GiB
RAM, using verified llama.cpp b11669 and pinned weights. The diagnostic saved a
profile through the production API used by the UI, then ran native QA generation,
embeddings, persistence, reopen, Search and Ask under that saved configuration.
All 51 recorded 3072-dimensional float32 vector fingerprints survived reopen
unchanged. Native full browser Generate/onboarding and interruption/resume were
not exercised by this diagnostic.

The workflow took **104.97 minutes**, including **84.6 minutes** for QA generation
and **17.4 minutes** for Ask. The highest sampled combined server RSS was
**36.4 GiB**; this is periodic process RSS, with possible shared-page double
counting, not total system or Apple Silicon unified-memory use. Both servers
used CPU inference and 14400-second server/client limits. These single-run
observations establish neither M5 capacity nor a controlled performance result.

All five required facts in the known five-line fixture were correctly documented.
Extra guidance still has malformed-value exception overgeneralization, input
assumptions and example defects; raw-token logging is unsafe advice when tokens
are credentials. The original Ask correctly answers equality/removal with help
from an authored dependency control. Its prompt had conflicting citation
instructions, and the answer did not supply a compiler location-citation chain.
This is bounded functional evidence, not held-out or broad factual accuracy.

The separate October 26 follow-ups corrected Search omission counts for repeated
headings and aligned documentation-Ask instructions with document-title citations.
Native Search passed in **3.08 seconds**, with ranking, scope, selected sections,
corpus and model binding unchanged; the FAQ now reports 15 omitted sections.
The Ask prompt regression selection passed **75 tests**, with **1 existing
expected failure**, while retaining compiler citation and repair behavior.
The final Python suite passed **5151 tests**, with **5 skipped, 1 expected failure
and 561 warnings**, in **409.79 seconds**, with no test modules excluded.
The earlier timeout browser selection passed **9 tests** using intercepted replies.
The post-correction native Ask completed in **1005.52 seconds (16.8 minutes)**.
Independent review found its equality/removal answer correct and its document-title
citations supported by the actual context, still with authored dependency
assistance. It made one embedding request and one chat synthesis; no compiler
citation chain is claimed. HTTP/static and real project-MCP metadata readiness
also passed, separately from browser onboarding. The original source hash
checkpoint and final outputs remain preserved. BF16/GPU serving and M5 hardware
qualification remain separate open work.

## Earlier managed-profile checkpoints

**The selected accuracy candidate, Qwen3.8-27B in thinking mode with Qwen3-Embedding-8B at 3072 dimensions, is now implemented as a separate managed Mac profile. It remains unqualified.** The previous Qwen3.5-9B profile is retained for existing libraries and has demonstrated factual errors. No M5 memory or performance qualification has been performed, and no changes have been pushed remotely.

| Ordered item | Current evidence | Completion status |
| --- | --- | --- |
| 1. UI model lifecycle and library/query flows | Both profiles covered; 115 focused backend checks and 46 selected browser checks passed | Implemented and regression-tested with fixture inference |
| 2. Durable generation cancellation/resume | 146 backend and 27 affected browser checks passed across the final runs | Implemented; retain in final acceptance |
| 3. Native compatibility, dependencies and packaged Spools | 236 compatibility checks, then 28 registration/recovery checks after the last fix | Implemented; retain in final acceptance |
| 4. Generation and retrieval quality | Hosted retrieval comparison reproduced; additional source-grounded generation reviews still find false claims, and remote timeouts prevented completion of the planned evaluations | **Open: generation and synthesis quality failed; broader and native-Q4 evaluation required** |
| 5. M5/Metal memory, latency, context and batch qualification | No measurements on either target memory configuration | **Outside user-managed acceptance; still required for Mac-specific performance claims** |
| 6. Fresh installation, full Python/browser acceptance and operator documentation | Earlier 5252 Python checks passed, including 29 evaluation guardrails; 46 selected browser and 19 installed-wheel checks predate the user-managed UI | See the linked follow-up for current acceptance; hardware-specific claims remain item 5 |

Completing an investigation with a failed result does not complete its quality requirement. No performance estimate, simulated-inference check or hosted BF16 result substitutes for M5 measurements of the pinned Q4 artifacts.

## Source-grounded accuracy evaluation before the user-managed UI

The [remote accuracy report](../evaluation/remote-accuracy-2026-10-23/README.md)
preserves frozen source excerpts, exact requests, final-only responses, factual
reviews and unsuccessful runs. No production prompt, scope rule or model setting
was changed during this evaluation. Six new evaluation/test modules were built
with failing-first guardrails; the full suite passed **5252 tests**, with **9
skipped, 1 expected failure and 462 warnings**, in **536.64 seconds**. All **843
frozen source/test/config hashes** remained unchanged. Focused Ruff passes for the
six new modules. The existing wheel and browser checks predate these test-only
additions; they were not rerun or relabelled as new executions.

The QA policy experiment completed 10 of 18 planned outputs before a read timeout.
Every produced output was reviewed: material errors occurred in **5/5 baseline**
and **3/5 candidate** documents. The candidate also lost a previously correct
read-recency fact. It was **rejected**, without consuming reserved confirmation
cases. These are counts from known regressions, not population accuracy estimates.

The real-source baseline completed four of 16 planned documents before its own
timeout. **All four contained material false or unsupported claims**, including
atomic-write guarantees, catch-all exception handling, recursion safety and list
hashability. One covered all required facts while adding false claims elsewhere.
Only configuration and dependency excerpts were reached; no second repetitions
or embedding/catalog baseline documents were completed. Selected excerpts omit
other implementation details, and the domain-specific `depends_on` coverage target
is explicitly qualified in the review.

The unchanged 50-question embedding corpus was replayed through current storage,
search and MCP code with recorded responses. Both profiles reproduced every
metric, including **48/50 designated-document top-five hits** and **48/50 MCP
evidence deliveries**. This spent no provider funds and supplies regression
evidence, not a new independent embedding evaluation. The original relevance-label
limitations remain recorded.

The first dependency-scoped application run stopped before Search and Ask. It
persisted eight catalog descriptions and one explanation, with normalized
3072-dimensional vectors; factual review found an unsupported constructor claim
and synchronous/async errors. A single fresh sequential reattempt returned only
one embedding response, then terminated with a connection timeout without any
completed catalog prose or Search/Ask result. The two unanswered chat attempts
remain counted. Neither a timeout nor a passing offline scope
test establishes that live dependency retrieval succeeded. Native Q4 and M5
qualification remain open.

This evaluation accounted for **$0.16743713 across 41 HTTP attempts**, including
full reservations for five unanswered requests. Including the earlier verification,
the conservative aggregate is **$0.33085641** against the $4.50 ceiling. All four
new runs are terminal; this is usage/reservation accounting, not a billing statement.

## Selected 27B thinking profile

The `qwen-mac-27b-q4-thinking-v1` profile pins
`bartowski/Qwen3.8-27B-GGUF` revision
`87410e7b9a5c21e6ca70c0c9cc5237de909fe621`, file
`Qwen3.8-27B-Q4_K_M.gguf` (17,442,399,968 bytes), SHA-256
`880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673`.
Publisher LFS metadata was read on 2026-10-22. The embedding artifact and its
normalization recipe remain identical to the previous Mac profile. Both weight
files total 22,119,204,896 bytes; this is **not a runtime RAM estimate**.

The profile explicitly requests thinking with `xhigh` effort, temperature 1.0,
top-p 0.95, top-k 20, min-p 0, presence penalty 0 and native repeat penalty 1.
The native server uses `deepseek` response separation. Template preflight carries
the same reasoning controls as generation; only final content reaches document
storage and Ask. Reasoning-only, malformed and truncated responses fail. The
32,768-token context has an **8192-token completion ceiling**, and full document
jobs reserve that amount. It includes thinking and final output together; there is
no additional 8192-token final-document allowance. Eight explicitly named Ask
selector phases receive a bounded 2048-token thinking allowance for this exact
profile; other smaller callers retain their limits. The profile's chat timeout is 600 seconds, subject to explicit shorter
per-call timeouts.

The profile identity includes the thinking/xhigh contract and exact artifacts.
It does not freeze operational settings such as sampling values, context size,
timeouts or aliases. Qualification evidence must therefore record those settings
separately.

This is a **32GB qualification target; 24GB operation remains unqualified**. The
runtime currently keeps both model servers resident. It does not automatically
unload one model to fit the other. Neither a 24GB Gemma fallback nor automatic
model swapping is implemented by this change.

The old `qwen-mac-q4-v1` profile retains its exact identity and pins. Existing
libraries are never silently rebound to 27B. The new profile creates a distinct
project/library for the same source. Matching embedding spaces still allow the
existing packaged-Spool vector-reuse path; this does not imply that an entire
library can be switched in place.

### Using the selected profile through the UI

1. In onboarding, select **Experimental Qwen local on Mac (27B thinking + 8B embeddings, Q4)**.
2. Use **Download models**, then **Start models**. Downloads verify the pinned
   sizes and SHA-256 hashes. The runtime requires native Apple Silicon macOS.
3. Wait for the selected profile to be ready, then build the library. Preview
   remains available before model startup.
4. Reopen the saved project in the console for Ask/Search and saved build controls.
   Its saved model pair remains authoritative.
5. To work with a project using the other profile, use **Stop models**, then
   **Start models** for that project's saved profile. Stop preserves resumable
   work; Cancel build remains terminal.

Both the browser and backend require the running profile ID and model identity to
match the selected project. Unknown saved markers and altered artifact identities
fail explicitly instead of bypassing managed-runtime checks. Source selection and
`depends_on` semantics are unchanged.

### Implementation evidence for the selected profile

Observed failing tests preceded the profile/runtime, template-preflight and UI
changes. The profile/runtime set passed **66 tests**; adapter plus legacy provider
and timeout regressions passed **48**. Native document generation, storage,
reopening, dependencies and clews now run against both profiles; their combined
workflow/library/Spool/adapter set passed **77 tests**. UI/backend profile guards
passed **115 Python tests**, and the focused managed-model/run browser set passed
**19 checks**. These sets overlap and must not be summed as independent coverage.
The changed profile/runtime/adapter files and new tests pass their focused Ruff
checks; unrelated repository lint findings have not been suppressed.

The hosted 27B thinking diagnostic is described by its
[preflight review](../evaluation/mac-local-models-2026-10-22/generation-27b-thinking-preflight.md).
It uses the existing DeepInfra mechanism, production explanation/QA prompts and
frozen synthetic sources. Its transformed final-only archives explicitly omit
private reasoning. Hosted results do not qualify the pinned Q4 artifact or M5.

All twelve hosted requests completed normally. Independent coding-agent review
found **9/12 documents with material errors** and **6/12 with complete, consistently
correct coverage of the frozen behavior targets**. The candidate fails the
correctness gate. The [complete review](../evaluation/mac-local-models-2026-10-22/generation-27b-thinking/review.md)
links each finding, immutable inputs and final outputs. Three previously studied
synthetic cases cannot establish a general accuracy rate or a comparative ranking.

### Budgeted verification and fixes, 2026-10-23

The [completion verification report](../evaluation/mac-local-models-2026-10-22/completion-verification/README.md)
records 60 paid HTTP requests across five frozen runs, totaling **$0.16341928**
in conservative usage accounting against a $4.50 ceiling. No further funds were
spent on retries of failed content judgments. All five runs in that checkpoint settled.

Three production fixes followed observed failures and failing offline guardrails:

- Four short selector calls exhausted their total thinking/output budgets. The
  shared completion helper now adds 2048 tokens for eight explicitly named Ask
  selector phases, only for the exact selected local model-pair identity and
  within its configured ceiling. Diagnostics and context preflight use the
  actual cap. Four original truncations disappeared in the replay. Legacy and
  hosted profiles, prose calls, unknown phases and saved durable requests retain
  their existing limits. Larger reservations reduce available prompt capacity;
  oversized contexts are rejected before generation.
- Catalog descriptions previously inferred runtime behavior from metadata alone.
  Their shared prompt now describes the supplied interface and identifies unknown
  behavior. Both the captured-call replay and hosted application workflow produced
  metadata-grounded descriptions. Existing saved descriptions and durable requests
  are not rewritten.
- The compact planner invented intermediate methods after its truncation was
  resolved. Its prompt now restricts obligations to the question, forbids assumed
  answers and invented identifiers/routes, and makes five a ceiling. Three
  consecutive confirmations of the same original question each produced one
  grounded obligation. This is a narrow repeated-case check, not general planner
  qualification. The parser, retrieval and dependency closure are unchanged.

A paired **24-document** generation experiment compared the existing recipe with
stricter source-grounding instructions across six frozen sources. The candidate
reduced material-error documents from **11/12 to 7/12** and increased complete,
consistently correct required-behavior coverage from **2/12 to 9/12**. It still
lost a correct fact and introduced another false claim, failing the frozen
correctness, coverage and non-regression gates. **The candidate recipe was
rejected; production explanation and QA prompts were not changed.** These small
sample counts are not general accuracy estimates. See the
[complete paired review](../evaluation/mac-local-models-2026-10-22/completion-verification/generation-recipe-r1/review.md).

The ten-call replay completed without truncation, but independent review found
unsupported claims in the planner, cascade answer and theme. The planner received
the subsequent narrow fix above. The cascade answer inferred a runtime call count
from duplicate same-site graph edges, and the theme inferred shared objects and
ordering from sparse summaries. These failures remain open. Downstream replay
inputs use scripted upstream fixture responses, so this does not establish a
complete real-model compiler-aware Ask workflow.

A separate real hosted application run passed **15 orchestration checks** covering
catalog, generation, 3072-dimensional document/chunk/section embedding storage,
Search/Ask, unchanged repeats, transactional rollback and reopening. However,
**both generated documents failed factual review**, and Ask repeated an error
from the retrieved documentation. Its captured request also contains conflicting
compiler-evidence and document-citation instructions. These findings are preserved
in the [workflow review](../evaluation/mac-local-models-2026-10-22/completion-verification/hosted-workflow-r1/review.md).
The harness used hosted model identities and mapped thinking/sampling options;
it did not execute pinned GGUF weights, native durable UI generation or Metal.

The adapter rejects truncated and reasoning-only responses before applying document
output. Optional Ask selectors may still use their existing fallbacks; synthesis
failures propagate. Passing structural/orchestration checks does not demonstrate
correct answers or that fallbacks preserve quality. Trace retains its explicit
30-second timeout; its recorded successful call does not establish a latency SLO.

## Implemented software and checks

### Final software acceptance after budgeted fixes

The final full Python suite passed **5223 tests**, with **9 skipped, 1 expected
failure, 462 warnings and no failures**, in **371.14 seconds**. All **837 frozen
source/test/config hashes** were verified unchanged after the run. The
[software summary](../evaluation/mac-local-models-2026-10-22/completion-verification/software/summary.json)
links complete JUnit results, captured stdout tail, skip reasons and source
snapshots. An earlier 5221-pass checkpoint predates the final planner fix and is
preserved separately. A first mixed-version run collected old capture expectations
while those files were being updated and failed once; it is also retained.

The final wheel installed outside the checkout into a fresh isolated environment
and passed **19 checks**: both model contracts, new selector/catalog/planner
behavior, four CLI invocations, two actual MCP sessions exposing 57 tools each,
served UI assets and onboarding model choices. All **17 checked files** matched
checkout, wheel and installation. The install used **73 cached distributions**,
Python 3.14.2 and MCP 1.30.0 on Intel macOS. The initial sandbox loopback denial and
subsequent successful authorized retry are preserved in the
[installed-wheel archive](../evaluation/mac-local-models-2026-10-22/completion-verification/installed-wheel/README.md).
No model ran during packaging checks; package metadata remains 0.3.0.

All **13 new verification/test modules** pass focused Ruff. Existing repository
findings were not suppressed; repository-wide lint is not claimed. The earlier
**46 selected browser checks** remain applicable to unchanged UI assets; no UI
code changed during this verification pass. They use fixture inference and do
not establish real-model UI accuracy or target-hardware performance.

### Earlier software acceptance before budgeted fixes

The earlier full Python run passed **5141 tests**, with **9 skipped, 1 expected
failure, 462 warnings and no failures**, in 354.71 seconds. An earlier full run
exposed two interruption fixtures that marked the new default runtime ready while
requesting a legacy-profile project. The profile guard correctly rejected them.
The fixtures now identify the running profile explicitly and exercise both
profiles; 34 focused checks passed before the full suite was repeated.

All **46 selected browser checks passed** across seven specs, with no failures,
pending cases or skips: console (5), onboarding (4), model selection (6), Mac model
controls (11), durable Mac runs (8), security (9), and backend atomicity (3).
These are offline checks using simulated inference; the backend atomicity cases
cross the actual isolated MCP boundary. Hosted journeys, container deployment,
Metal execution and real-model UI quality are not established by this run.

Focused Ruff checks pass for the changed profile, runtime, adapter, project
configuration and their tests. The 29 existing `web/server.py` diagnostics remain
recorded, not suppressed. Passing repository-wide lint is not claimed.

The [earlier software checkpoint](../evaluation/mac-local-models-2026-10-22/accuracy-software-checkpoints/summary.json)
preserves the full-suite and browser logs, the failed fixture run, focused lint,
and saved TDD logs. The additional focused UI red/green results were observed in
tool output; they do not have separate archived log files. Its
[hash manifest](../evaluation/mac-local-models-2026-10-22/accuracy-software-checkpoints/manifest.json)
also covers the changed application/test source snapshot and test-environment
package inventory. The MCP dependency cap was applied after the full Python run;
the test environment already used MCP 1.26.0, and fresh-wheel acceptance checks
the corrected installation metadata separately.

### Earlier fresh installed-wheel verification

A fresh install exposed a concrete packaging failure: the former `mcp>=1.26.0`
requirement admitted MCP 2.3.0, which removed the v1 FastMCP import Ariadne uses.
The installed CLI could load, but actual MCP initialization failed. The fix adds
`<2`; the lockfile changes only requirement metadata, with no dependency upgrades.

The rebuilt wheel installed 73 distributions in a second fresh environment,
selecting **MCP 1.30.0** without lock constraints or shared site-packages. All
**17 checks passed**, including installed imports, both pinned profiles, four CLI
invocations, two actual MCP initialize/list-tools sessions exposing 57 tools each,
five served UI assets and the onboarding model choices. Fourteen checked
source/UI asset hashes match the wheel and current source.

The first HTTP attempt was blocked by the sandbox's loopback restriction. The
approved outside-sandbox invocation completed the same check successfully. The
[installation archive](../evaluation/mac-local-models-2026-10-22/accuracy-distribution/README.md)
preserves the original dependency failure, minimal fix, fresh installation,
sandbox-limited attempt and final passing result. Both environments were outside
the checkout, on Intel macOS/Python
3.14.2, using cached dependency distributions. They do not establish installation
on M5 or a freshly fetched registry install. No models ran, and package metadata
remains 0.3.0.

### Earlier implementation checkpoints

The new candidate profile pins the generation and embedding GGUF revisions, filenames, byte sizes and SHA-256 hashes. Its identity is distinct from the existing BF16 local profile. Existing binding and storage code remains responsible for preventing incompatible reuse.

The native adapter uses llama.cpp's template rendering and tokenization endpoints and checks the server context limit against the effective output reservation (8192 tokens for full document jobs; selected-profile Ask selectors now include the bounded thinking allowance described above). Embeddings retain the existing native 4096 to normalized 3072-component float32 recipe. The tests exercise document, chunk and section persistence, malformed-vector rejection without changing existing knowledge, reopening, sidecar ownership and retained `depends_on` configuration.

The runtime verifies downloaded bytes before publishing files, owns its server processes, and handles failed startup, cancellation and shutdown. Process supervisors stop their model child if Ariadne's controlling pipe closes. These behaviors have deterministic HTTP and real disposable-process tests; native Metal execution remains unmeasured.

The web backend and shared onboarding/console controls expose the managed candidate and download/start/stop actions, progress, failures, unsupported hardware and its experimental quality status. Offline preview remains available; managed builds and Ask/Search require runtime readiness, while external projects remain independent. The UI retains the selected project for queries. Download/start cancellation and durable generation controls are implemented; their evidence is distinguished below.

After observed failing guardrails, the focused UI completion run passed **19 browser tests** (8 Mac lifecycle/selection/query cases, 6 existing model-selection cases and 5 console cases), and **52 backend tests** passed. Two additional regressions were reproduced and fixed: returning through the browser's back/forward cache resumes runtime polling, and old success-notification timers cannot dismiss a newer build error. These use simulated inference and do not qualify actual model quality or target hardware.

### Durable generation controls (item 2)

Saved local projects expose persisted runs in both onboarding and the console. Run discovery and cancellation work with model serving stopped. Resume uses the original project configuration and model binding; it does not create a replacement build. The browser reports applied **jobs**, which include catalog and embedding work, not just generated documents. Closing the tab leaves execution running; reopening discovers durable state even when the previous web job/event stream no longer exists.

**Cancel build is terminal:** it records cancellation before interrupting a positively matched owned worker, retains completed documents, and prevents late results from being applied. **Stop models is resumable:** it interrupts and reaps owned managed workers before stopping model serving, without terminally cancelling the persisted run. Restarting and resuming can reuse a saved model response. Completed, failed, obsolete and cancelled runs cannot be submitted for resume through the web API.

Tests first reproduced overlapping start/resume admissions, new work entering during Stop cleanup, cancellation interrupting an unrelated session, and disconnected cancellation requests leaving a pending browser caller. Shared project admission and owned cleanup now cover these races. External and hosted projects remain independent of managed-runtime Stop. Browser event history retains active jobs and the latest 64 completed jobs; persisted local runs remain discoverable after this history expires.

Two integration tests cross the actual MCP subprocess boundary, native HTTP fixtures, the run store and the persisted library. They verify cancellation-before-reaping; Stop-before-model-shutdown ordering; unchanged previously applied document/chunk/section rows; offline inspection through a fresh process; and restart/resume with **one generation request total**, rather than repeating the saved response. These are software integration checks with fixture-backed inference, not actual model or M5 measurements.

The item-2 checkpoint passed **146 backend tests** (336 warnings, 29.11 seconds):

```sh
.venv/bin/python -m pytest \
  tests/test_local_run_controls.py tests/test_web_job_lifecycle.py \
  tests/test_web_local_runs.py tests/test_onboarding_projects.py \
  tests/test_mac_onboarding.py tests/test_mac_runtime_api.py \
  tests/test_web_server.py tests/test_project_interruption.py \
  tests/test_web_run_integration.py -q --tb=short --disable-warnings
```

All **27 affected browser tests passed across the final runs**: 19 existing model-control, selection and console tests in the combined run, and 8 durable-run tests after correcting a delayed-response test fixture. The latter cover reopening/resume, offline cancellation, missing event-stream recovery, escaped errors/retry, hosted recovery in both response orderings, failed status polling and browser history restoration. A genuine hosted recovery regression was observed before its fix; the initial second-order failure was a fixture error, not product evidence. Local evidence logs are under `build/mac-local-models-tdd/runs-backend-final.log`, `runs-ui-final.log` and `runs-ui-final-corrected-fixture.log`. This is a focused checkpoint, not a new full-suite result.

### Native workflow and Spool compatibility (item 3)

Thirteen new native-protocol checks pass generation, validation, storage, provenance and reopening for explanation, architecture, QA, catalog, gotcha and diagram documents. They also exercise catalog synchronization/description and unchanged repeats, coherent/incoherent themes, forward and reverse `depends_on` closure in Search/Ask, and clew reopening, embedding reuse, route selection, compiler evidence and citations with both selector modes. These paths required no production changes. Responses are deterministic fixtures, so these are integration assertions rather than model-quality judgments.

Packaged Spools did need changes. A bound builder now records its actual embedding identity, including artifact revision, format and embedding recipe. Exact embedding-space matches preserve document, chunk and section vectors without inference, even when the generation model differs. A mismatch re-embeds every vector-bearing record, including theme children, under the consumer's document recipe before modifying the destination. Legacy manifests remain readable; omitted embedding metadata cannot disguise vectors present in the artifact. Malformed identities and collisions with unrelated source documents fail explicitly.

Build artifacts open independently of the active source library. Install/reinstall updates documents and derived metadata in one database transaction. Ordinary failures restore prior rows and cache files. A pending-install marker prevents a newly published cache from registering before commit; abrupt process exit before or after commit leaves it disabled until a successful reinstall. Reader checks cover concurrent publication during manifest loading. These are process-exit tests, not power-loss or filesystem durability certification. The shared metadata schema initializers now preserve their caller's transaction instead of implicitly committing it.

The shared completion helper also honors the configured local timeout (180 seconds for the legacy 9B candidate; 600 seconds for the selected 27B profile). It previously replaced that with its historical 60-second default. Explicit per-call limits and hosted defaults remain covered. Its focused regression run passed **51 tests** after two failing native timeout guardrails.

The compatibility checkpoint passed **236 tests**, with 10 warnings, in 60.29 seconds:

```sh
.venv/bin/python -m pytest tests/test_spool*.py \
  tests/test_mac_spool_pack.py tests/test_version_facts.py tests/test_surface_tags.py \
  tests/test_mac_workflows.py tests/test_mac_library.py \
  tests/test_chat_timeout_contract.py tests/test_web_run_integration.py \
  -q --tb=short --disable-warnings
```

After the final manifest-reader race fix, **28 registration/recovery tests passed** in 7.21 seconds (`test_spool_install_recovery`, `test_spool_config`, `test_spool_cache`, `test_spool_enable_cmd`, `test_spool_untrusted`). The four new item-3 test modules pass Ruff. Logs: `build/mac-local-models-tdd/compatibility-final.log` and `spool-reader-race-green.log`. Existing unrelated lint findings and the existing source-scope expected failure were not suppressed or changed. The [checkpoint archive](../evaluation/mac-local-models-2026-10-22/software-checkpoints/manifest.json) preserves item-2/item-3 logs and later evaluation-harness checks with SHA-256 hashes, including observed failing guardrails and their passing runs.

### Earlier baseline checks

These results precede the item-1/item-2/item-3 completion work above; they are not final acceptance of the current diff.

The combined backend command passed **128 tests**, with 66 aiohttp application-key warnings:

```sh
.venv/bin/python -m pytest \
  tests/test_mac_model_profile.py tests/test_model_config.py \
  tests/test_llamacpp_provider.py tests/test_local_provider.py \
  tests/test_local_embeddings.py tests/test_local_contract_fixtures.py \
  tests/test_model_binding.py tests/test_bound_hosted_models.py \
  tests/test_local_runtime.py tests/test_local_runtime_worker.py \
  tests/test_mac_library.py tests/test_mac_onboarding.py \
  tests/test_mac_runtime_api.py tests/test_onboarding_projects.py \
  tests/test_web_server.py -q
```

A subsequent full Python run (`.venv/bin/python -m pytest -q --disable-warnings`) passed **4865 tests**, with **9 skipped, 1 expected failure and 108 warnings**, in 273.94 seconds. This full run preceded the UI completion checks above. The original three missing-control browser failures were TDD guardrails and are now passing. Existing unrelated Ruff diagnostics have not been suppressed. Neither software result qualifies model output quality or M5 operation.

The [regression summary](../evaluation/mac-local-models-2026-10-22/python-regression.json) records the non-passing cases: fixture cases without applicable symbols/dependencies, unavailable saved calibration traces, two previously abandoned traversal modules, and an existing source-scope test marked expected failure. After this full run, a new test exposed a collision between managed and externally configured projects with identical model weights. Their project identities are now separate; the subsequent focused project/storage/runtime API run passed **26 tests**.

Further failing regressions reproduced readiness during shutdown, saved-project rebuilds bypassing readiness, and supervisor module shadowing from the working directory. Fixes reject managed builds while stopping, check saved-project build admission, and launch the owned supervisor file with isolated Python. Focused runs passed **39**, **32**, and finally **50 tests**, respectively. External project readiness and dependency closures remain unchanged. These later results are separate from the earlier full-suite run. The nine skips comprise seven test cases and two collection skips; the full-run timing above is the pytest CLI value.

## Earlier actual-model checkpoint: legacy 9B

This checkpoint used the legacy **Qwen3.5-9B Q4** generation artifact and **Qwen3-Embedding-8B Q4**, not the selected 27B artifact. Both legacy-pair weight files were downloaded and matched their pinned hashes in `local_models.py`. The actual runtime was the official llama.cpp **b11669**, commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`, x86_64 build, running with GPU offload disabled. Its archive SHA-256 was `cab28238d78158109b48deb27b05d50e0dcef56275d2e76da9ca8281526c7532`.

The host is an Intel Core i9 Mac with 64 GB RAM. The CPU probe used a 32768-token chat context, an 8192-token generation reservation and concurrency one. Operational timeouts were raised for this CPU experiment. This is output-quality evidence from the actual quantized artifacts; it is not evidence for Apple Silicon performance, Metal allocation or the proposed 24/32 GB operating envelope.

Three existing synthetic cases from `tests/source_verification_cases.py` were processed through the production DocGenerator, LibraryWriter, Search and Ask paths. Only checked-in source fixtures were executed to verify expected behavior; generated code was not executed. Each source had a separate root. Expected answers were kept outside model prompts.

The comparison used the repository's existing OpenAI profile, GPT-6 Astra with `text-embedding-3-large`, through the same application workflow and identical source files. Its recorded request-accounting estimate was **$0.27134719**, under the probe's $5 estimated-spending limit. That is harness accounting, not a billing statement. One run per pair and three examples cannot establish an overall accuracy rate.

| Case | Local generated documentation | OpenAI generated documentation |
| --- | --- | --- |
| Unguarded cache | Falsely says ordinary reads update recency; omits the negative-capacity `KeyError`; gives an incorrect exception type for plain dictionaries | Correctly distinguishes reads/writes, zero/negative capacity and the relevant exceptions |
| Guarded cache | Wrong order after updating an existing key; omits the truthiness guard's negative-capacity behavior; claims an invalid unconditional capacity invariant | Correct update order and explicit explanation of why negative capacity returns with an empty cache |
| Token expiry | Correct target behavior, but reverses clock-skew consequences in additional advice | Correct equality/expiry and removal behavior; a separate coding-agent review found no similarly material source errors |

All six documents passed structural validation and were stored. That does not make them factually correct. In particular, the local documents demonstrate source-level errors before retrieval or Ask can affect the result.

### Isolated thinking-mode diagnostic

The same 9B artifact was also tested on the unguarded cache case with thinking enabled, temperature 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 0 and repeat penalty 1. The request reused the production generation messages from the OpenAI comparison and contained the fixture source, without oracle answers. This changes both mode and sampling; it does not isolate the causal effect of thinking alone. The production non-thinking contract was not changed.

The request completed with `finish_reason: stop`, 291 prompt tokens and 4068 completion tokens, within the same 8192-token output cap. It took 1931.34 seconds on the Intel CPU host. Its [final document](../evaluation/mac-local-models-2026-10-22/thinking-cache/cache-qa.md) still groups zero and negative capacity together as immediately discarding data, omitting the negative-capacity `KeyError`. This diagnostic therefore does not resolve the target failure. It tests one generated document, not storage, retrieval, Ask, representative quality or M5 performance. The [request](../evaluation/mac-local-models-2026-10-22/thinking-cache/request.json), [report](../evaluation/mac-local-models-2026-10-22/thinking-cache/report.json), final response and [experiment script](../evaluation/mac-local-models-2026-10-22/thinking-probe.py.txt) are preserved; private reasoning text is not documentation and was not archived.

The original non-thinking request and managed server command did not explicitly set `min_p`. The pinned runtime's [source default is 0.05](https://github.com/ggml-org/llama.cpp/blob/b11669/common/common.h#L235), but the original server's effective value and possible initialization overrides were not captured. This is a sampling-control limitation, not an established explanation for the quality failures. The later thinking diagnostic explicitly used zero and still missed the target behavior. Capture effective sampling settings in subsequent comparisons.

### Remote validation using the previous Qwen mechanism

At the user's direction, the next evaluation used the existing DeepInfra recorder and client mechanism with exactly **`Qwen/Qwen3.5-9B`** and **`Qwen/Qwen3-Embedding-8B`**. DeepInfra serves the [generation model in BF16](https://deepinfra.com/Qwen/Qwen3.5-9B/api); its embedding precision and immutable weight revisions are undisclosed. This checks the same base model names remotely, while local Q4 execution and M5 hardware remain unqualified.

The run replayed the three frozen production generation prompts twice each, with thinking disabled and sampling settings explicitly recorded before inference. All **six generation requests and four embedding requests returned HTTP 200**. All generations completed normally and passed structural validation. A separate coding-agent review found **material source-level errors in all six documents**, including false read-recency behavior, an invented infinite loop despite a terminating guard, and reversed token-removal order. Some target facts were correct; the verdict concerns errors present in each complete document. This small diagnostic is not a general accuracy estimate.

All nine embedding vectors passed native-width and finite-value checks and the existing 4096-to-3072 float32 conversion. Independent normalization of the raw vectors matched the saved vectors within 4.73e-9 per component. This verifies vector handling; six documents and three queries do not qualify embedding accuracy or application retrieval. No library, Ask/Search or UI workflow ran in this remote diagnostic.

The run took **648.97 seconds**, with **$0.00160012** in harness request accounting. The [review](../evaluation/mac-local-models-2026-10-22/remote-deepinfra-9b/review.md) records per-document errors, source behavior, limitations and the reproduction command. The [protocol](../evaluation/mac-local-models-2026-10-22/remote-deepinfra-9b/protocol.json), [raw report](../evaluation/mac-local-models-2026-10-22/remote-deepinfra-9b/report.json), exact requests/responses, documents, vectors and selected source snapshots are archived with hashes. Raw run metadata remains pending review; the separate review carries the factual verdict.

The new opt-in runner was implemented through failing-then-passing tests. Its [final regression run](../evaluation/mac-local-models-2026-10-22/remote-probe-regression.json), including the prior Qwen recorder and hosted-provider tests, passed **31 tests**. Ruff passed for the runner and its tests. These checks are additional to the earlier full Python run, not a new full-suite result. Production provider contracts, generation prompts and retrieval/dependency behavior were not changed for this experiment.

### Fresh retrieval comparison (item 4)

The existing frozen 50-question/22-excerpt benchmark completed with **51 successful embedding requests**, followed by an offline replay of every recorded response. Both models found the designated document for **48/50** questions and delivered its target evidence through MCP for **48/50**. Chunk retrieval was **45/46 for hosted Qwen3-Embedding-8B** and **44/46 for text-embedding-3-large**. Both profiles passed 53 implementation checks, and replay reproduced every metric exactly. The harness had passed 20 offline tests before the live run.

The [comparison review](../evaluation/mac-local-models-2026-10-22/retrieval-current-network/review.md) reports all metrics, timing/pacing, $0.00191526 in request accounting, reproduction and limits. A separate [label and miss review](../evaluation/mac-local-models-2026-10-22/retrieval-current-network/label-review.md) found some omitted valid alternatives without changing the frozen labels. The actual document/MCP misses remained substantive: no adequate alternative answer was present in their returned document sets. The corpus tests known-target retrieval, with simple section layouts; it does not establish general answer accuracy. Hosted precision/revision remains undisclosed, so this result does not qualify the pinned local Q4 embedding artifact.

### Controlled generation investigation (item 4)

A [frozen paired trial](../evaluation/mac-local-models-2026-10-22/generation-scope-trial/protocol.json) compares the original QA prompt with one appended generic source-scope instruction. It schedules six source cases, two arms and two repeats: 24 documents, concurrency one, at most 32 HTTP attempts and a $1 accounting cap. Model, source, output limit and sampling are identical within each pair. The three additional cases were used in earlier 27B work; they are new to this 9B comparison, not independent unseen holdouts. Expected behavior stays outside the transmitted requests.

The new harness was developed through observed failing guardrails and passed **43 tests** plus Ruff. A separate agent reviewed its request boundary, schedule, evidence retention and limits before live execution. The protocol requires separate review of whole-document factual errors and target-behavior coverage; an omission or abstention is not a correct answer. Calling a candidate promising requires fewer materially erroneous documents, no lost correct behavior and no reduction in coverage. Production qualification additionally requires full correct coverage, zero material errors and broader representative evaluation; this small trial cannot grant it.

The trial stopped on a 180-second provider timeout at request 14, with **13/24 documents completed**. All six first-repeat pairs completed; separate source review found definite material errors in **5/6 baseline documents** and **6/6 candidate documents**, with no fully covered required-behavior target in either arm. The candidate introduced a wrong zero-capacity exception and an incorrect negative-index boundary, losing previously correct behavior. These are descriptive subset findings, not a completed-trial improvement estimate. The [trial review](../evaluation/mac-local-models-2026-10-22/generation-scope-trial/review.md) records rejection of the instruction, the incomplete experiment, all findings, 1865.87 seconds elapsed and $0.00451565 in accounting including the timed-out request's reservation. No production prompt has been changed. Generation quality remains an open gate.

A second [six-call diagnostic](../evaluation/mac-local-models-2026-10-22/generation-presence-zero/protocol.json) kept the original prompts and changed only the candidate's presence penalty from 1.5 to 0. Both arms used the same 600-second per-operation HTTP timeout, with at most ten attempts and $1 in accounting. This tested the model author's warning that higher presence penalties can reduce performance; it did not presume that the documented general non-thinking preset was a configuration error. The extension passed **50 offline tests** plus Ruff after observed failing guardrails, and received independent review before execution.

All **six requests returned HTTP 200**, completed normally, and produced structurally valid documents. The run took **719.47 seconds**, with **$0.00169345** in request accounting. Separate whole-document [source review](../evaluation/mac-local-models-2026-10-22/generation-presence-zero/review.md) found material errors in **2/3 documents per arm**, with full required-behavior coverage in **1/3 per arm**. Clear code as well as prose counted toward coverage; neither token-expiry document was classified as materially false. The cache candidates nevertheless failed, and the guarded-cache candidate lost the baseline's correct zero-capacity result and positive-capacity bound. The setting change therefore failed the frozen correctness, coverage and non-regression requirements and was rejected. These three previously examined cases and one sample per arm do not establish causality or general accuracy. No production sampling setting changed.

## Separate downstream limitation

Neither pair delivered all three complete, grounded target answers through the earlier local/OpenAI application-workflow comparison. The local cache answer abstained; its guarded-cache answer reached the correct yes/no result using unsupported reasoning from missing documentation. The token-expiry answer was correct.

The OpenAI Ask traces show a separate, existing evidence-delivery limitation: each prompt contained exactly the first 3000 characters of the generated document. Both correct negative-capacity sections fell beyond that prefix. OpenAI therefore gave only a partial unguarded-cache answer and abstained on the guarded case. Token expiry was answered correctly because its target facts were inside the prefix.

The recorded prompts also contain conflicting citation instructions: the system requires compiler-derived file/line evidence, while the documentation fallback asks for document citations. These findings are recorded, not silently repaired as part of adding the Mac adapter. Any change needs its own narrowly defined regression expectations, including existing dependency and compiler-aware behavior.

The first local run did not save exact Ask prompts. Its standalone Search output is not proof of the exact Ask context. This limitation prevents assigning all local answer omissions to either the model or context selection. The tiny source-scoped test also does not measure comparative embedding quality.

## Preserved evidence and remaining work

The [local report](../evaluation/mac-local-models-2026-10-22/local/report.json), [OpenAI report](../evaluation/mac-local-models-2026-10-22/openai/report.json), full unedited documents, answers and Search results are retained alongside the [experiment script](../evaluation/mac-local-models-2026-10-22/probe.py.txt). OpenAI Ask traces, all twelve request bodies and the [cost-accounting report](../evaluation/mac-local-models-2026-10-22/openai/cost-accounting.json) are retained too. The latter retains a stale `running` field because this experiment did not finalize the enclosing harness metadata; all twelve requests returned HTTP 200 and the final outputs were saved. The script is an archived experiment, not a supported product command or a release gate. The first local run predates its added trace capture. Reviews were performed by coding agents; no human or blinded review is claimed. The exact dirty application source snapshot was not captured at inference time, which limits reproducibility.

Examples: [local cache document](../evaluation/mac-local-models-2026-10-22/local/cache-qa.md), [OpenAI cache document](../evaluation/mac-local-models-2026-10-22/openai/cache-qa.md), [OpenAI cache Ask trace](../evaluation/mac-local-models-2026-10-22/openai/cache-trace.json).

Still required:

- Resolve the generation candidate's demonstrated quality failures before recommending it.
- Preserve the passing UI model lifecycle and durable-generation regressions through subsequent quality fixes.
- Qualify context, batch and concurrency settings on actual target hardware.
- Expand the completed hosted short-call diagnostics into representative full workflows, including selector fallback behavior, grounded synthesis/themes and native-Q4/trace-bridge limits.
- Complete a broader judged retrieval comparison and repeat factual-quality evaluation on representative repositories.
- Measure peak memory and latency on the intended M5 configurations; the earlier memory ballpark is not a measured guarantee.
- Rerun appropriate regression checks after subsequent implementation changes; the current full Python and selected browser acceptance runs pass.
