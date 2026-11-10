# Test coverage and consolidation

The development environment includes pytest-cov 7.1 or newer. Existing
configuration measures Python lines and branches. Coverage is opt-in; the normal
pytest command remains available without instrumentation.

The locked toolchain was verified with Python 3.14.2, coverage 7.13.5 and
pytest-cov 7.1.0. Configuration explicitly selects `core = "ctrace"`: the default
SysMonitor core on this combination produced incomplete per-test attribution.
pytest-cov switches contexts manually, and this coverage version does not use the
SysMonitor context callbacks. A nonempty context list alone did not reveal the
missing candidate tests. Confirm attribution before treating a coverage run as
evidence that one test covers another's execution paths.

The pytest-cov dependency floor matches the verified plugin generation. Its
installed package metadata requires coverage with TOML support at version 7.10.6
or newer; this is not a claim about the earliest release supporting each setting.
Use the lockfile to reproduce the verified versions and recheck context attribution
when upgrading tools.

Run commands from the repository root with the development environment installed:

```sh
mkdir -p build/coverage
COVERAGE_FILE="$PWD/build/coverage/.coverage" \
  .venv/bin/python -m pytest --cov --cov-context=test \
  --cov-report=term-missing --cov-report=json --cov-report=html
```

`--cov` without a value uses the source and omission rules in `pyproject.toml`.
`--cov-context=test` records each test's setup, call and teardown phases. The HTML
report and raw coverage database expose those contexts. JSON omits the repeated
context names to keep full-suite reports compact. An absolute `COVERAGE_FILE` keeps
this run separate from an existing root `.coverage` and avoids ambiguity if a test changes
the working directory. Do not use `--cov-append` when comparing separate runs.

Confirm the actual tracer and expected per-test contexts before using the result
to remove a test. For diagnosis, set `COVERAGE_DEBUG=core` and
`COVERAGE_DEBUG_FILE="$PWD/build/coverage/core.log"` on the pytest invocation and
inspect the recorded core selection. A requested core is not proof of the actual
tracer if an installation falls back. Some tests legitimately execute no measured
production lines, so distinguish that from missing attribution using known
nonempty production calls.

Open `build/coverage/html/index.html` in a browser to inspect executed/missing
lines and the tests associated with them. `build/coverage/coverage.json` is the
machine-readable report. Keep `build/coverage/.coverage` as well: the coverage data
contains per-context executed lines and branch arcs needed for precise comparisons.
JSON is the aggregate line/branch report. Even when enabled explicitly, its line
contexts alone do not provide per-test branch attribution.

## Before and after a consolidation

Use separate directories and keep the same tracer, source selection, omissions
and branch settings for both measurements. Per-test comparisons must use the
validated CTracer data on both sides; do not compare a SysMonitor test context
against a CTracer test context. For example, record a baseline:

```sh
coverage_run="$PWD/build/coverage/before"
mkdir -p "$coverage_run"
COVERAGE_FILE="$coverage_run/.coverage" \
  .venv/bin/python -m pytest --cov --cov-context=test \
  --cov-report=term-missing \
  --cov-report="json:$coverage_run/coverage.json" \
  --cov-report="html:$coverage_run/html"
```

After consolidating, repeat with `coverage_run="$PWD/build/coverage/after"`.
A focused run can name the candidate test modules after `pytest`; its percentage
is not directly comparable with a full-suite run. Preserve the exact test selection
and command alongside each focused comparison.

Consolidation requires both execution evidence and assertion review:

1. Match candidate tests to their recorded node IDs, including parameter IDs.
   Compare the production lines and branch arcs exercised by each candidate with
   the retained test or retained set of tests.
2. Inspect run-only coverage and the union of setup/run/teardown. Collection-time
   execution, including imports, has no individual test owner. Shared fixture or
   import warm-up can otherwise look like unique behavior.
3. Retain distinct inputs, failure boundaries, route behavior and assertion
   guarantees. The same lines can implement different cases; hitting them is not
   evidence that the same outcomes were checked. Preserve unique assertions before
   removing an overlapping test.
4. Re-run the affected tests and compare before/after line and arc sets. Then run
   the broader regression check appropriate to the change. Record any lost
   execution coverage explicitly rather than relying on an unchanged rounded
   percentage. Targeted fault injection can check whether retained assertions
   still reject the failure being guarded.

CoverageData's `set_query_context()` selects an exact context;
`set_query_contexts()` uses regular expressions. Escape and anchor node IDs when
using the latter: parameter names often contain regex characters. Its `lines()`
and `arcs()` methods honor the selected contexts. Ignore empty per-context file
sets when comparing and normalize paths consistently.

No arbitrary coverage percentage or target test-count reduction is imposed here.
Coverage measures execution reach; it does not establish factual model accuracy,
correct assertions or complete behavior coverage.

## Measurement limits

This configuration measures the pytest process and its normal threads. It does
not enable subprocess coverage. In the installed pytest-cov 7 series, automatic
subprocess measurement was removed; child CLI/MCP workers and external tools must
not be counted as measured solely because their parent test ran.

Coverage's subprocess patch can be introduced separately and validated for the
actual child environments. It requires compatible Python startup support and
inherited configuration; children in other environments or abnormal exits may
remain unmeasured. Ordinary subprocess bootstrapping also does not automatically
carry pytest's changing per-test dynamic context into the child. Keep aggregate
subprocess measurement and individual-test attribution distinct.

For before/after comparisons, do not change subprocess tracing in only one run.
Likewise, inspect the measured file list before interpreting the denominator.
Executed evaluation utilities can legitimately appear beyond the application files
found by source discovery. This suite also compiles fixture text with logical
filenames that have no corresponding source file. Preserve the raw coverage data
and `couldnt-parse` warnings for those filenames; reported file totals need not
match the raw measured-file count. Do not hide warnings, evaluation execution or
application code through new exclusions to improve the reported percentage.

## Local Qwen and Gemma critical paths

The native-profile tests exercise legacy Qwen, Qwen accuracy and Gemma with the
same Qwen embedding recipe. They use independent loopback HTTP fixtures, real
SQLite libraries and, for dashboard interruption tests, real MCP subprocesses.
They do not measure model quality, inference speed or memory usage on an M5.

| Boundary | Assertions to retain |
| --- | --- |
| Generation | Exact model and template options in preflight and completion; async/sync final output; context limits; malformed or reasoning-only responses rejected |
| Failure handling | Bounded retries, no hosted fallback, mismatched models and changed persisted requests rejected before inference |
| Embeddings and storage | Query/document framing, batching, native 4096 → normalized 3072 vectors, invalid vectors leave existing knowledge intact |
| Library ownership | Exact model-pair binding on reopen; no rebinding an existing Qwen project to Gemma |
| Retrieval and portability | Search/Ask dependency scope, clew evidence and synthesis, compatible Spool reuse without reembedding, rollback on failed installs |
| Runtime and dashboard | Verified artifacts, profile switching only while stopped, owned process cleanup, cancellation, offline run inspection and restart/resume without regenerating saved responses |

Collect the focused regression suite from the repository root. Use a filesystem
source (`--cov=.`), not repeated dotted module names: pre-importing dotted sources
while configuring coverage caused a NumPy import-order failure on Python 3.14.
The existing omission rules and CTracer setting still apply.

```sh
local_coverage_run="$PWD/build/coverage/local-models/current"
mkdir -p "$local_coverage_run"
COVERAGE_FILE="$local_coverage_run/.coverage" COVERAGE_DEBUG=core \
  COVERAGE_DEBUG_FILE="$local_coverage_run/core.log" \
  .venv/bin/python -m pytest -q --disable-warnings \
  tests/test_model_config.py tests/test_model_binding.py \
  tests/test_local_provider.py tests/test_local_embeddings.py \
  tests/test_local_boundaries.py tests/test_local_atomic_application.py \
  tests/test_local_spool.py tests/test_local_spool_recovery.py \
  tests/test_local_maintenance.py tests/test_local_workflows.py \
  tests/test_local_contract_fixtures.py tests/test_mac_model_profile.py \
  tests/test_mac_accuracy_profile.py tests/test_gemma_profile.py \
  tests/test_llamacpp_provider.py tests/test_local_runtime.py \
  tests/test_local_runtime_worker.py tests/test_mac_library.py \
  tests/test_mac_workflows.py tests/test_mac_spool_pack.py \
  tests/test_spool_install_recovery.py tests/test_mac_onboarding.py \
  tests/test_mac_runtime_api.py tests/test_external_models.py \
  tests/test_external_models_api.py tests/test_local_run_controls.py \
  tests/test_web_local_runs.py tests/test_web_run_integration.py \
  tests/test_project_interruption.py tests/test_mac_selector_budget.py \
  tests/test_ask_synthesis_config.py tests/test_section_fallback.py \
  tests/test_clew_model_support.py \
  --cov=. --cov-branch --cov-context=test --cov-report= \
  --junitxml="$local_coverage_run/tests.xml"

local_coverage_files="local_models.py,model_config.py,local_runtime.py,local_runtime_worker.py"
local_coverage_files="$local_coverage_files,docgen/llm/local.py,inference.py,embedding.py,llm.py,writer.py"
local_coverage_files="$local_coverage_files,library/model_binding.py,library/prepared.py,library/core.py,library/search.py,library/clew_embeddings.py"
local_coverage_files="$local_coverage_files,spool_pack.py,docgen/spool.py,docgen/local_pipeline.py"
local_coverage_files="$local_coverage_files,web/external_models.py,web/projects.py,web/local_runs.py,web/jobs.py,web/server.py"
local_coverage_files="$local_coverage_files,ariadne_mcp/service_admin.py,ariadne_mcp/service_search.py,ariadne_mcp/service_analysis.py"
COVERAGE_FILE="$local_coverage_run/.coverage" .venv/bin/python -m coverage json \
  --include="$local_coverage_files" -o "$local_coverage_run/coverage.json"
COVERAGE_FILE="$local_coverage_run/.coverage" .venv/bin/python -m coverage html \
  --include="$local_coverage_files" -d "$local_coverage_run/html"
```

These 25 complete modules include hosted and unrelated functionality. Their
aggregate percentage is a scope-specific measurement, not a percentage of
critical scenarios verified. Check per-test contexts for both `accuracy` and
`gemma`, including the full profile IDs used by runtime fixtures. In particular,
inspect the local embedding functions separately from the hosted embedding code.

The supervisor needs a separate subprocess-instrumented run. This opts into
coverage's subprocess patch only for that run; it does not change the repository's
normal pytest configuration. Keep its raw database distinct from the main run.

```sh
mkdir -p build/coverage/local-models/worker-current
.venv/bin/python - <<'PY'
from pathlib import Path
config = Path('pyproject.toml').read_text().replace(
    '[tool.coverage.run]\n', '[tool.coverage.run]\npatch = ["subprocess"]\n')
Path('build/coverage/local-models/worker-coverage.toml').write_text(config)
PY
local_worker_run="$PWD/build/coverage/local-models/worker-current"
COVERAGE_FILE="$local_worker_run/.coverage" COVERAGE_DEBUG=core \
  COVERAGE_DEBUG_FILE="$local_worker_run/core.log" \
  .venv/bin/python -m pytest -q tests/test_local_runtime_worker.py \
  --cov=. --cov-config=build/coverage/local-models/worker-coverage.toml \
  --cov-branch --cov-context=test --cov-report=
COVERAGE_FILE="$local_worker_run/.coverage" .venv/bin/python -m coverage json \
  --include=local_runtime_worker.py -o "$local_worker_run/coverage.json"
COVERAGE_FILE="$local_worker_run/.coverage" .venv/bin/python -m coverage html \
  --include=local_runtime_worker.py -d "$local_worker_run/html"
```

Verify that the worker report contains executed supervisor lines; a passing parent
test alone is insufficient. These subprocess hits provide aggregate coverage,
not pytest per-test attribution inside the child. The separate run checks closed
parent pipes, termination signals, forced kill of an unresponsive server and
natural server exit. Dashboard MCP children remain outside the main tracer; their
end-to-end behavior is checked through persisted state and process-exit assertions.

To include the measured supervisor in the 25-module report while retaining both
raw inputs:

```sh
local_combined_run="$PWD/build/coverage/local-models/combined-current"
mkdir -p "$local_combined_run"
COVERAGE_FILE="$local_combined_run/.coverage" .venv/bin/python -m coverage combine \
  --keep "$local_coverage_run/.coverage" "$local_worker_run/.coverage"
COVERAGE_FILE="$local_combined_run/.coverage" .venv/bin/python -m coverage json \
  --include="$local_coverage_files" -o "$local_combined_run/coverage.json"
COVERAGE_FILE="$local_combined_run/.coverage" .venv/bin/python -m coverage html \
  --include="$local_coverage_files" -d "$local_combined_run/html"
```

### Audit recorded on 2026-11-10

The initial 551-test selection passed. After extending the tests, 776 regression
cases and four additional failure-path cases passed, for 780 distinct cases. The
five supervisor cases also passed in a separate subprocess-instrumented rerun.
Type checking and `git diff --check` passed. Warnings remain, including SQLite
resource warnings also present in the baseline.

Run-phase context checks confirmed 31 scenario groups for both Qwen accuracy and
Gemma, covering the boundaries above. Existing legacy-Qwen checks were retained.
The comparison used identical source files, exclusions, branch settings and
CTracer instrumentation, with the supervisor measured separately on both sides.
No previously executed production lines or branches were lost.

| Component | Line coverage | Branch coverage |
| --- | ---: | ---: |
| Model profile definitions | 100.0% | 100.0% |
| Local chat transport and provider | 99.2% | 93.9% |
| Local embedding request/batching function | 100.0% | 100.0% |
| Native-vector reduction/validation function | 96.8% | 93.8% |
| Managed runtime | 95.6% | 88.1% |
| Runtime supervisor | 100.0% | 100.0% |
| All 25 selected modules | 70.1% | 56.7% |

The full selected scope increased from 69.6% lines / 56.0% branches to
70.1% / 56.7%, adding 35 executed lines and 14 branches. The main improvement is
model-specific assertion coverage: Qwen and Gemma share much of their execution
code, so adding a missing Gemma scenario need not increase the aggregate percentage.
Remaining misses include platform checks, archive error guards, invalid internal
retry budgets and a defensive reduced-vector norm check. This is not a claim that
every branch of the application is tested.

Local evidence is retained under `build/coverage/local-models/`: `audit.json`
records exact test contexts, file hashes and comparisons; `verified/html/index.html`
and `verified/coverage.json` contain the combined report; the individual `before`,
`after`, `after-guards`, `worker-before` and `worker-after` directories retain raw
data and logs. Generated reports remain ignored by Git.

## Complete local inference scenario

`tests/test_local_e2e.py` runs one shared scenario against deterministic HTTP
servers and, optionally, real local model servers. It generates a document,
embeds and stores it alongside compiler routes, closes and reopens the library,
embeds a query and retrieves the stored document, then calls the production Ask
service for obligation decomposition, evidence selection and final synthesis.
Assertions check persisted normalized 3072-dimensional vectors, the actual query
embedding request, all generation phases, resolved obligations, expanded source
evidence and a nonempty source-referencing answer. They do not compare prose to
a golden answer or score factual accuracy.

The scenario explicitly uses `ask_pipeline: compact`; it does not test the default
cascade pipeline, the browser, or the production SCIP indexer. Its compiler graph
and source files are controlled fixtures. HTTP-fixture cases cover the legacy Qwen
9B, Qwen 27B thinking, Gemma 12B and BF16/vLLM profiles, each paired with its
configured Qwen embedding model. EmbeddingGemma is not a supported profile.

Run the deterministic cases with:

```sh
.venv/bin/python -m pytest -q tests/test_local_e2e.py -k 'not live'
```

For real inference, start each desired model pair and supply its Ariadne YAML
configuration. Both endpoints must use `127.0.0.1`; the test blocks requests to
other hosts. The same assertions run without scripted model responses:

```sh
ARIADNE_LIVE_MODEL_CONFIGS='["/absolute/path/to/servers.yaml"]' \
ARIADNE_E2E_REPORT_DIR="$PWD/build/local-e2e-live/my-new-run" \
  .venv/bin/python -m pytest -q tests/test_local_e2e.py -k live
```

Use a fresh report directory for each run. Multiple configurations can be supplied
in the JSON array; run large model pairs sequentially to limit resident memory.
The report retains stage results and generation traces, including returned text
and errors, even when an assertion fails. Without the environment variable, the
live case is skipped. The test does not download weights or manage server processes. Real BF16
qualification additionally needs the hardware required by its vLLM deployment;
passing its HTTP-fixture case does not establish real BF16 execution.

### Complete-flow results recorded on 2026-11-10

The four HTTP-fixture cases passed. Together with `tests/test_compact_path.py`,
the regression run passed 19 tests and skipped the unconfigured live case. Type
checking and `git diff --check` passed. These new tests are not included in the
earlier coverage-percentage snapshot above.

Real native runs used the pinned Q4 artifacts and llama.cpp b11669 on this Intel
Mac's CPU, with endpoint timeouts extended to 3600 seconds. They are functional
checks, not M5 performance qualification. All launched servers were stopped.

| Profile, paired with Qwen embeddings | Real complete-flow result |
| --- | --- |
| Gemma 12B Q4 | Passed, including the explicit query-embedding request assertion; 170.40 seconds in pytest |
| Legacy Qwen 9B Q4 | Failed decomposition: returned `CO1:`/`CO2:`/`CO3:` instead of parseable `C1:` labels; no obligations resolved |
| Qwen 27B Q4 thinking | Failed evidence selection: exhausted its 2112-token budget with `finish_reason='length'`; the service returned a document-only fallback, which the complete-flow assertion rejected |
| BF16/vLLM | Not run with real weights here; its pinned deployment requires NVIDIA GPUs |

The legacy failure is retained under
`build/local-e2e-live/qwen-mac-q4-v1/`. The final Qwen 27B and Gemma runs are under
`build/local-e2e-live/20261110-201504/`. Each profile directory contains `run.json`,
`pytest.log`, server logs and `scenario/<profile-hash>/result.json`. The Qwen 27B
pytest run took 2215.16 seconds. Earlier runs, including a launcher port-reuse
failure before Qwen 27B started, remain separate from these final results.
At that checkpoint, no production behavior had been changed to make these live
checks pass; both Qwen failures were unresolved.

The legacy 9B profile was rerun after the user freed memory, with per-stage timers
added to the shared test and memory sampling in the local launcher. Evidence is
under `build/local-e2e-live/20261110-224230/qwen-mac-q4-v1/`. Route embedding took
0.914 seconds, document generation 34.376 seconds, document embedding/storage
7.682 seconds, library reopening 0.009 seconds and query embedding/search 1.449
seconds. Ask took 26.578 seconds and stopped at decomposition; its model call took
22.009 seconds according to the server log. The returned labels were `CO-1:` and
`CO-2:`, so no obligations parsed. Selection and synthesis were not reached.
The full launcher run took 102.53 seconds including verification/startup/shutdown.

Swap usage and cumulative swap-in/out counters remained zero; the compression
counter did not increase. Peak sampled combined server RSS was 18.55 GiB. RSS is
a process-residency measurement, not a prediction of total M5 unified-memory use.
This rerun reproduced the output-format failure with memory available. The timing
instrumentation changes neither prompts nor production behavior.

### Qwen 9B decomposition investigation

The original compact planner specified the abstract format `C<n>` without spelling
out literal labels. Complete, untruncated 9B responses repeatedly invented `CO1:`
or `CO-1:` labels, which the production parser correctly rejected. The compact
planner now explicitly lists `C1:` through `C5:`, requires capital C immediately
followed by the number and colon, and prohibits alternative label formatting.
Question-scoping instructions, the parser, model settings and token limits remain
unchanged. This is a prompt clarification, not constrained decoding or a guarantee
that every future response will follow the format.

An isolated real-model comparison used three question shapes (the payment
workflow, a single return-value question, and a comparison), plus two additional
payment-question calls with the clarified prompt. The original prompt produced
valid labels in 2/3 calls, again failing the payment case; the clarified prompt
produced valid labels in 5/5 calls. Full request/response traces are retained in
`build/local-e2e-live/20261110-224849/qwen-mac-q4-v1/scenario/prompt-probe.json`.
These counts measure label conformance, not factual quality. An earlier diagnostic
setup attempt in `20261110-224743` failed library binding before any model calls;
it is excluded from these counts.

Regression cases retain rejection of both observed malformed label variants.
The affected prompt, compact-path, local E2E and remote HTTP-fixture tests passed
31 cases, with the unconfigured live case skipped. Type checking also passed.

The clarified prompt then passed the unchanged complete live E2E assertions with
both Qwen 9B and Gemma, using Qwen embeddings. Evidence is under
`build/local-e2e-live/20261110-225256/`. Both runs reached all four generation
phases, resolved their obligations, expanded the source evidence and produced a
final answer. These are functional checks; they do not certify factual accuracy.

| Stage | Qwen 9B | Gemma 12B |
| --- | ---: | ---: |
| Embed routes | 0.893 s | 0.902 s |
| Generate document | 27.972 s | 29.840 s |
| Embed and store document | 6.086 s | 2.417 s |
| Reopen library | 0.008 s | 0.008 s |
| Embed query and search | 1.492 s | 1.472 s |
| Complete Ask | 184.210 s | 176.441 s |
| Pytest duration | 221.37 s | 211.83 s |
| Launcher duration including verification/startup/shutdown | 251.06 s | 244.22 s |

All launched servers were stopped. These single-run CPU timings vary with output
length and machine conditions; they are not a controlled performance comparison.
The earlier 27B thinking-profile selection-budget failure was not addressed or
rerun in this investigation.
