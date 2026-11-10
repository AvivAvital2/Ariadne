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
