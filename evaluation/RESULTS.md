# Findings

Two batteries exist, both against the **same environment** — a Databricks
spool (Spark + Delta + databricks-sdk-py pinned to DBR 17.3 LTS, built
from the shipped recipe): the **development battery**, run against two
internal consumer repos while the routing layer was built — its consumer
repos are private, so it ships as reported numbers with methodology — and
the **public battery** in this directory, whose synthetic archetype
consumers anyone can onboard and rerun (and whose environment anyone can
rebuild from the recipe's pinned SHAs). Sharing the environment makes the
two batteries' numbers directly comparable. The public battery's numbers
belong in this file the first time `build_eval_store.sh` +
`run_battery.py` are run on a release; until then this section reports
the development results only.

## The archetype model

Consumer repos differ in how much of their catalog is grounded in the
environment, and that spectrum — not any per-question tuning — is what the
battery samples:

- **Adopter** — no environment calls at all; migrating toward it. Seam
  questions are the most environment-dependent.
- **Peripheral** — some usage at the edges, with prose/vocabulary
  saturated by the environment's name. This archetype exposed
  name-contaminated routing: the environment's own name resolving as
  strong *consumer* evidence and flipping seam questions the wrong way.
- **Integrated** — the environment is part of the product; code-graph
  reach should lead. (The public battery covers its retrieval rows;
  reach-tier rows require a dependency-indexed consumer artifact and are
  not yet exercised — an honest gap, not an oversight.)

## Public battery results (2026-07-28)

Store: pack v1.5.0 (Spark 4.0.0 + Delta 4.0.0 + databricks-sdk-py 0.121.0
at the recipe's pinned SHAs; 199,260 environment docs) + the three fixture
consumers (70 docs). Query vectors are committed (`query_vectors.npz`), so
this table reproduces offline with `uv run python evaluation/run_battery.py`.

```
label                      archetype   want    topS   topR   breadth  GT    verdict
P-seam-serialization       peripheral  speak   0.556  0.482   16/50   #3    speak OK
P-target-concurrency       peripheral  speak   0.574  0.327   50/50   #1    speak OK
P-version-liquid           peripheral  speak   0.616  0.228   50/50   #1    speak OK
P-control-saturated        peripheral  silent  0.398  0.466    0/50   -     silent OK
P-control-domain           peripheral  silent  0.363  0.567    0/50   -     silent OK
P-nonsense                 peripheral  silent  0.296  0.159   50/50   -     silent OK
A-equivalence              adopter     speak   0.579  0.443   50/50   #1    speak OK
A-migration                adopter     speak   0.523  0.557    0/50   #4    silent MISS
A-control-saturated        adopter     silent  0.497  0.444    6/50   -     silent OK
I-seam-broadcast           integrated  speak   0.535  0.308   50/50   #1    speak OK
I-target-vacuum            integrated  speak   0.473  0.319   50/50   #1    speak OK
I-control-domain           integrated  silent  0.512  0.405   11/50   -     silent OK

participation: 11/12 correct   ground-truth-in-context: 7/7   junk admissions: 0
```

Reading the edges honestly:

- **The miss (A-migration)** is the adopter-seam hard case: the
  ground-truth doc ranks #4 on the environment side, but the consumer's
  own pipeline docs outscore the environment's window (topR 0.557 >
  topS 0.523), so the breadth criterion stays silent. The answer doc was
  *found*; participation suppressed it. This is the same residual class
  the development battery carries (its discovery-question cell) — a
  known open cell, not a surprise.
- **P-nonsense shows the junk floor working**: breadth alone would speak
  (50/50 docs beat the repo's 0.159 on an off-domain question), but the
  environment's best cosine (0.296) sits under the 0.35 floor, so it
  stays silent. Breadth measures *relative* strength; the floor catches
  the case where both sides are weak.
- **I-control-domain sits 11/50** — one doc short of speaking. Saturated
  vocabulary ("compacted" is environment language) keeps this class the
  thinnest margin in the battery, matching the development battery's
  accepted-noise finding.

### Theme coherence (same store)

A second question the same store answers: does Leiden theme discovery hold
up at scale, or degrade into junk clusters? `run_battery.py` reads the
coherence gate's verdict straight off the store:

```
theme coherence gate: 1900/2092 clusters coherent (90.8%)   192 incoherent (hidden by default)
```

Every Leiden community is handed to the LLM summarizer, which returns
`INCOHERENT` for a cluster with no shared concern (a vendored JS bundle, a
wall of generated stubs); those are flagged `coherent=0` and hidden from
`ariadne themes` and search by default. Over this store — the Databricks
environment (Spark + Delta + SDK) plus the three fixture consumers, 2,092
clusters — **90.8% pass the gate**, and the 192 rejected are exactly the
noise the gate exists to catch. It reproduces offline from the built store
(no LLM, no embeddings — it reads the stored flag) with
`uv run python evaluation/run_battery.py`; measure the same split on your own
corpus with `ariadne themes stats`.

## Development battery results (Databricks spool, 2026-07)

- **Participation:** 18/20 correct on the strict archetype matrix
  (speak/silent + primary side). The two misses are labeled, bounded
  noise on saturated controls — the environment speaks where it should
  stay quiet, with its docs clearly attributed (annoyance, not
  contamination).
- **Ground-truth-in-context:** 8/9 on real-pipeline seam questions; junk
  admissions 0.
- **Sourcing ladder** (same question set, growing as failure classes were
  found): no environment pack 0/13 → shipped baseline 8/13 → routing
  package 9/13 → provenance rows 11/13. Each step is a mechanism from
  design (name-blind dominance, breadth-with-floor participation, version
  facts, provenance rows) validated against the battery — never a
  per-question fix.

## Scope of this battery

This development battery measures routing, participation, and whether known
ground truth reached context. It does not support a conclusion about model
training saturation. The earlier bare-model A/B result is historical and is not
a current Ariadne-versus-bare-LLM finding.

For the reviewed compiler-aware comparison and its stated limits, see the
[compiler-aware comparison](chain-benchmark/COMPILER_AWARE_COMPARISON.md).

## Reading the numbers

Participation and junk are the contract ("decisive at the seam, quiet away
from it"). Ground-truth-in-context records whether the known material reached
context. The offline battery remains free and diffable in CI.

## Impact-analysis accuracy (E5) — pilot, 2026-07-29

First measurement of `impact_radius`'s "affected tests" (via `find_tests_for`
— import-graph edges + a `test_<name>` naming heuristic) against an
INDEPENDENT oracle: which test files stop collecting when a module is made
unimportable — their real transitive importers. On a mid-size Python repo
(~1,100 tests across 108 files), 6 source files. Offline / $0 / mutation-free
(a pytest meta-path plugin breaks the target import; the repo is never
written). Harness: `evaluation/impact_pilot.py`.

```
source file             predicted  observed  recall  precision
core data-type module        2        108      0.02     1.00
core database module         1        108      0.01     1.00
core DuckDB-IO module        0        108      0.00     1.00
core formatter base          6        108      0.04     0.67
scoring: feature             8          3      0.00     0.00
scoring: significance        0          3      0.00     1.00
── micro-avg (7 matched / 17 predicted / 438 observed) ──  recall 0.016  precision 0.412
```

Read honestly:

- **The recall gap is real, and matches the tool's own caveat.** `find_tests_for`
  resolves only DIRECT import edges; it does not compute the transitive import
  closure. The package `__init__` and shared conftests pull the core modules
  into effectively every test, so breaking one cascades to all 108 files — the
  tool reports ~2. `impact_radius`'s docstring already calls its transitive
  count "a conservative file-level lower bound"; this quantifies it: ~2% of the
  true blast radius for widely-imported files. Safe as a lower bound; it badly
  under-warns on core files.
- **The naming heuristic costs precision.** The scoring "feature" module: 8
  predicted, none among the 3 that broke — `test_*feature*` name-matches that
  don't import it.
- **What this is NOT.** The oracle measures import/removal impact ("what breaks
  if this module changes or goes away") — a superset of the tool's direct-call
  model — so part of the gap is definitional (direct vs transitive). n=6, one
  repo: directional, not a verdict.
- **The lever:** transitive-closure resolution moves recall most; tightening or
  dropping the naming heuristic moves precision.

### Fix + re-measure (2026-07-29)

Root cause was deeper than "direct vs transitive": `doc_graph`'s `imports`
edges were **empty library-wide** (the AST graph-build step was never wired
into onboard), so the affected-tests path had silently degraded to a
non-source-scoped filename heuristic. Fix: route it through the **SCIP call
graph** (the same edges the file-side impact already uses), source-scoped.
Re-measured on the same 6 files:

```
                        recall (before→after)   precision (before→after)
core data-type module        0.02 → 0.31             1.00 → 0.73
core database module         0.01 → 0.39             1.00 → 0.76
core DuckDB-IO module         0.00 → 0.12             1.00 → 0.72
core formatter base           0.04 → 0.02             0.67 → 0.67
scoring: feature              0.00 → 0.00             0.00 → 1.00
scoring: significance         0.00 → 0.00             1.00 → 1.00
── micro-average ──          0.016 → 0.205           0.412 → 0.744
```

- **Recall ~13× (1.6% → 20.5%), precision 41% → 74%.** Cross-source naming
  false positives are gone (source-scoped SCIP); heavily-called core files
  gain real recall.
- **The residual is an indexer limit, not plumbing.** SCIP models *calls*, the
  oracle measures *imports*, and scip-python resolves callers for only ~7% of
  this corpus's symbols — so leaf files with no caller edges stay at 0, and the
  core files miss import-only / fixture dependents. Closing the rest needs
  better call-edge coverage (or import edges), logged as follow-up. The eval
  did its job twice: found the gap, then proved the fix.

### Call-based oracle cross-check (2026-07-29)

`find_tests_for` predicts CALL-based impact (SCIP call edges), so the fairer
oracle should also be call-based, not import-based. A second oracle
(`impact_pilot.py --mode call`, plugin `e5call_plugin.py`) makes the target
module's own callables RAISE WHEN EXERCISED — classes get a raising `__init__`,
functions a raising stub — then runs the full suite; a test fails iff it
actually calls into the module. Still $0 / mutation-free (the source is never
written). It sharpened the picture rather than lifting the grade:

- **It cannot measure the most-connected modules — a limit of the *concept*,
  not the implementation.** You can't make a module's symbols "raise only when
  called at runtime" if its public surface includes enums, base classes, or
  constants that are consumed at *import* time. Concretely: a member-less stub
  replacing an enum is read by a downstream class body (`SomeEnum.MEMBER` in a
  class definition) during import → `AttributeError` → `conftest` ImportError →
  the whole pytest session aborts before any test runs. No per-file failures are
  emitted, so the two most heavily-imported files register `observed 0` — impact
  so total the per-file oracle can't see it. These are exactly the files the
  import-oracle measures best (recall 0.31 / 0.39).
- **The naive 6-file aggregate (recall 0.19, precision 0.058) is an artifact**
  of those two degenerate rows dumping ~100 predicted files into the denominator
  with 0 matched. The fair grade is the micro-average over the 4 measurable
  files: **recall 0.19, precision 0.33.**
- **Where it can measure, it agrees on the ceiling.** Recall (0.19) matches the
  import-oracle's 0.20; precision is *lower* (0.33 vs 0.74) because SCIP's static
  call edges over-predict what a test actually executes at runtime — predicted
  callers that never exercise the call count against precision, where the
  import-oracle's broader "importers" set absorbed them. The two leaf files stay
  at recall 0: scip-python resolved zero caller edges for them, the same limit
  the import pass hit.

Bottom line: the call-oracle is the conceptually-right target but a blunter
instrument, and it confirms from a second angle that the recall ceiling is
scip-python's call-edge coverage (~7% of symbols) — the logged follow-up. The
import-oracle stays the headline (recall 0.20 / precision 0.74) because it is
the measurement that reaches the core modules. Neither oracle flatters the
tool; both point to the same lever.
