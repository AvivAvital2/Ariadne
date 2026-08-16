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
this table reproduces offline with `uv run python evals/run_battery.py`.

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
`uv run python evals/run_battery.py`; measure the same split on your own
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
[compiler-aware comparison](../evaluation/chain-benchmark/COMPILER_AWARE_COMPARISON.md).

## Reading the numbers

Participation and junk are the contract ("decisive at the seam, quiet away
from it"). Ground-truth-in-context records whether the known material reached
context. The offline battery remains free and diffable in CI.
