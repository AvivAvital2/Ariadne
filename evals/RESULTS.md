# Findings

Two batteries exist: the **development battery**, run against a Databricks
spool (Spark + Delta + databricks-sdk-py pinned to DBR 17.3 LTS) and two
internal consumer repos while the routing layer was built — its consumer
repos are private, so it ships as reported numbers with methodology — and
the **public battery** in this directory (httpx mini-environment +
synthetic consumers), which anyone can rebuild and rerun. The public
battery's numbers belong in this file the first time
`build_eval_store.sh` + `run_battery.py` are run on a release; until then
this section reports the development results only.

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

## The honest finding: training saturation

An answer-level A/B (bare frontier models, zero tools, vs the full
product) on version-drift questions about Spark/Delta scored **8/8 for
the bare models** — that corpus is training-saturated down to `@Since`
sub-versions. So for famous environments this system does **not** claim
"knows more than the model." The measured claim is different:

- the product's 8/8 answers are **pinned and cited** — every version fact
  traces to a corpus SHA, not to training-data recall that silently goes
  stale;
- the anti-gap-fill property holds: the ground truth is *in the context*,
  so the model doesn't confabulate around your pinned versions;
- the knowledge delta lives where training is thin — post-cutoff
  releases, internal platforms, niche standards. That is untested here by
  construction, and this file will say so until someone measures it.

## Reading the numbers

Participation and junk are the contract ("decisive at the seam, quiet
away from it"); ground-truth-in-context is the value; neither is an
answer-quality score. LLM-judged answer quality is deliberately out of
scope for the offline battery — it would reintroduce a paid,
non-deterministic judge into a harness whose job is to be free, offline,
and diffable in CI.
