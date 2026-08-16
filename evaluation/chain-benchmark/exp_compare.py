"""Baseline-versus-candidate comparison under the Pareto acceptance rules.

A candidate is accepted in exactly three ways: reviewed claims reach a
later meaningful stage without regression or material cost growth; the
same claims reach the same stages while the workload/cost surface falls
materially; or a structural store blocker is repaired without downstream
regression (asserted by the caller, not inferred here). Raw candidate
growth alone never qualifies, and the recorded failed trajectory (raw up,
final down, bodies +84.6%) must be rejected automatically — that
rejection is a calibration requirement, not an example.
"""
from __future__ import annotations
STAGE_ORDER = (
    "store", "raw", "internal_menu", "retained", "materialized",
    "ledger", "final")

DEFAULT_LIMITS = {
    "body_growth_without_conversion": 0.10,
    "final_prompt_growth_without_conversion": 0.05,
    "material_reduction": 0.10,
}


def claim_frontiers(grade_report: dict) -> dict:
    return {
        (row["question"], row["claim"]): row["frontier"]
        for row in grade_report["claims"]}


def workload(artifacts_payload: dict) -> dict:
    questions = artifacts_payload["questions"]
    return {
        "raw": sum(len(q["raw_pool"]) for q in questions),
        "bodies": sum(len(q["retained"]["body_symbols"]) for q in questions),
        "excerpts": sum(
            len(q["materialized_excerpts"]) for q in questions),
        "ledger_chars": sum(len(q["ledger"]) for q in questions),
        "final_chars": sum(len(q["final_artifact"]) for q in questions),
        "prompt_tokens": sum(
            int(q.get("prompt_tokens_total") or 0) for q in questions),
        "model_calls": sum(
            int(q.get("model_calls") or 0) for q in questions),
    }


def compare(*, baseline_grade: dict, candidate_grade: dict,
            baseline_artifacts: dict, candidate_artifacts: dict,
            limits: dict | None = None) -> dict:
    """Per-claim stage deltas plus an accept/reject decision with reasons."""
    bounds = dict(DEFAULT_LIMITS)
    bounds.update(limits or {})
    base = claim_frontiers(baseline_grade)
    cand = claim_frontiers(candidate_grade)
    if set(base) != set(cand):
        missing = sorted(set(base) - set(cand), key=str)
        added = sorted(set(cand) - set(base), key=str)
        raise ValueError(
            "claim key sets differ between baseline and candidate: "
            f"missing={missing[:3]} added={added[:3]}")
    order = {stage: index for index, stage in enumerate(STAGE_ORDER)}
    order["none"] = -1

    converted = []
    regressed = []
    for key in sorted(base, key=str):
        before = order.get(base[key], -1)
        after = order.get(cand.get(key, "none"), -1)
        if after > before:
            converted.append(
                {"claim": list(key), "from": base[key], "to": cand[key]})
        elif after < before:
            regressed.append(
                {"claim": list(key), "from": base[key],
                 "to": cand.get(key, "none")})

    base_load = workload(baseline_artifacts)
    cand_load = workload(candidate_artifacts)
    growth = {
        key: ((cand_load[key] - base_load[key]) / base_load[key]
              if base_load[key] else 0.0)
        for key in base_load}
    meaningful = [row for row in converted
                  if order[row["to"]] >= order["internal_menu"]]
    reasons = []
    if regressed:
        reasons.append(f"{len(regressed)} claim(s) regressed")
    if growth["bodies"] > bounds["body_growth_without_conversion"] and not any(
            order[row["to"]] >= order["materialized"] for row in converted):
        reasons.append(
            f"bodies grew {growth['bodies']:.1%} without a "
            "materialization-stage conversion")
    if (growth["final_chars"] >
            bounds["final_prompt_growth_without_conversion"]
            and not any(order[row["to"]] >= order["ledger"]
                        for row in converted)):
        reasons.append(
            f"final artifact grew {growth['final_chars']:.1%} without a "
            "ledger/final conversion")
    raw_only = (converted and not meaningful)
    if raw_only:
        reasons.append(
            "only raw-stage conversions; raw growth alone never qualifies")

    reductions = []
    threshold = -bounds["material_reduction"]
    if growth["prompt_tokens"] <= threshold:
        reductions.append(
            f"prompt tokens fell {-growth['prompt_tokens']:.1%} — "
            "cost improvement")
    if cand_load["model_calls"] < base_load["model_calls"]:
        reductions.append(
            f"model calls fell {base_load['model_calls']} -> "
            f"{cand_load['model_calls']} — cost improvement")
    if growth["bodies"] <= threshold or growth["excerpts"] <= threshold:
        reductions.append(
            "materialization surface fell materially — cost improvement")

    accepted = not reasons and bool(meaningful or converted or reductions)
    if accepted:
        reasons = (["pareto conditions met"] if (meaningful or converted)
                   else []) + reductions
    elif not reasons:
        reasons.append(
            "no stage movement and no material workload reduction")

    return {
        "accepted": accepted,
        "reasons": reasons,
        "converted": converted,
        "regressed": regressed,
        "workload_baseline": base_load,
        "workload_candidate": cand_load,
        "workload_growth": growth,
    }
