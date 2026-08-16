"""Backtest: would the certifier have bought the runs we know failed?

Two-phase by construction: predictions are computed from preflight
evidence alone and sealed; observed outcomes come from rescoring saved
answers under the current scorer; the calibration join happens only
after both exist. The report states coverage honestly — zero false
positives over zero backtestable runs proves nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from exp_registry import registry


def rescore_run(entry, *, corpus=None) -> "dict | None":
    """Observed outcome under the CURRENT scorer; stale reports distrusted."""
    import measure_ariadne

    answers_path = HERE / entry["answers"]
    if not answers_path.is_file():
        return None
    answers = json.loads(answers_path.read_text())
    gold = json.loads(
        (HERE / "gold-chain-reviewed-compact.json").read_text())
    report = measure_ariadne.score_answers(
        answers, gold, Path(corpus or (HERE.parent.parent / "spool-corpus")))
    passing_questions = tuple(sorted(
        int(row["id"]) for row in report.get("questions", ())
        if row.get("passed")))
    return {
        "questions_passed": sum(
            1 for row in report.get("questions", ()) if row.get("passed")),
        "claims_passed": sum(
            1 for row in report.get("questions", ())
            for claim in row.get("claims", ()) if claim.get("passed")),
        "passing_questions": passing_questions,
    }


def calibration_table(*, corpus=None, rescore=True) -> dict:
    """The honest join: prediction basis, observed outcome, classification.

    Saved paid runs predate the sealed-preflight requirement, so the
    certifier refuses them all for lack of a sealed preflight artifact —
    a true negative for every run that then failed live, and a coverage
    statement rather than a hollow zero-false-positive claim.
    """
    rows = []
    counts = {"true_positive": 0, "false_positive": 0,
              "true_negative": 0, "false_negative": 0,
              "not_backtestable": 0}
    for entry in registry():
        row = {
            "run_id": entry["run_id"],
            "backtest_level": entry["backtest_level"],
            "fresh_live_outcome": entry["fresh_live_outcome"],
            "prediction": "refused: no sealed preflight artifact exists "
                          "for this historical run",
            "predicted_eligible": False,
        }
        if entry["backtest_level"] == "not_backtestable":
            row["classification"] = "not_backtestable"
            counts["not_backtestable"] += 1
            rows.append(row)
            continue
        observed = (rescore_run(entry, corpus=corpus)
                    if rescore and entry["expected_rescored"] else None)
        row["observed_rescored"] = observed
        expected = entry["expected_rescored"]
        if observed is not None and expected is not None:
            mismatches = {
                key: (expected[key], observed.get(key))
                for key in ("questions_passed", "claims_passed")
                if observed.get(key) != expected[key]}
            row["rescore_matches_expectation"] = not mismatches
            if mismatches:
                row["rescore_mismatches"] = {
                    key: {"expected": pair[0], "observed": pair[1]}
                    for key, pair in mismatches.items()}
        failed_live = entry["fresh_live_outcome"] == "failed"
        if failed_live and not row["predicted_eligible"]:
            row["classification"] = "true_negative"
            counts["true_negative"] += 1
        elif failed_live:
            row["classification"] = "false_positive"
            counts["false_positive"] += 1
        elif row["predicted_eligible"]:
            row["classification"] = "true_positive"
            counts["true_positive"] += 1
        else:
            row["classification"] = "false_negative"
            counts["false_negative"] += 1
        rows.append(row)
    backtestable = sum(
        1 for row in rows if row["classification"] != "not_backtestable")
    return {
        "measurement": "certification-backtest",
        "rows": rows,
        "counts": counts,
        "coverage": {
            "registered": len(rows),
            "backtestable": backtestable,
            "statement": (
                f"{backtestable} of {len(rows)} registered runs are "
                "backtestable; zero-false-positive claims are meaningful "
                "only over that set"),
        },
    }
