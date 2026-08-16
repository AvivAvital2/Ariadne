"""Registry of saved paid runs: what exists, and what it can calibrate.

Every saved run is registered with an honest backtest level — an
outcome-only artifact must never be silently relabeled as
preflight-reconstructable, and saved reports are never trusted without a
rescore under the current scorer.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

BACKTEST_LEVELS = (
    "full_trace", "answer_and_diagnostics", "outcome_only",
    "not_backtestable")

SAVED_RUNS = (
    {
        "run_id": "live22-diagnostic",
        "answers": "live22-diagnostic-answers.json",
        "report": "live22-diagnostic-report.json",
        "trace_dir": "live22-diagnostic-answers-traces",
        "questions": (4, 6, 8, 10, 12, 13, 15, 16, 17, 23, 27, 31, 62, 66,
                      67, 84, 87, 88, 107, 147, 156, 187),
        "backtest_level": "full_trace",
        "expected_rescored": {"questions_passed": 2, "claims_passed": 9,
                              "passing_questions": (67, 147)},
        "fresh_live_outcome": "failed",
    },
    {
        "run_id": "handoff-fix",
        "answers": "handoff-fix-live-answers.json",
        "report": "handoff-fix-live-report.json",
        "trace_dir": "handoff-fix-live-answers-traces",
        "questions": (6, 10, 17),
        "backtest_level": "full_trace",
        "expected_rescored": {"questions_passed": 0, "claims_passed": 2},
        "fresh_live_outcome": "failed",
    },
    {
        "run_id": "ariadne-pilot-rest",
        "answers": "ariadne-pilot-rest.json",
        "report": "ariadne-pilot-rest-gate.json",
        "trace_dir": None,
        "questions": (),
        "backtest_level": "outcome_only",
        "expected_rescored": None,
        "fresh_live_outcome": "failed",
    },
    {
        "run_id": "ariadne-pilot-after-scip",
        "answers": "ariadne-pilot-after-scip.json",
        "report": None,
        "trace_dir": None,
        "questions": (),
        "backtest_level": "outcome_only",
        "expected_rescored": None,
        "fresh_live_outcome": "failed",
    },
)


def registry() -> list:
    """Registered runs with on-disk availability resolved honestly."""
    rows = []
    for entry in SAVED_RUNS:
        row = dict(entry)
        answers = HERE / entry["answers"] if entry["answers"] else None
        trace_dir = (HERE / entry["trace_dir"]
                     if entry["trace_dir"] else None)
        row["answers_present"] = bool(answers and answers.is_file())
        row["trace_present"] = bool(trace_dir and trace_dir.is_dir())
        if not row["answers_present"]:
            row["backtest_level"] = "not_backtestable"
        elif entry["backtest_level"] == "full_trace" and not (
                row["trace_present"]):
            row["backtest_level"] = "answer_and_diagnostics"
        rows.append(row)
    return rows
