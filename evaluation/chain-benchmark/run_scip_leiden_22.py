#!/usr/bin/env python3
"""Run the standalone SCIP-Leiden experiment over the established 22 questions.

This wrapper deliberately delegates every question to ``scip_leiden_live.py``.
It does not implement retrieval, selection, materialization, or scoring itself.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNNER = HERE / "scip_leiden_live.py"
QUESTION_IDS = (4, 6, 8, 10, 12, 13, 15, 16, 17, 23, 27, 31, 62, 66, 67,
                84, 87, 88, 107, 147, 156, 187)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new directory for per-question and aggregate artifacts")
    parser.add_argument("--db", type=Path, default=ROOT / "ariadne.db")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "evaluation/spool-clean-room/questions_debcrumb_ask.json")
    parser.add_argument("--gold", type=Path, default=HERE / "gold-chain-reviewed.json")
    parser.add_argument("--corpus", type=Path, default=ROOT / "spool-corpus")
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--card-k", type=int, default=12)
    parser.add_argument("--owner-bridge-k", type=int, default=1)
    parser.add_argument("--selector-max-tokens", type=int, default=256)
    parser.add_argument("--formulation-max-tokens", type=int, default=512)
    return parser


def _command(args: argparse.Namespace, question_id: int, answers: Path, report: Path) -> list[str]:
    return [
        sys.executable, "-u", str(RUNNER),
        "--families", str(args.families), "--facts", str(args.facts),
        "--db", str(args.db), "--questions", str(args.questions),
        "--gold", str(args.gold), "--corpus", str(args.corpus),
        "--source", args.source, "--question-id", str(question_id),
        "--card-k", str(args.card_k), "--owner-bridge-k", str(args.owner_bridge_k),
        "--selector-max-tokens", str(args.selector_max_tokens),
        "--formulation-max-tokens", str(args.formulation_max_tokens),
        "--answers", str(answers), "--report", str(report),
    ]


def _summary(out_dir: Path, failures: dict[int, int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(out_dir.glob("q*-report.json")):
        report = json.loads(report_path.read_text())
        score = report["score"]["summary"]
        rows.append({
            "id": int(report["question_id"]),
            "passed_questions": int(score["passed_questions"]),
            "passed_claims": int(score["passed_claims"]),
            "claims": int(score["claims"]),
            "total_cost_usd": float(score["total_cost_usd"]),
            "total_elapsed_s": float(score["total_elapsed_s"]),
        })
    return {
        "schema": "scip-leiden-live22/v1",
        "question_ids": list(QUESTION_IDS),
        "completed": rows,
        "operational_failures": sorted(failures),
        "passed_questions": sum(row["passed_questions"] for row in rows),
        "questions_scored": len(rows),
        "passed_claims": sum(row["passed_claims"] for row in rows),
        "claims": sum(row["claims"] for row in rows),
        "total_cost_usd": round(sum(row["total_cost_usd"] for row in rows), 6),
        "total_elapsed_s": round(sum(row["total_elapsed_s"] for row in rows), 1),
    }


def main() -> int:
    args = _parser().parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _parser().error("ANTHROPIC_API_KEY must be set before starting the batch")
    required = (RUNNER, args.families, args.facts, args.db, args.questions, args.gold)
    if not all(path.is_file() for path in required):
        _parser().error("runner, stores, database, questions, and gold must exist")
    if not args.corpus.is_dir():
        _parser().error("--corpus must be a directory")
    if args.out_dir.exists():
        _parser().error("--out-dir must not already exist")
    args.out_dir.mkdir(parents=True)

    failures: dict[int, int] = {}
    for index, question_id in enumerate(QUESTION_IDS, start=1):
        answers = args.out_dir / f"q{question_id}-answers.json"
        report = args.out_dir / f"q{question_id}-report.json"
        print(f"[{index}/{len(QUESTION_IDS)}] Q{question_id} starting", flush=True)
        completed = subprocess.run(_command(args, question_id, answers, report), check=False)
        # The delegated runner returns 1 for a scored claim failure.  A report
        # means that result is valid benchmark data, not an operational error.
        if completed.returncode and not report.is_file():
            failures[question_id] = completed.returncode
            print(f"[{index}/{len(QUESTION_IDS)}] Q{question_id} operational failure "
                  f"(exit {completed.returncode})", flush=True)

    summary = _summary(args.out_dir, failures)
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"SCIP-Leiden 22: {summary['passed_questions']}/{summary['questions_scored']} "
        f"questions; {summary['passed_claims']}/{summary['claims']} claims; "
        f"${summary['total_cost_usd']:.4f}; {summary['total_elapsed_s']:.1f}s",
        flush=True)
    if failures:
        print("Operational failures: " + ", ".join(map(str, sorted(failures))), flush=True)
    print(f"Artifacts: {args.out_dir}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
