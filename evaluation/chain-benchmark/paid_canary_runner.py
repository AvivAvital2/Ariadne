#!/usr/bin/env python3
"""User-run paid canary wrapper: sequential, budgeted, checkpointed.

The framework prepares the command; a human runs this file. It executes
questions one at a time at concurrency 1, checkpoints an answers file
per question, accumulates actual token cost from each question's usage
telemetry, and refuses to START a question whose certified worst-case
cost exceeds the remaining budget. It stops AFTER the current question
if the budget is exceeded. It requires the same authorization the
preparer validated: a fresh nonce and a passed canary certificate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def question_cost(answers_path: Path, prices: dict, model: str) -> float:
    rows = json.loads(answers_path.read_text())
    tariff = prices["models"][model]
    total = 0.0
    for row in rows:
        for usage in row.get("usage_rows", ()):
            total += (int(usage.get("input_tokens") or 0)
                      * tariff["input_per_mtok"] / 1e6)
            total += (int(usage.get("output_tokens") or 0)
                      * tariff["output_per_mtok"] / 1e6)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", required=True)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--price-config", required=True)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--answers", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--trace-dir", default="")
    args = parser.parse_args()

    if not os.environ.get("ARIADNE_PAID_RUN_NONCE"):
        print("REFUSED: ARIADNE_PAID_RUN_NONCE is not set")
        return 2
    from exp_certificate import load_certificate
    certificate = load_certificate(args.certificate)
    if (certificate.get("type") != "paid-canary-eligibility"
            or not certificate.get("passed")):
        print("REFUSED: certificate cannot authorize a paid canary")
        return 2
    prices = json.loads(Path(args.price_config).read_text())
    worst_case = float(
        certificate["limits"]["max_usd_per_question"])

    questions = [int(part) for part in args.only.split(",") if part]
    spent = 0.0
    for index, question_id in enumerate(questions, start=1):
        remaining = args.max_usd - spent
        if worst_case > remaining:
            print(f"BUDGET STOP before q{question_id}: certified "
                  f"worst-case ${worst_case:.2f} exceeds remaining "
                  f"${remaining:.2f} (spent ${spent:.2f})")
            return 3
        answers = Path(args.answers or f"paid-canary-q{question_id}.json")
        if answers.exists() and not args.answers:
            print(f"skip q{question_id}: checkpoint exists at {answers}")
            continue
        command = [
            sys.executable, "-u", str(HERE / "measure_ariadne.py"),
            "--only", str(question_id), "--concurrency", "1",
            "--source", args.source,
            "--answers", str(answers),
        ]
        if args.report:
            command += ["--report",
                        f"{args.report}.q{question_id}.json"]
        if args.trace_dir:
            command += ["--trace-dir", args.trace_dir]
        print(f"[{index}/{len(questions)}] q{question_id}: running "
              f"(spent ${spent:.2f} of ${args.max_usd:.2f})", flush=True)
        code = subprocess.call(command)
        if code != 0:
            print(f"q{question_id} exited {code}; stopping")
            return code
        cost = question_cost(answers, prices, args.model)
        spent += cost
        print(f"q{question_id}: actual ${cost:.3f}; cumulative "
              f"${spent:.3f}", flush=True)
        if spent > args.max_usd:
            print(f"BUDGET EXCEEDED after q{question_id}: "
                  f"${spent:.2f} > ${args.max_usd:.2f}; stopping")
            return 3
    print(f"canary complete: ${spent:.3f} across "
          f"{len(questions)} question(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
