#!/usr/bin/env python3
"""Rescore saved benchmark answers with the CURRENT scorer. No API calls.

Saved live reports were written by whichever scorer ran that day; scoring
rules have since been corrected (e.g. short quoted lines were once
silently excluded). This tool regrades a saved answers file against the
reviewed gold using today's scorer, so a stale on-disk report is never
mistaken for the baseline. The output is stamped with the measurement
name, input paths, and the gold hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import measure_ariadne as reviewed_measure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument(
        "--gold", default=str(HERE / "gold-chain-reviewed-compact.json"))
    parser.add_argument("--corpus", default=str(ROOT / "spool-corpus"))
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    answers = json.loads(Path(args.answers).read_text())
    gold_text = Path(args.gold).read_text()
    only = (
        {int(value) for value in args.only.split(",")}
        if args.only else None)
    report = reviewed_measure.score_answers(
        answers, json.loads(gold_text), Path(args.corpus), only=only)
    payload = {
        "measurement": "rescore-current-scorer",
        "answers_file": str(args.answers),
        "gold_file": str(args.gold),
        "gold_sha256": hashlib.sha256(gold_text.encode()).hexdigest(),
        **report,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    reviewed_measure._print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
