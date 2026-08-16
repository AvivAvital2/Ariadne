#!/usr/bin/env python3
"""One coherent experiment system for the chain benchmark.

Thin dispatch only — the work lives in focused modules:

  offline   gold-blind production-shaped run (shadow_eval run; sealed)
  grade     grade a sealed artifact against reviewed gold (shadow_eval)
  classify  first-loss reach-gap classification (reach_gaps)
  compare   baseline vs candidate under the Pareto rules (exp_compare)
  oracle    downstream preservation invariant (offline_earliest_failure)
  backtest  calibration table over registered saved runs (exp_backtest)
  certify   issue a sealed certificate from preflight evidence only
  paid      PREPARE a paid command; never executes anything itself

The runner is gold-blind; the grader refuses unsealed or tampered
artifacts; comparison rejects raw-only growth automatically; prediction
accepts no outcome data. A free result is never a readiness claim — at
best it makes a paid canary worth its price.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: The only flags measure_ariadne.py actually understands; anything else
#: the preparer must consume itself rather than forward into a command
#: that fails when pasted.
MEASURE_ARIADNE_FLAGS = {
    "--gold", "--questions", "--answers", "--report", "--trace-dir",
    "--source", "--corpus", "--only", "--concurrency", "--resume",
    "--timeout", "--score-only", "--dry-run", "--require-perfect",
}


def forward(script: str, arguments: list) -> int:
    return subprocess.call(
        [sys.executable, str(HERE / script), *arguments])


def prepare_paid(arguments: list) -> int:
    """Validate authorization, then PRINT the command. Never executes."""
    parser = argparse.ArgumentParser(prog="experiment.py paid")
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--max-usd", type=float, default=None)
    parser.add_argument("--only", default="")
    parser.add_argument("--certificate", default="")
    parser.add_argument("--price-config",
                        default=str(HERE / "price-config-v1.json"))
    parser.add_argument("--answers", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--db", default=str(HERE.parent.parent / "ariadne.db"))
    try:
        args = parser.parse_args(arguments)
    except SystemExit:
        return 2

    def refuse(message: str) -> int:
        print(f"REFUSED: {message}")
        return 2

    if not args.allow_paid:
        return refuse("--allow-paid is required")
    if args.max_usd is None or args.max_usd <= 0:
        return refuse("--max-usd must be a positive dollar budget")
    identifiers = [part for part in args.only.split(",") if part.strip()]
    if not identifiers:
        return refuse("--only must name explicit question ids")
    if not all(part.strip().isdigit() for part in identifiers):
        return refuse(f"malformed question ids in --only: {args.only!r}")
    if not os.environ.get("ARIADNE_PAID_RUN_NONCE"):
        return refuse("ARIADNE_PAID_RUN_NONCE is not set; a paid run "
                      "requires a fresh user-provided nonce")
    if not args.certificate:
        return refuse("--certificate is required")
    from exp_certificate import load_certificate
    try:
        certificate = load_certificate(args.certificate)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return refuse(f"certificate rejected: {error}")
    if certificate.get("type") != "paid-canary-eligibility":
        return refuse(
            f"certificate type {certificate.get('type')!r} cannot "
            "authorize a paid canary")
    if not certificate.get("passed"):
        return refuse("certificate did not pass: "
                      + "; ".join(certificate.get("reasons", ())))
    from exp_fingerprint import fast_db_fingerprint
    current = fast_db_fingerprint(args.db)
    certified = certificate.get("database_fingerprint") or {}
    if (current["size"], current["mtime_ns"]) != (
            certified.get("size"), certified.get("mtime_ns")):
        return refuse("database changed since certification; the "
                      "certificate is stale")
    for output in (args.answers, args.report):
        if output and Path(output).exists():
            return refuse(f"output path exists and will not be "
                          f"overwritten: {output}")

    command = [
        sys.executable, "-u", str(HERE / "paid_canary_runner.py"),
        "--only", ",".join(identifiers),
        "--max-usd", str(args.max_usd),
        "--source", args.source,
        "--price-config", args.price_config,
        "--certificate", args.certificate,
    ]
    if args.answers:
        command += ["--answers", args.answers]
    if args.report:
        command += ["--report", args.report]
    if args.trace_dir:
        command += ["--trace-dir", args.trace_dir]
    for flag in command:
        if flag.startswith("--") and flag not in (
                MEASURE_ARIADNE_FLAGS
                | {"--max-usd", "--price-config", "--certificate"}):
            return refuse(f"internal error: unexpected flag {flag}")
    print("Paid command PREPARED (not executed). Run it yourself:")
    print("  " + shlex.join(command))
    print("Reminder: a canary certificate means only 'eligible for a "
          "paid canary', never '22/22 ready'.")
    return 0


def run_certify(arguments: list) -> int:
    parser = argparse.ArgumentParser(prog="experiment.py certify")
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--grade", required=True)
    parser.add_argument("--cost-projection", required=True)
    parser.add_argument("--type", default="paid-canary-eligibility")
    parser.add_argument("--issued-at", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(arguments)

    from exp_certificate import issue, predict_canary_eligibility
    artifacts = json.loads(Path(args.artifacts).read_text())
    grade = json.loads(Path(args.grade).read_text())
    cost = json.loads(Path(args.cost_projection).read_text())
    ceiling = sum(
        1 for row in grade["claims"] if row.get("store"))
    verdict = predict_canary_eligibility(
        grade_report=grade, artifacts_payload=artifacts,
        cost_projection=cost, store_ceiling=ceiling)
    certificate = issue(
        certificate_type=args.type, verdict=verdict,
        grade_report=grade, artifacts_payload=artifacts,
        cost_projection=cost, issued_at=args.issued_at)
    Path(args.out).write_text(json.dumps(
        certificate, indent=1, sort_keys=True))
    print(f"certificate: passed={certificate['passed']}")
    for reason in certificate["reasons"]:
        print(f"  - {reason}")
    return 0 if certificate["passed"] else 1


def run_backtest(arguments: list) -> int:
    parser = argparse.ArgumentParser(prog="experiment.py backtest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-rescore", action="store_true")
    args = parser.parse_args(arguments)

    from exp_backtest import calibration_table
    table = calibration_table(rescore=not args.no_rescore)
    Path(args.out).write_text(json.dumps(table, indent=1, sort_keys=True))
    print(table["coverage"]["statement"])
    for name, count in table["counts"].items():
        print(f"  {name}: {count}")
    return 0 if table["counts"]["false_positive"] == 0 else 1


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    command, arguments = sys.argv[1], sys.argv[2:]
    if command == "offline":
        return forward("shadow_eval.py", ["run", *arguments])
    if command == "grade":
        return forward("shadow_eval.py", ["grade", *arguments])
    if command == "classify":
        return forward("reach_gaps.py", arguments)
    if command == "oracle":
        return forward("offline_earliest_failure.py",
                       ["--selection", "oracle", *arguments])
    if command == "backtest":
        return run_backtest(arguments)
    if command == "certify":
        return run_certify(arguments)
    if command == "compare":
        from exp_compare import compare
        paths = dict(zip(
            ("baseline_grade", "candidate_grade",
             "baseline_artifacts", "candidate_artifacts"), arguments))
        if len(paths) != 4:
            print("usage: experiment.py compare BASE_GRADE CAND_GRADE "
                  "BASE_ARTIFACTS CAND_ARTIFACTS")
            return 2
        loaded = {key: json.loads(Path(value).read_text())
                  for key, value in paths.items()}
        result = compare(**loaded)
        print(json.dumps(
            {key: result[key] for key in
             ("accepted", "reasons", "converted", "regressed",
              "workload_growth")}, indent=1, sort_keys=True))
        return 0 if result["accepted"] else 1
    if command == "paid":
        return prepare_paid(arguments)
    print(f"unknown command: {command}")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
