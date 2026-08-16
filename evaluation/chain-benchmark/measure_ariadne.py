#!/usr/bin/env python3
"""Run Ariadne alone and measure its source proof against reviewed gold chains.

The runtime arm is always ``ariadne_arm.py``. There is no raw/bare model mode,
no Docker arm, and no LLM judge. Scoring is deterministic: every accepted gold
claim must recover its reviewed symbols, definition coordinates, compiler-edge
sites, and source witness fragments from hash-verified evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
GOLD = HERE / "gold-chain-reviewed-compact.json"
QUESTIONS = ROOT / "evaluation/spool-clean-room/questions_debcrumb_ask.json"
ARIADNE_ARM = ROOT / "evaluation/spool-clean-room/ariadne_arm.py"
SCORE_CHAIN = ROOT / "evaluation/spool-clean-room/score_chain.py"

_spec = importlib.util.spec_from_file_location(
    "_ariadne_gold_strict_score", SCORE_CHAIN)
_strict = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_strict)


def _same_file(left: str | Path, right: str | Path) -> bool:
    first = str(left).replace("\\", "/").removeprefix("/corpus/").lstrip("/")
    second = str(right).replace("\\", "/").removeprefix("/corpus/").lstrip("/")
    return (first == second or first.endswith("/" + second)
            or second.endswith("/" + first))


def _normalize(value: str) -> str:
    return " ".join(str(value).split())


def _citation_locations(answer: dict) -> list[dict]:
    locations = []
    for citation in answer.get("citations", []) or []:
        file = str(citation.get("file") or "")
        line = int(citation.get("line") or 0)
        line_end = int(citation.get("line_end") or line)
        if file and line > 0:
            locations.append({"file": file, "line": line, "line_end": line_end})
        call_site = str(citation.get("call_site") or "")
        if ":" in call_site:
            call_file, call_line = call_site.rsplit(":", 1)
            if call_line.isdigit():
                locations.append({
                    "file": call_file, "line": int(call_line),
                    "line_end": max(int(call_line), line_end)})
    return locations


def verified_evidence(answer: dict, corpus: Path, *, index=None, cache=None) -> dict:
    index = index if index is not None else _strict.index(corpus)
    cache = cache if cache is not None else {}
    quotes = _strict.verified_quotes(answer, index, corpus, cache, 0, min_chars = 0)
    citations = _citation_locations(answer)
    spans = []
    for quote in quotes:
        relative = str(quote["file"].relative_to(corpus)).replace("\\", "/")
        start = int(quote["line"])
        end = start
        for citation in citations:
            if (_same_file(relative, citation["file"])
                    and int(citation["line"]) == start):
                end = max(end, start + 5, int(citation["line_end"]))
        try:
            line_count = len(quote["file"].read_text(errors="ignore").splitlines())
            end = min(end, line_count)
        except OSError:
            pass
        spans.append({
            "file": relative, "line_start": start, "line_end": end,
            "text": str(quote.get("text") or "")})
    return {
        "provenance_ok": _strict.provenance_ok(answer, corpus),
        "spans": spans,
        "verified_quotes": len(quotes),
    }


def _selected_candidate(claim: dict, anchor: dict) -> dict:
    anchor_name = str(anchor["anchor"])
    selected = claim.get("review", {}).get(
        "selected_candidate_by_anchor", {}).get(anchor_name)
    for candidate in anchor.get("candidates", []):
        if selected in {
                candidate.get("canonical_id"), candidate.get("qualified_name")}:
            return candidate
    raise ValueError(
        f"claim {claim.get('id')}: selected candidate missing for {anchor_name}")


def _selected_paths(claim: dict) -> list[dict]:
    available = {
        str(path.get("id")): path for path in claim.get("candidate_paths", [])}
    selected = []
    for path_id in claim.get("review", {}).get("selected_path_ids", []):
        path = available.get(str(path_id))
        if path is None:
            raise ValueError(
                f"claim {claim.get('id')}: selected path missing: {path_id}")
        selected.append(path)
    if not selected and len(claim.get("anchors", [])) > 1:
        raise ValueError(f"claim {claim.get('id')}: no selected paths")
    return selected


def _claim_expectations(claim: dict) -> dict:
    if claim.get("review", {}).get("status") != "accepted":
        raise ValueError(f"claim {claim.get('id')} is not accepted")
    paths = _selected_paths(claim)
    symbols = {}
    definitions = {}
    for anchor in claim.get("anchors", []):
        candidate = _selected_candidate(claim, anchor)
        qualified = str(candidate.get("qualified_name") or "")
        if qualified:
            symbols[qualified] = qualified
        key = (str(candidate.get("file") or ""),
               int(candidate.get("line_start") or 0),
               int(candidate.get("line_end") or candidate.get("line_start") or 0),
               qualified)
        if key[0] and key[1] > 0:
            definitions[key] = key
    edges = {}
    for path in paths:
        for node in path.get("nodes", []):
            qualified = str(node.get("qualified_name") or "")
            if qualified:
                symbols[qualified] = qualified
            key = (str(node.get("file") or ""),
                   int(node.get("line_start") or 0),
                   int(node.get("line_end") or node.get("line_start") or 0),
                   qualified)
            if key[0] and key[1] > 0:
                definitions[key] = key
        for edge in path.get("edges", []):
            key = (str(edge.get("file") or ""), int(edge.get("line") or 0),
                   str(edge.get("caller") or edge.get("caller_canonical_id") or ""),
                   str(edge.get("callee") or edge.get("callee_canonical_id") or ""))
            if key[0] and key[1] > 0:
                edges[key] = key
    witnesses = []
    for witness in claim.get("witnesses", []):
        for fragment in witness.get("contains", []) or []:
            witnesses.append({
                "id": str(witness.get("id") or "<unnamed>"),
                "file": str(witness.get("file") or ""),
                "line_start": int(witness.get("line_start") or 0),
                "line_end": int(witness.get("line_end")
                                or witness.get("line_start") or 0),
                "fragment": str(fragment),
            })
    return {
        "symbols": sorted(symbols),
        "definitions": sorted(definitions),
        "edges": sorted(edges),
        "witness_fragments": witnesses,
    }


def _observed_symbols(answer: dict) -> set[str]:
    values = set()
    for key in ("selected_symbols", "hydrated_symbols"):
        values.update(str(value) for value in answer.get(key, []) or [] if value)
    values.update(
        str(citation.get("qualified_name"))
        for citation in answer.get("citations", []) or []
        if citation.get("qualified_name"))
    return values


def _span_covers(spans: list[dict], file: str, line_start: int,
                 line_end: int) -> bool:
    return any(
        _same_file(span["file"], file)
        and int(span["line_start"]) <= line_end
        and int(span["line_end"]) >= line_start
        for span in spans)


def score_claim(answer: dict, claim: dict, evidence: dict) -> dict:
    expected = _claim_expectations(claim)
    observed_symbols = _observed_symbols(answer)
    missing_symbols = [
        symbol for symbol in expected["symbols"] if symbol not in observed_symbols]
    missing_definitions = []
    for file, line_start, line_end, symbol in expected["definitions"]:
        if not _span_covers(evidence["spans"], file, line_start, line_end):
            missing_definitions.append(
                f"{symbol}@{file}:{line_start}-{line_end}")
    missing_edges = []
    for file, line, caller, callee in expected["edges"]:
        if not _span_covers(evidence["spans"], file, line, line):
            missing_edges.append(f"{caller}->{callee}@{file}:{line}")
    missing_witness_fragments = []
    for witness in expected["witness_fragments"]:
        fragment = _normalize(witness["fragment"])
        covered = any(
            _same_file(span["file"], witness["file"])
            and int(span["line_start"]) <= witness["line_end"]
            and int(span["line_end"]) >= witness["line_start"]
            and fragment in _normalize(span["text"])
            for span in evidence["spans"])
        if not covered:
            missing_witness_fragments.append(
                f"{witness['id']}:{witness['fragment']}")
    passed = bool(
        evidence["provenance_ok"]
        and not missing_symbols
        and not missing_definitions
        and not missing_edges
        and not missing_witness_fragments)
    return {
        "id": claim.get("id"),
        "assertion": claim.get("assertion"),
        "passed": passed,
        "expected_symbols": len(expected["symbols"]),
        "expected_definitions": len(expected["definitions"]),
        "expected_edges": len(expected["edges"]),
        "expected_witness_fragments": len(expected["witness_fragments"]),
        "missing_symbols": missing_symbols,
        "missing_definitions": missing_definitions,
        "missing_edges": missing_edges,
        "missing_witness_fragments": missing_witness_fragments,
    }


def _validate_gold(gold: dict) -> None:
    if gold.get("status") != "reviewed-gold":
        raise ValueError("gold artifact status must be reviewed-gold")
    ids = [int(question["id"]) for question in gold.get("questions", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate gold question id")
    for question in gold.get("questions", []):
        if question.get("review", {}).get("status") != "accepted":
            raise ValueError(f"Q{question.get('id')}: question is not accepted")
        for claim in question.get("claims", []):
            _claim_expectations(claim)
def score_answers(answers: list[dict], gold: dict, corpus: Path,
                  only: set[int] | None = None) -> dict:
    _validate_gold(gold)
    answer_ids = [int(answer["id"]) for answer in answers]
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("duplicate answer question id")
    by_id = {int(answer["id"]): answer for answer in answers}
    questions = [
        question for question in gold.get("questions", [])
        if only is None or int(question["id"]) in only]
    index, cache = _strict.index(corpus), {}
    results = []
    for question in questions:
        question_id = int(question["id"])
        answer = by_id.get(question_id)
        if answer is None:
            claims = []
            for claim in question.get("claims", []):
                expected = _claim_expectations(claim)
                claims.append({
                    "id": claim.get("id"), "assertion": claim.get("assertion"),
                    "passed": False,
                    "expected_symbols": len(expected["symbols"]),
                    "expected_definitions": len(expected["definitions"]),
                    "expected_edges": len(expected["edges"]),
                    "expected_witness_fragments": len(
                        expected["witness_fragments"]),
                    "missing_symbols": list(expected["symbols"]),
                    "missing_definitions": [
                        f"{symbol}@{file}:{start}-{end}"
                        for file, start, end, symbol in expected["definitions"]],
                    "missing_edges": [
                        f"{caller}->{callee}@{file}:{line}"
                        for file, line, caller, callee in expected["edges"]],
                    "missing_witness_fragments": [
                        f"{item['id']}:{item['fragment']}"
                        for item in expected["witness_fragments"]],
                })
            results.append({
                "id": question_id, "question": question.get("question"),
                "answer_present": False, "passed": False,
                "provenance_ok": False, "verified_quotes": 0,
                "confidence": None, "elapsed_s": 0.0,
                "total_cost_usd": 0.0, "claims": claims})
            continue
        evidence = verified_evidence(answer, corpus, index=index, cache=cache)
        claims = [
            score_claim(answer, claim, evidence)
            for claim in question.get("claims", [])]
        results.append({
            "id": question_id, "question": question.get("question"),
            "answer_present": True,
            "passed": bool(claims) and all(claim["passed"] for claim in claims),
            "provenance_ok": evidence["provenance_ok"],
            "verified_quotes": evidence["verified_quotes"],
            "confidence": answer.get("confidence"),
            "chain_confidence": answer.get("chain_confidence"),
            "elapsed_s": float(answer.get("elapsed_s") or 0.0),
            "total_cost_usd": float(answer.get("total_cost_usd") or 0.0),
            "claims": claims,
        })
    claim_rows = [claim for question in results for claim in question["claims"]]
    expected_symbols = sum(row["expected_symbols"] for row in claim_rows)
    expected_definitions = sum(row["expected_definitions"] for row in claim_rows)
    expected_edges = sum(row["expected_edges"] for row in claim_rows)
    expected_witnesses = sum(
        row["expected_witness_fragments"] for row in claim_rows)
    missing_symbols = sum(len(row["missing_symbols"]) for row in claim_rows)
    missing_definitions = sum(len(row["missing_definitions"]) for row in claim_rows)
    missing_edges = sum(len(row["missing_edges"]) for row in claim_rows)
    missing_witnesses = sum(
        len(row["missing_witness_fragments"]) for row in claim_rows)

    def recall(expected: int, missing: int) -> float:
        return 1.0 if expected == 0 else (expected - missing) / expected

    return {
        "measurement": "ariadne-only-reviewed-gold-proof",
        "summary": {
            "questions": len(results),
            "answers": sum(row["answer_present"] for row in results),
            "passed_questions": sum(row["passed"] for row in results),
            "question_score": (
                sum(row["passed"] for row in results) / len(results)
                if results else 0.0),
            "claims": len(claim_rows),
            "passed_claims": sum(row["passed"] for row in claim_rows),
            "claim_score": (
                sum(row["passed"] for row in claim_rows) / len(claim_rows)
                if claim_rows else 0.0),
            "symbol_recall": recall(expected_symbols, missing_symbols),
            "definition_evidence_recall": recall(
                expected_definitions, missing_definitions),
            "edge_evidence_recall": recall(expected_edges, missing_edges),
            "witness_evidence_recall": recall(
                expected_witnesses, missing_witnesses),
            "cost_scope": (
                "captured LLM completions only; embedding API charges excluded"),
            "total_cost_usd": round(sum(
                row["total_cost_usd"] for row in results), 6),
            "total_elapsed_s": round(sum(row["elapsed_s"] for row in results), 1),
            "max_elapsed_s": round(max(
                (row["elapsed_s"] for row in results), default=0.0), 1),
        },
        "questions": results,
    }
def ariadne_command(
        resume: bool = False, *, python: Path, questions: Path,
        answers: Path, source: str, corpus: Path, only: str | None,
        concurrency: int, timeout: float,
        trace_dir: Path | None = None) -> list[str]:
    """Build the Ariadne-only command, including diagnostic capture."""
    command = [
        str(python), "-u", str(ARIADNE_ARM),
        "--questions", str(questions),
        "--out", str(answers),
        "--source", source,
        "--corpus", str(corpus),
        "--concurrency", str(concurrency),
        "--timeout", str(timeout),
    ]
    if trace_dir is not None:
        command.extend(["--trace-dir", str(trace_dir)])
    if only:
        command.extend(["--only", only])
    if resume:
        command.append("--resume")
    return command


def _print_report(report: dict) -> None:
    print("\n  id  claims  quotes   cost   conf  result")
    for question in report["questions"]:
        passed = sum(claim["passed"] for claim in question["claims"])
        required = len(question["claims"])
        result = "PASS" if question["passed"] else "FAIL"
        confidence = question.get("confidence") or "-"
        print(
            f"{question['id']:>4}  {passed:>2}/{required:<2}  "
            f"{question['verified_quotes']:>6}  "
            f"${question['total_cost_usd']:>5.3f}  "
            f"{confidence:>5}  {result}")
    summary = report["summary"]
    print(
        f"\nARIADNE REVIEWED-GOLD SCORE: "
        f"{summary['passed_questions']}/{summary['questions']} questions; "
        f"{summary['passed_claims']}/{summary['claims']} claims")
    print(
        "proof recall: "
        f"symbols={summary['symbol_recall']:.0%}, "
        f"definitions={summary['definition_evidence_recall']:.0%}, "
        f"edges={summary['edge_evidence_recall']:.0%}, "
        f"witnesses={summary['witness_evidence_recall']:.0%}")
    print(f"reported LLM cost: ${summary['total_cost_usd']:.4f}")
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument(
        "--answers", default=str(HERE / "ariadne-gold-answers.json"))
    parser.add_argument(
        "--report", default=str(HERE / "ariadne-gold-measurement.json"))
    parser.add_argument(
        "--trace-dir",
        help=("compressed per-question diagnostic traces; defaults beside "
              "--answers"))
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--corpus", default=str(ROOT / "spool-corpus"))
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume compatible checkpointed answers with intact traces")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--score-only", action="store_true",
        help="score the existing --answers file without running Ariadne")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-perfect", action="store_true")
    args = parser.parse_args(argv)

    gold_path = Path(args.gold)
    questions_path = Path(args.questions)
    answers_path = Path(args.answers)
    report_path = Path(args.report)
    trace_dir = (
        Path(args.trace_dir) if args.trace_dir
        else answers_path.with_name(answers_path.stem + "-traces"))
    corpus = Path(args.corpus)
    selected = (
        {int(value) for value in args.only.split(",")}
        if args.only else None)
    command = ariadne_command(
        python=Path(sys.executable), questions=questions_path,
        answers=answers_path, source=args.source, corpus=corpus,
        only=args.only, concurrency=max(1, args.concurrency),
        timeout=args.timeout, resume=args.resume, trace_dir=trace_dir)
    if args.dry_run:
        print("ARIADNE-ONLY command:")
        print(" ".join(command))
        return 0

    run_status = 0
    if not args.score_only:
        missing_keys = [
            key for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            if not os.environ.get(key)]
        if missing_keys:
            print("missing required key(s): " + ", ".join(missing_keys),
                  file=sys.stderr)
            return 2
        run_status = subprocess.run(
            command, cwd=ROOT, check=False).returncode
    if not answers_path.is_file():
        print(
            f"answers file was not produced: {answers_path}",
            file=sys.stderr)
        return run_status or 2

    gold = json.loads(gold_path.read_text())
    answers = json.loads(answers_path.read_text())
    report = score_answers(answers, gold, corpus, only=selected)
    report["gold_file"] = str(gold_path)
    report["gold_sha256"] = hashlib.sha256(
        gold_path.read_bytes()).hexdigest()
    report["answers_file"] = str(answers_path)
    report["diagnostic_trace_dir"] = str(trace_dir)
    report["diagnostic_traces"] = {
        str(answer["id"]): answer["diagnostic_trace"]
        for answer in answers
        if answer.get("diagnostic_trace")}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    _print_report(report)
    print(
        f"diagnostic traces: {len(report['diagnostic_traces'])}/"
        f"{report['summary']['answers']} -> {trace_dir}")
    print(f"detail -> {report_path}")
    if args.require_perfect and (
            report["summary"]["passed_questions"]
            != report["summary"]["questions"]):
        return 1
    return run_status


if __name__ == "__main__":
    raise SystemExit(main())
