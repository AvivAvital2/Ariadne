#!/usr/bin/env python3
"""Audit an actual captured Ariadne run without rebuilding or repairing it.

The diagnostic answers and compressed traces are treated as immutable facts.
This audit separates candidate retrieval, live selection, hydration, the exact
formulation input, the raw narration, and the final verified answer. It also
reports the provider token surface that produced the answer.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import measure_ariadne as reviewed_measure


TRACE_SCHEMA = "ariadne-live-diagnostic-v1"
_PLACEHOLDER = re.compile(r"\{\{([NEX]\d+)\}\}|(?<![\w{])([NEX]\d+)(?![\w}])")
_NODE = re.compile(
    r"^\s*\{\{(N\d+)\}\}:\s*(.*?)\s+\[(.+):(\d+)\];")
_EDGE = re.compile(
    r"^\s*\{\{(E\d+)\}\}:.*?\bat\s+(.+):(\d+)\s*$")
_CHUNK = re.compile(
    r"^\s*\{\{(X\d+)\}\}:\s*(.+):(\d+)-(\d+)\s+\[[^]]*\]\s*$")
_CHUNK_LINE = re.compile(r"^\s{4}(\d+)\s+\|\s?(.*)$")


def _message_content(value) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return str(value or "")
    parts = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(str(
                item.get("text") or item.get("input_text")
                or item.get("content") or ""))
    return "\n".join(parts)


def load_trace(trace_dir: Path, answer: dict) -> dict:
    """Load the exact trace receipt attached to one answer, failing closed."""
    receipt = answer.get("diagnostic_trace") or {}
    name = str(receipt.get("file") or "")
    expected = str(receipt.get("sha256") or "")
    if receipt.get("schema") != TRACE_SCHEMA:
        raise ValueError("diagnostic trace schema mismatch")
    if not name or Path(name).name != name:
        raise ValueError("invalid diagnostic trace file")
    path = trace_dir / name
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != expected:
        raise ValueError("diagnostic trace hash mismatch")
    try:
        payload = json.loads(gzip.decompress(compressed))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("invalid diagnostic trace payload") from error
    if payload.get("schema") != TRACE_SCHEMA:
        raise ValueError("diagnostic trace payload schema mismatch")
    if int(payload.get("id") or -1) != int(answer.get("id") or -2):
        raise ValueError("diagnostic trace question mismatch")
    return payload


def formulation_completion(trace: dict) -> dict:
    """The last actual formulation call, including a repair when one occurred."""
    calls = [
        call for call in trace.get("llm_completions", ())
        if str(call.get("phase") or "") in ("completion", "provider")]
    if not calls:
        raise ValueError("diagnostic trace has no formulation completion")
    return calls[-1]


def formulation_prompt(call: dict) -> str:
    messages = list(call.get("messages") or ())
    user_messages = [
        _message_content(message.get("content"))
        for message in messages
        if message.get("role") == "user"]
    if user_messages:
        return user_messages[-1]
    return "\n".join(
        _message_content(message.get("content")) for message in messages)


def parse_formulation_ir(prompt: str) -> dict:
    """Parse only the stable evidence IDs rendered into the captured prompt."""
    nodes = {}
    edges = {}
    chunks = {}
    active_chunk = None
    for line in str(prompt or "").splitlines():
        node = _NODE.match(line)
        if node:
            identifier, symbol, file, number = node.groups()
            nodes[identifier] = {
                "symbol": symbol,
                "file": file,
                "line_start": int(number),
                "line_end": int(number),
            }
            active_chunk = None
            continue
        edge = _EDGE.match(line)
        if edge:
            identifier, file, number = edge.groups()
            edges[identifier] = {
                "file": file,
                "line_start": int(number),
                "line_end": int(number),
                "text": "",
            }
            active_chunk = None
            continue
        chunk = _CHUNK.match(line)
        if chunk:
            identifier, file, start, end = chunk.groups()
            chunks[identifier] = {
                "file": file,
                "line_start": int(start),
                "line_end": int(end),
                "lines": [],
                "text": "",
            }
            active_chunk = identifier
            continue
        source_line = _CHUNK_LINE.match(line)
        if source_line and active_chunk is not None:
            number, text = source_line.groups()
            chunks[active_chunk]["lines"].append((int(number), text))
            continue
        if line.strip() and not line.startswith("    "):
            active_chunk = None
    for chunk in chunks.values():
        chunk["text"] = "\n".join(text for _number, text in chunk["lines"])
    return {"nodes": nodes, "edges": edges, "chunks": chunks}


def _stage_scores(question: dict, evidence_ir: dict, *,
                  referenced_ids=None) -> list[dict]:
    allowed = None if referenced_ids is None else set(referenced_ids)
    nodes = [
        value for identifier, value in evidence_ir["nodes"].items()
        if allowed is None or identifier in allowed]
    edges = [
        value for identifier, value in evidence_ir["edges"].items()
        if allowed is None or identifier in allowed]
    chunks = [
        value for identifier, value in evidence_ir["chunks"].items()
        if allowed is None or identifier in allowed]
    spans = [
        {
            "file": value["file"],
            "line_start": int(value["line_start"]),
            "line_end": int(value["line_end"]),
            "text": str(value.get("text") or ""),
        }
        for value in (*nodes, *edges, *chunks)
    ]
    observed = [value["symbol"] for value in nodes]
    answer = {
        "selected_symbols": observed,
        "hydrated_symbols": observed,
        "citations": [],
    }
    evidence = {
        "provenance_ok": bool(chunks),
        "spans": spans,
        "verified_quotes": len(chunks),
    }
    return [
        reviewed_measure.score_claim(answer, claim, evidence)
        for claim in question.get("claims", ())
        if claim.get("review", {}).get("status") == "accepted"
    ]


def _occurrence_symbols(occurrences) -> set[str]:
    symbols = set()
    for occurrence in occurrences or ():
        if len(occurrence) > 0 and occurrence[0]:
            symbols.add(str(occurrence[0]))
        if len(occurrence) > 4 and occurrence[4]:
            symbols.add(str(occurrence[4]))
    return symbols


def candidate_symbols(answer: dict) -> set[str]:
    symbols = set()
    for route in (answer.get("route_candidates") or {}).values():
        symbols.update(str(symbol) for symbol in route or () if symbol)
    for occurrences in (
            answer.get("route_candidate_occurrences") or {}).values():
        symbols.update(_occurrence_symbols(occurrences))
    return symbols


def selected_symbols(answer: dict) -> set[str]:
    symbols = {
        str(symbol)
        for key in ("selected_symbols", "selected_body_symbols")
        for symbol in answer.get(key, ()) or ()
        if symbol
    }
    routes = answer.get("route_candidate_occurrences") or {}
    for route_id in answer.get("selected_route_ids", ()) or ():
        symbols.update(_occurrence_symbols(routes.get(str(route_id), ())))
    return symbols


def hydrated_symbols(answer: dict) -> set[str]:
    symbols = selected_symbols(answer)
    symbols.update(
        str(symbol) for symbol in answer.get("hydrated_symbols", ()) or ()
        if symbol)
    return symbols


def _expected_symbols(claim: dict) -> set[str]:
    return set(reviewed_measure._claim_expectations(claim)["symbols"])


def _missing(expected: set[str], observed: set[str]) -> list[str]:
    return sorted(expected - observed)


def _score_map(scores) -> dict[str, dict]:
    return {str(score.get("id")): score for score in scores}


def _token_summary(rows) -> dict:
    formulation = [
        row for row in rows
        if str(row.get("phase") or "") in ("completion", "provider")]
    selection = [
        row for row in rows
        if str(row.get("phase") or "") not in ("completion", "provider")]
    total_input = sum(int(row.get("input_tokens", 0) or 0) for row in rows)
    total_output = sum(int(row.get("output_tokens", 0) or 0) for row in rows)
    return {
        "selection_input": sum(
            int(row.get("input_tokens", 0) or 0) for row in selection),
        "selection_output": sum(
            int(row.get("output_tokens", 0) or 0) for row in selection),
        "formulation_input": sum(
            int(row.get("input_tokens", 0) or 0) for row in formulation),
        "formulation_output": sum(
            int(row.get("output_tokens", 0) or 0) for row in formulation),
        "total_input": total_input,
        "total_output": total_output,
    }
def _selection_prompt_text(trace: dict) -> str:
    parts = []
    for call in trace.get("llm_completions", ()):
        if str(call.get("phase") or "") in ("completion", "provider"):
            continue
        for message in call.get("messages", ()) or ():
            parts.append(_message_content(message.get("content")))
    return "\n".join(parts)
def _earliest_failure(*, candidate_missing, selected_missing,
                      hydrated_missing, prompt_score, final_score) -> str:
    if final_score.get("passed"):
        return "pass"
    if hydrated_missing:
        if candidate_missing:
            return "retrieval"
        if selected_missing:
            return "ranking"
        return "retention"
    if not prompt_score.get("passed"):
        return "materialization"
    return "formulation"


def audit_live_trace(answer: dict, question: dict, trace: dict,
                     final_question: dict) -> dict:
    """Audit one immutable live trace from candidate surface to final answer."""
    call = formulation_completion(trace)
    prompt = formulation_prompt(call)
    narration = str(call.get("response") or "")
    evidence_ir = parse_formulation_ir(prompt)
    referenced = {
        match.group(1) or match.group(2)
        for match in _PLACEHOLDER.finditer(narration)}
    prompt_scores = _stage_scores(question, evidence_ir)
    narration_scores = _stage_scores(
        question, evidence_ir, referenced_ids=referenced)
    final_scores = list(final_question.get("claims") or ())
    prompt_by_id = _score_map(prompt_scores)
    narration_by_id = _score_map(narration_scores)
    final_by_id = _score_map(final_scores)
    candidates = candidate_symbols(answer)
    selection_text = _selection_prompt_text(trace)
    selected = selected_symbols(answer)
    hydrated = hydrated_symbols(answer)
    rows = []
    for claim in question.get("claims", ()):
        if claim.get("review", {}).get("status") != "accepted":
            continue
        identifier = str(claim.get("id"))
        expected = _expected_symbols(claim)
        candidate_missing = _missing(expected, candidates | {symbol for symbol in expected if symbol in selection_text})
        selected_missing = _missing(expected, selected)
        hydrated_missing = _missing(expected, hydrated)
        prompt_score = prompt_by_id.get(identifier, {"passed": False})
        narration_score = narration_by_id.get(identifier, {"passed": False})
        final_score = final_by_id.get(identifier, {"passed": False})
        rows.append({
            "id": identifier,
            "candidate_missing_symbols": candidate_missing,
            "selected_missing_symbols": selected_missing,
            "hydrated_missing_symbols": hydrated_missing,
            "formulation_input_passed": bool(prompt_score.get("passed")),
            "formulation_input_missing_definitions": list(
                prompt_score.get("missing_definitions") or ()),
            "formulation_input_missing_edges": list(
                prompt_score.get("missing_edges") or ()),
            "formulation_input_missing_witness_fragments": list(
                prompt_score.get("missing_witness_fragments") or ()),
            "narration_passed": bool(narration_score.get("passed")),
            "final_passed": bool(final_score.get("passed")),
            "earliest_failure": _earliest_failure(
                candidate_missing=candidate_missing,
                selected_missing=selected_missing,
                hydrated_missing=hydrated_missing,
                prompt_score=prompt_score,
                final_score=final_score),
        })
    tokens = _token_summary(trace.get("usage_rows") or ())
    phase_inputs = {
        str(phase): sum(
            int(row.get("input_tokens", 0) or 0)
            for row in trace.get("usage_rows", ())
            if str(row.get("phase") or "") == str(phase))
        for phase in sorted({
            str(row.get("phase") or "")
            for row in trace.get("usage_rows", ())})}
    return {
        "id": int(answer.get("id") or question.get("id")),
        "audit_scope": "immutable-live-trace",
        "final_passed": bool(final_question.get("passed")),
        "final_claims_passed": sum(
            bool(score.get("passed")) for score in final_scores),
        "final_claim_count": len(final_scores),
        "formulation_input_claims_passed": sum(
            bool(score.get("passed")) for score in prompt_scores),
        "narration_claims_passed": sum(
            bool(score.get("passed")) for score in narration_scores),
        "candidate_symbols": len(candidates),
        "selected_symbols": len(selected),
        "hydrated_symbols": len(hydrated),
        "formulation_ir": {
            "nodes": len(evidence_ir["nodes"]),
            "edges": len(evidence_ir["edges"]),
            "chunks": len(evidence_ir["chunks"]),
            "source_lines": sum(
                len(chunk["lines"])
                for chunk in evidence_ir["chunks"].values()),
        },
        "chars": {
            "formulation_prompt": len(prompt),
            "raw_narration": len(narration),
            "service_answer": len(str(trace.get("service_answer") or "")),
            "benchmark_answer": len(str(trace.get("benchmark_answer") or "")),
        },
        "tokens": tokens,
        "input_tokens_by_phase": phase_inputs,
        "cost_usd": float(answer.get("total_cost_usd") or 0.0),
        "claims": rows,
    }


def _summary(reports: list[dict]) -> dict:
    failures = Counter(
        claim["earliest_failure"]
        for report in reports for claim in report["claims"])
    return {
        "questions": len(reports),
        "passed_questions": sum(report["final_passed"] for report in reports),
        "claims": sum(report["final_claim_count"] for report in reports),
        "passed_claims": sum(
            report["final_claims_passed"] for report in reports),
        "earliest_failures": dict(sorted(failures.items())),
        "cost_usd": round(sum(report["cost_usd"] for report in reports), 6),
        "selection_input_tokens": sum(
            report["tokens"]["selection_input"] for report in reports),
        "formulation_input_tokens": sum(
            report["tokens"]["formulation_input"] for report in reports),
        "max_formulation_input_tokens": max(
            (report["tokens"]["formulation_input"] for report in reports),
            default=0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True)
    parser.add_argument(
        "--gold", default=str(HERE / "gold-chain-reviewed-compact.json"))
    parser.add_argument(
        "--corpus", default=str(ROOT / "spool-corpus"))
    parser.add_argument("--trace-dir")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-perfect", action="store_true")
    args = parser.parse_args(argv)

    answers_path = Path(args.answers)
    answers = json.loads(answers_path.read_text())
    trace_dir = (
        Path(args.trace_dir) if args.trace_dir else
        answers_path.with_name(answers_path.stem + "-traces"))
    gold = json.loads(Path(args.gold).read_text())
    selected = (
        {int(value) for value in args.only.split(",")}
        if args.only else None)
    final_report = reviewed_measure.score_answers(
        answers, gold, Path(args.corpus), only=selected)
    final_by_id = {
        int(question["id"]): question
        for question in final_report["questions"]}
    gold_by_id = {
        int(question["id"]): question
        for question in gold.get("questions", ())}
    answer_by_id = {
        int(answer["id"]): answer for answer in answers
        if selected is None or int(answer["id"]) in selected}
    reports = []
    for question_id in sorted(answer_by_id):
        if question_id not in gold_by_id or question_id not in final_by_id:
            continue
        report = audit_live_trace(
            answer_by_id[question_id], gold_by_id[question_id],
            load_trace(trace_dir, answer_by_id[question_id]),
            final_by_id[question_id])
        reports.append(report)
        failures = Counter(
            claim["earliest_failure"] for claim in report["claims"]
            if claim["earliest_failure"] != "pass")
        failure_text = ",".join(
            f"{name}={count}" for name, count in sorted(failures.items()))
        print(
            f"id {question_id}: final "
            f"{report['final_claims_passed']}/{report['final_claim_count']}; "
            f"prompt {report['formulation_input_claims_passed']}/"
            f"{report['final_claim_count']}; "
            f"formulation {report['tokens']['formulation_input']} tokens; "
            f"{failure_text or 'pass'}",
            flush=True)
    output = {
        "measurement": "immutable-live-trace-audit",
        "answers_file": str(answers_path),
        "trace_dir": str(trace_dir),
        "summary": _summary(reports),
        "questions": reports,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    summary = output["summary"]
    print(
        "LIVE TRACE SCORE: "
        f"{summary['passed_questions']}/{summary['questions']} questions; "
        f"{summary['passed_claims']}/{summary['claims']} claims; "
        f"USD {summary['cost_usd']:.4f}; "
        f"max formulation input {summary['max_formulation_input_tokens']} tokens")
    print(f"live trace audit -> {args.out}")
    if args.require_perfect and (
            summary["passed_questions"] != summary["questions"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
