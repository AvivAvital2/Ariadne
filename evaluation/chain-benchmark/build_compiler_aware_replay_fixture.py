#!/usr/bin/env python3
"""Create the compact public replay fixture from a recorded Ariadne run.

The resulting gzip contains every recorded selection, citation, transition,
and completeness field for the twelve-question public panel.  It excludes
answer prose, prompts, traces, private paths, cost, and provider-usage data.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FIELDS = (
    "id", "question", "selected_symbols", "selected_body_symbols",
    "hydrated_symbols", "citations", "transition_claims", "file_hashes",
    "chain_complete", "formulation_complete", "scope_complete",
    "selection_complete", "route_selection_status", "selected_route_ids",
    "selected_section_ids", "chain_summary", "confidence_reasons",
    "completeness_reasons", "formulation_reasons", "scope_reasons",
    "selection_reasons",
)
PRIVATE_FIELDS = {
    "answer", "benchmark_corpus", "diagnostic_trace", "elapsed_s",
    "sources", "token_usage", "tool_calls", "total_cost_usd",
    "usage_by_phase", "phase_timings", "llm_calls", "num_turns",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    if normalized.startswith("/corpus/"):
        return normalized.removeprefix("/corpus/")
    if normalized.startswith("/"):
        raise ValueError(f"refusing private absolute path: {value}")
    return normalized


def _sanitize(value: Any, *, field: str = "") -> Any:
    if isinstance(value, list):
        return [_sanitize(item, field=field) for item in value]
    if isinstance(value, dict):
        if any(key in PRIVATE_FIELDS for key in value):
            raise ValueError("private run field reached replay fixture")
        if field == "file_hashes":
            return {_relative_path(str(key)): _sanitize(item)
                    for key, item in value.items()}
        return {str(key): _sanitize(item, field=str(key))
                for key, item in value.items()}
    if isinstance(value, str) and field in {"file", "call_site"}:
        return _relative_path(value)
    if isinstance(value, str) and value.startswith("/"):
        raise ValueError(f"refusing private absolute path in {field or 'value'}")
    return value


def _panel(record: dict) -> dict[int, str]:
    values = {
        int(question["benchmark_question_id"]): str(question["question"])
        for question in record.get("questions") or ()}
    if len(values) != 12:
        raise ValueError("public record must contain exactly twelve questions")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", required=True, type=Path,
                        help="private recorded Ariadne answers JSON")
    parser.add_argument("--record", type=Path,
                        default=HERE / "compiler-aware-comparison-record.json")
    parser.add_argument("--out", required=True, type=Path,
                        help="new public gzip replay fixture")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError(f"refusing to overwrite existing fixture: {args.out}")

    with args.record.open() as handle:
        panel = _panel(json.load(handle))
    with args.answers.open() as handle:
        answers = json.load(handle)
    by_id = {int(answer["id"]): answer for answer in answers}
    if set(by_id) & set(panel) != set(panel):
        raise ValueError("recorded answers do not contain every panel question")

    records = []
    for question_id, question in panel.items():
        answer = by_id[question_id]
        if str(answer.get("question")) != question:
            raise ValueError(f"Q{question_id}: recorded question differs from public record")
        records.append({field: _sanitize(answer[field], field=field)
                        for field in FIELDS if field in answer})
    payload = {
        "schema": "compiler-aware-recorded-replay/v1",
        "scope": ("All recorded selection and evidence state for the twelve-question "
                  "public panel; excludes private model and run payloads."),
        "record_sha256": _sha256(args.record),
        "private_answers_sha256": _sha256(args.answers),
        "questions": records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as compressed:
            compressed.write(encoded)
    print(f"wrote {len(records)} panel records ({args.out.stat().st_size:,} bytes) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
