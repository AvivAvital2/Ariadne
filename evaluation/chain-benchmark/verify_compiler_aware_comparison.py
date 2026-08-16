#!/usr/bin/env python3
"""Verify the public compiler-aware comparison panel without model calls.

The public panel deliberately covers only twelve displayed questions and their
twenty-five reviewed claims.  This verifier recomputes its aggregate results
from the public records.  With ``--source-root`` it additionally verifies each
referenced source hash, definition/rule location, and witness fragment against
an independently obtained DBR 17.3 LTS source tree.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PANEL_IDS = (4, 6, 8, 13, 15, 16, 62, 66, 67, 88, 147, 187)
METRICS = ("symbols", "definitions", "relation_sites", "witnesses")
REPOSITORIES = ("spark", "delta", "databricks-sdk-py")
REPLAY_FIELDS = {
    "id", "question", "selected_symbols", "selected_body_symbols",
    "hydrated_symbols", "citations", "transition_claims", "file_hashes",
    "chain_complete", "formulation_complete", "scope_complete",
    "selection_complete", "route_selection_status", "selected_route_ids",
    "selected_section_ids", "chain_summary", "confidence_reasons",
    "completeness_reasons", "formulation_reasons", "scope_reasons",
    "selection_reasons",
}


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _load_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and value.startswith("/")


def _fail(message: str) -> None:
    raise ValueError(message)


def _panel_questions(record: dict, proof: dict) -> tuple[list[dict], dict[int, dict]]:
    questions = list(record.get("questions") or ())
    observed_ids = tuple(int(item.get("benchmark_question_id", -1)) for item in questions)
    if observed_ids != PANEL_IDS:
        _fail(f"comparison panel ids differ: {observed_ids!r}")
    if any(int(item.get("display_number", 0)) != index
           for index, item in enumerate(questions, start=1)):
        _fail("comparison display numbers must be contiguous 1--12")
    proof_questions = {
        int(item.get("benchmark_question_id", -1)): item
        for item in proof.get("questions") or ()}
    if tuple(proof_questions) != PANEL_IDS:
        _fail("proof manifest must contain exactly the comparison panel")
    return questions, proof_questions


def _check_manifest_alignment(questions: list[dict], proof_by_id: dict[int, dict]) -> None:
    for question in questions:
        question_id = int(question["benchmark_question_id"])
        proof = proof_by_id[question_id]
        if proof.get("question") != question.get("question"):
            _fail(f"Q{question_id}: question text differs between public records")
        claims = list(question.get("claims") or ())
        proof_claims = list(proof.get("claims") or ())
        if [item.get("id") for item in proof_claims] != [item.get("id") for item in claims]:
            _fail(f"Q{question_id}: claim order or membership differs")
        expected_complete = all(bool(item.get("ariadne", {}).get("complete"))
                                for item in claims)
        if bool(proof.get("recorded_complete")) != expected_complete:
            _fail(f"Q{question_id}: recorded completion disagrees with claim results")
        for claim, selected in zip(claims, proof_claims, strict=True):
            if selected.get("assertion") != claim.get("assertion"):
                _fail(f"Q{question_id}/{claim.get('id')}: assertion differs")
            if bool(selected.get("recorded_complete")) != bool(
                    claim.get("ariadne", {}).get("complete")):
                _fail(f"Q{question_id}/{claim.get('id')}: completion differs")


def _transition_key(item: dict) -> tuple[str, tuple[str, ...]]:
    return str(item.get("text") or ""), tuple(item.get("locations") or ())


def _check_replay(questions: list[dict], proof_by_id: dict[int, dict], replay: dict,
                  *, record_path: Path) -> None:
    if replay.get("schema") != "compiler-aware-recorded-replay/v1":
        _fail("unrecognized recorded replay schema")
    if replay.get("record_sha256") != _sha256(record_path):
        _fail("recorded replay was built from a different public comparison record")
    if _contains_absolute_path(replay):
        _fail("recorded replay contains an absolute path")
    records = {int(item.get("id", -1)): item for item in replay.get("questions") or ()}
    if tuple(records) != PANEL_IDS:
        _fail("recorded replay must contain exactly the comparison panel")
    for question in questions:
        question_id = int(question["benchmark_question_id"])
        record = records[question_id]
        unexpected = set(record) - REPLAY_FIELDS
        if unexpected:
            _fail(f"Q{question_id}: replay contains excluded fields: {sorted(unexpected)}")
        if record.get("question") != question.get("question"):
            _fail(f"Q{question_id}: replay question differs from public record")
        proof = proof_by_id[question_id]
        if record.get("selected_symbols") != proof.get("selected_symbols"):
            _fail(f"Q{question_id}: selected symbol list differs from proof manifest")
        if record.get("selected_body_symbols") != proof.get("selected_body_symbols"):
            _fail(f"Q{question_id}: selected body symbol list differs from proof manifest")
        citations = {json.dumps(item, sort_keys=True) for item in record.get("citations") or ()}
        transitions = {_transition_key(item) for item in record.get("transition_claims") or ()}
        for claim in proof.get("claims") or ():
            for citation in claim.get("selected_source_citations") or ():
                if json.dumps(citation, sort_keys=True) not in citations:
                    _fail(f"Q{question_id}/{claim.get('id')}: selected citation absent from replay")
            for transition in claim.get("supported_directed_transitions") or ():
                if _transition_key(transition) not in transitions:
                    _fail(f"Q{question_id}/{claim.get('id')}: transition absent from replay")


def _totals(questions: list[dict]) -> dict:
    ariadne = {metric: [0, 0] for metric in METRICS}
    bare = {metric: [0, 0] for metric in METRICS}
    claims = 0
    ariadne_claims = 0
    for question in questions:
        for claim in question["claims"]:
            claims += 1
            ariadne_claims += bool(claim["ariadne"]["complete"])
            for metric in METRICS:
                for system, total in (("ariadne", ariadne), ("bare", bare)):
                    value = claim[system]["coverage"][metric]
                    total[metric][0] += int(value["present"])
                    total[metric][1] += int(value["required"])
    return {
        "questions": len(questions),
        "claims": claims,
        "ariadne_questions": sum(
            all(claim["ariadne"]["complete"] for claim in question["claims"])
            for question in questions),
        "ariadne_claims": ariadne_claims,
        "bare_questions": sum(
            bool(question["outcomes"]["bare_complete"]) for question in questions),
        "ariadne": ariadne,
        "bare": bare,
    }


def _resolve_source(root: Path, relative: str, expected_hash: str) -> Path:
    candidates = [root / repository / relative for repository in REPOSITORIES]
    candidates.append(root / relative)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest == expected_hash:
            return candidate
    _fail(f"no hash-matching source file for {relative}")


def _verify_source(questions: list[dict], root: Path) -> int:
    checked: dict[tuple[str, str], list[str]] = {}
    for question in questions:
        for claim in question["claims"]:
            source = claim["reviewed_source_provenance"]
            for item in source["source_files"]:
                key = (item["file"], item["sha256"])
                if key not in checked:
                    checked[key] = _resolve_source(root, *key).read_text(
                        errors="replace").splitlines()
            for definition in source["definitions"]:
                lines = checked[(definition["file"], next(
                    item["sha256"] for item in source["source_files"]
                    if item["file"] == definition["file"]))]
                if not 1 <= int(definition["line_start"]) <= int(definition["line_end"]) <= len(lines):
                    _fail(f"definition range is outside {definition['file']}")
            for relation in source["relation_sites"]:
                lines = checked[(relation["file"], next(
                    item["sha256"] for item in source["source_files"]
                    if item["file"] == relation["file"]))]
                if not 1 <= int(relation["line"]) <= len(lines):
                    _fail(f"relation site is outside {relation['file']}")
            for witness in source["witnesses"]:
                lines = checked[(witness["file"], next(
                    item["sha256"] for item in source["source_files"]
                    if item["file"] == witness["file"]))]
                start, end = int(witness["line_start"]), int(witness["line_end"])
                if not 1 <= start <= end <= len(lines):
                    _fail(f"witness range is outside {witness['file']}")
                excerpt = "\n".join(lines[start - 1:end])
                for fragment in witness["fragments"]:
                    if fragment not in excerpt:
                        _fail(f"witness fragment missing: {witness['id']}: {fragment}")
    return len(checked)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path,
                        default=HERE / "compiler-aware-comparison-record.json")
    parser.add_argument("--proof", type=Path,
                        default=HERE / "compiler-aware-ariadne-proof-manifest.json")
    parser.add_argument("--replay", type=Path,
                        default=HERE / "compiler-aware-recorded-replay.json.gz")
    parser.add_argument("--source-root", type=Path,
                        help="root containing spark/, delta/, and databricks-sdk-py/")
    args = parser.parse_args()

    record = _load(args.record)
    questions, proof_by_id = _panel_questions(record, _load(args.proof))
    _check_manifest_alignment(questions, proof_by_id)
    _check_replay(questions, proof_by_id, _load_gzip_json(args.replay),
                  record_path=args.record)
    totals = _totals(questions)
    if (totals["questions"], totals["claims"], totals["ariadne_questions"],
            totals["ariadne_claims"], totals["bare_questions"]) != (12, 25, 8, 19, 2):
        _fail(f"unexpected panel totals: {totals}")
    if args.source_root:
        print(f"source evidence: {_verify_source(questions, args.source_root)} files verified")
    print(f"recorded replay: {args.replay.stat().st_size:,} bytes verified")
    print("compiler-aware panel: "
          f"Ariadne {totals['ariadne_questions']}/12 questions, "
          f"{totals['ariadne_claims']}/25 claims; "
          f"bare {totals['bare_questions']}/12 complete chains")
    for system in ("ariadne", "bare"):
        text = ", ".join(f"{name} {present}/{required}"
                         for name, (present, required) in totals[system].items())
        print(f"{system} evidence: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
