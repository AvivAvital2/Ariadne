#!/usr/bin/env python3
"""Review gold-chain candidates without using an LLM or network service.

The tool deliberately separates mechanically proven connectivity from human
correctness decisions. It renders compact, source-grounded alternatives and
applies only explicit, fingerprinted selections to the candidate oracle.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def _selection_key(selection: dict) -> tuple:
    return (
        tuple(sorted(str(value) for value in selection.get("path_ids", []))),
        tuple(sorted(
            (str(anchor), str(canonical))
            for anchor, canonical in
            selection.get("selected_candidate_by_anchor", {}).items())),
    )


def selection_id(selection: dict) -> str:
    """Return a stable identifier independent of path or mapping order."""
    paths, endpoints = _selection_key(selection)
    payload = json.dumps(
        {"paths": paths, "endpoints": endpoints},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def unique_selections(claim: dict) -> list[dict]:
    """Collapse coherent sets which differ only by traversal ordering."""
    found = {}
    for raw in claim.get("coherent_path_sets", []):
        if raw.get("complete") is not True:
            continue
        normalized = copy.deepcopy(raw)
        normalized["path_ids"] = sorted(
            str(value) for value in raw.get("path_ids", []))
        normalized["selected_candidate_by_anchor"] = dict(sorted(
            (str(anchor), str(canonical))
            for anchor, canonical in
            raw.get("selected_candidate_by_anchor", {}).items()))
        normalized["selection_id"] = selection_id(normalized)
        key = _selection_key(normalized)
        previous = found.get(key)
        if previous is None or int(normalized.get("score", 0)) < int(
                previous.get("score", 0)):
            found[key] = normalized
    return sorted(found.values(), key=lambda item: (
        int(item.get("score", 0)), item["selection_id"]))
def selection_problems(claim: dict, selection: dict) -> list[str]:
    """Return mechanical reasons a coherent selection cannot be reviewed."""
    paths = {
        str(path.get("id")): path
        for path in claim.get("candidate_paths", [])}
    chosen = selection.get("selected_candidate_by_anchor", {})
    problems = []
    for witness in claim.get("witnesses", []):
        witness_id = str(witness.get("id", "<unnamed>"))
        for gap in witness.get("materialization", {}).get("gaps", []):
            problems.append(
                f"witness {witness_id}: materialization gap: {gap}")
        for error in witness.get("proof_errors", []):
            problems.append(f"witness {witness_id}: {error}")
    for path_id in selection.get("path_ids", []):
        path = paths.get(str(path_id))
        if path is None:
            problems.append(f"unknown path: {path_id}")
            continue
        if path.get("all_edges_compiler_verified") is not True:
            problems.append(f"{path_id}: contains an unverified compiler edge")
        for error in path.get("proof_errors", []):
            problems.append(f"{path_id}: {error}")
        for gap in path.get("materialization", {}).get("gaps", []):
            problems.append(f"{path_id}: materialization gap: {gap}")
        endpoints = path.get("endpoint_candidates", {})
        for side, anchor in zip(("left", "right"), path.get("connects", [])):
            expected = chosen.get(anchor)
            actual = endpoints.get(side, {}).get("canonical_id")
            if expected is not None and str(actual) != str(expected):
                problems.append(
                    f"{path_id}: endpoint for {anchor} is {actual}, "
                    f"selection requires {expected}")
    required_transitions = []
    for raw_transition in claim.get("transitions", []):
        if len(raw_transition) == 2:
            transition = tuple(str(value) for value in raw_transition)
            if transition not in required_transitions:
                required_transitions.append(transition)
    if required_transitions:
        required = set(required_transitions)
        reverse_transitions = {
            tuple(str(value) for value in transition)
            for transition in claim.get("reverse_transitions", [])
            if len(transition) == 2}
        proof_pairs = {
            ((right, left) if (left, right) in reverse_transitions
             else (left, right))
            for left, right in required_transitions}
        proven = set()
        for path_id in selection.get("path_ids", []):
            path = paths.get(str(path_id))
            if path is None:
                continue
            connected = tuple(str(value) for value in path.get("connects", []))
            if len(connected) != 2:
                continue
            reverse = (connected[1], connected[0])
            orientation = str(path.get("orientation", "left-to-right"))
            if orientation == "right-to-left":
                transition = reverse
            elif orientation == "bidirectional-reference":
                if connected in proof_pairs:
                    transition = connected
                elif reverse in proof_pairs:
                    transition = reverse
                else:
                    transition = connected
            else:
                transition = connected
            proven.add(transition)
        for left, right in required_transitions:
            causal = (left, right)
            proof = (right, left) if causal in reverse_transitions else causal
            if proof not in proven:
                problems.append(
                    f"missing required transition: {left} -> {right}")
    return list(dict.fromkeys(problems))


def _claim_diagnostics(report: dict, claim: dict) -> dict:
    selections = unique_selections(claim)
    assignments = {
        tuple(sorted(item["selected_candidate_by_anchor"].items()))
        for item in selections}
    coherent_count = len(claim.get("coherent_path_sets", []))
    cap = int(report.get("parameters", {}).get("coherent_set_limit", 0) or 0)
    return {
        "coherent_sets": coherent_count,
        "unique_selections": len(selections),
        "endpoint_assignments": len(assignments),
        "exhausted_paths": sum(
            path.get("search", {}).get("exhausted") is True
            for path in claim.get("candidate_paths", [])),
        "at_report_cap": bool(cap and coherent_count >= cap),
    }


def review_items(report: dict) -> list[dict]:
    """Return claims in descending need for semantic adjudication."""
    items = []
    for question in report.get("questions", []):
        for claim in question.get("claims", []):
            diagnostics = _claim_diagnostics(report, claim)
            reviewed = claim.get("review", {}).get("status") == "accepted"
            items.append({
                "question": question,
                "claim": claim,
                "diagnostics": diagnostics,
                "_priority": (
                    1 if reviewed else 0,
                    -diagnostics["endpoint_assignments"],
                    -int(diagnostics["at_report_cap"]),
                    -diagnostics["exhausted_paths"],
                    -diagnostics["unique_selections"],
                    int(question.get("id", 0)),
                    str(claim.get("id", "")),
                ),
            })
    items.sort(key=lambda item: item["_priority"])
    for item in items:
        item.pop("_priority", None)
    return items


def _safe_candidate_paths(root: Path, file: str) -> list[Path]:
    raw = Path(file)
    direct = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        direct.relative_to(root)
    except ValueError:
        return []
    candidates = [direct] if direct.is_file() else []
    if not raw.is_absolute():
        try:
            children = list(root.iterdir())
        except OSError:
            children = []
        for child in children:
            if not child.is_dir():
                continue
            candidate = (child / raw).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if candidate.is_file() and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def source_context(
        source_root: str | Path,
        file: str,
        *,
        line: int,
        context_lines: int,
        cache: dict,
        expected_sha256: str | None = None,
) -> str | None:
    """Read bounded context only when a path resolves uniquely inside the root."""
    root = Path(source_root).resolve()
    key = (str(root), str(file))
    if key not in cache:
        candidates = _safe_candidate_paths(root, str(file))
        if len(candidates) != 1:
            cache[key] = None
        else:
            try:
                data = candidates[0].read_bytes()
                text = data.decode("utf-8")
            except (OSError, UnicodeError):
                cache[key] = None
            else:
                cache[key] = (
                    text.splitlines(), hashlib.sha256(data).hexdigest())
    loaded = cache.get(key)
    if loaded is None:
        return None
    lines, digest = loaded
    if expected_sha256 and digest != expected_sha256:
        return None
    target = int(line)
    if target < 1 or target > len(lines):
        return None
    radius = max(int(context_lines), 0)
    start = max(target - radius, 1)
    end = min(target + radius, len(lines))
    return "\n".join(
        f"{number} | {lines[number - 1]}"
        for number in range(start, end + 1))


def _proof_excerpt(path: dict, file: str, line: int) -> tuple[str | None, str | None]:
    for excerpt in path.get("materialization", {}).get("excerpts", []):
        if str(excerpt.get("file", "")) != str(file):
            continue
        start = int(excerpt.get("line_start", 0))
        end = int(excerpt.get("line_end", start))
        if start <= int(line) <= end:
            content = str(excerpt.get("content", ""))
            formatted = "\n".join(
                f"{start + offset} | {value}"
                for offset, value in enumerate(content.splitlines()))
            return formatted, str(excerpt.get("sha256", "")) or None
    return None, None


def _endpoint_name(claim: dict, anchor: str, canonical_id: str) -> str:
    for anchor_row in claim.get("anchors", []):
        if str(anchor_row.get("anchor")) != str(anchor):
            continue
        for candidate in anchor_row.get("candidates", []):
            if str(candidate.get("canonical_id")) == str(canonical_id):
                return str(candidate.get("qualified_name") or canonical_id)
    for path in claim.get("candidate_paths", []):
        for node in path.get("nodes", []):
            if str(node.get("canonical_id")) == str(canonical_id):
                return str(node.get("qualified_name") or canonical_id)
    return str(canonical_id)


def _render_coordinate(
        path: dict,
        source_root: Path | None,
        file: str,
        line: int,
        context_lines: int,
        cache: dict,
) -> list[str]:
    proof, expected_hash = _proof_excerpt(path, file, line)
    context = None
    if source_root is not None:
        context = source_context(
            source_root, file, line=line, context_lines=context_lines,
            cache=cache, expected_sha256=expected_hash)
    shown = context or proof
    lines = [f"- evidence: {file}:{line}"]
    if shown:
        lines.extend(["  ```text", *[f"  {value}" for value in shown.splitlines()],
                      "  ```"])
    else:
        lines.append("  source context unavailable")
    return lines
def _render_claim_witness(witness: dict) -> list[str]:
    witness_id = str(witness.get("id", "<unnamed>"))
    file = str(witness.get("file", ""))
    line_start = int(witness.get("line_start", 0))
    line_end = int(witness.get("line_end", line_start))
    tick = chr(96)
    lines = [
        f"### Claim witness {tick}{witness_id}{tick}",
        "",
        f"- evidence: {file}:{line_start}-{line_end}",
    ]
    excerpts = witness.get("materialization", {}).get("excerpts", [])
    if excerpts:
        excerpt = excerpts[0]
        content = str(excerpt.get("content", ""))
        start = int(excerpt.get("line_start", line_start))
        lines.extend([
            f"- file sha256: {excerpt.get('sha256', '')}",
            "  " + tick * 3 + "text",
            *[
                f"  {start + offset} | {value}"
                for offset, value in enumerate(content.splitlines())
            ],
            "  " + tick * 3,
        ])
    else:
        lines.append("  source witness unavailable")
    required = witness.get("contains", [])
    if required:
        lines.append("- required fragments: " + ", ".join(
            f"{tick}{value}{tick}" for value in required))
    for gap in witness.get("materialization", {}).get("gaps", []):
        lines.append(f"- MATERIALIZATION GAP: {gap}")
    for error in witness.get("proof_errors", []):
        lines.append(f"- PROOF ERROR: {error}")
    lines.append("")
    return lines


def _render_path(
        path: dict,
        *,
        source_root: Path | None,
        context_lines: int,
        cache: dict,
) -> list[str]:
    node_names = [
        str(node.get("qualified_name") or node.get("canonical_id"))
        for node in path.get("nodes", [])]
    search = path.get("search", {})
    lines = [
        f"#### Route `{path.get('id', '')}`",
        "",
        "Chain: " + " -> ".join(node_names),
        "",
        "Traversal: "
        f"{path.get('orientation', '?')}; {path.get('hop_count', 0)} hop(s); "
        f"expanded={search.get('expanded_nodes', '?')}; "
        f"exhausted={bool(search.get('exhausted', False))}",
        "",
    ]
    seen = set()
    for node in path.get("nodes", []):
        coordinate = (
            str(node.get("file", "")), int(node.get("line_start", 0)))
        if coordinate in seen:
            continue
        seen.add(coordinate)
        lines.append(
            f"- definition: {node.get('qualified_name', node.get('canonical_id', ''))}")
        lines.extend(_render_coordinate(
            path, source_root, coordinate[0], coordinate[1],
            context_lines, cache))
    for edge in path.get("edges", []):
        coordinate = (str(edge.get("file", "")), int(edge.get("line", 0)))
        lines.append(
            f"- {edge.get('edge_type', 'edge')} "
            f"{edge.get('caller', edge.get('caller_canonical_id', ''))} -> "
            f"{edge.get('callee', edge.get('callee_canonical_id', ''))} "
            f"({edge.get('traversal', '?')})")
        lines.extend(_render_coordinate(
            path, source_root, coordinate[0], coordinate[1],
            context_lines, cache))
    lines.append("")
    return lines


def render_review_bundle(
        report: dict,
        *,
        source_root: str | Path | None = None,
        context_lines: int = 2,
        selection_limit: int = 3,
        only_claims: set[tuple[int, str]] | None = None,
) -> str:
    """Render ambiguity-first review material with deduplicated alternatives."""
    root = Path(source_root).resolve() if source_root else None
    cache = {}
    all_items = review_items(report)
    items = [
        item for item in all_items
        if only_claims is None or (
            int(item["question"].get("id", 0)),
            str(item["claim"].get("id", ""))) in only_claims]
    lines = [
        "# Gold-chain adjudication",
        "",
        "This file presents candidates; it does not certify correctness.",
        "Accept a selection only after its endpoints, every edge, source "
        "context, and omitted alternatives have been checked.",
        "",
        f"Claims queued: {len(items)}",
        "",
    ]
    for item in items:
        question = item["question"]
        claim = item["claim"]
        diagnostics = item["diagnostics"]
        selections = unique_selections(claim)
        limit = len(selections) if int(selection_limit) <= 0 else int(selection_limit)
        shown = selections[:limit]
        lines.extend([
            f"## Q{question.get('id')} / {claim.get('id', '')}",
            "",
            f"Question: {question.get('question', '')}",
            "",
            f"Assertion to verify: {claim.get('assertion', '')}",
            "",
            "Diagnostics: "
            f"{diagnostics['endpoint_assignments']} endpoint assignment(s), "
            f"{diagnostics['exhausted_paths']} exhausted returned path(s), "
            f"report-cap={diagnostics['at_report_cap']}; "
            f"showing {len(shown)} of {len(selections)} unique selection(s).",
            "",
        ])
        if claim.get("witnesses"):
            lines.extend(["Claim witnesses:", ""])
            for witness in claim.get("witnesses", []):
                lines.extend(_render_claim_witness(witness))
        referenced_paths = []
        for selection in shown:
            lines.extend([
                f"### Selection `{selection['selection_id']}` "
                f"(score={selection.get('score', 0)})",
                "",
                "Endpoints:",
            ])
            for anchor, canonical in selection[
                    "selected_candidate_by_anchor"].items():
                qualified = _endpoint_name(claim, anchor, canonical)
                lines.append(
                    f"- {anchor}: {qualified} [canonical={canonical}]")
            lines.extend([
                "- routes: " + ", ".join(
                    f"`{path_id}`" for path_id in selection["path_ids"]),
                "",
            ])
            referenced_paths.extend(selection["path_ids"])
        path_by_id = {
            str(path.get("id")): path
            for path in claim.get("candidate_paths", [])}
        for path_id in dict.fromkeys(referenced_paths):
            path = path_by_id.get(path_id)
            if path is None:
                lines.extend([
                    f"#### Missing route `{path_id}`",
                    "",
                    "The candidate artifact is internally inconsistent.",
                    "",
                ])
                continue
            lines.extend(_render_path(
                path, source_root=root, context_lines=context_lines,
                cache=cache))
        if len(shown) < len(selections):
            lines.extend([
                f"Omitted selections: {len(selections) - len(shown)}. "
                "Re-render this claim with a larger --selection-limit before "
                "accepting it.",
                "",
            ])
        lines.extend([
            "Decision: pending",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def report_fingerprint(report: dict) -> str:
    """Bind a decisions template to the exact candidate artifact."""
    candidate = copy.deepcopy(report)
    for question in candidate.get("questions", []):
        question["review"] = {}
        for claim in question.get("claims", []):
            claim["review"] = {}
    payload = json.dumps(
        candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def decision_template(report: dict) -> dict:
    """Create an explicitly pending decision document."""
    questions = []
    for question in report.get("questions", []):
        claims = []
        for claim in question.get("claims", []):
            selections = unique_selections(claim)
            claims.append({
                "id": claim.get("id"),
                "status": "pending",
                "selection_id": "",
                "available_selection_ids": [
                    item["selection_id"] for item in selections],
                "claim_correct": None,
                "complete": None,
                "notes": "",
            })
        questions.append({
            "id": question.get("id"),
            "status": "pending",
            "answer": "",
            "notes": "",
            "claims": claims,
        })
    return {
        "candidate_fingerprint": report_fingerprint(report),
        "instructions": (
            "Set a claim to accepted only after reviewing its source. "
            "Choose a listed selection_id and set claim_correct and complete "
            "to true. Accept a question only after all of its claims are accepted."),
        "questions": questions,
    }


def _indexed_by_id(rows: list[dict], *, label: str) -> dict:
    indexed = {}
    for row in rows:
        value = row.get("id")
        if value in indexed:
            raise ValueError(f"duplicate {label} id: {value}")
        indexed[value] = row
    return indexed
def apply_decisions(report: dict, decisions: dict) -> dict:
    """Apply explicit selections without inferring correctness or completeness."""
    expected = decisions.get("candidate_fingerprint")
    if expected and str(expected) != report_fingerprint(report):
        raise ValueError("candidate fingerprint does not match this report")
    reviewed = copy.deepcopy(report)
    questions = _indexed_by_id(
        reviewed.get("questions", []), label="question")
    decision_questions = _indexed_by_id(
        decisions.get("questions", []), label="decision question")
    for question_id, decision in decision_questions.items():
        question = questions.get(question_id)
        if question is None:
            raise ValueError(f"unknown question: {question_id}")
        claims = _indexed_by_id(question.get("claims", []), label="claim")
        claim_decisions = _indexed_by_id(
            decision.get("claims", []), label="decision claim")
        for claim_id, claim_decision in claim_decisions.items():
            claim = claims.get(claim_id)
            if claim is None:
                raise ValueError(
                    f"Q{question_id}: unknown claim: {claim_id}")
            status = str(claim_decision.get("status", "pending"))
            if status == "pending":
                continue
            if status != "accepted":
                raise ValueError(
                    f"Q{question_id}:{claim_id}: unsupported status {status}")
            if claim_decision.get("claim_correct") is not True:
                raise ValueError(
                    f"Q{question_id}:{claim_id}: claim_correct must be true")
            if claim_decision.get("complete") is not True:
                raise ValueError(
                    f"Q{question_id}:{claim_id}: complete must be true")
            selections = {
                item["selection_id"]: item
                for item in unique_selections(claim)}
            chosen_id = str(claim_decision.get("selection_id", ""))
            chosen = selections.get(chosen_id)
            if chosen is None:
                raise ValueError(
                    f"Q{question_id}:{claim_id}: unknown selection {chosen_id}")
            problems = selection_problems(claim, chosen)
            if problems:
                raise ValueError(
                    f"Q{question_id}:{claim_id}: selection is not "
                    "mechanically valid: " + "; ".join(problems))
            claim["review"] = {
                "status": "accepted",
                "selected_candidate_by_anchor": copy.deepcopy(
                    chosen["selected_candidate_by_anchor"]),
                "selected_path_ids": list(chosen["path_ids"]),
                "claim_correct": True,
                "complete": True,
                "notes": str(claim_decision.get("notes", "")),
            }
        question_status = str(decision.get("status", "pending"))
        if question_status not in {"pending", "accepted"}:
            raise ValueError(
                f"Q{question_id}: unsupported status {question_status}")
        answer = str(decision.get("answer", ""))
        notes = str(decision.get("notes", ""))
        if question_status == "accepted":
            if not answer.strip():
                raise ValueError(
                    f"Q{question_id}: accepted question answer is empty")
            missing = [
                claim_id for claim_id, claim in claims.items()
                if claim.get("review", {}).get("status") != "accepted"]
            if missing:
                raise ValueError(
                    f"Q{question_id}: accepted question has pending claims: "
                    + ", ".join(str(value) for value in missing))
            question["review"] = {
                "status": "accepted",
                "answer": answer,
                "notes": notes,
            }
        else:
            previous = question.get("review", {})
            question["review"] = {
                "status": "pending",
                "answer": answer or str(previous.get("answer", "")),
                "notes": notes or str(previous.get("notes", "")),
            }
    counts = review_status(reviewed)
    if counts["accepted_questions"] == counts["questions"]:
        reviewed["status"] = "reviewed-gold"
    return reviewed


def review_status(report: dict) -> dict:
    questions = report.get("questions", [])
    claims = [
        claim for question in questions
        for claim in question.get("claims", [])]
    return {
        "questions": len(questions),
        "accepted_questions": sum(
            question.get("review", {}).get("status") == "accepted"
            for question in questions),
        "claims": len(claims),
        "accepted_claims": sum(
            claim.get("review", {}).get("status") == "accepted"
            for claim in claims),
    }


def parse_claim_filter(value: str | None) -> set[tuple[int, str]] | None:
    if not value:
        return None
    selected = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"invalid selector {item!r}; expected QID:claim-id")
        question, claim = item.split(":", 1)
        if not question.isdigit() or not claim:
            raise ValueError(
                f"invalid selector {item!r}; expected QID:claim-id")
        selected.add((int(question), claim))
    return selected or None


def _configured_source_root(report: dict) -> Path | None:
    source = str(report.get("source", ""))
    try:
        from config import get_config
        configured = get_config().get_all_source_paths().get(source)
    except Exception:
        return None
    return Path(configured) if configured else None


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
def merge_candidate_reports(reports: list[dict]) -> dict:
    """Merge partial candidate reports; later reports replace matching claims."""
    if not reports:
        raise ValueError("at least one candidate report is required")
    merged = copy.deepcopy(reports[0])
    expected_source = str(merged.get("source", ""))
    question_order = []
    questions = {}
    claim_order = {}
    claims = {}
    for report in reports:
        source = str(report.get("source", ""))
        if source != expected_source:
            raise ValueError(
                f"candidate source mismatch: {source!r} != {expected_source!r}")
        for question in report.get("questions", []):
            question_id = int(question["id"])
            if question_id not in questions:
                row = copy.deepcopy(question)
                row["claims"] = []
                questions[question_id] = row
                question_order.append(question_id)
                claim_order[question_id] = []
                claims[question_id] = {}
            else:
                current_text = str(questions[question_id].get("question", ""))
                incoming_text = str(question.get("question", ""))
                if current_text and incoming_text and current_text != incoming_text:
                    raise ValueError(
                        f"Q{question_id}: question text differs across reports")
            for claim in question.get("claims", []):
                claim_id = str(claim["id"])
                if claim_id not in claims[question_id]:
                    claim_order[question_id].append(claim_id)
                claims[question_id][claim_id] = copy.deepcopy(claim)
    for question_id in question_order:
        questions[question_id]["claims"] = [
            claims[question_id][claim_id]
            for claim_id in claim_order[question_id]]
    merged["questions"] = [questions[question_id] for question_id in question_order]
    merged["merge_provenance"] = {
        "report_count": len(reports),
        "question_count": len(question_order),
        "claim_count": sum(len(values) for values in claim_order.values()),
        "overlay_policy": "later-report-wins-per-claim",
    }
    return merged
def compact_reviewed_report(report: dict) -> dict:
    """Keep only accepted endpoints and compiler paths needed for scoring."""
    if report.get("status") != "reviewed-gold":
        raise ValueError("compact input must be reviewed-gold")
    compact = copy.deepcopy(report)
    compact["format"] = "compact-reviewed-gold-v1"
    compact.pop("source_root", None)
    provenance = compact.get("provenance")
    if isinstance(provenance, dict) and provenance.get("database"):
        provenance["database"] = Path(str(provenance["database"])).name
    for question in compact.get("questions", []):
        if question.get("review", {}).get("status") != "accepted":
            raise ValueError(
                f"Q{question.get('id')}: question is not accepted")
        for claim in question.get("claims", []):
            review = claim.get("review", {})
            if review.get("status") != "accepted":
                raise ValueError(
                    f"Q{question.get('id')}:{claim.get('id')}: claim is not accepted")
            selected_candidates = review.get(
                "selected_candidate_by_anchor", {})
            for anchor in claim.get("anchors", []):
                anchor_name = str(anchor.get("anchor") or "")
                selected = selected_candidates.get(anchor_name)
                candidates = [
                    candidate for candidate in anchor.get("candidates", [])
                    if selected in {
                        candidate.get("canonical_id"),
                        candidate.get("qualified_name"),
                    }
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f"Q{question.get('id')}:{claim.get('id')}: "
                        f"selected candidate for {anchor_name} is not unique")
                anchor["candidates"] = candidates
            available_paths = {
                str(candidate.get("id")): candidate
                for candidate in claim.get("candidate_paths", [])
            }
            selected_path_ids = [
                str(value) for value in review.get("selected_path_ids", [])]
            if any(path_id not in available_paths
                   for path_id in selected_path_ids):
                raise ValueError(
                    f"Q{question.get('id')}:{claim.get('id')}: "
                    "selected path is unavailable")
            claim["candidate_paths"] = [
                available_paths[path_id] for path_id in selected_path_ids]
            for key in (
                    "candidate_coverage", "coherent_path_sets",
                    "unresolved_anchors"):
                claim.pop(key, None)
    return compact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render")
    render.add_argument("--candidates", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--source-root", default="")
    render.add_argument("--context-lines", type=int, default=2)
    render.add_argument("--selection-limit", type=int, default=3)
    render.add_argument("--only-claims", default="")

    template = subparsers.add_parser("template")
    template.add_argument("--candidates", required=True)
    template.add_argument("--out", required=True)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--candidates", required=True)
    apply_parser.add_argument("--decisions", required=True)
    apply_parser.add_argument("--out", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--candidates", required=True, nargs="+")
    merge.add_argument("--out", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--candidates", required=True)
    compact_parser = subparsers.add_parser("compact")
    compact_parser.add_argument("--reviewed", required=True)
    compact_parser.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "compact":
        compact = compact_reviewed_report(_load(args.reviewed))
        Path(args.out).write_text(json.dumps(compact, indent=2) + "\n")
        print(f"compact reviewed gold -> {args.out}")
        return 0
    if args.command == "merge":
        reports = [_load(path) for path in args.candidates]
        merged = merge_candidate_reports(reports)
        Path(args.out).write_text(json.dumps(merged, indent=2) + "\n")
        provenance = merged["merge_provenance"]
        question_count = provenance["question_count"]
        claim_count = provenance["claim_count"]
        print(
            f"merged -> {args.out}; "
            f"{question_count} questions, {claim_count} claims")
        return 0
    if args.command == "render":
        report = _load(args.candidates)
        root = (
            Path(args.source_root) if args.source_root
            else _configured_source_root(report))
        try:
            only_claims = parse_claim_filter(args.only_claims)
        except ValueError as error:
            parser.error(str(error))
        Path(args.out).write_text(render_review_bundle(
            report, source_root=root, context_lines=args.context_lines,
            selection_limit=args.selection_limit, only_claims=only_claims))
        print(f"review bundle -> {args.out}")
        return 0
    if args.command == "template":
        report = _load(args.candidates)
        Path(args.out).write_text(
            json.dumps(decision_template(report), indent=2) + "\n")
        print(f"decision template -> {args.out}")
        return 0
    if args.command == "apply":
        report = _load(args.candidates)
        decisions = _load(args.decisions)
        reviewed = apply_decisions(report, decisions)
        Path(args.out).write_text(
            json.dumps(reviewed, indent=2) + "\n")
        counts = review_status(reviewed)
        print(
            f"reviewed -> {args.out}; "
            f"{counts['accepted_questions']}/{counts['questions']} questions, "
            f"{counts['accepted_claims']}/{counts['claims']} claims accepted")
        return 0
    counts = review_status(_load(args.candidates))
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
