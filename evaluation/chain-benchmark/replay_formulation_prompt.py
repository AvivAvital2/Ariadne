#!/usr/bin/env python3
"""Replay Ariadne's selected formulation prompt without an LLM or embeddings."""
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

import profile_evidence_walk as profiler


def recorded_selection(answer: dict):
    """Rebuild the exact route occurrence selection persisted by the arm."""
    from library.chain_menu import Selection

    route_ids = tuple(dict.fromkeys(
        str(route_id) for route_id in answer.get("selected_route_ids", ())))
    occurrences = []
    known = set()
    candidates = answer.get("route_candidate_occurrences") or {}
    for route_id in route_ids:
        for occurrence in candidates.get(route_id, ()):
            key = tuple(occurrence)
            if key not in known:
                known.add(key)
                occurrences.append(key)
    return Selection(
        symbols=list(dict.fromkeys(
            str(symbol) for symbol in answer.get("selected_symbols", ()))),
        route_ids=route_ids,
        section_ids=tuple(dict.fromkeys(
            str(section) for section in answer.get("selected_section_ids", ()))),
        occurrence_keys=tuple(occurrences))


def fragment_recall(prompt: str, question: dict) -> dict:
    """Require every reviewed literal fragment to occur in the rendered prompt."""
    claims = []
    found_total = 0
    fragment_total = 0
    for claim in question.get("claims", ()):
        witnesses = []
        for witness in claim.get("witnesses", ()):
            fragments = [str(value) for value in witness.get("contains", ())]
            missing = [fragment for fragment in fragments if fragment not in prompt]
            found = len(fragments) - len(missing)
            found_total += found
            fragment_total += len(fragments)
            witnesses.append({
                "id": witness.get("id"),
                "found": found,
                "fragments": len(fragments),
                "missing": missing,
                "passed": not missing,
            })
        claims.append({
            "id": claim.get("id"),
            "witnesses": witnesses,
            "passed": all(witness["passed"] for witness in witnesses),
        })
    return {
        "claims_passed": sum(bool(claim["passed"]) for claim in claims),
        "claims": len(claims),
        "fragments_found": found_total,
        "fragments": fragment_total,
        "details": claims,
    }
def scoped_all_selection(menu, question: str, *, max_routes: int = 32):
    """Apply production scope ranking, then preserve every retained route."""
    from library.chain_menu import all_route_selection, scope_route_menu
    scoped = scope_route_menu(menu, question, max_families=max_routes)
    return all_route_selection(scoped), scoped


def replay_prompt(answer: dict, question: dict, *, source: str, selection_mode: str = "recorded", max_routes: int = 32) -> tuple[dict, str]:
    """Rebuild the post-selection evidence IR and audit reviewed fragments."""
    specs = profiler.recorded_route_specs(answer)
    if not specs:
        raise ValueError("no recorded routes available")

    from ariadne_mcp.service import AriadneService
    from library.chain_answer import evidence_for
    from library.chain_menu import (
        Selection, _occurrence_key, fetch_selected, hydrate_selected_hops)
    from library.chain_story import build_story_ir, render_story_evidence

    service = AriadneService.get()
    matches = profiler._matches(specs, question, source)
    evidence = evidence_for(
        service.library, [],
        question=str(answer.get("question") or question.get("question") or ""),
        source=source, clew_matches=matches,
        positioning_documents=(), defer_source=True)
    if selection_mode == "scoped-all":
        from library.chain_menu import route_menu_for
        route_menu = route_menu_for(service.library, evidence.hops, source=source)
        selection, _scoped_menu = scoped_all_selection(
            route_menu, str(answer.get("question") or question.get("question") or ""),
            max_routes=max_routes)
    elif selection_mode == "recorded":
        selection = recorded_selection(answer)
    else:
        raise ValueError(f"unknown selection mode: {selection_mode}")
    available = {_occurrence_key(hop) for hop in evidence.hops}
    requested = tuple(selection.occurrence_keys)
    matched = tuple(key for key in requested if key in available)
    if requested and not matched:
        raise ValueError(
            "none of the recorded occurrences exist in the replayed evidence graph")
    effective = Selection(
        symbols=list(selection.symbols), route_ids=selection.route_ids,
        section_ids=selection.section_ids,
        occurrence_keys=matched)
    source_root = str(
        service.config.get_all_source_paths().get(source) or "") or None
    selected_body_symbols = None
    if selection_mode == "scoped-all":
        from library.chain_menu import (
            all_definition_body_selection, definition_body_menu)
        selected_body_symbols = all_definition_body_selection(
            definition_body_menu(evidence.hops, effective)).symbols
    hydrated_hops, source_gaps = hydrate_selected_hops(
        service.library, evidence.hops, effective, source=source,
        source_root=source_root, definition_body_symbols = selected_body_symbols)
    fetched = fetch_selected(service.library, effective, hydrated_hops)
    story = build_story_ir(hydrated_hops, effective, fetched)
    prompt = render_story_evidence(story)
    hydrated_names = {
        hop.citation.qualified_name for hop in hydrated_hops}
    report = {
        "selection_mode": selection_mode, "selected_body_symbols": list(selected_body_symbols or ()), "id": int(answer.get("id") or question.get("id")),
        "requested_routes": len(selection.route_ids),
        "requested_symbols": len(selection.symbols),
        "requested_occurrences": len(requested),
        "matched_occurrences": len(matched),
        "hydrated_hops": len(hydrated_hops),
        "hydrated_definitions": len(fetched.definitions),
        "missing_selected_symbols": sorted(
            set(selection.symbols) - hydrated_names),
        "story_nodes": len(story.nodes),
        "story_edges": len(story.edges),
        "source_excerpts": sum(
            len(node.excerpts) for node in story.nodes),
        "source_gaps": list(source_gaps),
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "fragment_recall": fragment_recall(prompt, question),
        "limitations": [
            "historical artifacts retain section labels but not their document/index mapping; background sections are omitted",
            "proof auditing uses source excerpts only; generated descriptions and sections never satisfy fragments",
        ],
    }
    return report, prompt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answers", default=str(HERE / "q4-trace-answers.baseline.json"))
    parser.add_argument(
        "--gold", default=str(HERE / "gold-chain-reviewed.json"))
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--source", default="databricks")
    parser.add_argument(
        "--selection", choices=("recorded", "scoped-all"), default="recorded")
    parser.add_argument("--max-routes", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-out")
    args = parser.parse_args(argv)

    answers = {int(row["id"]): row for row in json.loads(
        Path(args.answers).read_text())}
    gold_payload = json.loads(Path(args.gold).read_text())
    gold = {int(row["id"]): row for row in gold_payload.get("questions", ())}
    wanted = set(answers) & set(gold)
    if args.only:
        wanted &= {int(value) for value in args.only.split(",")}
    if args.prompt_out and len(wanted) != 1:
        raise ValueError("--prompt-out requires exactly one question")

    output = []
    prompts = {}
    for qid in sorted(wanted):
        print(f"id {qid} prompt replay starting", flush=True)
        report, prompt = replay_prompt(
            answers[qid], gold[qid], source=args.source, selection_mode = args.selection, max_routes = args.max_routes)
        output.append(report)
        prompts[qid] = prompt
        recall = report["fragment_recall"]
        print(
            f"id {qid}: gold claims {recall['claims_passed']}/{recall['claims']}; "
            f"fragments {recall['fragments_found']}/{recall['fragments']}; "
            f"{report['story_nodes']} nodes, {report['source_excerpts']} excerpts",
            flush=True)
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    if args.prompt_out:
        Path(args.prompt_out).write_text(next(iter(prompts.values())) + "\n")
    print(f"formulation prompt replay -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
