#!/usr/bin/env python3
"""Audit the exact source evidence persisted by a paid Ariadne answer artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from replay_formulation_prompt import fragment_recall, recorded_selection

def citation_from_occurrence(occurrence, *, source: str):
    """Restore one compiler-derived citation from its persisted 10-field key."""
    if len(occurrence) != 10:
        raise ValueError("recorded occurrence must contain exactly 10 fields")
    (qualified_name, file, line_start, line_end, parent_qualified_name,
     call_site_file, call_site_line, relation, hop, stop_reason) = occurrence
    from library.structural_assembly import StructuralCitation
    return StructuralCitation(
        qualified_name=str(qualified_name), file=str(file),
        line_start=int(line_start), line_end=int(line_end), source_name=source,
        parent_qualified_name=str(parent_qualified_name),
        call_site_file=str(call_site_file), call_site_line=int(call_site_line),
        relation=str(relation), hop=int(hop), stop_reason=str(stop_reason))


def artifact_citations(answer: dict, *, source: str) -> list:
    """Restore only unique occurrences belonging to the persisted selected routes."""
    selection = recorded_selection(answer)
    return [
        citation_from_occurrence(occurrence, source=source)
        for occurrence in selection.occurrence_keys]

def exact_replay(answer: dict, question: dict, *, source: str, body_symbols=None,
                 complete_transitions=False, semantic_slices=False,
                 compact_ledger=False) -> tuple[dict, str]:
    """Replay persisted occurrences through the production hydration policy."""
    citations = artifact_citations(answer, source=source)
    if not citations:
        raise ValueError("no selected artifact occurrences available")

    from ariadne_mcp.service import AriadneService
    from library.chain_bundle import BundleHop
    from library.chain_menu import (
        DefinitionBodySelection,
        complete_definition_body_selection,
        complete_selection_with_body_dependencies,
        definition_body_menu,
        fetch_selected,
        hydrate_selected_hops)
    from library.chain_story import build_story_ir, render_story_evidence
    from library.source_chunks import source_chunk_values

    service = AriadneService.get()
    source_root = service.config.get_all_source_paths().get(source)
    if source_root is None:
        raise ValueError(f"{source}: source root unavailable")
    selection = recorded_selection(answer)
    if body_symbols is None:
        recorded_bodies = tuple(answer.get("selected_body_symbols") or ())
        body_symbols = recorded_bodies or None
    seed_hops = tuple(BundleHop(citation=citation) for citation in citations)
    if body_symbols is not None and complete_transitions:
        body_menu = definition_body_menu(seed_hops, selection)
        body_symbols = complete_definition_body_selection(
            body_menu,
            DefinitionBodySelection(symbols=tuple(body_symbols))).symbols
    question_text = str(
        answer.get("question") or question.get("question") or "").strip()
    coverage_plan = str(
        ((answer.get("graph_diagnostics") or {}).get("clew_selection") or {})
        .get("coverage_plan") or "").strip()
    body_query = "\n\n".join(
        value for value in (question_text, coverage_plan) if value)
    selected_hops, source_gaps = hydrate_selected_hops(
        service.library, seed_hops, selection, source=source,
        source_root=source_root,
        definition_body_symbols=body_symbols,
        definition_body_query=(
            body_query if semantic_slices and body_query else None),
        reference_query=question_text)
    selection = complete_selection_with_body_dependencies(
        selection, selected_hops, body_symbols)
    fetched = fetch_selected(service.library, selection, selected_hops)
    story = build_story_ir(selected_hops, selection, fetched)
    prompt = render_story_evidence(story, compact_source=compact_ledger)
    ledger_text = "\n".join(source_chunk_values(story.chunks).values())
    hydrated_names = {
        hop.citation.qualified_name for hop in selected_hops}
    excerpt_kinds = Counter(
        excerpt.kind for node in story.nodes for excerpt in node.excerpts)
    report = {
        "id": int(answer.get("id") or question.get("id")),
        "replay_mode": "artifact-exact",
        "compact_ledger": bool(compact_ledger),
        "selected_routes": len(selection.route_ids),
        "selected_symbols": len(selection.symbols),
        "selected_definition_bodies": list(
            selection.symbols if body_symbols is None else body_symbols),
        "artifact_occurrences": len(citations),
        "completed_occurrences": len(selection.occurrence_keys),
        "curated_hops": len(selected_hops),
        "hydrated_definitions": len(fetched.definitions),
        "missing_selected_symbols": sorted(
            set(selection.symbols) - hydrated_names),
        "story_nodes": len(story.nodes),
        "story_edges": len(story.edges),
        "story_chunks": len(story.chunks),
        "source_excerpts": sum(
            len(node.excerpts) for node in story.nodes),
        "source_excerpt_kinds": dict(sorted(excerpt_kinds.items())),
        "source_gaps": list(source_gaps),
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "fragment_recall": fragment_recall(prompt, question),
        "chunk_fragment_recall": fragment_recall(ledger_text, question),
        "limitations": [
            "historical artifacts retain section labels but not document/index identities; background sections are omitted",
            "proof auditing uses persisted source occurrences only; generated descriptions never satisfy fragments",
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
        "--body-symbols",
        help="comma-separated definition bodies to materialize")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-out")
    parser.add_argument("--complete-transitions", action="store_true", help="include unselected intermediate route-transition bodies")
    parser.add_argument("--semantic-slices", action="store_true", help="slice selected definition bodies around question and obligation terms")
    parser.add_argument("--compact-ledger", action="store_true", help="render the exact-source chunk ledger instead of full bodies; measures narration-free recall")
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

    body_symbols = (tuple(value for value in args.body_symbols.split(",") if value)
                    if args.body_symbols else None)
    output = []
    prompts = {}
    for qid in sorted(wanted):
        print(f"id {qid} exact prompt audit starting", flush=True)
        if body_symbols is None:
            report, prompt = exact_replay(
                answers[qid], gold[qid], source=args.source, complete_transitions = args.complete_transitions, semantic_slices = args.semantic_slices, compact_ledger = args.compact_ledger)
        else:
            report, prompt = exact_replay(
                answers[qid], gold[qid], source=args.source,
                body_symbols=body_symbols, complete_transitions = args.complete_transitions, semantic_slices = args.semantic_slices, compact_ledger = args.compact_ledger)
        output.append(report)
        prompts[qid] = prompt
        recall = report["fragment_recall"]
        print(
            f"id {qid}: gold claims {recall['claims_passed']}/{recall['claims']}; "
            f"fragments {recall['fragments_found']}/{recall['fragments']}; "
            f"{report['artifact_occurrences']} artifact occurrences, "
            f"{report['source_excerpts']} excerpts", flush=True)
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    if args.prompt_out:
        Path(args.prompt_out).write_text(next(iter(prompts.values())) + "\n")
    print(f"exact formulation prompt audit -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
