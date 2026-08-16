#!/usr/bin/env python3
"""One-question live ablation for SCIP-only Leiden seed strands.

This is intentionally outside Ariadne's production request path.  Before a
model is called it uses only an external structural-family SQLite database,
precomputed source-token facts, and the raw SCIP store.  The first completion
selects two or three bounded directed strands.  Only then are exact source
coordinates materialized for the formulation completion.

Reviewed gold is loaded only after the answer is complete, by the ordinary
strict scorer.  It never informs the question frame, the Leiden lookup, cards,
selection, source reads, or prompt construction.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
QUESTIONS = ROOT / "evaluation/spool-clean-room/questions_debcrumb_ask.json"
STRUCTURAL = HERE / "scip_structural_families.py"
ARM = ROOT / "evaluation/spool-clean-room/ariadne_arm.py"
MEASURE = HERE / "measure_ariadne.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _question(question_id: int, path: Path) -> str:
    for row in json.loads(path.read_text()):
        if int(row.get("id", -1)) == question_id:
            text = str(row.get("after") or row.get("question") or "").strip()
            if text:
                return text
    raise ValueError(f"question {question_id} is absent from {path}")


def _cards(strands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give the selector compact identities, never source bodies."""
    cards = []
    for index, strand in enumerate(strands, start=1):
        cards.append({
            "id": f"S{index}",
            "strand_id": strand["strand_id"],
            "family_id": strand["family_id"],
            "seed": {
                key: strand[key]
                for key in ("symbol_id", "qualified_name", "file", "line_start", "line_end")
            },
            "roles": strand["matches"],
            "directed_edges": strand["directed_edges"],
        })
    return cards


def _symbol(conn: sqlite3.Connection, canonical_id: str) -> dict[str, Any] | None:
    row = conn.execute("""
        SELECT canonical_id, qualified_name, file, line_start, line_end
        FROM scip_symbols WHERE canonical_id=?
    """, (canonical_id,)).fetchone()
    if row is None:
        return None
    return {
        "symbol_id": str(row[0]), "qualified_name": str(row[1]), "file": str(row[2]),
        "line_start": int(row[3]), "line_end": int(row[4]),
    }


def _owner(qualified_name: str) -> str:
    return qualified_name.rpartition(".")[0]


def _local_terms(qualified_name: str, probe) -> set[str]:
    """Terms from a symbol's local type/member name, never its package."""
    return set(probe._terms(".".join(str(qualified_name).split(".")[-2:])))


def _planning_bridges(db_path: Path, cards: list[dict[str, Any]], probe) -> None:
    """Attach bounded, exact planner-to-executor type bridges to cards.

    A compiler graph commonly separates a logical-plan builder from its physical
    executor: the builder returns logical type L, a strategy pattern-matches L,
    and the strategy constructs the executor.  This is neither a guessed call
    nor a family-wide expansion.  All four links must exist in raw SCIP, and
    only the immediate builder caller in the same lexical owner is retained.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for card in cards:
            seed = card["seed"]
            seed_terms = _local_terms(str(seed["qualified_name"]), probe)
            bridges: list[dict[str, Any]] = []
            strategies = conn.execute("""
                SELECT DISTINCT caller_canonical_id, file, line
                FROM scip_edges
                WHERE edge_type='type_ref' AND callee_canonical_id=?
                ORDER BY caller_canonical_id, file, line
            """, (seed["symbol_id"],)).fetchall()
            for strategy_id, strategy_file, strategy_line in strategies:
                strategy = _symbol(conn, str(strategy_id))
                if strategy is None or not probe._is_production_path(strategy["file"]):
                    continue
                logical_rows = conn.execute("""
                    SELECT DISTINCT callee_canonical_id, file, line
                    FROM scip_edges
                    WHERE edge_type='type_ref' AND caller_canonical_id=?
                      AND callee_canonical_id != ?
                    ORDER BY callee_canonical_id, file, line
                """, (strategy["symbol_id"], seed["symbol_id"])).fetchall()
                for logical_id, strategy_logical_file, strategy_logical_line in logical_rows:
                    logical = _symbol(conn, str(logical_id))
                    if logical is None or not (seed_terms & _local_terms(logical["qualified_name"], probe)):
                        continue
                    builders = conn.execute("""
                        SELECT DISTINCT caller_canonical_id, file, line
                        FROM scip_edges
                        WHERE edge_type='type_ref' AND callee_canonical_id=?
                          AND caller_canonical_id != ?
                        ORDER BY caller_canonical_id, file, line
                    """, (logical["symbol_id"], strategy["symbol_id"])).fetchall()
                    for builder_id, builder_logical_file, builder_logical_line in builders:
                        builder = _symbol(conn, str(builder_id))
                        if builder is None or not probe._is_production_path(builder["file"]):
                            continue
                        planners = conn.execute("""
                            SELECT DISTINCT caller_canonical_id, file, line
                            FROM scip_edges
                            WHERE edge_type='call' AND callee_canonical_id=?
                            ORDER BY caller_canonical_id, file, line
                        """, (builder["symbol_id"],)).fetchall()
                        for planner_id, planner_file, planner_line in planners:
                            planner = _symbol(conn, str(planner_id))
                            if (planner is None
                                    or not probe._is_production_path(planner["file"])
                                    or _owner(planner["qualified_name"]) != _owner(builder["qualified_name"])):
                                continue
                            bridges.append({
                                "planner": planner, "builder": builder,
                                "logical": logical, "strategy": strategy,
                                "edges": [
                                    {"caller_name": planner["qualified_name"],
                                     "callee_name": builder["qualified_name"], "relation": "call",
                                     "file": str(planner_file), "line": int(planner_line)},
                                    {"caller_name": builder["qualified_name"],
                                     "callee_name": logical["qualified_name"], "relation": "type_ref",
                                     "file": str(builder_logical_file), "line": int(builder_logical_line)},
                                    {"caller_name": strategy["qualified_name"],
                                     "callee_name": logical["qualified_name"], "relation": "type_ref",
                                     "file": str(strategy_logical_file), "line": int(strategy_logical_line)},
                                    {"caller_name": strategy["qualified_name"],
                                     "callee_name": seed["qualified_name"], "relation": "type_ref",
                                     "file": str(strategy_file), "line": int(strategy_line)},
                                ],
                            })
            unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
            for bridge in bridges:
                key = tuple(bridge[key]["symbol_id"] for key in ("planner", "builder", "logical", "strategy"))
                unique.setdefault(key, bridge)
            card["planning_bridges"] = list(unique.values())[:2]
    finally:
        conn.close()


def _short_name(qualified_name: str) -> str:
    """Compact a selector card without changing the exact stored identity."""
    parts = str(qualified_name).split(".")
    return ".".join(parts[-3:]) if len(parts) > 3 else str(qualified_name)


def _selector_prompt(question: str, frame: dict[str, Any], cards: list[dict[str, Any]]) -> str:
    blocks = []
    for card in cards:
        seed = card["seed"]
        edge_lines = [
            f'  {_short_name(edge["caller_name"])} {edge["relation"]} '
            f'{_short_name(edge["callee_name"])}'
            for edge in card["directed_edges"]
        ]
        for bridge in card.get("planning_bridges", []):
            edge_lines.append("  planning/type bridge (not a direct call): "
                              + " -> ".join(_short_name(edge["caller_name"])
                                            for edge in bridge["edges"][:1])
                              + " -> " + _short_name(bridge["builder"]["qualified_name"])
                              + f" --type_ref { _short_name(bridge['logical']['qualified_name'])} --> "
                              + _short_name(seed["qualified_name"]))
        blocks.append(
            f'{card["id"]}: seed {seed["qualified_name"]}\n'
            + "\n".join(edge_lines))
    return (
        "Select compiler-verified directed strands for a code explanation. "
        "Do not answer the user.  Do not invent or infer symbols.\n\n"
        f"Question:\n{question}\n\n"
        f"Deterministic question frame: components={frame['components']}; "
        f"behaviour_roles={frame['roles']}.\n\n"
        "Each card is one exact SCIP seed with at most one ownership bridge and "
        "two one-edge call continuations.  Select the two or three cards that "
        "jointly provide a complete directed explanation for every requested "
        "component and behaviour.  If none does, reply exactly `KEEP: NONE`.\n"
        "Reply with exactly one line using card ids; for example "
        "`KEEP: S1 S3`.\n\n"
        + "\n\n".join(blocks)
    )


def _parse_selection(reply: str, cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = re.findall(r"(?m)^\s*KEEP\s*:\s*(.*?)\s*$", reply or "")
    if len(lines) != 1:
        raise ValueError("selector did not return one KEEP line")
    selected_ids = lines[0].split()
    if selected_ids == ["NONE"]:
        raise ValueError("selector reports no complete directed explanation")
    by_id = {card["id"]: card for card in cards}
    if not 2 <= len(selected_ids) <= 3:
        raise ValueError("selector must keep two or three strand cards")
    if len(set(selected_ids)) != len(selected_ids) or any(item not in by_id for item in selected_ids):
        raise ValueError("selector returned duplicate or unknown card ids")
    # Selection order has no semantic meaning: a model may naturally name the
    # execution consequence before its entry card.  Normalize it so evidence
    # and traces stay byte-stable without rejecting an otherwise valid choice.
    selected = set(selected_ids)
    return [card for card in cards if card["id"] in selected]


def _citations(cards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Materialize only selected seeds and exact selected-edge windows."""
    citations: list[dict[str, Any]] = []
    symbols: set[str] = set()
    for card in cards:
        seed = card["seed"]
        citations.append({
            "file": seed["file"], "line": int(seed["line_start"]),
            # A selected seed is already an exact SCIP definition, so its
            # bounded body—not an arbitrary first excerpt—is the evidence
            # projection.  This preserves later join/classification witnesses
            # while refusing pathological generated/monolithic extents.
            "line_end": min(int(seed["line_end"]), int(seed["line_start"]) + 511),
            "qualified_name": seed["qualified_name"],
        })
        symbols.add(str(seed["qualified_name"]))
        for edge in card["directed_edges"]:
            citations.append({
                "file": edge["file"], "line": int(edge["line"]),
                "line_end": int(edge["line"]) + 6,
                "qualified_name": edge["caller_name"],
            })
            symbols.update((str(edge["caller_name"]), str(edge["callee_name"])))
        for bridge in card.get("planning_bridges", []):
            # The bridge is discovered before selection but its source is read
            # only after this card is selected.  Keep two exact definition
            # bodies plus their four compiler sites; no strategy/file body is
            # widened merely because it supplies a type intersection.
            for node in (bridge["planner"], bridge["builder"]):
                citations.append({
                    "file": node["file"], "line": int(node["line_start"]),
                    "line_end": min(int(node["line_end"]), int(node["line_start"]) + 511),
                    "qualified_name": node["qualified_name"],
                })
                symbols.add(str(node["qualified_name"]))
            for edge in bridge["edges"]:
                citations.append({
                    "file": edge["file"], "line": int(edge["line"]),
                    "line_end": int(edge["line"]) + 6,
                    "qualified_name": edge["caller_name"],
                })
                symbols.update((str(edge["caller_name"]), str(edge["callee_name"])))
    deduped: list[dict[str, Any]] = []
    seen = set()
    for citation in citations:
        key = citation["file"], citation["line"], citation["line_end"], citation["qualified_name"]
        if key not in seen:
            seen.add(key)
            deduped.append(citation)
    return deduped, sorted(symbols)


def _display_provenance(cards: list[dict[str, Any]]) -> str:
    """Render selected compiler facts after formulation without another model call."""
    lines = ["", "Evidence paths (selected raw SCIP relations):"]
    seen_relations: set[tuple[str, str, str, str, int]] = set()
    seen_bridges: set[tuple[str, str, str, str]] = set()
    for card in cards:
        seed = card["seed"]
        lines.append(
            f"- {card['id']} definition: {_short_name(seed['qualified_name'])} "
            f"({seed['file']}:{seed['line_start']}-{seed['line_end']})")
        for edge in card["directed_edges"]:
            key = (str(edge["caller_name"]), str(edge["relation"]),
                   str(edge["callee_name"]), str(edge["file"]), int(edge["line"]))
            if key in seen_relations:
                continue
            seen_relations.add(key)
            lines.append(
                f"  {_short_name(edge['caller_name'])} --{edge['relation']}→ "
                f"{_short_name(edge['callee_name'])} ({edge['file']}:{edge['line']})")
        for bridge in card.get("planning_bridges", []):
            key = tuple(str(bridge[name]["symbol_id"])
                        for name in ("planner", "builder", "logical", "strategy"))
            if key in seen_bridges:
                continue
            seen_bridges.add(key)
            planner, builder = bridge["planner"], bridge["builder"]
            logical, strategy = bridge["logical"], bridge["strategy"]
            sites = ", ".join(f"{edge['file']}:{edge['line']}"
                              for edge in bridge["edges"])
            lines.append(
                "  bridge: "
                f"{_short_name(planner['qualified_name'])} --call→ "
                f"{_short_name(builder['qualified_name'])} --type_ref→ "
                f"{_short_name(logical['qualified_name'])} ←type_ref-- "
                f"{_short_name(strategy['qualified_name'])} --type_ref→ "
                f"{_short_name(seed['qualified_name'])} ({sites})")
    return "\n".join(lines)


def _formulation_prompt(question: str, evidence: str) -> str:
    return (
        "Answer the code question using only the exact source excerpts below. "
        "Do not invent symbols or behaviours. Return 2 to 4 complete plain-text "
        "sentences, at most 140 words total. Start with the direct answer, cover "
        "every distinct subject or alternative the user asks about, and explain "
        "only the source-supported causal order from input or trigger through "
        "transformation, decision, and outcome when those stages apply. If the "
        "question asks for a comparison, use the final sentence for the difference. "
        "If the excerpts do not establish a requested link, say so plainly. Do not "
        "use Markdown code fences, bullets, tables, or verbatim source.\n\n"
        f"Question:\n{question}\n\n"
        f"Selected compiler/source evidence:\n{evidence}"
    )


def _close_unterminated_fence(text: str) -> str:
    """Keep a capped model completion from swallowing the evidence appendix."""
    return text + "\n```" if text.count("```") % 2 else text


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _retrieve(args, structural, probe, question: str) -> dict[str, Any]:
    family_conn = sqlite3.connect(f"file:{args.families}?mode=ro", uri=True)
    fact_conn = sqlite3.connect(f"file:{args.facts}?mode=ro", uri=True)
    source_conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        return structural._query(
            family_conn, question, probe, family_k=args.card_k,
            edge_k=args.edge_k, facts=fact_conn, fact_k=args.fact_k,
            owner_bridge_k=args.owner_bridge_k, source_sites=source_conn)
    finally:
        source_conn.close()
        fact_conn.close()
        family_conn.close()


async def _live(args, question: str, cards: list[dict[str, Any]], frame: dict[str, Any], *,
                score_answer: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    from llm import capture_completion_usage, chat_complete

    arm = _load("_leiden_arm", ARM)
    measure = _load("_leiden_measure", MEASURE) if score_answer else None
    started = time.monotonic()
    with capture_completion_usage() as usage:
        reply = await chat_complete(
            [{"role": "system", "content": "Return only the requested selector grammar."},
             {"role": "user", "content": _selector_prompt(question, frame, cards)}],
            max_tokens=args.selector_max_tokens, phase="scip-leiden-strand-select")
        selected = _parse_selection(reply, cards)
        citations, symbols = _citations(selected)
        evidence, files, hashes = arm._materialize_citations(citations, args.corpus)
        if not evidence:
            raise RuntimeError("selected SCIP source coordinates did not materialize")
        answer = _close_unterminated_fence(await chat_complete(
            [{"role": "system", "content": "Be precise and evidence-bound."},
             {"role": "user", "content": _formulation_prompt(question, evidence)}],
            max_tokens=args.formulation_max_tokens, phase="scip-leiden-formulation"))
    row = {
        "id": args.question_id, "question": question, "benchmark_source": args.source,
        "benchmark_corpus": str(args.corpus.resolve()),
        # The strict benchmark needs the exact materialized source as proof;
        # a product response would render ``answer`` alone, without this annex.
        "answer": answer + "\n\nVERBATIM SCIP EVIDENCE\n" + evidence,
        "display_answer": answer + _display_provenance(selected),
        "sources": [], "confidence": "experimental",
        "chain_confidence": "experimental", "files_read": files, "file_hashes": hashes,
        "chain_files": files, "citations": citations, "selected_symbols": symbols,
        "hydrated_symbols": symbols, "tool_calls": [], "llm_calls": len(usage),
        "elapsed_s": round(time.monotonic() - started, 1), **arm._usage_summary(usage),
    }
    score = None
    if score_answer:
        assert measure is not None
        gold = json.loads(args.gold.read_text())
        score = measure.score_answers([row], gold, args.corpus, only={args.question_id})
    report = {
        "schema": "scip-leiden-live-ablation/v1", "question_id": args.question_id,
        "frame": frame, "cards": cards, "selector_reply": reply,
        "selected_card_ids": [card["id"] for card in selected],
        "source_materialization": {"citation_count": len(citations), "files": files,
                                   "evidence_chars": len(evidence), "file_hashes": hashes},
        "usage": row["usage_by_phase"], "score": score,
    }
    return row, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "ariadne.db")
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--question-id", type=int, default=4)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--gold", type=Path, default=HERE / "gold-chain-reviewed.json")
    parser.add_argument("--answer-only", action="store_true",
                        help="do not read reviewed gold or score this live answer")
    parser.add_argument("--corpus", type=Path, default=ROOT / "spool-corpus")
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--card-k", type=int, default=12)
    parser.add_argument("--edge-k", type=int, default=12)
    parser.add_argument("--fact-k", type=int, default=32)
    parser.add_argument("--owner-bridge-k", type=int, default=4)
    parser.add_argument("--selector-max-tokens", type=int, default=64)
    parser.add_argument("--formulation-max-tokens", type=int, default=512)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(value <= 0 for value in (args.card_k, args.edge_k, args.fact_k, args.owner_bridge_k,
                                    args.selector_max_tokens, args.formulation_max_tokens)):
        parser.error("all numeric bounds must be positive")
    if args.dry_run:
        if args.answers or args.report:
            parser.error("--dry-run does not create answer/report files")
    elif not args.answers or not args.report:
        parser.error("live runs require --answers and --report")
    required_files = (args.db, args.families, args.facts, args.questions)
    if not args.answer_only:
        required_files += (args.gold,)
    if not all(path.is_file() for path in required_files):
        parser.error("--db, --families, --facts, and --questions must exist; --gold is required unless --answer-only")
    if not args.corpus.is_dir():
        parser.error("--corpus must be a directory")
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("ANTHROPIC_API_KEY is required for the configured provider")

    structural = _load("_leiden_structural", STRUCTURAL)
    probe = structural._load_probe()
    question = _question(args.question_id, args.questions)
    retrieval = _retrieve(args, structural, probe, question)
    cards = _cards(retrieval["candidate_strands"])
    _planning_bridges(args.db, cards, probe)
    if len(cards) < 2:
        raise RuntimeError("SCIP Leiden retrieval has fewer than two bounded strands")
    if args.dry_run:
        print(json.dumps({
            "schema": "scip-leiden-live-ablation-preview/v1", "question_id": args.question_id,
            "frame": retrieval["frame"], "cards": cards,
            "selector_prompt_chars": len(_selector_prompt(question, retrieval["frame"], cards)),
        }, indent=2, sort_keys=True))
        return 0
    row, report = asyncio.run(_live(
        args, question, cards, retrieval["frame"], score_answer=not args.answer_only))
    _write(args.answers, [row])
    _write(args.report, report)
    if args.answer_only:
        print(
            f"Q{args.question_id} SCIP-Leiden answer-only run: unscored; "
            f"${row['total_cost_usd']:.4f}; {row['elapsed_s']:.1f}s")
    else:
        summary = report["score"]["summary"]
        print(
            f"Q{args.question_id} SCIP-Leiden live test: {summary['passed_questions']}/1 questions; "
            f"{summary['passed_claims']} claims; ${summary['total_cost_usd']:.4f}; {row['elapsed_s']:.1f}s")
    print(f"answers -> {args.answers}\nreport -> {args.report}")
    return 0 if args.answer_only or summary["passed_questions"] == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
