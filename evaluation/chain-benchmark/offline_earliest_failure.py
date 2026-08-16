#!/usr/bin/env python3
"""Per-claim earliest-failure audit of the deterministic answer pipeline.

Runs the production stages with no LLM and no paid API: cached question
vectors drive retrieval, the deterministic clause-covering selector replaces
route selection, every offered body is completed, and the exact-source chunk
ledger is assembled. Each reviewed claim is classified to exactly one
earliest failing stage:

    retrieval -> ranking -> retention -> materialization -> ledger

``ledger`` is the deterministic stand-in for formulation: the fragment was
materialized but the chunk ledger — the narration's only source of exact
code — does not carry it.
"""
from __future__ import annotations
import gc

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

STAGES = ("retrieval", "ranking", "retention", "materialization", "ledger")


def anchor_symbols(claim) -> list[str]:
    """Reviewed qualified symbols; structured targets first, legacy strings kept."""
    symbols = []
    for anchor in claim.get("anchors", ()):
        if isinstance(anchor, dict):
            target = anchor.get("target") or {}
            symbol = str(target.get("symbol") or anchor.get("symbol")
                         or anchor.get("anchor") or "")
        else:
            symbol = str(anchor)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols
def witness_owner_symbols(service, question, *, source) -> list[str]:
    """Tightest indexed symbol enclosing each reviewed witness span."""
    owners = []
    with service.library._conn_provider.acquire() as conn:
        for claim in question.get("claims", ()):
            for witness in claim.get("witnesses", ()):
                file = str(witness.get("file") or "")
                line_start = int(witness.get("line_start") or 0)
                line_end = int(witness.get("line_end") or line_start)
                if not file or not line_start:
                    continue
                rows = conn.execute(
                    "SELECT qualified_name FROM scip_symbols "
                    "WHERE source_name = ? AND file LIKE ? "
                    "AND line_start <= ? AND line_end >= ? "
                    "AND canonical_id NOT GLOB 'local *' "
                    "ORDER BY (line_end - line_start), line_start LIMIT 24",
                    (source, f"%{file}", line_end, line_start)).fetchall()
                if not rows:
                    # A header-comment span sits above every extent; the
                    # definition directly below it is the witness owner.
                    rows = conn.execute(
                        "SELECT qualified_name FROM scip_symbols "
                        "WHERE source_name = ? AND file LIKE ? "
                        "AND line_start > ? AND line_start <= ? "
                        "AND canonical_id NOT GLOB 'local *' "
                        "ORDER BY line_start LIMIT 1",
                        (source, f"%{file}", line_end, line_end + 3)).fetchall()
                for row in rows:
                    if row[0] not in owners:
                        owners.append(row[0])
    return owners
def oracle_matches(pool, anchors, question, *, per_anchor=4, limit=24):
    """Pool routes that carry reviewed anchors, topped up deterministically.

    Evaluation-only: this uses the reviewed oracle to hold selection constant
    so retention, materialization, and ledger failures are measured without
    selector noise. Runtime code never sees these anchors.
    """
    from library.clews import deterministic_clew_matches

    chosen = []
    chosen_ids = set()
    for anchor in anchors:
        needle = anchor.lower()
        kept = 0
        for match in pool:
            if kept >= per_anchor or len(chosen) >= limit:
                break
            if match.clew.id in chosen_ids:
                continue
            if any(needle in symbol.lower() for symbol in match.clew.route):
                chosen.append(match)
                chosen_ids.add(match.clew.id)
                kept += 1
    for match in deterministic_clew_matches(question, pool, limit=8):
        if len(chosen) >= limit:
            break
        if match.clew.id not in chosen_ids:
            chosen.append(match)
            chosen_ids.add(match.clew.id)
    # Production's coverage plan hands the walk obligations and resolved
    # targets; the oracle stands in for the plan with the reviewed anchors.
    from attrs import evolve
    obligations = tuple(range(1, max(len(anchors), 1) + 1))
    targets = tuple(
        (obligation, anchor)
        for obligation, anchor in enumerate(anchors, 1))
    return [evolve(match, obligations=obligations, target_symbols=targets)
            for match in chosen]


def stage_pipeline(service, question_id, question, vector, *, source,
                   selection_mode="deterministic", anchors=()):
    """Deterministic production stages; returns artifacts and stage timings."""
    import numpy as np

    from library.chain_answer import catalog_positioning_documents, evidence_for
    from library.chain_menu import (
        all_definition_body_selection,
        all_route_selection,
        complete_definition_body_selection,
        complete_selection_with_body_dependencies,
        definition_body_menu,
        fetch_selected,
        hydrate_selected_hops,
        project_selected_evidence,
        route_menu_for,
        route_section_embedding_scores,
        scope_route_menu,
        Selection,
    )
    from library.chain_story import build_story_ir, render_formulation_spine
    from library.clews import (deterministic_clew_matches,
                               document_clew_matches, nearest_clew_matches,
                               select_clew_matches)
    from library.source_chunks import source_chunk_values

    timings = {}
    started = time.perf_counter()
    query = np.asarray(vector, dtype=np.float32)
    positioning = catalog_positioning_documents(
        service.library, question,
        sources=(source, f"spool:{source}"), limit=8,
        query_embedding=query, matrix_provider=lambda: None)
    with service.library._conn_provider.acquire() as conn:
        recalled = nearest_clew_matches(
            conn, query, source_name=source, top_k=5000, min_similarity=-1.0)
        accepted = select_clew_matches(question, recalled).accepted
        documents = document_clew_matches(
            conn, positioning, question, source_name=source, limit=48)
    known = {match.clew.id for match in accepted}
    pool = accepted + [match for match in documents
                       if match.clew.id not in known]
    timings["retrieval"] = time.perf_counter() - started

    started = time.perf_counter()
    if selection_mode == "oracle":
        matches = oracle_matches(pool, anchors, question)
    else:
        matches = deterministic_clew_matches(question, pool, limit=8)
    timings["clew_selection"] = time.perf_counter() - started

    started = time.perf_counter()
    evidence = evidence_for(
        service.library, positioning, source=source, clew_matches=matches,
        question=question, defer_source=True,
        positioning_documents=positioning)
    timings["evidence_walk"] = time.perf_counter() - started

    started = time.perf_counter()
    menu = route_menu_for(service.library, evidence.hops, source=source)
    route_scores = {}
    if menu.sections:
        section_scores = route_section_embedding_scores(
            service.library, menu, Selection(route_ids=tuple(menu.routes)),
            query)
        route_scores = {
            route_id: max(
                (section_scores[label]
                 for label in menu.route_sections.get(route_id, ())
                 if label in section_scores),
                default=0.0)
            for route_id in menu.routes}
    required_symbols = ()
    if selection_mode == "oracle":
        # Production narrows routes with an LLM component/route selection;
        # the oracle stands in for a correct one by reserving every route
        # symbol the reviewed anchors resolve to (evaluation-only).
        wanted = []
        for anchor in anchors:
            needle = anchor.lower()
            member = ".".join(needle.rsplit(".", 2)[-2:])
            for route in menu.routes.values():
                for symbol in route:
                    lowered = symbol.lower()
                    if ((needle in lowered or member in lowered)
                            and symbol not in wanted):
                        wanted.append(symbol)
        required_symbols = tuple(wanted)
    scoped = scope_route_menu(menu, question, route_scores=route_scores,
                              required_symbols=required_symbols)
    selection = all_route_selection(scoped)
    body_menu = definition_body_menu(evidence.hops, selection)
    body_selection = complete_definition_body_selection(
        body_menu, all_definition_body_selection(body_menu))
    from library.body_plan import derive_definition_body_plan
    body_plan = derive_definition_body_plan(
        hops=evidence.hops,
        retained_symbols=tuple(selection.symbols),
        bindings=())
    body_symbols = list(body_selection.symbols)
    timings["selection"] = time.perf_counter() - started

    started = time.perf_counter()
    source_root = str(
        service.config.get_all_source_paths().get(source) or "") or None
    selected_hops, source_gaps = hydrate_selected_hops(
        service.library, evidence.hops, selection, source=source,
        source_root=source_root,
        definition_body_symbols=tuple(body_symbols))
    selection = complete_selection_with_body_dependencies(
        selection, selected_hops, body_symbols)
    timings["hydration"] = time.perf_counter() - started

    started = time.perf_counter()
    fetched = fetch_selected(service.library, selection, selected_hops)
    story = build_story_ir(selected_hops, selection, fetched)
    spine = render_formulation_spine(story)
    ledger = "\n".join(source_chunk_values(story.chunks).values())
    projected = project_selected_evidence(
        evidence, selection, hydrated_hops=selected_hops,
        source_gaps=source_gaps)
    timings["materialization"] = time.perf_counter() - started
    return {
        "pool": pool, "matches": matches, "evidence": evidence,
        "menu": menu, "scoped": scoped, "selection": selection,
        "body_symbols": body_symbols, "selected_hops": selected_hops,
        "story": story, "spine": spine, "ledger": ledger,
        "projected": projected, "source_gaps": source_gaps,
        "timings": timings, "body_plan": body_plan,
    }


def surfaces(artifacts) -> dict:
    """Membership surfaces each stage exposes, computed once per question."""
    pool_symbols = " ".join(
        symbol for match in artifacts["pool"] for symbol in match.clew.route)
    selected_symbols = " ".join(
        symbol for match in artifacts["matches"] for symbol in match.clew.route)
    hop_symbols = " ".join(
        hop.citation.qualified_name for hop in artifacts["evidence"].hops)
    chosen_symbols = " ".join(artifacts["selection"].symbols)
    story_symbols = " ".join(
        node.symbol for node in artifacts["story"].nodes)
    projected_symbols = " ".join(
        hop.citation.qualified_name for hop in artifacts["projected"].hops)
    excerpts = [
        (excerpt.file, excerpt.line_start, excerpt.line_end, excerpt.content)
        for hop in artifacts["selected_hops"]
        for excerpt in hop.source_excerpts]
    menu_symbols = " ".join(
        symbol for route in artifacts["menu"].routes.values()
        for symbol in route)
    scoped_symbols = " ".join(
        symbol for route in artifacts["scoped"].routes.values()
        for symbol in route)
    return {
        "pool": " ".join((pool_symbols, selected_symbols, hop_symbols,
                          menu_symbols)).lower(),
        "clew_pool": pool_symbols.lower(),
        "walked": hop_symbols.lower(),
        "scoped": scoped_symbols.lower(),
        "selected": chosen_symbols.lower(),
        "final": (story_symbols + " " + projected_symbols).lower(),
        "excerpts": excerpts,
        "ledger": artifacts["ledger"],
        "spine": artifacts["spine"],
    }


def classify_claim(claim, surface) -> dict:
    """Exactly one earliest failing stage, or pass, per reviewed claim."""
    symbols = anchor_symbols(claim)
    missing = {stage: [] for stage in STAGES}
    presence = {}
    for symbol in symbols:
        # Canonical prefixes differ across stores; owner.member is identity.
        needle = symbol.lower()
        member = ".".join(needle.rsplit(".", 2)[-2:])

        def present(surface_text):
            return needle in surface_text or member in surface_text

        presence[symbol] = {
            name: present(surface[name])
            for name in ("clew_pool", "walked", "scoped", "selected", "final")}
        if not present(surface["pool"]):
            missing["retrieval"].append(symbol)
        elif not present(surface["selected"]):
            missing["ranking"].append(symbol)
        elif not present(surface["final"]):
            missing["retention"].append(symbol)
    fragments_total = 0
    fragments_found = 0
    for witness in claim.get("witnesses", ()):
        span = (str(witness.get("file") or ""),
                int(witness.get("line_start") or 0),
                int(witness.get("line_end") or 0))
        covered = any(
            file.endswith(span[0]) and line_start <= span[1]
            and span[2] <= line_end
            for file, line_start, line_end, _ in surface["excerpts"]
            if span[0])
        for fragment in witness.get("contains", ()):
            fragments_total += 1
            if fragment in surface["ledger"]:
                fragments_found += 1
            elif any(fragment in content
                     for _, _, _, content in surface["excerpts"]) or covered:
                missing["ledger"].append(fragment)
            else:
                missing["materialization"].append(fragment)
    earliest = next(
        (stage for stage in STAGES if missing[stage]), "pass")
    return {
        "id": claim.get("id"),
        "earliest_failure": earliest,
        "anchor_surfaces": presence,
        "fragments_found": fragments_found,
        "fragments": fragments_total,
        "missing_symbols": {
            stage: values for stage, values in missing.items()
            if values and stage in ("retrieval", "ranking", "retention")},
        "missing_fragments": {
            stage: values for stage, values in missing.items()
            if values and stage in ("materialization", "ledger")},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(HERE / "gold-chain-reviewed.json"))
    parser.add_argument("--embedding-cache",
                        default=str(HERE / "question-embeddings.npz"))
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--selection", default="deterministic",
                        choices=("deterministic", "oracle"),
                        help="oracle holds selection constant using reviewed anchors (evaluation-only)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import numpy as np

    gold = {int(row["id"]): row for row in json.loads(
        Path(args.gold).read_text()).get("questions", ())}
    with np.load(args.embedding_cache, allow_pickle=False) as cached:
        vectors = {int(key[1:]): cached[key] for key in cached.files
                   if key.startswith("q")}
    ids = sorted(set(gold) & set(vectors))
    if args.only:
        wanted = {int(value) for value in args.only.split(",")}
        ids = [qid for qid in ids if qid in wanted]
    absent = sorted(set(gold) - set(vectors))
    if absent:
        print(f"WARNING: no cached embedding for question(s) {absent}; skipped",
              flush=True)

    from ariadne_mcp.service import AriadneService
    service = AriadneService.get()

    rows = []
    totals = {stage: 0 for stage in (*STAGES, "pass")}
    fragment_totals = [0, 0]
    for qid in ids:
        question = gold[qid]
        started = time.perf_counter()
        anchors = list(dict.fromkeys(
            symbol for claim in question.get("claims", ())
            for symbol in anchor_symbols(claim)))
        if args.selection == "oracle":
            # The reviewed candidate paths are the constructed gold chains;
            # holding selection to them isolates the downstream stages. A
            # witness span is a requirement too, wherever the chain points,
            # so its enclosing symbol joins the reserved set.
            anchors = list(dict.fromkeys(anchors + [
                str(node.get("qualified_name") or "")
                for claim in question.get("claims", ())
                for path in claim.get("candidate_paths", ())
                for node in path.get("nodes", ())
                if node.get("qualified_name")]
                + witness_owner_symbols(
                    service, question, source=args.source)))
        artifacts = stage_pipeline(
            service, qid, str(question.get("question") or ""),
            vectors[qid], source=args.source,
            selection_mode=args.selection, anchors=anchors)
        surface = surfaces(artifacts)
        claims = [classify_claim(claim, surface)
                  for claim in question.get("claims", ())]
        for claim in claims:
            totals[claim["earliest_failure"]] += 1
            fragment_totals[0] += claim["fragments_found"]
            fragment_totals[1] += claim["fragments"]
        story = artifacts["story"]
        rows.append({
            "id": qid,
            "pool_routes": len(artifacts["pool"]),
            "selected_routes": len(artifacts["matches"]),
            "evidence_hops": len(artifacts["evidence"].hops),
            "menu_routes": len(artifacts["menu"].routes),
            "scoped_routes": len(artifacts["scoped"].routes),
            "selected_bodies": len(artifacts["body_symbols"]),
            "selection_occurrences": len(
                artifacts["selection"].occurrence_keys),
            "hydrated_hops": len(artifacts["selected_hops"]),
            "story_nodes": len(story.nodes),
            "story_edges": len(story.edges),
            "story_chunks": len(story.chunks),
            "spine_chars": len(artifacts["spine"]),
            "source_gaps": list(artifacts["source_gaps"]),
            "timings": {key: round(value, 3)
                        for key, value in artifacts["timings"].items()},
            "elapsed": round(time.perf_counter() - started, 3),
            "claims": claims,
        })
        statuses = ", ".join(
            f"{claim['id']}={claim['earliest_failure']}" for claim in claims)
        print(f"id {qid}: {statuses}; ledger "
              f"{sum(claim['fragments_found'] for claim in claims)}/"
              f"{sum(claim['fragments'] for claim in claims)} fragments; "
              f"{rows[-1]['elapsed']}s", flush=True)
        del artifacts, surface, story
        gc.collect()

    report = {
        "selection_mode": args.selection,
        "stage_totals": totals,
        "fragments_found": fragment_totals[0],
        "fragments": fragment_totals[1],
        "questions": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("summary: " + ", ".join(
        f"{key}={value}" for key, value in totals.items())
        + f"; fragments {fragment_totals[0]}/{fragment_totals[1]}")
    print(f"earliest-failure audit -> {args.out}")
    return 0 if totals["pass"] == sum(totals.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
