#!/usr/bin/env python3
"""Two-process blind shadow evaluator: a gold-blind runner, then a grader.

Phase A (``run``) executes the real production stages with cached
embeddings and deterministic selection and writes an immutable artifact
per question: seeds, raw candidate pool, model-facing menus, retained
selection, materialized excerpts, exact-source ledger, and the
deterministic final artifact. It never receives or reads reviewed gold —
its inputs are the questions-only file, the production store, and the
embedding cache.

Phase B (``grade``) runs after Phase A exits: it reads the blind
artifacts plus reviewed gold, computes the store-recoverable ceiling from
the live store (never hardcoded), and assigns each claim its furthest
consecutive stage:

    0 store  1 raw  2 menu  3 retained  4 materialized  5 ledger  6 final

It may grade artifacts; it never mutates or reruns selection.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
STAGES = (
    "store", "raw", "internal_menu", "retained", "materialized",
    "ledger", "final")
def blind_artifact(service, question_id, question, vector, *, source) -> dict:
    """One question's immutable blind artifact. No reviewed data enters.

    Every raw occurrence carries exact-identity resolution: the four-way
    conjunction (source, qualified name, file, extent) resolved against
    the store as exact/ambiguous/missing — never a first-row guess.
    """
    from offline_earliest_failure import stage_pipeline

    from library.question_facets import extract_question_facets

    artifacts = stage_pipeline(
        service, question_id, question, vector, source=source,
        selection_mode="deterministic")
    resolver = OccurrenceResolver(service, source)
    raw_pool = [
        {"qualified_name": citation.qualified_name,
         "file": citation.file,
         "line_start": citation.line_start,
         "line_end": citation.line_end,
         "parent_qualified_name": citation.parent_qualified_name,
         "call_site_file": citation.call_site_file,
         "call_site_line": citation.call_site_line,
         "relation": citation.relation,
         "hop": citation.hop,
         "stop_reason": citation.stop_reason,
         **resolver.resolve(
             citation.qualified_name, citation.file,
             citation.line_start, citation.line_end)}
        for citation in (
            hop.citation for hop in artifacts["evidence"].hops)]
    excerpts = [
        [excerpt.file, excerpt.line_start, excerpt.line_end, excerpt.kind,
         excerpt.sha256, excerpt.content]
        for hop in artifacts["selected_hops"]
        for excerpt in hop.source_excerpts]
    final_artifact = artifacts["spine"]
    body_plan = artifacts["body_plan"]
    return {
        "id": question_id,
        "question": question,
        "source": source,
        "mode": "blind-facet",
        "resolution_census": OccurrenceResolver.census(raw_pool),
        "seeds": {
            "pool_clews": sorted({
                symbol for match in artifacts["pool"]
                for symbol in match.clew.route}),
            "selected_clews": sorted({
                symbol for match in artifacts["matches"]
                for symbol in match.clew.route}),
        },
        "seed_union": [
            {"symbol": entry["symbol"], "origins": list(entry["origins"])}
            for entry in artifacts["evidence"].seed_provenance],
        "expansion": dict(getattr(
            artifacts["evidence"], "expansion_diagnostics", {}) or {}),
        "body_plan": {
            "required": [
                [requirement.body_ref.qualified_name,
                 requirement.body_ref.file,
                 requirement.body_ref.line_start,
                 requirement.body_ref.line_end,
                 requirement.reason]
                for requirement in body_plan.required],
            "optional_total": len(body_plan.optional),
            "gaps": list(body_plan.gaps),
            "cap_events": len(body_plan.cap_events),
        },
        "facets": [
            {"id": facet.id, "exact_text": facet.exact_text,
             "kind": facet.kind, "identifiers": list(facet.identifiers),
             "roles": list(facet.roles)}
            for facet in extract_question_facets(question)],
        "raw_pool": raw_pool,
        "menu": {
            "routes": {label: list(route)
                       for label, route in artifacts["menu"].routes.items()},
            "scoped_routes": {
                label: list(route)
                for label, route in artifacts["scoped"].routes.items()},
        },
        "retained": {
            "symbols": list(artifacts["selection"].symbols),
            "occurrence_keys": [
                list(key) for key in artifacts["selection"].occurrence_keys],
            "body_symbols": list(artifacts["body_symbols"]),
        },
        "materialized_excerpts": excerpts,
        "ledger": artifacts["ledger"],
        "final_artifact": final_artifact,
        "source_gaps": list(artifacts["source_gaps"]),
        "timings": artifacts["timings"],
        "stage_hashes": {
            "raw_pool": _digest(raw_pool),
            "menu": _digest(sorted(artifacts["scoped"].routes)),
            "retained": _digest(sorted(artifacts["selection"].symbols)),
            "required_bodies": _digest(sorted(
                (requirement.body_ref.qualified_name,
                 requirement.body_ref.file,
                 requirement.body_ref.line_start,
                 requirement.body_ref.line_end)
                for requirement in body_plan.required)),
            "ledger": _digest(artifacts["ledger"]),
            "final": _digest(final_artifact),
        },
    }


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def run_blind(argv) -> int:
    """Phase A. The word gold appears in this module only to be excluded."""
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--only", default="")
    parser.add_argument("--strong-db-fingerprint", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    questions = json.loads(Path(args.questions).read_text())
    banned = {"claims", "anchors", "witnesses", "review",
              "candidate_paths", "selected_path_ids"}
    for row in questions:
        leaked = banned.intersection(row)
        if leaked:
            raise SystemExit(
                f"ABORT: questions file carries reviewed fields {leaked}")
    only = {int(value) for value in args.only.split(",") if value}
    vectors = np.load(args.embedding_cache, allow_pickle=False)

    from ariadne_mcp.service import AriadneService
    service = AriadneService.get()

    rows = []
    for row in questions:
        question_id = int(row["id"])
        if only and question_id not in only:
            continue
        key = f"q{question_id}"
        if key not in vectors:
            print(f"q{question_id}: no cached embedding; skipped")
            continue
        started = time.perf_counter()
        rows.append(blind_artifact(
            service, question_id, str(row["question"]),
            vectors[key], source=args.source))
        print(f"q{question_id}: blind artifact in "
              f"{time.perf_counter() - started:.1f}s", flush=True)
    from exp_seal import seal
    payload = seal({
        "schema": "ariadne-blind-shadow-v3", "mode": "blind-facet",
        "provenance": build_provenance(
            args, strong_db=bool(args.strong_db_fingerprint)),
        "questions": rows})
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {args.out} ({len(rows)} questions)")
    return 0


def required_items(claim) -> dict:
    """The reviewed items a claim needs, from its accepted review."""
    selected_paths = set(
        (claim.get("review") or {}).get("selected_path_ids") or ())
    symbols: set = set()
    canonical: set = set()
    edges: set = set()
    for anchor in claim.get("anchors", ()):
        target = (anchor.get("target") or {}).get("symbol")
        if target:
            symbols.add(str(target))
    chosen = (claim.get("review") or {}).get(
        "selected_candidate_by_anchor") or {}
    canonical.update(str(value) for value in chosen.values())
    for path in claim.get("candidate_paths", ()):
        if selected_paths and path.get("id") not in selected_paths:
            continue
        for node in path.get("nodes", ()):
            if node.get("qualified_name"):
                symbols.add(str(node["qualified_name"]))
            if node.get("canonical_id"):
                canonical.add(str(node["canonical_id"]))
        for edge in path.get("edges", ()):
            edges.add((str(edge["caller_canonical_id"]),
                       str(edge["callee_canonical_id"]),
                       str(edge["edge_type"])))
    witnesses = [
        {"file": witness.get("file"),
         "line_start": int(witness.get("line_start") or 0),
         "line_end": int(witness.get("line_end") or 0),
         "contains": [str(text) for text in witness.get("contains", ())]}
        for witness in claim.get("witnesses", ())]
    return {"symbols": symbols, "canonical": canonical, "edges": edges,
            "witnesses": witnesses}
def store_recoverable(conn, items) -> bool:
    in_store, _gaps = store_diagnostics(conn, items)
    return in_store
def claim_stage_flags(artifact, items) -> dict:
    """Independent per-stage presence flags for one claim's required items.

    The raw surface is the union of the clew candidate pool and the walked
    evidence pool — everything production retrieval put on the table before
    any model-facing cut.
    """
    seeds = artifact.get("seeds") or {}
    raw_names = {
        row["qualified_name"] if isinstance(row, dict) else row[0]
        for row in artifact["raw_pool"]}
    raw_names.update(seeds.get("pool_clews") or ())
    raw_names.update(seeds.get("selected_clews") or ())
    menu_names = {
        symbol for route in artifact["menu"]["scoped_routes"].values()
        for symbol in route}
    retained_names = set(artifact["retained"]["symbols"]) | {
        key[0] for key in artifact["retained"]["occurrence_keys"]}
    excerpt_ranges = [
        (row[0], row[1], row[2]) for row in
        artifact["materialized_excerpts"]]
    ledger = artifact["ledger"]
    final = artifact["final_artifact"]

    symbols = items["symbols"]
    flags = {
        "raw": bool(symbols) and symbols <= raw_names,
        "internal_menu": bool(symbols) and symbols <= menu_names,
        "retained": bool(symbols) and symbols <= retained_names,
    }
    covered = []
    for witness in items["witnesses"]:
        span_covered = any(
            file == witness["file"]
            and line_start <= witness["line_start"]
            and witness["line_end"] <= line_end
            for file, line_start, line_end in excerpt_ranges)
        covered.append(span_covered)
    flags["materialized"] = flags["retained"] and all(covered)
    fragments = [
        fragment for witness in items["witnesses"]
        for fragment in witness["contains"]]
    flags["ledger"] = flags["materialized"] and all(
        fragment in ledger for fragment in fragments)
    flags["final"] = flags["ledger"] and all(
        fragment in final for fragment in fragments)
    return flags
def grade(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--gold", default=str(
        HERE / "gold-chain-reviewed-compact.json"))
    parser.add_argument("--db", default=str(ROOT / "ariadne.db"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    from exp_seal import verify_seal
    payload = json.loads(Path(args.artifacts).read_text())
    verify_seal(payload)
    artifacts = {int(row["id"]): row for row in payload["questions"]}
    gold = json.loads(Path(args.gold).read_text())
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    vector = {stage: 0 for stage in STAGES}
    rows = []
    for question in gold["questions"]:
        question_id = int(question["id"])
        artifact = artifacts.get(question_id)
        for claim in question.get("claims", ()):
            items = required_items(claim)
            in_store, store_gaps = store_diagnostics(conn, items)
            record = {
                "question": question_id, "claim": claim.get("id"),
                "store": in_store, "store_gaps": store_gaps}
            flags = (claim_stage_flags(artifact, items)
                     if artifact is not None else
                     {stage: False for stage in STAGES[1:]})
            record["flags"] = flags
            # The frontier is the furthest CONSECUTIVE stage: a claim
            # retained by a net without ever being menu-visible stops
            # at raw for the vector, and the flags keep the detail.
            frontier = 0
            if in_store:
                for index, stage in enumerate(STAGES[1:], start=1):
                    if not flags.get(stage):
                        break
                    frontier = index
            record["frontier"] = STAGES[frontier] if (
                in_store or frontier) else "none"
            if in_store:
                for index, stage in enumerate(STAGES):
                    if index == 0 or index <= frontier:
                        vector[stage] += 1
            rows.append(record)

    total = len(rows)
    report = {
        "measurement": "blind-shadow-grade",
        "artifacts_file": str(args.artifacts),
        "gold_sha256": hashlib.sha256(
            Path(args.gold).read_bytes()).hexdigest(),
        "claims_total": total,
        "stage_vector": vector,
        "claims": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print("STAGE VECTOR (store-recoverable ceiling computed, "
          "not hardcoded; consecutive frontier):")
    for stage in STAGES:
        print(f"  {stage:>13}: {vector[stage]}/{total}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "grade"))
    args, rest = parser.parse_known_args()
    if args.command == "run":
        return run_blind(rest)
    return grade(rest)


def store_diagnostics(conn, items) -> tuple:
    """(recoverable, gaps) — every gap names the exact missing store item."""
    gaps = []
    for canonical_id in sorted(items["canonical"]):
        row = conn.execute(
            "SELECT 1 FROM scip_symbols WHERE canonical_id = ?",
            (canonical_id,)).fetchone()
        if row is None:
            gaps.append(f"symbol-absent:{canonical_id}")
    for caller, callee, edge_type in sorted(items["edges"]):
        row = conn.execute(
            "SELECT 1 FROM scip_edges WHERE caller_canonical_id = ? "
            "AND callee_canonical_id = ? AND edge_type = ?",
            (caller, callee, edge_type)).fetchone()
        if row is None:
            gaps.append(f"edge-absent:{edge_type}:{caller}->{callee}")
    return (not gaps, gaps)
def build_provenance(args, *, strong_db: bool = False) -> dict:
    """Prove which implementation, store, and configuration produced this.

    The runtime manifest covers every imported runtime file — tracked or
    not — because this checkout is dirty and a hand-picked module list
    cannot attribute a result. The database fingerprint level is
    recorded; paid-canary certification requires the strong level.
    """
    from exp_fingerprint import (
        effective_configuration,
        fast_db_fingerprint,
        runtime_manifest,
        strong_db_fingerprint,
    )

    def file_sha(path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    database = ROOT / "ariadne.db"
    fingerprint = (
        strong_db_fingerprint(database) if strong_db
        else fast_db_fingerprint(database))
    return {
        "runtime_manifest": runtime_manifest(ROOT),
        "database_fingerprint": fingerprint,
        "effective_configuration": effective_configuration(),
        "command": {
            "argv": list(sys.argv),
            "questions": str(args.questions),
            "embedding_cache": str(args.embedding_cache),
            "source": str(args.source),
            "only": str(getattr(args, "only", "") or ""),
        },
        "embedding_cache_sha256": file_sha(args.embedding_cache),
        "questions_sha256": file_sha(args.questions),
        "offline_policy": "no-network; cached embeddings only",
    }


class OccurrenceResolver:
    """Exact-identity resolution by four-way conjunction, never first-row.

    Production citations do not yet carry canonical ids, so the evaluator
    resolves (source, qualified name, file, extent) against the store:
    one row is exact, zero is missing, several are ambiguous — and an
    ambiguous required occurrence must fail certification rather than be
    guessed at.
    """

    def __init__(self, service, source: str):
        self._service = service
        self._source = source
        self._cache: dict = {}

    def resolve(self, qualified_name, file, line_start, line_end) -> dict:
        key = (str(qualified_name), str(file),
               int(line_start or 0), int(line_end or 0))
        if key in self._cache:
            return self._cache[key]
        with self._service.library._conn_provider.acquire() as conn:
            rows = conn.execute(
                "SELECT canonical_id, kind FROM scip_symbols "
                "WHERE source_name = ? AND qualified_name = ? "
                "AND file = ? AND line_start = ? AND line_end = ? "
                "AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                (self._source, key[0], key[1], key[2], key[3],
                 "local *")).fetchall()
        if len(rows) == 1:
            record = {
                "canonical_id": rows[0][0],
                "kind": rows[0][1],
                "resolution": "exact",
                "resolution_candidates": []}
        elif not rows:
            record = {
                "canonical_id": "unresolved",
                "kind": "",
                "resolution": "missing",
                "resolution_candidates": []}
        else:
            record = {
                "canonical_id": "unresolved",
                "kind": rows[0][1],
                "resolution": "ambiguous",
                "resolution_candidates": [row[0] for row in rows]}
        self._cache[key] = record
        return record

    @staticmethod
    def census(rows) -> dict:
        counts: dict = {"exact": 0, "ambiguous": 0, "missing": 0}
        for row in rows:
            counts[row["resolution"]] = counts.get(
                row["resolution"], 0) + 1
        return counts


if __name__ == "__main__":
    raise SystemExit(main())
