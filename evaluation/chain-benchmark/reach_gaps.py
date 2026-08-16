#!/usr/bin/env python3
"""Reach-gap classifier: why store-recoverable items miss the blind pool.

Grader-side only — runs AFTER the blind runner, reads its artifacts plus
reviewed gold, and computes for every store-recoverable claim that failed
the raw stage, per missing symbol, a MUTUALLY EXCLUSIVE first-loss
headline plus secondary annotations. The path basis is the recorded
production seed union (every channel), and the shortest relation-valid
path is grader diagnostics only — gold never feeds back into production.

First-loss headline ladder (exactly one per item):
  absent-store             target symbol resolves to no store row
  seeded-not-cited         production seeded it and cited nothing
  boundary:<relation-dir>  first policy-illegal hop on the shortest path
  reachable-not-taken:<n>  every hop legal at length n (rank/caps/assembly)
  disconnected             no path within the search bound

Secondary annotations (may overlap):
  question-named, catalog-exists, clew-pool-miss

Reports item counts AND unique affected-claim counts per family, plus a
per-claim dominant family — item labels must never be read as claim
impact.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from shadow_eval import required_items, store_recoverable

ALLOWED_HOPS = {
    ("call", "forward"), ("type_ref", "forward"),
    ("call", "reverse"), ("type_ref", "reverse"),
    ("contains", "owner"),
}


def resolve_ids(conn, source, names) -> dict:
    resolved: dict = {}
    for name in names:
        rows = conn.execute(
            "SELECT canonical_id FROM scip_symbols WHERE source_name = ? "
            "AND qualified_name = ? AND canonical_id NOT GLOB ? "
            "ORDER BY canonical_id", (source, name, "local *")).fetchall()
        resolved[name] = [row[0] for row in rows]
    return resolved


def neighbors(conn, frontier) -> list:
    """(neighbor, via, relation, direction) for every labeled hop."""
    found = []
    marks = ",".join("?" * len(frontier))
    for caller, callee, edge_type in conn.execute(
            f"SELECT caller_canonical_id, callee_canonical_id, edge_type "
            f"FROM scip_edges WHERE caller_canonical_id IN ({marks})",
            list(frontier)):
        found.append((callee, caller, edge_type, "forward"))
    for caller, callee, edge_type in conn.execute(
            f"SELECT caller_canonical_id, callee_canonical_id, edge_type "
            f"FROM scip_edges WHERE callee_canonical_id IN ({marks})",
            list(frontier)):
        found.append((caller, callee, edge_type, "reverse"))
    owner_rows = conn.execute(
        f"SELECT s.canonical_id, o.canonical_id FROM scip_symbols s "
        f"JOIN scip_symbols o ON o.qualified_name = s.parent_qualified_name "
        f"AND o.source_name = s.source_name "
        f"WHERE s.canonical_id IN ({marks})", list(frontier))
    for member, owner in owner_rows:
        found.append((owner, member, "contains", "owner"))
    member_rows = conn.execute(
        f"SELECT o.canonical_id, s.canonical_id FROM scip_symbols o "
        f"JOIN scip_symbols s ON s.parent_qualified_name = o.qualified_name "
        f"AND s.source_name = o.source_name "
        f"WHERE o.canonical_id IN ({marks})", list(frontier))
    for owner, member in member_rows:
        found.append((member, owner, "contains", "member"))
    return found


def shortest_labeled_path(conn, seed_ids, target_ids, *, max_depth=6):
    targets = set(target_ids)
    parents = {seed: None for seed in seed_ids}
    frontier = sorted(parents)
    for _depth in range(max_depth):
        if not frontier:
            break
        next_frontier = []
        for start in range(0, len(frontier), 400):
            chunk = frontier[start:start + 400]
            for node, via, relation, direction in sorted(
                    neighbors(conn, chunk)):
                if node in parents or node.startswith("local "):
                    continue
                parents[node] = (via, relation, direction)
                next_frontier.append(node)
                if node in targets:
                    path = []
                    cursor = node
                    while parents[cursor] is not None:
                        via_node, hop_relation, hop_direction = (
                            parents[cursor])
                        path.append((via_node, hop_relation,
                                     hop_direction, cursor))
                        cursor = via_node
                    return list(reversed(path))
        frontier = sorted(next_frontier)
    return None
def classify(conn, source, artifact, missing_names, question) -> list:
    """Mutually exclusive first-loss per missing symbol, plus annotations.

    The path basis is the recorded production seed union — every symbol
    production actually seeded from any channel — not just selected
    clews. Headline ladder: absent-store, seeded-not-cited (it WAS a
    seed and still produced no pool citation), boundary:<relation-dir>
    (first policy-illegal hop on the shortest path), reachable-not-taken
    (all hops legal — ranking/caps/assembly loss), disconnected.
    """
    seed_union = artifact.get("seed_union") or []
    seed_names = sorted({
        str(entry["symbol"]) for entry in seed_union}) or sorted(set(
            (artifact.get("seeds") or {}).get("selected_clews") or ()))
    pool_names = set((artifact.get("seeds") or {}).get("pool_clews") or ())
    seed_ids = [
        canonical for ids in resolve_ids(conn, source, seed_names).values()
        for canonical in ids]
    question_tokens = set(
        token.lower() for token in re.findall(r"[A-Za-z_]\w+", question))
    records = []
    for name in sorted(missing_names):
        target_ids = resolve_ids(conn, source, [name])[name]
        record = {"symbol": name, "annotations": []}
        leaf = name.rsplit(".", 1)[-1].lower()
        if leaf in question_tokens:
            record["annotations"].append("question-named")
        catalog = conn.execute(
            "SELECT 1 FROM documents WHERE source_name = ? AND "
            "json_extract(metadata, '$.qualified_name') = ? LIMIT 1",
            (source, name)).fetchone()
        if catalog is not None:
            record["annotations"].append("catalog-exists")
        if name not in pool_names:
            record["annotations"].append("clew-pool-miss")
        if not target_ids:
            record["first_loss"] = "absent-store"
            records.append(record)
            continue
        if name in set(seed_names):
            record["first_loss"] = "seeded-not-cited"
            records.append(record)
            continue
        path = (shortest_labeled_path(conn, seed_ids, target_ids)
                if seed_ids else None)
        if path is None:
            record["first_loss"] = "disconnected"
        else:
            record["path_length"] = len(path)
            record["path"] = [
                [via, relation, direction, node]
                for via, relation, direction, node in path]
            boundary = next(
                ((relation, direction) for _via, relation, direction, _node
                 in path if (relation, direction) not in ALLOWED_HOPS),
                None)
            if boundary is not None:
                record["first_loss"] = (
                    f"boundary:{boundary[0]}-{boundary[1]}")
            else:
                record["first_loss"] = (
                    f"reachable-not-taken:{len(path)}")
        records.append(record)
    return records
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--grade", required=True)
    parser.add_argument("--gold", default=str(
        HERE / "gold-chain-reviewed-compact.json"))
    parser.add_argument("--db", default=str(ROOT / "ariadne.db"))
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    artifacts = {
        int(row["id"]): row
        for row in json.loads(Path(args.artifacts).read_text())["questions"]}
    grade = json.loads(Path(args.grade).read_text())
    gold_questions = {
        int(question["id"]): question
        for question in json.loads(Path(args.gold).read_text())["questions"]}
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)

    item_histogram: Counter = Counter()
    claim_families: dict = {}
    rows = []
    for record in grade["claims"]:
        if not record["store"] or record["flags"]["raw"]:
            continue
        question_id = int(record["question"])
        artifact = artifacts.get(question_id)
        question = gold_questions[question_id]
        claim = next(
            claim for claim in question["claims"]
            if claim.get("id") == record["claim"])
        items = required_items(claim)
        raw_names = {row[0] for row in artifact["raw_pool"]}
        raw_names.update((artifact.get("seeds") or {}).get(
            "pool_clews") or ())
        missing = sorted(items["symbols"] - raw_names)
        classified = classify(
            conn, args.source, artifact, missing,
            str(question.get("question") or ""))
        families = Counter(
            entry["first_loss"] for entry in classified)
        for family, count in families.items():
            item_histogram[family] += count
            claim_families.setdefault(family, set()).add(
                (question_id, record["claim"]))
        dominant = families.most_common(1)[0][0] if families else None
        rows.append({
            "question": question_id, "claim": record["claim"],
            "dominant_family": dominant,
            "missing": classified})

    dominant_histogram = Counter(
        row["dominant_family"] for row in rows if row["dominant_family"])
    report = {
        "measurement": "reach-gap-first-loss",
        "claims_classified": len(rows),
        "item_histogram": dict(sorted(
            item_histogram.items(), key=lambda item: -item[1])),
        "claims_affected": {
            family: len(claims)
            for family, claims in sorted(
                claim_families.items(),
                key=lambda item: -len(item[1]))},
        "claims_by_dominant_family": dict(sorted(
            dominant_histogram.items(), key=lambda item: -item[1])),
        "claims": rows,
    }
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"claims classified: {len(rows)}")
    print("items by first loss:")
    for family, count in item_histogram.most_common():
        print(f"  {family}: {count} items, "
              f"{len(claim_families[family])} claims")
    print("claims by dominant family:")
    for family, count in dominant_histogram.most_common():
        print(f"  {family}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
