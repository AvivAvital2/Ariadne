"""Exact source-instance-aware store grading.

The legacy store grader was source-instance blind: a canonical id present
anywhere satisfied it, so a Spark Connect shadow could substitute for a
classic production definition and an ownership edge could stand in for a
call. Exact grading requires the reviewed source, module, file, extent,
relation type, direction, and site — and every failed item carries
exactly one primary reason from a deterministic taxonomy. The result is
``exact_store_complete``; the old number remains ``legacy_store_complete``
and the two are never conflated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFINITION_TAXONOMY = (
    "absent_symbol", "wrong_source_instance", "wrong_module",
    "wrong_file", "wrong_extent", "wrong_definition_hash",
    "ambiguous_identity")
EDGE_TAXONOMY = (
    "absent_edge", "wrong_edge_type", "wrong_edge_direction",
    "wrong_edge_site")


@dataclass(frozen=True)
class DefinitionExpectation:
    source: str
    canonical_id: str
    qualified_name: str
    file: str
    line_start: int
    line_end: int
    kind: str = ""
    owner: str = ""


@dataclass(frozen=True)
class EdgeExpectation:
    caller_canonical_id: str
    callee_canonical_id: str
    edge_type: str
    file: str
    line: int


@dataclass(frozen=True)
class StoreGap:
    item: str
    primary_reason: str
    detail: str = ""
    secondary: tuple = ()


def definition_expectations(claim, *, source: str) -> list:
    """Exact expectations from review-selected anchors and path nodes."""
    expectations: dict = {}
    selected = set(
        ((claim.get("review") or {}).get("selected_candidate_by_anchor")
         or {}).values())
    for anchor in claim.get("anchors", ()):
        for candidate in anchor.get("candidates", ()):
            if candidate.get("canonical_id") in selected:
                expectations[candidate["canonical_id"]] = (
                    DefinitionExpectation(
                        source=source,
                        canonical_id=str(candidate["canonical_id"]),
                        qualified_name=str(
                            candidate.get("qualified_name") or ""),
                        file=str(candidate.get("file") or ""),
                        line_start=int(candidate.get("line_start") or 0),
                        line_end=int(candidate.get("line_end") or 0),
                        kind=str(candidate.get("kind") or ""),
                        owner=str(
                            candidate.get("parent_qualified_name") or "")))
    selected_paths = set(
        (claim.get("review") or {}).get("selected_path_ids") or ())
    for path in claim.get("candidate_paths", ()):
        if selected_paths and path.get("id") not in selected_paths:
            continue
        for node in path.get("nodes", ()):
            canonical = str(node.get("canonical_id") or "")
            if canonical and canonical not in expectations:
                expectations[canonical] = DefinitionExpectation(
                    source=source,
                    canonical_id=canonical,
                    qualified_name=str(node.get("qualified_name") or ""),
                    file=str(node.get("file") or ""),
                    line_start=int(node.get("line_start") or 0),
                    line_end=int(node.get("line_end") or 0),
                    kind=str(node.get("kind") or ""))
    return list(expectations.values())


def edge_expectations(claim) -> list:
    selected_paths = set(
        (claim.get("review") or {}).get("selected_path_ids") or ())
    expectations: dict = {}
    for path in claim.get("candidate_paths", ()):
        if selected_paths and path.get("id") not in selected_paths:
            continue
        for edge in path.get("edges", ()):
            key = (str(edge["caller_canonical_id"]),
                   str(edge["callee_canonical_id"]),
                   str(edge["edge_type"]),
                   str(edge.get("file") or ""),
                   int(edge.get("line") or 0))
            expectations[key] = EdgeExpectation(*key)
    return list(expectations.values())


def _module_root(file: str) -> str:
    return file.split("/", 1)[0] if file else ""


def grade_definition(conn, expectation: DefinitionExpectation) -> (
        "StoreGap | None"):
    """One primary reason per failure, in deterministic taxonomy order."""
    rows = conn.execute(
        "SELECT source_name, file, line_start, line_end, kind "
        "FROM scip_symbols WHERE canonical_id = ?",
        (expectation.canonical_id,)).fetchall()
    item = (f"{expectation.qualified_name}"
            f"@{expectation.file}:{expectation.line_start}")
    if not rows:
        return StoreGap(item, "absent_symbol")
    same_source = [row for row in rows
                   if row[0] == expectation.source]
    if not same_source:
        return StoreGap(
            item, "wrong_source_instance",
            detail=f"present only under {sorted({r[0] for r in rows})}")
    exact = [row for row in same_source
             if row[1] == expectation.file
             and row[2] == expectation.line_start
             and row[3] == expectation.line_end]
    if len(exact) == 1:
        if expectation.kind and exact[0][4] and (
                exact[0][4] != expectation.kind):
            return StoreGap(
                item, "ambiguous_identity",
                detail=f"kind {exact[0][4]!r} != {expectation.kind!r}")
        return None
    if len(exact) > 1:
        return StoreGap(
            item, "ambiguous_identity",
            detail=f"{len(exact)} identical exact rows")
    files = sorted({row[1] for row in same_source})
    if expectation.file not in files:
        expected_root = _module_root(expectation.file)
        if any(_module_root(file) != expected_root for file in files):
            return StoreGap(
                item, "wrong_module",
                detail=f"defined in {files[:3]}")
        return StoreGap(item, "wrong_file", detail=f"in {files[:3]}")
    return StoreGap(
        item, "wrong_extent",
        detail=f"extents {[(r[2], r[3]) for r in same_source][:3]}")


def grade_edge(conn, expectation: EdgeExpectation) -> "StoreGap | None":
    item = (f"{expectation.edge_type}:"
            f"{expectation.caller_canonical_id[:40]}->"
            f"{expectation.callee_canonical_id[:40]}")
    endpoint_rows = conn.execute(
        "SELECT edge_type, file, line FROM scip_edges "
        "WHERE caller_canonical_id = ? AND callee_canonical_id = ?",
        (expectation.caller_canonical_id,
         expectation.callee_canonical_id)).fetchall()
    typed = [row for row in endpoint_rows
             if row[0] == expectation.edge_type]
    if typed:
        sited = [row for row in typed
                 if row[1] == expectation.file
                 and int(row[2]) == expectation.line]
        if sited:
            return None
        return StoreGap(
            item, "wrong_edge_site",
            detail=f"sites {[(r[1], r[2]) for r in typed][:3]}")
    if endpoint_rows:
        return StoreGap(
            item, "wrong_edge_type",
            detail=f"present as {sorted({r[0] for r in endpoint_rows})}")
    reversed_rows = conn.execute(
        "SELECT edge_type FROM scip_edges "
        "WHERE caller_canonical_id = ? AND callee_canonical_id = ?",
        (expectation.callee_canonical_id,
         expectation.caller_canonical_id)).fetchall()
    if any(row[0] == expectation.edge_type for row in reversed_rows):
        return StoreGap(item, "wrong_edge_direction")
    return StoreGap(item, "absent_edge")


def grade_claim(conn, claim, *, source: str) -> dict:
    gaps = []
    for expectation in definition_expectations(claim, source=source):
        gap = grade_definition(conn, expectation)
        if gap is not None:
            gaps.append(gap)
    for expectation in edge_expectations(claim):
        gap = grade_edge(conn, expectation)
        if gap is not None:
            gaps.append(gap)
    return {
        "exact_store_complete": not gaps,
        "gaps": [
            {"item": gap.item, "primary_reason": gap.primary_reason,
             "detail": gap.detail}
            for gap in gaps],
    }
