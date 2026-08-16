#!/usr/bin/env python3
"""Build reviewable source-grounded candidates for the 22-question gold standard.

Oracle assertions and anchors are evaluation data. They are used here to locate and
verify candidate paths, and never enter Ariadne retrieval, prompts, or production code.
The output is not gold until every claim's review block is explicitly accepted.

No embedding or LLM APIs are used.
"""
from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NONPRODUCTION = ("/test/", "/tests/", "/benchmark/", "/benchmarks/",
                 "/target/", "/generated/", "/example/", "/examples/",
                 "/sample/", "/samples/")
EDGE_TYPES = ("call", "type_ref", "implements", "contains", "companion")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(text: str) -> tuple[str, ...]:
    parts = re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+",
        (text or "").replace("_", " "))
    return tuple(part.lower() for part in parts if len(part) > 1)


def _nonproduction(path: str) -> bool:
    normalized = f"/{(path or '').lower().strip('/')}"
    return any(part in normalized for part in NONPRODUCTION)


def _chunks(values, size: int = 300):
    values = list(values)
    for start in range(0, len(values), size):
        yield values[start:start + size]
def _node_rows(conn: sqlite3.Connection, source: str, canonical_ids) -> dict[str, dict]:
    found = {}
    for chunk in _chunks(sorted(set(canonical_ids))):
        if not chunk:
            continue
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                "SELECT canonical_id,qualified_name,file,line_start,line_end,kind,"
                "display_name,parent_qualified_name,language FROM scip_symbols "
                f"WHERE source_name=? AND canonical_id IN ({marks}) "
                "AND canonical_id NOT GLOB ?", [source, *chunk, "local *"]):
            if _nonproduction(str(row[2])):
                continue
            found[str(row[0])] = {
                "canonical_id": str(row[0]), "qualified_name": str(row[1]),
                "file": str(row[2]), "line_start": int(row[3]),
                "line_end": int(row[4] or row[3]), "kind": str(row[5] or ""),
                "display_name": str(row[6] or ""),
                "parent_qualified_name": str(row[7] or ""),
                "language": str(row[8] or "")}
    return found
def _anchor_rows(conn: sqlite3.Connection, source: str, needle: str,
                 repos: list[str], *, bound: int = 5000) -> list[tuple]:
    """Recall repo-scoped matches before a deterministic global fallback."""
    columns = (
        "canonical_id,qualified_name,file,line_start,line_end,kind,"
        "display_name,parent_qualified_name,language")
    predicate = "(qualified_name LIKE ? OR display_name LIKE ?)"
    pattern = f"%{needle}%"
    rows = []
    seen = set()

    def collect(query: str, parameters: list) -> None:
        for row in conn.execute(query, parameters):
            canonical_id = str(row[0])
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            rows.append(tuple(row))

    for repo in repos:
        repo_pattern = f"%{repo.lower()}%"
        collect(
            f"SELECT {columns} FROM scip_symbols "
            f"WHERE source_name=? AND {predicate} "
            "AND canonical_id NOT GLOB ? "
            "AND (lower(qualified_name) LIKE ? OR lower(file) LIKE ?) "
            "ORDER BY qualified_name,canonical_id LIMIT ?",
            [source, pattern, pattern, "local *",
             repo_pattern, repo_pattern, max(bound, 0)])
    collect(
        f"SELECT {columns} FROM scip_symbols "
        f"WHERE source_name=? AND {predicate} "
        "AND canonical_id NOT GLOB ? "
        "ORDER BY qualified_name,canonical_id LIMIT ?",
        [source, pattern, pattern, "local *", max(bound, 0)])
    return rows
def normalize_anchor_spec(anchor: str | dict) -> dict:
    """Normalize legacy labels and exact compiler-addressable anchor targets."""
    if isinstance(anchor, str):
        return {
            "anchor": anchor,
            "symbol": anchor,
            "strict": False,
        }
    if not isinstance(anchor, dict):
        raise TypeError("anchor must be a string or object")
    symbol = str(
        anchor.get("symbol")
        or anchor.get("qualified_name")
        or anchor.get("anchor")
        or anchor.get("label")
        or "").strip()
    label = str(
        anchor.get("anchor")
        or anchor.get("label")
        or symbol).strip()
    if not label or not symbol:
        raise ValueError("structured anchor requires anchor and symbol")
    normalized = dict(anchor)
    normalized["anchor"] = label
    normalized["symbol"] = symbol
    normalized["strict"] = bool(anchor.get("strict", True))
    return normalized
def _structured_anchor_rows(
        conn: sqlite3.Connection, source: str, spec: dict) -> list[tuple]:
    """Resolve an exact target, recovering duplicate definition locations from SCIP."""
    columns = (
        "canonical_id,qualified_name,file,line_start,line_end,kind,"
        "display_name,parent_qualified_name,language")
    canonical_id = str(spec.get("canonical_id", "")).strip()
    if canonical_id:
        predicate = "canonical_id=?"
        target = canonical_id
    else:
        predicate = "lower(qualified_name)=lower(?)"
        target = str(spec["symbol"])
    rows = [tuple(row) for row in conn.execute(
        f"SELECT {columns} FROM scip_symbols "
        f"WHERE source_name=? AND {predicate} "
        "AND canonical_id NOT GLOB ? "
        "ORDER BY qualified_name,canonical_id",
        [source, target, "local *"])]
    file_target = str(spec.get("file", "")).strip().replace("\\", "/")
    if not file_target:
        return rows

    def matches_file(value: str) -> bool:
        normalized = str(value).replace("\\", "/")
        return (
            normalized == file_target
            or normalized.endswith("/" + file_target)
        )

    if any(matches_file(str(row[2])) for row in rows):
        return rows

    requested_line = spec.get("line")
    recovered = []
    for row in rows:
        canonical = str(row[0])
        sites = conn.execute(
            "SELECT file,MIN(line),MAX(line) FROM scip_edges "
            "WHERE caller_canonical_id=? GROUP BY file ORDER BY file",
            [canonical],
        )
        for site_file, minimum_line, maximum_line in sites:
            if not matches_file(str(site_file)):
                continue
            if _nonproduction(str(site_file)):
                continue
            start = (
                int(requested_line)
                if requested_line is not None
                else int(minimum_line or 0)
            )
            end = max(start, int(maximum_line or start))
            if requested_line is not None and int(maximum_line or 0) < start:
                continue
            recovered.append((
                row[0], row[1], str(site_file), start, end,
                row[5], row[6], row[7], row[8],
            ))
    return rows + recovered
def _selected_endpoint_nodes(
        nodes: dict[str, dict], node_ids: list[str], *,
        left_id: str, left_candidate: dict,
        right_id: str, right_candidate: dict) -> list[dict]:
    """Keep the exact selected location when one canonical id has many definitions."""
    selected = {
        left_id: left_candidate,
        right_id: right_candidate,
    }
    location_keys = (
        "qualified_name", "file", "line_start", "line_end", "kind",
        "display_name", "parent_qualified_name", "language",
    )
    result = []
    for node_id in node_ids:
        node = dict(nodes[node_id])
        endpoint = selected.get(node_id)
        if endpoint is not None:
            for key in location_keys:
                if key in endpoint:
                    node[key] = endpoint[key]
        result.append(node)
    return result
def anchor_candidates(conn: sqlite3.Connection, source: str, anchor: str | dict,
                      assertion: str, repos: list[str], *, limit: int) -> list[dict]:
    """Rank compiler endpoints, honoring exact structured oracle targets."""
    spec = normalize_anchor_spec(anchor)
    target = str(spec["symbol"])
    needle = target.rsplit(".", 1)[-1]
    rows = _structured_anchor_rows(conn, source, spec) if bool(spec.get('strict')) else _anchor_rows(conn, source, needle, repos)
    anchor_tokens = set(_tokens(target))
    context_tokens = set(_tokens(assertion))
    if re.fullmatch(r"[A-Z][A-Za-z0-9]*", needle):
        anchor_role = "type"
    elif re.fullmatch(r"[A-Z][A-Z0-9_]*", needle):
        anchor_role = "constant"
    else:
        anchor_role = "member"
    strict = bool(spec.get("strict"))
    target_lower = target.lower()
    canonical_target = str(spec.get("canonical_id", "")).strip()
    owner_target = str(spec.get("owner", "")).strip().lower()
    file_target = str(spec.get("file", "")).strip().replace("\\", "/")
    kind_target = str(spec.get("kind", "")).strip().lower()
    line_target = spec.get("line")
    ranked = []
    for row in rows:
        canonical, qualified, file, start, end, kind, display, parent, language = row
        if _nonproduction(str(file)):
            continue
        qualified_text = str(qualified)
        qualified_lower = qualified_text.lower()
        canonical_text = str(canonical)
        if canonical_target:
            target_rank = 0 if canonical_text == canonical_target else 4
            target_match = (
                "canonical-exact" if target_rank == 0 else "canonical-mismatch")
        elif qualified_lower == target_lower:
            target_rank = 0
            target_match = "qualified-exact"
        elif qualified_lower.endswith("." + target_lower):
            target_rank = 1
            target_match = "qualified-suffix"
        else:
            target_rank = 3
            target_match = "lexical"
        if strict and target_rank > 1:
            continue
        normalized_file = str(file).replace("\\", "/")
        if file_target and not (
                normalized_file == file_target
                or normalized_file.endswith("/" + file_target)):
            continue
        if kind_target and str(kind or "").lower() != kind_target:
            continue
        if line_target is not None and not (
                int(start) <= int(line_target) <= int(end or start)):
            continue
        parent_text = str(parent or "").lower()
        if owner_target and not (
                parent_text == owner_target
                or parent_text.endswith("." + owner_target)
                or qualified_lower.startswith(owner_target + ".")
                or ("." + owner_target + ".") in qualified_lower):
            continue
        terminal = qualified_text.rsplit(".", 1)[-1]
        segments = set(re.split(r"[.#/$]+", qualified_text))
        candidate_text = " ".join(
            (qualified_text, str(display or ""), str(parent or ""), str(file)))
        candidate_tokens = set(_tokens(candidate_text))
        identity_rank = (
            0 if str(display).lower() == needle.lower()
            or terminal.lower() == needle.lower() else
            1 if needle.lower() in {segment.lower() for segment in segments} else
            2 if terminal.lower().endswith(needle.lower()) else 3)
        repo_hints = [repo for repo in repos
                      if repo.lower() in f"{qualified} {file}".lower()]
        context_overlap = len(candidate_tokens & context_tokens)
        anchor_overlap = len(candidate_tokens & anchor_tokens)
        is_type_kind = str(kind) in {
            "Class", "Object", "Trait", "Interface", "Enum"}
        callable_kind = str(kind) in {
            "Method", "Function", "Constructor", "Class", "Object",
            "Trait", "Interface"}
        role_kind_rank = (
            0 if anchor_role == "type" and is_type_kind else
            1 if anchor_role == "type" else
            0 if anchor_role != "type" and callable_kind else 1)
        if strict:
            if anchor_role in {"type", "constant"}:
                score = (
                    target_rank, identity_rank, -len(repo_hints), role_kind_rank,
                    -context_overlap, -anchor_overlap, len(candidate_tokens),
                    qualified_text, canonical_text)
            else:
                score = (
                    target_rank, -len(repo_hints), -context_overlap,
                    -anchor_overlap, identity_rank, role_kind_rank,
                    len(candidate_tokens), qualified_text, canonical_text)
        elif anchor_role in {"type", "constant"}:
            score = (
                identity_rank, -len(repo_hints), role_kind_rank,
                -context_overlap, -anchor_overlap, len(candidate_tokens),
                qualified_text, canonical_text)
        else:
            score = (
                -len(repo_hints), -context_overlap, -anchor_overlap,
                identity_rank, role_kind_rank, len(candidate_tokens),
                qualified_text, canonical_text)
        ranked.append((score, {
            "canonical_id": canonical_text, "qualified_name": qualified_text,
            "file": str(file), "line_start": int(start),
            "line_end": int(end or start), "kind": str(kind or ""),
            "display_name": str(display or ""),
            "parent_qualified_name": str(parent or ""),
            "language": str(language or ""), "repo_hints": repo_hints,
            "context_overlap": context_overlap,
            "anchor_overlap": anchor_overlap, "anchor_role": anchor_role,
            "identity_rank": identity_rank,
            "target_match": target_match,
            "match": "exact" if identity_rank == 0 else
                     "owner" if identity_rank == 1 else
                     "suffix" if identity_rank == 2 else "substring"}))
    ranked.sort(key=lambda item: item[0])
    result = []
    for rank, (_score, candidate) in enumerate(ranked[:max(limit, 0)]):
        candidate["rank"] = rank
        result.append(candidate)
    return result
def _companion_edges(
        conn: sqlite3.Connection, source: str, frontier: set[str], *,
        incoming: bool) -> list[tuple]:
    nodes = _node_rows(conn, source, frontier)
    counterpart_ids = set()
    relationships = []
    type_kinds = {"class", "trait", "interface", "enum"}
    term_kinds = {"object", "module"}
    for canonical_id, node in nodes.items():
        kind = str(node.get("kind", "")).lower()
        if kind in type_kinds and canonical_id.endswith("#"):
            counterpart = canonical_id[:-1] + "."
            caller, callee = counterpart, canonical_id
        elif kind in term_kinds and canonical_id.endswith("."):
            counterpart = canonical_id[:-1] + "#"
            caller, callee = canonical_id, counterpart
        else:
            continue
        counterpart_ids.add(counterpart)
        relationships.append((caller, callee, canonical_id, counterpart))
    counterparts = _node_rows(conn, source, counterpart_ids)
    edges = []
    for caller, callee, canonical_id, counterpart in relationships:
        other = counterparts.get(counterpart)
        current = nodes.get(canonical_id)
        if other is None or current is None:
            continue
        if other.get("qualified_name") != current.get("qualified_name"):
            continue
        if incoming and callee not in frontier:
            continue
        if not incoming and caller not in frontier:
            continue
        term = nodes.get(caller) or counterparts.get(caller)
        if term is None:
            continue
        edges.append((
            caller, callee, "companion", term["file"],
            int(term["line_start"]), "exact-structural"))
    return edges


def _has_shared_callee_turn(edges: list[dict]) -> bool:
    for left, right in zip(edges, edges[1:]):
        if "companion" in {left.get("edge_type"), right.get("edge_type")}:
            continue
        if (
                left.get("callee_canonical_id")
                == right.get("callee_canonical_id")
                and left.get("traversal") == "caller_to_callee"
                and right.get("traversal") == "callee_to_caller"):
            return True
    return False
def _load_scoped_edges(conn: sqlite3.Connection, source: str,
                       frontier: set[str], *, incoming: bool,
                       bound: int) -> list[tuple]:
    rows = _companion_edges(conn, source, frontier, incoming=incoming)
    for chunk in _chunks(sorted(frontier)):
        marks = ",".join("?" * len(chunk))
        edge_marks = ",".join("?" * len(EDGE_TYPES))
        column = "callee_canonical_id" if incoming else "caller_canonical_id"
        queried = conn.execute(
            "SELECT caller_canonical_id,callee_canonical_id,edge_type,file,line,confidence "
            f"FROM scip_edges WHERE {column} IN ({marks}) "
            f"AND edge_type IN ({edge_marks}) LIMIT ?",
            [*chunk, *EDGE_TYPES, max(bound, 0)])
        rows.extend(row for row in queried if not _nonproduction(str(row[3])))
    candidate_ids = {
        str(row[0] if incoming else row[1]) for row in rows}
    scoped = set(_node_rows(conn, source, candidate_ids))
    return [tuple(row) for row in rows
            if str(row[0] if incoming else row[1]) in scoped]
def _scoped_edges(conn: sqlite3.Connection, source: str, frontier: set[str], *,
                  incoming: bool, bound: int,
                  edge_cache: dict | None = None) -> list[tuple]:
    requested = max(bound, 0)
    if not frontier or requested == 0:
        return []
    if edge_cache is None:
        return _load_scoped_edges(
            conn, source, frontier, incoming=incoming, bound=requested)
    missing = []
    for canonical_id in sorted(frontier):
        cached = edge_cache.get((source, incoming, canonical_id))
        if cached is None or cached[0] < requested:
            missing.append(canonical_id)
    for canonical_id in missing:
        loaded = _load_scoped_edges(
            conn, source, {canonical_id}, incoming=incoming,
            bound=requested)
        owned = [
            tuple(edge) for edge in loaded
            if str(edge[1] if incoming else edge[0]) == canonical_id]
        edge_cache[(source, incoming, canonical_id)] = (
            requested, tuple(owned[:requested]))
    result = []
    for canonical_id in sorted(frontier):
        cached = edge_cache.get((source, incoming, canonical_id))
        if cached is not None:
            result.extend(cached[1])
    return result


def _reconstruct(hit: str, parents: dict[str, tuple[str, tuple, str]]) -> tuple[list[str], list[dict]]:
    nodes = [hit]
    steps = []
    current = hit
    while current in parents:
        previous, edge, traversal = parents[current]
        caller, callee, edge_type, file, line, confidence = edge
        steps.append({
            "caller_canonical_id": str(caller),
            "callee_canonical_id": str(callee), "edge_type": str(edge_type),
            "file": str(file), "line": int(line),
            "confidence": str(confidence or ""), "traversal": traversal,
            "compiler_verified": True,
            "compiler_source": (
                "scip-parent" if str(edge_type) == "contains"
                else "scip-edge")})
        nodes.append(previous)
        current = previous
    nodes.reverse()
    steps.reverse()
    return nodes, steps
def shortest_path(conn: sqlite3.Connection, source: str, starts: set[str],
                  targets: set[str], *, mode: str, max_depth: int,
                  max_frontier: int, edge_cache: dict | None = None,
                  search_node_limit: int | None = None) -> tuple[list[str], list[dict]]:
    starts = set(_node_rows(conn, source, starts))
    targets = set(_node_rows(conn, source, targets))
    overlap = sorted(starts & targets)
    if overlap:
        return [overlap[0]], []
    frontier = set(sorted(starts)[:max(max_frontier, 0)])
    visited = set(starts)
    parents: dict[str, tuple[str, tuple, str]] = {}
    expanded = 0
    for _depth in range(max(max_depth, 0)):
        if not frontier:
            break
        if (search_node_limit is not None
                and expanded + len(frontier) > max(search_node_limit, 0)):
            break
        expanded += len(frontier)
        rows = []
        if mode in {"directed", "undirected"}:
            rows.extend((row, "caller_to_callee") for row in _scoped_edges(
                conn, source, frontier, incoming=False,
                bound=max_frontier * 12, edge_cache=edge_cache))
        if mode == "undirected":
            rows.extend((row, "callee_to_caller") for row in _scoped_edges(
                conn, source, frontier, incoming=True,
                bound=max_frontier * 12, edge_cache=edge_cache))
        ranked = sorted(rows, key=lambda item: (
            0 if item[0][2] == "call" else 1,
            str(item[0][3]), int(item[0][4]), str(item[0][0]), str(item[0][1]),
            item[1]))
        next_frontier = set()
        for edge, traversal in ranked:
            caller, callee = str(edge[0]), str(edge[1])
            previous, candidate = ((caller, callee) if traversal == "caller_to_callee"
                                   else (callee, caller))
            if previous not in frontier or candidate in visited or candidate in next_frontier:
                continue
            parents[candidate] = (previous, tuple(edge), traversal)
            next_frontier.add(candidate)
            if candidate in targets:
                return _reconstruct(candidate, parents)
            if len(next_frontier) >= max_frontier:
                break
        visited.update(next_frontier)
        frontier = next_frontier
    return [], []


def candidate_path(conn: sqlite3.Connection, source: str, left: list[dict],
                   right: list[dict], *, max_depth: int,
                   max_frontier: int, edge_cache: dict | None = None,
                   search_node_limit: int | None = None) -> dict | None:
    shared_cache = {} if edge_cache is None else edge_cache
    left_by_id = {item["canonical_id"]: item for item in left}
    right_by_id = {item["canonical_id"]: item for item in right}
    overlap = set(left_by_id) & set(right_by_id)
    left_by_id = {key: value for key, value in left_by_id.items()
                  if key not in overlap}
    right_by_id = {key: value for key, value in right_by_id.items()
                   if key not in overlap}
    if not left_by_id or not right_by_id:
        return None
    attempts = (
        ("left-to-right", set(left_by_id), set(right_by_id), "directed"),
        ("right-to-left", set(right_by_id), set(left_by_id), "directed"),
        ("bidirectional-reference", set(left_by_id), set(right_by_id), "undirected"),
    )
    for orientation, starts, targets, mode in attempts:
        node_ids, edges = shortest_path(
            conn, source, starts, targets, mode=mode,
            max_depth=max_depth, max_frontier=max_frontier,
            edge_cache=shared_cache, search_node_limit=search_node_limit)
        if not node_ids:
            continue
        nodes = _node_rows(conn, source, node_ids)
        if len(nodes) != len(node_ids):
            continue
        for edge in edges:
            edge["caller"] = nodes.get(edge["caller_canonical_id"], {}).get(
                "qualified_name", edge["caller_canonical_id"])
            edge["callee"] = nodes.get(edge["callee_canonical_id"], {}).get(
                "qualified_name", edge["callee_canonical_id"])
        if orientation == "right-to-left":
            left_id, right_id = node_ids[-1], node_ids[0]
        else:
            left_id, right_id = node_ids[0], node_ids[-1]
        if left_id not in left_by_id or right_id not in right_by_id:
            continue

        def endpoint_summary(candidate: dict) -> dict:
            return {key: candidate[key]
                    for key in ("canonical_id", "qualified_name", "rank")
                    if key in candidate}

        return {
            "orientation": orientation,
            "nodes": [nodes[node_id] for node_id in node_ids],
            "edges": edges, "hop_count": len(edges),
            "endpoint_candidates": {
                "left": endpoint_summary(left_by_id[left_id]),
                "right": endpoint_summary(right_by_id[right_id])},
            "all_edges_compiler_verified": all(
                edge["compiler_verified"] for edge in edges)}
    return None
def _edge_record(edge: tuple, traversal: str) -> dict:
    caller, callee, edge_type, file, line, confidence = edge
    compiler_source = (
        "scip-parent" if str(edge_type) == "contains" else
        "scip-companion" if str(edge_type) == "companion" else
        "scip-edge")
    return {
        "caller_canonical_id": str(caller),
        "callee_canonical_id": str(callee),
        "edge_type": str(edge_type),
        "file": str(file),
        "line": int(line or 0),
        "confidence": str(confidence or ""),
        "traversal": traversal,
        "compiler_verified": True,
        "compiler_source": compiler_source}
def _path_scope(file: str) -> str:
    normalized = str(file or "").replace("\\", "/").strip("/")
    return normalized.split("/", 1)[0] if normalized else ""


def _search_neighbors(
        conn: sqlite3.Connection, source: str, node: str, *,
        backward: bool, mode: str, max_frontier: int,
        edge_cache: dict | None, preferred_file: str = "", 
definition_sites: dict[str, dict] | None = None) -> list[tuple[str, tuple, str]]:
    "Return graph neighbors and route-direction traversal metadata."
    bound = max(int(max_frontier), 1)
    rows = []
    if mode == "directed":
        if backward:
            for edge in _scoped_edges(
                    conn, source, {node}, incoming=True, bound=bound,
                    edge_cache=edge_cache):
                if str(edge[1]) == node:
                    rows.append((
                        str(edge[0]), tuple(edge), "caller_to_callee"))
        else:
            for edge in _scoped_edges(
                    conn, source, {node}, incoming=False, bound=bound,
                    edge_cache=edge_cache):
                if str(edge[0]) == node:
                    rows.append((
                        str(edge[1]), tuple(edge), "caller_to_callee"))
    else:
        for edge in _scoped_edges(
                conn, source, {node}, incoming=False, bound=bound,
                edge_cache=edge_cache):
            if str(edge[0]) != node:
                continue
            rows.append((
                str(edge[1]), tuple(edge),
                "callee_to_caller" if backward else "caller_to_callee"))
        for edge in _scoped_edges(
                conn, source, {node}, incoming=True, bound=bound,
                edge_cache=edge_cache):
            if str(edge[1]) != node:
                continue
            rows.append((
                str(edge[0]), tuple(edge),
                "caller_to_callee" if backward else "callee_to_caller"))
    if definition_sites:
        remapped_rows = []
        for neighbor, edge, traversal in rows:
            edge = tuple(edge)
            if str(edge[2]) == "companion":
                site = definition_sites.get(str(edge[0]), {})
                site_file = str(site.get("file", ""))
                if site_file:
                    edge = (
                        edge[0], edge[1], edge[2], site_file,
                        int(site.get("line_start", edge[4]) or edge[4]), edge[5])
            remapped_rows.append((neighbor, edge, traversal))
        rows = remapped_rows
    preferred_scope = _path_scope(preferred_file)
    if preferred_scope:
        local_rows = [
            item for item in rows
            if _path_scope(str(item[1][3])) == preferred_scope]
        if local_rows:
            rows = local_rows
    edge_priority = {
        "call": 0, "type_ref": 1, "implements": 2,
        "contains": 3, "companion": 4}
    unique = {}
    for neighbor, edge, traversal in rows:
        key = (
            neighbor, str(edge[0]), str(edge[1]), str(edge[2]),
            str(edge[3]), int(edge[4] or 0), traversal)
        unique.setdefault(key, (neighbor, edge, traversal))
    ranked = sorted(unique.values(), key=lambda item: (
        edge_priority.get(str(item[1][2]), 9),
        str(item[1][3]), int(item[1][4] or 0),
        str(item[1][0]), str(item[1][1]), item[2]))
    return ranked[:bound]


def _reconstruct_bidirectional(
        meeting: str, forward_origin: str, backward_origin: str,
        forward_parents: dict, backward_parents: dict
) -> tuple[list[str], list[dict]]:
    forward_state = (meeting, forward_origin)
    forward_nodes = [meeting]
    forward_edges = []
    current = forward_state
    while current in forward_parents:
        previous, edge, traversal = forward_parents[current]
        forward_nodes.append(previous[0])
        forward_edges.append(_edge_record(edge, traversal))
        current = previous
    forward_nodes.reverse()
    forward_edges.reverse()

    backward_state = (meeting, backward_origin)
    backward_nodes = []
    backward_edges = []
    current = backward_state
    while current in backward_parents:
        following, edge, traversal = backward_parents[current]
        backward_nodes.append(following[0])
        backward_edges.append(_edge_record(edge, traversal))
        current = following
    return (
        [*forward_nodes, *backward_nodes],
        [*forward_edges, *backward_edges])


def _bidirectional_search(
        conn: sqlite3.Connection, source: str, starts: list[dict],
        targets: list[dict], *, mode: str, max_depth: int,
        max_frontier: int, edge_cache: dict,
        search_node_limit: int | None,
        excluded_pairs: set[tuple[str, str]]
) -> dict | None:
    """Find the best endpoint pair without enumerating their Cartesian product."""
    forward_heap = []
    backward_heap = []
    forward_distance = {}
    backward_distance = {}
    forward_parents = {}
    backward_parents = {}
    forward_at_node = {}
    backward_at_node = {}
    settled_forward = set()
    settled_backward = set()

    def seed(candidates, heap, distance, at_node):
        for candidate in candidates:
            node = str(candidate["canonical_id"])
            state = (node, node)
            score = (int(candidate.get("rank", 0)), 0)
            if score >= distance.get(state, (10 ** 9, 10 ** 9)):
                continue
            distance[state] = score
            at_node.setdefault(node, set()).add(state)
            heapq.heappush(heap, (*score, node, node))
    start_ids = {str(candidate["canonical_id"]) for candidate in starts}
    target_ids = {str(candidate["canonical_id"]) for candidate in targets}
    preferred_files = {str(candidate["canonical_id"]): str(candidate.get("file", "")) for candidate in [*starts, *targets]}
    definition_sites = {str(candidate["canonical_id"]): dict(candidate) for candidate in [*starts, *targets]}
    eligible_starts = [
        candidate for candidate in starts
        if any(
            (str(candidate["canonical_id"]), target_id)
            not in excluded_pairs
            for target_id in target_ids)]
    eligible_targets = [
        candidate for candidate in targets
        if any(
            (start_id, str(candidate["canonical_id"]))
            not in excluded_pairs
            for start_id in start_ids)]
    seed(eligible_starts, forward_heap, forward_distance, forward_at_node)
    seed(eligible_targets, backward_heap, backward_distance, backward_at_node)
    if not forward_heap or not backward_heap:
        return None

    best = None
    def consider(node: str) -> None:
        nonlocal best
        for forward_state in sorted(forward_at_node.get(node, ())):
            for backward_state in sorted(backward_at_node.get(node, ())):
                pair = (forward_state[1], backward_state[1])
                if pair in excluded_pairs:
                    continue
                forward_score = forward_distance[forward_state]
                backward_score = backward_distance[backward_state]
                hop_count = forward_score[1] + backward_score[1]
                if hop_count > max(max_depth, 0):
                    continue
                _node_ids, route_edges = _reconstruct_bidirectional(
                    node, pair[0], pair[1],
                    forward_parents, backward_parents)
                if mode == "undirected" and _has_shared_callee_turn(route_edges):
                    continue
                key = (
                    forward_score[0] + backward_score[0],
                    hop_count, pair[0], pair[1], node)
                if best is None or key < best[0]:
                    best = (key, node, pair[0], pair[1])

    for node in set(forward_at_node) & set(backward_at_node):
        consider(node)

    def peek(heap, distance, settled):
        while heap:
            rank, hops, node, origin = heap[0]
            state = (node, origin)
            if state in settled or distance.get(state) != (rank, hops):
                heapq.heappop(heap)
                continue
            return rank, hops
        return None
    def frontier_minima(heap, distance, settled):
        minima = {}
        for rank, hops, node, origin in heap:
            state = (node, origin)
            if state in settled or distance.get(state) != (rank, hops):
                continue
            score = (rank, hops)
            if score < minima.get(origin, (10 ** 9, 10 ** 9)):
                minima[origin] = score
        return minima

    def pop(heap, distance, settled):
        while heap:
            rank, hops, node, origin = heapq.heappop(heap)
            state = (node, origin)
            if state in settled or distance.get(state) != (rank, hops):
                continue
            return state, (rank, hops)
        return None

    expanded = 0
    exhausted = False
    hard_limit = (
        None if search_node_limit is None
        else max(int(search_node_limit), 1) * 2)
    forward_turn = True
    while forward_heap and backward_heap:
        forward_lower = peek(
            forward_heap, forward_distance, settled_forward)
        backward_lower = peek(
            backward_heap, backward_distance, settled_backward)
        if forward_lower is None or backward_lower is None:
            break
        forward_minima = frontier_minima(
            forward_heap, forward_distance, settled_forward)
        backward_minima = frontier_minima(
            backward_heap, backward_distance, settled_backward)
        compatible_lower = min((
            forward_score[0] + backward_score[0],
            forward_score[1] + backward_score[1])
            for forward_origin, forward_score in forward_minima.items()
            for backward_origin, backward_score in backward_minima.items()
            if (forward_origin, backward_origin) not in excluded_pairs
        ) if any(
            (forward_origin, backward_origin) not in excluded_pairs
            for forward_origin in forward_minima
            for backward_origin in backward_minima
        ) else None
        if best is not None and (
                compatible_lower is None
                or compatible_lower >= best[0][:2]):
            break
        if hard_limit is not None and expanded >= hard_limit:
            exhausted = True
            break

        use_forward = forward_turn
        forward_turn = not forward_turn
        if use_forward:
            popped = pop(
                forward_heap, forward_distance, settled_forward)
            if popped is None:
                continue
            state, score = popped
            settled_forward.add(state)
            distance = forward_distance
            parents = forward_parents
            at_node = forward_at_node
            opposite_at_node = backward_at_node
            heap = forward_heap
            backward = False
        else:
            popped = pop(
                backward_heap, backward_distance, settled_backward)
            if popped is None:
                continue
            state, score = popped
            settled_backward.add(state)
            distance = backward_distance
            parents = backward_parents
            at_node = backward_at_node
            opposite_at_node = forward_at_node
            heap = backward_heap
            backward = True

        expanded += 1
        node, origin = state
        consider(node)
        if score[1] >= max(max_depth, 0):
            continue
        for neighbor, edge, traversal in _search_neighbors(
                conn, source, node, backward=backward, mode=mode,
                max_frontier=max_frontier, edge_cache=edge_cache, preferred_file = preferred_files.get(origin, ""), definition_sites = definition_sites):
            next_state = (neighbor, origin)
            next_score = (score[0], score[1] + 1)
            if next_score >= distance.get(
                    next_state, (10 ** 9, 10 ** 9)):
                continue
            distance[next_state] = next_score
            parents[next_state] = (state, edge, traversal)
            at_node.setdefault(neighbor, set()).add(next_state)
            heapq.heappush(
                heap, (*next_score, neighbor, origin))
            if neighbor in opposite_at_node:
                consider(neighbor)

    if best is None:
        return None
    _key, meeting, forward_origin, backward_origin = best
    node_ids, edges = _reconstruct_bidirectional(
        meeting, forward_origin, backward_origin,
        forward_parents, backward_parents)
    if len(node_ids) != len(set(node_ids)):
        return None
    return {
        "node_ids": node_ids,
        "edges": edges,
        "start_id": forward_origin,
        "target_id": backward_origin,
        "search": {
            "algorithm": "bidirectional-dijkstra",
            "expanded_nodes": expanded,
            "limit": search_node_limit,
            "exhausted": exhausted,
            "mode": mode}}
def bidirectional_candidate_path(
        conn: sqlite3.Connection, source: str, left: list[dict],
        right: list[dict], *, max_depth: int, max_frontier: int,
        edge_cache: dict | None = None,
        search_node_limit: int | None = None,
        excluded_pairs: set[tuple[str, str]] | None = None
) -> dict | None:
    """Resolve the best ranked route after comparing every traversal mode."""
    shared_cache = {} if edge_cache is None else edge_cache
    left_by_id = {
        str(item["canonical_id"]): item for item in left}
    right_by_id = {
        str(item["canonical_id"]): item for item in right}
    valid_ids = set(_node_rows(
        conn, source, set(left_by_id) | set(right_by_id)))
    left_by_id = {
        key: value for key, value in left_by_id.items()
        if key in valid_ids}
    right_by_id = {
        key: value for key, value in right_by_id.items()
        if key in valid_ids}
    overlap = set(left_by_id) & set(right_by_id)
    left_by_id = {
        key: value for key, value in left_by_id.items()
        if key not in overlap}
    right_by_id = {
        key: value for key, value in right_by_id.items()
        if key not in overlap}
    if not left_by_id or not right_by_id:
        return None
    excluded = set(excluded_pairs or set())
    minimum_endpoint_rank_sum = min(
        (
            int(left_candidate.get("rank", 0))
            + int(right_candidate.get("rank", 0))
            for left_id, left_candidate in left_by_id.items()
            for right_id, right_candidate in right_by_id.items()
            if (left_id, right_id) not in excluded
        ),
        default=None)
    if minimum_endpoint_rank_sum is None:
        return None
    attempts = (
        ("left-to-right", list(left_by_id.values()),
         list(right_by_id.values()), "directed", False),
        ("right-to-left", list(right_by_id.values()),
         list(left_by_id.values()), "directed", True),
        ("bidirectional-reference", list(left_by_id.values()),
         list(right_by_id.values()), "undirected", False))
    orientation_penalty = {
        "left-to-right": 0,
        "right-to-left": 1,
        "bidirectional-reference": 2}
    route_candidates = []
    for orientation, starts, targets, mode, reversed_logical in attempts:
        attempt_depth = max_depth
        if route_candidates:
            best_key = min(item[0] for item in route_candidates)
            if best_key[0] == minimum_endpoint_rank_sum:
                attempt_depth = min(max_depth, best_key[1] - 1)
                if attempt_depth <= 0:
                    break
        if mode == "undirected" and attempt_depth < 2:
            break
        oriented_excluded = (
            {(right_id, left_id) for left_id, right_id in excluded}
            if reversed_logical else excluded)
        found = _bidirectional_search(
            conn, source, starts, targets, mode=mode,
            max_depth=attempt_depth, max_frontier=max_frontier,
            edge_cache=shared_cache,
            search_node_limit=search_node_limit,
            excluded_pairs=oriented_excluded)
        if found is None:
            continue
        node_ids = found["node_ids"]
        nodes = _node_rows(conn, source, node_ids)
        if len(nodes) != len(node_ids):
            continue
        edges = found["edges"]
        for edge in edges:
            edge["caller"] = nodes.get(
                edge["caller_canonical_id"], {}).get(
                    "qualified_name", edge["caller_canonical_id"])
            edge["callee"] = nodes.get(
                edge["callee_canonical_id"], {}).get(
                    "qualified_name", edge["callee_canonical_id"])
        if reversed_logical:
            left_id = found["target_id"]
            right_id = found["start_id"]
        else:
            left_id = found["start_id"]
            right_id = found["target_id"]
        if left_id not in left_by_id or right_id not in right_by_id:
            continue

        def endpoint_summary(candidate: dict) -> dict:
            return {
                key: candidate[key]
                for key in ("canonical_id", "qualified_name", "rank")
                if key in candidate}

        candidate = {
            "orientation": orientation,
            "nodes": _selected_endpoint_nodes(nodes, node_ids, left_id=left_id, left_candidate=left_by_id[left_id], right_id=right_id, right_candidate=right_by_id[right_id]),
            "edges": edges,
            "hop_count": len(edges),
            "endpoint_candidates": {
                "left": endpoint_summary(left_by_id[left_id]),
                "right": endpoint_summary(right_by_id[right_id])},
            "search": found["search"],
            "all_edges_compiler_verified": all(
                edge["compiler_verified"] for edge in edges)}
        endpoint_rank_sum = (
            int(left_by_id[left_id].get("rank", 0))
            + int(right_by_id[right_id].get("rank", 0)))
        route_key = (
            endpoint_rank_sum,
            candidate["hop_count"],
            orientation_penalty[orientation],
            tuple(node_ids))
        route_candidates.append((route_key, candidate))
    if not route_candidates:
        return None
    route_candidates.sort(key=lambda item: item[0])
    return route_candidates[0][1]


def candidate_paths(conn: sqlite3.Connection, source: str, left: list[dict],
                    right: list[dict], *, max_depth: int, max_frontier: int,
                    endpoint_limit: int, path_limit: int,
                    edge_cache: dict | None = None,
                    search_node_limit: int | None = None) -> list[dict]:
    """Retain ranked routes without enumerating every endpoint pair."""
    if path_limit <= 0 or endpoint_limit <= 0:
        return []
    shared_cache = {} if edge_cache is None else edge_cache
    left_candidates = left[:max(endpoint_limit, 0)]
    right_candidates = right[:max(endpoint_limit, 0)]
    excluded_pairs = set()
    paths = []
    seen_routes = set()
    while len(paths) < path_limit:
        path = bidirectional_candidate_path(
            conn, source, left_candidates, right_candidates,
            max_depth=max_depth, max_frontier=max_frontier,
            edge_cache=shared_cache,
            search_node_limit=search_node_limit,
            excluded_pairs=excluded_pairs)
        if path is None:
            break
        endpoints = path["endpoint_candidates"]
        pair = (
            str(endpoints["left"]["canonical_id"]),
            str(endpoints["right"]["canonical_id"]))
        if pair in excluded_pairs:
            break
        excluded_pairs.add(pair)
        route_key = (
            tuple(node["canonical_id"] for node in path["nodes"]),
            tuple((
                edge["caller_canonical_id"],
                edge["callee_canonical_id"],
                edge["edge_type"], edge["traversal"])
                for edge in path["edges"]))
        if route_key in seen_routes:
            continue
        seen_routes.add(route_key)
        orientation_penalty = {
            "left-to-right": 0,
            "right-to-left": 1,
            "bidirectional-reference": 2,
        }.get(path["orientation"], 3)
        endpoint_rank_sum = (
            int(endpoints["left"].get("rank", 0))
            + int(endpoints["right"].get("rank", 0)))
        path["selection_score"] = {
            "endpoint_rank_sum": endpoint_rank_sum,
            "orientation_penalty": orientation_penalty,
            "hop_count": path["hop_count"]}
        paths.append(path)
    paths.sort(key=lambda item: (
        item["selection_score"]["endpoint_rank_sum"],
        item["selection_score"]["hop_count"],
        item["selection_score"]["orientation_penalty"],
        item["endpoint_candidates"]["left"].get("canonical_id", ""),
        item["endpoint_candidates"]["right"].get("canonical_id", "")))
    for rank, path in enumerate(paths):
        path["alternative_rank"] = rank
    return paths


def path_proof_errors(path: dict) -> list[str]:
    """Return all reasons a candidate path lacks exact source proof."""
    errors = []
    nodes = path.get("nodes", [])
    edges = path.get("edges", [])
    materialization = path.get("materialization", {})
    excerpts = materialization.get("excerpts", [])
    for gap in materialization.get("gaps", []):
        errors.append(f"materialization gap: {gap}")
    if not nodes:
        errors.append("path has no nodes")
    for node in nodes:
        label = node.get("qualified_name") or node.get("canonical_id", "<unknown>")
        file = str(node.get("file", ""))
        line = int(node.get("line_start", 0))
        if _nonproduction(file):
            errors.append(f"non-production node: {file}:{line}")
        proven = any(
            excerpt.get("kind") == "definition"
            and excerpt.get("file") == file
            and int(excerpt.get("line_start", 0)) <= line
            <= int(excerpt.get("line_end", 0))
            and bool(str(excerpt.get("content", "")).strip())
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}",
                                  str(excerpt.get("sha256", ""))))
            for excerpt in excerpts)
        if not proven:
            errors.append(f"missing definition proof: {label}")
    for edge in edges:
        file = str(edge.get("file", ""))
        line = int(edge.get("line", 0))
        if not edge.get("compiler_verified"):
            errors.append(f"unverified compiler edge: {file}:{line}")
        if _nonproduction(file):
            errors.append(f"non-production edge site: {file}:{line}")
        expected_kind = (
            "definition" if edge.get("edge_type") == "contains"
            else "call_site")
        proven = any(
            excerpt.get("kind") == expected_kind
            and excerpt.get("file") == file
            and int(excerpt.get("line_start", 0)) <= line
            <= int(excerpt.get("line_end", 0))
            and bool(str(excerpt.get("content", "")).strip())
            and bool(re.fullmatch(r"[0-9a-fA-F]{64}",
                                  str(excerpt.get("sha256", ""))))
            for excerpt in excerpts)
        if not proven:
            label = (
                "containment" if edge.get("edge_type") == "contains"
                else "call-site")
            errors.append(f"missing {label} proof: {file}:{line}")
    return list(dict.fromkeys(errors))


def materialize_path(path: dict | None, *, source: str,
                     source_root: str | None) -> dict:
    if path is None or source_root is None:
        return {"excerpts": [], "gaps": (["source root unavailable"]
                if source_root is None else [])}
    from library.source_materialization import materialize_citations
    from library.structural_assembly import StructuralCitation
    nodes_by_name = {node["qualified_name"]: node for node in path["nodes"]}
    citations = []
    for node in path["nodes"]:
        citations.append(StructuralCitation(
            qualified_name=node["qualified_name"], file=node["file"],
            line_start=node["line_start"], line_end=node["line_end"],
            source_name=source, relation="localized", hop=0,
            call_site_file=node["file"], call_site_line=node["line_start"],
            stop_reason="gold_candidate"))
    for hop, edge in enumerate(path["edges"], 1):
        callee = nodes_by_name.get(edge["callee"])
        if callee is None:
            continue
        relation = (
            "calls" if edge["edge_type"] == "call" else
            "contains" if edge["edge_type"] == "contains" else
            "references")
        citations.append(StructuralCitation(
            qualified_name=callee["qualified_name"], file=callee["file"],
            line_start=callee["line_start"], line_end=callee["line_end"],
            source_name=source, relation=relation, hop=hop,
            call_site_file=edge["file"], call_site_line=edge["line"],
            stop_reason="gold_candidate", parent_qualified_name=edge["caller"]))
    materialized = materialize_citations(citations, {source: source_root})
    return {"excerpts": [asdict(item) for item in materialized.excerpts],
            "gaps": list(materialized.gaps)}
def materialize_claim_witnesses(
        witness_specs: list[dict], *, source: str,
        source_root: str | None) -> list[dict]:
    """Materialize and validate exact non-topological claim evidence."""
    from library.source_materialization import materialize_citations

    results = []
    seen_ids = set()
    for raw in witness_specs:
        spec = dict(raw)
        witness_id = str(spec.get("id", "")).strip()
        file = str(spec.get("file", "")).strip()
        line_start = int(spec.get("line_start", 0) or 0)
        line_end = int(spec.get("line_end", line_start) or 0)
        required = spec.get("contains", [])
        if isinstance(required, str):
            required = [required]
        required = [str(value) for value in required if str(value)]
        errors = []
        if not witness_id:
            errors.append("witness id is empty")
        elif witness_id in seen_ids:
            raise ValueError(f"duplicate witness id: {witness_id}")
        seen_ids.add(witness_id)
        if not file:
            errors.append("witness file is empty")
        if _nonproduction(file):
            errors.append(f"non-production witness: {file}:{line_start}")
        if line_start < 1 or line_end < line_start:
            errors.append(
                f"invalid witness range: {file}:{line_start}-{line_end}")

        if errors:
            materialization = {"excerpts": [], "gaps": []}
        else:
            roots = {source: source_root} if source_root is not None else {}
            fetched = materialize_citations(
                [], roots,
                extra_ranges=((
                    source, file, line_start, line_end, "claim_witness"),))
            materialization = {
                "excerpts": [asdict(item) for item in fetched.excerpts],
                "gaps": list(fetched.gaps)}
            excerpts = materialization["excerpts"]
            if len(excerpts) != 1:
                errors.append(
                    f"missing exact witness range: "
                    f"{file}:{line_start}-{line_end}")
            else:
                excerpt = excerpts[0]
                digest = str(excerpt.get("sha256", ""))
                if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    errors.append("witness source hash is missing")
                content = str(excerpt.get("content", ""))
                for fragment in required:
                    if fragment not in content:
                        errors.append(
                            f"missing required fragment: {fragment}")
        results.append({
            "id": witness_id,
            "file": file,
            "line_start": line_start,
            "line_end": line_end,
            "contains": required,
            "materialization": materialization,
            "proof_errors": list(dict.fromkeys(errors)),
        })
    return results


def claim_path_coverage(anchor_names: list[str], paths: list[dict]) -> dict:
    """Describe whether the supplied paths connect every anchor as one claim."""
    ordered = list(dict.fromkeys(anchor_names))
    parent = {anchor: anchor for anchor in ordered}

    def find(anchor: str) -> str:
        while parent[anchor] != anchor:
            parent[anchor] = parent[parent[anchor]]
            anchor = parent[anchor]
        return anchor

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    covered = set(ordered if len(ordered) <= 1 else [])
    for path in paths:
        connected = [anchor for anchor in path.get("connects", [])
                     if anchor in parent]
        covered.update(connected)
        for left, right in zip(connected, connected[1:]):
            union(left, right)
    grouped = {}
    for anchor in ordered:
        grouped.setdefault(find(anchor), []).append(anchor)
    components = list(grouped.values())
    missing = [anchor for anchor in ordered if anchor not in covered]
    complete = not missing and len(components) <= 1
    return {
        "covered_anchors": [anchor for anchor in ordered if anchor in covered],
        "missing_anchors": missing, "components": components,
        "complete": complete}
def coherent_path_sets(
        anchor_names: list[str], paths: list[dict], *, limit: int,
        required_transitions: list[list[str]] | None = None,
        reverse_transitions: list[list[str]] | None = None) -> list[dict]:
    "Find coherent endpoint assignments that honor the declared claim topology."
    ordered = list(dict.fromkeys(anchor_names))
    if len(ordered) < 2 or limit <= 0:
        return []
    anchor_set = set(ordered)
    normalized_transitions = []
    required_pairs = set()
    for raw_transition in required_transitions or []:
        if len(raw_transition) != 2:
            raise ValueError(
                f"required transition must name two anchors: {raw_transition!r}")
        left, right = (str(value) for value in raw_transition)
        pair = tuple((left, right))
        if left == right or left not in anchor_set or right not in anchor_set:
            raise ValueError(
                f"invalid required transition: {raw_transition!r}")
        if pair in required_pairs:
            continue
        required_pairs.add(pair)
        normalized_transitions.append([left, right])
    normalized_reverse_transitions = []
    reverse_pairs = set()
    for raw_transition in reverse_transitions or []:
        if len(raw_transition) != 2:
            raise ValueError(
                f"reverse transition must name two anchors: {raw_transition!r}")
        pair = tuple(str(value) for value in raw_transition)
        if pair not in required_pairs:
            raise ValueError(
                f"reverse transition is not required: {raw_transition!r}")
        if pair in reverse_pairs:
            continue
        reverse_pairs.add(pair)
        normalized_reverse_transitions.append(list(pair))
    proof_pairs = {
        ((pair[1], pair[0]) if pair in reverse_pairs else pair): pair
        for pair in required_pairs}
    if len(proof_pairs) != len(required_pairs):
        raise ValueError("required transitions have ambiguous proof directions")
    usable = []
    for path in paths:
        connected = path.get("connects", [])
        endpoints = path.get("endpoint_candidates", {})
        if len(connected) != 2 or not set(connected) <= anchor_set:
            continue
        connected_pair = tuple(str(value) for value in connected)
        reverse_pair = (connected_pair[1], connected_pair[0])
        orientation = str(path.get("orientation", "left-to-right"))
        if orientation == "right-to-left":
            actual_pair = reverse_pair
        elif orientation == "bidirectional-reference" and required_pairs:
            if connected_pair in proof_pairs:
                actual_pair = connected_pair
            elif reverse_pair in proof_pairs:
                actual_pair = reverse_pair
            else:
                actual_pair = connected_pair
        else:
            actual_pair = connected_pair
        if required_pairs:
            transition_pair = proof_pairs.get(actual_pair)
            if transition_pair is None:
                continue
        else:
            transition_pair = actual_pair
        left_id = endpoints.get("left", {}).get("canonical_id")
        right_id = endpoints.get("right", {}).get("canonical_id")
        if not left_id or not right_id or path.get("proof_errors"):
            continue
        path_score = path.get("selection_score", {})
        score = (
            int(path_score.get("endpoint_rank_sum", 0)) * 100
            + int(path_score.get(
                "hop_count", path.get("hop_count", 0))) * 10
            + int(path_score.get("orientation_penalty", 0)))
        usable.append((
            score, str(path.get("id", "")), path,
            {connected[0]: str(left_id), connected[1]: str(right_id)},
            transition_pair))
    usable.sort(key=lambda item: (item[0], item[1]))
    first_anchor = ordered[0]
    results = {}
    result_budget = max(limit * 10, limit)

    def search(
            covered: set[str], assignments: dict[str, str],
            selected: list[str], selected_pairs: set[tuple[str]],
            score: int) -> None:
        if len(results) >= result_budget:
            return
        if covered == anchor_set:
            if not required_pairs <= selected_pairs:
                return
            key = (tuple(sorted(set(selected))), tuple(sorted(assignments.items())))
            current = results.get(key)
            if current is None or score < current["score"]:
                results[key] = {
                    "path_ids": sorted(set(selected)),
                    "selected_candidate_by_anchor": dict(assignments),
                    "score": score,
                    "complete": True,
                    "required_transitions": list(normalized_transitions),
                    "reverse_transitions": list(
                        normalized_reverse_transitions),
                    "required_transitions_complete": True}
            return
        for (
                path_score, path_id, _path, endpoint_map,
                transition_pair) in usable:
            connected = set(endpoint_map)
            if not connected & covered:
                continue
            if connected <= covered:
                continue
            if any(
                    anchor in assignments
                    and assignments[anchor] != canonical_id
                    for anchor, canonical_id in endpoint_map.items()):
                continue
            next_assignments = dict(assignments)
            next_assignments.update(endpoint_map)
            search(
                covered | connected, next_assignments,
                [*selected, path_id],
                selected_pairs | {transition_pair},
                score + path_score)

    search({first_anchor}, {}, [], set(), 0)
    found = sorted(results.values(), key=lambda item: (
        item["score"], item["path_ids"],
        sorted(item["selected_candidate_by_anchor"].items())))
    return found[:limit]
def _render_claim_witness_summary(witness: dict) -> list[str]:
    witness_id = str(witness.get("id", "<unnamed>"))
    file = str(witness.get("file", ""))
    line_start = int(witness.get("line_start", 0))
    line_end = int(witness.get("line_end", line_start))
    lines = [f"- Witness {witness_id}: {file}:{line_start}-{line_end}"]
    excerpts = witness.get("materialization", {}).get("excerpts", [])
    if excerpts:
        content = str(excerpts[0].get("content", ""))
        lines.extend(["  source:", *[
            f"    {line}" for line in content.splitlines()]])
    for gap in witness.get("materialization", {}).get("gaps", []):
        lines.append(f"  - MATERIALIZATION GAP: {gap}")
    for error in witness.get("proof_errors", []):
        lines.append(f"  - PROOF ERROR: {error}")
    return lines


def render_review_queue(report: dict) -> str:
    """Render a compact, model-free queue for explicit human adjudication."""
    lines = [
        "# Gold-chain candidate review",
        "",
        f"Status: {report.get('status', 'candidate oracle')}",
        "",
        "Accept a claim only after checking every selected endpoint, edge, "
        "source excerpt, and omitted alternative.",
        ""]
    for question in report.get("questions", []):
        lines.extend([
            f"## Q{question['id']}: {question.get('question', '')}",
            "",
            f"Question review: {question.get('review', {}).get('status', 'pending')}",
            ""])
        for claim in question.get("claims", []):
            coverage = claim.get("candidate_coverage", {})
            coherent = [
                item for item in claim.get("coherent_path_sets", [])
                if item.get("complete")]
            lines.extend([
                f"### {claim.get('id', '')}",
                "",
                str(claim.get("assertion", "")),
                "",
                "Candidate connectivity: "
                f"{'complete' if coverage.get('complete') else 'incomplete'}; "
                f"missing={coverage.get('missing_anchors', [])}; "
                f"components={coverage.get('components', [])}",
                f"Coherent complete sets: {len(coherent)}",
                "",
                "Anchors:",
                ""])
            for anchor in claim.get("anchors", []):
                lines.append(f"- {anchor.get('anchor', '')}")
                for candidate in anchor.get("candidates", [])[:5]:
                    lines.append(
                        "  - [{rank}] {name} — {file}:{line} "
                        "(match={match}, context={context}, repos={repos})".format(
                            rank=candidate.get("rank", "?"),
                            name=candidate.get("qualified_name", ""),
                            file=candidate.get("file", ""),
                            line=candidate.get("line_start", "?"),
                            match=candidate.get("match", ""),
                            context=candidate.get("context_overlap", 0),
                            repos=candidate.get("repo_hints", [])))
            if claim.get("witnesses"):
                lines.extend(["", "Claim witnesses:", ""])
                for witness in claim.get("witnesses", []):
                    lines.extend(_render_claim_witness_summary(witness))
            lines.extend(["", "Routes:", ""])
            for path in claim.get("candidate_paths", []):
                endpoints = path.get("endpoint_candidates", {})
                left = endpoints.get("left", {}).get(
                    "qualified_name",
                    endpoints.get("left", {}).get("canonical_id", ""))
                right = endpoints.get("right", {}).get(
                    "qualified_name",
                    endpoints.get("right", {}).get("canonical_id", ""))
                lines.append(
                    f"- {path.get('id', '')}: {left} → {right}; "
                    f"{path.get('orientation', '')}, {path.get('hop_count', 0)} hop(s)")
                proof_errors = path.get("proof_errors", [])
                if proof_errors:
                    for error in proof_errors:
                        lines.append(f"  - PROOF ERROR: {error}")
                else:
                    lines.append("  - source proof: complete")
            if coherent:
                lines.extend(["", "Coherent selections:", ""])
                for item in coherent:
                    lines.append(
                        f"- score={item.get('score', 0)} "
                        f"paths={item.get('path_ids', [])} "
                        f"endpoints={item.get('selected_candidate_by_anchor', {})}")
            lines.extend([
                "",
                "Decision: pending",
                "Selected endpoints: {}",
                "Selected paths: []",
                "Completeness notes:",
                ""])
    return "\n".join(lines).rstrip() + "\n"


def validate_review(path: Path) -> list[str]:
    report = json.loads(path.read_text())
    errors = []
    for question in report.get("questions", []):
        question_prefix = "Q{}".format(question["id"])
        question_review = question.get("review", {})
        if question_review.get("status") != "accepted":
            errors.append(f"{question_prefix}: question review is not accepted")
        if not str(question_review.get("answer", "")).strip():
            errors.append(f"{question_prefix}: reviewed answer is empty")
        for claim in question.get("claims", []):
            prefix = "{}:{}".format(question_prefix, claim["id"])
            for witness in claim.get("witnesses", []):
                witness_id = str(witness.get("id", "<unnamed>"))
                for gap in witness.get("materialization", {}).get("gaps", []):
                    errors.append(
                        f"{prefix}: witness {witness_id}: materialization gap: {gap}")
                for error in witness.get("proof_errors", []):
                    errors.append(f"{prefix}: witness {witness_id}: {error}")
            review = claim.get("review", {})
            if review.get("status") != "accepted":
                errors.append(f"{prefix}: review is not accepted")
            if review.get("claim_correct") is not True:
                errors.append(f"{prefix}: claim correctness not confirmed")
            if review.get("complete") is not True:
                errors.append(f"{prefix}: completeness not confirmed")
            chosen = review.get("selected_candidate_by_anchor", {})
            anchor_names = []
            for anchor in claim.get("anchors", []):
                anchor_name = anchor["anchor"]
                anchor_names.append(anchor_name)
                if anchor_name not in chosen:
                    errors.append(f"{prefix}: no reviewed candidate for {anchor_name}")
                    continue
                candidates = anchor.get("candidates", [])
                available_candidates = {
                    value for candidate in candidates
                    for value in (candidate.get("canonical_id"),
                                  candidate.get("qualified_name")) if value}
                if available_candidates and chosen[anchor_name] not in available_candidates:
                    errors.append(
                        f"{prefix}: unknown candidate for {anchor_name}: "
                        f"{chosen[anchor_name]}")
            available_paths = {
                item["id"]: item for item in claim.get("candidate_paths", [])}
            selected_paths = []
            for selected in review.get("selected_path_ids", []):
                candidate = available_paths.get(selected)
                if candidate is None:
                    errors.append(f"{prefix}: unknown selected path {selected}")
                    continue
                selected_paths.append(candidate)
                if not candidate.get("all_edges_compiler_verified"):
                    errors.append(f"{prefix}:{selected}: unverified compiler edge")
                endpoints = candidate.get("endpoint_candidates", {})
                connected = candidate.get("connects", [])
                for side, anchor_name in zip(("left", "right"), connected):
                    endpoint_id = endpoints.get(side, {}).get("canonical_id")
                    if (endpoint_id and anchor_name in chosen
                            and chosen[anchor_name] != endpoint_id):
                        errors.append(
                            f"{prefix}:{selected}: selected endpoint for "
                            f"{anchor_name} does not match {endpoint_id}")
                for proof_error in path_proof_errors(candidate):
                    errors.append(f"{prefix}:{selected}: {proof_error}")
            coverage = claim_path_coverage(anchor_names, selected_paths)
            if not coverage["complete"]:
                errors.append(
                    f"{prefix}: selected paths do not connect every anchor; "
                    f"missing={coverage['missing_anchors']} "
                    f"components={coverage['components']}")
            elif len(anchor_names) > 1 and not coherent_path_sets(
                    anchor_names, selected_paths, limit=1, required_transitions = claim.get("transitions", []), reverse_transitions = claim.get("reverse_transitions", [])):
                errors.append(
                    f"{prefix}: selected paths have no coherent endpoint assignment or required transition proof")
    return errors
def parse_claim_filter(value: str | None) -> set[tuple[int, str]] | None:
    if not value or not value.strip():
        return None
    selected = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"invalid claim selector {item!r}; expected QID:claim-id")
        question_text, claim_id = item.split(":", 1)
        if not question_text.isdigit() or not claim_id.strip():
            raise ValueError(
                f"invalid claim selector {item!r}; expected QID:claim-id")
        selected.add((int(question_text), claim_id.strip()))
    return selected or None


def claim_selected(question_id: int, claim_id: str,
                   selected: set[tuple[int, str]] | None) -> bool:
    return selected is None or (question_id, claim_id) in selected
def recohere_report(report: dict, requirements: list[dict], *, limit: int) -> dict:
    # Recompute claim topology from existing paths without querying SCIP.
    refreshed = json.loads(json.dumps(report))
    requirement_questions = {int(item["id"]): item for item in requirements}
    for question in refreshed.get("questions", []):
        question_id = int(question["id"])
        requirement_question = requirement_questions.get(question_id)
        if requirement_question is None:
            raise ValueError(f"Q{question_id}: missing requirement")
        requirement_claims = {
            str(item["id"]): item
            for item in requirement_question.get("claims", [])}
        for claim in question.get("claims", []):
            claim_id = str(claim["id"])
            requirement = requirement_claims.get(claim_id)
            if requirement is None:
                raise ValueError(f"Q{question_id}:{claim_id}: missing requirement")
            candidate_anchors = [
                str(item["anchor"]) for item in claim.get("anchors", [])]
            required_anchors = [
                str(item["anchor"])
                for item in requirement.get("anchors", [])]
            if candidate_anchors != required_anchors:
                raise ValueError(
                    f"Q{question_id}:{claim_id}: candidate anchors differ from requirements")
            claim["transitions"] = json.loads(json.dumps(
                requirement.get("transitions", [])))
            reverse = requirement.get("reverse_transitions", [])
            if reverse:
                claim["reverse_transitions"] = json.loads(json.dumps(reverse))
            else:
                claim.pop("reverse_transitions", None)
            claim["candidate_coverage"] = claim_path_coverage(
                candidate_anchors, claim.get("candidate_paths", []))
            claim["coherent_path_sets"] = coherent_path_sets(
                candidate_anchors, claim.get("candidate_paths", []),
                limit=limit,
                required_transitions=claim.get("transitions", []),
                reverse_transitions=claim.get("reverse_transitions", []))
            claim["review"] = {
                "status": "pending",
                "selected_candidate_by_anchor": {},
                "selected_path_ids": [],
                "claim_correct": None,
                "complete": None,
                "notes": "",
            }
        question["review"] = {
            "status": "pending", "answer": "", "notes": ""}
    refreshed["status"] = "candidate-oracle; not gold until reviewed"
    refreshed.setdefault("parameters", {})["coherent_set_limit"] = limit
    refreshed["topology_refresh"] = {
        "mode": "existing-paths-only",
        "question_count": len(refreshed.get("questions", [])),
        "claim_count": sum(
            len(question.get("claims", []))
            for question in refreshed.get("questions", [])),
    }
    return refreshed
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(ROOT / "ariadne.db"))
    parser.add_argument("--requirements", default=str(HERE / "requirements.json"))
    parser.add_argument("--questions", default=str(
        ROOT / "evaluation/spool-clean-room/questions_debcrumb_ask.json"))
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--source-root", default="")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument(
        "--only-claims",
        help="comma-separated QID:claim-id selectors")
    parser.add_argument("--candidate-limit", type=int, default=12)
    parser.add_argument("--endpoint-limit", type=int, default=4)
    parser.add_argument("--paths-per-anchor-pair", type=int, default=3)
    parser.add_argument("--coherent-set-limit", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-frontier", type=int, default=200)
    parser.add_argument("--search-node-limit", type=int, default=120,
                        help="maximum expanded graph nodes per endpoint pair")
    parser.add_argument("--validate", help="validate a human-reviewed candidate file")
    parser.add_argument("--recohere", help="recompute topology from existing candidate paths")
    parser.add_argument("--review-out", help="write compact Markdown review queue")
    parser.add_argument("--out")
    args = parser.parse_args()
    try:
        selected_claims = parse_claim_filter(args.only_claims)
    except ValueError as exc:
        parser.error(str(exc))
    if args.validate:
        errors = validate_review(Path(args.validate))
        for error in errors:
            print(f"ERROR: {error}")
        print(f"review validation: {len(errors)} error(s)")
        return 1 if errors else 0
    if not args.out:
        parser.error("--out is required unless --validate is used")

    requirements_path = Path(args.requirements)
    if args.recohere:
        requirement_rows = json.loads(requirements_path.read_text())
        report = json.loads(Path(args.recohere).read_text())
        refreshed = recohere_report(
            report, requirement_rows, limit=args.coherent_set_limit)
        Path(args.out).write_text(json.dumps(refreshed, indent=2) + "\n")
        claim_count = refreshed["topology_refresh"]["claim_count"]
        print(f"recohered -> {args.out}; {claim_count} claims")
        return 0
    questions_path = Path(args.questions)
    requirements = {int(item["id"]): item
                    for item in json.loads(requirements_path.read_text())}
    questions = {int(item["id"]): item.get("after") or item.get("question") or ""
                 for item in json.loads(questions_path.read_text())}
    ids = sorted(set(requirements) & set(questions))
    if args.only:
        wanted = {int(value) for value in args.only.split(",")}
        ids = [qid for qid in ids if qid in wanted]
    if selected_claims is not None:
        selected_question_ids = {qid for qid, _claim_id in selected_claims}
        ids = [qid for qid in ids if qid in selected_question_ids]
    source_root = args.source_root or None
    if source_root is None:
        try:
            from config import get_config
            configured = get_config().get_all_source_paths().get(args.source)
            source_root = str(configured) if configured else None
        except Exception:
            source_root = None

    conn = sqlite3.connect(args.db)
    output = []
    try:
        for qid in ids:
            claim_rows = []
            edge_cache = {}
            for claim in requirements[qid].get("claims", []):
                if not claim_selected(qid, claim["id"], selected_claims):
                    continue
                print(f"Q{qid}:{claim['id']}: building...", flush=True)
                anchors = []
                by_anchor = {}
                for raw_anchor in claim.get("anchors", []):
                    anchor_spec = normalize_anchor_spec(raw_anchor)
                    anchor_name = anchor_spec["anchor"]
                    if anchor_name in by_anchor:
                        raise ValueError(
                            f"Q{qid}:{claim['id']}: duplicate anchor label {anchor_name}")
                    candidates = anchor_candidates(
                        conn, args.source, anchor_spec, claim["assertion"],
                        claim.get("repos", []), limit=args.candidate_limit)
                    by_anchor[anchor_name] = candidates
                    anchors.append({
                        "anchor": anchor_name,
                        "target": anchor_spec,
                        "candidates": candidates,
                        "resolved": bool(candidates),
                    })
                paths = []
                anchor_names = list(by_anchor)
                for left_index, left_name in enumerate(anchor_names):
                    for right_name in anchor_names[left_index + 1:]:
                        alternatives = candidate_paths(
                            conn, args.source,
                            by_anchor[left_name], by_anchor[right_name],
                            max_depth=args.max_depth,
                            max_frontier=args.max_frontier,
                            endpoint_limit=args.endpoint_limit,
                            path_limit=args.paths_per_anchor_pair,
                            edge_cache=edge_cache, search_node_limit = args.search_node_limit)
                        for candidate in alternatives:
                            path_id = (
                                f"{left_name}->{right_name}#"
                                f"{candidate['alternative_rank'] + 1}")
                            candidate["id"] = path_id
                            candidate["connects"] = [left_name, right_name]
                            candidate["materialization"] = materialize_path(
                                candidate, source=args.source,
                                source_root=source_root)
                            candidate["proof_errors"] = path_proof_errors(candidate)
                            paths.append(candidate)
                coverage = claim_path_coverage(anchor_names, paths)
                coherent = coherent_path_sets(
                    anchor_names, paths, limit=args.coherent_set_limit, required_transitions = claim.get("transitions", []), reverse_transitions = claim.get("reverse_transitions", []))
                witnesses = materialize_claim_witnesses(
                    claim.get("witnesses", []), source=args.source,
                    source_root=source_root)
                row = {
                    "id": claim["id"], "assertion": claim["assertion"],
                    "repos": claim.get("repos", []), "anchors": anchors,
                    "candidate_paths": paths,
                    "candidate_coverage": coverage,
                    "coherent_path_sets": coherent,
                    "unresolved_anchors": [item["anchor"] for item in anchors
                                           if not item["resolved"]],
                    "review": {
                        "status": "pending",
                        "selected_candidate_by_anchor": {},
                        "selected_path_ids": [], "claim_correct": None,
                        "complete": None, "alternative_paths": [], "notes": ""}, "witnesses": witnesses, "transitions": claim.get("transitions", [])}
                claim_rows.append(row)
                print(
                    f"Q{qid}:{claim['id']}: "
                    f"{sum(item['resolved'] for item in anchors)}/{len(anchors)} "
                    f"anchors, {len(paths)} path(s), "
                    f"{len(coherent)} coherent set(s)",
                    flush=True)
            if not claim_rows:
                continue
            output.append({"id": qid, "question": questions[qid],
                           "claims": claim_rows,
                           "review": {"status": "pending", "answer": "", "notes": ""}})
    finally:
        conn.close()
    report = {
        "status": "candidate-oracle; not gold until reviewed",
        "source": args.source, "source_root": source_root,
        "parameters": {
            "candidate_limit": args.candidate_limit,
            "endpoint_limit": args.endpoint_limit,
            "paths_per_anchor_pair": args.paths_per_anchor_pair,
            "coherent_set_limit": args.coherent_set_limit,
            "max_depth": args.max_depth,
            "max_frontier": args.max_frontier,
            "only_claims": (
                sorted(f"{qid}:{claim_id}" for qid, claim_id in selected_claims)
                if selected_claims is not None else []), "search_node_limit": args.search_node_limit},
        "provenance": {
            "database": str(Path(args.db).resolve()),
            "database_sha256": _sha256(Path(args.db)),
            "requirements_sha256": _sha256(requirements_path),
            "questions_sha256": _sha256(questions_path)},
        "questions": output}
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    if args.review_out:
        review_out = Path(args.review_out)
        review_out.write_text(render_review_queue(report))
        print(f"review queue -> {review_out}")
    claim_count = sum(len(question["claims"]) for question in output)
    print(f"candidate oracle -> {out} ({len(output)} questions, {claim_count} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
