#!/usr/bin/env python3
"""Build and query SCIP-only Leiden families in an external SQLite database.

This experiment intentionally has no document-table, embedding, theme-summary,
or provider dependency.  It projects exact same-source SCIP relations to a
weighted undirected graph for community discovery, while separately preserving
the original directed relation/site records for later strand construction.

Leiden answers only "which symbols belong to the same structural concern?".
It does *not* supply execution direction.  The query output therefore turns
exact member/source seeds into small directed strands before a later live arm
may select them and materialize source.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
PROBE_PATH = HERE / "claim_ledger_probe.py"
QUESTIONS = ROOT / "evaluation/spool-clean-room/questions_debcrumb_ask.json"

# Calls carry execution flow.  Implements/type references connect an API to
# its implementation.  Ownership is intentionally weak: it keeps members of
# a type coherent without allowing a large owner to swallow call communities.
# Only execution/contract/ownership relations form the discovery projection.
# A type reference can be an important proof edge later, but including every
# reference in community detection creates giant import-like components and
# obscures causal call neighbourhoods.  All four relations are still retained
# after clustering for directed strand construction.
_DISCOVERY_WEIGHTS = {
    "call": 4.0,
    "implements": 2.0,
    "contains": 0.20,
}
_PRESERVED_RELATIONS = frozenset((*_DISCOVERY_WEIGHTS, "type_ref"))

# These are question-language cues, not repository vocabulary.  They let the
# experiment keep explicit code components (``Spark V2``, ``DeltaTable``) apart
# from the prose which describes what the user wants to know.
_QUESTION_CAPITALS = frozenset({
    "a", "an", "and", "are", "can", "does", "how", "i", "in", "is",
    "it", "not", "the", "this", "what", "when", "where", "which", "why", "with",
})
_PROSE_CUES = frozenset({
    "also", "each", "fate", "for", "from", "into", "its", "order", "plain",
    "relative", "result", "row", "rows", "table", "than", "the", "then", "to",
    "versu", "versus", "what", "when", "with",
})
_ROLE_VOCABULARY = {
    "decision": frozenset({"action", "condition", "decid", "delete", "insert", "match", "rule", "update"}),
    "analysis": frozenset({"analysi", "analyz", "plan", "resolve", "rewrite"}),
    "execution": frozenset({"apply", "dispatch", "exec", "execute", "handle", "process", "run"}),
    "input": frozenset({"input", "load", "parse", "read", "receiv", "request"}),
    "ordering": frozenset({"after", "before", "order", "sequenc", "stage"}),
    "output": frozenset({"emit", "generat", "output", "produc", "result", "return", "write"}),
    "routing": frozenset({"divert", "path", "route"}),
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _load_probe():
    spec = importlib.util.spec_from_file_location("_structural_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    try:
        values = {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError as error:
        raise argparse.ArgumentTypeError("--only must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("--only must not be empty")
    return values


def _question_rows(path: Path, only: set[int] | None) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    selected = []
    for row in rows:
        question_id = int(row.get("id", -1))
        if only is not None and question_id not in only:
            continue
        text = str(row.get("after") or row.get("question") or "").strip()
        if text:
            selected.append({"id": question_id, "question": text})
    if only is not None and {item["id"] for item in selected} != only:
        raise ValueError(f"unknown question ids: {sorted(only - {item['id'] for item in selected})}")
    return selected


def _source_fingerprint(conn: sqlite3.Connection, source: str) -> str:
    digest = hashlib.sha256()
    for row in conn.execute(
        "SELECT canonical_id, file, line_start, line_end FROM scip_symbols "
        "WHERE source_name=? ORDER BY canonical_id", (source,),
    ):
        digest.update("\x1f".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _edge_rows(conn: sqlite3.Connection, source: str, probe) -> Iterable[tuple]:
    marks = ",".join("?" * len(_PRESERVED_RELATIONS))
    # Dedup by exact relation endpoints.  Repeated call/type-reference sites
    # strengthen that pair logarithmically; they never become duplicate graph
    # edges whose arbitrary ordering could perturb Leiden membership.
    sql = f"""
        SELECT e.caller_canonical_id, e.callee_canonical_id, e.edge_type,
               count(*) AS occurrences,
               min(e.file || char(31) || printf('%012d', e.line)) AS site_key,
               caller.qualified_name, caller.file, caller.line_start, caller.line_end,
               callee.qualified_name, callee.file, callee.line_start, callee.line_end
        FROM scip_edges AS e
        JOIN scip_symbols AS caller ON caller.canonical_id=e.caller_canonical_id
        JOIN scip_symbols AS callee ON callee.canonical_id=e.callee_canonical_id
        WHERE e.edge_type IN ({marks})
          AND caller.source_name=? AND callee.source_name=?
        GROUP BY e.caller_canonical_id, e.callee_canonical_id, e.edge_type
        ORDER BY e.caller_canonical_id, e.callee_canonical_id, e.edge_type
    """
    for row in conn.execute(sql, (*sorted(_PRESERVED_RELATIONS), source, source)):
        # ``min(file), min(line)`` can fabricate a file/line pair when an edge
        # has several sites.  Select one lexicographically ordered *pair* so a
        # later source citation always denotes a real compiler record.
        site_file, raw_line = str(row[4]).rsplit(chr(31), 1)
        site_line = int(raw_line)
        row = (*row[:4], site_file, site_line, *row[5:])
        caller_file, callee_file = str(row[7]), str(row[11])
        if not all(probe._is_production_path(path) for path in (caller_file, callee_file, site_file)):
            continue
        yield row


def _family_id(member_ids: list[str]) -> str:
    digest = hashlib.sha256("\n".join(member_ids).encode()).hexdigest()[:20]
    return f"sf-{digest}"


def _build(db: Path, output: Path, source: str, *, resolution: float,
           seed: int, min_size: int) -> dict[str, Any]:
    """Create an immutable structural-family database outside ``ariadne.db``."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite structural family database: {output}")
    import igraph as ig
    import leidenalg as la

    probe = _load_probe()
    source_conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    target = sqlite3.connect(output)
    started = time.monotonic()
    try:
        symbols: dict[str, tuple[str, str, int, int]] = {}
        undirected: dict[tuple[str, str], float] = defaultdict(float)
        for row in _edge_rows(source_conn, source, probe):
            (caller_id, callee_id, relation, occurrences, site_file, site_line,
             caller_name, caller_file, caller_start, caller_end,
             callee_name, callee_file, callee_start, callee_end) = row
            caller_id, callee_id, relation = str(caller_id), str(callee_id), str(relation)
            if relation not in _DISCOVERY_WEIGHTS:
                continue
            symbols.setdefault(caller_id, (str(caller_name), str(caller_file), int(caller_start), int(caller_end)))
            symbols.setdefault(callee_id, (str(callee_name), str(callee_file), int(callee_start), int(callee_end)))
            left, right = sorted((caller_id, callee_id))
            if left == right:
                continue
            undirected[(left, right)] += _DISCOVERY_WEIGHTS[relation] * min(4.0, 1.0 + math.log2(int(occurrences)))

        node_ids = sorted(symbols)
        node_index = {symbol_id: index for index, symbol_id in enumerate(node_ids)}
        edge_pairs = [(node_index[left], node_index[right]) for left, right in undirected]
        graph = ig.Graph(n=len(node_ids), edges=edge_pairs, directed=False)
        graph.es["weight"] = [undirected[pair] for pair in undirected]
        membership = la.find_partition(
            graph, la.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=resolution, seed=seed, n_iterations=10,
        ).membership
        grouped: dict[int, list[str]] = defaultdict(list)
        for index, community in enumerate(membership):
            grouped[int(community)].append(node_ids[index])
        # Tiny disconnected artifacts are intentionally not families.  They
        # remain in no family rather than being attached heuristically.
        groups = [sorted(members) for members in grouped.values() if len(members) >= min_size]
        groups.sort(key=lambda members: (members[0], len(members)))
        membership_by_symbol: dict[str, str] = {}
        members_by_family: dict[str, list[str]] = {}
        families: list[tuple[str, int, str]] = []
        for members in groups:
            family = _family_id(members)
            families.append((family, len(members), hashlib.sha256("\n".join(members).encode()).hexdigest()))
            members_by_family[family] = members
            membership_by_symbol.update({member: family for member in members})

        target.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE families (
                family_id TEXT PRIMARY KEY,
                member_count INTEGER NOT NULL,
                membership_sha256 TEXT NOT NULL
            );
            CREATE TABLE family_members (
                family_id TEXT NOT NULL,
                symbol_id TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                file TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                PRIMARY KEY (family_id, symbol_id)
            );
            CREATE INDEX family_members_symbol ON family_members(symbol_id);
            CREATE TABLE family_edges (
                family_id TEXT NOT NULL,
                caller_id TEXT NOT NULL,
                callee_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                occurrences INTEGER NOT NULL,
                site_file TEXT NOT NULL,
                site_line INTEGER NOT NULL,
                PRIMARY KEY (family_id, caller_id, callee_id, relation, site_file, site_line)
            );
            CREATE INDEX family_edges_family ON family_edges(family_id, relation);
            CREATE VIRTUAL TABLE family_member_search USING fts5(
                family_id UNINDEXED, symbol_id UNINDEXED, surface,
                tokenize='porter unicode61'
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        target.executemany("INSERT INTO families VALUES (?,?,?)", families)
        member_rows = [
            (family, symbol_id, *symbols[symbol_id])
            for family, members in sorted(members_by_family.items())
            for symbol_id in members
        ]
        target.executemany("INSERT INTO family_members VALUES (?,?,?,?,?,?)", member_rows)
        target.executemany(
            "INSERT INTO family_member_search(family_id, symbol_id, surface) VALUES (?,?,?)",
            ((family, symbol_id, probe._surface(name, file))
             for family, symbol_id, name, file, _start, _end in member_rows),
        )
        directed_edge_count = 0
        edge_batch: list[tuple[str, str, str, str, int, str, int]] = []
        # Re-read structural rows instead of retaining millions of Python edge
        # tuples while Leiden runs.  This keeps the full-store build bounded
        # while preserving all relation types for later directed strands.
        for row in _edge_rows(source_conn, source, probe):
            caller_id, callee_id, relation = str(row[0]), str(row[1]), str(row[2])
            family = membership_by_symbol.get(caller_id)
            if family is None or family != membership_by_symbol.get(callee_id):
                continue
            edge_batch.append((family, caller_id, callee_id, relation, int(row[3]), str(row[4]), int(row[5])))
            if len(edge_batch) == 5_000:
                target.executemany("INSERT INTO family_edges VALUES (?,?,?,?,?,?,?)", edge_batch)
                directed_edge_count += len(edge_batch)
                edge_batch.clear()
        if edge_batch:
            target.executemany("INSERT INTO family_edges VALUES (?,?,?,?,?,?,?)", edge_batch)
            directed_edge_count += len(edge_batch)
        metadata = {
            "schema": "scip-structural-leiden-families/v1",
            "source": source,
            "source_fingerprint": _source_fingerprint(source_conn, source),
            "resolution": str(resolution), "seed": str(seed), "min_size": str(min_size),
            "node_count": str(len(node_ids)), "undirected_edge_count": str(len(undirected)),
            "directed_edge_count": str(directed_edge_count), "family_count": str(len(families)),
        }
        target.executemany("INSERT INTO meta VALUES (?,?)", sorted(metadata.items()))
        target.commit()
        return {**metadata, "elapsed_seconds": round(time.monotonic() - started, 3),
                "family_db_bytes": output.stat().st_size}
    except BaseException:
        target.rollback()
        raise
    finally:
        target.close()
        source_conn.close()


def _quoted(token: str) -> str:
    return f'"{token.replace(chr(34), "")}"'


def _question_frame(question: str, probe) -> dict[str, Any]:
    """Extract code components and behavioural direction without a model.

    Components are explicit identifier-shaped words in the user question.  The
    remaining lexical terms describe the requested behaviour.  Neither list is
    repository-specific and no gold/evaluation data participates in this step.
    """
    tokens = tuple(dict.fromkeys(probe._terms(question)))
    if not tokens:
        raise ValueError("question produced no lexical retrieval terms")
    words = list(_WORD.finditer(question))
    raw_groups: list[list[str]] = []
    current: list[str] = []
    previous_index: int | None = None
    for index, match in enumerate(words):
        word = match.group(0)
        lower = word.lower()
        explicit = (
            len(word) > 1
            and lower not in _QUESTION_CAPITALS
            and word[0].isupper()
            and (not word.isupper() or len(word) <= 4 or any(char.isdigit() for char in word))
        )
        if explicit and previous_index == index - 1:
            current.append(word)
        elif explicit:
            if current:
                raw_groups.append(current)
            current = [word]
        elif current:
            raw_groups.append(current)
            current = []
        previous_index = index if explicit else None
    if current:
        raw_groups.append(current)
    components = tuple(dict.fromkeys(
        value for value in (tuple(dict.fromkeys(probe._terms(" ".join(group)))) for group in raw_groups)
        if value
    ))
    component_terms = {token for group in components for token in group}
    semantic_terms = tuple(token for token in tokens if token not in component_terms and token not in _PROSE_CUES)
    roles = tuple(
        role for role, vocabulary in _ROLE_VOCABULARY.items()
        if set(tokens) & vocabulary
    )
    variants: list[dict[str, Any]] = []
    for index, group in enumerate(components):
        component_query = " AND ".join(_quoted(token) for token in group)
        if semantic_terms:
            variants.append({
                "label": f"component-{index + 1}", "components": group,
                "query": f"({component_query}) AND ({' OR '.join(_quoted(token) for token in semantic_terms[:12])})",
            })
        variants.append({
            "label": f"component-{index + 1}-name", "components": group,
            "query": component_query,
        })
    if semantic_terms:
        variants.append({
            "label": "behaviour-fallback", "components": (),
            "query": " OR ".join(_quoted(token) for token in semantic_terms),
        })
    # Stable de-duplication means query ordering is itself deterministic and
    # becomes part of the recorded frame/provenance.
    seen_queries: set[str] = set()
    variants = [item for item in variants if not (item["query"] in seen_queries or seen_queries.add(item["query"]))]
    return {
        "terms": tokens, "components": components, "semantic_terms": semantic_terms,
        "roles": roles, "queries": variants,
    }


def _role_score(name: str, roles: tuple[str, ...], probe) -> int:
    terms = set(probe._terms(name))
    return sum(len(terms & _ROLE_VOCABULARY[role]) for role in roles)


def _is_runtime_source_fact(file: str, probe) -> bool:
    """Apply the source policy to path-like executable files as well."""
    if not probe._is_production_path(file):
        return False
    return not ({"test", "tests", "benchmark", "benchmarks"} & set(
        re.split(r"[-_./\\]+", str(file).lower())))


def _member_hits(conn: sqlite3.Connection, frame: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    """Return exact SCIP members, retaining which generic query found each."""
    hits: dict[tuple[str, str], dict[str, Any]] = {}
    for query_index, variant in enumerate(frame["queries"]):
        rows = conn.execute("""
            SELECT search.family_id, search.symbol_id, member.qualified_name,
                   member.file, member.line_start, member.line_end,
                   bm25(family_member_search) AS rank
            FROM family_member_search AS search
            JOIN family_members AS member
              ON member.family_id=search.family_id AND member.symbol_id=search.symbol_id
            WHERE family_member_search MATCH ?
            ORDER BY rank, search.family_id, search.symbol_id
            LIMIT ?
        """, (variant["query"], limit)).fetchall()
        for position, row in enumerate(rows, start=1):
            item = {
                "family_id": str(row[0]), "symbol_id": str(row[1]),
                "qualified_name": str(row[2]), "file": str(row[3]),
                "line_start": int(row[4]), "line_end": int(row[5]),
                "rank": float(row[6]), "query_index": query_index,
                "query_label": variant["label"], "query_position": position,
            }
            key = item["family_id"], item["symbol_id"]
            if key not in hits or (item["query_index"], item["rank"], item["symbol_id"]) < (
                    hits[key]["query_index"], hits[key]["rank"], hits[key]["symbol_id"]):
                hits[key] = item
    return list(hits.values())


def _source_fact_hits(conn: sqlite3.Connection, frame: dict[str, Any], probe, *, fact_k: int) -> list[dict[str, Any]]:
    """Retrieve exact source-backed symbols using the same generic frame."""
    hits: dict[str, dict[str, Any]] = {}
    per_query = max(fact_k * 2, 16)
    for query_index, variant in enumerate(frame["queries"]):
        rows = conn.execute("""
            SELECT f.symbol_id, f.qualified_name, f.file, f.line_start, f.line_end,
                   bm25(symbol_fact_search) AS rank
            FROM symbol_fact_search
            JOIN symbol_facts AS f ON f.rowid=symbol_fact_search.rowid
            WHERE symbol_fact_search MATCH ?
            ORDER BY rank, f.symbol_id
            LIMIT ?
        """, (variant["query"], per_query)).fetchall()
        for position, row in enumerate(rows, start=1):
            if not _is_runtime_source_fact(str(row[2]), probe):
                continue
            item = {
                "symbol_id": str(row[0]), "qualified_name": str(row[1]),
                "file": str(row[2]), "line_start": int(row[3]), "line_end": int(row[4]),
                "rank": float(row[5]), "query_index": query_index,
                "query_label": variant["label"], "query_position": position,
            }
            previous = hits.get(item["symbol_id"])
            if previous is None or (item["query_index"], item["rank"], item["symbol_id"]) < (
                    previous["query_index"], previous["rank"], previous["symbol_id"]):
                hits[item["symbol_id"]] = item
    return sorted(hits.values(), key=lambda item: (
        item["query_index"], item["rank"], item["symbol_id"]))[:fact_k]


def _owner_promotions(conn: sqlite3.Connection, hits: list[dict[str, Any]], frame: dict[str, Any], probe) -> list[dict[str, Any]]:
    """Promote a repeatedly matched member family to its exact SCIP owner.

    For example, many ``MergeRowsExec.<member>`` matches establish the class as
    a structural entry seed.  This uses only qualified-name containment and an
    exact member row; it never guesses a sibling or collapses overloaded names.
    """
    support: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for hit in hits:
        parts = hit["qualified_name"].split(".")
        for end in range(len(parts) - 1, 0, -1):
            support[hit["family_id"], ".".join(parts[:end])].append(hit)
    promoted: list[dict[str, Any]] = []
    requested_behaviour = set(frame["semantic_terms"])
    for (family, qualified_name), children in support.items():
        if len({child["symbol_id"] for child in children}) < 2:
            continue
        # Never promote a package-like common ancestor.  An owner must itself
        # name at least one requested behaviour (e.g. ``MergeRowsExec`` for a
        # merge question), not merely share a component such as ``Spark``.
        if not (set(probe._terms(qualified_name)) & requested_behaviour):
            continue
        row = conn.execute("""
            SELECT symbol_id, file, line_start, line_end
            FROM family_members
            WHERE family_id=? AND qualified_name=?
            ORDER BY symbol_id
            LIMIT 1
        """, (family, qualified_name)).fetchone()
        if row is None:
            continue
        representative = min(children, key=lambda child: (
            child["query_index"], child["rank"], child["symbol_id"]))
        promoted.append({
            "family_id": family, "symbol_id": str(row[0]), "qualified_name": qualified_name,
            "file": str(row[1]), "line_start": int(row[2]), "line_end": int(row[3]),
            "rank": representative["rank"], "query_index": representative["query_index"],
            "query_label": representative["query_label"],
            "query_position": representative["query_position"],
            "owner_support": len({child["symbol_id"] for child in children}),
        })
    return promoted


def _edge_item(row: tuple, names: dict[str, str], seed_id: str, *, bridge: bool,
               source_sites: sqlite3.Connection | None, site_cache: dict[tuple[str, str, str], tuple[str, int]]) -> dict[str, Any]:
    caller_id, callee_id, relation, occurrences, site_file, site_line = row
    if source_sites is not None:
        key = str(caller_id), str(callee_id), str(relation)
        exact = site_cache.get(key)
        if exact is None:
            exact = source_sites.execute("""
                SELECT file, line
                FROM scip_edges
                WHERE caller_canonical_id=? AND callee_canonical_id=? AND edge_type=?
                ORDER BY file, line
                LIMIT 1
            """, key).fetchone()
            if exact is None:
                raise RuntimeError(f"selected structural edge is absent from source SCIP: {key}")
            exact = str(exact[0]), int(exact[1])
            site_cache[key] = exact
        site_file, site_line = exact
    return {
        "caller_id": str(caller_id), "caller_name": names.get(str(caller_id), str(caller_id)),
        "callee_id": str(callee_id), "callee_name": names.get(str(callee_id), str(callee_id)),
        "relation": str(relation), "occurrences": int(occurrences),
        "file": str(site_file), "line": int(site_line),
        "seed_adjacent": str(caller_id) == seed_id or str(callee_id) == seed_id,
        "ownership_bridge": bridge,
    }


def _strand_edges(conn: sqlite3.Connection, family: str, seed_id: str, roles: tuple[str, ...], probe, *,
                  edge_k: int, owner_bridge_k: int, source_sites: sqlite3.Connection | None,
                  site_cache: dict[tuple[str, str, str], tuple[str, int]]) -> list[dict[str, Any]]:
    """Return one bounded, directed strand rooted at one exact SCIP symbol."""
    direct = conn.execute("""
        SELECT caller_id, callee_id, relation, occurrences, site_file, site_line
        FROM family_edges
        WHERE family_id=? AND (caller_id=? OR callee_id=?)
    """, (family, seed_id, seed_id)).fetchall()
    all_ids = {seed_id}
    for row in direct:
        all_ids.update((str(row[0]), str(row[1])))
    if all_ids:
        marks = ",".join("?" * len(all_ids))
        names = {
            str(row[0]): str(row[1])
            for row in conn.execute(
                f"SELECT symbol_id, qualified_name FROM family_members WHERE family_id=? AND symbol_id IN ({marks})",
                (family, *sorted(all_ids)),
            )
        }
    else:
        names = {}
    bridge_rows = [row for row in direct if str(row[2]) == "contains" and str(row[0]) == seed_id]
    bridge_rows.sort(key=lambda row: (
        -_role_score(names.get(str(row[1]), ""), roles, probe), str(row[4]), int(row[5]),
        names.get(str(row[1]), ""), str(row[1]),
    ))
    bridge_rows = bridge_rows[:owner_bridge_k]
    direct_rows = [row for row in direct if row not in bridge_rows and str(row[2]) != "contains"]

    def edge_sort(row: tuple) -> tuple:
        relation = str(row[2])
        return (
            -_role_score(names.get(str(row[1]), ""), roles, probe),
            {"call": 0, "implements": 1, "type_ref": 2, "contains": 3}[relation],
            str(row[4]), int(row[5]), names.get(str(row[1]), ""), str(row[1]),
        )

    output: list[dict[str, Any]] = []
    continuations_used = 0
    for row in sorted(direct_rows, key=edge_sort):
        output.append(_edge_item(
            row, names, seed_id, bridge=False, source_sites=source_sites, site_cache=site_cache))
        # A direct executable seed may need one more compiler-verified call to
        # express the outcome it delegates to.  Keep each continuation to one
        # edge and permit only two direct alternatives: this is a bounded path
        # witness for a comparison, not a resumed graph traversal.
        if (continuations_used < 2 and str(row[0]) == seed_id and str(row[2]) == "call"):
            continuations_used += 1
            callee_id = str(row[1])
            continuation_rows = conn.execute("""
                SELECT caller_id, callee_id, relation, occurrences, site_file, site_line
                FROM family_edges
                WHERE family_id=? AND caller_id=? AND relation != 'contains'
            """, (family, callee_id)).fetchall()
            continuation_ids = {callee_id} | {str(next_row[1]) for next_row in continuation_rows}
            missing = continuation_ids - names.keys()
            if missing:
                marks = ",".join("?" * len(missing))
                names.update({
                    str(next_row[0]): str(next_row[1])
                    for next_row in conn.execute(
                        f"SELECT symbol_id, qualified_name FROM family_members "
                        f"WHERE family_id=? AND symbol_id IN ({marks})",
                        (family, *sorted(missing)),
                    )
                })
            if continuation_rows:
                best = sorted(continuation_rows, key=edge_sort)[0]
                output.append(_edge_item(
                    best, names, seed_id, bridge=False, source_sites=source_sites, site_cache=site_cache))
    # A bridge is emitted next to the bounded direct calls of that member, so
    # the result stays a readable path rather than a bag of owner descendants.
    for bridge in bridge_rows:
        output.append(_edge_item(
            bridge, names, seed_id, bridge=True, source_sites=source_sites, site_cache=site_cache))
        member_id = str(bridge[1])
        member_rows = conn.execute("""
            SELECT caller_id, callee_id, relation, occurrences, site_file, site_line
            FROM family_edges
            WHERE family_id=? AND caller_id=? AND relation != 'contains'
        """, (family, member_id)).fetchall()
        member_ids = {member_id}
        for row in member_rows:
            member_ids.add(str(row[1]))
        missing = member_ids - names.keys()
        if missing:
            marks = ",".join("?" * len(missing))
            names.update({
                str(row[0]): str(row[1])
                for row in conn.execute(
                    f"SELECT symbol_id, qualified_name FROM family_members WHERE family_id=? AND symbol_id IN ({marks})",
                    (family, *sorted(missing)),
                )
            })
        output.extend(
            _edge_item(row, names, seed_id, bridge=True, source_sites=source_sites, site_cache=site_cache)
            for row in sorted(member_rows, key=edge_sort))
    deduped: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for edge in output:
        key = edge["caller_id"], edge["callee_id"], edge["relation"], edge["file"], edge["line"]
        deduped.setdefault(key, edge)
    return list(deduped.values())[:edge_k]


def _query(conn: sqlite3.Connection, question: str, probe, *, family_k: int,
           edge_k: int, facts: sqlite3.Connection | None, fact_k: int,
           owner_bridge_k: int, source_sites: sqlite3.Connection | None) -> dict[str, Any]:
    frame = _question_frame(question, probe)
    member_hits = _member_hits(conn, frame, limit=max(family_k * 32, 96))
    source_facts = _source_fact_hits(facts, frame, probe, fact_k=fact_k) if facts else []
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    def add(candidate: dict[str, Any], origin: str) -> None:
        family = candidate["family_id"]
        key = family, candidate["symbol_id"]
        item = candidates.setdefault(key, {
            "family_id": family, "symbol_id": candidate["symbol_id"],
            "qualified_name": candidate["qualified_name"], "file": candidate["file"],
            "line_start": candidate["line_start"], "line_end": candidate["line_end"],
            "matches": [], "owner_support": 0,
        })
        item["owner_support"] = max(item["owner_support"], int(candidate.get("owner_support", 0)))
        item["matches"].append({
            "origin": origin, "query_index": candidate["query_index"],
            "query_label": candidate["query_label"], "query_position": candidate["query_position"],
            "rank": candidate["rank"],
        })

    for hit in member_hits:
        add(hit, "structural-member")
    for hit in _owner_promotions(conn, member_hits, frame, probe):
        add(hit, "structural-owner")
    for fact in source_facts:
        for (family,) in conn.execute(
                "SELECT family_id FROM family_members WHERE symbol_id=? ORDER BY family_id", (fact["symbol_id"],)):
            add({**fact, "family_id": str(family)}, "source-fact")

    for item in candidates.values():
        item["matches"].sort(key=lambda match: (
            match["query_index"], match["rank"], match["origin"]))
        item["role_score"] = _role_score(item["qualified_name"], frame["roles"], probe)
        item["role_matched_source_fact"] = any(
            match["origin"] == "source-fact" for match in item["matches"]
        ) and item["role_score"] > 0
        item["family_member_count"] = int(conn.execute(
            "SELECT member_count FROM families WHERE family_id=?", (item["family_id"],)
        ).fetchone()[0])

        has_source_fact = any(match["origin"] == "source-fact" for match in item["matches"])
        if item["role_matched_source_fact"]:
            item["selection_kind"] = 0
        elif item["owner_support"] and item["role_score"]:
            item["selection_kind"] = 1
        elif has_source_fact:
            item["selection_kind"] = 2
        else:
            item["selection_kind"] = 3
    ordered = sorted(candidates.values(), key=lambda item: (
        item["matches"][0]["query_index"], item["selection_kind"],
        item["matches"][0]["rank"] if item["selection_kind"] == 0 else 0.0,
        -item["owner_support"], -item["role_score"],
        item["matches"][0]["rank"], item["qualified_name"], item["symbol_id"],
    ))
    # Retain a first card for every explicit component query before giving a
    # broad concern another card.  This protects comparison questions without
    # ever widening to a whole Leiden community.
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in ordered:
        for label in dict.fromkeys(match["query_label"] for match in item["matches"]):
            if label.startswith("component-") and not label.endswith("-name"):
                by_label[label].append(item)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for label in sorted(by_label):
        for item in by_label[label]:
            key = item["family_id"], item["symbol_id"]
            if key not in seen:
                selected.append(item)
                seen.add(key)
                break
    for item in ordered:
        key = item["family_id"], item["symbol_id"]
        if len(selected) >= family_k:
            break
        if key not in seen:
            selected.append(item)
            seen.add(key)
    strands = []
    site_cache: dict[tuple[str, str, str], tuple[str, int]] = {}
    for item in selected[:family_k]:
        strands.append({
            **item,
            "strand_id": f'{item["family_id"]}:{item["symbol_id"]}',
            "directed_edges": _strand_edges(
                conn, item["family_id"], item["symbol_id"], frame["roles"], probe,
                edge_k=edge_k, owner_bridge_k=owner_bridge_k,
                source_sites=source_sites, site_cache=site_cache),
        })
    return {
        "question": question, "frame": frame, "source_fact_count": len(source_facts),
        "structural_member_hit_count": len(member_hits), "candidate_strands": strands,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "ariadne.db")
    parser.add_argument("--families", type=Path, required=True,
                        help="external structural-family SQLite path")
    parser.add_argument("--facts", type=Path,
                        help="external symbol-facts SQLite path; no documents or embeddings")
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--question")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--only")
    parser.add_argument("--family-k", type=int, default=6)
    parser.add_argument("--edge-k", type=int, default=12)
    parser.add_argument("--fact-k", type=int, default=32)
    parser.add_argument("--owner-bridge-k", type=int, default=4,
                        help="maximum direct members exposed from one owner seed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (not args.db.is_file() or args.resolution <= 0 or args.min_size <= 0
            or args.family_k <= 0 or args.edge_k <= 0 or args.fact_k <= 0
            or args.owner_bridge_k <= 0):
        parser.error("invalid input path or non-positive numeric parameter")
    if bool(args.question) == bool(args.questions):
        parser.error("supply exactly one of --question or --questions")
    if args.only and not args.questions:
        parser.error("--only requires --questions")
    if args.build:
        metadata = _build(args.db, args.families, args.source, resolution=args.resolution,
                          seed=args.seed, min_size=args.min_size)
    else:
        if not args.families.is_file():
            parser.error("--families does not exist; pass --build to create it")
        metadata = {"reused": True}
    probe = _load_probe()
    conn = sqlite3.connect(f"file:{args.families}?mode=ro", uri=True)
    fact_conn = None
    source_site_conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        if args.facts:
            if not args.facts.is_file():
                parser.error("--facts does not exist")
            fact_conn = sqlite3.connect(f"file:{args.facts}?mode=ro", uri=True)
        if args.question:
            queries = [{"question": args.question}]
        else:
            queries = _question_rows(args.questions, _parse_ids(args.only))
        results = [
            {**query, **_query(conn, query["question"], probe,
                                family_k=args.family_k, edge_k=args.edge_k,
                                facts=fact_conn, fact_k=args.fact_k,
                                owner_bridge_k=args.owner_bridge_k,
                                source_sites=source_site_conn)}
            for query in queries
        ]
    finally:
        if fact_conn is not None:
            fact_conn.close()
        source_site_conn.close()
        conn.close()
    payload = {"schema": "scip-structural-leiden-family-query/v1",
               "build": metadata, "results": results}
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "families": str(args.families), "output": str(args.out),
        "queries": len(results),
        "candidate_strands": [len(item["candidate_strands"]) for item in results],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
