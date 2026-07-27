"""Hybrid graph construction for theme discovery (Themes plan, Phase 2).

Builds an undirected weighted graph by combining:
- Structural edges from doc_graph (imports / documents / topic_member /
  theme_member) — already populated by AST parser, ariadne_topic, etc.
- Semantic k-NN edges over per-element embeddings — this module's
  responsibility, persisted as edge_type='semantic_neighbor'.

Design notes:
- Semantic edges are stored canonicalized (lexicographically smaller id
  first) to halve storage; consumers treat them as undirected.
- Heavy deps (hnswlib, igraph) are imported inside functions so the module
  loads even when those packages aren't installed (for environments that
  only need the catalog/Phase 1 surface).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import igraph as ig

    from library import Library


# Edge types whose stored weight is taken at face value during graph composition.
_STRUCTURAL_EDGE_TYPES = ('imports', 'documents', 'topic_member', 'theme_member')


def _scoped_view(library: "Library", source: str | None, *, scope=None):
    """Build a ScopedLibrary appropriate for this graph operation.

    An explicit ``scope`` (a closure of source names) is used verbatim —
    for clustering over an arbitrary opted-in set, e.g. {project(s) + spool}.
    Otherwise, with ``source`` given, scope to that source's closure; without
    either, a global view (every configured source). Reads go through this
    wrapper; the chokepoint enforces the underlying ``library`` isn't read
    directly.
    """
    if scope is not None:
        from library import ScopedLibrary
        return ScopedLibrary(library, frozenset(scope))
    from config import get_config
    from scope_resolution import (
        make_global_scoped_library, make_scoped_library,
    )

    cfg = get_config()
    if source is None:
        return make_global_scoped_library(cfg, library)
    return make_scoped_library(cfg, library, source)


def _catalog_doc_ids(
    library: "Library", source: str | None = None, *, scope=None,
) -> list[str]:
    """Return doc_ids of catalog elements, scoped.

    An explicit ``scope`` closure admits exactly those sources (a
    ``ScopedLibrary`` on that set already filters to them, so no re-narrowing
    is needed) — used for a spool's cross-source pass over an opted-in set.
    Otherwise the legacy ``source`` filter narrows a single-source closure
    back to that source (the closure also brings in dependency docs, which
    semantic-edge code didn't want to mix).
    """
    scoped = _scoped_view(library, source, scope=scope)
    docs = scoped.list_documents_lite(content_type='catalog')
    if scope is None and source is not None:
        docs = [d for d in docs if d.source_name == source]
    return sorted(d.id for d in docs)


def _build_index(
    matrix: np.ndarray,
    *,
    ef_construction: int = 200,
    M: int = 16,
    ef_search: int = 64,
):
    """Build an HNSW index over `matrix` (each row is a unit-normalized embedding)."""
    import hnswlib

    n, dim = matrix.shape
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.set_ef(ef_search)
    index.add_items(matrix, ids=list(range(n)))
    return index


def _delete_semantic_for_ids(conn, ids: list[str]) -> None:
    """Delete semantic_neighbor edges where source or target is in `ids`."""
    if not ids:
        return
    placeholders = ','.join('?' * len(ids))
    conn.execute(
        f"DELETE FROM doc_graph WHERE edge_type = 'semantic_neighbor' "
        f"AND (source_id IN ({placeholders}) OR target_id IN ({placeholders}))",
        ids + ids,
    )


def _delete_semantic_within_ids(conn, ids: list[str]) -> None:
    """Delete semantic_neighbor edges where BOTH endpoints are in `ids`.

    Used for a scoped (cross-source spool) rebuild: it clears only the
    within-scope edges it is about to recompute, so an edge from an in-scope
    element to an out-of-scope one — e.g. a base project↔project edge — is
    left intact. (Contrast ``_delete_semantic_for_ids``, which clears any edge
    touching an id and is right for a single-source refresh.)
    """
    if not ids:
        return
    placeholders = ','.join('?' * len(ids))
    conn.execute(
        f"DELETE FROM doc_graph WHERE edge_type = 'semantic_neighbor' "
        f"AND source_id IN ({placeholders}) AND target_id IN ({placeholders})",
        ids + ids,
    )


def _clear_prior_semantic(conn, source, scope, ids: list[str]) -> None:
    """Clear the semantic edges a (re)build is about to replace.

    Global (no source, no scope): all of them. Scoped (a spool cross-source
    pass): only edges within the scope's ids, preserving cross-scope edges.
    Single-source: any edge touching one of its ids.
    """
    if source is None and scope is None:
        conn.execute("DELETE FROM doc_graph WHERE edge_type = 'semantic_neighbor'")
    elif scope is not None:
        _delete_semantic_within_ids(conn, ids)
    else:
        _delete_semantic_for_ids(conn, ids)


def build_semantic_edges(
    library: "Library",
    *,
    k: int = 5,
    min_sim: float = 0.6,
    source: str | None = None,
    scope=None,
) -> int:
    """Build/refresh all semantic_neighbor edges in doc_graph.

    Idempotent: clears prior semantic_neighbor edges (scoped to `source` if
    provided, else all) before inserting.

    Args:
        library: the Library to read embeddings from and write edges into.
        k: number of nearest neighbors per element.
        min_sim: cosine-similarity threshold; edges below this are dropped.
        source: optional source_name filter; only catalog elements from this
            source are indexed and the existing semantic edges among them are
            replaced. Edges outside this scope are untouched.
        scope: optional closure of source names (e.g. {project, spool}) to
            index together — builds cross-source edges among exactly those
            sources, replacing only within-scope edges (see
            ``_delete_semantic_within_ids``). Used by the spool cross-check.

    Returns:
        Number of edges inserted.
    """
    doc_ids = _catalog_doc_ids(library, source, scope=scope)
    if len(doc_ids) < 2:
        with library._conn_provider.acquire() as conn:
            _clear_prior_semantic(conn, source, scope, doc_ids)
        return 0

    scoped = _scoped_view(library, source, scope=scope)
    embeddings = scoped.get_embeddings_for_ids(doc_ids)
    valid_ids = [did for did in doc_ids if did in embeddings]
    if len(valid_ids) < 2:
        with library._conn_provider.acquire() as conn:
            _clear_prior_semantic(conn, source, scope, valid_ids)
        return 0

    matrix = np.stack([embeddings[did] for did in valid_ids])
    index = _build_index(matrix)
    query_k = min(k + 1, len(valid_ids))
    labels, distances = index.knn_query(matrix, k=query_k)

    # Canonicalize to (smaller_id, larger_id, max_sim).
    edge_weights: dict[tuple[str, str], float] = {}
    for i in range(len(valid_ids)):
        src_id = valid_ids[i]
        for j_idx in range(query_k):
            j = int(labels[i, j_idx])
            if j == i:
                continue
            sim = 1.0 - float(distances[i, j_idx])
            if sim < min_sim:
                continue
            tgt_id = valid_ids[j]
            a, b = (src_id, tgt_id) if src_id < tgt_id else (tgt_id, src_id)
            existing = edge_weights.get((a, b))
            if existing is None or sim > existing:
                edge_weights[(a, b)] = sim

    with library._conn_provider.acquire() as conn:
        _clear_prior_semantic(conn, source, scope, valid_ids)
        if edge_weights:
            conn.executemany(
                "INSERT OR REPLACE INTO doc_graph "
                "(source_id, target_id, edge_type, weight) "
                "VALUES (?, ?, 'semantic_neighbor', ?)",
                [(a, b, sim) for (a, b), sim in edge_weights.items()],
            )

    return len(edge_weights)


def update_semantic_edges_for(
    library: "Library",
    element_ids: list[str],
    *,
    k: int = 5,
    min_sim: float = 0.6,
) -> int:
    """Refresh semantic_neighbor edges incident to a specific subset of elements.

    Useful after a small subset re-embeds: existing edges that touch any of
    these elements are deleted, and new edges are computed by querying the
    full catalog index for the targets only.

    Returns:
        Number of edges (re)inserted that involve any of `element_ids`.
    """
    if not element_ids:
        return 0

    all_ids = _catalog_doc_ids(library)
    if len(all_ids) < 2:
        return 0

    # update_semantic_edges_for refreshes edges incident to a subset of
    # already-embedded elements. It operates across all sources by
    # design (the catalog index is global), so reads go through a
    # global-closure ScopedLibrary.
    scoped = _scoped_view(library, None)
    embeddings = scoped.get_embeddings_for_ids(all_ids)
    valid_ids = [did for did in all_ids if did in embeddings]
    if len(valid_ids) < 2:
        return 0

    target_set = [did for did in element_ids if did in embeddings]
    if not target_set:
        return 0

    id_to_idx = {did: i for i, did in enumerate(valid_ids)}
    matrix = np.stack([embeddings[did] for did in valid_ids])
    index = _build_index(matrix)
    query_k = min(k + 1, len(valid_ids))

    target_indices = [id_to_idx[did] for did in target_set]
    target_matrix = matrix[target_indices]
    labels, distances = index.knn_query(target_matrix, k=query_k)

    edge_weights: dict[tuple[str, str], float] = {}
    for ti, did in enumerate(target_set):
        src_idx = target_indices[ti]
        for j_idx in range(query_k):
            j = int(labels[ti, j_idx])
            if j == src_idx:
                continue
            sim = 1.0 - float(distances[ti, j_idx])
            if sim < min_sim:
                continue
            tgt_id = valid_ids[j]
            a, b = (did, tgt_id) if did < tgt_id else (tgt_id, did)
            existing = edge_weights.get((a, b))
            if existing is None or sim > existing:
                edge_weights[(a, b)] = sim

    with library._conn_provider.acquire() as conn:
        _delete_semantic_for_ids(conn, target_set)
        if edge_weights:
            conn.executemany(
                "INSERT OR REPLACE INTO doc_graph "
                "(source_id, target_id, edge_type, weight) "
                "VALUES (?, ?, 'semantic_neighbor', ?)",
                [(a, b, sim) for (a, b), sim in edge_weights.items()],
            )

    return len(edge_weights)


def load_hybrid_graph(
    library: "Library",
    *,
    semantic_edge_scale: float = 0.5,
    source: str | None = None,
    scope=None,
) -> tuple['ig.Graph', list[str]]:
    """Load the merged weighted graph from doc_graph for clustering.

    Combines structural edges (imports / documents / topic_member /
    theme_member) at face value with semantic_neighbor edges scaled by
    `semantic_edge_scale`. If a node pair is connected by multiple edge
    types, their composed weights are summed so downstream clustering sees
    one weighted edge per pair (no double-counting).

    Returns:
        (igraph.Graph, node_ids) where g.vs['name'] = node_ids.
    """
    import igraph as ig

    doc_ids = _catalog_doc_ids(library, source, scope=scope)
    id_to_index = {did: idx for idx, did in enumerate(doc_ids)}

    with library._conn_provider.acquire() as conn:
        rows = conn.execute(
            'SELECT source_id, target_id, edge_type, weight FROM doc_graph'
        ).fetchall()

    composed: dict[tuple[int, int], float] = {}
    for src_id, tgt_id, edge_type, weight in rows:
        if src_id not in id_to_index or tgt_id not in id_to_index:
            continue
        if edge_type == 'semantic_neighbor':
            w = float(weight) * semantic_edge_scale
        elif edge_type in _STRUCTURAL_EDGE_TYPES:
            w = float(weight)
        else:
            continue
        a, b = id_to_index[src_id], id_to_index[tgt_id]
        if a == b:
            continue
        u, v = (a, b) if a < b else (b, a)
        composed[(u, v)] = composed.get((u, v), 0.0) + w

    edge_list = list(composed.keys())
    weights = [composed[e] for e in edge_list]

    g = ig.Graph(n=len(doc_ids), edges=edge_list, directed=False)
    g.vs['name'] = doc_ids
    if weights:
        g.es['weight'] = weights
    return g, doc_ids


__all__ = [
    'build_semantic_edges',
    'load_hybrid_graph',
    'update_semantic_edges_for',
]
