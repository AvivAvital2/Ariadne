"""Leiden community detection over the hybrid graph (Themes plan, Phase 3).

cluster_themes is the orchestrator: load hybrid graph -> run Leiden ->
filter tiny clusters -> map to stable cluster_ids via Jaccard -> persist
themes/theme_members/cluster_history.

A placeholder theme document is created for each new cluster so the
`themes.doc_id` FK is satisfied; Phase 4 (the summarizer) updates that
document's content with the LLM-generated theme markdown.

**Operates below the closure chokepoint.** Community detection runs
over the whole library's catalog elements; per-source clustering would
miss cross-source clusters by construction. Raw ``library.X(...)`` here
is intentional — see ``designs/directional-closure-scoping.md`` §
"Library-internal modules — legitimately unscoped".
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

from attrs import frozen

from docgen.graph_builder import _scoped_view, load_hybrid_graph
from library.themes import ClusterMapping
from schema import generate_deterministic_id

if TYPE_CHECKING:
    import igraph as ig

    from library import Library


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@frozen
class ClusterRun:
    """Summary of one cluster_themes invocation.

    clusters: stable cluster_id -> set of member doc_ids
    new_cluster_ids: cluster_ids with no Jaccard match to a prior run
    deleted_cluster_ids: prior cluster_ids absent from this run
    membership_changes: doc_id -> (old_cluster_id|None, new_cluster_id|None)
    """
    run_id: int
    clusters: dict[str, set[str]]
    new_cluster_ids: set[str]
    deleted_cluster_ids: set[str]
    membership_changes: dict[str, tuple[str | None, str | None]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_hash(members: set[str]) -> str:
    """Deterministic short id from a set of member ids; insensitive to order."""
    h = hashlib.blake2b(digest_size=12)
    for mid in sorted(members):
        h.update(mid.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()


def _association_key(scope) -> str:
    """Partition key for a clustering pass.

    The base global pass (scope=None) owns the '' partition, so existing
    themes — which migrate to association='' — keep reconciling against it
    unchanged. A scoped pass over an opted-in {project(s) + spool} set owns a
    partition keyed by its sorted source names, independent of every other
    pass.
    """
    if scope is None:
        return ''
    return '|'.join(sorted(scope))


def _mint_cluster_id(association: str, members: set[str]) -> str:
    """Fresh stable id for a cluster with no prior match in its partition.

    Namespaced by association so identical member sets in different passes
    (e.g. a pure-project cluster that recurs in both the base pass and a spool
    pass) don't collide on the global themes.cluster_id PK. The base pass
    (association='') keeps the bare hash, so existing ids are unchanged.
    """
    digest = _stable_hash(members)
    return f'{association}#{digest}' if association else digest


def _cross_source_clusters(
    scoped, clusters: dict[int, set[str]],
) -> dict[int, set[str]]:
    """Keep only clusters that span a spool source AND a non-spool source.

    A scoped spool pass clusters the whole {project ∪ spool} graph; a
    pure-project cluster merely duplicates a base theme and a pure-spool
    cluster is spool-internal, so neither is a cross-source theme worth
    persisting. ``scoped`` is the pass's ScopedLibrary view.
    """
    from spools import is_spool_source
    spool_ids = {
        d.id for d in scoped.list_documents_lite(content_type='catalog')
        if is_spool_source(d.source_name)
    }
    return {
        cid: members for cid, members in clusters.items()
        if (members & spool_ids) and (members - spool_ids)
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _theme_doc_id(cluster_id: str) -> str:
    """Deterministic doc id for a theme placeholder, stable across runs."""
    return generate_deterministic_id('theme', cluster_id)


def _run_leiden(g: "ig.Graph", *, resolution: float, seed: int) -> list[int]:
    """Run Leiden RBConfigurationVertexPartition; return per-vertex cluster ints.

    If the graph has no edges (e.g. semantic edges weren't built before
    this call), every vertex becomes its own singleton cluster. We bail
    out explicitly rather than letting leidenalg raise a confusing
    ``KeyError: 'Attribute does not exist'`` when it tries to read the
    'weight' edge attribute that doesn't exist on an empty edge set.
    """
    if g.ecount() == 0:
        return list(range(g.vcount()))

    import leidenalg as la

    # n_iterations=10, NOT -1 (run-to-convergence). On the live ~95k-vertex
    # graph Leiden's modularity is converged by ~10-20 passes: @10 captures
    # ~90% of the (already <0.2%) total gain over @2, while -1 spends ~2x the
    # time chasing noise-level refinements the min_cluster_size filter discards.
    # On small graphs @10 converges in <=2 passes anyway, so it matches -1 there
    # — one fixed value, no size branch. seed pins determinism. See
    # benchmark_clustering.py / benchmark_leiden_scale.py for the measurements.
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=10,
    )
    return list(partition.membership)


def _filter_tiny_clusters(
    g: "ig.Graph",
    membership: list[int],
    min_size: int,
) -> list[int]:
    """Reassign tiny-cluster members to their highest-weight non-tiny neighbor cluster.

    Members whose tiny cluster has no non-tiny neighbor stay where they are; the
    caller drops the resulting still-tiny clusters from the persisted set.
    """
    if min_size <= 1:
        return list(membership)

    cluster_sizes = Counter(membership)
    tiny = {c for c, sz in cluster_sizes.items() if sz < min_size}
    if not tiny:
        return list(membership)

    has_weight = 'weight' in g.es.attributes() if g.ecount() else False
    new_membership = list(membership)
    for v_idx, c in enumerate(membership):
        if c not in tiny:
            continue
        cluster_scores: dict[int, float] = defaultdict(float)
        for n_idx in g.neighbors(v_idx):
            n_c = membership[n_idx]
            if n_c in tiny:
                continue
            edge_id = g.get_eid(v_idx, n_idx, error=False)
            if edge_id == -1:
                continue
            w = float(g.es[edge_id]['weight']) if has_weight else 1.0
            cluster_scores[n_c] += w
        if cluster_scores:
            new_membership[v_idx] = max(cluster_scores, key=cluster_scores.get)

    return new_membership


def _stabilize_cluster_ids(
    new_clusters: dict[int, set[str]],
    prior_clusters: dict[str, set[str]],
    threshold: float,
    association: str = '',
) -> dict[int, str]:
    """Greedy Jaccard match: new int id -> stable string id.

    Largest new clusters get first pick of prior matches; if no prior cluster
    overlaps with this new one above `threshold`, mint a fresh hash-based id.
    """
    mapping: dict[int, str] = {}
    used_prior: set[str] = set()
    sorted_news = sorted(new_clusters.items(), key=lambda kv: -len(kv[1]))

    for new_id, new_members in sorted_news:
        best_prior: str | None = None
        best_score = 0.0
        for prior_id, prior_members in prior_clusters.items():
            if prior_id in used_prior:
                continue
            score = _jaccard(new_members, prior_members)
            if score > best_score:
                best_prior = prior_id
                best_score = score
        if best_prior is not None and best_score >= threshold:
            mapping[new_id] = best_prior
            used_prior.add(best_prior)
        else:
            mapping[new_id] = _mint_cluster_id(association, new_members)
    return mapping


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def cluster_themes(
    library: "Library",
    *,
    resolution: float = 1.0,
    seed: int = 42,
    min_cluster_size: int = 3,
    stability_threshold: float = 0.5,
    source: str | None = None,
    scope=None,
    semantic_edge_scale: float = 0.5,
) -> ClusterRun:
    """Run Leiden over the hybrid graph and persist stable theme assignments.

    Each new cluster gets a placeholder theme document with content
    "(pending summarization)"; Phase 4's summarizer updates that document
    when it generates real content.
    """
    g, node_ids = load_hybrid_graph(
        library, semantic_edge_scale=semantic_edge_scale, source=source,
        scope=scope,
    )
    run_id = (library.latest_cluster_run() or 0) + 1

    if g.vcount() == 0:
        return ClusterRun(
            run_id=run_id, clusters={}, new_cluster_ids=set(),
            deleted_cluster_ids=set(), membership_changes={},
        )

    membership = _run_leiden(g, resolution=resolution, seed=seed)
    membership = _filter_tiny_clusters(g, membership, min_cluster_size)

    # Group members by cluster, dropping any still-tiny clusters that the
    # filter couldn't reassign (no non-tiny neighbor).
    new_clusters_int: dict[int, set[str]] = defaultdict(set)
    for v_idx, c in enumerate(membership):
        new_clusters_int[c].add(node_ids[v_idx])
    new_clusters_int = {
        c: members for c, members in new_clusters_int.items()
        if len(members) >= min_cluster_size
    }

    association = _association_key(scope)
    if scope is not None:
        # A scoped spool pass keeps only genuinely cross-source clusters
        # (spanning the spool AND a project); pure clusters are dropped
        # before stabilization so they never consume a prior id or persist.
        scoped_view = _scoped_view(library, source, scope=scope)
        new_clusters_int = _cross_source_clusters(scoped_view, new_clusters_int)
    prior_themes = library.list_themes(
        coherent_only=False, association=association,
    )
    prior_clusters: dict[str, set[str]] = {
        t.cluster_id: {eid for eid, _ in library.get_theme_members(t.cluster_id)}
        for t in prior_themes
    }
    prior_doc_ids: dict[str, str] = {t.cluster_id: t.doc_id for t in prior_themes}

    int_to_stable = _stabilize_cluster_ids(
        new_clusters_int, prior_clusters, stability_threshold, association,
    )
    new_clusters: dict[str, set[str]] = {
        int_to_stable[c]: members for c, members in new_clusters_int.items()
    }

    persisted_ids = set(new_clusters)
    new_cluster_ids = persisted_ids - set(prior_clusters)
    deleted_cluster_ids = set(prior_clusters) - persisted_ids

    # Diff old vs new memberships for the summary.
    prior_element_to_cluster: dict[str, str] = {
        eid: cid for cid, members in prior_clusters.items() for eid in members
    }
    new_element_to_cluster: dict[str, str] = {
        eid: cid for cid, members in new_clusters.items() for eid in members
    }
    membership_changes: dict[str, tuple[str | None, str | None]] = {}
    for eid in set(prior_element_to_cluster) | set(new_element_to_cluster):
        old = prior_element_to_cluster.get(eid)
        new = new_element_to_cluster.get(eid)
        if old != new:
            membership_changes[eid] = (old, new)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------

    for cluster_id, members in new_clusters.items():
        is_new = cluster_id in new_cluster_ids
        if is_new:
            doc_id = _theme_doc_id(cluster_id)
            if library.get_document(doc_id) is None:
                library.add_document(
                    content_type='theme',
                    title=f'Theme {cluster_id[:12]}',
                    content='(pending summarization)',
                    source_files=[],
                    metadata={'cluster_id': cluster_id, 'pending': True},
                    doc_id=doc_id,
                )
            library.add_theme(
                cluster_id=cluster_id,
                doc_id=doc_id,
                member_count=len(members),
                resolution=resolution,
                summary_hash='',  # Phase 4 fills this when it summarizes.
                coherent=True,
                dirty=True,
                association=association,
            )
        else:
            # Existing theme: refresh member count + last_built_at; mark dirty
            # iff membership changed (so Phase 4 knows to re-summarize).
            members_changed = members != prior_clusters.get(cluster_id)
            from schema import _now_iso
            with library._conn_provider.acquire() as conn:
                conn.execute(
                    'UPDATE themes SET member_count = ?, last_built_at = ? '
                    'WHERE cluster_id = ?',
                    (len(members), _now_iso(), cluster_id),
                )
            if members_changed:
                library.mark_theme_dirty(cluster_id)
        library.set_theme_members(
            cluster_id, [(eid, 1.0) for eid in sorted(members)],
        )

    # Delete orphaned themes (cascades to theme_members + the theme doc).
    for cid in deleted_cluster_ids:
        doc_id = prior_doc_ids.get(cid)
        if doc_id is not None:
            # delete_document cascades to themes (themes.doc_id FK ON DELETE CASCADE),
            # which cascades to theme_members.
            library.delete_document(doc_id)
        else:
            library.delete_theme(cid)

    # Cluster history. Per plan §4.3, every cluster touched this run gets a
    # row: surviving clusters with prev=self and overlap>0; brand-new clusters
    # with prev=None and overlap=None; orphaned (deleted) priors with prev=self
    # and overlap=0 so the deletion is auditable.
    mappings: list[ClusterMapping] = []
    for new_int, stable_id in int_to_stable.items():
        if stable_id in prior_clusters:
            overlap = _jaccard(new_clusters_int[new_int], prior_clusters[stable_id])
            mappings.append(ClusterMapping(
                cluster_id=stable_id,
                prev_cluster_id=stable_id,
                overlap_ratio=overlap,
            ))
        else:
            mappings.append(ClusterMapping(
                cluster_id=stable_id,
                prev_cluster_id=None,
                overlap_ratio=None,
            ))
    for orphan_id in deleted_cluster_ids:
        mappings.append(ClusterMapping(
            cluster_id=orphan_id,
            prev_cluster_id=orphan_id,
            overlap_ratio=0.0,
        ))
    library.record_cluster_history(run_id=run_id, mappings=mappings)

    return ClusterRun(
        run_id=run_id,
        clusters=new_clusters,
        new_cluster_ids=new_cluster_ids,
        deleted_cluster_ids=deleted_cluster_ids,
        membership_changes=membership_changes,
    )


__all__ = [
    'ClusterRun',
    'cluster_themes',
]
