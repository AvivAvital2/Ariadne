"""Tests for docgen.cluster (Themes plan, Phase 3).

Covers Leiden-driven theme discovery: cluster recovery, stability across runs,
tiny-cluster filtering, history recording, and persistence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _add_catalog(
    library: Library,
    doc_id: str,
    embedding: list[float],
    *,
    source_name: str = 'test',
) -> None:
    library.add_document(
        content_type='catalog',
        title=f'el {doc_id}',
        content='dummy',
        source_files=[],
        embedding=_unit(embedding),
        metadata={'kind': 'element', 'source_name': source_name},
        doc_id=doc_id,
    )
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE documents SET source_name = ? WHERE id = ?',
            (source_name, doc_id),
        )


def _populate_two_clusters(library: Library) -> None:
    """3 elements close to [1,0,...] (cluster A) and 3 to [0,0,1,...] (cluster B)."""
    for i, v in enumerate([
        [1.0, 0.05, 0, 0, 0, 0, 0, 0],
        [1.0, 0.04, 0, 0, 0, 0, 0, 0],
        [1.0, 0.06, 0, 0, 0, 0, 0, 0],
    ]):
        _add_catalog(library, f'A{i}', v)
    for i, v in enumerate([
        [0, 0, 1.0, 0.05, 0, 0, 0, 0],
        [0, 0, 1.0, 0.04, 0, 0, 0, 0],
        [0, 0, 1.0, 0.06, 0, 0, 0, 0],
    ]):
        _add_catalog(library, f'B{i}', v)


@pytest.fixture(autouse=True)
def _test_config(monkeypatch, tmp_path):
    """Configure the sources the fixtures use so the clustering chokepoint
    admits their docs. The contract under test is clustering recovery
    from embeddings; the source name is an environmental detail."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, ('test', 'src1', 'src2'))


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'cluster-test.db')
    yield lib
    lib.close()


# ---------------------------------------------------------------------------
# Recovery & stability
# ---------------------------------------------------------------------------


class TestClusterThemes:
    def test_recovers_two_communities(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        run = cluster_themes(library, min_cluster_size=3)

        assert len(run.clusters) == 2
        groups = {frozenset(members) for members in run.clusters.values()}
        assert frozenset({'A0', 'A1', 'A2'}) in groups
        assert frozenset({'B0', 'B1', 'B2'}) in groups

    def test_stable_cluster_ids_on_rerun(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        run1 = cluster_themes(library, min_cluster_size=3)
        run2 = cluster_themes(library, min_cluster_size=3)

        # The exact membership is the same → Jaccard match → cluster_ids preserved.
        assert set(run1.clusters.keys()) == set(run2.clusters.keys())

    def test_catalog_doc_ids_order_is_deterministic(self, library: Library) -> None:
        # The doc_id list from _catalog_doc_ids becomes the Leiden vertex order
        # (g.vs['name']). It must be deterministic — independent of updated_at
        # (which list_documents_lite sorts by, DESC, with no tiebreaker) — or
        # cluster IDs churn across incremental syncs that merely touch
        # updated_at, forcing needless (paid) re-summarization.
        from docgen.graph_builder import _catalog_doc_ids

        _add_catalog(library, 'aaa', [1.0, 0, 0, 0, 0, 0, 0, 0])
        _add_catalog(library, 'bbb', [0, 1.0, 0, 0, 0, 0, 0, 0])
        _add_catalog(library, 'ccc', [0, 0, 1.0, 0, 0, 0, 0, 0])
        # Force updated_at so the DESC ordering is the REVERSE of id-sorted.
        with library._conn_provider.acquire() as conn:
            for doc_id, ts in (
                ('aaa', '2020-01-01'),
                ('bbb', '2020-01-02'),
                ('ccc', '2020-01-03'),
            ):
                conn.execute(
                    'UPDATE documents SET updated_at = ? WHERE id = ?', (ts, doc_id),
                )
            conn.commit()

        assert _catalog_doc_ids(library) == ['aaa', 'bbb', 'ccc']

    def test_persists_themes_and_members(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        cluster_themes(library, min_cluster_size=3)

        themes = library.list_themes(coherent_only=False)
        assert len(themes) == 2

        member_sets = [
            frozenset(eid for eid, _ in library.get_theme_members(t.cluster_id))
            for t in themes
        ]
        assert frozenset({'A0', 'A1', 'A2'}) in member_sets
        assert frozenset({'B0', 'B1', 'B2'}) in member_sets

    def test_records_cluster_history(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        run = cluster_themes(library, min_cluster_size=3)

        assert library.latest_cluster_run() == run.run_id
        assert run.run_id == 1  # first run from a fresh library
        with library._conn_provider.acquire() as conn:
            history_count = conn.execute(
                'SELECT COUNT(*) FROM cluster_history WHERE run_id = ?',
                (run.run_id,),
            ).fetchone()[0]
        assert history_count == len(run.clusters)

    def test_run_ids_increment(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        r1 = cluster_themes(library, min_cluster_size=3)
        r2 = cluster_themes(library, min_cluster_size=3)
        assert r2.run_id == r1.run_id + 1

    def test_tiny_cluster_filtering_drops_below_threshold(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        # Set min_size high enough that both 3-element clusters are tiny and
        # have no non-tiny neighbor → all clusters drop.
        run = cluster_themes(library, min_cluster_size=10)
        assert len(run.clusters) == 0

    def test_empty_library_returns_empty_run(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        run = cluster_themes(library, min_cluster_size=3)
        assert run.clusters == {}
        assert run.new_cluster_ids == set()
        assert run.deleted_cluster_ids == set()


# ---------------------------------------------------------------------------
# Drift / merge / membership churn
# ---------------------------------------------------------------------------


class TestClusterDrift:
    def test_majority_overlap_preserves_cluster_id(self, library: Library) -> None:
        """Adding one element to a cluster should preserve its cluster_id (Jaccard ≥ 0.5)."""
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        run1 = cluster_themes(library, min_cluster_size=3)

        # Find which cluster_id contains "A0".
        a_cluster_id = next(
            cid for cid, members in run1.clusters.items() if 'A0' in members
        )

        # Add a 4th element to cluster A.
        _add_catalog(library, 'A3', [1.0, 0.05, 0, 0, 0, 0, 0, 0])
        build_semantic_edges(library, k=5, min_sim=0.6)
        run2 = cluster_themes(library, min_cluster_size=3)

        # The cluster containing A0 should still have the same cluster_id —
        # Jaccard({A0,A1,A2,A3}, {A0,A1,A2}) = 3/4 = 0.75 ≥ 0.5
        a_cluster_id_2 = next(
            cid for cid, members in run2.clusters.items() if 'A0' in members
        )
        assert a_cluster_id_2 == a_cluster_id


# ---------------------------------------------------------------------------
# ClusterMapping wiring
# ---------------------------------------------------------------------------


class TestClusterMerge:
    """Plan §5.3 acceptance: merge test — when two prior clusters become one,
    history must record the orphaned cluster (prev_cluster_id self-referential,
    overlap_ratio=0) per plan §4.3 sequence diagram.
    """

    def test_merge_records_orphaned_prior_in_history(
        self, library: Library, monkeypatch,
    ) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        # First run: two distinct clusters.
        run1 = cluster_themes(library, min_cluster_size=3)
        assert len(run1.clusters) == 2
        prior_cluster_ids = set(run1.clusters)

        # Force Leiden to return one big cluster spanning both groups.
        def force_one_cluster(g, *, resolution, seed):
            return [0] * g.vcount()
        monkeypatch.setattr('docgen.cluster._run_leiden', force_one_cluster)

        run2 = cluster_themes(library, min_cluster_size=3)

        # Exactly one surviving cluster, one deleted.
        assert len(run2.clusters) == 1
        assert len(run2.deleted_cluster_ids) == 1
        surviving_id = next(iter(run2.clusters))
        orphan_id = next(iter(run2.deleted_cluster_ids))

        # Both should be in run2's cluster_history records.
        with library._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT cluster_id, prev_cluster_id, overlap_ratio '
                'FROM cluster_history WHERE run_id = ? ORDER BY cluster_id',
                (run2.run_id,),
            ).fetchall()
        history = {r[0]: (r[1], r[2]) for r in rows}

        assert surviving_id in history, 'surviving cluster missing from history'
        assert orphan_id in history, (
            'orphaned cluster missing from history — plan §4.3 requires '
            'M->H: record (run_id, cluster_id, prev_cluster_id=cluster_id, overlap=0) '
            'for deleted clusters'
        )

        # Surviving cluster: prev = self, overlap > 0.
        prev, overlap = history[surviving_id]
        assert prev == surviving_id
        assert overlap is not None and overlap > 0.0

        # Orphaned cluster: prev = self (self-referential per plan), overlap = 0.
        prev, overlap = history[orphan_id]
        assert prev == orphan_id
        assert overlap == 0.0

    def test_merge_deletes_orphaned_theme_doc(
        self, library: Library, monkeypatch,
    ) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        run1 = cluster_themes(library, min_cluster_size=3)
        prior_doc_ids = {
            cid: library.get_theme(cid).doc_id  # type: ignore[union-attr]
            for cid in run1.clusters
        }

        def force_one_cluster(g, *, resolution, seed):
            return [0] * g.vcount()
        monkeypatch.setattr('docgen.cluster._run_leiden', force_one_cluster)
        run2 = cluster_themes(library, min_cluster_size=3)

        orphan_id = next(iter(run2.deleted_cluster_ids))
        # The orphan's theme doc should be gone (or its theme row at minimum).
        assert library.get_theme(orphan_id) is None
        # Doc deletion is a stronger guarantee per plan §4.3 / §5.3.
        assert library.get_document(prior_doc_ids[orphan_id]) is None


class TestClusterReassignmentAndScope:
    """Behavioral tests for parameters that aren't covered above."""

    def test_min_cluster_size_reassigns_singleton_to_strongest_neighbor(
        self, library: Library, monkeypatch,
    ) -> None:
        """A tiny cluster with a non-tiny neighbor should have its members
        reassigned, not just dropped. Plan §4.2 explicitly: 'Drop tiny
        clusters; members reassigned to next-best.'
        """
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        # Six A/B elements in two real clusters, plus one isolated 'tinyX'
        # connected by a strong import edge to A0.
        _populate_two_clusters(library)
        _add_catalog(library, 'tinyX', [0.99, 0.05, 0, 0, 0, 0, 0, 0])
        with library._conn_provider.acquire() as conn:
            conn.execute(
                "INSERT INTO doc_graph (source_id, target_id, edge_type, weight) "
                "VALUES (?, ?, 'imports', 5.0)",
                ('A0', 'tinyX'),
            )
        build_semantic_edges(library, k=5, min_sim=0.6)

        # Force Leiden to return three clusters: A's, B's, {tinyX}.
        def force_three(g, *, resolution, seed):
            membership = []
            for v in g.vs:
                name = v['name']
                if name.startswith('A'):
                    membership.append(0)
                elif name.startswith('B'):
                    membership.append(1)
                else:  # tinyX
                    membership.append(2)
            return membership
        monkeypatch.setattr('docgen.cluster._run_leiden', force_three)

        run = cluster_themes(library, min_cluster_size=3)

        # The {tinyX} cluster (size 1) should have been dropped, with tinyX
        # reassigned to A's cluster (its strongest neighbor).
        assert len(run.clusters) == 2
        a_members = next(
            members for members in run.clusters.values() if 'A0' in members
        )
        assert 'tinyX' in a_members, (
            "tinyX should have been reassigned to A's cluster (strongest neighbor); "
            "instead got clusters: " + str(run.clusters)
        )

    def test_source_scope_isolation(self, library: Library) -> None:
        """cluster_themes(source='X') must not see elements from other sources."""
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        # Two sources, each with three elements.
        for i, v in enumerate([
            [1.0, 0.05, 0, 0, 0, 0, 0, 0],
            [1.0, 0.04, 0, 0, 0, 0, 0, 0],
            [1.0, 0.06, 0, 0, 0, 0, 0, 0],
        ]):
            _add_catalog(library, f'src1_A{i}', v, source_name='src1')
        for i, v in enumerate([
            [0, 0, 1.0, 0.05, 0, 0, 0, 0],
            [0, 0, 1.0, 0.04, 0, 0, 0, 0],
            [0, 0, 1.0, 0.06, 0, 0, 0, 0],
        ]):
            _add_catalog(library, f'src2_B{i}', v, source_name='src2')

        build_semantic_edges(library, k=5, min_sim=0.6)
        run = cluster_themes(library, min_cluster_size=3, source='src1')

        for cluster_id, members in run.clusters.items():
            for mid in members:
                assert mid.startswith('src1_'), (
                    f'src2 element {mid!r} leaked into source-scoped clustering'
                )

    def test_resolution_parameter_is_passed_to_leiden(
        self, library: Library, monkeypatch,
    ) -> None:
        """The resolution kwarg must reach leidenalg, otherwise the config
        knob is dead. Verified by capturing the value at the boundary.
        """
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        captured: list[float] = []

        def capturing_leiden(g, *, resolution, seed):
            captured.append(resolution)
            return [0] * g.vcount()

        monkeypatch.setattr('docgen.cluster._run_leiden', capturing_leiden)
        cluster_themes(library, resolution=2.5, min_cluster_size=3)

        assert captured == [2.5]

    def test_seed_parameter_is_passed_to_leiden(
        self, library: Library, monkeypatch,
    ) -> None:
        """Same as above for the seed (drives determinism)."""
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        captured_seed: list[int] = []

        def capturing_leiden(g, *, resolution, seed):
            captured_seed.append(seed)
            return [0] * g.vcount()

        monkeypatch.setattr('docgen.cluster._run_leiden', capturing_leiden)
        cluster_themes(library, seed=12345, min_cluster_size=3)

        assert captured_seed == [12345]

    def test_membership_changes_populated_after_merge(
        self, library: Library, monkeypatch,
    ) -> None:
        """The ClusterRun.membership_changes field should record old→new
        cluster transitions for every element that moved.
        """
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        def force_one_cluster(g, *, resolution, seed):
            return [0] * g.vcount()
        monkeypatch.setattr('docgen.cluster._run_leiden', force_one_cluster)

        run = cluster_themes(library, min_cluster_size=3)

        # Half the elements were in the orphan cluster; they all moved to the
        # surviving cluster. So membership_changes is non-empty.
        assert len(run.membership_changes) >= 3, (
            f'expected membership_changes to record transitions; got {run.membership_changes}'
        )

    def test_new_cluster_ids_populated_on_first_run(self, library: Library) -> None:
        """First run from an empty themes table → every cluster is 'new'."""
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        run = cluster_themes(library, min_cluster_size=3)
        assert run.new_cluster_ids == set(run.clusters)
        assert run.deleted_cluster_ids == set()


class TestClusterMappingsWritten:
    def test_history_records_prev_cluster_id_for_inherited(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        cluster_themes(library, min_cluster_size=3)
        cluster_themes(library, min_cluster_size=3)

        # Second run's history should have prev_cluster_id=cluster_id (inherited).
        with library._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT cluster_id, prev_cluster_id, overlap_ratio '
                'FROM cluster_history WHERE run_id = 2'
            ).fetchall()
        assert len(rows) == 2
        for cid, prev, overlap in rows:
            assert prev == cid  # inherited the prior id
            assert overlap is not None
            assert overlap >= 0.5
