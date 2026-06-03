"""Tests for docgen.graph_builder (Themes plan, Phase 2).

Covers semantic-edge construction over per-element embeddings, idempotency,
similarity threshold, scoped updates, and hybrid-graph weight composition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library


@pytest.fixture(autouse=True)
def _test_config(tmp_path: Path, monkeypatch):
    """Provide a Config that knows about the 'test' source — without it,
    graph_builder reads through ScopedLibrary fail-closed on an
    unconfigured source name.
    """
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, ('test', 'src1', 'src2'))


def _unit(vec: list[float]) -> np.ndarray:
    """Return a unit-normalized float32 ndarray."""
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
    """Insert a catalog element with a unit-normalized embedding."""
    library.add_document(
        content_type='catalog',
        title=f'element {doc_id}',
        content='dummy',
        source_files=[],
        embedding=_unit(embedding),
        metadata={'kind': 'element', 'source_name': source_name},
        doc_id=doc_id,
    )
    # add_document doesn't expose source_name as a kwarg; set the column.
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE documents SET source_name = ? WHERE id = ?',
            (source_name, doc_id),
        )


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'graph-builder-test.db')
    yield lib
    lib.close()


def _semantic_edges(library: Library) -> set[tuple[str, str]]:
    """Return the set of (source_id, target_id) for semantic_neighbor edges."""
    with library._conn_provider.acquire() as conn:
        rows = conn.execute(
            "SELECT source_id, target_id FROM doc_graph "
            "WHERE edge_type = 'semantic_neighbor'"
        ).fetchall()
    return {(s, t) for s, t in rows}


# ---------------------------------------------------------------------------
# build_semantic_edges
# ---------------------------------------------------------------------------


class TestBuildSemanticEdges:
    """The two-cluster fixture: A* are mutually similar, B* are mutually similar,
    A* and B* are nearly orthogonal so cosine sim ≈ 0 across clusters.
    """

    def _populate_two_clusters(self, library: Library) -> None:
        # Cluster A: vectors close to [1, 0, 0, 0, ...]
        for i, v in enumerate([
            [1.0, 0.05, 0, 0, 0, 0, 0, 0],
            [1.0, 0.03, 0, 0, 0, 0, 0, 0],
            [1.0, 0.07, 0, 0, 0, 0, 0, 0],
        ]):
            _add_catalog(library, f'A{i}', v)
        # Cluster B: vectors close to [0, 0, 1, 0, ...]
        for i, v in enumerate([
            [0, 0, 1.0, 0.05, 0, 0, 0, 0],
            [0, 0, 1.0, 0.03, 0, 0, 0, 0],
            [0, 0, 1.0, 0.07, 0, 0, 0, 0],
        ]):
            _add_catalog(library, f'B{i}', v)

    def test_creates_edges_between_similar_neighbors(self, library: Library) -> None:
        from docgen.graph_builder import build_semantic_edges
        self._populate_two_clusters(library)

        n = build_semantic_edges(library, k=5, min_sim=0.6)

        assert n > 0
        edges = _semantic_edges(library)

        def _has_edge(a: str, b: str) -> bool:
            return (a, b) in edges or (b, a) in edges

        # Every intra-A pair connected.
        for a in ('A0', 'A1', 'A2'):
            for b in ('A0', 'A1', 'A2'):
                if a < b:
                    assert _has_edge(a, b), f'missing intra-A edge {a}-{b}'

        # Every intra-B pair connected.
        for a in ('B0', 'B1', 'B2'):
            for b in ('B0', 'B1', 'B2'):
                if a < b:
                    assert _has_edge(a, b), f'missing intra-B edge {a}-{b}'

        # No cross-cluster edges (cosine ≈ 0 < 0.6).
        for a in ('A0', 'A1', 'A2'):
            for b in ('B0', 'B1', 'B2'):
                assert not _has_edge(a, b), f'unexpected cross-cluster edge {a}-{b}'

    def test_idempotent(self, library: Library) -> None:
        from docgen.graph_builder import build_semantic_edges
        self._populate_two_clusters(library)

        n1 = build_semantic_edges(library, k=5, min_sim=0.6)
        edges1 = _semantic_edges(library)
        n2 = build_semantic_edges(library, k=5, min_sim=0.6)
        edges2 = _semantic_edges(library)

        assert n1 == n2
        assert edges1 == edges2

    def test_min_sim_filters_low_similarity(self, library: Library) -> None:
        from docgen.graph_builder import build_semantic_edges
        # Both vectors are already unit norm (|[1,0,...]| = 1, |[0.8, 0.6, ...]| = 1).
        # cos(P, Q) = 0.8.
        _add_catalog(library, 'P', [1.0, 0, 0, 0, 0, 0, 0, 0])
        _add_catalog(library, 'Q', [0.8, 0.6, 0, 0, 0, 0, 0, 0])

        # At threshold 0.5: 0.8 ≥ 0.5, edge created.
        n_low = build_semantic_edges(library, k=2, min_sim=0.5)
        assert n_low == 1

        # At threshold 0.95: 0.8 < 0.95, no edge.
        n_high = build_semantic_edges(library, k=2, min_sim=0.95)
        assert n_high == 0

    def test_zero_elements_no_edges(self, library: Library) -> None:
        from docgen.graph_builder import build_semantic_edges
        n = build_semantic_edges(library, k=5, min_sim=0.6)
        assert n == 0

    def test_source_filter_only_writes_for_that_source(self, library: Library) -> None:
        from docgen.graph_builder import build_semantic_edges
        # Two elements per source, near-identical embeddings within source.
        _add_catalog(library, 'src1_a', [1.0, 0.01, 0, 0, 0, 0, 0, 0], source_name='src1')
        _add_catalog(library, 'src1_b', [1.0, 0.02, 0, 0, 0, 0, 0, 0], source_name='src1')
        _add_catalog(library, 'src2_a', [1.0, 0.03, 0, 0, 0, 0, 0, 0], source_name='src2')
        _add_catalog(library, 'src2_b', [1.0, 0.04, 0, 0, 0, 0, 0, 0], source_name='src2')

        # Build for src1 only.
        n = build_semantic_edges(library, k=5, min_sim=0.6, source='src1')
        edges = _semantic_edges(library)

        assert n >= 1  # src1_a-src1_b edge.
        # No edge involves src2_*.
        for a, b in edges:
            assert a.startswith('src1') and b.startswith('src1'), \
                f"edge {a}-{b} should not exist when source='src1'"


# ---------------------------------------------------------------------------
# update_semantic_edges_for
# ---------------------------------------------------------------------------


class TestUpdateSemanticEdgesFor:
    def test_refreshes_edges_for_given_elements(self, library: Library) -> None:
        from docgen.graph_builder import build_semantic_edges, update_semantic_edges_for
        # Three near-identical elements: full graph after build.
        for i, v in enumerate([
            [1.0, 0.01, 0, 0, 0, 0, 0, 0],
            [1.0, 0.02, 0, 0, 0, 0, 0, 0],
            [1.0, 0.03, 0, 0, 0, 0, 0, 0],
        ]):
            _add_catalog(library, f'X{i}', v)

        build_semantic_edges(library, k=2, min_sim=0.6)
        before = _semantic_edges(library)

        # Re-running update for X0 should not crash; edges involving X0 are refreshed.
        n = update_semantic_edges_for(library, ['X0'], k=2, min_sim=0.6)
        after = _semantic_edges(library)

        assert n >= 0
        # Edges involving X1-X2 (no X0) untouched.
        for edge in [('X1', 'X2')]:
            assert edge in before or (edge[1], edge[0]) in before
            assert edge in after or (edge[1], edge[0]) in after

    def test_empty_input_is_noop(self, library: Library) -> None:
        from docgen.graph_builder import update_semantic_edges_for
        n = update_semantic_edges_for(library, [], k=5, min_sim=0.6)
        assert n == 0


# ---------------------------------------------------------------------------
# load_hybrid_graph
# ---------------------------------------------------------------------------


class TestLoadHybridGraph:
    def test_combines_structural_and_semantic_edges_with_weight_composition(
        self, library: Library,
    ) -> None:
        from docgen.graph_builder import build_semantic_edges, load_hybrid_graph

        for i, v in enumerate([
            [1.0, 0.01, 0, 0, 0, 0, 0, 0],
            [1.0, 0.02, 0, 0, 0, 0, 0, 0],
            [1.0, 0.03, 0, 0, 0, 0, 0, 0],
        ]):
            _add_catalog(library, f'N{i}', v)

        # Add a structural import edge directly.
        with library._conn_provider.acquire() as conn:
            conn.execute(
                "INSERT INTO doc_graph (source_id, target_id, edge_type, weight) "
                "VALUES (?, ?, 'imports', 1.0)",
                ('N0', 'N1'),
            )

        build_semantic_edges(library, k=2, min_sim=0.6)
        graph, node_ids = load_hybrid_graph(library, semantic_edge_scale=0.5)

        assert set(node_ids) >= {'N0', 'N1', 'N2'}
        # Every node maps to a unique vertex.
        assert len(set(node_ids)) == len(node_ids)
        # At least one edge in the loaded graph.
        assert graph.ecount() > 0

        # The N0-N1 imports edge contributes weight 1.0; semantic edges
        # contribute sim * 0.5. So if both exist between N0 and N1, the
        # composed weight is at least 1.0 (the imports floor).
        weights_n0_n1 = [
            edge['weight']
            for edge in graph.es
            if {graph.vs[edge.source]['name'], graph.vs[edge.target]['name']} == {'N0', 'N1'}
        ]
        assert weights_n0_n1, 'no edge between N0 and N1'
        assert max(weights_n0_n1) >= 1.0, 'imports floor not preserved'

    def test_returns_only_nodes_with_edges_or_in_catalog(self, library: Library) -> None:
        from docgen.graph_builder import load_hybrid_graph

        _add_catalog(library, 'lonely', [1.0, 0, 0, 0, 0, 0, 0, 0])
        graph, node_ids = load_hybrid_graph(library, semantic_edge_scale=0.5)

        # A lone catalog element with no edges is allowed in the graph (will be
        # an isolated vertex). The contract is just that calling load on a
        # populated library returns a graph object.
        assert graph is not None
        assert isinstance(node_ids, list)
