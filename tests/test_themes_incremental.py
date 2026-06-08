"""Tests for incremental theme update (Themes plan, Phase 5).

Covers local_reassign (cheap path) and incremental_theme_update (orchestrator)
that picks between cheap reassignment and full cluster_themes based on
drift ratio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library
from writer import LibraryWriter


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr
def _add_catalog(
    library: Library,
    doc_id: str,
    embedding: list[float],
    *,
    description: str | None = None,
    source_name: str = "test",
) -> None:
    metadata: dict[str, object] = {
        "kind": "element",
        "source_name": source_name,
        "qualified_name": doc_id,
        "subtype": "function",
    }
    if description is not None:
        metadata["description"] = description
    library.add_document(
        content_type="catalog",
        title=doc_id,
        content=f"function {doc_id}",
        source_files=[],
        embedding=_unit(embedding),
        metadata=metadata,
        doc_id=doc_id,
    )
    with library._conn_provider.acquire() as conn:
        conn.execute(
            "UPDATE documents SET source_name = ? WHERE id = ?",
            (source_name, doc_id),
        )


def _populate_two_clusters(library: Library) -> None:
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


def _add_graph_edge(library, src, tgt, edge_type, weight=1.0) -> None:
    with library._conn_provider.acquire() as conn:
        a, b = (src, tgt) if src < tgt else (tgt, src)
        conn.execute(
            'INSERT OR REPLACE INTO doc_graph (source_id, target_id, edge_type, weight) '
            'VALUES (?, ?, ?, ?)',
            (a, b, edge_type, weight),
        )


@pytest.fixture(autouse=True)
def _test_config(monkeypatch, tmp_path):
    """Configure the ``'test'`` source so the chokepoint admits these
    fixture docs. The contract under test is incremental theme refresh;
    source naming is environmental."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'test')


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'themes-incremental-test.db')
    yield lib
    lib.close()


@pytest.fixture
def mocked_embedding(monkeypatch):
    async def fake_embed(self, text):
        return np.zeros(3072, dtype=np.float32)

    async def fake_embed_batch(self, texts):
        return [np.zeros(3072, dtype=np.float32) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr('embedding.EmbeddingService.embed_batch', fake_embed_batch)
    monkeypatch.setattr('embedding.EmbeddingService._get_client', fake_get_client)
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)


def _coherent_response() -> str:
    return (
        '# Theme: Mocked\n\n'
        '## What this is\nA mocked theme.\n\n'
        '## Why this is a coherent theme\nBecause the test says so.\n\n'
        '## Key participants\n- **m1** — does m1 things\n\n'
        '## Cross-cutting concerns\nNone.\n\n'
        '## Caveats\nNone apparent.\n'
    )


@pytest.fixture
def mocked_chat_coherent(monkeypatch):
    async def fake_chat(messages, *, model=None, **kwargs):
        return _coherent_response()
    monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)


# ---------------------------------------------------------------------------
# local_reassign
# ---------------------------------------------------------------------------


class TestLocalReassign:
    def test_moves_element_to_dominant_neighbor_cluster(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges
        from docgen.themes import local_reassign

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        run = cluster_themes(library, min_cluster_size=3)

        # Find which cluster has B0 in it.
        b_cluster_id = next(
            cid for cid, members in run.clusters.items() if 'B0' in members
        )

        # Add a new element. Wire heavy structural edges to B-cluster members.
        _add_catalog(library, 'NEW', [0.5, 0.5, 0, 0, 0, 0, 0, 0])
        _add_graph_edge(library, 'NEW', 'B0', 'imports', weight=1.0)
        _add_graph_edge(library, 'NEW', 'B1', 'imports', weight=1.0)

        affected = local_reassign(library, {'NEW'})

        assert b_cluster_id in affected
        # NEW now appears in B's member list.
        members = library.get_theme_members(b_cluster_id)
        assert 'NEW' in {eid for eid, _ in members}

    def test_no_neighbors_no_reassignment(self, library: Library) -> None:
        from docgen.themes import local_reassign

        _add_catalog(library, 'isolated', [1.0, 0, 0, 0, 0, 0, 0, 0])
        affected = local_reassign(library, {'isolated'})

        assert affected == set()

    def test_element_already_in_best_cluster_is_a_noop(self, library: Library) -> None:
        """If the element's strongest neighbors are in the cluster it's already
        in (and only that cluster), local_reassign should make no change.
        """
        from docgen.themes import local_reassign

        # Two elements, both in cluster c1, connected.
        library.add_document(
            content_type='theme', title='t', content='x',
            source_files=[], metadata={}, doc_id='doc-c1',
        )
        _add_catalog(library, 'stable', [1.0, 0, 0, 0, 0, 0, 0, 0])
        _add_catalog(library, 'neighbor', [1.0, 0.01, 0, 0, 0, 0, 0, 0])
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=2, resolution=1.0, summary_hash='h',
        )
        library.set_theme_members('c1', [('stable', 1.0), ('neighbor', 1.0)])
        _add_graph_edge(library, 'stable', 'neighbor', 'imports', weight=2.0)

        affected = local_reassign(library, {'stable'})

        assert affected == set(), 'no change expected when element is already in best cluster'
        # Membership unchanged.
        assert {m[0] for m in library.get_theme_members('c1')} == {'stable', 'neighbor'}

    def test_element_in_multiple_clusters_cleaned_to_one(
        self, library: Library,
    ) -> None:
        """Pathological starting state: element in multiple clusters. After
        local_reassign, it should be in exactly one (the best) cluster.
        """
        from docgen.themes import local_reassign

        # Two themes c1 and c2; element 'el' in BOTH (anomaly).
        for tid in ('c1', 'c2'):
            library.add_document(
                content_type='theme', title=f't{tid}', content='x',
                source_files=[], metadata={}, doc_id=f'doc-{tid}',
            )
            library.add_theme(
                cluster_id=tid, doc_id=f'doc-{tid}',
                member_count=2, resolution=1.0, summary_hash='h',
            )
        _add_catalog(library, 'el', [1.0, 0, 0, 0, 0, 0, 0, 0])
        _add_catalog(library, 'other_c1', [1.0, 0.01, 0, 0, 0, 0, 0, 0])
        _add_catalog(library, 'other_c2', [1.0, 0.02, 0, 0, 0, 0, 0, 0])

        library.set_theme_members('c1', [('el', 1.0), ('other_c1', 1.0)])
        library.set_theme_members('c2', [('el', 1.0), ('other_c2', 1.0)])
        # Strong edge to c1 only — c1 should win.
        _add_graph_edge(library, 'el', 'other_c1', 'imports', weight=10.0)
        _add_graph_edge(library, 'el', 'other_c2', 'imports', weight=1.0)

        local_reassign(library, {'el'})

        clusters = library.get_themes_for_element('el')
        assert clusters == ['c1'], f'expected only c1, got {clusters}'

    def test_marks_affected_themes_dirty(self, library: Library) -> None:
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges
        from docgen.themes import local_reassign

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        # Mark all themes clean to verify dirty toggling.
        all_themes = library.list_themes(coherent_only=False)
        library.mark_themes_clean([t.cluster_id for t in all_themes])
        assert library.get_dirty_themes() == []

        # Add a new element that joins cluster B.
        _add_catalog(library, 'NEW', [0, 0, 1.0, 0, 0, 0, 0, 0])
        _add_graph_edge(library, 'NEW', 'B0', 'imports', weight=1.0)

        affected = local_reassign(library, {'NEW'})
        assert affected

        # Affected cluster is now dirty.
        assert set(library.get_dirty_themes()) == affected


# ---------------------------------------------------------------------------
# incremental_theme_update — path selection
# ---------------------------------------------------------------------------


class TestRefreshThemes:
    """Pull-based refresh — the function discovers what's stale itself.

    The contract is that callers don't need to compute or pass changed
    element ids. State is encoded by:
      - cluster_history.created_at (last cluster run time)
      - documents.updated_at (per-element)
      - themes.dirty (per-cluster)
    """

    @pytest.mark.asyncio
    async def test_disabled_short_circuits(
        self, library: Library, mocked_chat_coherent, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes

        # Anything monkeypatched here had better not be called.
        cluster_called = {'called': False}

        def tracking(library, **kwargs):
            cluster_called['called'] = True

        monkeypatch.setattr('docgen.themes.cluster_themes', tracking)

        async with LibraryWriter(library) as writer:
            summary = await themes.refresh_themes(library, writer, enabled=False)

        assert summary['path'] == 'disabled'
        assert summary['summarized'] == 0
        assert cluster_called['called'] is False

    @pytest.mark.asyncio
    async def test_no_catalog_returns_no_catalog_path(
        self, library: Library, mocked_chat_coherent, mocked_embedding,
    ) -> None:
        from docgen import themes

        async with LibraryWriter(library) as writer:
            summary = await themes.refresh_themes(library, writer)

        assert summary['path'] == 'no_catalog'

    @pytest.mark.asyncio
    async def test_first_run_does_initial_build(
        self, library: Library, mocked_chat_coherent, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        # No cluster_history yet, so this is an initial build.
        async with LibraryWriter(library) as writer:
            summary = await themes.refresh_themes(library, writer)

        assert summary['path'] == 'initial_build'
        assert summary['recluster_full'] is True
        # Two clusters of 3 should have been built and summarized.
        assert summary['summarized'] == 2

    @pytest.mark.asyncio
    async def test_idempotent_when_nothing_changed(
        self, library: Library, mocked_chat_coherent, mocked_embedding, monkeypatch,
    ) -> None:
        """Two consecutive refresh_themes calls — the second sees no catalog
        changes since the first's cluster run and does no LLM work.
        """
        from docgen import themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        chat_count = 0

        async def counting_chat(messages, *, model=None, **kwargs):
            nonlocal chat_count
            chat_count += 1
            return _coherent_response()

        monkeypatch.setattr('docgen.themes.chat_complete', counting_chat)

        async with LibraryWriter(library) as writer:
            await themes.refresh_themes(library, writer)
        first_count = chat_count

        async with LibraryWriter(library) as writer:
            summary = await themes.refresh_themes(library, writer)

        assert summary['path'] == 'noop'
        assert chat_count == first_count, 'second refresh should issue zero LLM calls'
