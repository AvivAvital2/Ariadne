"""Tests for docgen.semantic_index — the persisted HNSW edge-kNN index.

Slice 1: build the index over embedded catalog elements, persist it (save/load),
a freshness stamp that tracks the embedded-catalog set, and neighbor queries
that map hnswlib's integer labels back to doc_ids.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from docgen.semantic_index import (
    INDEX_NAME,
    SemanticIndex,
    build_semantic_index,
    ensure_semantic_index,
)
from library import Library


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _add_catalog(library: Library, doc_id: str, embedding: list[float]) -> None:
    library.add_document(
        content_type='catalog', title=f'el {doc_id}', content='dummy',
        source_files=[], embedding=_unit(embedding),
        metadata={'kind': 'element'}, doc_id=doc_id,
    )


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'semidx-test.db')
    yield lib
    lib.close()


def _two_clusters(library: Library) -> None:
    for i, v in enumerate([
        [1.0, 0.05, 0, 0, 0, 0, 0, 0],
        [1.0, 0.03, 0, 0, 0, 0, 0, 0],
        [1.0, 0.07, 0, 0, 0, 0, 0, 0],
    ]):
        _add_catalog(library, f'A{i}', v)
    for i, v in enumerate([
        [0, 0, 1.0, 0.05, 0, 0, 0, 0],
        [0, 0, 1.0, 0.03, 0, 0, 0, 0],
        [0, 0, 1.0, 0.07, 0, 0, 0, 0],
    ]):
        _add_catalog(library, f'B{i}', v)


class TestSemanticIndexPersistence:
    def test_build_reload_query_maps_labels_to_doc_ids(
        self, library: Library, tmp_path: Path,
    ) -> None:
        """Build → persist → reload → query returns the right *doc_ids* (not
        hnswlib's integer labels), and the index survives the disk round-trip."""
        _two_clusters(library)
        out = tmp_path / '.ariadne'
        build_semantic_index(library, out)

        si = SemanticIndex.load(out)
        assert si is not None
        neighbors = si.query_neighbors(_unit([1.0, 0.04, 0, 0, 0, 0, 0, 0]), k=3)
        ids = [doc_id for doc_id, _sim in neighbors]
        # A cluster-A query returns cluster-A neighbors (and they are doc_ids).
        assert set(ids) <= {'A0', 'A1', 'A2'}, ids
        assert all(isinstance(sim, float) for _id, sim in neighbors)

    def test_freshness_stamp_detects_catalog_change(
        self, library: Library, tmp_path: Path,
    ) -> None:
        """The persisted index is fresh until the embedded-catalog set changes,
        then stale — so a reload knows to rebuild."""
        _two_clusters(library)
        out = tmp_path / '.ariadne'
        build_semantic_index(library, out)
        si = SemanticIndex.load(out)
        with library._conn_provider.acquire() as conn:
            assert si.is_fresh(conn)
        _add_catalog(library, 'C0', [0, 1.0, 0, 0, 0, 0, 0, 0])  # corpus changed
        with library._conn_provider.acquire() as conn:
            assert not si.is_fresh(conn)

    def test_ensure_builds_then_reuses_when_fresh(
        self, library: Library, tmp_path: Path, monkeypatch,
    ) -> None:
        """ensure_semantic_index builds on first call, then reuses the persisted
        index when fresh — it must NOT rebuild (the whole point of persistence)."""
        _two_clusters(library)
        out = tmp_path / '.ariadne'
        first = ensure_semantic_index(library, out)
        assert first is not None and first.index is not None

        import docgen.semantic_index as M

        def _must_not_build(*_a, **_k):
            raise AssertionError('ensure must not rebuild when the index is fresh')
        monkeypatch.setattr(M, 'build_semantic_index', _must_not_build)
        reused = ensure_semantic_index(library, out)
        assert reused is not None and reused.index is not None

    def test_empty_catalog_yields_empty_index(
        self, library: Library, tmp_path: Path,
    ) -> None:
        """No embedded catalog → a valid (empty) index that queries to []."""
        out = tmp_path / '.ariadne'
        build_semantic_index(library, out)
        si = SemanticIndex.load(out)
        assert si is not None
        assert si.query_neighbors(_unit([1.0, 0, 0, 0, 0, 0, 0, 0]), k=3) == []

    def test_load_returns_none_when_absent(self, tmp_path: Path) -> None:
        assert SemanticIndex.load(tmp_path / 'nope') is None

    def test_load_degrades_when_index_file_missing(
        self, library: Library, tmp_path: Path,
    ) -> None:
        """Meta present but the .bin gone (partial cache) → empty index, not a
        crash; queries return []."""
        _two_clusters(library)
        out = tmp_path / '.ariadne'
        build_semantic_index(library, out)
        (out / INDEX_NAME).unlink()
        si = SemanticIndex.load(out)
        assert si is not None and si.index is None
        assert si.query_neighbors(_unit([1.0, 0, 0, 0, 0, 0, 0, 0]), k=3) == []

    def test_ensure_defaults_to_ariadne_dir_next_to_db(
        self, library: Library,
    ) -> None:
        """Called without an out_dir, ensure builds into ``.ariadne/`` beside
        the database file (shared with the embedding matrix)."""
        _two_clusters(library)
        si = ensure_semantic_index(library)  # no out_dir
        expected = Path(library._conn_provider.path).parent / '.ariadne'
        assert (expected / INDEX_NAME).exists()
        assert si is not None and si.index is not None

    def test_ensure_rebuilds_when_stale(
        self, library: Library, tmp_path: Path,
    ) -> None:
        """When the corpus changes, ensure rebuilds the persisted index so the
        new element is indexed and the stamp is current again."""
        _two_clusters(library)
        out = tmp_path / '.ariadne'
        ensure_semantic_index(library, out)
        _add_catalog(library, 'C0', [0, 1.0, 0, 0, 0, 0, 0, 0])  # corpus changed

        si = ensure_semantic_index(library, out)
        with library._conn_provider.acquire() as conn:
            assert si.is_fresh(conn)  # rebuilt to the current corpus
        nbrs = si.query_neighbors(_unit([0, 1.0, 0, 0, 0, 0, 0, 0]), k=1)
        assert nbrs and nbrs[0][0] == 'C0'  # the new element is now indexed
