"""Tier 2 — embedding ranking strategies selected by candidate-set size.

Large sets rank against the mmap matrix; small sets load candidates from SQLite.
See designs/embedding-matrix-tier2-serve.md. Built via the evolving-TDD loop,
then split into focused tests.

Fixtures are synthetic only: source ``src1``, docs ``d1, d2, …``, tiny
``dim = 4`` hand-chosen vectors.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library
from library.embedding_matrix import EmbeddingMatrix, build_doc_embedding_matrix
from library.embedding_ranking import (
    MATRIX_MIN_CANDIDATES,
    MatrixRanker,
    SqliteRanker,
    select_ranker,
)

QUERY = np.array([0.1, 1.0, 0.0, 0.0], dtype=np.float32)  # d2 > d1 > d3


@pytest.fixture
def lib(tmp_path: Path) -> Library:
    library = Library(tmp_path / 'lib.db')
    yield library
    library.close()


def _add(library: Library, doc_id: str, vec: list[float]) -> None:
    library.add_document(
        content_type='explanation',
        title=f'title-{doc_id}',
        content=f'content-{doc_id}',
        embedding=np.array(vec, dtype=np.float32),
        doc_id=doc_id,
        source_name='src1',
    )


def test_select_small_set_uses_sqlite(lib: Library) -> None:
    assert isinstance(select_ranker(5, lambda: object(), lib, threshold=1000), SqliteRanker)


def test_select_large_set_uses_matrix(lib: Library) -> None:
    assert isinstance(select_ranker(2000, lambda: object(), lib, threshold=1000), MatrixRanker)


def test_select_large_set_without_matrix_uses_sqlite(lib: Library) -> None:
    # Large set but no fresh matrix available → SQLite.
    assert isinstance(select_ranker(2000, lambda: None, lib, threshold=1000), SqliteRanker)


def test_select_default_threshold_read_at_call_time(lib: Library) -> None:
    # threshold=None → reads MATRIX_MIN_CANDIDATES; boundary is inclusive.
    assert isinstance(select_ranker(MATRIX_MIN_CANDIDATES, lambda: object(), lib), MatrixRanker)
    assert isinstance(select_ranker(MATRIX_MIN_CANDIDATES - 1, lambda: object(), lib), SqliteRanker)


def test_sqlite_ranker_ranks_and_handles_empty(lib: Library) -> None:
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    _add(lib, 'd2', [0.0, 1.0, 0.0, 0.0])
    assert [i for i, _ in SqliteRanker(lib).rank(QUERY, ['d1', 'd2'], 2)] == ['d2', 'd1']
    assert SqliteRanker(lib).rank(QUERY, ['nonexistent'], 2) == []


def test_matrix_ranker_delegates(lib: Library, tmp_path: Path) -> None:
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    _add(lib, 'd2', [0.0, 1.0, 0.0, 0.0])
    build_doc_embedding_matrix(lib, tmp_path)
    matrix_ranker = MatrixRanker(EmbeddingMatrix.load(tmp_path))
    assert [i for i, _ in matrix_ranker.rank(QUERY, ['d1', 'd2'], 2)] == ['d2', 'd1']
