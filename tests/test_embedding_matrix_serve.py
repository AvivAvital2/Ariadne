"""Tier 2 (SERVE) — EmbeddingMatrix (mmap load + rank) and its integration into
SearchMixin._rank_ids_by_embedding.

See designs/embedding-matrix-tier2-serve.md. Built via the evolving-TDD loop
(one growing test per sub-feature), then split into focused tests for failure
localization.

Fixtures are synthetic only: source ``src1``, docs ``d1, d2, …``, tiny
``dim = 4`` hand-chosen vectors. The integration tests detect *which* ranking
path ran by making the on-disk matrix disagree with the DB (swap embeddings
without bumping ``updated_at``, so the stamp — hence freshness — is unchanged).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from library import Library
from library.embedding_matrix import EmbeddingMatrix, build_doc_embedding_matrix
from search import batch_dot_similarity, top_k_indices

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


def _set_embedding(library: Library, doc_id: str, vec: list[float]) -> None:
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE documents SET embedding = ? WHERE id = ?',
            (np.array(vec, dtype=np.float32).tobytes(), doc_id),
        )


def _bump_updated_at(library: Library, doc_id: str) -> None:
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE documents SET updated_at = ? WHERE id = ?',
            ('2099-01-01T00:00:00', doc_id),
        )


def _matrix_of_three(library: Library, out_dir: Path) -> EmbeddingMatrix:
    _add(library, 'd1', [1.0, 0.0, 0.0, 0.0])
    _add(library, 'd2', [0.0, 1.0, 0.0, 0.0])
    _add(library, 'd3', [0.0, 0.0, 1.0, 0.0])
    build_doc_embedding_matrix(library, out_dir)
    matrix = EmbeddingMatrix.load(out_dir)
    assert matrix is not None
    return matrix


# --- EmbeddingMatrix -------------------------------------------------------

def test_load_and_rank(lib: Library, tmp_path: Path) -> None:
    matrix = _matrix_of_three(lib, tmp_path)
    ranked = matrix.rank(QUERY, ['d1', 'd2', 'd3'], 2)
    assert [doc_id for doc_id, _ in ranked] == ['d2', 'd1']


def test_load_absent_returns_none(tmp_path: Path) -> None:
    assert EmbeddingMatrix.load(tmp_path / 'nonexistent') is None


def test_is_fresh_detects_staleness(lib: Library, tmp_path: Path) -> None:
    matrix = _matrix_of_three(lib, tmp_path)
    with lib._conn_provider.acquire() as conn:
        assert matrix.is_fresh(conn) is True
    _bump_updated_at(lib, 'd1')
    with lib._conn_provider.acquire() as conn:
        assert matrix.is_fresh(conn) is False


def test_rank_parity_with_sqlite(lib: Library, tmp_path: Path) -> None:
    matrix = _matrix_of_three(lib, tmp_path)
    embeddings_map = lib.get_embeddings_for_ids(['d1', 'd2', 'd3'])
    ordered = [d for d in ['d1', 'd2', 'd3'] if d in embeddings_map]
    sims = batch_dot_similarity(QUERY, np.stack([embeddings_map[d] for d in ordered]))
    expected = [(ordered[i], float(sims[i])) for i in top_k_indices(sims, 3)]
    assert matrix.rank(QUERY, ['d1', 'd2', 'd3'], 3) == expected




def test_rank_skips_unknown_ids(lib: Library, tmp_path: Path) -> None:
    matrix = _matrix_of_three(lib, tmp_path)
    assert [d for d, _ in matrix.rank(QUERY, ['d1', 'unknown'], 5)] == ['d1']
    assert matrix.rank(QUERY, ['unknown'], 5) == []


def test_rank_scores_only_candidate_rows_not_whole_matrix(
    lib: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rank() must score only the candidate rows, never the full matrix.

    Scoring the whole matrix runs the similarity kernel over every row and
    pages in the entire (multi-GB, mmap'd) matrix on each query — synchronous
    work on the single event-loop thread that serialises concurrent readers.
    The candidate set is tiny relative to the corpus, so the kernel must see
    only the candidate rows. We assert it is handed exactly the 2 candidate
    rows, not the whole 3-row matrix.
    """
    import search
    matrix = _matrix_of_three(lib, tmp_path)  # 3-row matrix
    row_counts: list[int] = []
    real = search.batch_dot_similarity

    def _spy(query, embeddings):
        row_counts.append(int(embeddings.shape[0]))
        return real(query, embeddings)

    monkeypatch.setattr(search, 'batch_dot_similarity', _spy)
    ranked = matrix.rank(QUERY, ['d1', 'd3'], 2)  # 2 of the 3 rows are candidates
    assert [d for d, _ in ranked] == ['d1', 'd3']  # result unchanged
    assert row_counts == [2], (
        f'similarity kernel scored {row_counts} rows; expected [2] — the whole '
        'matrix is being scanned instead of just the candidate rows'
    )


# --- integration into _rank_ids_by_embedding -------------------------------

class _FakeEmbed:
    def __init__(self, vec: np.ndarray) -> None:
        self._vec = vec

    async def embed(self, query: str) -> np.ndarray:
        return self._vec


class _RaisingEmbed:
    async def embed(self, query: str) -> np.ndarray:
        raise RuntimeError('embed boom')


def _make_service(library: Library):
    from ariadne_mcp.service import AriadneService

    svc = AriadneService()
    svc._library = library
    return svc


def _service_with_disagreeing_matrix(library: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Matrix (on disk) sees d1=e1, d2=e2; the DB has them swapped (updated_at
    intact, so the matrix still reads as fresh). Query e1 then ranks d1 first via
    the matrix, d2 first via SQLite — revealing which path ran.

    Patches the candidate threshold to 1 so the matrix strategy is selected for
    these tiny (2-candidate) fixtures."""
    monkeypatch.setattr('library.embedding_ranking.MATRIX_MIN_CANDIDATES', 1)
    _add(library, 'd1', [1.0, 0.0, 0.0, 0.0])
    _add(library, 'd2', [0.0, 1.0, 0.0, 0.0])
    build_doc_embedding_matrix(library, tmp_path / '.ariadne')
    _set_embedding(library, 'd1', [0.0, 1.0, 0.0, 0.0])
    _set_embedding(library, 'd2', [1.0, 0.0, 0.0, 0.0])
    svc = _make_service(library)
    svc._embedding_service = _FakeEmbed(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    return svc


def test_integration_falls_back_when_stale(lib: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _service_with_disagreeing_matrix(lib, tmp_path, monkeypatch)
    _bump_updated_at(lib, 'd1')  # stamp changes → is_fresh False
    ranked = asyncio.run(svc._rank_ids_by_embedding(['d1', 'd2'], 'q', 2))
    assert ranked[0][0] == 'd2'  # SQLite path


def test_integration_falls_back_when_no_artifact(lib: Library, tmp_path: Path) -> None:
    _add(lib, 'd9', [1.0, 0.0, 0.0, 0.0])  # no matrix built
    svc = _make_service(lib)
    svc._embedding_service = _FakeEmbed(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    ranked = asyncio.run(svc._rank_ids_by_embedding(['d9'], 'q', 2))
    assert ranked[0][0] == 'd9'


def test_integration_falls_through_when_candidate_absent(lib: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _service_with_disagreeing_matrix(lib, tmp_path, monkeypatch)
    # fresh matrix, but the candidate isn't in it → rank() empty → SQLite (also empty here)
    ranked = asyncio.run(svc._rank_ids_by_embedding(['unknown'], 'zzz', 2))
    assert ranked == []


def test_integration_embedding_error_falls_back_to_text(lib: Library, tmp_path: Path) -> None:
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    svc = _make_service(lib)
    svc._embedding_service = _RaisingEmbed()
    ranked = asyncio.run(svc._rank_ids_by_embedding(['d1'], 'title-d1', 2))
    assert isinstance(ranked, list)
    assert ranked and ranked[0][0] == 'd1'  # text fallback matched the title


def test_integration_caches_matrix_handle(lib: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two ranks on one service: the second reuses the cached handle (the load is
    # not repeated), and the result is unchanged.
    svc = _service_with_disagreeing_matrix(lib, tmp_path, monkeypatch)
    first = asyncio.run(svc._rank_ids_by_embedding(['d1', 'd2'], 'q', 2))
    assert first[0][0] == 'd1'
    second = asyncio.run(svc._rank_ids_by_embedding(['d1', 'd2'], 'q', 2))
    assert second[0][0] == 'd1'


def test_integration_matrix_only_above_threshold(lib: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Disagreeing setup (matrix: d1=e1; DB: d1=e2). With threshold 2, a 1-candidate
    # query is ranked by SQLite (DB → d1 scores 0.0 for query e1) while a 2-candidate
    # query is ranked by the matrix (d1 scores 1.0) — proving the count-based choice.
    svc = _service_with_disagreeing_matrix(lib, tmp_path, monkeypatch)
    monkeypatch.setattr('library.embedding_ranking.MATRIX_MIN_CANDIDATES', 2)

    small = asyncio.run(svc._rank_ids_by_embedding(['d1'], 'q', 1))
    assert small[0][0] == 'd1'
    assert small[0][1] == pytest.approx(0.0, abs=1e-6)  # SQLite read the swapped DB

    big = asyncio.run(svc._rank_ids_by_embedding(['d1', 'd2'], 'q', 2))
    assert big[0][0] == 'd1'
    assert big[0][1] == pytest.approx(1.0, abs=1e-6)  # matrix
