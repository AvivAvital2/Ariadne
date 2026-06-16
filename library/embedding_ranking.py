"""Embedding ranking strategies, selected by candidate-set size.

Large candidate sets rank fastest against the shared mmap'd matrix (one flat
full-matrix matmul, ~70 ms regardless of count); small sets are cheaper loading
just their own embeddings from SQLite than paying that matmul. ``select_ranker``
picks the right strategy for a given candidate count.

See designs/embedding-matrix-tier2-serve.md.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from library import Library
    from library.embedding_matrix import EmbeddingMatrix

# At or above this candidate count the shared mmap matrix wins; below it the
# SQLite per-candidate load is cheaper than the matrix's flat full-matmul cost.
# That matmul runs over the WHOLE corpus, so its cost — and thus this crossover —
# grows with total doc count; raise this as the corpus grows. Measured crossover
# is ~3000 candidates for ~80k docs (matmul ~67ms vs ~20us/candidate to load).
MATRIX_MIN_CANDIDATES = 3000


class MatrixRanker:
    """Ranks candidates against the preloaded mmap matrix — best for large
    candidate sets (one flat full-matrix matmul, independent of count)."""

    def __init__(self, matrix: 'EmbeddingMatrix') -> None:
        self._matrix = matrix

    def rank(
        self, query_embedding: 'NDArray[np.float32]', candidate_ids, limit: int, weights=None
    ) -> list[tuple[str, float]]:
        return self._matrix.rank(query_embedding, candidate_ids, limit, weights=weights)


class SqliteRanker:
    """Loads only the candidates' embeddings from SQLite — best for small sets,
    where that beats a full-matrix matmul. Returns ``[]`` when no candidate has
    an embedding (the caller then applies its text fallback)."""

    def __init__(self, library: 'Library') -> None:
        self._library = library

    def rank(
        self, query_embedding: 'NDArray[np.float32]', candidate_ids, limit: int, weights=None
    ) -> list[tuple[str, float]]:
        from search import batch_dot_similarity, top_k_indices

        embeddings = self._library.get_embeddings_for_ids(candidate_ids)
        if not embeddings:
            return []
        ordered = [cid for cid in candidate_ids if cid in embeddings]
        matrix = np.stack([embeddings[cid] for cid in ordered])
        similarities = batch_dot_similarity(query_embedding, matrix)
        if weights is not None:
            similarities = similarities * np.array(
                [weights.get(cid, 1.0) for cid in ordered], dtype=np.float32)
        return [(ordered[i], float(similarities[i])) for i in top_k_indices(similarities, limit)]


def select_ranker(
    candidate_count: int,
    matrix_provider: 'Callable[[], EmbeddingMatrix | None]',
    library: 'Library',
    *,
    threshold: int | None = None,
) -> 'MatrixRanker | SqliteRanker':
    """Pick the ranking strategy for this candidate count.

    ``matrix_provider`` is a zero-arg callable returning a fresh
    ``EmbeddingMatrix`` or ``None`` (absent/stale). ``threshold`` defaults to
    ``MATRIX_MIN_CANDIDATES``, read at call time so it stays patchable. A large
    candidate set uses the matrix only when one is available; otherwise (and for
    small sets) it loads candidates from SQLite.
    """
    if threshold is None:
        threshold = MATRIX_MIN_CANDIDATES
    if candidate_count >= threshold:
        matrix = matrix_provider()
        if matrix is not None:
            return MatrixRanker(matrix)
    return SqliteRanker(library)
