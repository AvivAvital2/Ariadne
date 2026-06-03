"""Tests for the Claude Library search module."""
from __future__ import annotations

import numpy as np

from search import batch_cosine_similarity, batch_dot_similarity, cosine_similarity, top_k_indices


class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Test similarity of identical vectors."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        similarity = cosine_similarity(a, b)
        assert abs(similarity - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        """Test similarity of orthogonal vectors."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        similarity = cosine_similarity(a, b)
        assert abs(similarity) < 1e-6

    def test_opposite_vectors(self) -> None:
        """Test similarity of opposite vectors."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([-1.0, -2.0, -3.0], dtype=np.float32)

        similarity = cosine_similarity(a, b)
        assert abs(similarity + 1.0) < 1e-6

    def test_zero_vector(self) -> None:
        """Test similarity with zero vector."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)

        similarity = cosine_similarity(a, b)
        assert similarity == 0.0

    def test_normalized_vectors(self) -> None:
        """Test with pre-normalized vectors."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.707107, 0.707107], dtype=np.float32)  # 45 degrees

        similarity = cosine_similarity(a, b)
        assert abs(similarity - 0.707107) < 1e-5


class TestBatchCosineSimilarity:
    """Tests for the batch_cosine_similarity function."""

    def test_batch_similarity(self) -> None:
        """Test batch similarity computation."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Same as query
            [0.0, 1.0, 0.0],  # Orthogonal
            [-1.0, 0.0, 0.0],  # Opposite
        ], dtype=np.float32)

        similarities = batch_cosine_similarity(query, embeddings)

        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-6
        assert abs(similarities[1]) < 1e-6
        assert abs(similarities[2] + 1.0) < 1e-6

    def test_batch_with_zero_query(self) -> None:
        """Test batch similarity with zero query vector."""
        query = np.zeros(3, dtype=np.float32)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        similarities = batch_cosine_similarity(query, embeddings)
        assert all(s == 0.0 for s in similarities)

    def test_batch_empty_embeddings(self) -> None:
        """Test batch similarity with empty embeddings."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.empty((0, 3), dtype=np.float32)

        similarities = batch_cosine_similarity(query, embeddings)
        assert len(similarities) == 0


class TestTopKIndices:
    """Tests for the top_k_indices function."""

    def test_top_k(self) -> None:
        """Test getting top-k indices."""
        similarities = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)

        top_3 = top_k_indices(similarities, k=3)
        assert top_3 == [3, 1, 2]  # indices of 0.9, 0.5, 0.3

    def test_k_larger_than_array(self) -> None:
        """Test when k is larger than array length."""
        similarities = np.array([0.5, 0.3], dtype=np.float32)

        top_5 = top_k_indices(similarities, k=5)
        assert len(top_5) == 2
        assert top_5 == [0, 1]

    def test_empty_array(self) -> None:
        """Test with empty array."""
        similarities = np.array([], dtype=np.float32)
        result = top_k_indices(similarities, k=3)
        assert result == []

    def test_k_equals_one(self) -> None:
        """Test getting only top result."""
        similarities = np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32)

        top_1 = top_k_indices(similarities, k=1)
        assert top_1 == [2]

    def test_ties_handling(self) -> None:
        """Test handling of tied values."""
        similarities = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        top_2 = top_k_indices(similarities, k=2)
        assert len(top_2) == 2
        # All have same score, so any 2 indices are valid
        assert all(i in [0, 1, 2] for i in top_2)


class TestBatchDotSimilarity:
    """Tests for the batch_dot_similarity function (pre-normalized vectors)."""

    def test_matches_cosine_on_normalized_vectors(self) -> None:
        """Dot product on unit vectors must equal cosine similarity."""
        query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
        ], dtype=np.float32)
        # Normalize both
        query_n = query / np.linalg.norm(query)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_n = embeddings / norms

        dot_sims = batch_dot_similarity(query_n, emb_n)
        cos_sims = batch_cosine_similarity(query_n, emb_n)

        np.testing.assert_allclose(dot_sims, cos_sims, atol=1e-6)

    def test_zero_query(self) -> None:
        """Zero query should return all zeros."""
        query = np.zeros(3, dtype=np.float32)
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        sims = batch_dot_similarity(query, embeddings)
        assert sims[0] == 0.0

    def test_empty_embeddings(self) -> None:
        """Empty embedding matrix should return empty array."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.empty((0, 3), dtype=np.float32)
        sims = batch_dot_similarity(query, embeddings)
        assert len(sims) == 0

    def test_identical_normalized_vectors(self) -> None:
        """Identical unit vectors should have similarity ~1.0."""
        v = np.array([0.6, 0.8, 0.0], dtype=np.float32)  # already unit
        embeddings = np.array([v], dtype=np.float32)
        sims = batch_dot_similarity(v, embeddings)
        assert abs(sims[0] - 1.0) < 1e-6
