"""Semantic search functionality for the Claude Library.

This module provides vector similarity search using cosine similarity.
"""
from __future__ import annotations

__all__ = ['batch_cosine_similarity', 'batch_dot_similarity', 'cosine_similarity', 'top_k_indices']

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute the cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score between 0 and 1.
        (Embeddings are typically normalized, so values are in [0, 1].)
    """
    # Handle potential zero vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_cosine_similarity(
    query: NDArray[np.float32],
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute cosine similarity between a query and multiple embeddings.

    This is more efficient than calling cosine_similarity in a loop.

    Args:
        query: Query vector of shape (dim,).
        embeddings: Matrix of embeddings of shape (n, dim).

    Returns:
        Array of similarity scores of shape (n,).
    """
    # Normalize query
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(embeddings), dtype=np.float32)
    query_normalized = query / query_norm

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    embeddings_normalized = embeddings / norms

    # Compute similarities
    similarities = embeddings_normalized @ query_normalized
    return similarities.astype(np.float32)


def batch_dot_similarity(
    query: NDArray[np.float32],
    embeddings: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Dot product similarity for pre-normalized embeddings.

    When embeddings are unit vectors (norm=1), dot product equals cosine
    similarity. Skips the per-query normalization of all stored embeddings
    that batch_cosine_similarity performs.

    Args:
        query: Query vector of shape (dim,). Will be normalized.
        embeddings: Matrix of PRE-NORMALIZED embeddings of shape (n, dim).

    Returns:
        Array of similarity scores of shape (n,).
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(len(embeddings), dtype=np.float32)
    return (embeddings @ (query / query_norm)).astype(np.float32)


def top_k_indices(
    similarities: NDArray[np.float32],
    k: int,
) -> list[int]:
    """Get indices of top-k most similar items.

    Args:
        similarities: Array of similarity scores.
        k: Number of top results to return.

    Returns:
        List of indices sorted by descending similarity.
    """
    if len(similarities) == 0:
        return []

    k = min(k, len(similarities))

    # Use argpartition for efficiency when k << n
    if k < len(similarities) // 2:
        # Get indices of top k (unordered)
        top_k_unsorted = np.argpartition(similarities, -k)[-k:]
        # Sort those k indices by their similarity scores
        sorted_indices = top_k_unsorted[np.argsort(similarities[top_k_unsorted])[::-1]]
    else:
        # For larger k, just sort everything
        sorted_indices = np.argsort(similarities)[::-1][:k]

    return sorted_indices.tolist()
