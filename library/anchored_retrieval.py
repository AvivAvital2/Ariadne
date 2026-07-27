"""Anchored-ground retrieval ranking.

See ``designs/spool-anchored-retrieval.md``. The user repo is the protected
**anchor** (the subject — where the question lives); a spool is subordinate
**ground** (the environment/context the subject operates in). The ranking:

- ranks the anchor by query similarity and guarantees a floor of anchor slots
  the ground can never displace (a no-spool scope has an empty ground set, so
  the result is exactly today's query-ranked top-k);
- admits ground docs by a combined score of BOTH query similarity and
  similarity to the anchored docs, relevance-gated (so a query-similar but
  anchor-distant spool doc is not admitted — the spool is ground *for the
  repo*, not a free-floating match);
- diversifies the admitted ground (MMR) so it covers complementary facets
  rather than near-duplicates of the top ground doc;
- applies an optional per-doc quality multiplier to demote low-value catalog
  bloat before the gate.

Embeddings are assumed L2-normalized (cosine == dot product), matching the
document embedding matrix the ranker feeds off.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Defaults are conservative starting points; the search path may override them.
DEFAULT_W_QUERY = 0.5
DEFAULT_W_ANCHOR = 0.5
DEFAULT_GATE = 0.35
DEFAULT_DIVERSITY = 0.5


def _cos(a: 'NDArray', b: 'NDArray') -> float:
    """Cosine similarity for L2-normalized vectors (== dot product)."""
    import numpy as np

    return float(np.dot(a, b))


def anchored_rank(
    query_emb: 'NDArray',
    anchor: 'list[tuple[str, NDArray]]',
    ground: 'list[tuple[str, NDArray]]',
    *,
    limit: int,
    anchor_floor: int,
    w_q: float = DEFAULT_W_QUERY,
    w_a: float = DEFAULT_W_ANCHOR,
    gate: float = DEFAULT_GATE,
    diversity: float = DEFAULT_DIVERSITY,
    weights: 'dict[str, float] | None' = None,
) -> list[str]:
    """Return up to ``limit`` doc ids: anchor (protected) + gated/diverse ground.

    Args:
        query_emb: the (normalized) query embedding.
        anchor: ``(doc_id, embedding)`` for the user repo's in-scope docs.
        ground: ``(doc_id, embedding)`` for the spool's in-scope docs.
        limit: max results.
        anchor_floor: anchor slots guaranteed against ground displacement.
        w_q, w_a: weights on query similarity and anchor similarity in the
            ground doc's combined score.
        gate: minimum combined score for a ground doc to be admitted (dynamic
            sizing — an unrelated question admits ~no ground).
        diversity: MMR redundancy penalty in ``[0, 1]`` on the admitted ground.
        weights: optional ``{doc_id: multiplier}`` applied to every doc's
            score (anchor query-sim and ground combined, before the gate).
            Carries provenance weighting (code > human-doc) and bloat
            demotion (``< 1`` for low-value catalog docs).
    """
    weights = weights or {}

    # -- anchor ranked by (weighted) query similarity; floor is protected ---
    anchor_ranked = sorted(
        ((doc_id, _cos(query_emb, emb) * weights.get(doc_id, 1.0))
         for doc_id, emb in anchor),
        key=lambda pair: pair[1],
        reverse=True,
    )
    n_reserved = max(0, min(anchor_floor, limit, len(anchor_ranked)))
    reserved = [doc_id for doc_id, _ in anchor_ranked[:n_reserved]]
    anchor_rest = anchor_ranked[n_reserved:]
    remaining = limit - len(reserved)
    if remaining <= 0:
        return reserved

    # -- ground: combined score (query AND anchor), quality, relevance gate --
    anchor_embs = [emb for _, emb in anchor]
    scored: list[tuple[str, 'NDArray', float]] = []
    for doc_id, emb in ground:
        query_sim = _cos(query_emb, emb)
        anchor_sim = max((_cos(emb, ae) for ae in anchor_embs), default=0.0)
        combined = (w_q * query_sim + w_a * anchor_sim) * weights.get(doc_id, 1.0)
        if combined >= gate:
            scored.append((doc_id, emb, combined))

    # -- MMR select the ground, capped at the remaining slots ---------------
    ground_emb = {doc_id: emb for doc_id, emb, _ in scored}
    selected: list[str] = []
    pool = scored[:]
    while pool and len(selected) < remaining:
        if not selected:
            best = max(pool, key=lambda item: item[2])
        else:
            best = max(
                pool,
                key=lambda item: item[2] - diversity * max(
                    _cos(item[1], ground_emb[s]) for s in selected
                ),
            )
        selected.append(best[0])
        pool.remove(best)

    # -- fill any leftover slots with the anchor remainder ------------------
    leftover = remaining - len(selected)
    anchor_tail = [doc_id for doc_id, _ in anchor_rest[:max(leftover, 0)]]

    return reserved + selected + anchor_tail


__all__ = ['anchored_rank']
