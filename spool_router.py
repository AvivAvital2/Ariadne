"""Theme-routed aisle selection (designs/spool-expert-aisles.md §3).

The cheap coarse tier of the expert-aisles architecture: given a question and
the enabled spool *aisles* (each a standalone Ariadne carrying its themes as
embedding vectors), decide WHICH aisle(s) to consult. Matching a question
against a small set of per-aisle theme vectors is nearly free, so enabling many
aisles costs nothing per question — only the aisle(s) whose themes actually
match are woken, and each then runs its own precise retrieval (the second tier).

Pure similarity math over precomputed theme vectors; the real embedder is the
caller's concern (embed the question once, pass the vector in).
"""
from __future__ import annotations

import math

from attrs import frozen


@frozen
class Aisle:
    """A consultable spool aisle: an id + its themes as embedding vectors (for
    routing) + how to reach it + its advisory taxonomy. ``endpoint`` is how to
    reach the aisle's MCP (opaque to routing); ``taxonomy`` is the aisle's
    concern/opportunity/gotcha lens, injected when the aisle is consulted.
    Routing only reads ``theme_embeddings``."""
    name: str
    theme_embeddings: tuple[tuple[float, ...], ...]
    endpoint: str = ''
    taxonomy: tuple[str, ...] = ()


def _cosine(a: 'tuple[float, ...]', b: 'tuple[float, ...]') -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def route(
    query_embedding: 'tuple[float, ...]',
    aisles: 'list[Aisle]',
    *,
    threshold: float = 0.25,
    top_k: int = 3,
) -> 'list[Aisle]':
    """Aisles whose themes are relevant to ``query_embedding``, best-first,
    capped at ``top_k``. An aisle qualifies on its BEST-matching theme (max
    cosine), and only if that clears ``threshold``. Returns [] when nothing
    clears the bar — the caller then answers from the library alone, waking no
    aisle (the no-slowdown property)."""
    scored: list[tuple[float, Aisle]] = []
    for aisle in aisles:
        best = max(
            (_cosine(query_embedding, theme)
             for theme in aisle.theme_embeddings),
            default=0.0,
        )
        if best >= threshold:
            scored.append((best, aisle))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [aisle for _score, aisle in scored[:top_k]]
