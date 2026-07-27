"""Consulting spool expert aisles (designs/spool-expert-aisles.md §3-4).

Wires the router (:mod:`spool_router`) to actual aisle consultation:

- **load** the enabled spools as consultable aisles (themes for routing,
  endpoint to reach them, advisory taxonomy);
- **route** a question to the relevant aisle(s) and **consult** each — via a
  ``consult`` seam so the transport is pluggable (MCP-backed in production, a
  subprocess ``ariadne mcp`` per aisle; a fake in tests). The consults are
  independent, so production may fan them out across threads;
- **combine** the grounded aisle answers back with the project context.

The orchestration never merges stores — it consults separate expert Ariadnes
and folds their answers in, so aisles stay pure and reusable across projects.
"""
from __future__ import annotations

from attrs import frozen

from spool_router import ROUTE_THRESHOLD, ROUTE_TOP_K, Aisle, route


@frozen
class AisleAnswer:
    """What one aisle returns from a consult: the aisle id, its grounded answer,
    and the symbols/docs it cited (so combine can surface provenance)."""
    aisle: str
    answer: str
    citations: tuple[str, ...] = ()


def load_aisles(
    enabled: 'list[str]',
    *,
    themes_loader,
    taxonomy_loader,
    endpoint_for,
) -> 'list[Aisle]':
    """Build a consultable :class:`Aisle` per enabled spool: its routing theme
    vectors (``themes_loader``), advisory taxonomy (``taxonomy_loader``), and
    MCP endpoint (``endpoint_for``). An aisle whose spool produced NO themes is
    dropped — it can't be routed to, so registering it would be dead weight.
    """
    aisles: list[Aisle] = []
    for name in enabled:
        themes = tuple(themes_loader(name) or ())
        if not themes:
            continue
        aisles.append(Aisle(
            name=name,
            theme_embeddings=themes,
            endpoint=endpoint_for(name),
            taxonomy=tuple(taxonomy_loader(name) or ()),
        ))
    return aisles


def consult_relevant(
    question: str,
    query_embedding: 'tuple[float, ...]',
    aisles: 'list[Aisle]',
    *,
    consult,
    threshold: float = ROUTE_THRESHOLD,
    top_k: int = ROUTE_TOP_K,
) -> 'list[AisleAnswer]':
    """Route ``question`` to the relevant aisle(s) and consult each.

    ``consult(aisle, question) -> AisleAnswer`` is the transport seam (an
    MCP-backed call in production; a fake in tests). Returns [] when routing
    wakes no aisle — the caller then answers from the library alone, paying
    nothing for the enabled-but-irrelevant aisles. The consults are independent
    and may be parallelised by the caller.
    """
    picked = route(query_embedding, aisles, threshold=threshold, top_k=top_k)
    return [consult(aisle, question) for aisle in picked]


def combine(project_context: str, aisle_answers: 'list[AisleAnswer]') -> str:
    """Fold the aisle answers back into the project context — the grounded
    material a final synthesis reasons over. No aisle answers → the project
    context is returned unchanged (nothing to consult)."""
    if not aisle_answers:
        return project_context
    parts = [project_context, '', '# Consulted expert aisles']
    for ans in aisle_answers:
        parts.append(f'\n## {ans.aisle}\n{ans.answer}')
        if ans.citations:
            parts.append(f'  (grounded in: {", ".join(ans.citations)})')
    return '\n'.join(parts)
