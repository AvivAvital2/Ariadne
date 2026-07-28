"""Themes-tool implementation for the Ariadne MCP server (Themes plan, Phase 6).

`themes_action` is the routing helper used by the `ariadne_themes` MCP tool;
it takes a Library and the action parameters, then returns a JSON-serializable
dict matching the response shape contracted in the plan §5.6.

**Operates below the closure chokepoint.** Themes are cross-source
library internals (a cluster can span members from any source; theme
summary docs are created by ``docgen/cluster.py`` without a
``source_name`` so they can't be filtered by per-source closure). Raw
``library.X(...)`` access here is intentional — see
``designs/directional-closure-scoping.md`` § "Library-internal modules
— legitimately unscoped".
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from library import Library


_ACTIONS = ('list', 'get', 'members', 'stats')


def _theme_to_summary(library: "Library", theme) -> dict:
    """Compact response shape for action='list'."""
    doc = library.get_document(theme.doc_id)
    title = doc.title if doc else f'Theme {theme.cluster_id[:12]}'
    return {
        'cluster_id': theme.cluster_id,
        'title': title,
        'member_count': theme.member_count,
        'coherent': theme.coherent,
        'dirty': theme.dirty,
        'last_summarized_at': theme.last_summarized_at,
    }


def _action_list(
    library: "Library",
    *,
    coherent_only: bool,
    source: str | None,
    limit: int,
) -> dict:
    themes = library.list_themes(coherent_only=coherent_only, source=source)
    payload = [_theme_to_summary(library, t) for t in themes[:limit]]
    return {'themes': payload, 'total': len(themes)}


def _action_get(library: "Library", cluster_id: str | None) -> dict:
    if not cluster_id:
        return {'error': "cluster_id is required for action='get'"}
    theme = library.get_theme(cluster_id)
    if theme is None:
        return {'error': f'unknown cluster_id: {cluster_id}'}
    doc = library.get_document(theme.doc_id)
    return {
        'cluster_id': theme.cluster_id,
        'title': doc.title if doc else None,
        'content': doc.content if doc else None,
        'member_count': theme.member_count,
        'coherent': theme.coherent,
        'dirty': theme.dirty,
        'last_summarized_at': theme.last_summarized_at,
        'summary_hash': theme.summary_hash,
    }


def _action_members(library: "Library", cluster_id: str | None) -> dict:
    if not cluster_id:
        return {'error': "cluster_id is required for action='members'"}
    pairs = library.get_theme_members(cluster_id)
    out: list[dict] = []
    for element_id, weight in pairs:
        doc = library.get_document(element_id)
        out.append({
            'element_id': element_id,
            'title': doc.title if doc else element_id,
            'weight': weight,
        })
    # Sorted by weight desc — strongest cluster ties first.
    out.sort(key=lambda m: m['weight'], reverse=True)
    return {'members': out}


def _action_stats(library: "Library", *, source: str | None) -> dict:
    """Coherence-rate readout — coherent / incoherent / total + rate.

    The number behind "does theme discovery hold up at scale?": how many
    clusters passed the LLM coherence gate. ``coherent_rate`` is 0.0 for
    an empty corpus (no division by zero).
    """
    counts = library.theme_coherence_counts(source=source)
    total = counts['total']
    rate = counts['coherent'] / total if total else 0.0
    return {
        'coherent': counts['coherent'],
        'incoherent': counts['incoherent'],
        'total': total,
        'coherent_rate': rate,
    }


def themes_action(
    library: "Library",
    action: Literal['list', 'get', 'members', 'stats'] = 'list',
    cluster_id: str | None = None,
    coherent_only: bool = True,
    source: str | None = None,
    limit: int = 50,
) -> dict:
    """Dispatch a themes inspection request and return a serializable dict.

    Themes are library-internal cross-source data; this function takes
    the raw ``library`` (not a ``ScopedLibrary``) by design — see
    module docstring.
    """
    if action == 'list':
        return _action_list(
            library, coherent_only=coherent_only, source=source, limit=limit,
        )
    if action == 'get':
        return _action_get(library, cluster_id)
    if action == 'members':
        return _action_members(library, cluster_id)
    if action == 'stats':
        return _action_stats(library, source=source)
    return {'error': f'unknown action: {action!r}; valid actions: {_ACTIONS}'}


__all__ = ['themes_action']
