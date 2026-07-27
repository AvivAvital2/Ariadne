"""Per-request scope resolution.

A request enters Ariadne through an MCP tool, a CLI command, or
ariadne.api directly. Each entry point must turn whatever the caller
supplied (an explicit source argument, just a cwd, or nothing) into a
``ScopedLibrary`` before any data is read — that's what makes the
chokepoint structural rather than per-call best-effort.

This module is the one place that resolution lives so both surfaces
share it. Phase 3 of ``designs/directional-closure-scoping.md``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from library import Library, ScopedLibrary


def resolve_source_name(
    config: 'Config',
    source: str | None,
    *,
    cwd: Path | None = None,
    use_cwd: bool = True,
) -> str | None:
    """Pick a source name using the standard resolution order.

    Order:
      1. explicit ``source`` argument if non-empty
      2. cwd-based detection (``Config.get_source_scope(cwd)``) — skipped
         when ``use_cwd=False``. The MCP server sets ``use_cwd=False``
         because its process cwd is the Ariadne install, not the user's
         project, so cwd detection would always mis-resolve to whatever
         source contains Ariadne.

    Returns ``None`` when nothing resolves. There is deliberately **no**
    ``default_source`` fallback: a silent default answers from the wrong
    repo (the failure this guards against), so an undetermined source must
    fail closed — see :func:`make_scoped_library`. Callers that genuinely
    want every source use :func:`make_global_scoped_library` explicitly.
    """
    if source:
        return source
    if use_cwd:
        if cwd is None:
            cwd = Path.cwd()
        resolved = config.get_source_scope(cwd)
        if resolved:
            return resolved
    return None


def make_global_scoped_library(
    config: 'Config',
    library: 'Library',
) -> 'ScopedLibrary':
    """Closure-scoped view covering every configured source.

    For internal maintenance operations that genuinely need to see the
    whole library — theme generation/refresh, cross-source graph
    materialization, integrity checks — the closure should span all
    sources. This is "scoped to global": the chokepoint discipline still
    runs every read through ``ScopedLibrary``, but the closure
    encompasses everything so no actual filtering occurs.

    Distinct from ``make_scoped_library(config, library, None)`` which
    falls back to cwd / default_source and fails-closed if neither
    resolves — global scope is an explicit choice for internal callers
    that intentionally cross sources.
    """
    from library import ScopedLibrary
    config.hydrate_relations(library.all_source_relations())

    sources = frozenset(config.sources)
    if not sources:
        raise LookupError(
            'No sources configured — cannot build a global ScopedLibrary.',
        )
    return ScopedLibrary(library, sources)


def make_scoped_library(
    config: 'Config',
    library: 'Library',
    source: str | None,
    *,
    cwd: Path | None = None,
    use_cwd: bool = True,
) -> 'ScopedLibrary':
    """Resolve a source argument into a closure-scoped Library view.

    Wraps ``resolve_source_name`` with the fail-closed contract: if no
    source resolves, raise ``LookupError`` with a message guiding the
    caller toward a fix. Returning an unscoped library — or scoping to
    every configured source — would silently widen results past the
    caller's intent, which is the failure mode this helper exists to
    prevent.
    """
    from library import ScopedLibrary
    config.hydrate_relations(library.all_source_relations())

    resolved = resolve_source_name(config, source, cwd=cwd, use_cwd=use_cwd)
    if resolved is None:
        configured = sorted(config.sources)
        raise LookupError(
            'No source context — pass source= explicitly or run within '
            'a configured project tree. Currently configured sources: '
            f'{configured}.',
        )
    try:
        closure = config.scope_closure(resolved)
    except KeyError as e:
        # Unify the fail-closed contract: callers (CLI, MCP tools, and
        # Claude per ``mcp_server.py`` instructions) expect LookupError
        # as the single signal for "source not resolvable". KeyError IS
        # a LookupError subclass, but the instruction text says
        # LookupError specifically; re-raise with the same shape so the
        # message stays user-friendly.
        configured = sorted(config.sources)
        raise LookupError(
            f'Unknown source {resolved!r}; configured sources: '
            f'{configured}. Pass source= as one of the configured '
            'names.',
        ) from e
    except ValueError as e:
        # Config.scope_closure raises ValueError on depends_on cycle.
        # Surface it through the same LookupError channel so callers don't
        # need a separate handler for the misconfig case.
        raise LookupError(
            f'Misconfigured source graph: {e}. Fix the cycle in '
            'ariadne.yaml depends_on.',
        ) from e
    # Registered spools join the query scope (enabled + cached + pin OK —
    # designs/spool-environment-plugin.md §18.6.4); an unresolvable spool is
    # simply absent here, its gap surfaces via `ariadne spools` / honest-gap.
    from spools import active_spool_sources
    return ScopedLibrary(library, closure | active_spool_sources(config))
