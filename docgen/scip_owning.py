"""The single owning-symbol resolver: map a ``(relative_path, 0-indexed line)``
to the ``canonical_id`` of the enclosing code symbol, read straight from the
SCIP index's occurrences.

This is the one correct answer to "what encloses this position", replacing five
divergent ``scip_symbols`` line-range + ``kind``-filter queries that returned
nothing on real scip-python output (``kind`` unset, relative-vs-absolute paths,
name-token-only ranges). See ``designs/owning-symbol-resolution-fix.md``.

Two properties resolution depends on, and why this reads occurrences rather than
the ``scip_symbols`` table:

* ``enclosing_range`` — a definition's *body* span (a separate SCIP field). A
  definition's plain ``range`` is only the name token (one line for
  scip-python), which cannot contain a body call site. We use the body span when
  present, falling back to the name token otherwise.
* descriptor kind — scip-python leaves ``SymbolInformation.kind`` unset, so we
  classify the owner by its canonical-id *descriptor* (via ``scip_descriptors``),
  keeping only genuine containers — method/function, type/class, term/field —
  and excluding parameters, type-parameters, packages, and ``local N`` symbols.
  A parameter's ``enclosing_range`` spans the whole function body, so without
  this filter a body literal would bind to a parameter symbol that is not a
  persisted graph node and therefore cannot be traversed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from docgen.scip_descriptors import _parse_descriptors

if TYPE_CHECKING:
    from docgen.scip_extractor import ScipIndex

# Descriptor kinds that can own a call site. ``parameter`` / ``package`` /
# ``typealias`` and ``local N`` symbols are deliberately excluded.
_OWNING_KINDS = frozenset({'method', 'type', 'term'})


def _is_owning_symbol(symbol: str) -> bool:
    """True if ``symbol`` is a genuine owning container — a method/function, a
    type/class, or a term/field. A query inside a body belongs to the enclosing
    function, never to one of its parameters, so parameter (and package /
    type-alias / ``local N``) symbols are not owners."""
    if symbol.startswith('local '):
        return False
    descriptors = symbol.split(' ')[-1]
    desc = _parse_descriptors(descriptors)
    if not desc:
        return False
    return desc[-1][1] in _OWNING_KINDS


def build_owning_resolver(
    index: 'ScipIndex',
) -> Callable[[str, int], 'str | None']:
    """Return ``owning(relative_path, line0) -> canonical_id | None``.

    ``line0`` is **0-indexed** (the SCIP wire convention); callers working in
    1-indexed AST coordinates must pass ``line - 1``. Resolution picks the
    smallest enclosing definition span containing the line; ties resolve to the
    first occurrence in document order (deterministic). ``None`` is a clean
    signal — a module-level site with no enclosing owner — not an error.
    """
    by_file: dict[str, list[tuple[int, int, str]]] = {}
    for doc in index.documents:
        for occ in doc.occurrences:
            if not (occ.is_definition and occ.range):
                continue
            if not _is_owning_symbol(occ.symbol):
                continue
            span = occ.enclosing_range or occ.range
            start = span[0]
            end = span[2] if len(span) >= 4 else span[0]
            by_file.setdefault(doc.relative_path, []).append(
                (start, end, occ.symbol))

    def owning(relative_path: str, line0: int) -> 'str | None':
        best: tuple[int, str] | None = None
        for start, end, sym in by_file.get(relative_path, ()):
            if start <= line0 <= end and (best is None or end - start < best[0]):
                best = (end - start, sym)
        return best[1] if best else None

    return owning


__all__ = ['build_owning_resolver']
