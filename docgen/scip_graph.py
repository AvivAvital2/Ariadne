"""Symbols and edges, built from the SCIP model.

Stage one of the north star. ``docgen/scip_index.py`` decides what SCIP *means*; this
projects that into the rows a walk travels. The two are separate on purpose — the module
this replaces mixed index reading, graph building, persistence and eleven query methods in
one object, and that mixing is why the same class of defect kept recurring in it.

Construction is a **pure function** of an index: same index in, same rows out, nothing to
wire up and nothing to forget to call. Four things it gets right by construction, each
because the model already settled them:

* extents are body extents (``ScipDocument.extent_of``), so a cited hop is quotable;
* local ids are already document-scoped, so no row is shared between files;
* ``is_implementation`` becomes an ``implements`` edge, a relation the store never held;
* an edge is typed by **what it points at**, not by where it was found.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from docgen.scip_descriptors import (
    _enclosing_symbol_from_symbol,
    _symbol_descriptor_kind,
)

from attrs import field, frozen

if TYPE_CHECKING:  # pragma: no cover - typing only
    from docgen.scip_index import ScipDocument, ScipIndex
_TRAVERSABLE_OWNER_DESCRIPTOR_KINDS = frozenset({"type", "term", "method"})
@frozen
class CrossSourceSymbol:
    """A definition, with the coordinates a citation needs."""

    canonical_id: str
    source_name: str
    language: str
    file: str
    line_start: int
    line_end: int
    kind: str
    display_name: str
    qualified_name: str
    parent_qualified_name: 'str | None'


@frozen
class CrossSourceEdge:
    """One relation between two definitions, with the site that proves it."""

    caller: CrossSourceSymbol
    callee: CrossSourceSymbol
    edge_type: str
    file: str
    line: int
    confidence: str = 'exact'


@frozen
class GraphRows:
    """What one source contributes, plus what could not be attributed."""

    symbols: dict = field(factory=dict)
    edges: list = field(factory=list)
    unresolved_callees: int = 0
    unattributed_sites: int = 0


def classify_edge(callee_id: str) -> str:
    """``'call'`` for an invocation, ``'type_ref'`` for a type or attribute mention.

    SCIP records every non-definition occurrence identically, so the moniker's own grammar
    is the only signal: a callable ends ``).``, a class or trait in ``#``, an attribute or
    module value in a bare ``.``. Labelling everything a call put 145,817 Class-targeted
    edges into the databricks call graph, which ``callers`` and ``impact_radius`` then
    reported as callers.

    ``).`` and not ``().``: an overload carries a disambiguator — ``foo(+1).`` — and Scala
    constructors are always ``<init>(+N).``, so matching the empty-parens form alone drops
    every one of them. A synthetic id with none of these suffixes stays a call, so
    non-SCIP sources are unaffected.
    """
    if callee_id.startswith('local '):
        return 'type_ref'
    if callee_id.endswith(').'):
        return 'call'
    if callee_id.endswith('#') or callee_id.endswith('.'):
        return 'type_ref'
    return 'call'


def _tightest_enclosing(extents: list[tuple[int, int, str]],
                        line: int) -> 'str | None':
    """The innermost definition whose body contains ``line``.

    Largest start wins, so a call inside a nested function attributes to the nested one.
    """
    best: tuple[int, str] | None = None
    for start, end, symbol in extents:
        if start <= line <= end and (best is None or start > best[0]):
            best = (start, symbol)
    return best[1] if best is not None else None
def _symbols_of(document: 'ScipDocument', *, source_name: str,
                language: str) -> dict:
    """Every definition in one document, keyed by canonical id."""
    from docgen.scip_descriptors import _qualified_name_from_symbol

    out: dict = {}
    metadata = {info.symbol: info for info in document.symbols}
    for occurrence in document.definitions():
        if occurrence.is_parameter:
            # `Foo#bar().(self)` is not a standalone definition; promoting it pollutes
            # the resolver and collides with same-named nested methods.
            continue
        info = metadata.get(occurrence.symbol)
        start, end = document.extent_of(occurrence)
        qualified_name, parent = _qualified_name_from_symbol(
            occurrence.symbol, language)
        out[occurrence.symbol] = CrossSourceSymbol(
            canonical_id=occurrence.symbol,
            source_name=source_name,
            language=language,
            file=document.relative_path,
            line_start=start,
            line_end=end,
            kind=(info.effective_kind or '') if info else '',
            display_name=info.effective_display_name if info else '',
            qualified_name=qualified_name,
            parent_qualified_name=parent,
        )
    return out


def build_symbols(index: 'ScipIndex', *, source_name: str,
                  language: str) -> dict:
    """Every definition in an index, keyed by canonical id."""
    symbols: dict = {}
    for document in index.documents:
        symbols.update(_symbols_of(document, source_name=source_name,
                                   language=language))
    return symbols
def build_edges(index: 'ScipIndex', *, source_name: str, language: str,
                symbols: dict, resolve_external=None) -> tuple[list, int, int]:
    """Edges for one index, resolved against ``symbols``.

    ``symbols`` is deliberately the **whole** symbol set, not just this source's: two
    indexes can agree on a canonical id, and that agreement is a genuine cross-source
    edge. ``resolve_external`` is consulted only when a callee is absent from it — the
    moniker names a definition in another source that differs by package or version, so
    the canonical id misses while the qualified name matches.
    """
    edges: list = []
    unresolved = 0
    unattributed = 0
    for document in index.documents:
        metadata = {info.symbol: info for info in document.symbols}
        ownership_seen = set()
        for occurrence in document.definitions():
            child_id = occurrence.symbol
            info = metadata.get(child_id)
            parent_id = (
                (info.enclosing_symbol if info is not None else "")
                or _enclosing_symbol_from_symbol(child_id)
            )
            if not parent_id:
                continue
            if _symbol_descriptor_kind(parent_id) not in _TRAVERSABLE_OWNER_DESCRIPTOR_KINDS:
                continue
            parent = symbols.get(parent_id)
            child = symbols.get(child_id)
            if parent is None or child is None or parent.canonical_id == child.canonical_id:
                continue
            ownership = (parent.canonical_id, child.canonical_id)
            if ownership in ownership_seen:
                continue
            ownership_seen.add(ownership)
            edges.append(CrossSourceEdge(
                caller=parent, callee=child, edge_type="contains",
                file=child.file, line=child.line_start,
            ))
        # A call belongs to the callable it sits in. Locals are held in a SEPARATE
        # candidate list and consulted only when no named definition encloses the site,
        # because a local is not something that calls anything.
        #
        # Ranking them together put a fifth of the call graph behind a variable:
        # `x = foo()` defines the local on the same line as the call, its start line is
        # larger than the enclosing function's, and `_tightest_enclosing` prefers the
        # innermost scope — so `x` won. Measured on the rebuilt graph, 160,012 of 777,182
        # call edges were owned by a local, and every sampled one was a mis-attributed
        # call site: 84.4% left the enclosing callable with no edge to that callee at all
        # (`check_tests.main` never reached `run_pytest()`), and the rest survived only
        # because the same function happened to call the same callee from another line.
        # Stage one walks these edges, and a hop owned by a variable is not a hop.
        #
        # `dc138d3` kept locals as candidates so a call really on a local's line still
        # attributed and file-level impact still counted the dependency. That is what the
        # fallback preserves: module-level `X = foo()` has no callable around it, so the
        # local keeps the edge rather than the reference being dropped. A local also keeps
        # its OWN occurrence range and is never given a synthesised body span.
        enclosing: list[tuple[int, int, str]] = []
        local_sites: list[tuple[int, int, str]] = []
        for occurrence in document.definitions():
            if occurrence.is_parameter:
                continue
            if occurrence.is_local:
                start, end = occurrence.identifier_lines
                local_sites.append((start, end, occurrence.symbol))
            else:
                enclosing.append((*document.extent_of(occurrence), occurrence.symbol))

        for occurrence in document.occurrences:
            if occurrence.is_definition:
                continue
            callee = symbols.get(occurrence.symbol)
            confidence = 'exact'
            if callee is None and resolve_external is not None:
                callee = resolve_external(occurrence.symbol, language)
                confidence = 'resolved'
            if callee is None:
                unresolved += 1
                continue
            line = occurrence.identifier_lines[0]
            caller_id = (_tightest_enclosing(enclosing, line)
                         or _tightest_enclosing(local_sites, line))
            caller = symbols.get(caller_id) if caller_id else None
            if caller is None:
                unattributed += 1
                continue
            edges.append(CrossSourceEdge(
                caller=caller, callee=callee,
                edge_type=classify_edge(occurrence.symbol),
                file=document.relative_path, line=line, confidence=confidence,
            ))

        # `is_implementation` — the relation the store never held. It is declared on the
        # symbol, not at an occurrence, so its site is the implementor's own definition.
        for implementor_id, interface_id in document.implementations():
            implementor = symbols.get(implementor_id)
            interface = symbols.get(interface_id)
            if implementor is None or interface is None:
                unresolved += 1
                continue
            edges.append(CrossSourceEdge(
                caller=implementor, callee=interface, edge_type='implements',
                file=implementor.file, line=implementor.line_start,
            ))
    return edges, unresolved, unattributed


def build_rows(index: 'ScipIndex', *, source_name: str, language: str,
               resolve_external=None) -> GraphRows:
    """Project a loaded index into symbols and edges.

    Pure: no connection, no config, no ordering requirement. A caller cannot forget to
    call a second step, which is the failure that left ``doc_graph`` holding 769 imports
    and 491 calls against 2.47M available edges.
    """
    symbols = build_symbols(index, source_name=source_name, language=language)
    edges, unresolved, unattributed = build_edges(
        index, source_name=source_name, language=language, symbols=symbols,
        resolve_external=resolve_external)
    return GraphRows(symbols=symbols, edges=edges,
                     unresolved_callees=unresolved,
                     unattributed_sites=unattributed)
