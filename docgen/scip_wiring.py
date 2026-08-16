"""The wiring gate — SCIP ingest invariants, checked loudly.

Every SCIP defect in this codebase's history succeeded silently: ``build_graph`` was never
on the onboard path, ``resolve_external_to`` was gated on an id production never produces,
staleness was computed and never threaded, body extents were computed and discarded, and
``local N`` ids were never namespaced so a single row fused 4,446 files. In each case the
pipeline reported success and the numbers looked plausible.

That is the failure mode this module exists to end. A store either satisfies the ingest
invariants or it is wrong, and the report says which one broke. It is the definition of
"the wiring works", and it runs before anything downstream is trusted.

Stage one of the north star — *travel index* — cannot be believed without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlite3 import Connection

#: Below this share of named definitions carrying a multi-line extent, the store is
#: storing identifier spans rather than bodies and nothing it cites can be quoted. A
#: floor, not a target: a corpus of one-line declarations legitimately sits low, so the
#: check only catches the collapse (the live store measured exactly 0.0%).
MIN_MULTI_LINE_SHARE = 0.01


@dataclass(frozen=True)
class WiringCheck:
    """One invariant, its verdict, and the numbers behind it."""

    name: str
    ok: bool
    detail: str
    measured: dict = field(default_factory=dict)


@dataclass(frozen=True)
class WiringReport:
    checks: list[WiringCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    def failures(self) -> list[WiringCheck]:
        return [check for check in self.checks if not check.ok]


def _one(conn: 'Connection', sql: str, args: tuple = ()) -> int:
    return conn.execute(sql, args).fetchone()[0] or 0


def _local_ids_namespaced(conn: 'Connection') -> WiringCheck:
    """No bare ``local N`` id.

    SCIP numbers local bindings per document while ``canonical_id`` is a global primary
    key, so a bare ``local 5`` is one row shared by every document that emitted that
    index — measured on the live store, edges from 4,446 distinct files pointed at it,
    and 19.5% of the call graph routed through such nodes.
    """
    # `local 1` is bare; `local src1:a.py:1` is scoped. The scoped form keeps the
    # `local ` prefix on purpose so every existing detector still fires, so the
    # invariant is "nothing is ONLY local N" — GLOB, because the character after the
    # space is a digit for a bare id and never one for a scoped id.
    bare = _one(conn,
                "SELECT COUNT(*) FROM scip_symbols "
                "WHERE canonical_id GLOB 'local [0-9]*'")
    total = _one(conn, 'SELECT COUNT(*) FROM scip_symbols')
    return WiringCheck(
        name='local_ids_namespaced',
        ok=total > 0 and bare == 0,
        detail=('no store' if total == 0 else
                'ok' if bare == 0 else
                f'{bare} bare local ids fuse unrelated documents'),
        measured={'bare_local_ids': bare, 'symbols': total},
    )
def _locals_never_cross_sources(conn: 'Connection') -> WiringCheck:
    """No cross-source edge runs through a document-local binding.

    Cross-source edges are the point of the cross-source graph, not a defect in it:
    stage one's travel index crosses repositories on them, ``ariadne callers`` is
    documented as a cross-source caller tree, and ``_resolve_external`` mints them
    deliberately when a reference's moniker differs from the definition's by package or
    version. The earlier form of this check banned every crossing, which a correct
    multi-source store can never satisfy -- the only way to pass it was to switch
    resolution off and surrender the capability.

    What can never be legitimate is a crossing *through a local*. A local binding is
    numbered per document and private to it, so it is not something another source can
    refer to; when it sits at one end of a cross-source edge, two documents were handed
    the same identity. On the live store 499,248 of 499,249 cross-source edges ran
    through one, while a correctly ingested store held 50,182 locals and crossed on none
    of them.

    Matched on the ``local `` prefix rather than the bare ``local N`` shape, so a scoping
    scheme that is per-document but not per-source -- which ``local_ids_namespaced``
    accepts -- still cannot weld two repositories together unnoticed.
    """
    _endpoints = (
        'FROM scip_edges e '
        'JOIN scip_symbols c ON c.canonical_id = e.caller_canonical_id '
        'JOIN scip_symbols d ON d.canonical_id = e.callee_canonical_id '
        'WHERE c.source_name <> d.source_name '
    )
    fused = _one(conn,
                 'SELECT COUNT(*) ' + _endpoints +
                 "AND (c.canonical_id LIKE 'local %' "
                 "OR d.canonical_id LIKE 'local %')")
    crossing = _one(conn, 'SELECT COUNT(*) ' + _endpoints)
    edges = _one(conn, 'SELECT COUNT(*) FROM scip_edges')
    return WiringCheck(
        name='locals_never_cross_sources',
        ok=edges > 0 and fused == 0,
        detail=('no edges' if edges == 0 else
                'ok' if fused == 0 else
                f'{fused} cross-source edges run through a document-local id, '
                f'so unrelated documents share one identity'),
        measured={'fused_cross_source_edges': fused,
                  'cross_source_edges': crossing, 'edges': edges},
    )


def _definition_extents_present(conn: 'Connection') -> WiringCheck:
    """Named definitions carry body extents, not identifier spans.

    Without this a cited hop cannot be quoted: ``get_element_body`` has one line to read.
    """
    named = _one(conn, "SELECT COUNT(*) FROM scip_symbols "
                       "WHERE canonical_id NOT LIKE '%local %'")
    multi = _one(conn, "SELECT COUNT(*) FROM scip_symbols "
                       "WHERE canonical_id NOT LIKE '%local %' "
                       'AND line_end > line_start')
    share = (multi / named) if named else 0.0
    return WiringCheck(
        name='definition_extents_present',
        ok=named > 0 and share >= MIN_MULTI_LINE_SHARE,
        detail=('no named definitions' if named == 0 else
                'ok' if share >= MIN_MULTI_LINE_SHARE else
                'every definition is an identifier span, so nothing can be quoted'),
        measured={'multi_line': multi, 'named': named, 'share': round(share, 4)},
    )


def _implements_edges_present(conn: 'Connection') -> WiringCheck:
    """SCIP's ``is_implementation`` relationships reached the graph.

    ``SymbolInformation.relationships`` is what answers *which implementation runs* at a
    polymorphic call site. Measured before the rebuild: zero readers in the codebase, and
    an edge-type set of exactly ``call`` and ``type_ref``.
    """
    implements = _one(conn, "SELECT COUNT(*) FROM scip_edges "
                            "WHERE edge_type = 'implements'")
    edges = _one(conn, 'SELECT COUNT(*) FROM scip_edges')
    return WiringCheck(
        name='implements_edges_present',
        ok=edges > 0 and implements > 0,
        detail=('no edges' if edges == 0 else
                'ok' if implements > 0 else
                'is_implementation relationships were discarded at ingest'),
        measured={'implements_edges': implements, 'edges': edges},
    )


#: The closed set of ingest invariants. Adding a defect class means adding a check here,
#: so it can never again be found by hand months later.
CHECKS = (
    _local_ids_namespaced,
    _locals_never_cross_sources,
    _definition_extents_present,
    _implements_edges_present,
)


def wiring_report(conn: 'Connection') -> WiringReport:
    """Run every ingest invariant against ``conn`` and report what broke."""
    return WiringReport(checks=[check(conn) for check in CHECKS])
