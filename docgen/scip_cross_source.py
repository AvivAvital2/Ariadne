"""Cross-source SCIP graph (SCIP-everywhere, Phase 2).

Joins multiple sources' SCIP indexes into a single graph keyed on
canonical SCIP symbol IDs. Within each language, symbol strings are
canonical and unambiguous, so cross-source joining is just string
equality on the canonical_id. Cross-language joins do not happen by
construction — ``scip-python python ...`` symbols cannot collide with
``scip-java maven ...`` symbols.

Per design decisions in ``designs/scip-everywhere.md``:

- **No fallbacks** (#4). A source either has a current ``.scip``
  registered via ``add_source()`` or it contributes nothing. There is no
  imports-based degraded mode.
- **Pure consumer** (#2). This module reads ``ScipIndex`` instances; it
  never invokes any indexer.
- **Permissive symbol resolution** (#3). ``resolve_symbol()`` matches
  exact qualified_name → suffix → substring; the algorithm is
  deterministic so the same input + indexed corpus produces the same
  result.

The module exposes three small frozen types — ``CrossSourceSymbol``,
``CrossSourceEdge``, ``SymbolResolution`` — and one builder,
``CrossSourceGraph``. Tests synthesize ``ScipIndex`` instances directly
without touching the protobuf bindings.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from attrs import evolve, field, frozen

from docgen.scip_descriptors import _qualified_name_from_symbol
import logging

if TYPE_CHECKING:
    from sqlite3 import Connection

    from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
_CONFIDENCE_RANK = {'recovered': 0, 'derived': 1, 'resolved': 2, 'exact': 3}
DEFAULT_ASSERT_MIN_CONFIDENCE = 'resolved'
def _resolve_assert_floor() -> str:
    """The configured SQL read-boundary floor (design §3a's
    ``sql_assert_min_confidence``, default ``resolved``) — the single place the
    config key is consulted. An active config that doesn't define the key (e.g. a
    partial test double) resolves to the default, since an unset key means the
    same; the read boundary never hard-depends on the key being present."""
    from config import get_config

    return getattr(get_config(), 'sql_assert_min_confidence',
                   DEFAULT_ASSERT_MIN_CONFIDENCE)


def _floor_rank(min_confidence: 'str | None') -> int:
    """Rank of the effective confidence floor: the explicit ``min_confidence``
    when given, else the configured default (:func:`_resolve_assert_floor`). The
    one place the read boundary (query views + graph projection) resolves its
    floor, so the ``sql_assert_min_confidence`` config key takes effect uniformly."""
    return _CONFIDENCE_RANK[min_confidence or _resolve_assert_floor()]
DEFAULT_MAX_DATA_EDGES = 1_000_000

logger = logging.getLogger(__name__)


# Schema lives at library level (``library_scip.py``) so the slim
# consumer can apply it without pulling in docgen modules at import
# time. CrossSourceGraph populates the tables; slim-side query code
# can read them directly.


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@frozen
class CrossSourceSymbol:
    """A symbol definition in any indexed source.

    ``canonical_id`` is the SCIP wire-format symbol string — globally
    unique by SCIP convention. The other fields are denormalized for
    convenience; they all derive from the SCIP index this symbol came
    from.
    """
    canonical_id: str
    source_name: str
    language: str
    file: str  # relative path within the source root
    line_start: int  # 1-indexed
    line_end: int  # 1-indexed, inclusive
    kind: str  # SCIP SymbolKind name (Class, Method, Field, ...)
    display_name: str
    qualified_name: str
    parent_qualified_name: str | None


@frozen
class CrossSourceEdge:
    """A reference (call/use) from one symbol to another, resolved by
    SCIP. ``confidence`` always reads ``'exact'`` in v1; the field is
    kept as a future-proofing escape hatch."""
    caller: CrossSourceSymbol
    callee: CrossSourceSymbol
    edge_type: str  # 'call' for v1
    file: str  # where the reference appears (relative path)
    line: int  # 1-indexed
    confidence: str = 'exact'


@frozen
class SymbolResolution:
    """Result of ``CrossSourceGraph.resolve_symbol(query)``.

    Exactly one of ``symbol`` or ``candidates`` is non-trivial:
    - Single match: ``symbol`` is set, ``candidates`` is empty,
      ``match_tier`` indicates how it matched.
    - Ambiguous: ``symbol`` is None, ``candidates`` lists all best-tier
      candidates, ``match_tier`` describes the tier that produced them.
    - No match: both ``symbol`` and ``candidates`` are None/empty,
      ``match_tier`` is ``'none'``.
    """
    symbol: CrossSourceSymbol | None
    candidates: tuple[CrossSourceSymbol, ...] = ()
    match_tier: str = 'none'  # 'exact' | 'suffix' | 'substring' | 'none'
@frozen
class SharedDatabase:
    """An explicit declaration (config, §6) that the named sources share one
    physical database, so their identically-located table/column is the SAME
    node. Cross-source column identity is emitted ONLY through this gate, never
    a name match. ``database``/``db_schema`` optionally narrow which physical
    schema is shared."""
    sources: frozenset
    database: str | None = None
    db_schema: str | None = None


def shared_databases_from_config(raw) -> 'list[SharedDatabase]':
    """Parse the ``shared_database`` config block (a list of dicts) into
    declarations; optional ``database``/``schema`` keys default to None.
    None or an empty list yields no declarations (the gate stays shut)."""
    return [
        SharedDatabase(
            sources=frozenset(entry['sources']),
            database=entry.get('database'),
            db_schema=entry.get('schema'),
        )
        for entry in raw or []
    ]


def _shared_node_id(source_name, database, db_schema, table_name, column_name,
                    declarations):
    """Cross-source column identity (§6, opt-in): if a ``shared_database``
    declaration names ``source_name`` and its optional database/schema match
    this row, return a SOURCE-INDEPENDENT canonical id so the identically-
    located table/column in every member source collapses to one node. Else
    None — columns stay per-source (the default; the coupling is never inferred
    from a name match)."""
    for decl in declarations:
        if source_name not in decl.sources:
            continue
        if decl.database is not None and decl.database != (database or None):
            continue
        if decl.db_schema is not None and decl.db_schema != (db_schema or None):
            continue
        key = '+'.join(sorted(decl.sources))
        col = f'#{column_name}' if column_name else ''
        return f'data sql @shared:{key} {database or "_"}.{db_schema or "_"}.{table_name}{col}'
    return None


# ---------------------------------------------------------------------------
# CrossSourceGraph
# ---------------------------------------------------------------------------


@frozen(eq=False)
class _SourceEntry:
    name: str
    index: 'ScipIndex'
    language: str


class CrossSourceGraph:
    """Cross-source SCIP graph builder.

    Lifecycle:
        graph = CrossSourceGraph()
        graph.add_source('scalaproject', index=..., language='python')
        graph.add_source('biggerproject-backend', index=..., language='python')
        graph.materialize()
        graph.consumers_of_source('scalaproject')
        graph.callers_of(canonical_id)
        graph.resolve_symbol('LicenseService.validate_token')

    ``materialize()`` is idempotent and may be called any time after
    ``add_source()`` calls — it rebuilds from the current registered
    sources.
    """

    def __init__(self) -> None:
        # Multiple indexer runs can register under the same source_name
        # (per design decision #7: a polyglot source like scalaproject has
        # scip-java + scip-python + scip-typescript outputs that all
        # belong to one logical source). Hence dict-of-lists.
        self._sources: dict[str, list[_SourceEntry]] = {}
        self._symbols: dict[str, CrossSourceSymbol] = {}
        self._edges: list[CrossSourceEdge] = []
        # Source names known to the graph — either via add_source (with
        # ScipIndex) or via load_from (DB-loaded, no live index).
        # has_scip checks this set so load-from-DB callers can query
        # without re-registering sources.
        self._known_source_names: set[str] = set()
        self._rst_autodoc: dict[str, list[str]] = {}
        self._edges_by_callee: dict = {}
        self._edges_by_caller: dict = {}
        # Dirty by default: a graph built by assigning _edges directly (e.g.
        # catalog_enrich) must still rebuild the index on the first query.
        self._edge_index_dirty: bool = True

    # -- registration -----------------------------------------------------

    def add_source(
        self,
        source_name: str,
        *,
        index: 'ScipIndex',
        language: str,
    ) -> None:
        """Register a source's loaded ScipIndex.

        Per decision #4, only sources registered here participate in
        cross-source features. A source whose ``.scip`` is missing or
        stale should fail loudly upstream (in ``ScipIndex.load``); it
        must NOT be passed to this method as a degraded entry.
        """
        self._sources.setdefault(source_name, []).append(
            _SourceEntry(
                name=source_name, index=index, language=language,
            ),
        )
        self._known_source_names.add(source_name)

    def has_scip(self, source_name: str) -> bool:
        """True iff a source has been seen by the graph — either
        registered via ``add_source`` (with a live ScipIndex) or loaded
        from the DB via ``load_from``."""
        return source_name in self._known_source_names

    # -- materialization --------------------------------------------------

    def materialize(self) -> None:
        """Rebuild the symbol index and edge list from registered sources.

        Two passes:
        1. Collect every definition occurrence as a CrossSourceSymbol,
           keyed by canonical_id.
        2. For every non-definition occurrence whose target is a known
           symbol, resolve the caller (tightest enclosing definition in
           the same document) and emit one CrossSourceEdge.

        References whose target is not in the registered set are
        silently dropped (the target lives in an unindexed source — per
        decision #4, no fallback edge is generated).
        """
        self._symbols = {}
        self._edges = []

        # Pass 1: collect definitions across every registered indexer
        # (a polyglot source has multiple entries under one name).
        for entries in self._sources.values():
            for entry in entries:
                for doc in entry.index.documents:
                    self._collect_definitions(doc, entry)

        # Pass 2: emit reference edges
        for entries in self._sources.values():
            for entry in entries:
                for doc in entry.index.documents:
                    self._collect_edges(doc, entry)
        self._edge_index_dirty = True

    def _collect_definitions(
        self, doc: '_ScipDoc', entry: _SourceEntry,
    ) -> None:
        """For each definition occurrence in ``doc``, build and store a
        CrossSourceSymbol.

        Parameter-descriptor definitions (scip-python's
        ``Foo#bar().(self)``) are skipped — they're not standalone
        callable definitions and ingesting them as ``CrossSourceSymbol``
        rows pollutes the resolver's substring tier and collides with
        same-named nested-method QNs. The parser still recognizes them
        (so chained descriptors don't truncate); they just aren't
        promoted to graph nodes.
        """
        from docgen.scip_descriptors import _parse_descriptors

        symbol_meta = {s.symbol: s for s in doc.symbols}

        for occ in doc.occurrences:
            if not occ.is_definition:
                continue
            # Skip parameter symbols. The cheap prefilter: a parameter
            # descriptor ``(name)`` makes the symbol end with ``)``.
            if occ.symbol.endswith(')'):
                descriptors = occ.symbol.rsplit(' ', 1)[-1]
                parsed = _parse_descriptors(descriptors)
                if parsed and parsed[-1][1] == 'parameter':
                    continue
            meta = symbol_meta.get(occ.symbol)
            line_start, line_end = _occ_line_range_1indexed(occ)
            qn, parent_qn = _qualified_name_from_symbol(
                occ.symbol, entry.language,
            )
            self._symbols[occ.symbol] = CrossSourceSymbol(
                canonical_id=occ.symbol,
                source_name=entry.name,
                language=entry.language,
                file=doc.relative_path,
                line_start=line_start,
                line_end=line_end,
                kind=meta.kind if meta else '',
                display_name=meta.display_name if meta else _last_descriptor(occ.symbol),
                qualified_name=qn,
                parent_qualified_name=parent_qn,
            )

    def _collect_edges(
        self, doc: '_ScipDoc', entry: _SourceEntry,
    ) -> None:
        """For each non-definition occurrence in ``doc``, find the
        callee (lookup in self._symbols) and the caller (tightest
        enclosing definition in this doc)."""
        # Pre-compute per-doc list of (line_start, line_end, symbol) for
        # caller resolution.
        defs_in_doc: list[tuple[int, int, str]] = []
        for occ in doc.occurrences:
            if occ.is_definition:
                ls, le = _occ_line_range_1indexed(occ)
                defs_in_doc.append((ls, le, occ.symbol))

        for occ in doc.occurrences:
            if occ.is_definition:
                continue
            callee = self._symbols.get(occ.symbol)
            if callee is None:
                # Target lives in an unindexed source — drop per decision #4
                continue

            ref_line = _occ_line_start_1indexed(occ)
            caller_id = _tightest_enclosing(defs_in_doc, ref_line)
            if caller_id is None:
                # Reference at file scope (e.g., import line) — no
                # enclosing function/method/class. Skip; reverse-augment
                # only cares about edges with a defined caller.
                continue
            caller = self._symbols.get(caller_id)
            if caller is None:
                continue

            # Skip self-references — a defining occurrence can have
            # additional non-definition occurrences at the same site for
            # supplementary roles (read/write). We only want true
            # call/use edges that change the caller-callee relationship.
            if caller.canonical_id == callee.canonical_id:
                continue

            self._edges.append(CrossSourceEdge(
                caller=caller,
                callee=callee,
                edge_type='call',
                file=doc.relative_path,
                line=ref_line,
            ))

    # -- queries ----------------------------------------------------------

    def consumers_of_source(
        self, source_name: str,
    ) -> list[CrossSourceEdge]:
        """All edges where the callee is in ``source_name`` and the
        caller is in a different source. This is the input to the
        reverse-augment phase."""
        return [
            e for e in self._edges
            if e.callee.source_name == source_name
            and e.caller.source_name != source_name
        ]
    def callers_of(self, symbol_id: str) -> 'list[CrossSourceEdge]':
        """All edges referencing this symbol. Same-source edges included —
    caller_of is a graph query, not a cross-source query."""
        if self._edge_index_dirty:
            self._rebuild_edge_index()
        return self._edges_by_callee.get(symbol_id, [])
    
    
    def callees_of(self, symbol_id: str) -> 'list[CrossSourceEdge]':
        """All edges originating from this symbol."""
        if self._edge_index_dirty:
            self._rebuild_edge_index()
        return self._edges_by_caller.get(symbol_id, [])
    
    
    def _rebuild_edge_index(self) -> None:
        """(Re)build the endpoint indexes from ``_edges`` (design §6): O(1) per
    hop instead of an O(E) scan, important because data edges are plausibly
    the most numerous class. The flat ``_edges`` list stays for source-scoped
    queries; this is rebuilt lazily after any edge mutation marks it dirty."""
        self._edges_by_callee = {}
        self._edges_by_caller = {}
        for edge in self._edges:
            self._edges_by_callee.setdefault(edge.callee.canonical_id, []).append(edge)
            self._edges_by_caller.setdefault(edge.caller.canonical_id, []).append(edge)
        self._edge_index_dirty = False

    def edges_in_source(self, source_name: str) -> list[CrossSourceEdge]:
        """All edges where BOTH caller and callee are in
        ``source_name`` — within-source edges only.

        Used by cli_graph (Phase 5) to replace ast-grep-derived
        approximate per-source edges with SCIP-precise ones for
        sources that have a current ``.scip``."""
        return [
            e for e in self._edges
            if e.caller.source_name == source_name
            and e.callee.source_name == source_name
        ]

    def symbols_in(
        self, source_name: str,
    ) -> tuple[CrossSourceSymbol, ...]:
        """All defined symbols in a registered source."""
        return tuple(
            s for s in self._symbols.values() if s.source_name == source_name
        )

    def symbols_with_zero_references(
        self, source_name: str,
    ) -> list[CrossSourceSymbol]:
        """Symbols defined in ``source_name`` that are never referenced
        anywhere in the registered corpus. Used as input to
        ``ariadne improve --dead-code``."""
        referenced = {e.callee.canonical_id for e in self._edges}
        return [
            s for s in self._symbols.values()
            if s.source_name == source_name and s.canonical_id not in referenced
        ]

    # -- persistence (Phase 2f) -------------------------------------------

    def save_to(self, conn: 'Connection') -> None:
        """Write the materialized state to ariadne.db.

        For every source registered via ``add_source``, prior rows in
        ``scip_symbols`` and ``scip_edges`` for that source are deleted
        before the current state is written. Sources NOT in the current
        registration are left untouched, so re-saving one source does
        not wipe data for another.

        Schema must already exist; call ``init_scip_schema(conn)`` (or
        rely on Library's __attrs_post_init__) before invoking this.
        """
        registered = tuple(self._sources.keys())
        if not registered:
            return

        placeholders = ','.join('?' for _ in registered)
        # Delete edges FIRST, while scip_symbols still has the rows
        # the join uses to identify which edges belong to these sources.
        # Reversing this order silently leaks orphan edges through
        # re-saves: the join returns empty after the symbols delete and
        # the edges DELETE matches nothing.
        conn.execute(
            f'DELETE FROM scip_edges '
            f'WHERE caller_canonical_id IN '
            f'  (SELECT canonical_id FROM scip_symbols '
            f'   WHERE source_name IN ({placeholders})) '
            f'OR callee_canonical_id IN '
            f'  (SELECT canonical_id FROM scip_symbols '
            f'   WHERE source_name IN ({placeholders}))',
            registered + registered,
        )
        # Now safe to delete the symbols themselves.
        conn.execute(
            f'DELETE FROM scip_symbols WHERE source_name IN ({placeholders})',
            registered,
        )

        # Insert symbols owned by registered sources
        for sym in self._symbols.values():
            if sym.source_name not in self._sources:
                continue
            conn.execute(
                'INSERT OR REPLACE INTO scip_symbols VALUES '
                '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    sym.canonical_id, sym.source_name, sym.language,
                    sym.file, sym.line_start, sym.line_end,
                    sym.kind, sym.display_name,
                    sym.qualified_name, sym.parent_qualified_name,
                ),
            )

        # Insert edges where at least one endpoint is in a registered
        # source (we don't double-write edges between two unregistered
        # sources, but materialize() doesn't produce those anyway).
        for edge in self._edges:
            if (
                edge.caller.source_name not in self._sources
                and edge.callee.source_name not in self._sources
            ):
                continue
            conn.execute(
                'INSERT OR REPLACE INTO scip_edges VALUES '
                '(?, ?, ?, ?, ?, ?)',
                (
                    edge.caller.canonical_id, edge.callee.canonical_id,
                    edge.edge_type, edge.file, edge.line, edge.confidence,
                ),
            )

    def load_from(self, conn: 'Connection') -> None:
        """Replace in-memory state with what's in ariadne.db.

        Reads every row from ``scip_symbols`` and ``scip_edges`` and
        rebuilds the internal indexes. Useful for query-time tools
        (``ariadne callers``, etc.) that don't have the original SCIP
        files, only the cached graph in the DB.
        """
        self._symbols = {}
        self._edges = []
        self._sources = {}
        self._known_source_names = set()

        for row in conn.execute(
            'SELECT canonical_id, source_name, language, file, '
            '       line_start, line_end, kind, display_name, '
            '       qualified_name, parent_qualified_name '
            'FROM scip_symbols',
        ):
            sym = CrossSourceSymbol(
                canonical_id=row[0],
                source_name=row[1],
                language=row[2],
                file=row[3],
                line_start=row[4],
                line_end=row[5],
                kind=row[6],
                display_name=row[7],
                qualified_name=row[8],
                parent_qualified_name=row[9],
            )
            self._symbols[sym.canonical_id] = sym
            self._known_source_names.add(sym.source_name)

        for row in conn.execute(
            'SELECT caller_canonical_id, callee_canonical_id, edge_type, '
            '       file, line, confidence '
            'FROM scip_edges',
        ):
            caller = self._symbols.get(row[0])
            callee = self._symbols.get(row[1])
            if caller is None or callee is None:
                # Orphan edge — symbol referenced but not stored. Skip.
                continue
            self._edges.append(CrossSourceEdge(
                caller=caller, callee=callee,
                edge_type=row[2], file=row[3], line=row[4],
                confidence=row[5],
            ))
        self._rst_autodoc = {}
        for row in conn.execute(
            'SELECT symbol_qualified_name, rst_section_qualified_name '
            'FROM rst_autodoc_links',
        ):
            self._rst_autodoc.setdefault(row[0], []).append(row[1])
        self._edge_index_dirty = True
    
    
    def add_data_layer(self, conn, min_confidence=None,
                       shared_database=None, max_data_edges=DEFAULT_MAX_DATA_EDGES):
        """Project the SQL data model into the graph as ordinary nodes and
    edges (design §6).

    ``schema_symbols`` rows become ``CrossSourceSymbol`` nodes
    (``kind='Table'|'Column'``, ``language='sql'``); each ``data_access``
    row becomes a ``CrossSourceEdge`` carrying the access **role** as
    ``edge_type`` (``'filter'``/``'project'``/``'write'``/…), and each
    code-first node's ``producer_symbol_id`` becomes a ``'maps_to'`` edge.
    Edges are registered with the data node as the **callee** (the
    app/model symbol as caller), so ``callers_of(table|column)`` finds
    every access AND the maps_to link — which is exactly what reverse
    ``impact_radius`` walks.

    Only facts at/above ``min_confidence`` on the shared ladder
    (``exact > resolved > derived > recovered``, default ``resolved``) are
    asserted; weaker facts are held back as gaps, never projected into the
    graph (design §3a/§6a — the read-boundary safety valve).

    Must run after ``load_from``: it resolves consumer/producer app
    symbols against the already-loaded ``scip_symbols``. The SCIP
    definition pass ingests SCIP occurrences only, so data nodes are
    registered here, alongside it — not through it.

    Cross-source column identity (§6) is opt-in: when ``shared_database``
    names the sources that share one physical schema, their identically-
    located table/column collapses to ONE source-independent node, so an
    A-writes / B-reads coupling traverses in a single walk; absent a
    declaration each source's column stays distinct (never name-matched).

    ``max_data_edges`` bounds the data layer held in ``self._edges`` for
    ``impact_radius`` (design §9a): once that many access edges are
    projected, projection stops and a warning is logged rather than
    truncating silently. Returns the count of access edges projected.
    """
        _kind = {
            'table': 'Table', 'column': 'Column',
            'view': 'View', 'index': 'Index',
        }
        floor = _floor_rank(min_confidence)
        declarations = shared_database or []

        def _asserts(confidence):
            return _CONFIDENCE_RANK.get(confidence, -1) >= floor

        # Per-source canonical_id -> fused node id, for the opt-in gate below.
        remap: dict = {}

        # Pass 1: nodes (+ the maps_to edge from a code-first producer).
        for row in conn.execute(
            'SELECT canonical_id, source_name, node_type, database, '
            '       db_schema, table_name, column_name, producer_symbol_id, '
            '       confidence '
            'FROM schema_symbols',
        ):
            (cid, source_name, node_type, database, db_schema,
             table_name, column_name, producer_id, confidence) = row
            if not _asserts(confidence):
                continue
            db = database or '_'
            schema = db_schema or '_'
            table_qn = f'{db}.{schema}.{table_name}'
            qualified_name = table_qn + (f'#{column_name}' if column_name else '')
            # Opt-in cross-source identity: a declared shared database collapses
            # the column to one source-independent node; else it stays per-source.
            shared = _shared_node_id(
                source_name, database, db_schema, table_name, column_name,
                declarations)
            node_cid = shared or cid
            if shared is not None:
                remap[cid] = shared
            # A code-first node anchors at its producer symbol's range;
            # SQL-first nodes have no source line yet (sentinel 0).
            producer = self._symbols.get(producer_id) if producer_id else None
            node = CrossSourceSymbol(
                canonical_id=node_cid,
                source_name=source_name,
                language='sql',
                file=producer.file if producer else '',
                line_start=producer.line_start if producer else 0,
                line_end=producer.line_start if producer else 0,
                kind=_kind.get(node_type, node_type),
                display_name=column_name or table_name,
                qualified_name=qualified_name,
                parent_qualified_name=table_qn if column_name else None,
            )
            self._symbols[node_cid] = node
            self._known_source_names.add(source_name)
            if producer is not None:
                self._edges.append(CrossSourceEdge(
                    caller=producer, callee=node, edge_type='maps_to',
                    file=producer.file, line=producer.line_start,
                    confidence=confidence,
                ))

        # Pass 2: access edges (nodes from pass 1 are now resolvable).
        projected = 0
        for row in conn.execute(
            'SELECT consumer_symbol_id, schema_symbol_id, role, '
            '       call_site_file, call_site_line, confidence '
            'FROM data_access',
        ):
            consumer_id, schema_id, role, cs_file, cs_line, confidence = row
            if not _asserts(confidence):
                continue
            consumer = self._symbols.get(consumer_id)
            node = self._symbols.get(remap.get(schema_id, schema_id))
            if consumer is None or node is None:
                # Orphan: app symbol or data node absent. Skip, as
                # load_from does for orphan SCIP edges.
                continue
            if projected >= max_data_edges:
                logger.warning(
                    'data-edge budget reached (%d): stopping data-layer projection; '
                    'impact_radius may be incomplete. Raise max_data_edges (or scope '
                    'the query) to project the rest.', max_data_edges)
                break
            self._edges.append(CrossSourceEdge(
                caller=consumer, callee=node, edge_type=role,
                file=cs_file or '', line=cs_line or 0,
                confidence=confidence,
            ))
            projected += 1
        self._edge_index_dirty = True
        return projected

    # -- symbol resolution (decision #3) ----------------------------------

    def resolve_symbol(self, query: str) -> SymbolResolution:
        """Permissive symbol lookup: exact qualified_name → suffix →
        substring. Returns the unique match if any tier finds one;
        otherwise the candidate list at the best-matching tier; or
        a 'none' result if the query matches nothing.

        Ties are broken deterministically by ``(qualified_name,
        canonical_id)`` ascending, so identical inputs across runs
        return identical outputs.
        """
        all_symbols = list(self._symbols.values())

        # Tier 1: exact qualified_name match
        exact = [s for s in all_symbols if s.qualified_name == query]
        if len(exact) == 1:
            return SymbolResolution(symbol=exact[0], match_tier='exact')
        if len(exact) > 1:
            return SymbolResolution(
                symbol=None,
                candidates=tuple(_sorted_candidates(exact)),
                match_tier='exact',
            )

        # Tier 2: suffix match. The query must match a trailing
        # ``.``-bounded segment of the qualified_name.
        def is_suffix(query: str, qn: str) -> bool:
            if qn == query:
                return True
            return qn.endswith('.' + query)

        suffix = [s for s in all_symbols if is_suffix(query, s.qualified_name)]
        if len(suffix) == 1:
            return SymbolResolution(symbol=suffix[0], match_tier='suffix')
        if len(suffix) > 1:
            return SymbolResolution(
                symbol=None,
                candidates=tuple(_sorted_candidates(suffix)),
                match_tier='suffix',
            )

        # Tier 3: substring match against qualified_name OR display_name
        substring = [
            s for s in all_symbols
            if query in s.qualified_name or query in s.display_name
        ]
        if len(substring) == 1:
            return SymbolResolution(symbol=substring[0], match_tier='substring')
        if len(substring) > 1:
            return SymbolResolution(
                symbol=None,
                candidates=tuple(_sorted_candidates(substring)),
                match_tier='substring',
            )

        return SymbolResolution(symbol=None, candidates=(), match_tier='none')
    
    def rst_sections_documenting(self, symbol_qualified_name: str) -> tuple[str, ...]:
        """rst section qualified-names that document ``symbol_qualified_name``
    (reverse autodoc lookup, ingested by :meth:`load_from`). () when none.
    """
        return tuple(sorted(self._rst_autodoc.get(symbol_qualified_name, ())))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _occ_line_range_1indexed(occ: '_ScipOccurrence') -> tuple[int, int]:
    """Convert SCIP wire range (0-indexed) → Ariadne 1-indexed
    ``(line_start, line_end)``. Handles 3-tuple (same line) and 4-tuple
    (multi-line) forms.
    """
    r = list(occ.range)
    if len(r) == 3:
        return (r[0] + 1, r[0] + 1)
    return (r[0] + 1, r[2] + 1)


def _occ_line_start_1indexed(occ: '_ScipOccurrence') -> int:
    """1-indexed start line of an occurrence."""
    return int(occ.range[0]) + 1


def _tightest_enclosing(
    defs: list[tuple[int, int, str]], ref_line: int,
) -> str | None:
    """Find the symbol of the tightest definition whose 1-indexed range
    contains ``ref_line``. Returns None if no definition contains it.

    ``defs`` is a list of ``(line_start, line_end, symbol)`` tuples; we
    pick the entry with the largest ``line_start`` whose
    ``[line_start, line_end]`` interval includes ``ref_line``. The
    largest start = innermost scope.
    """
    best: tuple[int, str] | None = None
    for ls, le, sym in defs:
        if ls <= ref_line <= le:
            if best is None or ls > best[0]:
                best = (ls, sym)
    return best[1] if best is not None else None


def _last_descriptor(symbol: str) -> str:
    """Best-effort display-name fallback when SymbolInformation is
    missing. Strip the trailing kind suffix and return the last
    descriptor's name; keeps the resolver's display matching usable
    even with under-populated indexes.
    """
    # Take the descriptor segment (after the last whitespace token).
    descriptors = symbol.rsplit(' ', 1)[-1]
    # Strip trailing kind char if present
    if descriptors and descriptors[-1] in '/.#:':
        descriptors = descriptors[:-1]
    # Strip method disambiguator like '()' from the tail
    if descriptors.endswith('()'):
        descriptors = descriptors[:-2]
    # The last name is whatever follows the last delimiter.
    for sep in '/#.:':
        idx = descriptors.rfind(sep)
        if idx >= 0:
            descriptors = descriptors[idx + 1:]
            break
    return descriptors


def _sorted_candidates(
    symbols: list[CrossSourceSymbol],
) -> list[CrossSourceSymbol]:
    """Sort candidates by (qualified_name, canonical_id) for
    deterministic output order."""
    return sorted(symbols, key=lambda s: (s.qualified_name, s.canonical_id))


# ---------------------------------------------------------------------------
# Manifest loader (Phase 2g) — bridges ariadne discover's manifest.json to
# the cross-source graph.
# ---------------------------------------------------------------------------


# Kind in manifest.json → language for CrossSourceGraph.add_source.
# scip-java handles both Scala and Java; we default to 'scala' since the
# QN derivation behavior is identical across both for non-overloaded
# callables (the only place language matters in _qualified_name_from_symbol
# is the JVM signature decoding, which only fires for non-empty disambigs).
_KIND_TO_LANGUAGE = {
    'java': 'scala',
    'python': 'python',
    'typescript': 'javascript',
}


def load_source_from_manifest(
    graph: CrossSourceGraph,
    source_name: str,
    source_root: Path,
    *,
    max_staleness_days: int = 7,
    index_factory=None,
) -> None:
    """Read ``<source_root>/.ariadne/manifest.json`` and register each
    declared SCIP index with the graph under ``source_name``.

    Per design decision #5, missing or stale ``.scip`` files raise
    structured errors (``ScipUnavailableError`` / ``ScipTooStaleError``)
    propagated from ``ScipIndex.load``. No silent fallback.

    Manifest entries without a ``scip_path`` are skipped — discovery
    has identified the location but ``ariadne index`` hasn't produced
    the artifact yet.

    ``index_factory`` is a dependency-injection point for tests; pass
    a callable matching ``ScipIndex.load`` to substitute synthetic
    indexes without touching disk. Defaults to the real loader.
    """
    import json

    from docgen.scip_extractor import ScipIndex

    if index_factory is None:
        index_factory = ScipIndex.load

    manifest_path = source_root / '.ariadne' / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'manifest not found at {manifest_path} — run '
            f'`ariadne discover --source {source_name}` first'
        )

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    for entry in manifest.get('indexers', ()):
        scip_relative = entry.get('scip_path')
        if not scip_relative:
            continue
        scip_path = source_root / '.ariadne' / scip_relative
        kind = entry.get('kind')
        language = _KIND_TO_LANGUAGE.get(kind)
        if language is None:
            raise ValueError(
                f"manifest for '{source_name}' contains unknown indexer "
                f"kind: {kind!r}"
            )
        index = index_factory(
            scip_path,
            repo=source_name,
            max_staleness_days=max_staleness_days,
        )

        # Phase 2h: translate Vue extracted-companion paths back to
        # the original .vue files when a vue-mapping.json is declared.
        vue_mapping_relative = entry.get('vue_mapping')
        if vue_mapping_relative:
            mapping_path = source_root / '.ariadne' / vue_mapping_relative
            if mapping_path.exists():
                from docgen.scip_extractor import apply_vue_mapping
                vue_mapping = json.loads(
                    mapping_path.read_text(encoding='utf-8'),
                )
                index = apply_vue_mapping(index, vue_mapping)

        # The .scip's document paths are relative to the indexer's cwd (the
        # package root), not the .scip file's own directory — so set source_root
        # to <repo>/<cwd> for source-file reads (ORM strategies etc.) to resolve
        # on multi-package repos (§5.0.1; the loader default is the .scip's dir).
        index = evolve(index, source_root=source_root / entry.get('cwd', '.'))
        graph.add_source(source_name, index=index, language=language)


__all__ = [
    'CrossSourceEdge',
    'CrossSourceGraph',
    'CrossSourceSymbol',
    'SymbolResolution',
    'load_source_from_manifest',
]
