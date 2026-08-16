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
import re
from docgen.scip_graph import (  # noqa: F401  (re-exported for callers)
    CrossSourceEdge,
    CrossSourceSymbol,
    classify_edge,
)

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
_BARE_LOCAL = re.compile(r'^local \d+$')


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
@frozen
class ReachSite:
    """One place a consumer artifact reaches INTO a target (spool): the calling
    symbol + file:line, and the edge confidence (``'resolved'`` = bridged by
    qualified-name moniker resolution across the store; ``'exact'`` = same
    canonical id)."""
    consumer_source: str
    caller: str
    file: str
    line: int
    confidence: str
@frozen
class ReachFinding:
    """One reached spool symbol: WHERE the consumer calls it (``sites``) plus the
    spool docs documenting it (``doc_ids``/``doc_titles`` — the WHAT-knowledge).
    The deterministic unit the reach-knowledge synthesis turns into
    "what breaks / what to tune on version X, and where"."""
    symbol: str
    qualified_name: "str | None"
    sites: tuple
    doc_ids: tuple
    doc_titles: tuple


def build_reach_findings(reach_result, symbols, find_docs):
    """Pair reach sites (from :meth:`CrossSourceGraph.reach_into`) with the docs
    documenting each reached symbol.

    ``reach_result``: ``{symbol_canonical_id: [ReachSite, ...]}``.
    ``symbols``: ``{canonical_id: CrossSourceSymbol}`` (e.g. ``graph._symbols``).
    ``find_docs``: ``callable(file) -> [doc]`` where each doc has ``.id`` and
    ``.title`` (e.g. a per-file ``find_documents_by_source_files``).

    Pure — no graph or DB access — so it unit-tests cleanly and the doc-join
    granularity (file-level today; symbol-level later) can evolve without
    touching the graph. A symbol with no documenting doc is still reported (we
    know WHERE even when there's no knowledge doc — the honest-gap surfaces
    downstream)."""
    findings = []
    for symbol_id, sites in reach_result.items():
        sym = symbols.get(symbol_id)
        file = getattr(sym, "file", None)
        docs = find_docs(file) if file else []
        findings.append(ReachFinding(
            symbol=symbol_id,
            qualified_name=getattr(sym, "qualified_name", None),
            sites=tuple(sites),
            doc_ids=tuple(d.id for d in docs),
            doc_titles=tuple(d.title for d in docs),
        ))
    return findings


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
        # producer canonical_id -> [consumer CrossSourceSymbol] (HTTP tier),
        # for cross-language blast radius.
        self._http_consumers: dict[str, list] = {}
        self._edges_by_callee: dict = {}
        self._edges_by_caller: dict = {}
        # Dirty by default: a graph built by assigning _edges directly (e.g.
        # catalog_enrich) must still rebuild the index on the first query.
        self._edge_index_dirty: bool = True
        self._resolve_external_to: frozenset = frozenset()
        self._qn_index: dict = {}
    
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
                name=source_name, index=index.scoped_to(source_name), language=language,
            ),
        )
        self._known_source_names.add(source_name)

    def has_scip(self, source_name: str) -> bool:
        """True iff a source has been seen by the graph — either
        registered via ``add_source`` (with a live ScipIndex) or loaded
        from the DB via ``load_from``."""
        return source_name in self._known_source_names
    def materialize(self, resolve_external_to=None) -> None:
        """Build symbols and edges from every registered source.

    Two phases, because external resolution needs the whole picture: symbols first across
    every source, then edges resolved against that complete set. Both phases delegate to
    ``docgen.scip_graph``, where identity, extents, edge typing and ``implements`` are
    settled once and cannot drift per consumer.

    ``resolve_external_to`` names sources a dropped reference may resolve INTO, matched by
    qualified name — the bridge for a repo whose moniker differs from a spool's definition
    only by package or version, so the canonical id misses where the name matches.
    """
        from docgen.scip_graph import build_edges, build_symbols

        self._symbols = {}
        self._edges = []
        self._unresolved_callees = 0
        self._unattributed_sites = 0
        self._resolve_external_to = frozenset(resolve_external_to or ())

        for entries in self._sources.values():
            for entry in entries:
                self._symbols.update(build_symbols(
                    entry.index, source_name=entry.name, language=entry.language))

        # The qualified-name index the external resolver matches against, built only over
        # the sources resolution is allowed to reach into.
        self._qn_index = {}
        if self._resolve_external_to:
            for symbol in self._symbols.values():
                if symbol.source_name in self._resolve_external_to:
                    self._qn_index.setdefault(symbol.qualified_name, []).append(symbol)

        resolver = self._resolve_external if self._resolve_external_to else None
        for entries in self._sources.values():
            for entry in entries:
                edges, unresolved, unattributed = build_edges(
                    entry.index, source_name=entry.name, language=entry.language,
                    symbols=self._symbols, resolve_external=resolver)
                self._edges.extend(edges)
                self._unresolved_callees += unresolved
                self._unattributed_sites += unattributed
        self._edge_index_dirty = True

    def _resolve_external(self, symbol, language):
        """Resolve an external-reference moniker to a UNIQUE definition in the
        resolvable sources, matched by qualified name. Returns the
        CrossSourceSymbol or None (resolution disabled / no qn / no match /
        ambiguous). Matching by qualified name rather than canonical id is what
        bridges a repo's package- and version-specific moniker to the spool's
        definition of the same symbol; the unique-match guard keeps unrelated
        same-named symbols from inventing an edge."""
        if not self._resolve_external_to:
            return None
        qn, _parent = _qualified_name_from_symbol(symbol, language)
        if not qn:
            return None
        candidates = self._qn_index.get(qn)
        if candidates and len(candidates) == 1:
            return candidates[0]
        return None

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
    
    
    def http_consumers_of(self, producer_id: str) -> list:
        """Client symbols that consume an endpoint PRODUCED by ``producer_id``
        — the reverse of trace_flow's forward HTTP hop. Empty when the symbol
        produces no consumed endpoint. Powers cross-language blast radius."""
        return self._http_consumers.get(producer_id, [])

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

        # Insert symbols owned by registered sources. Columns are NAMED: this was a
        # positional `VALUES (?, ?, ...)`, which is correct only while the tuple and the
        # table agree on order and arity, and writes values into the wrong fields the
        # moment a column is added anywhere but the end.
        for sym in self._symbols.values():
            if sym.source_name not in self._sources:
                continue
            conn.execute(
                'INSERT OR REPLACE INTO scip_symbols '
                '(canonical_id, source_name, language, file, line_start, line_end, '
                ' kind, display_name, qualified_name, parent_qualified_name) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
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
                'INSERT OR REPLACE INTO scip_edges '
                '(caller_canonical_id, callee_canonical_id, edge_type, file, line, '
                ' confidence) '
                'VALUES (?, ?, ?, ?, ?, ?)',
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

        # HTTP tier: producer symbol -> consumer symbols. Changing a producer
        # affects every client bound to its endpoint (reverse of trace_flow's
        # forward hop). Best-effort — tables may be absent in older DBs.
        self._http_consumers = {}
        try:
            for producer, consumer in conn.execute(
                'SELECT ae.producer_symbol_id, ac.consumer_symbol_id '
                'FROM api_endpoints ae '
                'JOIN api_calls ac ON ac.endpoint_id = ae.endpoint_id '
                'WHERE ae.producer_symbol_id IS NOT NULL',
            ):
                csym = self._symbols.get(consumer)
                if csym is not None:
                    self._http_consumers.setdefault(producer, []).append(csym)
        except Exception:
            pass
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

    def reach_into(self, target_sources, *, from_sources=None, resolved_only=False):
        """Cross-source edges reaching INTO ``target_sources`` (a spool),
        grouped by the target symbol the consumer calls.

        Returns ``{target_symbol_canonical_id: [ReachSite, ...]}``. Only edges
        whose caller is OUTSIDE the targets count (a consumer reaching in), so a
        target's internal edges are excluded. ``from_sources`` restricts to
        specific consumer sources; ``resolved_only`` keeps only moniker-resolved
        edges (the cross-store reach, not same-canonical-id matches). This is the
        graph-driven "where" the env-bridge and reach-knowledge synthesis stand on.
        """
        targets = frozenset(target_sources)
        out = {}
        for edge in self._edges:
            if edge.callee.source_name not in targets:
                continue
            if edge.caller.source_name in targets:
                continue
            if from_sources is not None and edge.caller.source_name not in from_sources:
                continue
            if resolved_only and edge.confidence != 'resolved':
                continue
            out.setdefault(edge.callee.canonical_id, []).append(ReachSite(
                consumer_source=edge.caller.source_name,
                caller=edge.caller.canonical_id,
                file=edge.file,
                line=edge.line,
                confidence=edge.confidence,
            ))
        return out


@frozen
class ImpactReport:
    """Aggregated reverse-edge walk: which files and symbols are affected if the
    starting symbol changes.
    """
    start_symbol: object  # CrossSourceSymbol
    affected_symbols: list  # list[CrossSourceSymbol] in walk order
    files: set[str] = field(factory=set)


def compute_impact_radius(
    graph: 'CrossSourceGraph', start_id: str, depth: int,
) -> ImpactReport:
    """Walk reverse edges N-deep, collecting every symbol and file that
    transitively depends on the starting symbol.
    """
    if start_id not in graph._symbols:
        # Empty report — caller decides whether to error
        return ImpactReport(start_symbol=None, affected_symbols=[], files=set())

    start_sym = graph._symbols[start_id]

    visited: set[str] = set()
    # The starting symbol IS affected — it's the thing being changed.
    affected: list = [start_sym]

    def _walk(sym_id: str, d: int) -> None:
        if d <= 0 or sym_id in visited:
            return
        visited.add(sym_id)
        for edge in graph.callers_of(sym_id):
            caller_id = edge.caller.canonical_id
            if caller_id not in visited:
                affected.append(edge.caller)
            _walk(caller_id, d - 1)
        # Reverse HTTP hop: a change to an endpoint PRODUCER affects every
        # client bound to its endpoint — cross-language blast radius grep
        # cannot reach.
        for consumer in graph.http_consumers_of(sym_id):
            if consumer.canonical_id not in visited:
                affected.append(consumer)
            _walk(consumer.canonical_id, d - 1)

    _walk(start_id, depth)

    files = {sym.file for sym in affected}
    return ImpactReport(
        start_symbol=start_sym,
        affected_symbols=affected,
        files=files,
    )


_NO_END = 1 << 30   # the last definition runs to end-of-file


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
    'ImpactReport',
    'SymbolResolution',
    'compute_impact_radius',
    'load_source_from_manifest',
]
