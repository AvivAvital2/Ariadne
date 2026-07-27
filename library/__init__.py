"""SQLite-backed storage for the Ariadne library.

This module provides the main Library class for storing and retrieving
LLM-generated documentation about source code.

The Library class is composed from domain-specific mixins:
- CoreMixin: document CRUD, chunks, sections, sync state
- SearchMixin: semantic search, scope filtering
- AnalyticsMixin: usage tracking, quality scores, gap analysis, trends
- GraphMixin: dependency graph building and querying
- QualityMixin: health checks, linting, debt analysis
- DebugMixin: debugging, tracing, code patterns, gotchas
- IntelligenceMixin: explanation, clustering, review, conflict resolution
"""
from __future__ import annotations

__all__ = ['Library', 'filter_by_branch']

import contextlib
import json
import logging
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from sqlite3 import Connection

import numpy as np
from attrs import define, field, frozen

if __name__ != '__main__':
    # Mixin imports
    from library.analytics import AnalyticsMixin
    from library.core import CoreMixin
    from library.debug import DebugMixin
    from library.graph import GraphMixin
    from library.intelligence import IntelligenceMixin
    from library.migrations import MigrationsMixin
    from library.quality import QualityMixin
    from library.scip import init_scip_schema
    from library.search import SearchMixin
    from library.themes import (
        _CLUSTER_HISTORY_SCHEMA,
        _SPOOL_ASSOC_SYNC_SCHEMA,
        _THEME_MEMBERS_SCHEMA,
        _THEMES_SCHEMA,
        ThemesMixin, _THEME_SYNCED_HASHES_SCHEMA)

_logger = logging.getLogger(__name__)


@frozen
class _ConnectionProvider:
    """Thread-local SQLite connection provider with WAL mode."""
    path: Path
    _local: threading.local = field(init=False, factory=threading.local)
    _closed: threading.Event = field(init=False, factory=threading.Event)

    def _open_connection(self) -> Connection:
        """Open a new connection with WAL mode enabled."""
        conn = sqlite3.connect(self.path, check_same_thread=True)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=normal')
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    @contextlib.contextmanager
    def acquire(self) -> Iterator[Connection]:
        """Get a thread-local connection."""
        if self._closed.is_set():
            raise RuntimeError('Connection provider has been closed')

        try:
            conn = self._local.conn
        except AttributeError:
            conn = self._open_connection()
            self._local.conn = conn

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close all connections."""
        self._closed.set()
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn


_DOCUMENTS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content_type TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_files TEXT NOT NULL,
    embedding BLOB,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL,
    content_token_count INTEGER
)
'''

_CHUNKS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB
)
'''

_SECTIONS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS sections (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    heading TEXT NOT NULL,
    description TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB,
    PRIMARY KEY (document_id, idx)
)
'''

_CREATE_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id);
'''

_SYNC_STATE_SCHEMA = '''
CREATE TABLE IF NOT EXISTS sync_state (
    source_name TEXT PRIMARY KEY,
    git_hash TEXT NOT NULL,
    synced_at TEXT NOT NULL
)
'''
_SOURCE_RELATIONS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS source_relations (
    source_name TEXT PRIMARY KEY,
    depends_on  TEXT NOT NULL DEFAULT '[]',
    parent      TEXT,
    branches    TEXT NOT NULL DEFAULT '[]'
)
'''

_USAGE_EVENTS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    query TEXT,
    result_count INTEGER NOT NULL DEFAULT 0,
    outcome TEXT NOT NULL DEFAULT 'call',
    feedback TEXT,
    returned_document_ids TEXT,
    quality_score INTEGER
)
'''

_USAGE_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_tool_name ON usage_events(tool_name);
CREATE INDEX IF NOT EXISTS idx_usage_outcome ON usage_events(outcome);
'''

_DOC_GRAPH_SCHEMA = '''
CREATE TABLE IF NOT EXISTS doc_graph (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (source_id, target_id, edge_type)
)
'''

_DOC_GRAPH_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_graph_source ON doc_graph(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_target ON doc_graph(target_id);
CREATE INDEX IF NOT EXISTS idx_graph_type ON doc_graph(edge_type);
'''

_QUERY_CACHE_SCHEMA = '''
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key TEXT PRIMARY KEY,
    branch TEXT NOT NULL,
    query_text TEXT,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
'''

_QUERY_CACHE_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_qcache_branch ON query_cache(branch);
CREATE INDEX IF NOT EXISTS idx_qcache_created ON query_cache(created_at);
'''


def filter_by_branch(docs: list, branch: str) -> list:
    """Filter documents by branch pattern.

    Stable docs always pass. Experimental/deprecated docs must match
    at least one branch pattern via fnmatch.
    """
    import fnmatch

    filtered = []
    for doc in docs:
        doc_status = doc.metadata.get('status', 'stable')
        if doc_status == 'stable':
            filtered.append(doc)
            continue
        doc_branches = doc.metadata.get('branches', [])
        if isinstance(doc_branches, list):
            for pattern in doc_branches:
                if fnmatch.fnmatch(branch, pattern):
                    filtered.append(doc)
                    break
    return filtered


@define(eq=False)
class Library(
    CoreMixin,
    SearchMixin,
    AnalyticsMixin,
    GraphMixin,
    QualityMixin,
    DebugMixin,
    IntelligenceMixin,
    ThemesMixin,
    MigrationsMixin,
):
    """SQLite-backed library for storing LLM-generated documentation.

    This class provides CRUD operations for documents and semantic search
    using vector embeddings. Methods are organized into domain mixins.

    Example:
        >>> library = Library(Path('ariadne.db'))
        >>> doc = library.add_document(
        ...     content_type='explanation',
        ...     title='LLM Caching',
        ...     content='...',
        ...     source_files=['mylib/core.py']
        ... )
        >>> results = library.search('How does caching work?', k=5)
        >>> library.close()

    Args:
        path: Path to the SQLite database file.
    """
    path: Path
    _conn_provider: _ConnectionProvider = field(init=False)

    @_conn_provider.default
    def _conn_provider_init(self) -> _ConnectionProvider:
        return _ConnectionProvider(self.path)

    def __attrs_post_init__(self) -> None:
        """Initialize the database schema."""
        with self._conn_provider.acquire() as conn:
            conn.execute(_DOCUMENTS_SCHEMA)
            conn.execute(_CHUNKS_SCHEMA)
            conn.execute(_SECTIONS_SCHEMA)
            conn.execute(_SYNC_STATE_SCHEMA)
            conn.execute(_SOURCE_RELATIONS_SCHEMA)
            conn.execute(_USAGE_EVENTS_SCHEMA)
            conn.executescript(_CREATE_INDEXES)
            conn.executescript(_USAGE_INDEXES)
            conn.execute(_DOC_GRAPH_SCHEMA)
            conn.executescript(_DOC_GRAPH_INDEXES)
            conn.execute(_QUERY_CACHE_SCHEMA)
            conn.executescript(_QUERY_CACHE_INDEXES)
            conn.execute(_THEMES_SCHEMA)
            conn.executescript(_THEME_MEMBERS_SCHEMA)
            conn.executescript(_CLUSTER_HISTORY_SCHEMA)
            conn.executescript(_THEME_SYNCED_HASHES_SCHEMA)
            conn.execute(_SPOOL_ASSOC_SYNC_SCHEMA)
            init_scip_schema(conn)
            
            # Migration: add string_literals.kind (plain|fstring) if missing
            lit_cols = {row[1] for row in conn.execute('PRAGMA table_info(string_literals)')}
            if 'kind' not in lit_cols:
                conn.execute("ALTER TABLE string_literals ADD COLUMN kind TEXT NOT NULL DEFAULT 'plain'")
            # Migration: add themes.association (the owning clustering pass) if
            # missing. Existing themes default to '' (the base global pass), so
            # they keep their ids and reconcile unchanged — no regeneration.
            theme_cols = {row[1] for row in conn.execute('PRAGMA table_info(themes)')}
            if 'association' not in theme_cols:
                conn.execute("ALTER TABLE themes ADD COLUMN association TEXT NOT NULL DEFAULT ''")
            # Migration: add returned_document_ids column if missing
            cols = {row[1] for row in conn.execute('PRAGMA table_info(usage_events)')}
            if 'returned_document_ids' not in cols:
                conn.execute('ALTER TABLE usage_events ADD COLUMN returned_document_ids TEXT')
            if 'quality_score' not in cols:
                conn.execute('ALTER TABLE usage_events ADD COLUMN quality_score INTEGER')
            # Migration: add source_name column to documents if missing
            doc_cols = {row[1] for row in conn.execute('PRAGMA table_info(documents)')}
            if 'source_name' not in doc_cols:
                conn.execute('ALTER TABLE documents ADD COLUMN source_name TEXT')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_name)')
                # Backfill source_name from metadata or source_files paths
                for row in conn.execute('SELECT id, metadata, source_files FROM documents').fetchall():
                    meta = json.loads(row[1]) if row[1] else {}
                    sf = json.loads(row[2]) if row[2] else []
                    sn = meta.get('source_name')
                    if not sn and meta.get('module_name', '').startswith('pythonproject'):
                        sn = 'pythonproject'
                    if not sn:
                        for f in sf:
                            if 'myproject' in f or 'pythonproject' in f:
                                sn = 'pythonproject'
                                break
                            elif 'benchmark' in f:
                                sn = 'benchmark'
                                break
                    if sn:
                        conn.execute('UPDATE documents SET source_name = ? WHERE id = ?', (sn, row[0]))
            # Migration: add content_token_count column to documents if missing
            if 'content_token_count' not in doc_cols:                                                                                                                   
                conn.execute('ALTER TABLE documents ADD COLUMN content_token_count INTEGER')                                                                            
                conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_token_count ON documents(content_token_count)')                                                  
            # Migration: normalize existing embeddings for dot-product search
            self._migrate_normalize_embeddings(conn)

    def _migrate_normalize_embeddings(self, conn: Connection) -> None:
        """Normalize all stored embeddings to unit vectors (one-time migration).

        After this, search can use dot product instead of cosine similarity,
        saving the per-query normalization of all stored embeddings.
        """
        # Check if already migrated by sampling the first embedding
        sample = conn.execute(
            'SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1'
        ).fetchone()
        if sample is None:
            return  # No embeddings to migrate
        emb = np.frombuffer(sample[0], dtype=np.float32)
        if abs(np.linalg.norm(emb) - 1.0) < 1e-4:
            return  # Already normalized

        _logger.info('Normalizing document embeddings for dot-product search...')
        for row in conn.execute(
            'SELECT id, embedding FROM documents WHERE embedding IS NOT NULL'
        ).fetchall():
            emb = np.frombuffer(row[1], dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0 and abs(norm - 1.0) > 1e-6:
                emb = emb / norm
                conn.execute('UPDATE documents SET embedding = ? WHERE id = ?', (emb.tobytes(), row[0]))

        _logger.info('Normalizing chunk embeddings...')
        for row in conn.execute(
            'SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL'
        ).fetchall():
            emb = np.frombuffer(row[1], dtype=np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0 and abs(norm - 1.0) > 1e-6:
                emb = emb / norm
                conn.execute('UPDATE chunks SET embedding = ? WHERE id = ?', (emb.tobytes(), row[0]))
        _logger.info('Embedding normalization complete.')

    def close(self) -> None:
        """Close the database connection."""
        self._conn_provider.close()

    def __enter__(self) -> Library:
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        self.close()


# Base-pass ('') themes are member-grounded (ScopedLibrary._admitted_theme_ids):
# a theme is admitted to a scope only when the scope's sources hold at least
# this share of its membership. Any-member admission leaked whole-cluster
# summaries across scopes (a 999-member private cluster was admitted on ONE
# stray in-scope member — a theme's content is written over ALL its members).
# Measured separation on the live store: leaking stragglers held <= 0.35%
# share; legitimate small cross-repo themes held >= 17%. A LONE in-scope
# member additionally admits only a cluster it co-dominates (>= 0.5 share) —
# "one stray member never admits a foreign cluster's summary". Provisional —
# calibrate with real distributions (same policy as the spool gate constants).
_BASE_THEME_MIN_MEMBER_SHARE = 0.10


class ScopedLibrary:
    """Closure-bounded view onto a ``Library``.

    Phase 2 of the closure-scoping design — see
    ``designs/directional-closure-scoping.md``. Every data-returning
    method on the underlying ``Library`` is mirrored here and filtered
    by ``closure`` (a ``frozenset[str]`` of source names). The intent
    is that consumers receive a ``ScopedLibrary``, never a raw
    ``Library``, so per-request scope is structurally enforced at the
    data-access layer instead of relied on at every call site.
    """

    def __init__(self, library: 'Library', closure: 'frozenset[str]') -> None:
        if not closure:
            raise ValueError(
                'ScopedLibrary closure must not be empty — an empty '
                'closure would silently produce empty results for every '
                'query. Resolve a non-empty closure (typically via '
                'Config.scope_closure(source_name)) before constructing '
                'the wrapper.',
            )
        self._library = library
        self._closure = closure
        # Cache the sorted-tuple form used for SQL params. The closure
        # is a frozenset set once at construction; recomputing on every
        # _filter_ids_by_closure call is just wasted work.
        self._closure_params: tuple[str, ...] = tuple(sorted(closure))
        self._scip_graph: object | None = None

    @property
    def closure(self) -> 'frozenset[str]':
        """The source-name closure this scoped view admits (read-only).

        Exposed so a caller that must read through the raw library (e.g.
        the PM audience-cache lookup, which needs ``list_documents``) can
        still source-scope its own filtering to exactly this set.
        """
        return self._closure

    def _scip(self):
        """Lazily load the SCIP graph from the underlying library DB.

        The graph is read-only at this point (Phase 2 doesn't write to
        it), so we build it once per ScopedLibrary and reuse on
        subsequent queries.
        """
        if self._scip_graph is None:
            from docgen.scip_cross_source import CrossSourceGraph
            graph = CrossSourceGraph()
            with self._library._conn_provider.acquire() as conn:
                graph.load_from(conn)
            self._scip_graph = graph
        return self._scip_graph

    def _edge_in_closure(self, edge) -> bool:
        return (
            edge.caller.source_name in self._closure
            and edge.callee.source_name in self._closure
        )

    def list_documents_lite(self, content_type=None):
        """Lite docs restricted to the scope.

        Filters out any document whose ``source_name`` is not in the
        closure. Theme docs (``content_type='theme'``) carry
        ``source_name=NULL`` (cross-source by design); they pass the
        theme gate instead — see :meth:`_admitted_theme_ids`.
        """
        rows = self._library.list_documents_lite(content_type=content_type)
        theme_ids = [
            d.id for d in rows
            if d.source_name is None and d.content_type == 'theme'
        ]
        admitted_themes = self._admitted_theme_ids(theme_ids)
        return [d for d in rows if self._admit(d, admitted_themes)]

    def _admit(self, doc, admitted_themes=None) -> bool:
        """Return True if ``doc`` should pass the scope filter.

        In-closure source matches admit it. A NULL-source theme is
        admitted only when it passes the theme gate — see
        :meth:`_admitted_theme_ids`. ``admitted_themes`` is an optional
        precomputed admitted-id set for batch callers; when absent the
        gate runs for just this doc. Everything else is rejected.
        """
        if doc.source_name in self._closure:
            return True
        if doc.source_name is None and doc.content_type == 'theme':
            if admitted_themes is None:
                admitted_themes = self._admitted_theme_ids([doc.id])
            return doc.id in admitted_themes
        return False

    def get_embeddings_for_ids(self, doc_ids):
        """Embedding lookup restricted to the closure.

        IDs that resolve to a document outside the closure are silently
        dropped — raising would surface their existence, which is itself
        a leak. An out-of-closure id is treated identically to an
        unknown id.
        """
        allowed = self._filter_ids_by_closure(doc_ids)
        return self._library.get_embeddings_for_ids(allowed)

    def find_documents_by_source_files(self, file_paths):
        """File-path lookup restricted to the closure.

        The underlying matcher uses basename substring matching on the
        ``source_files`` JSON column, so a file name shared by multiple
        sources can surface rows from each — the closure filter is what
        stops out-of-closure rows from leaking through that ambiguity.
        """
        rows = self._library.find_documents_by_source_files(file_paths)
        return [d for d in rows if self._admit(d)]

    def get_documents_batch(self, doc_ids):
        """Full-doc batch fetch restricted to the closure.

        Symmetric to ``get_embeddings_for_ids``: out-of-closure ids are
        silently dropped from the response. Treated identically to
        unknown ids.
        """
        allowed = self._filter_ids_by_closure(doc_ids)
        return self._library.get_documents_batch(allowed)

    def get_related(self, doc_id, max_hops=2, limit=10):
        """Graph-walk neighbors restricted to the closure.

        The underlying walker walks ``doc_graph`` by node id. Some edge
        kinds (notably ``edge_type='imports'``) use file paths as
        nodes, so a non-doc-id seed (e.g., the ``context_file`` argument
        used by ``mcp_service_search``) is a legitimate walker input.
        We therefore DON'T pre-reject the seed by checking it against
        ``documents.id`` — we let the walker run, then filter the
        result set by closure. The walker never returns the seed
        itself (``visited.pop(doc_id, None)`` in ``library_graph``),
        so there's no seed-existence leak vector.
        """
        related = self._library.get_related(
            doc_id, max_hops=max_hops, limit=limit,
        )
        if not related:
            return related
        allowed_ids = set(
            self._filter_ids_by_closure([r['id'] for r in related]),
        )
        return [r for r in related if r['id'] in allowed_ids]

    def get_related_batch(self, doc_ids, max_hops=2, limit=10):
        """Batch, closure-filtered twin of :meth:`get_related`.

        One bulk graph load in the underlying library; the closure filter
        runs once across all neighbours rather than per seed. The result
        for each id equals ``get_related(id)`` (the in-library batch is
        equivalence-tested in ``tests/test_get_related_batch.py``).
        """
        raw = self._library.get_related_batch(
            doc_ids, max_hops=max_hops, limit=limit,
        )
        all_ids = {r['id'] for results in raw.values() for r in results}
        if not all_ids:
            return raw
        allowed = set(self._filter_ids_by_closure(list(all_ids)))
        return {
            doc_id: [r for r in results if r['id'] in allowed]
            for doc_id, results in raw.items()
        }

    def get_document(self, doc_id):
        """Single-doc lookup restricted to the closure.

        Returns the underlying ``Library.get_document`` result iff the
        document's ``source_name`` is in the closure. Out-of-closure
        rows return ``None`` (treated identically to missing) — exposing
        the doc would leak its existence to a caller whose scope does
        not include it.
        """
        doc = self._library.get_document(doc_id)
        if doc is None or not self._admit(doc):
            return None
        return doc

    def get_theme(self, cluster_id):
        """Single-theme lookup restricted to the closure.

        Returns the theme iff its summary doc's ``source_name`` is in
        the closure. Same identity rule as :meth:`list_themes` and
        :meth:`get_theme_members`.
        """
        theme = self._library.get_theme(cluster_id)
        if theme is None:
            return None
        if not self._filter_ids_by_closure([theme.doc_id]):
            return None
        return theme

    def list_themes(self, *, coherent_only=True):
        """Themes restricted to those whose summary doc is in the closure.

        The summary doc is the theme's identity — if the summary's
        ``source_name`` is out of closure, the theme is invisible. We
        push the filter into the JOIN so the SQL never returns rows we
        would discard.
        """
        return self._library.list_themes(
            coherent_only=coherent_only,
            source_names=self._closure_params,
        )

    def get_theme_members(self, cluster_id):
        """Member list filtered to in-closure member docs.

        Out-of-closure members are dropped from the displayed set. If
        the theme's summary doc is itself out of closure, return ``[]``
        — revealing membership of an invisible theme would itself be a
        leak.
        """
        owning_doc = self._library._theme_owning_doc_id(cluster_id)
        if owning_doc is None:
            return []
        if not self._filter_ids_by_closure([owning_doc]):
            return []
        members = self._library.get_theme_members(cluster_id)
        if not members:
            return members
        allowed_ids = set(
            self._filter_ids_by_closure([m[0] for m in members]),
        )
        return [m for m in members if m[0] in allowed_ids]

    def scip_callers(self, symbol_id):
        """SCIP callers restricted to in-closure edges.

        An edge is in-closure iff BOTH its caller and its callee have
        a ``source_name`` in the closure. This is what implements the
        directional rule at the graph level — a leaf-source's reverse
        closure sees its consumers' call sites; a product's forward
        closure does not see other products that call into the shared
        symbol.
        """
        return [
            e for e in self._scip().callers_of(symbol_id)
            if self._edge_in_closure(e)
        ]

    def scip_callees(self, symbol_id):
        """SCIP callees restricted to in-closure edges. Symmetric to
        ``scip_callers``."""
        return [
            e for e in self._scip().callees_of(symbol_id)
            if self._edge_in_closure(e)
        ]

    def _filter_ids_by_closure(self, doc_ids):
        """Return the subset of ``doc_ids`` admitted by the scope.

        In-closure sources are admitted; NULL-source theme rows go
        through the theme gate (:meth:`_admitted_theme_ids`). Batches to
        stay under SQLite's variable limit."""
        if not doc_ids:
            return []
        if not self._closure:
            return []
        from library.sql_vars import chunk_ids
        doc_ids = list(doc_ids)
        closure_params = self._closure_params
        src_placeholders = ','.join('?' * len(closure_params))
        candidates: list[tuple[str, bool]] = []
        with self._library._conn_provider.acquire() as conn:
            for chunk in chunk_ids(doc_ids, reserved=len(closure_params)):
                id_placeholders = ','.join('?' * len(chunk))
                rows = conn.execute(
                    f'SELECT id, source_name FROM documents '
                    f'WHERE id IN ({id_placeholders}) '
                    f'AND ('
                    f'source_name IN ({src_placeholders}) '
                    f"OR (source_name IS NULL AND content_type = 'theme')"
                    f')',
                    chunk + list(closure_params),
                ).fetchall()
                for doc_id, source_name in rows:
                    needs_theme_gate = source_name not in self._closure
                    candidates.append((str(doc_id), needs_theme_gate))
        admitted_themes = self._admitted_theme_ids(
            [d for d, needs_gate in candidates if needs_gate],
        )
        return [
            d for d, needs_gate in candidates
            if not needs_gate or d in admitted_themes
        ]

    def _theme_association_admitted(self, association: str) -> bool:
        """A non-empty scoped ``association`` passes the closure gate.

        The association is a '|'-joined sorted source-name key naming
        every source the theme spans; it is admitted iff every one of
        those sources is in the closure. Because ``make_scoped_library``
        unions the enabled spool sources into the closure, a spool-scoped
        theme is visible only when that spool is enabled (and a
        project × spool theme only in that project's own scope). The base
        pass ('') is not decided here — it is member-grounded, see
        :meth:`_admitted_theme_ids`.
        """
        if not association:
            return False
        return all(
            token in self._closure for token in association.split('|')
        )
    def _admitted_theme_ids(self, doc_ids) -> set:
        """The subset of theme summary ``doc_ids`` that pass the theme gate.

        A scoped association is admitted iff every source it spans is in
        the closure (:meth:`_theme_association_admitted`). A base-pass
        ('') theme is member-grounded: admitted iff the closure's sources
        hold at least ``_BASE_THEME_MIN_MEMBER_SHARE`` of its membership,
        and a LONE in-scope member only admits a cluster it co-dominates
        (>= half the membership) —
        a theme's content summarizes ALL its members, so pass-level or
        any-single-member admission leaked foreign clusters' summaries
        into scopes holding a negligible share. A theme doc with no
        themes row (a stale orphan from an earlier rebuild) is never
        admitted — fail closed.
        """
        doc_ids = [d for d in doc_ids if d]
        if not doc_ids:
            return set()
        from library.sql_vars import chunk_ids
        closure_params = self._closure_params
        src_placeholders = ','.join('?' * len(closure_params))
        admitted: set = set()
        with self._library._conn_provider.acquire() as conn:
            for chunk in chunk_ids(list(doc_ids), reserved=len(closure_params)):
                id_placeholders = ','.join('?' * len(chunk))
                rows = conn.execute(
                    f'SELECT t.doc_id, t.association, '
                    f'(SELECT COUNT(*) FROM theme_members tm '
                    f'JOIN documents md ON md.id = tm.element_id '
                    f'WHERE tm.cluster_id = t.cluster_id '
                    f'AND md.source_name IN ({src_placeholders})), '
                    f'(SELECT COUNT(*) FROM theme_members tm '
                    f'WHERE tm.cluster_id = t.cluster_id) '
                    f'FROM themes t WHERE t.doc_id IN ({id_placeholders})',
                    list(closure_params) + chunk,
                ).fetchall()
                for doc_id, association, in_scope, total in rows:
                    if association:
                        if self._theme_association_admitted(str(association)):
                            admitted.add(str(doc_id))
                    elif total:
                        share = in_scope / total
                        lone_stray = in_scope == 1 and share < 0.5
                        if share >= _BASE_THEME_MIN_MEMBER_SHARE and not lone_stray:
                            admitted.add(str(doc_id))
        return admitted
