"""SCIP cross-source schema (SCIP-everywhere, Phase 2d).

Three tables that hold the materialized cross-source SCIP graph:

- ``scip_symbols`` — every symbol definition seen across all indexed
  sources. The canonical_id is the SCIP wire-format symbol string and
  is globally unique by construction.
- ``scip_edges`` — every reference (call/use) resolved by SCIP, joining
  caller→callee canonical_ids.
- ``scip_index_state`` — bookkeeping: which ``.scip`` file we last
  consumed for each source, with sha256 + indexed_at so we can detect
  and skip unchanged inputs.

The schema lives at library level (not under ``docgen/``) so the slim
consumer can ``init_scip_schema`` without transitively importing any
producer-only modules. The cross-source builder
(``docgen.scip_cross_source.CrossSourceGraph``) populates these tables;
slim-side query code reads them directly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlite3 import Connection


_SCIP_SYMBOLS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS scip_symbols (
    canonical_id          TEXT PRIMARY KEY,
    source_name           TEXT NOT NULL,
    language              TEXT NOT NULL,
    file                  TEXT NOT NULL,
    line_start            INTEGER NOT NULL,
    line_end              INTEGER NOT NULL,
    kind                  TEXT NOT NULL,
    display_name          TEXT NOT NULL,
    qualified_name        TEXT NOT NULL,
    parent_qualified_name TEXT
)
'''

_SCIP_EDGES_SCHEMA = '''
CREATE TABLE IF NOT EXISTS scip_edges (
    caller_canonical_id   TEXT NOT NULL,
    callee_canonical_id   TEXT NOT NULL,
    edge_type             TEXT NOT NULL,
    file                  TEXT NOT NULL,
    line                  INTEGER NOT NULL,
    confidence            TEXT NOT NULL,
    PRIMARY KEY (caller_canonical_id, callee_canonical_id, file, line)
)
'''

_SCIP_INDEX_STATE_SCHEMA = '''
CREATE TABLE IF NOT EXISTS scip_index_state (
    source_name      TEXT PRIMARY KEY,
    scip_path        TEXT NOT NULL,
    file_sha256      TEXT NOT NULL,
    indexed_at       TEXT NOT NULL,
    indexer_version  TEXT
)
'''

_SCIP_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_scip_symbols_source ON scip_symbols(source_name);
CREATE INDEX IF NOT EXISTS idx_scip_symbols_file   ON scip_symbols(file);
CREATE INDEX IF NOT EXISTS idx_scip_edges_callee   ON scip_edges(callee_canonical_id);
CREATE INDEX IF NOT EXISTS idx_scip_edges_caller   ON scip_edges(caller_canonical_id);
'''


# ---------------------------------------------------------------------------
# API surface schema (Wave 4, Phase 7a)
# ---------------------------------------------------------------------------

# Producer-side: declared HTTP endpoints, joined to scip_symbols via
# producer_symbol_id (the handler symbol). resolution_source records
# whether the binding came from an OpenAPI/Swagger spec, framework
# pattern matching, or manual override.
_API_ENDPOINTS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS api_endpoints (
    endpoint_id          TEXT PRIMARY KEY,
    source_name          TEXT NOT NULL,
    http_method          TEXT NOT NULL,
    path_template        TEXT NOT NULL,
    producer_symbol_id   TEXT,
    resolution_source    TEXT NOT NULL
)
'''

# Consumer-side: HTTP call sites resolved to the endpoint they hit.
# Composite PK across (consumer, endpoint, file, line) tolerates the
# same call site appearing multiple times in re-runs without
# ON CONFLICT IGNORE noise.
_API_CALLS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS api_calls (
    consumer_symbol_id   TEXT NOT NULL,
    endpoint_id          TEXT NOT NULL,
    call_site_file       TEXT NOT NULL,
    call_site_line       INTEGER NOT NULL,
    resolution_source    TEXT NOT NULL,
    confidence           TEXT NOT NULL,
    PRIMARY KEY (consumer_symbol_id, endpoint_id, call_site_file, call_site_line)
)
'''

_API_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_api_endpoints_source       ON api_endpoints(source_name);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_producer_sym ON api_endpoints(producer_symbol_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_consumer         ON api_calls(consumer_symbol_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_endpoint         ON api_calls(endpoint_id);
'''


# ---------------------------------------------------------------------------
# Layer C config-value index (Phase 2q)
# ---------------------------------------------------------------------------

# Persists Phase 2o's scanner output (HOCON / dotenv / etc.) so Layer C's
# resolution traversal (Phase 2s) can look up config values by key during
# reference walking. The UNIQUE on (source_name, file, key, line_start)
# tolerates re-ingest without duplicate-constraint violations;
# persist_config_values clears the source's rows before insert anyway.
_CONFIG_VALUES_SCHEMA = '''
CREATE TABLE IF NOT EXISTS config_values (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    file        TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    line_start  INTEGER NOT NULL,
    UNIQUE(source_name, file, key, line_start)
)
'''

_CONFIG_VALUES_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_config_values_source     ON config_values(source_name);
CREATE INDEX IF NOT EXISTS idx_config_values_source_key ON config_values(source_name, key);
'''


# ---------------------------------------------------------------------------
# Layer C string-literal index (Phase 2p)
# ---------------------------------------------------------------------------

# Persists every string literal extracted from indexed source files,
# attributed to its enclosing function/method (owning_symbol_id) where
# present. Layer C's resolution traversal (Phase 2s) queries this when
# walking back from a sink call site to find candidate literal
# arguments inside the call's enclosing scope.
_STRING_LITERALS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS string_literals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name      TEXT NOT NULL,
    file             TEXT NOT NULL,
    line_start       INTEGER NOT NULL,
    col_start        INTEGER NOT NULL,
    value            TEXT NOT NULL,
    owning_symbol_id TEXT,
    kind             TEXT NOT NULL DEFAULT 'plain'
)
'''

_STRING_LITERALS_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_string_literals_source         ON string_literals(source_name);
CREATE INDEX IF NOT EXISTS idx_string_literals_owning_symbol  ON string_literals(source_name, owning_symbol_id);
CREATE INDEX IF NOT EXISTS idx_string_literals_file           ON string_literals(source_name, file);
'''


# ---------------------------------------------------------------------------
# Layer C process_invocations (Phase 2t)
# ---------------------------------------------------------------------------

# Final Layer C output: one row per sink call site. ariadne_trace_flow
# (Phase 9) joins this with scip_edges and api_calls to build
# cross-language flow traces. ``target_path`` is nullable — unresolved
# call sites are still recorded so the trace can report "attempted but
# not resolvable". ``target_symbol_id`` is reserved for Phase 2t.b
# fuzzy file-matching.
_PROCESS_INVOCATIONS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS process_invocations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name       TEXT NOT NULL,
    caller_symbol_id  TEXT NOT NULL,
    target_path       TEXT,
    target_symbol_id  TEXT,
    confidence        TEXT NOT NULL,
    file              TEXT NOT NULL,
    line_start        INTEGER NOT NULL,
    line_end          INTEGER NOT NULL
)
'''

_PROCESS_INVOCATIONS_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_process_invocations_source ON process_invocations(source_name);
CREATE INDEX IF NOT EXISTS idx_process_invocations_caller ON process_invocations(source_name, caller_symbol_id);
CREATE INDEX IF NOT EXISTS idx_process_invocations_target ON process_invocations(target_symbol_id);
'''


# ---------------------------------------------------------------------------
# HTTP client call sites (Phase 8b)
# ---------------------------------------------------------------------------

# Per-language HTTP client extractors (8b.1 Python, 8b.2 JS, 8b.3 JVM)
# all write here. Phase 8c reads ``raw_url`` from these rows, resolves
# them against ``api_endpoints.path_template``, and inserts matched
# pairs into ``api_calls``.
#
# ``raw_url`` is the literal URL string captured at the call site —
# stored as-is, normalization happens in Phase 8c. ``http_method`` is
# nullable because some libraries (Scala play-ws fluent builders) carry
# the method on a chained call we don't track in v1; the URL is still
# useful to record.
#
# ``consumer_symbol_id`` is the enclosing function/method's
# ``scip_symbols.canonical_id``; NULL when the call sits at module
# level. ``sink_name`` is the ``SinkSpec.name`` that matched, useful
# for audit / debugging.
_HTTP_CLIENT_CALLS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS http_client_calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name         TEXT NOT NULL,
    consumer_symbol_id  TEXT,
    raw_url             TEXT NOT NULL,
    http_method         TEXT,
    call_site_file      TEXT NOT NULL,
    call_site_line      INTEGER NOT NULL,
    sink_name           TEXT NOT NULL,
    confidence          TEXT NOT NULL
)
'''

_HTTP_CLIENT_CALLS_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_http_client_calls_source   ON http_client_calls(source_name);
CREATE INDEX IF NOT EXISTS idx_http_client_calls_consumer ON http_client_calls(consumer_symbol_id);
CREATE INDEX IF NOT EXISTS idx_http_client_calls_file     ON http_client_calls(source_name, call_site_file);
'''
_CONFIG_READS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS config_reads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    key         TEXT NOT NULL,
    file        TEXT NOT NULL,
    line_start  INTEGER NOT NULL,
    col_start   INTEGER NOT NULL,
    value       TEXT,
    confidence  TEXT NOT NULL,
    UNIQUE(source_name, file, line_start, col_start, key)
)
'''

_CONFIG_READS_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_config_reads_source_key ON config_reads(source_name, key);
'''


_RST_AUTODOC_LINKS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS rst_autodoc_links (
    source_name                 TEXT NOT NULL,
    symbol_qualified_name       TEXT NOT NULL,
    rst_section_qualified_name  TEXT NOT NULL,
    UNIQUE(source_name, symbol_qualified_name, rst_section_qualified_name)
)
'''

_RST_AUTODOC_LINKS_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_rst_autodoc_links_symbol ON rst_autodoc_links(symbol_qualified_name);
'''


# ---------------------------------------------------------------------------
# SQL data model — schema symbols + data-access edges (design §4/§5)
# ---------------------------------------------------------------------------

# The schema *as data*: one row per table/column/view/index node, mirroring
# api_endpoints (nullable producer, provenance, per-source idempotence).
# ``producer_symbol_id`` links a code-first table/column back to the ORM
# model/attr SCIP symbol that defines it (the Layer-1 maps_to binding).
# ``confidence`` is the ordered enum 'exact' > 'resolved' > 'derived' >
# 'recovered'; the read boundary asserts only at/above the configured floor.
_SCHEMA_SYMBOLS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS schema_symbols (
    canonical_id       TEXT PRIMARY KEY,
    source_name        TEXT NOT NULL,
    node_type          TEXT NOT NULL,
    database           TEXT,
    db_schema          TEXT,
    table_name         TEXT NOT NULL,
    column_name        TEXT,
    column_type        TEXT,
    is_nullable        INTEGER,
    is_primary_key     INTEGER,
    references_id      TEXT,
    producer_symbol_id TEXT,
    last_changed_by    TEXT,
    resolution_source  TEXT NOT NULL,
    confidence         TEXT NOT NULL
)
'''

# The access edges: one row per code symbol -> table/column interaction,
# role-typed (SENT: 'filter'|'write'|'order'  RECV: 'project'  also 'ddl').
# Mirrors api_calls / config_reads. The composite UNIQUE tolerates re-runs
# (persist clears the source's rows first anyway).
_DATA_ACCESS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS data_access (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name        TEXT NOT NULL,
    consumer_symbol_id TEXT NOT NULL,
    schema_symbol_id   TEXT NOT NULL,
    role               TEXT NOT NULL,
    value_source       TEXT,
    binds_to           TEXT,
    call_site_file     TEXT,
    call_site_line     INTEGER,
    witness            TEXT NOT NULL,
    confidence         TEXT NOT NULL,
    UNIQUE(source_name, consumer_symbol_id, schema_symbol_id,
           call_site_file, call_site_line, role)
)
'''

_DATA_MODEL_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_schema_symbols_source        ON schema_symbols(source_name);
CREATE INDEX IF NOT EXISTS idx_schema_symbols_producer      ON schema_symbols(producer_symbol_id);
CREATE INDEX IF NOT EXISTS idx_schema_symbols_table         ON schema_symbols(source_name, table_name);
CREATE INDEX IF NOT EXISTS idx_data_access_source           ON data_access(source_name);
CREATE INDEX IF NOT EXISTS idx_data_access_consumer         ON data_access(consumer_symbol_id);
CREATE INDEX IF NOT EXISTS idx_data_access_schema_symbol    ON data_access(schema_symbol_id);
'''


_DATA_MODEL_GAPS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS data_model_gaps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    detail      TEXT NOT NULL
)
'''


def init_scip_schema(conn: 'Connection') -> None:
    """Create the SCIP tables (cross-source graph + API surface +
    config-value index + config-read index + string-literal index +
    process_invocations) and their indexes if missing.

    Idempotent — safe to call on an existing DB. Used by Library's
    __attrs_post_init__ so every fresh open of ariadne.db has the SCIP
    surface available even before any indexer or Swagger ingestion has
    populated it.
    """
    conn.execute(_SCIP_SYMBOLS_SCHEMA)
    conn.execute(_SCIP_EDGES_SCHEMA)
    conn.execute(_SCIP_INDEX_STATE_SCHEMA)
    conn.executescript(_SCIP_INDEXES)
    # API surface tables (Wave 4)
    conn.execute(_API_ENDPOINTS_SCHEMA)
    conn.execute(_API_CALLS_SCHEMA)
    conn.executescript(_API_INDEXES)
    # Layer C config-value index (Phase 2q)
    conn.execute(_CONFIG_VALUES_SCHEMA)
    conn.executescript(_CONFIG_VALUES_INDEXES)
    # Config-read index (Tier 2 config↔code bridge)
    conn.execute(_CONFIG_READS_SCHEMA)
    conn.executescript(_CONFIG_READS_INDEXES)
    # Layer C string-literal index (Phase 2p)
    conn.execute(_STRING_LITERALS_SCHEMA)
    conn.executescript(_STRING_LITERALS_INDEXES)
    # Layer C process_invocations (Phase 2t)
    conn.execute(_PROCESS_INVOCATIONS_SCHEMA)
    conn.executescript(_PROCESS_INVOCATIONS_INDEXES)
    # HTTP client call sites (Phase 8b)
    conn.execute(_HTTP_CLIENT_CALLS_SCHEMA)
    conn.executescript(_HTTP_CLIENT_CALLS_INDEXES)
    conn.execute(_RST_AUTODOC_LINKS_SCHEMA)
    conn.executescript(_RST_AUTODOC_LINKS_INDEXES)
    # SQL data model — schema symbols + data-access edges (design §4/§5)
    conn.execute(_SCHEMA_SYMBOLS_SCHEMA)
    conn.execute(_DATA_ACCESS_SCHEMA)
    conn.executescript(_DATA_MODEL_INDEXES)
    conn.execute(_DATA_MODEL_GAPS_SCHEMA)


def persist_rst_autodoc_links(conn, source_name, links):
    """Persist resolved rst->symbol autodoc links for reverse lookup.

    Only resolved links are stored (a dangling target points at no symbol).
    Replaces this source's prior links so a re-sync is idempotent.
    """
    conn.execute('DELETE FROM rst_autodoc_links WHERE source_name = ?', (source_name,))
    conn.executemany(
        'INSERT OR IGNORE INTO rst_autodoc_links '
        '(source_name, symbol_qualified_name, rst_section_qualified_name) '
        'VALUES (?, ?, ?)',
        [
            (source_name, lk.symbol_qualified_name, lk.section_qualified_name)
            for lk in links
            if lk.resolved
        ],
    )


def rst_sections_documenting(conn, symbol_qualified_name):
    """rst section qualified-names that document ``symbol_qualified_name``
    (the reverse of the autodoc link). Empty when none -- the symbol has no
    human rst documentation pointing at it."""
    rows = conn.execute(
        'SELECT rst_section_qualified_name FROM rst_autodoc_links '
        'WHERE symbol_qualified_name = ? ORDER BY rst_section_qualified_name',
        (symbol_qualified_name,),
    ).fetchall()
    return tuple(r[0] for r in rows)


__all__ = ['init_scip_schema']
