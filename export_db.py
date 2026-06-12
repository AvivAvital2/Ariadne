"""Export/import a single-source *slice* of the Ariadne library as a
self-contained SQLite database.

The goal: hand someone a standalone ``ariadne.db`` containing everything
needed to answer questions about ONE source (e.g. an SDK) — its documents,
their embeddings, chunks and sections — so they can point the MCP server /
Slack bridge at it and get the same answers the full library would give for
that source, with no re-generation.

This module is intentionally separate from ``export.py`` (markdown export):
that path drops embeddings and isn't source-scoped; this one produces a
queryable DB scoped to a single source, carrying embeddings verbatim so the
recipient pays nothing to stand it up.

Scope: every document Ariadne generated for the source travels — ALL content
types (explanation, architecture, qa, gotcha, diagram, catalog, finding) — with
their chunks, sections, and embeddings, plus the source's themes and "Related
Documents" graph and its relational/sync metadata. With ``include_scip=True`` the
SCIP call graph travels too (the source's symbols plus their 1-hop cross-source
neighbors), so callers/impact_radius resolve on the slice; ``local N`` symbol ids
are namespaced by owning source for cross-instance merge safety.
"""
from __future__ import annotations

__all__ = [
    'DEFAULT_EMBEDDING_MODEL',
    'EmbeddingModelMismatch',
    'ImportConflictError',
    'ImportReport',
    'SliceManifest',
    'SourceNotFoundError',
    'export_source_db',
    'import_source_db',
]

import sqlite3
from pathlib import Path

from attrs import frozen

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-large'

# Tables copied verbatim from bundle into target on import, documents first
# so chunk/section foreign keys resolve.
_MERGE_TABLES = ('documents', 'chunks', 'sections', 'themes', 'theme_members',
                 'doc_graph', 'scip_symbols', 'scip_edges', 'scip_index_state',
                 'source_relations', 'sync_state')

# A theme is in-slice iff it has >=1 member whose document is in-source.
_RELEVANT_CLUSTERS = (
    'SELECT tm.cluster_id FROM theme_members tm '
    'JOIN documents d ON d.id = tm.element_id '
    'WHERE d.source_name = ?'
)


class SourceNotFoundError(ValueError):
    """The requested source has no rows in the source database."""


class EmbeddingModelMismatch(ValueError):
    """A bundle's embedding model differs from the target's, so its vectors
    would be incomparable with the target's at search time."""


class ImportConflictError(ValueError):
    """A merge hit pre-existing document ids while ``on_conflict='fail'``."""


@frozen
class SliceManifest:
    """Summary of what an export wrote into the bundle."""
    source_name: str
    doc_count: int
    chunk_count: int
    section_count: int
    theme_count: int
    edge_count: int
    embedding_model: str
    embedding_dim: int
    includes_embeddings: bool


@frozen
class ImportReport:
    """Summary of a merge of a bundle into an existing database."""
    source_name: str
    documents_merged: int
    conflicts: int


def _cols(conn: sqlite3.Connection, table: str) -> list[str]:
    """Column names of ``table`` in declaration order."""
    return [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]


def _source_exists(conn: sqlite3.Connection, source_name: str) -> bool:
    """True if ``source_name`` is attested anywhere we'd export from."""
    for table in ('documents', 'source_relations', 'scip_index_state'):
        if conn.execute(
            f'SELECT 1 FROM {table} WHERE source_name=? LIMIT 1', (source_name,)
        ).fetchone():
            return True
    return False


def _copy_rows(src, dst, table, rows) -> int:
    """INSERT OR REPLACE ``rows`` (aligned to ``dst``'s columns) into ``table``."""
    cols = _cols(dst, table)
    placeholders = ','.join('?' * len(cols))
    dst.executemany(
        f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})",
        rows,
    )
    return len(rows)


def _select_for_dest(src, dst, table, where, params):
    """SELECT the destination table's columns from the source table."""
    cols = _cols(dst, table)
    return cols, src.execute(
        f"SELECT {','.join(cols)} FROM {table} WHERE {where}", params
    ).fetchall()


def _null_embedding(cols, rows):
    """Return ``rows`` with the ``embedding`` column blanked (drops vectors)."""
    ei = cols.index('embedding')
    return [tuple(None if i == ei else v for i, v in enumerate(r)) for r in rows]


def _namespace_locals(sym_rows, sym_cols, edge_rows, edge_cols):
    """Rewrite ``local N`` symbol ids to ``{source}::local N``.

    ``local N`` ids are renumbered per ``scip merge`` and so are NOT stable
    across instances; namespacing them by owning source means merging a slice
    into a populated DB can't collide one onto an unrelated local on the
    ``canonical_id`` primary key. Edge endpoints get the same rewrite so the
    graph stays internally consistent.
    """
    cid_i = sym_cols.index('canonical_id')
    src_i = sym_cols.index('source_name')
    remap = {
        r[cid_i]: f'{r[src_i]}::{r[cid_i]}'
        for r in sym_rows if r[cid_i].startswith('local ')
    }
    sym_rows = [
        tuple(remap.get(v, v) if i == cid_i else v for i, v in enumerate(r))
        for r in sym_rows
    ]
    cal_i = edge_cols.index('caller_canonical_id')
    cle_i = edge_cols.index('callee_canonical_id')
    edge_rows = [
        tuple(remap.get(v, v) if i in (cal_i, cle_i) else v for i, v in enumerate(r))
        for r in edge_rows
    ]
    return sym_rows, edge_rows


def _export_scip(src, dst, source_name):
    """Copy the source's SCIP symbols + their 1-hop neighbors and the edges
    touching them, namespacing local ids. Reachability (not just
    both-endpoints-in-source) is what lets callers/callees resolve in the slice.
    """
    s0 = 'SELECT canonical_id FROM scip_symbols WHERE source_name = ?'
    touching = f'caller_canonical_id IN ({s0}) OR callee_canonical_id IN ({s0})'
    edge_cols = _cols(dst, 'scip_edges')
    edge_rows = src.execute(
        f"SELECT {','.join(edge_cols)} FROM scip_edges WHERE {touching}",
        (source_name, source_name),
    ).fetchall()
    cal_i = edge_cols.index('caller_canonical_id')
    cle_i = edge_cols.index('callee_canonical_id')
    reach = {r[0] for r in src.execute(s0, (source_name,))}
    for e in edge_rows:
        reach.add(e[cal_i])
        reach.add(e[cle_i])
    sym_cols = _cols(dst, 'scip_symbols')
    cid_i = sym_cols.index('canonical_id')
    sym_rows = [
        r for r in src.execute(f"SELECT {','.join(sym_cols)} FROM scip_symbols")
        if r[cid_i] in reach
    ]
    sym_rows, edge_rows = _namespace_locals(sym_rows, sym_cols, edge_rows, edge_cols)
    _copy_rows(src, dst, 'scip_symbols', sym_rows)
    _copy_rows(src, dst, 'scip_edges', edge_rows)
    _state_cols, state_rows = _select_for_dest(src, dst, 'scip_index_state', 'source_name=?', (source_name,))
    _copy_rows(src, dst, 'scip_index_state', state_rows)


def export_source_db(
    source_db_path: str | Path,
    source_name: str,
    out_path: str | Path,
    *,
    include_embeddings: bool = True,
    include_scip: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> SliceManifest:
    """Write a standalone single-source slice of ``source_db_path`` to ``out_path``.

    Args:
        source_db_path: Path to a full Ariadne database.
        source_name: The one source to slice out.
        out_path: Where to write the bundle (a valid standalone ``ariadne.db``).
        include_embeddings: If False, documents/chunks/sections travel without
            their vectors — the recipient must ``rebuild`` (a few dollars).
        include_scip: If True, also carry the source's SCIP call graph — its
            symbols plus their 1-hop cross-source neighbors and the edges
            touching them, so callers/callees resolve. ``local N`` ids are
            namespaced by owning source for cross-instance merge safety.
        embedding_model: Stamped into the bundle so import can refuse a
            target that embeds with a different model.

    Raises:
        SourceNotFoundError: if ``source_name`` is attested nowhere.
    """
    from library import Library

    src = sqlite3.connect(str(source_db_path))
    try:
        if not _source_exists(src, source_name):
            raise SourceNotFoundError(
                f'no data for source {source_name!r} in {source_db_path}'
            )

        Library(Path(out_path)).close()  # initialise the standard schema
        dst = sqlite3.connect(str(out_path))
        try:
            dst.execute(
                'CREATE TABLE IF NOT EXISTS slice_meta (key TEXT PRIMARY KEY, value TEXT)'
            )

            # Documents: the source's own docs PLUS the summary docs of themes
            # that touch the source (theme summaries are written source_name=NULL).
            keep_docs = (
                f'source_name = ? OR id IN (SELECT t.doc_id FROM themes t '
                f'WHERE t.cluster_id IN ({_RELEVANT_CLUSTERS}))'
            )
            dcols, doc_rows = _select_for_dest(
                src, dst, 'documents', keep_docs, (source_name, source_name)
            )
            emb_idx = dcols.index('embedding')
            embs = [r[emb_idx] for r in doc_rows if r[emb_idx] is not None] if include_embeddings else []
            embedding_dim = len(embs[0]) // 4 if embs else 0  # float32 = 4 bytes
            if not include_embeddings:
                doc_rows = _null_embedding(dcols, doc_rows)
            doc_count = _copy_rows(src, dst, 'documents', doc_rows)

            # Chunks + sections for every kept document (same DB, so a subquery
            # sidesteps any SQLite variable-count limit).
            in_keep = f'document_id IN (SELECT id FROM documents WHERE {keep_docs})'
            table_counts = {}
            for table in ('chunks', 'sections'):
                cols, rows = _select_for_dest(src, dst, table, in_keep, (source_name, source_name))
                if not include_embeddings and 'embedding' in cols:
                    rows = _null_embedding(cols, rows)
                table_counts[table] = _copy_rows(src, dst, table, rows)

            # Themes touching the source: theme row + members restricted to
            # in-source elements (no members dangling at absent docs).
            _t, theme_rows = _select_for_dest(
                src, dst, 'themes', f'cluster_id IN ({_RELEVANT_CLUSTERS})', (source_name,)
            )
            theme_count = _copy_rows(src, dst, 'themes', theme_rows)
            tm_where = (
                f'cluster_id IN ({_RELEVANT_CLUSTERS}) '
                f'AND element_id IN (SELECT id FROM documents WHERE source_name = ?)'
            )
            _tm, tm_rows = _select_for_dest(src, dst, 'theme_members', tm_where, (source_name, source_name))
            _copy_rows(src, dst, 'theme_members', tm_rows)

            # doc_graph: edges whose endpoints are both in-source documents.
            dg_where = (
                'source_id IN (SELECT id FROM documents WHERE source_name = ?) '
                'AND target_id IN (SELECT id FROM documents WHERE source_name = ?)'
            )
            _dg, dg_rows = _select_for_dest(src, dst, 'doc_graph', dg_where, (source_name, source_name))
            edge_count = _copy_rows(src, dst, 'doc_graph', dg_rows)

            # Source-scoped relational + sync metadata.
            for table in ('source_relations', 'sync_state'):
                _cols_, rows = _select_for_dest(src, dst, table, 'source_name=?', (source_name,))
                _copy_rows(src, dst, table, rows)

            if include_scip:
                _export_scip(src, dst, source_name)

            dst.execute(
                "INSERT OR REPLACE INTO slice_meta (key, value) VALUES ('embedding_model', ?)",
                (embedding_model,),
            )
            dst.execute(
                "INSERT OR REPLACE INTO slice_meta (key, value) VALUES ('source_name', ?)",
                (source_name,),
            )
            dst.commit()
            dst.execute('PRAGMA wal_checkpoint(TRUNCATE)')  # fold WAL into one file to ship
        finally:
            dst.close()
    finally:
        src.close()

    return SliceManifest(
        source_name=source_name,
        doc_count=doc_count,
        chunk_count=table_counts['chunks'],
        section_count=table_counts['sections'],
        theme_count=theme_count,
        edge_count=edge_count,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        includes_embeddings=include_embeddings,
    )


def _count_existing(conn: sqlite3.Connection, table: str, ids: list[str]) -> int:
    """How many of ``ids`` already exist in ``table`` (batched under SQLite's
    variable limit)."""
    total = 0
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        placeholders = ','.join('?' * len(chunk))
        total += conn.execute(
            f'SELECT COUNT(*) FROM {table} WHERE id IN ({placeholders})', chunk
        ).fetchone()[0]
    return total


def import_source_db(
    target_db_path: str | Path,
    bundle_path: str | Path,
    *,
    on_conflict: str = 'replace',
    expected_embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> ImportReport:
    """Merge a bundle produced by :func:`export_source_db` into an existing DB.

    Args:
        on_conflict: ``'replace'`` (bundle wins), ``'skip'`` (keep existing
            rows), or ``'fail'`` (abort if any document id already exists).
        expected_embedding_model: The target's embedding model; a bundle
            stamped with anything else is rejected.

    Raises:
        ValueError: on an unknown ``on_conflict`` value.
        EmbeddingModelMismatch: if the bundle's stamped model differs.
        ImportConflictError: if ``on_conflict='fail'`` and ids collide.
    """
    from library import Library

    if on_conflict not in ('replace', 'skip', 'fail'):
        raise ValueError(f'invalid on_conflict: {on_conflict!r}')

    bnd = sqlite3.connect(str(bundle_path))
    try:
        stamped = bnd.execute(
            "SELECT value FROM slice_meta WHERE key='embedding_model'"
        ).fetchone()[0]
        if stamped != expected_embedding_model:
            raise EmbeddingModelMismatch(
                f'bundle embedded with {stamped!r}, target expects '
                f'{expected_embedding_model!r}'
            )
        source_name = bnd.execute(
            "SELECT value FROM slice_meta WHERE key='source_name'"
        ).fetchone()[0]

        Library(Path(target_db_path)).close()  # ensure target schema exists
        tgt = sqlite3.connect(str(target_db_path))
        try:
            bundle_doc_ids = [r[0] for r in bnd.execute('SELECT id FROM documents')]
            conflicts = _count_existing(tgt, 'documents', bundle_doc_ids)
            if on_conflict == 'fail' and conflicts:
                raise ImportConflictError(
                    f'{conflicts} documents already exist; aborting (on_conflict=fail)'
                )
            verb = 'INSERT OR IGNORE' if on_conflict == 'skip' else 'INSERT OR REPLACE'

            documents_merged = 0
            for table in _MERGE_TABLES:
                cols = _cols(tgt, table)
                placeholders = ','.join('?' * len(cols))
                rows = bnd.execute(f"SELECT {','.join(cols)} FROM {table}").fetchall()
                tgt.executemany(
                    f"{verb} INTO {table} ({','.join(cols)}) VALUES ({placeholders})", rows
                )
                if table == 'documents':
                    documents_merged = len(rows)
            tgt.commit()
        finally:
            tgt.close()
    finally:
        bnd.close()

    return ImportReport(
        source_name=source_name,
        documents_merged=documents_merged,
        conflicts=conflicts,
    )
