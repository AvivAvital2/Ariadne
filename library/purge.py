"""Remove ALL of a source's data from the library DB.

``ariadne source remove`` only edits ariadne.yaml — a removed source's
documents, chunks, and SCIP rows stay in the DB (occupying space and
still ranking in semantic search via the embedding matrix). ``purge_source``
is the DB-level counterpart: it deletes the source's documents with the
same cascade as ``migrations._delete_doc_with_refs`` (chunks / sections /
theme_members / doc_graph / themes), plus every *source-scoped auxiliary
table* — the SCIP + config + sync tables keyed by ``source_name``.

The auxiliary set is DISCOVERED from the live schema (``source_scoped_aux_tables``)
rather than hard-coded, so a newly added source-scoped table is purged
automatically instead of silently orphaning rows.

It does NOT touch the embedding matrix — that is a filesystem artifact the
caller rebuilds afterwards (otherwise search still ranks against the
now-deleted vectors). ``dry_run=True`` counts what would go without
deleting anything.
"""
from __future__ import annotations

from dataclasses import dataclass

# Children keyed by a document id (not source_name) — mirrors the authoritative
# cascade in migrations._delete_doc_with_refs. Each entry: (table, id_column).
# doc_graph is handled specially (two id columns).
_DOC_CHILD_TABLES = (
    ('chunks', 'document_id'),
    ('sections', 'document_id'),
    ('theme_members', 'element_id'),
    ('themes', 'doc_id'),
)

_DOC_SUBQUERY = 'SELECT id FROM documents WHERE source_name = ?'


@dataclass(frozen=True)
class PurgeSummary:
    """What ``purge_source`` removed (or, for a dry run, would remove).

    ``counts`` maps each touched table to its affected row count.
    """
    source_name: str
    dry_run: bool
    counts: dict

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def source_scoped_aux_tables(conn) -> list[str]:
    """Every table carrying a ``source_name`` column EXCEPT ``documents``.

    ``documents`` is excluded because its rows (and their doc-id-keyed
    children) are removed through the document cascade, not this per-source
    sweep. Discovered from the schema so the purge tracks new tables.
    """
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    scoped = []
    for table in tables:
        columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
        if 'source_name' in columns and table != 'documents':
            scoped.append(table)
    return sorted(scoped)


def purge_source(library, source_name: str, *, dry_run: bool = False) -> PurgeSummary:
    """Delete every row attributable to ``source_name`` from the library DB.

    Runs in a single transaction: count first (so the summary is exact even
    for a dry run), then — unless ``dry_run`` — delete the doc-id-keyed
    children, the source-scoped auxiliary tables, and finally the documents
    themselves (last, so the child subqueries still resolve).
    """
    counts: dict = {}
    with library._conn_provider.acquire() as conn:
        aux_tables = source_scoped_aux_tables(conn)

        def _count(sql: str, params: tuple) -> int:
            return int(conn.execute(sql, params).fetchone()[0])

        counts['documents'] = _count(
            'SELECT COUNT(*) FROM documents WHERE source_name = ?', (source_name,)
        )
        for table, id_col in _DOC_CHILD_TABLES:
            counts[table] = _count(
                f'SELECT COUNT(*) FROM {table} WHERE {id_col} IN ({_DOC_SUBQUERY})',
                (source_name,),
            )
        counts['doc_graph'] = _count(
            f'SELECT COUNT(*) FROM doc_graph WHERE source_id IN ({_DOC_SUBQUERY}) '
            f'OR target_id IN ({_DOC_SUBQUERY})',
            (source_name, source_name),
        )
        for table in aux_tables:
            counts[table] = _count(
                f'SELECT COUNT(*) FROM {table} WHERE source_name = ?', (source_name,)
            )

        if not dry_run:
            for table, id_col in _DOC_CHILD_TABLES:
                conn.execute(
                    f'DELETE FROM {table} WHERE {id_col} IN ({_DOC_SUBQUERY})',
                    (source_name,),
                )
            conn.execute(
                f'DELETE FROM doc_graph WHERE source_id IN ({_DOC_SUBQUERY}) '
                f'OR target_id IN ({_DOC_SUBQUERY})',
                (source_name, source_name),
            )
            for table in aux_tables:
                conn.execute(
                    f'DELETE FROM {table} WHERE source_name = ?', (source_name,)
                )
            # Documents LAST — the child/graph subqueries above resolve against it.
            conn.execute('DELETE FROM documents WHERE source_name = ?', (source_name,))

    return PurgeSummary(source_name=source_name, dry_run=dry_run, counts=counts)
