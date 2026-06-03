"""Producer-internal schema invariants for the core library tables.

``Library()`` runs schema migrations on open; the resulting
``documents`` / ``chunks`` / ``usage_events`` / ``doc_graph`` tables
back every read path in the codebase (MCP search, list, get_document,
hit/miss tracking, the graph injector). If any of these columns
disappears, the affected queries either error out at runtime or
silently return wrong data.

Salvaged from ``tests/contract/test_schema_invariants.py`` (the
slim-consumer fork). Producer-internal value stands on its own.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from library import Library


DOCUMENTS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'id', 'content_type', 'title', 'content',
    'source_files', 'embedding', 'metadata',
    'source_name', 'created_at', 'updated_at',
})

CHUNKS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'id', 'document_id', 'chunk_index', 'content', 'embedding',
})

USAGE_EVENTS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'id', 'timestamp', 'tool_name', 'query',
    'result_count', 'outcome', 'feedback',
    'returned_document_ids',
})

DOC_GRAPH_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'source_id', 'target_id', 'edge_type', 'weight',
})


@pytest.fixture
def fresh_library(tmp_path: Path):
    lib = Library(tmp_path / 'ariadne.db')
    yield lib
    lib.close()


def _columns(library: Library, table: str) -> set[str]:
    with library._conn_provider.acquire() as conn:
        rows = conn.execute(
            f'SELECT name FROM pragma_table_info("{table}")',
        ).fetchall()
    return {row[0] for row in rows}


@pytest.mark.parametrize(
    ('table', 'required'),
    [
        ('documents', DOCUMENTS_REQUIRED_COLUMNS),
        ('chunks', CHUNKS_REQUIRED_COLUMNS),
        ('usage_events', USAGE_EVENTS_REQUIRED_COLUMNS),
        ('doc_graph', DOC_GRAPH_REQUIRED_COLUMNS),
    ],
)
def test_required_columns_present(
    fresh_library: Library, table: str, required: frozenset[str],
) -> None:
    """A fresh ``Library`` carries every column the read paths name."""
    actual = _columns(fresh_library, table)
    missing = required - actual
    assert not missing, (
        f'{table} missing columns: {sorted(missing)}. '
        f'Either restore them in library.py / library_*.py migrations '
        f'or update callers + this test.'
    )


def test_documents_table_exists(fresh_library: Library) -> None:
    """The ``documents`` table is the primary read surface — its
    absence is catastrophic."""
    assert _columns(fresh_library, 'documents'), 'documents table is missing entirely'
