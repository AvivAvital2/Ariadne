"""`purge_source` — remove ALL of a source's data from the library DB.

`ariadne source remove` only edits ariadne.yaml; a removed source's
documents, chunks, and SCIP rows are left orphaned (they still occupy the
DB and still rank in semantic search via the embedding matrix). This pins
the DB-level purge: the documents cascade (chunks/sections/theme_members/
doc_graph/themes, mirroring migrations._delete_doc_with_refs) plus every
source-scoped auxiliary table, discovered from the live schema so a newly
added table is covered automatically. Cross-source isolation is the
load-bearing property. Synthetic fixtures only.
"""
from __future__ import annotations

import pytest

from library import Library
from library.purge import PurgeSummary, purge_source, source_scoped_aux_tables


def _add_doc(lib, doc_id, source_name):
    lib.add_document(
        'catalog', f'{doc_id} title', f'body of {doc_id}',
        source_files=[f'{source_name}/f.py'],
        metadata={'qualified_name': f'{source_name}.pkg.{doc_id}'},
        doc_id=doc_id,
        source_name=source_name,
    )


def _seed(conn, source_name, doc_id):
    """Insert one aux/child row into every table the purge must clear."""
    conn.execute(
        'INSERT INTO chunks (id, document_id, chunk_index, content, embedding) '
        'VALUES (?, ?, 0, ?, NULL)',
        (f'{doc_id}-c0', doc_id, 'chunk body'),
    )
    conn.execute(
        'INSERT INTO sections (document_id, idx, heading, description, content, embedding) '
        'VALUES (?, 0, ?, ?, ?, NULL)',
        (doc_id, 'h', 'd', 'section body'),
    )
    conn.execute(
        'INSERT INTO themes (cluster_id, doc_id, member_count, resolution, '
        'last_built_at, last_summarized_at, summary_hash, coherent, dirty, association) '
        "VALUES (?, ?, 1, 1.0, '2026-01-01', '2026-01-01', 'h', 1, 0, '')",
        (f'cl-{doc_id}', doc_id),
    )
    conn.execute(
        'INSERT INTO theme_members (cluster_id, element_id, weight, joined_at) '
        "VALUES (?, ?, 1.0, '2026-01-01')",
        (f'cl-{doc_id}', doc_id),
    )
    conn.execute(
        'INSERT INTO doc_graph (source_id, target_id, edge_type, weight) '
        "VALUES (?, ?, 'imports', 1.0)",
        (doc_id, f'{doc_id}-other'),
    )
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
        'line_start, line_end, kind, display_name, qualified_name) '
        "VALUES (?, ?, 'python', ?, 1, 2, 'function', ?, ?)",
        (f'{doc_id}#sym', source_name, f'{source_name}/f.py',
         'sym', f'{source_name}.pkg.{doc_id}'),
    )
    conn.execute(
        'INSERT INTO config_values (source_name, file, key, value, line_start) '
        'VALUES (?, ?, ?, ?, 1)',
        (source_name, f'{source_name}/app.conf', f'timeout.{doc_id}', '30'),
    )


@pytest.fixture
def seeded_lib(tmp_path):
    with Library(tmp_path / 'lib.db') as lib:
        _add_doc(lib, 'src1-d1', 'src1')
        _add_doc(lib, 'src1-d2', 'src1')
        _add_doc(lib, 'src2-d1', 'src2')
        with lib._conn_provider.acquire() as conn:
            _seed(conn, 'src1', 'src1-d1')
            _seed(conn, 'src1', 'src1-d2')
            _seed(conn, 'src2', 'src2-d1')
        yield lib


def _counts(lib, source_name):
    """Row counts still attributable to `source_name` across every table
    the purge should touch."""
    with lib._conn_provider.acquire() as conn:
        subq = 'SELECT id FROM documents WHERE source_name = ?'
        out = {
            'documents': conn.execute(
                'SELECT COUNT(*) FROM documents WHERE source_name = ?',
                (source_name,)).fetchone()[0],
            'chunks': conn.execute(
                f'SELECT COUNT(*) FROM chunks WHERE document_id IN ({subq})',
                (source_name,)).fetchone()[0],
            'sections': conn.execute(
                f'SELECT COUNT(*) FROM sections WHERE document_id IN ({subq})',
                (source_name,)).fetchone()[0],
            'theme_members': conn.execute(
                f'SELECT COUNT(*) FROM theme_members WHERE element_id IN ({subq})',
                (source_name,)).fetchone()[0],
            'doc_graph': conn.execute(
                f'SELECT COUNT(*) FROM doc_graph WHERE source_id IN ({subq})',
                (source_name,)).fetchone()[0],
            'scip_symbols': conn.execute(
                'SELECT COUNT(*) FROM scip_symbols WHERE source_name = ?',
                (source_name,)).fetchone()[0],
            'config_values': conn.execute(
                'SELECT COUNT(*) FROM config_values WHERE source_name = ?',
                (source_name,)).fetchone()[0],
        }
    return out


class TestPurgeSource:
    def test_purges_target_source_completely(self, seeded_lib):
        summary = purge_source(seeded_lib, 'src1')
        after = _counts(seeded_lib, 'src1')
        assert after == {k: 0 for k in after}, f'residue after purge: {after}'
        assert isinstance(summary, PurgeSummary)
        assert summary.counts['documents'] == 2
        assert summary.counts['chunks'] == 2
        assert summary.counts['scip_symbols'] == 2
        assert summary.counts['config_values'] == 2
        assert summary.total >= 2 + 2 + 2 + 2

    def test_leaves_other_sources_intact(self, seeded_lib):
        purge_source(seeded_lib, 'src1')
        other = _counts(seeded_lib, 'src2')
        assert other['documents'] == 1
        assert other['chunks'] == 1
        assert other['sections'] == 1
        assert other['theme_members'] == 1
        assert other['doc_graph'] == 1
        assert other['scip_symbols'] == 1
        assert other['config_values'] == 1

    def test_dry_run_counts_without_deleting(self, seeded_lib):
        summary = purge_source(seeded_lib, 'src1', dry_run=True)
        assert summary.dry_run is True
        assert summary.counts['documents'] == 2
        # Nothing removed.
        assert _counts(seeded_lib, 'src1')['documents'] == 2
        assert _counts(seeded_lib, 'src1')['scip_symbols'] == 2

    def test_discovery_covers_aux_tables_excludes_documents(self, seeded_lib):
        with seeded_lib._conn_provider.acquire() as conn:
            aux = source_scoped_aux_tables(conn)
        assert 'scip_symbols' in aux
        assert 'config_values' in aux
        assert 'documents' not in aux   # handled by the doc cascade, not the sweep
