"""The doc-to-SCIP seam: turning retrieved documents into seeds the graph can walk.

Measured on the live store, **0 of 91 files retrieved at production width join
``scip_symbols.file`` as stored**. Document paths are either repo-prefixed
(``delta/spark/src/...`` where SCIP holds ``spark/src/...``) or absolute
(``/Users/.../config.py``), because a spool corpus is indexed once per repo with the
repo as the indexer ``cwd``. Without this seam the ``(source, file, line_start)`` seek
that ``designs/answer-path.md`` §4.2 selects returns nothing, and no retrieved document
can seed a walk.

The strip candidates come from the indexer manifest — never a hardcoded repo list — and
each is **verified against the index** rather than trusted. The existing
``resolve_scip_file`` trusts the longest matching ``cwd`` and so over-strips
``spark/sql/core/...`` to ``core/...``: 36 of 91 against 59 for verification.

Synthetic fixtures only: source ``src1``.
"""
from __future__ import annotations

import sqlite3

import pytest

from library.scip import init_scip_schema
from library.structural_assembly import scip_paths_for, seeds_from_documents

# The index holds BOTH of these as real, distinct files — so preferring the longest
# strip would silently resolve the wrong one.
INDEXED = ('sql/core/app.py', 'core/app.py', 'pkg/beta.py')
CWDS = ('spark', 'spark/sql', 'repo')


def _symbol(conn, cid, *, file, qn, line_start=1, source='src1'):
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
        'line_start, line_end, kind, display_name, qualified_name, '
        'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (cid, source, 'python', file, line_start, line_start + 2, '', '', qn, ''),
    )


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    for n, file in enumerate(INDEXED, start=1):
        _symbol(c, f'scip-python python src1 0.1 `m{n}`/fn().', file=file,
                qn=f'm{n}.fn', line_start=n * 10)
    # named uniquely -> can anchor; named twice -> ambiguous, must be dropped
    _symbol(c, 'scip-python python src1 0.1 `m4`/Solitary#', file='sql/core/app.py',
            qn='m4.Solitary')
    c.execute("UPDATE scip_symbols SET display_name='Solitary' "
              "WHERE canonical_id LIKE '%m4%'")
    for n in (5, 6):
        _symbol(c, f'scip-python python src1 0.1 `m{n}`/Doubled#', file='core/app.py',
                qn=f'm{n}.Doubled')
    c.execute("UPDATE scip_symbols SET display_name='Doubled' "
              "WHERE canonical_id LIKE '%m5%' OR canonical_id LIKE '%m6%'")
    # a local binding in an indexed file: never a seed
    _symbol(c, 'local 1', file='pkg/beta.py', qn='local 1', line_start=4)
    # a same-named file in another source: must not be reachable from src1
    _symbol(c, 'scip-python python src2 0.1 `other`/fn().', file='pkg/beta.py',
            qn='other.fn', source='src2')
    c.commit()
    yield c
    c.close()


def test_the_least_strip_that_the_index_confirms_wins(conn):
    """`spark/sql/core/app.py` is `sql/core/app.py`, not `core/app.py`."""
    resolved, unresolved = scip_paths_for(
        conn, ['spark/sql/core/app.py'], source='src1', indexer_cwds=CWDS)

    assert resolved == {'spark/sql/core/app.py': 'sql/core/app.py'}
    assert unresolved == ()


def test_a_path_already_in_index_form_is_left_alone(conn):
    resolved, _ = scip_paths_for(conn, ['pkg/beta.py'], source='src1',
                                indexer_cwds=CWDS)

    assert resolved == {'pkg/beta.py': 'pkg/beta.py'}


def test_an_unindexed_path_is_reported_not_guessed(conn):
    """32 of the live 91 are genuinely absent — markdown, and an unindexed module."""
    resolved, unresolved = scip_paths_for(
        conn, ['delta/PROTOCOL.md', 'repo/pkg/beta.py'], source='src1',
        indexer_cwds=CWDS)

    assert resolved == {'repo/pkg/beta.py': 'pkg/beta.py'}
    assert unresolved == ('delta/PROTOCOL.md',)


def test_seeds_are_the_named_symbols_defined_in_the_resolved_files(conn):
    """Seeds carry no locals and nothing from another source."""
    result = seeds_from_documents(
        conn, [{'source_files': ['repo/pkg/beta.py']}], source='src1',
        indexer_cwds=CWDS)

    assert result.seeds == ['scip-python python src1 0.1 `m3`/fn().']
    assert result.from_files == 1
    assert result.unresolved_paths == ()


def test_documents_without_source_files_contribute_nothing_and_do_not_error(conn):
    result = seeds_from_documents(
        conn, [{'source_files': None}, {}], source='src1', indexer_cwds=CWDS)

    assert result.seeds == []
    assert result.unresolved_paths == ()


def test_a_prose_document_seeds_from_the_symbols_its_text_names(conn):
    """A markdown doc has no code symbols in its FILE, but names them in its body.

    Measured: 24 of the 32 discarded files name at least one indexed symbol, so the file
    route alone leaks half the retrieved evidence.
    """
    result = seeds_from_documents(
        conn,
        [{'source_files': ['delta/docs/guide.md'],
          'content': 'MERGE is planned by Solitary and then executed.'}],
        source='src1', indexer_cwds=CWDS)

    assert result.seeds == ['scip-python python src1 0.1 `m4`/Solitary#']
    assert result.from_mentions == 1
    assert result.from_files == 0
    assert result.unresolved_paths == ('delta/docs/guide.md',)


def test_an_ambiguous_mention_is_dropped_and_counted(conn):
    """`Doubled` names two symbols, so it cannot anchor anything — and it is reported."""
    result = seeds_from_documents(
        conn,
        [{'source_files': ['delta/docs/guide.md'],
          'content': 'The Doubled helper is used widely.'}],
        source='src1', indexer_cwds=CWDS)

    assert result.seeds == []
    assert result.ambiguous_mentions == 1


def test_the_file_route_wins_and_suppresses_scraping(conn):
    """A document whose file resolves is not text-scraped — no flooding the seed set."""
    result = seeds_from_documents(
        conn,
        [{'source_files': ['repo/pkg/beta.py'],
          'content': 'This also mentions Solitary and Doubled.'}],
        source='src1', indexer_cwds=CWDS)

    assert result.seeds == ['scip-python python src1 0.1 `m3`/fn().']
    assert result.from_mentions == 0
    assert result.ambiguous_mentions == 0
