"""One owner for the doc↔SCIP path seam.

Measured on the live store: **0 of 91 files retrieved at production width joined
``scip_symbols.file`` as stored.** Documents carry a repo-name prefix
(``delta/spark/src/...``) because a spool corpus is indexed once per repo with the repo as
the indexer ``cwd``; an ordinary source's document paths are absolute. So the
``(source, file, line_start)`` seek the answer path depends on returns nothing, and every
consumer that needed the join reinvented it — three times, differently.

Two rules, both measured:

* **Verify, don't trust.** Trusting the longest matching ``cwd`` over-strips
  ``spark/sql/core/...`` to ``core/...``: 36 of 91. Verifying each candidate against the
  index: 59 of 91.
* **Least strip wins.** The correct answer preserves the most path, so ``spark`` beats
  ``spark/sql`` when both are prefixes and both verify.

A path that resolves to nothing is **reported**, never guessed: of the live 91, 32 are
genuinely absent — 20 markdown/html, and 12 ``.java``/``.scala`` under a module that was
never indexed. "Not indexed" and "no such symbol" are different answers.

Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.scip_paths import scip_paths_for
from library.scip import init_scip_schema
import json

# The index holds BOTH as real, distinct files, so preferring the longest strip silently
# resolves the wrong one.
INDEXED = ('sql/core/app.py', 'core/app.py', 'pkg/beta.py')
CWDS = ('spark', 'spark/sql', 'repo')


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    for n, file in enumerate(INDEXED, start=1):
        c.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
            'line_start, line_end, kind, display_name, qualified_name, '
            "parent_qualified_name) VALUES (?,'src1','python',?,1,3,'','',?,'')",
            (f'sym{n}', file, f'm{n}.fn'))
    c.commit()
    yield c
    c.close()


def _resolve(conn, *paths, root=None):
    return scip_paths_for(conn, list(paths), source='src1', indexer_cwds=CWDS,
                          source_root=root)


def test_the_least_strip_the_index_confirms_wins(conn):
    resolved, unresolved = _resolve(conn, 'spark/sql/core/app.py')

    assert resolved == {'spark/sql/core/app.py': 'sql/core/app.py'}
    assert unresolved == ()


def test_a_path_already_in_index_form_is_left_alone(conn):
    resolved, _ = _resolve(conn, 'pkg/beta.py')

    assert resolved == {'pkg/beta.py': 'pkg/beta.py'}


def test_an_absolute_path_is_made_relative_to_the_source_root(conn):
    """An ordinary source's document paths are absolute — `/Users/.../config.py`."""
    resolved, _ = _resolve(conn, '/srv/checkout/pkg/beta.py', root='/srv/checkout')

    assert resolved == {'/srv/checkout/pkg/beta.py': 'pkg/beta.py'}


def test_an_unindexed_path_is_reported_not_guessed(conn):
    resolved, unresolved = _resolve(conn, 'delta/PROTOCOL.md', 'repo/pkg/beta.py')

    assert resolved == {'repo/pkg/beta.py': 'pkg/beta.py'}
    assert unresolved == ('delta/PROTOCOL.md',)


def test_a_cwd_that_is_not_a_prefix_is_not_applied(conn):
    """`repo` must not be stripped from a path that merely contains it."""
    resolved, unresolved = _resolve(conn, 'vendor/repo/pkg/beta.py')

    assert resolved == {}
    assert unresolved == ('vendor/repo/pkg/beta.py',)


def test_a_dot_cwd_is_ignored_rather_than_stripping_nothing_forever(conn):
    resolved, _ = scip_paths_for(conn, ['pkg/beta.py'], source='src1',
                                 indexer_cwds=('.', ''), source_root=None)

    assert resolved == {'pkg/beta.py': 'pkg/beta.py'}


def test_the_index_is_queried_once_per_candidate_not_once_per_path(conn):
    """The seam runs per retrieved document set; repeated candidates must be cached."""
    calls: list[str] = []
    conn.set_trace_callback(lambda sql: calls.append(sql)
                            if 'scip_symbols' in sql else None)
    try:
        _resolve(conn, 'pkg/beta.py', 'pkg/beta.py', 'pkg/beta.py')
    finally:
        conn.set_trace_callback(None)

    assert len(calls) == 1


def test_a_source_with_no_index_reports_everything_unresolved(conn):
    resolved, unresolved = scip_paths_for(conn, ['pkg/beta.py'], source='absent',
                                         indexer_cwds=CWDS)

    assert resolved == {}
    assert unresolved == ('pkg/beta.py',)


def test_indexer_cwds_come_from_the_manifest_not_a_hardcoded_list(tmp_path):
    """A multi-package source is indexed once per package, each with its own cwd."""
    import json

    from docgen.scip_paths import indexer_cwds

    (tmp_path / '.ariadne').mkdir()
    (tmp_path / '.ariadne' / 'manifest.json').write_text(json.dumps({
        'indexers': [{'cwd': 'backend'}, {'cwd': 'frontend'}, {'kind': 'no-cwd'}],
    }))

    assert indexer_cwds(tmp_path) == ('backend', 'frontend', '.')


def test_a_source_with_no_manifest_falls_back_to_the_single_root_default(tmp_path):
    from docgen.scip_paths import indexer_cwds

    assert indexer_cwds(tmp_path) == ('.',)
