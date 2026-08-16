"""Clews get their own vector surface, not a document content type.

A clew is a pre-generated route through the call graph — ``fit -> withTransformEvent ->
listenerBus -> getOrCreate`` — embedded so a question can match a *path* instead of a
starting symbol. ``scip_edges`` carries 2.78M edges and no embedding column, which is why the
walk has never seen the question: retrieval can only match a symbol.

Storing them as ``documents`` with a new ``content_type`` was the cheap route and it is wrong.
Twenty-two files enumerate content types, so a clew would inherit provenance weighting, gap
analysis, doc-type pickers and export — and any filter that lists types drops it silently. A
route is not prose and must not compete with prose in one ranked list.

So clews follow ``sections``: their own table, their own ``embedding`` column, queried
deliberately. That is the existing pattern for a non-document vector surface, not a new one.

Synthetic fixtures only: source ``src1``.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from library import Library; 

from library.chain_answer import evidence_for; 

from library.scip import init_scip_schema; 

from library.clews import Clew, add_clew, clews_for, init_clews_schema, nearest_clews

SOURCE = 'src1'


@pytest.fixture()
def conn():
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    init_clews_schema(connection)
    yield connection
    connection.close()


def _clew(entry: str, steps: list[str], files: list[str], vector=None) -> dict:
    return {
        'source_name': SOURCE,
        'entry_symbol': entry,
        'steps': steps,
        'route': [f'pkg.{step}' for step in steps],
        'files': files,
        'strategy': 'theme',
        'embedding': None if vector is None else np.asarray(vector, dtype=np.float32),
    }


class TestAClewIsItsOwnSurface:
    """Stored, scoped and retrieved without touching ``documents``."""

    def test_a_stored_clew_returns_its_route_and_the_files_to_seed_from(self, conn):
        add_clew(conn, **_clew('pkg.run', ['run', 'load', 'write'], ['a.py', 'b.py']))

        stored = clews_for(conn, source_name=SOURCE)

        assert len(stored) == 1
        clew = stored[0]
        assert isinstance(clew, Clew)
        assert clew.steps == ['run', 'load', 'write']
        assert clew.route == ['pkg.run', 'pkg.load', 'pkg.write']
        assert clew.files == ['a.py', 'b.py'], 'the files are what seeds a walk'
        assert clew.hops == 2, 'hops is derived from the route, never stored twice'

    def test_a_clew_belongs_to_one_source(self, conn):
        add_clew(conn, **_clew('pkg.run', ['run', 'load', 'write'], ['a.py']))
        other = _clew('pkg.other', ['other', 'load', 'write'], ['c.py'])
        other['source_name'] = 'src2'
        add_clew(conn, **other)

        assert len(clews_for(conn, source_name=SOURCE)) == 1
        assert len(clews_for(conn, source_name='src2')) == 1

    def test_the_same_route_stored_twice_is_one_clew(self, conn):
        """Generation pools several strategies, and they overlap by design.

        Measured on the databricks pack, pooling three strategies is worth +14 points of
        coverage over the best single one — so the same route arrives more than once and the
        store, not the generator, has to be the thing that dedupes.
        """
        add_clew(conn, **_clew('pkg.run', ['run', 'load', 'write'], ['a.py']))
        add_clew(conn, **_clew('pkg.run', ['run', 'load', 'write'], ['a.py']))

        assert len(clews_for(conn, source_name=SOURCE)) == 1

    def test_nearest_returns_the_closest_route_by_cosine(self, conn):
        add_clew(conn, **_clew('pkg.near', ['near', 'load', 'write'], ['a.py'],
                               vector=[1.0, 0.0, 0.0]))
        add_clew(conn, **_clew('pkg.far', ['far', 'load', 'write'], ['b.py'],
                               vector=[0.0, 1.0, 0.0]))

        found = nearest_clews(conn, np.asarray([0.9, 0.1, 0.0], dtype=np.float32),
                              source_name=SOURCE, top_k=1)

        assert [c.entry_symbol for c in found] == ['pkg.near']

    def test_a_clew_without_an_embedding_is_never_returned_as_nearest(self, conn):
        """An unembedded clew is not yet usable, and must not be silently ranked as distant.

        Embedding happens after generation and needs a provider key, so a pack can hold
        clews that are stored but not embedded. Treating a missing vector as a zero vector
        would make them the answer to every query.
        """
        add_clew(conn, **_clew('pkg.unembedded', ['unembedded', 'load', 'write'], ['a.py']))

        found = nearest_clews(conn, np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                              source_name=SOURCE, top_k=5)

        assert found == []
RUN = 'scip-python python src1 0.1 `m`/run().'
HELPER = 'scip-python python src1 0.1 `m`/helper().'


def _symbol(conn, cid, *, file, qn, line_start, line_end, parent=''):
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
        'line_start, line_end, kind, display_name, qualified_name, '
        'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (cid, SOURCE, 'python', file, line_start, line_end, '', '', qn, parent))
def _edge(conn, caller, callee, *, line, file='m.py'):
    conn.execute(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
        "edge_type, file, line, confidence) VALUES (?,?,'call',?,?,'exact')",
        (caller, callee, file, line))
class TestAClewCanPositionTheWalk:
    """A matched route seeds the walk. That is the whole point of storing clews.

    Retrieval today hands ``evidence_for`` documents, and their files become seeds — so a
    question can only influence *where the walk starts* through whatever prose retrieval
    matched. A clew is a route the walk already found, so passing its symbols seeds the walk
    directly on the path. Measured on the databricks pack, pooling clew strategies contains
    92.8% of the symbols answer keys require against 66.0% for one document-seeded walk.

    This asserts the mechanism at its strongest: **no documents at all**, and the chain still
    exists because the clew positioned it.
    """

    def test_clew_symbols_seed_the_walk_when_retrieval_returned_nothing(self, tmp_path):
        library = Library(tmp_path / 'clew.db')
        with library._conn_provider.acquire() as conn:
            init_scip_schema(conn)
            _symbol(conn, RUN, file='m.py', qn='m.run', line_start=5, line_end=20)
            _symbol(conn, HELPER, file='h.py', qn='m.helper', line_start=3, line_end=9)
            _edge(conn, RUN, HELPER, line=8)
            conn.commit()

        without = evidence_for(library, [], source=SOURCE)
        with_clew = evidence_for(library, [], source=SOURCE, clew_symbols=['m.run'])

        assert without.spine == '', 'no documents and no clew means no chain'
        assert with_clew.spine, 'the clew alone must be able to seed the walk'
        assert 'm.helper' in with_clew.spine, (
            'the walk continues from where the clew put it')
