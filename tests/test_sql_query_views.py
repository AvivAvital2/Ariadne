"""Query-view helper for the SQL data model (design §6).

``data_access_for`` answers "who writes / reads this table or column"
directly from ``data_access`` — the cheap, graph-free read path the
``ariadne_data`` MCP tool wraps. It applies the SAME confidence floor as
the graph projection (§3a/§6a: the read boundary is enforced in one
place, two views), and it splits SENT-write from reads so "what *writes*
users.email" is distinct from "what reads it". Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.sql_query_views import data_access_for
from library.scip import init_scip_schema

COLUMN_ID = 'data sql src1 _._.users#email'
WRITER = 'scip-python python src1 . src1/deactivate().'
FILTERER = 'scip-python python src1 . src1/find_user().'
PROJECTOR = 'scip-python python src1 . src1/list_emails().'
DDL_SITE = 'scip-python python src1 . src1/migration_0001().'
GUESSED_WRITER = 'scip-python python src1 . src1/legacy().'


def _access(conn, consumer, role, confidence='resolved'):
    conn.execute(
        'INSERT INTO data_access (source_name, consumer_symbol_id, '
        'schema_symbol_id, role, witness, confidence) VALUES (?,?,?,?,?,?)',
        ('src1', consumer, COLUMN_ID, role, 'rawsql', confidence),
    )


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def test_data_access_for_splits_writers_from_readers_and_honors_threshold(conn):
    # --- given: every role + a below-floor guess ----------------------
    _access(conn, WRITER, 'write')
    _access(conn, FILTERER, 'filter')
    _access(conn, PROJECTOR, 'project')
    _access(conn, DDL_SITE, 'ddl')                               # not app r/w
    _access(conn, GUESSED_WRITER, 'write', confidence='derived')  # below floor

    view = data_access_for(conn, COLUMN_ID)

    # --- then: writers = only the resolved write; readers = filter +
    #           project (sent-predicate + received column), sorted -------
    assert view['writes'] == [WRITER]
    assert view['reads'] == [FILTERER, PROJECTOR]
    # the derived write is held as a gap, the ddl site is not an app access
    assert GUESSED_WRITER not in view['writes'] + view['reads']
    assert DDL_SITE not in view['writes'] + view['reads']

    # --- and: the threshold is the gate — lowering it surfaces the
    #           derived write at the query boundary too -----------------
    relaxed = data_access_for(conn, COLUMN_ID, min_confidence='derived')
    assert set(relaxed['writes']) == {WRITER, GUESSED_WRITER}
