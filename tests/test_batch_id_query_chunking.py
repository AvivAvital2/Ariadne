"""Batch-by-id reads must chunk their ``WHERE id IN (...)`` so a large id
list never exceeds SQLite's variable limit.

Regression for the spool build that crashed in the themes phase with
``sqlite3.OperationalError: too many SQL variables`` — ``get_embeddings_for_ids``
was called with ~38k scoped doc ids in one query. ``get_documents_batch`` and
``get_sections_batch`` had the same unchunked pattern.

The test caps SQLite's variable limit low (well above a single-row insert) so
~150 ids reproduce the crash without needing tens of thousands of rows.
"""
from __future__ import annotations

import sqlite3

import numpy as np

import library.core as core
from library import Library

_VAR_CAP = 100   # SQLite variable ceiling for this test's connections
_N_DOCS = 150    # > _VAR_CAP → a single unchunked IN(...) query blows the cap


def test_batch_id_reads_chunk_under_sqlite_var_limit(tmp_path, monkeypatch):
    real_connect = sqlite3.connect

    def _capped(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _VAR_CAP)
        return conn

    monkeypatch.setattr(sqlite3, 'connect', _capped)
    # Chunk well below the cap. raising=False so the pre-fix module (no such
    # constant) still gets it set — the pre-fix unchunked query then blows the
    # cap, i.e. this test is red before the fix and green after.
    monkeypatch.setattr(core, '_SQL_MAX_VARS', 5, raising=False)

    lib = Library(tmp_path / 'lib.db')
    try:
        ids: list[str] = []
        for i in range(_N_DOCS):
            doc = lib.add_document(
                content_type='explanation',
                title=f'title-{i}',
                content=f'content-{i}',
                source_name='src1',
                embedding=np.array([float(i)], dtype=np.float32),
            )
            ids.append(doc.id)

        embs = lib.get_embeddings_for_ids(ids)
        assert set(embs) == set(ids)              # every id, merged across chunks

        docs = lib.get_documents_batch(ids)
        assert [d.id for d in docs] == ids         # input order preserved across chunks

        secs = lib.get_sections_batch(ids)
        assert set(secs) == set(ids)               # a bucket for every id
    finally:
        lib.close()
