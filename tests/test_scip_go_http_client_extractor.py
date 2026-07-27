"""Contract for the SCIP-driven Go net/http client extractor.

Mirrors the JS/TS (8b.2) and Python (8b.1) client extractors:

- **SCIP** classifies client call sites via ``DEFAULT_SINK_REGISTRY``
  filtered to ``language='go'`` (suffix match on the occurrence symbol).
- **ast-grep / tree-sitter-go** parses the call and locates the URL
  argument at the sink's ``arg_index``.
- The URL literal is read DIRECTLY from the AST node
  (``interpreted_string_literal`` / ``raw_string_literal``); Go isn't in
  ``string_literals`` and net/http URLs are literals in practice.

Output: rows in ``http_client_calls`` (``confidence='literal'``).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_go_http_client_extractor import ingest_go_http_clients
from docgen.scip_string_literal_extractor import ingest_string_literals

_PRE = 'scip-go gomod example.com/m v1 '
_HTTP_GET = _PRE + 'net/http/Get().'
_HTTP_POST = _PRE + 'net/http/Post().'
_CLIENT_GET = _PRE + 'net/http/Client#Get().'
_NEWREQUEST = _PRE + 'net/http/NewRequest().'


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _occ_at(text: str, marker: str, symbol: str) -> _ScipOccurrence:
    for lineno, line in enumerate(text.splitlines()):
        col = 0
        while True:
            i = line.find(marker, col)
            if i < 0:
                break
            before = i == 0 or not (line[i - 1].isalnum() or line[i - 1] == '_')
            j = i + len(marker)
            after = j >= len(line) or not (line[j].isalnum() or line[j] == '_')
            if before and after:
                return _ScipOccurrence(
                    symbol=symbol, range=(lineno, i, lineno, j),
                    is_definition=False,
                )
            col = i + 1
    raise ValueError(f'{marker!r} not found as a word in text')


def _run(tmp_path, conn, filename, text, occs, source_name='svc') -> int:
    """Pipeline: index string literals first (resolve_arg_value reads the
    URL from that seam), then extract client calls — matching the persist
    chain order."""
    (tmp_path / filename).write_text(text)
    index = ScipIndex(
        documents=(_ScipDoc(
            relative_path=filename, occurrences=tuple(occs), symbols=(),
        ),),
        source_root=tmp_path,
    )
    ingest_string_literals(
        source_name=source_name, source_root=tmp_path,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_go_http_clients(
        source_name=source_name, source_root=tmp_path,
        conn=conn, index_factory=lambda: index,
    )


def _calls(conn: sqlite3.Connection, source_name: str) -> list[tuple]:
    cur = conn.execute(
        'SELECT raw_url, http_method, sink_name, confidence '
        'FROM http_client_calls WHERE source_name = ? ORDER BY raw_url',
        (source_name,),
    )
    return list(cur.fetchall())


def test_http_get_literal_url(tmp_path, conn):
    text = 'package main\nfunc f() { http.Get("https://api.x/v1/a") }\n'
    n = _run(tmp_path, conn, 'c.go', text, [_occ_at(text, 'Get', _HTTP_GET)])
    assert n == 1
    assert _calls(conn, 'svc') == [
        ('https://api.x/v1/a', 'GET', 'net/http.Get', 'literal'),
    ]


def test_http_post_literal_url(tmp_path, conn):
    text = 'package main\nfunc f() { http.Post("https://api.x/v1/b", ct, body) }\n'
    _run(tmp_path, conn, 'c.go', text, [_occ_at(text, 'Post', _HTTP_POST)])
    rows = _calls(conn, 'svc')
    assert rows and rows[0][1] == 'POST' and rows[0][0] == 'https://api.x/v1/b'


def test_client_method_url(tmp_path, conn):
    text = 'package main\nfunc f() { client.Get("https://api.x/v1/c") }\n'
    _run(tmp_path, conn, 'c.go', text, [_occ_at(text, 'Get', _CLIENT_GET)])
    rows = _calls(conn, 'svc')
    assert rows and rows[0][2] == 'net/http.Client.Get'


def test_newrequest_url_is_second_arg(tmp_path, conn):
    """NewRequest(method, url, body): the URL is arg 1, method unresolved."""
    text = ('package main\n'
            'func f() { http.NewRequest("GET", "https://api.x/v1/d", nil) }\n')
    _run(tmp_path, conn, 'c.go', text, [_occ_at(text, 'NewRequest', _NEWREQUEST)])
    rows = _calls(conn, 'svc')
    assert rows and rows[0][0] == 'https://api.x/v1/d'
    assert rows[0][1] is None  # method arg not resolved yet


def test_non_literal_url_is_skipped(tmp_path, conn):
    text = 'package main\nfunc f() { http.Get(baseURL) }\n'
    n = _run(tmp_path, conn, 'c.go', text, [_occ_at(text, 'Get', _HTTP_GET)])
    assert n == 0
    assert _calls(conn, 'svc') == []


def test_non_go_documents_are_skipped(tmp_path, conn):
    text = 'fetch("https://api.x/v1/js")\n'
    (tmp_path / 'c.ts').write_text(text)
    index = ScipIndex(
        documents=(_ScipDoc(
            relative_path='c.ts',
            occurrences=(_occ_at(text, 'fetch', _HTTP_GET),), symbols=(),
        ),),
        source_root=tmp_path,
    )
    n = ingest_go_http_clients(
        source_name='svc', source_root=tmp_path, conn=conn,
        index_factory=lambda: index,
    )
    assert n == 0
