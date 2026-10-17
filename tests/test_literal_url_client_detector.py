"""Detector A — URL literals written directly in client functions.

The SPA idiom: each API-client function holds its endpoint URL as a literal
(``/api/users``), funneled through a shared request wrapper. The URL never
reaches a recognized ``fetch`` sink as a literal, so the sink extractor misses
it — but the literal IS attributable to its owning function via
``string_literals.owning_symbol_id``. This detector emits an
``http_client_calls`` row per such literal (method NULL — the literal doesn't
carry it), which the existing URL resolver then matches to an endpoint.

Server-side route-declaration literals (owned by an endpoint *producer*) are
excluded so the server doesn't "call" its own routes.
"""
from __future__ import annotations

import sqlite3

from docgen.scip_literal_url_extractor import ingest_literal_url_clients
from library.scip import init_scip_schema


def _conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    return c


def _add_literal(c, *, value, owner, file='src/api/client.js', line=10):
    c.execute(
        'INSERT INTO string_literals (source_name, file, line_start, '
        'col_start, value, owning_symbol_id, kind) '
        "VALUES ('app', ?, ?, 4, ?, ?, 'string')",
        (file, line, value, owner),
    )


def test_client_url_literal_becomes_http_client_call() -> None:
    c = _conn()
    _add_literal(c, value='/api/users', owner='ts/client.js/listUsers().')
    n = ingest_literal_url_clients(source_name='app', conn=c)
    assert n == 1
    row = c.execute(
        'SELECT consumer_symbol_id, raw_url, http_method, sink_name '
        'FROM http_client_calls'
    ).fetchone()
    assert row == ('ts/client.js/listUsers().', '/api/users', None, 'literal-url')


def test_server_route_declaration_literal_is_excluded() -> None:
    """A URL literal owned by an endpoint producer is the server DECLARING a
    route, not a client calling one — it must not become a client call."""
    c = _conn()
    producer = 'py/routers/set_user_status().'
    c.execute(
        'INSERT INTO api_endpoints (source_name, http_method, path_template, '
        "producer_symbol_id, resolution_source) "
        "VALUES ('app', 'PATCH', '/api/users/{id}/status', ?, 'pattern')",
        (producer,),
    )
    _add_literal(c, value='/api/users/{id}/status', owner=producer,
                 file='app/routers/users.py')
    n = ingest_literal_url_clients(source_name='app', conn=c)
    assert n == 0
    assert c.execute('SELECT COUNT(*) FROM http_client_calls').fetchone()[0] == 0


def test_non_path_literal_ignored() -> None:
    c = _conn()
    _add_literal(c, value='Content-Type', owner='ts/client.js/request().')
    assert ingest_literal_url_clients(source_name='app', conn=c) == 0
