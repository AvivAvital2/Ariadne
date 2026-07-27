"""Contract for the SCIP-driven Go HTTP route extractor.

Architecture mirrors the Express/Koa (8a.3), Flask/FastAPI (8a.2), and
Akka (8a.1) extractors:

- **SCIP** classifies which call sites are real router methods, via the
  trailing descriptor of the occurrence symbol (suffix match — the
  ``scip-go gomod <module> <version>`` preamble is variable and ignored).
- **ast-grep / tree-sitter-go** parses the call STRUCTURE: the
  ``call_expression`` → ``selector_expression`` (``field_identifier`` =
  method name, where scip-go anchors) and the ``argument_list`` whose
  first element holds the route path string literal.
- The path literal is read DIRECTLY from the AST node
  (``interpreted_string_literal`` / ``raw_string_literal``) — Go isn't in
  the ``string_literals`` table and route paths are literals in practice.

Frameworks covered: gin (``RouterGroup``/``Engine`` verb methods), echo
(``Echo``/``Group``), chi (``Mux``/``Router`` Title-case verbs), and
net/http (``http.HandleFunc`` / ``ServeMux`` → method ``ANY``).

NOTE: the symbol suffixes encode scip-go / SCIP descriptor conventions +
each framework's exported type names; validate against a real scip-go
index when one is available.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_go_route_extractor import ingest_go_routes
from docgen.scip_string_literal_extractor import ingest_string_literals

# Synthetic scip-go symbols — only the trailing descriptor matters.
_PRE = 'scip-go gomod example.com/m v1 '
_GIN_GET = _PRE + 'github.com/gin-gonic/gin/RouterGroup#GET().'
_ECHO_POST = _PRE + 'github.com/labstack/echo/v4/Echo#POST().'
_CHI_GET = _PRE + 'github.com/go-chi/chi/v5/Mux#Get().'
_NETHTTP_HANDLEFUNC = _PRE + 'net/http/HandleFunc().'


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _occ_at(text: str, marker: str, symbol: str) -> _ScipOccurrence:
    """SCIP occurrence at the first whole-word ``marker`` in ``text``
    (0-indexed line/col) — the position scip-go anchors the method call,
    and the position the Go extractor derives from the field_identifier."""
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
                    symbol=symbol,
                    range=(lineno, i, lineno, j),
                    is_definition=False,
                )
            col = i + 1
    raise ValueError(f'{marker!r} not found as a word in text')


def _make_index(
    rel: str, occs: list[_ScipOccurrence], source_root: Path,
) -> ScipIndex:
    return ScipIndex(
        documents=(_ScipDoc(
            relative_path=rel, occurrences=tuple(occs), symbols=(),
        ),),
        source_root=source_root,
    )


def _endpoints(conn: sqlite3.Connection, source_name: str) -> set[tuple[str, str]]:
    cur = conn.execute(
        'SELECT http_method, path_template FROM api_endpoints '
        'WHERE source_name = ?',
        (source_name,),
    )
    return {(r[0], r[1]) for r in cur.fetchall()}


def _run(tmp_path, conn, filename, text, occs, source_name='svc'):
    """Pipeline: index string literals first (the seam the route extractor
    reads the path from), then extract routes — same order the persist
    chain runs (persist_string_literals precedes the route extractors)."""
    (tmp_path / filename).write_text(text)
    index = _make_index(filename, occs, tmp_path)
    ingest_string_literals(
        source_name=source_name, source_root=tmp_path,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_go_routes(
        source_name=source_name, source_root=tmp_path,
        conn=conn, index_factory=lambda: index,
    )


def test_gin_verb_route_with_param_normalized(tmp_path, conn):
    text = 'package main\nfunc f() { r.GET("/users/:id", getUser) }\n'
    _run(tmp_path, conn, 'routes.go', text, [_occ_at(text, 'GET', _GIN_GET)])
    assert ('GET', '/users/{id}') in _endpoints(conn, 'svc')


def test_echo_verb_route(tmp_path, conn):
    text = 'package main\nfunc f() { e.POST("/login", h) }\n'
    _run(tmp_path, conn, 'routes.go', text, [_occ_at(text, 'POST', _ECHO_POST)])
    assert ('POST', '/login') in _endpoints(conn, 'svc')


def test_chi_titlecase_verb_route(tmp_path, conn):
    text = 'package main\nfunc f() { r.Get("/health", h) }\n'
    _run(tmp_path, conn, 'routes.go', text, [_occ_at(text, 'Get', _CHI_GET)])
    assert ('GET', '/health') in _endpoints(conn, 'svc')


def test_net_http_handlefunc_is_any_method(tmp_path, conn):
    text = 'package main\nfunc f() { http.HandleFunc("/metrics", h) }\n'
    _run(
        tmp_path, conn, 'main.go', text,
        [_occ_at(text, 'HandleFunc', _NETHTTP_HANDLEFUNC)],
    )
    assert ('ANY', '/metrics') in _endpoints(conn, 'svc')


def test_raw_string_path_literal(tmp_path, conn):
    text = 'package main\nfunc f() { r.GET(`/raw/:x`, h) }\n'
    _run(tmp_path, conn, 'routes.go', text, [_occ_at(text, 'GET', _GIN_GET)])
    assert ('GET', '/raw/{x}') in _endpoints(conn, 'svc')


def test_unclassified_method_is_ignored(tmp_path, conn):
    """A same-named method that isn't a known router type is not a route."""
    text = 'package main\nfunc f() { cache.GET("/nope", h) }\n'
    other = _PRE + 'example.com/m/cache/Store#GET().'
    _run(tmp_path, conn, 'store.go', text, [_occ_at(text, 'GET', other)])
    assert _endpoints(conn, 'svc') == set()


def test_non_go_documents_are_skipped(tmp_path, conn):
    text = 'app.get("/js", h)\n'
    (tmp_path / 'app.js').write_text(text)
    index = _make_index('app.js', [_occ_at(text, 'get', _GIN_GET)], tmp_path)
    ingest_go_routes(
        source_name='svc', source_root=tmp_path, conn=conn,
        index_factory=lambda: index,
    )
    assert _endpoints(conn, 'svc') == set()
