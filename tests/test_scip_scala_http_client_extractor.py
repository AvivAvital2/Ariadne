"""Contract for SCIP-driven Scala HTTP client extractor (Phase 8b.3).

Architecture mirrors Phase 8b.1 / 8b.2 and reuses the Phase 8a.1
(Akka) tree-sitter-scala helpers:

- **SCIP** filters which call sites are real JVM HTTP-client
  primitives via the Phase 2r ``DEFAULT_SINK_REGISTRY`` filtered to
  ``language='jvm'``. v1 covers ``play.WSClient.url()``; sttp / Akka
  HTTP client / OkHttp builder need Phase 2s chained-call walking
  and are deferred.
- **ast-grep / tree-sitter-scala** walks the call STRUCTURE: the
  ``call_expression`` and its arguments. Same grammar Phase 8a.1
  uses for Akka.
- **Phase 2p ``string_literals``** supplies the URL value at the
  first arg's position. ``s"..."`` / ``f"..."`` interpolated forms
  aren't indexed by Phase 2p, so the lookup naturally returns
  ``None`` for those.
- **Phase 2d ``scip_symbols``** supplies ``consumer_symbol_id`` —
  the enclosing ``def`` (Method).

Output: rows in ``http_client_calls`` with ``http_method=None`` for
play-ws (the verb comes from a chained call we don't track in v1).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
)
from docgen.scip_string_literal_extractor import ingest_string_literals


_PLAY_WS_URL_SYM = (
    'scip-java semanticdb maven com.typesafe.play '
    'play-ahc-ws_2.13 2.8.0 '
    'play/api/libs/ws/WSClient#url().'
)


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _occ_at(
    text: str, marker: str, symbol: str, *, nth: int = 0,
) -> _ScipOccurrence:
    found = 0
    pos = -1
    n = len(text)
    i = 0
    while i <= n - len(marker):
        if text.startswith(marker, i):
            before_ok = (
                i == 0
                or not (text[i - 1].isalnum() or text[i - 1] == '_')
            )
            j = i + len(marker)
            after_ok = (
                j >= n
                or not (text[j].isalnum() or text[j] == '_')
            )
            if before_ok and after_ok:
                if found == nth:
                    pos = i
                    break
                found += 1
                i = j
                continue
        i += 1
    if pos < 0:
        raise ValueError(
            f'marker {marker!r} (nth={nth}) not found as a word',
        )
    line = text.count('\n', 0, pos)
    line_start = text.rfind('\n', 0, pos) + 1
    col = pos - line_start
    return _ScipOccurrence(
        symbol=symbol,
        range=(line, col, line, col + len(marker)),
        is_definition=False,
    )


def _make_index(
    src_file: Path,
    source_root: Path,
    occurrences: list[_ScipOccurrence],
) -> ScipIndex:
    rel = src_file.relative_to(source_root)
    return ScipIndex(
        documents=(_ScipDoc(
            relative_path=str(rel),
            occurrences=tuple(occurrences),
            symbols=(),
        ),),
        source_root=source_root,
    )


def _add_scip_symbol(
    conn: sqlite3.Connection,
    *,
    canonical_id: str,
    source_name: str,
    file: str,
    line_start: int,
    line_end: int,
    kind: str = 'Method',
    qualified_name: str = '',
    parent_qualified_name: str | None = None,
    language: str = 'scala',
) -> None:
    conn.execute(
        '''INSERT INTO scip_symbols
           (canonical_id, source_name, language, file,
            line_start, line_end, kind, display_name,
            qualified_name, parent_qualified_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (canonical_id, source_name, language, file,
         line_start, line_end, kind,
         qualified_name.rsplit('.', 1)[-1] or canonical_id,
         qualified_name, parent_qualified_name),
    )
    conn.commit()


def _query_client_calls(
    conn: sqlite3.Connection, source_name: str,
) -> list[dict]:
    cur = conn.execute(
        '''SELECT consumer_symbol_id, raw_url, http_method,
                  call_site_file, call_site_line, sink_name, confidence
           FROM http_client_calls WHERE source_name = ?
           ORDER BY call_site_file, call_site_line''',
        (source_name,),
    )
    cols = [
        'consumer_symbol_id', 'raw_url', 'http_method',
        'call_site_file', 'call_site_line', 'sink_name',
        'confidence',
    ]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _run_pipeline(
    *, source_name: str, source_root: Path,
    conn: sqlite3.Connection, index: ScipIndex,
) -> int:
    from docgen.scip_scala_http_client_extractor import (
        ingest_scala_http_clients,
    )

    ingest_string_literals(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_scala_http_clients(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )


# ---------------------------------------------------------------------------
# play-ws WSClient.url
# ---------------------------------------------------------------------------


class TestPlayWsUrl:
    def test_url_with_literal(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = 'val r = wsClient.url("http://api/users")\n'
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        # http_method=None until Phase 2s walks the chained call
        assert rows[0]['http_method'] is None
        assert rows[0]['sink_name'] == 'play.WSClient.url'

    def test_url_chained_with_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``wsClient.url("/api").get()`` — the SCIP occurrence on
        ``url`` matches the inner call. The chained ``.get()`` is a
        separate call_expression with its own callee position; if
        nothing in the registry matches its symbol, no extra row.
        v1 doesn't record the GET method because chained-call
        resolution is Phase 2s territory."""
        text = (
            'val r = wsClient.url("http://api/users").get()\n'
        )
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        assert rows[0]['sink_name'] == 'play.WSClient.url'

    def test_url_with_variable_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'val URL = "http://api/x"\n'
            'val r = wsClient.url(URL)\n'
        )
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        # Variable URL — not indexed at the call site by Phase 2p.
        # Phase 2s would resolve via the val's definition.
        assert rows == []

    def test_url_with_interpolated_string_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'val prefix = "api"\n'
            'val r = wsClient.url(s"http://$prefix/users")\n'
        )
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        # ``s"..."`` interpolated strings aren't indexed by Phase 2p
        for r in rows:
            assert '$' not in r['raw_url']
            # The plain "api" literal might be in string_literals from
            # the `val prefix = "api"` line, but it isn't at the call's
            # arg position, so the row's URL won't be 'api'.
            assert r['raw_url'] != 'api'


# ---------------------------------------------------------------------------
# Consumer symbol resolution
# ---------------------------------------------------------------------------


class TestConsumerSymbolResolution:
    def test_call_inside_def_resolves_consumer(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'class UserService {\n'
            '  def fetch(): Future[String] = {\n'
            '    wsClient.url("http://api/users").get().map(_.body)\n'
            '  }\n'
            '}\n'
        )
        src = tmp_path / 'UserService.scala'
        src.write_text(text)
        method_id = 'scip:UserService#fetch().'
        # Class symbol (kind=Class — must be filtered by kind filter)
        _add_scip_symbol(
            conn, canonical_id='scip:UserService#',
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=5,
            kind='Class', qualified_name='UserService',
        )
        # Method symbol (kind=Method — qualifies)
        _add_scip_symbol(
            conn, canonical_id=method_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=2, line_end=4,
            kind='Method', qualified_name='UserService.fetch',
            parent_qualified_name='UserService',
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['consumer_symbol_id'] == method_id

    def test_call_at_top_level_consumer_is_null(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = 'val r = wsClient.url("http://api/x")\n'
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['consumer_symbol_id'] is None


# ---------------------------------------------------------------------------
# Phase 2s wiring — variable URLs resolve via scip_symbols + string_literals
# ---------------------------------------------------------------------------


class TestPhase2sVariableResolution:
    """Phase 2s shipped as a library; this test asserts Phase 8b.3
    consumes it for Scala. A class field whose initializer is a
    string literal resolves through ``resolve_arg_value``."""

    def test_field_resolves_via_scip_symbols(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'class Service {\n'
            '  val apiUrl = "http://api/users"\n'
            '  def fetch(): Unit = {\n'
            '    wsClient.url(apiUrl).get()\n'
            '  }\n'
            '}\n'
        )
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        method_id = 'scip:Service#fetch().'
        _add_scip_symbol(
            conn, canonical_id='scip:Service#',
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=6, kind='Class',
            qualified_name='Service', language='scala',
        )
        _add_scip_symbol(
            conn, canonical_id=method_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=3, line_end=5, kind='Method',
            qualified_name='Service.fetch',
            parent_qualified_name='Service', language='scala',
        )
        # The val whose initializer carries the literal — Phase 2s
        # finds it via qualified_name match
        _add_scip_symbol(
            conn, canonical_id='scip:Service#apiUrl.',
            source_name='myapi', file=str(src.resolve()),
            line_start=2, line_end=2, kind='Field',
            qualified_name='Service.apiUrl',
            parent_qualified_name='Service', language='scala',
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        assert rows[0]['confidence'] == 'resolved-constant'
        assert rows[0]['consumer_symbol_id'] == method_id

    def test_unknown_identifier_still_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired negative — undefined identifier is skipped; the
        literal sibling still emits."""
        text = (
            'class Service {\n'
            '  def fetch(): Unit = {\n'
            '    wsClient.url("http://api/literal").get()\n'
            '    wsClient.url(UNDEFINED).get()\n'
            '  }\n'
            '}\n'
        )
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        method_id = 'scip:Service#fetch().'
        _add_scip_symbol(
            conn, canonical_id=method_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=2, line_end=5, kind='Method',
            qualified_name='Service.fetch', language='scala',
        )
        # No scip_symbol for UNDEFINED
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM, nth=0),
            _occ_at(text, 'url', _PLAY_WS_URL_SYM, nth=1),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        urls = [r['raw_url'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the literal call; got {urls}'
        )
        assert rows[0]['raw_url'] == 'http://api/literal'


# ---------------------------------------------------------------------------
# Adversarial — SCIP filtering, malformed input
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_unrelated_symbol_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``url`` as a method on some non-WSClient type — symbol
        doesn't match the play-ws registry suffix. No row."""
        text = 'val r = otherClient.url("http://api/x")\n'
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'url',
                'scip-java semanticdb maven other libxx 1.0 '
                'other/lib/Client#url().',
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert rows == []

    def test_no_scip_occurrence_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = 'val r = wsClient.url("http://api/x")\n'
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert rows == []

    def test_missing_index_file_no_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_scala_http_client_extractor import (
            ingest_scala_http_clients,
        )
        rc = ingest_scala_http_clients(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# Re-ingest semantics
# ---------------------------------------------------------------------------


class TestReIngest:
    def test_replaces_same_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text1 = 'val r = wsClient.url("http://api/old")\n'
        src = tmp_path / 'Service.scala'
        src.write_text(text1)
        index1 = _make_index(src, tmp_path, [
            _occ_at(text1, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index1,
        )
        text2 = 'val r = wsClient.url("http://api/new")\n'
        src.write_text(text2)
        index2 = _make_index(src, tmp_path, [
            _occ_at(text2, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index2,
        )
        rows = _query_client_calls(conn, 'myapi')
        urls = [r['raw_url'] for r in rows]
        assert 'http://api/old' not in urls
        assert 'http://api/new' in urls

    def test_preserves_other_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        conn.execute(
            '''INSERT INTO http_client_calls
               (source_name, raw_url, http_method,
                call_site_file, call_site_line, sink_name, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            ('other', 'http://other/x', 'GET',
             '/x.scala', 1, 'fetch', 'literal'),
        )
        conn.commit()
        text = 'val r = wsClient.url("http://api/mine")\n'
        src = tmp_path / 'Service.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'url', _PLAY_WS_URL_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        my = [r['raw_url'] for r in _query_client_calls(conn, 'myapi')]
        other = [
            r['raw_url'] for r in _query_client_calls(conn, 'other')
        ]
        assert 'http://api/mine' in my
        assert other == ['http://other/x']
