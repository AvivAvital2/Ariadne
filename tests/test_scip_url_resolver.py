"""Contract for Phase 8c URL→endpoint resolver.

Bridges Phase 8b's ``http_client_calls`` table (raw client URLs) to
Phase 7a's ``api_endpoints.path_template`` table (server-side declared
endpoints) by writing matched pairs into ``api_calls``.

Algorithm:

1. Read every ``http_client_calls`` row for ``source_name``.
2. For each row with a non-NULL ``consumer_symbol_id``:
   a. Normalize ``raw_url`` to a path (strip scheme/host/query/fragment).
   b. Find ``api_endpoints`` rows whose ``path_template`` matches the
      path (segment count + wildcard ``{name}`` substitution + literal
      segment equality) and whose ``http_method`` matches the client's
      (or where the client's method is NULL — play-ws fluent builders
      defer method to a chained call).
   c. Insert the matched ``(consumer, endpoint, file, line)`` tuple
      into ``api_calls`` with ``resolution_source='http-client'`` and
      a confidence tag (``'exact-literal'`` for one match,
      ``'ambiguous'`` for multiple).
3. Re-resolve clears prior ``http-client`` rows from
   ``api_calls`` for the source.

Skipped:

- Client rows with NULL ``consumer_symbol_id`` (top-level calls).
  ``api_calls.consumer_symbol_id`` is NOT NULL by Phase 7a's schema;
  we'd need a schema change to record consumerless edges.
- Client rows whose URL doesn't parse to a path (malformed input).
- Client rows where no endpoint's path_template matches.

These tests are RED until ``docgen/scip_url_resolver.py`` exists.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _add_endpoint(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    method: str,
    path_template: str,
    endpoint_id: str | None = None,
    producer_symbol_id: str | None = None,
    resolution_source: str = 'pattern',
) -> str:
    if endpoint_id is None:
        endpoint_id = f'ep:{source_name}:{method}:{path_template}'
    conn.execute(
        '''INSERT INTO api_endpoints
           (endpoint_id, source_name, http_method, path_template,
            producer_symbol_id, resolution_source)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (endpoint_id, source_name, method, path_template,
         producer_symbol_id, resolution_source),
    )
    conn.commit()
    return endpoint_id


def _add_client_call(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    consumer_symbol_id: str | None,
    raw_url: str,
    http_method: str | None,
    call_site_file: str = '/x.py',
    call_site_line: int = 1,
    sink_name: str = 'requests.get',
    confidence: str = 'literal',
) -> None:
    conn.execute(
        '''INSERT INTO http_client_calls
           (source_name, consumer_symbol_id, raw_url, http_method,
            call_site_file, call_site_line, sink_name, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (source_name, consumer_symbol_id, raw_url, http_method,
         call_site_file, call_site_line, sink_name, confidence),
    )
    conn.commit()


def _query_api_calls(
    conn: sqlite3.Connection, source_name: str | None = None,
) -> list[dict]:
    """Return list of dicts. ``source_name=None`` returns all rows."""
    if source_name is None:
        cur = conn.execute(
            '''SELECT consumer_symbol_id, endpoint_id,
                      call_site_file, call_site_line,
                      resolution_source, confidence
               FROM api_calls
               ORDER BY consumer_symbol_id, endpoint_id''',
        )
    else:
        # Filter by joining through http_client_calls when needed —
        # api_calls itself doesn't carry source_name (PK is the call
        # site triple). For tests, we set up isolation per fixture.
        cur = conn.execute(
            '''SELECT consumer_symbol_id, endpoint_id,
                      call_site_file, call_site_line,
                      resolution_source, confidence
               FROM api_calls
               ORDER BY consumer_symbol_id, endpoint_id''',
        )
    cols = [
        'consumer_symbol_id', 'endpoint_id', 'call_site_file',
        'call_site_line', 'resolution_source', 'confidence',
    ]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# URL path extraction
# ---------------------------------------------------------------------------


class TestUrlPathExtraction:
    """Direct tests on ``_extract_path``. Covers scheme/host/query/
    fragment stripping. Uses the helper directly to surface the
    contract without going through the full pipeline."""

    def test_http_url_path_extracted(self) -> None:
        from docgen.scip_url_resolver import _extract_path
        assert _extract_path('http://api.com/users') == '/users'

    def test_https_url_path_extracted(self) -> None:
        from docgen.scip_url_resolver import _extract_path
        assert _extract_path('https://api.com/users') == '/users'

    def test_relative_path_kept(self) -> None:
        from docgen.scip_url_resolver import _extract_path
        assert _extract_path('/users') == '/users'

    def test_query_stripped(self) -> None:
        from docgen.scip_url_resolver import _extract_path
        assert _extract_path('/users?limit=10') == '/users'

    def test_fragment_stripped(self) -> None:
        from docgen.scip_url_resolver import _extract_path
        assert _extract_path('/users#section') == '/users'

    def test_query_and_fragment_stripped(self) -> None:
        from docgen.scip_url_resolver import _extract_path
        assert (
            _extract_path('https://api/users?id=1#sec')
            == '/users'
        )

    def test_url_with_port(self) -> None:
        from docgen.scip_url_resolver import _extract_path
        assert (
            _extract_path('http://api.com:8080/users')
            == '/users'
        )

    def test_url_with_no_path(self) -> None:
        """``http://api.com`` (no path) should yield ``/`` (root)."""
        from docgen.scip_url_resolver import _extract_path
        assert _extract_path('http://api.com') == '/'

    def test_unparseable_returns_none(self) -> None:
        """Garbage that doesn't look like a URL or path returns None
        so the caller can skip cleanly."""
        from docgen.scip_url_resolver import _extract_path
        assert _extract_path('') is None
        assert _extract_path('not a url') is None


# ---------------------------------------------------------------------------
# Template matching
# ---------------------------------------------------------------------------


class TestTemplateMatching:
    """Direct tests on ``_matches_template``."""

    def test_exact_match(self) -> None:
        from docgen.scip_url_resolver import _matches_template
        assert _matches_template('/users', '/users') is True

    def test_wildcard_match(self) -> None:
        from docgen.scip_url_resolver import _matches_template
        assert _matches_template('/users/123', '/users/{id}') is True

    def test_multi_wildcard_match(self) -> None:
        from docgen.scip_url_resolver import _matches_template
        assert _matches_template(
            '/users/123/posts/456',
            '/users/{userId}/posts/{postId}',
        ) is True

    def test_segment_count_mismatch(self) -> None:
        from docgen.scip_url_resolver import _matches_template
        assert _matches_template('/users', '/users/{id}') is False
        assert (
            _matches_template('/users/1/extra', '/users/{id}')
            is False
        )

    def test_literal_segment_mismatch(self) -> None:
        from docgen.scip_url_resolver import _matches_template
        assert (
            _matches_template('/items/123', '/users/{id}') is False
        )

    def test_wildcard_does_not_span_slash(self) -> None:
        """A ``{wildcard}`` matches one segment, not multiple
        slash-separated parts."""
        from docgen.scip_url_resolver import _matches_template
        assert (
            _matches_template('/users/1/2', '/users/{id}') is False
        )


# ---------------------------------------------------------------------------
# resolve_urls_to_endpoints — end-to-end pipeline
# ---------------------------------------------------------------------------


class TestResolveUrlsToEndpoints:
    def test_literal_url_matches_exact_endpoint(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        ep_id = _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/users',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='consumer:fetchUsers',
            raw_url='http://api.com/users',
            http_method='GET',
        )
        n = resolve_urls_to_endpoints(
            conn=conn, source_name='client',
        )
        assert n == 1
        rows = _query_api_calls(conn)
        assert len(rows) == 1
        assert rows[0]['consumer_symbol_id'] == 'consumer:fetchUsers'
        assert rows[0]['endpoint_id'] == ep_id
        assert rows[0]['confidence'] == 'exact-literal'
        assert rows[0]['resolution_source'] == 'http-client'

    def test_wildcard_endpoint_matches_concrete_url(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        ep_id = _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/users/{id}',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='consumer:getUser',
            raw_url='https://api.com/users/123',
            http_method='GET',
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        rows = _query_api_calls(conn)
        assert len(rows) == 1
        assert rows[0]['endpoint_id'] == ep_id

    def test_method_mismatch_no_row(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        _add_endpoint(
            conn, source_name='api',
            method='POST', path_template='/users',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='consumer:fetch',
            raw_url='/users',
            http_method='GET',  # mismatch
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        assert _query_api_calls(conn) == []

    def test_null_method_matches_any_endpoint(
        self, conn: sqlite3.Connection,
    ) -> None:
        """play-ws ``WSClient.url()`` records ``http_method=None``
        because the verb is on a chained call. The resolver should
        still match such a row to any endpoint with the right path,
        independent of method. (Phase 2s would tighten this once
        chained-call resolution lands.)"""
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/users',
        )
        _add_endpoint(
            conn, source_name='api',
            method='POST', path_template='/users',
            endpoint_id='ep:other',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='consumer:scalaCall',
            raw_url='http://api.com/users',
            http_method=None,  # play-ws case
            sink_name='play.WSClient.url',
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        rows = _query_api_calls(conn)
        # Two endpoints both match ⇒ ambiguous resolution
        assert len(rows) == 1
        assert rows[0]['confidence'] == 'ambiguous'

    def test_consumer_null_skipped(
        self, conn: sqlite3.Connection,
    ) -> None:
        """``api_calls.consumer_symbol_id`` is NOT NULL by Phase 7a
        schema. Module-level client calls are recorded in
        ``http_client_calls`` but skipped by the resolver — there's
        no caller side to attribute the edge to."""
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/users',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id=None,  # top-level call
            raw_url='/users',
            http_method='GET',
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        assert _query_api_calls(conn) == []

    def test_no_endpoint_match_no_row(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/items',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='consumer:x',
            raw_url='/users',
            http_method='GET',
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        assert _query_api_calls(conn) == []

    def test_multi_match_marks_ambiguous(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        # Two endpoints both match the URL: same method, same path
        # template (one declared per source — overlap is realistic
        # when a client targets multiple backends with equivalent APIs).
        _add_endpoint(
            conn, source_name='api1',
            method='GET', path_template='/users',
            endpoint_id='ep:api1',
        )
        _add_endpoint(
            conn, source_name='api2',
            method='GET', path_template='/users',
            endpoint_id='ep:api2',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='consumer:x',
            raw_url='/users',
            http_method='GET',
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        rows = _query_api_calls(conn)
        # First-wins semantics; one row only, marked ambiguous
        assert len(rows) == 1
        assert rows[0]['confidence'] == 'ambiguous'
        # Picked the lex-first endpoint_id
        assert rows[0]['endpoint_id'] == 'ep:api1'

    def test_re_resolve_replaces_prior_rows(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/old',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='consumer:x',
            raw_url='/old',
            http_method='GET',
            call_site_file='/x.py', call_site_line=1,
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        assert len(_query_api_calls(conn)) == 1

        # Replace the http_client_calls row with a different URL
        conn.execute(
            'DELETE FROM http_client_calls WHERE source_name = ?',
            ('client',),
        )
        _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/new',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='consumer:x',
            raw_url='/new',
            http_method='GET',
            call_site_file='/x.py', call_site_line=1,
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        rows = _query_api_calls(conn)
        # Old row replaced with new; only one entry
        assert len(rows) == 1
        assert rows[0]['endpoint_id'] == 'ep:api:GET:/new'

    def test_other_source_resolutions_preserved(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Resolving for source ``client_a`` doesn't disturb prior
        resolutions for source ``client_b``."""
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/x',
            endpoint_id='ep:x',
        )
        _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/y',
            endpoint_id='ep:y',
        )
        # client_b first
        _add_client_call(
            conn, source_name='client_b',
            consumer_symbol_id='consumer:b',
            raw_url='/y', http_method='GET',
            call_site_file='/b.py', call_site_line=1,
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client_b')

        # client_a second
        _add_client_call(
            conn, source_name='client_a',
            consumer_symbol_id='consumer:a',
            raw_url='/x', http_method='GET',
            call_site_file='/a.py', call_site_line=1,
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client_a')

        rows = _query_api_calls(conn)
        consumers = {r['consumer_symbol_id'] for r in rows}
        # Both consumers' edges remain
        assert 'consumer:a' in consumers
        assert 'consumer:b' in consumers


# ---------------------------------------------------------------------------
# Path-template normalization edge cases
# ---------------------------------------------------------------------------


class TestPathParamNormalization:
    """Phase 8a normalizes framework-specific path syntaxes
    (``<id>``, ``:id``) to OpenAPI ``{id}`` before persisting to
    ``api_endpoints``. The resolver assumes that normalization has
    happened, so it only matches against ``{name}`` form."""

    def test_openapi_brace_param_matches_concrete(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_url_resolver import (
            resolve_urls_to_endpoints,
        )

        ep_id = _add_endpoint(
            conn, source_name='api',
            method='GET', path_template='/users/{id}',
        )
        _add_client_call(
            conn, source_name='client',
            consumer_symbol_id='c:1',
            raw_url='http://api.com/users/42',
            http_method='GET',
        )
        resolve_urls_to_endpoints(conn=conn, source_name='client')
        rows = _query_api_calls(conn)
        assert len(rows) == 1
        assert rows[0]['endpoint_id'] == ep_id
