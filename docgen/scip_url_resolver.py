"""Phase 8c — URL → endpoint resolver.

Bridges Phase 8b's ``http_client_calls`` (raw client URLs at known
call sites) to Phase 7a's ``api_endpoints.path_template`` (server-side
declared endpoints), writing matched pairs to ``api_calls`` so that
``ariadne_trace_flow`` (Phase 9) can join client → server hops.

Algorithm:

1. Read every ``http_client_calls`` row for ``source_name`` whose
   ``consumer_symbol_id`` is non-NULL (``api_calls.consumer`` is NOT
   NULL by Phase 7a's schema; module-level calls are out-of-scope).
2. For each row:
   a. Normalize ``raw_url`` to a path (strip scheme/host/query/
      fragment).
   b. Find ``api_endpoints`` rows whose ``path_template`` matches the
      path (segment count + literal-or-wildcard equality) and whose
      ``http_method`` matches the client's. Client method ``NULL``
      (play-ws fluent builders) matches ANY endpoint method — Phase
      2s with chained-call resolution will tighten this later.
   c. Insert a row into ``api_calls`` with
      ``resolution_source='http-client'`` and confidence:

      - ``'exact-literal'`` for a single match
      - ``'ambiguous'`` when multiple endpoints match (first-wins by
        ``endpoint_id`` lex order; the ambiguity is recorded so
        downstream consumers can disambiguate)

3. Re-resolve clears prior ``http-client`` rows in ``api_calls``
   whose call site is in this source's current ``http_client_calls``.
   Other sources' resolutions are preserved.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlite3 import Connection


def _extract_path(raw_url: str) -> str | None:
    """Extract just the path component of a URL string.

    Returns the path (always starting with ``/``) for absolute URLs
    (``http://...``, ``https://...``) and for already-bare paths
    (``/users``). Returns ``None`` for input that doesn't parse as
    a URL or path — empty string, missing leading slash on a relative,
    etc.
    """
    if not raw_url:
        return None
    s = raw_url
    if s.startswith(('http://', 'https://')):
        scheme_end = s.find('://') + 3
        idx = s.find('/', scheme_end)
        if idx == -1:
            return '/'
        s = s[idx:]
    elif not s.startswith('/'):
        return None
    s = s.split('?', 1)[0]
    s = s.split('#', 1)[0]
    if not s.startswith('/'):
        return None
    return s


def _matches_template(url_path: str, template: str) -> bool:
    """Return True if ``url_path`` matches ``template`` segment-by-
    segment.

    Templates use OpenAPI-style ``{name}`` for wildcards; a wildcard
    matches exactly one path segment (no ``/``). Phase 8a normalizes
    framework-specific syntaxes (Flask ``<id>``, Express ``:id``,
    FastAPI ``{id:int}``) to the OpenAPI form before persistence, so
    the resolver doesn't need framework-specific knowledge.
    """
    url_parts = url_path.split('/')
    tmpl_parts = template.split('/')
    if len(url_parts) != len(tmpl_parts):
        return False
    for u, t in zip(url_parts, tmpl_parts):
        if t.startswith('{') and t.endswith('}') and len(t) >= 3:
            # Wildcard segment — matches one segment, never spans /
            if '/' in u:
                return False
            # Otherwise any segment value matches
        else:
            if u != t:
                return False
    return True


def resolve_urls_to_endpoints(
    *,
    conn: 'Connection',
    source_name: str,
) -> int:
    """Resolve ``http_client_calls`` for ``source_name`` against
    ``api_endpoints``, writing matched edges into ``api_calls``.

    Returns the number of edges inserted.
    """
    # Clear prior 'http-client' rows whose call site is in this
    # source's current http_client_calls. Other sources untouched.
    conn.execute(
        '''DELETE FROM api_calls
           WHERE resolution_source = 'http-client'
             AND (call_site_file, call_site_line) IN (
                 SELECT call_site_file, call_site_line
                 FROM http_client_calls
                 WHERE source_name = ?
             )''',
        (source_name,),
    )

    cursor = conn.execute(
        '''SELECT consumer_symbol_id, raw_url, http_method,
                  call_site_file, call_site_line
           FROM http_client_calls
           WHERE source_name = ?
             AND consumer_symbol_id IS NOT NULL''',
        (source_name,),
    )
    client_calls = cursor.fetchall()
    if not client_calls:
        conn.commit()
        return 0

    # Read all endpoints (cross-source: a client may call any
    # source's endpoints).
    endpoints = conn.execute(
        '''SELECT endpoint_id, http_method, path_template
           FROM api_endpoints'''
    ).fetchall()

    rows: list[tuple] = []
    seen: set[tuple] = set()
    for consumer, raw_url, method, call_file, call_line in client_calls:
        path = _extract_path(raw_url)
        if path is None:
            continue
        matches: list[tuple[str, str]] = []
        for ep_id, ep_method, ep_template in endpoints:
            if method is not None and ep_method != method:
                continue
            if _matches_template(path, ep_template):
                matches.append((ep_id, ep_method))
        if not matches:
            continue
        # First-wins by endpoint_id lex order — deterministic.
        matches.sort()
        ep_id, _ = matches[0]
        confidence = (
            'ambiguous' if len(matches) > 1 else 'exact-literal'
        )
        key = (consumer, ep_id, call_file, call_line)
        if key in seen:
            continue
        seen.add(key)
        rows.append((
            consumer,
            ep_id,
            call_file,
            call_line,
            'http-client',
            confidence,
        ))

    if rows:
        conn.executemany(
            '''INSERT OR REPLACE INTO api_calls
               (consumer_symbol_id, endpoint_id,
                call_site_file, call_site_line,
                resolution_source, confidence)
               VALUES (?, ?, ?, ?, ?, ?)''',
            rows,
        )
    conn.commit()
    return len(rows)


__all__ = [
    'resolve_urls_to_endpoints',
    '_extract_path',
    '_matches_template',
]
