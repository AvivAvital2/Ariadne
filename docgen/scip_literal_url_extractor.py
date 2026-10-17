"""Detector A — client HTTP calls inferred from URL literals in situ.

Complements the sink-based extractor (``scip_js_http_client_extractor`` etc.),
which needs the URL to appear as a literal AT a recognized HTTP-client call
(``fetch("...")``). Real SPA clients don't look like that: each API function
holds its endpoint URL as a literal and funnels it through a shared request
wrapper, so the literal never reaches the ``fetch`` site as a literal. But the
literal is still attributable to its owning function via
``string_literals.owning_symbol_id``.

This detector emits an ``http_client_calls`` row for every URL-shaped string
literal whose owning symbol is NOT an endpoint producer (a producer-owned
literal is the *server* declaring its own route, not a client calling one).
``http_method`` is NULL — a bare URL literal doesn't carry the verb; the URL
resolver's NULL-matches-any rule links it to the endpoint by path. The two
detectors are complementary: this one catches literals wherever written; the
sink extractor catches recognized client-call sites with a literal argument.
Both feed ``http_client_calls`` → the URL resolver dedups and matches.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlite3 import Connection


def _looks_like_api_path(value: str) -> bool:
    """A URL path the resolver could match to an endpoint: absolute
    (``/...``), single-segment-or-more, no scheme/host/whitespace, and not a
    protocol-relative ``//`` or a filesystem-looking path. Permissive on
    purpose — the URL resolver only writes ``api_calls`` for literals that
    match a real endpoint template, so junk paths cost at most a harmless
    ``http_client_calls`` row, never a spurious edge."""
    if not value.startswith('/') or value.startswith('//'):
        return False
    if any(ch.isspace() for ch in value):
        return False
    # Must have a non-empty first segment.
    return len(value) > 1


def ingest_literal_url_clients(*, source_name: str, conn: 'Connection') -> int:
    """Emit ``http_client_calls`` from URL-shaped string literals owned by
    non-producer functions. Idempotent: clears this detector's prior rows
    (``sink_name='literal-url'``) for the source before re-inserting.
    Returns the number of rows inserted.
    """
    conn.execute(
        "DELETE FROM http_client_calls "
        "WHERE source_name = ? AND sink_name = 'literal-url'",
        (source_name,),
    )
    rows = conn.execute(
        'SELECT value, file, line_start, owning_symbol_id '
        'FROM string_literals '
        'WHERE source_name = ? AND owning_symbol_id IS NOT NULL '
        '  AND value LIKE ? '
        '  AND owning_symbol_id NOT IN ('
        '      SELECT producer_symbol_id FROM api_endpoints '
        '      WHERE source_name = ? AND producer_symbol_id IS NOT NULL)',
        (source_name, '/%', source_name),
    ).fetchall()

    inserted = 0
    for value, file, line, owner in rows:
        if not _looks_like_api_path(value):
            continue
        conn.execute(
            'INSERT INTO http_client_calls '
            '(source_name, consumer_symbol_id, raw_url, http_method, '
            ' call_site_file, call_site_line, sink_name, confidence) '
            "VALUES (?, ?, ?, NULL, ?, ?, 'literal-url', 'literal')",
            (source_name, owner, value, file, line),
        )
        inserted += 1
    return inserted
