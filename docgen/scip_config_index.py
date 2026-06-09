"""Config-value index persistence (Phase 2q / Layer C).

Bridges Phase 2o's scanner output (in-memory ``ConfigValue`` instances)
to ariadne.db's ``config_values`` table, queryable by Layer C's
resolution traversal (Phase 2s) when walking SCIP refs from a sink
call site backward through config-key lookups.

Re-ingest semantics: :func:`persist_config_values` clears existing
rows for ``source_name`` before inserting the new set. Mirrors the
swagger_ingest pattern (Phase 7c) — the manifest + scan is the source
of truth; the database is a cache.

Read API:

- :func:`query_config_values_by_key` — exact key match within a
  source; returns a list since the same key can appear in multiple
  files (layered HOCON includes, per-environment overrides).
- :func:`query_config_values_for_source` — full dump for a source.
"""
from __future__ import annotations

from pathlib import Path
from attrs import frozen
from typing import TYPE_CHECKING

from docgen.scip_config_scanners import ConfigValue

if TYPE_CHECKING:
    from sqlite3 import Connection


def persist_config_values(
    *,
    source_name: str,
    config_values: list[ConfigValue],
    conn: 'Connection',
) -> int:
    """Persist ``config_values`` to ariadne.db, replacing the source's
    prior rows.

    Empty input is a valid signal — it represents "this source has no
    config values" and clears any pre-existing rows.

    Returns the number of rows actually inserted.
    """
    # Re-ingest: clear the source's rows first. This makes the function
    # idempotent over the manifest — re-running with the same input
    # produces the same DB state.
    conn.execute(
        'DELETE FROM config_values WHERE source_name = ?',
        (source_name,),
    )

    if not config_values:
        conn.commit()
        return 0

    rows = [
        (
            source_name,
            str(cv.file),
            cv.key,
            cv.value,
            cv.line_start,
        )
        for cv in config_values
    ]
    conn.executemany(
        'INSERT INTO config_values '
        '(source_name, file, key, value, line_start) '
        'VALUES (?, ?, ?, ?, ?)',
        rows,
    )
    conn.commit()
    return len(rows)


def query_config_values_by_key(
    *,
    source_name: str,
    key: str,
    conn: 'Connection',
) -> list[ConfigValue]:
    """Return all ConfigValues for ``(source_name, key)``. Multiple
    matches happen when the same key is set in multiple config files
    (e.g., layered HOCON includes, per-env overrides). Caller decides
    which to use based on context.

    Empty list if no match — no exceptions for missing rows.
    """
    cur = conn.execute(
        'SELECT file, key, value, line_start '
        'FROM config_values '
        'WHERE source_name = ? AND key = ?',
        (source_name, key),
    )
    return [
        ConfigValue(
            file=Path(row[0]),
            key=row[1],
            value=row[2],
            line_start=row[3],
        )
        for row in cur.fetchall()
    ]


def query_config_values_for_source(
    *,
    source_name: str,
    conn: 'Connection',
) -> list[ConfigValue]:
    """Return every ConfigValue persisted for ``source_name``. Used by
    Layer C when it needs to scan all known config values (e.g.,
    when the resolution traversal can't pin a specific key)."""
    cur = conn.execute(
        'SELECT file, key, value, line_start '
        'FROM config_values '
        'WHERE source_name = ?',
        (source_name,),
    )
    return [
        ConfigValue(
            file=Path(row[0]),
            key=row[1],
            value=row[2],
            line_start=row[3],
        )
        for row in cur.fetchall()
    ]
@frozen
class ConfigRead:
    """One code site that reads a config key (a Typesafe Config getter
    call) — the read-side analog of :class:`ConfigValue`. ``value`` is the
    resolved default from ``config_values`` (``None`` when the key is read
    but never declared); ``confidence`` is ``'config-resolved'`` for a
    verified getter call or ``'string-match'`` for the
    unsupported-language fallback. Produced by
    ``scip_config_usage_extractor.extract_config_reads`` and persisted to
    the ``config_reads`` table by :func:`persist_config_reads`."""
    file: Path
    line: int        # 1-indexed
    col: int         # 0-indexed
    key: str
    value: str | None
    confidence: str
def persist_config_reads(
    *,
    source_name: str,
    config_reads: 'list[ConfigRead]',
    conn: 'Connection',
) -> int:
    """Persist config-read sites to ariadne.db, replacing the source's
    prior rows.

    Empty input is a valid signal — it clears any pre-existing rows for
    the source (an extract that found no reads). Mirrors
    :func:`persist_config_values`. Returns the number of rows inserted.
    """
    conn.execute(
        'DELETE FROM config_reads WHERE source_name = ?',
        (source_name,),
    )

    if not config_reads:
        conn.commit()
        return 0

    rows = [
        (
            source_name,
            r.key,
            str(r.file),
            r.line,
            r.col,
            r.value,
            r.confidence,
        )
        for r in config_reads
    ]
    conn.executemany(
        'INSERT INTO config_reads '
        '(source_name, key, file, line_start, col_start, value, confidence) '
        'VALUES (?, ?, ?, ?, ?, ?, ?)',
        rows,
    )
    conn.commit()
    return len(rows)


def query_config_reads_by_key(
    *,
    source_name: str,
    key: str,
    conn: 'Connection',
) -> list[ConfigRead]:
    """Return all config-read sites for ``(source_name, key)``. Multiple
    matches are the norm — one key is typically read from several call
    sites. Empty list if no match."""
    cur = conn.execute(
        'SELECT file, line_start, col_start, key, value, confidence '
        'FROM config_reads '
        'WHERE source_name = ? AND key = ?',
        (source_name, key),
    )
    return [
        ConfigRead(
            file=Path(row[0]),
            line=row[1],
            col=row[2],
            key=row[3],
            value=row[4],
            confidence=row[5],
        )
        for row in cur.fetchall()
    ]


__all__ = [
    'persist_config_reads', 'persist_config_values', 'query_config_reads_by_key', 'query_config_values_by_key', 'query_config_values_for_source'
]
