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


__all__ = [
    'persist_config_values',
    'query_config_values_by_key',
    'query_config_values_for_source',
]
