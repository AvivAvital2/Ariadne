"""Contract for the config-value index (Phase 2q / Layer C).

Persists Phase 2o's :class:`ConfigValue` output to ``ariadne.db`` as
the ``config_values`` table, queryable by Layer C's resolution
traversal (Phase 2s) when walking SCIP refs from a sink call site
backward through config-key lookups.

Schema columns (asserted by ``TestSchema``):
``source_name``, ``file``, ``key``, ``value``, ``line_start``.

Re-ingest semantics: ``persist_config_values`` clears existing rows
for ``source_name`` before inserting the new set. Mirrors the
swagger_ingest pattern from Phase 7c — same source's manifest +
re-scan = idempotent.

Query API:

- ``query_config_values_by_key(source_name, key, conn)`` — exact key
  match within a source; returns a list since the same key can
  appear in multiple files (e.g., layered HOCON includes).
- ``query_config_values_for_source(source_name, conn)`` — full dump
  for a source.

These tests are RED until ``docgen/scip_config_index.py`` exists and
``library_scip.init_scip_schema`` creates the ``config_values`` table.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def conn():
    """Fresh in-memory SQLite with the SCIP schema applied."""
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_init_scip_schema_creates_config_values_table(
        self, conn: sqlite3.Connection,
    ) -> None:
        """``init_scip_schema`` (called by Library.__attrs_post_init__)
        also creates the config_values table."""
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='config_values'"
        )
        assert cur.fetchone() is not None

    def test_required_columns_present(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Schema columns: source_name, file, key, value, line_start.
        All required for Layer C lookups."""
        cur = conn.execute('PRAGMA table_info(config_values)')
        cols = {row[1] for row in cur.fetchall()}
        for col in ('source_name', 'file', 'key', 'value', 'line_start'):
            assert col in cols, f'config_values missing column: {col}'

    def test_indexed_for_source_and_key_lookup(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Layer C will query by ``(source_name, key)`` repeatedly
        during traversal — index must exist for performance."""
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='config_values'"
        )
        index_names = {row[0] for row in cur.fetchall()}
        # At least one index covering source_name + key
        assert any(
            'source' in name.lower() or 'key' in name.lower()
            for name in index_names
        )


# ---------------------------------------------------------------------------
# persist_config_values — write path
# ---------------------------------------------------------------------------


class TestPersist:
    def test_inserts_rows(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """A fresh persist with N values writes N rows; returned count
        matches."""
        from docgen.scip_config_index import persist_config_values
        from docgen.scip_config_scanners import ConfigValue

        values = [
            ConfigValue(
                file=tmp_path / 'app.conf',
                key='resources.python',
                value='/usr/bin/python3',
                line_start=42,
            ),
            ConfigValue(
                file=tmp_path / 'app.conf',
                key='resources.azureml',
                value='/opt/azureml/.venv/bin/python',
                line_start=43,
            ),
        ]
        count = persist_config_values(
            source_name='scalaproject',
            config_values=values,
            conn=conn,
        )
        assert count == 2

        cur = conn.execute(
            'SELECT key, value FROM config_values '
            "WHERE source_name='scalaproject' ORDER BY key"
        )
        rows = cur.fetchall()
        assert len(rows) == 2
        assert rows[0] == ('resources.azureml', '/opt/azureml/.venv/bin/python')
        assert rows[1] == ('resources.python', '/usr/bin/python3')

    def test_re_ingest_replaces_old_rows(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Persisting twice for the same source replaces the old rows.
        Mirrors swagger_ingest re-ingest semantics — the manifest is
        the source of truth."""
        from docgen.scip_config_index import persist_config_values
        from docgen.scip_config_scanners import ConfigValue

        # First ingest
        v1 = [
            ConfigValue(tmp_path / 'a.conf', 'old_key', 'old_val', 1),
        ]
        persist_config_values(
            source_name='s', config_values=v1, conn=conn,
        )
        # Second ingest — completely different content
        v2 = [
            ConfigValue(tmp_path / 'a.conf', 'new_key', 'new_val', 2),
        ]
        persist_config_values(
            source_name='s', config_values=v2, conn=conn,
        )

        cur = conn.execute(
            "SELECT key FROM config_values WHERE source_name='s'"
        )
        keys = {row[0] for row in cur.fetchall()}
        assert keys == {'new_key'}, (
            'Re-ingest should replace, not merge'
        )

    def test_isolated_per_source(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Persisting for source A doesn't disturb source B's rows."""
        from docgen.scip_config_index import persist_config_values
        from docgen.scip_config_scanners import ConfigValue

        v_a = [ConfigValue(tmp_path / 'a', 'k', 'val_a', 1)]
        v_b = [ConfigValue(tmp_path / 'b', 'k', 'val_b', 1)]
        persist_config_values(
            source_name='A', config_values=v_a, conn=conn,
        )
        persist_config_values(
            source_name='B', config_values=v_b, conn=conn,
        )

        cur = conn.execute(
            "SELECT value FROM config_values WHERE source_name='A'"
        )
        assert cur.fetchone()[0] == 'val_a'
        cur = conn.execute(
            "SELECT value FROM config_values WHERE source_name='B'"
        )
        assert cur.fetchone()[0] == 'val_b'

    def test_empty_list_clears_source(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Re-ingesting with an empty list clears the source's rows.
        The empty input represents 'no config values for this source'
        — equivalent to running a scan that found nothing."""
        from docgen.scip_config_index import persist_config_values

        # Pre-populate via a prior persist
        from docgen.scip_config_scanners import ConfigValue
        persist_config_values(
            source_name='s',
            config_values=[
                ConfigValue(Path('/x'), 'k', 'v', 1),
            ],
            conn=conn,
        )

        # Empty re-ingest clears
        count = persist_config_values(
            source_name='s', config_values=[], conn=conn,
        )
        assert count == 0
        cur = conn.execute(
            'SELECT COUNT(*) FROM config_values '
            "WHERE source_name='s'"
        )
        assert cur.fetchone()[0] == 0

    def test_file_path_persisted_as_string(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Path objects serialize as strings in SQLite — verify the
        round-trip works (assignment AND read-back)."""
        from docgen.scip_config_index import persist_config_values
        from docgen.scip_config_scanners import ConfigValue

        config_path = tmp_path / 'sub' / 'app.conf'
        v = [ConfigValue(config_path, 'k', 'v', 7)]
        persist_config_values(
            source_name='s', config_values=v, conn=conn,
        )

        cur = conn.execute(
            "SELECT file FROM config_values WHERE source_name='s'"
        )
        stored = cur.fetchone()[0]
        # Path round-trip: stored as str(path); equal to original
        assert stored == str(config_path)


# ---------------------------------------------------------------------------
# query_config_values_by_key — read path
# ---------------------------------------------------------------------------


class TestQueryByKey:
    def test_returns_match(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Looking up an existing key returns one ConfigValue with the
        right fields populated."""
        from docgen.scip_config_index import (
            persist_config_values,
            query_config_values_by_key,
        )
        from docgen.scip_config_scanners import ConfigValue

        v = [
            ConfigValue(
                tmp_path / 'app.conf',
                'resources.python',
                '/usr/bin/python3',
                42,
            ),
        ]
        persist_config_values(
            source_name='c', config_values=v, conn=conn,
        )

        results = query_config_values_by_key(
            source_name='c',
            key='resources.python',
            conn=conn,
        )
        assert len(results) == 1
        r = results[0]
        assert r.key == 'resources.python'
        assert r.value == '/usr/bin/python3'
        assert r.line_start == 42
        assert str(r.file).endswith('app.conf')

    def test_unknown_key_returns_empty(
        self, conn: sqlite3.Connection,
    ) -> None:
        """No match → empty list (not None)."""
        from docgen.scip_config_index import query_config_values_by_key

        results = query_config_values_by_key(
            source_name='c', key='nonexistent', conn=conn,
        )
        assert results == []

    def test_multiple_files_with_same_key(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Same key in multiple files (e.g., layered HOCON includes,
        per-environment overrides) returns ALL matches. Layer C's
        traversal can decide which to use based on context."""
        from docgen.scip_config_index import (
            persist_config_values,
            query_config_values_by_key,
        )
        from docgen.scip_config_scanners import ConfigValue

        v = [
            ConfigValue(tmp_path / 'a.conf', 'shared_key', 'from_a', 1),
            ConfigValue(tmp_path / 'b.conf', 'shared_key', 'from_b', 1),
        ]
        persist_config_values(
            source_name='c', config_values=v, conn=conn,
        )

        results = query_config_values_by_key(
            source_name='c', key='shared_key', conn=conn,
        )
        assert len(results) == 2
        files = {str(r.file) for r in results}
        assert files == {
            str(tmp_path / 'a.conf'),
            str(tmp_path / 'b.conf'),
        }

    def test_isolated_per_source(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Querying one source doesn't return another source's rows
        even if the key matches."""
        from docgen.scip_config_index import (
            persist_config_values,
            query_config_values_by_key,
        )
        from docgen.scip_config_scanners import ConfigValue

        persist_config_values(
            source_name='A',
            config_values=[
                ConfigValue(tmp_path / 'a', 'k', 'val_a', 1),
            ],
            conn=conn,
        )
        persist_config_values(
            source_name='B',
            config_values=[
                ConfigValue(tmp_path / 'b', 'k', 'val_b', 1),
            ],
            conn=conn,
        )

        results = query_config_values_by_key(
            source_name='A', key='k', conn=conn,
        )
        assert len(results) == 1
        assert results[0].value == 'val_a'


# ---------------------------------------------------------------------------
# query_config_values_for_source — bulk read for a source
# ---------------------------------------------------------------------------


class TestQueryForSource:
    def test_returns_all_values(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Returns every ConfigValue for the source, across all keys
        and files."""
        from docgen.scip_config_index import (
            persist_config_values,
            query_config_values_for_source,
        )
        from docgen.scip_config_scanners import ConfigValue

        v = [
            ConfigValue(tmp_path / 'a', 'k1', 'v1', 1),
            ConfigValue(tmp_path / 'a', 'k2', 'v2', 2),
            ConfigValue(tmp_path / 'b', 'k3', 'v3', 1),
        ]
        persist_config_values(
            source_name='s', config_values=v, conn=conn,
        )

        all_values = query_config_values_for_source(
            source_name='s', conn=conn,
        )
        assert len(all_values) == 3
        keys = {cv.key for cv in all_values}
        assert keys == {'k1', 'k2', 'k3'}

    def test_unknown_source_returns_empty(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_index import (
            query_config_values_for_source,
        )

        assert query_config_values_for_source(
            source_name='nonexistent', conn=conn,
        ) == []
