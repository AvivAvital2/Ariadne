"""Contract for Phase 2q — config-value index extractor.

Walks ``source_root`` for config files (``.conf`` HOCON, ``.yaml`` /
``.yml``, ``.env``), parses each format, and persists flattened
``(key, value, file, line_start)`` rows to ``config_values``. The
table is the substrate Phase 2s consults when resolving sink-site
arguments that come from ``config.getString("key")``-style getters.

Key conventions:

- Nested keys flatten to dot-paths (``a.b.c`` for HOCON / YAML).
- Value column stores the resolved string. List values are skipped
  in v1 (Phase 2s territory).
- Comments / blank lines / file-format noise → no rows.
- Re-ingest replaces all rows for ``source_name``.

These tests follow the lesson from Phase 2t — stub returns 0, every
test fails behaviorally rather than on missing imports, every "skip"
case is paired with a positive baseline so a stub can't pass it
trivially.
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


def _query_config_values(
    conn: sqlite3.Connection, source_name: str,
) -> list[dict]:
    cur = conn.execute(
        '''SELECT file, key, value, line_start
           FROM config_values WHERE source_name = ?
           ORDER BY file, key, line_start''',
        (source_name,),
    )
    cols = ['file', 'key', 'value', 'line_start']
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _kv(rows: list[dict]) -> dict[str, str]:
    """Helper: flatten rows to ``{key: value}`` for assertions where
    file/line don't matter."""
    return {r['key']: r['value'] for r in rows}


# ---------------------------------------------------------------------------
# HOCON
# ---------------------------------------------------------------------------


class TestHocon:
    def test_flat_key_value_pair(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'app.conf').write_text(
            'name = "deploy-prod"\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        rows = _query_config_values(conn, 'myapi')
        assert _kv(rows).get('name') == 'deploy-prod'

    def test_nested_block_keys_flatten_to_dot_path(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'app.conf').write_text(
            'resources {\n'
            '  python = "/usr/bin/python3"\n'
            '}\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('resources.python') == '/usr/bin/python3'

    def test_dotted_key_path(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'app.conf').write_text(
            'resources.azureMLPythonCommand = '
            '"models/azureml/local_scripts/train.py"\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        # Scalaproject canonical case — Phase 2s consumer needs this key
        assert (
            kv.get('resources.azureMLPythonCommand')
            == 'models/azureml/local_scripts/train.py'
        )

    def test_multiple_keys_emit_multiple_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'app.conf').write_text(
            'host = "localhost"\n'
            'port = 8080\n'
            'debug = true\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('host') == 'localhost'
        # Numeric and boolean values stringify
        assert kv.get('port') == '8080'
        assert kv.get('debug') == 'true'


# ---------------------------------------------------------------------------
# YAML / kustomize
# ---------------------------------------------------------------------------


class TestYaml:
    def test_flat_yaml_key_value(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'app.yaml').write_text(
            'name: my-service\n'
            'replicas: 3\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('name') == 'my-service'
        assert kv.get('replicas') == '3'

    def test_nested_yaml_keys_flatten(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'app.yml').write_text(
            'resources:\n'
            '  limits:\n'
            '    memory: 512Mi\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('resources.limits.memory') == '512Mi'

    def test_yaml_list_values_skipped_scalar_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired bite — scalars emit, lists skip. v1 only stores
        scalar (string-coercible) values; sequence resolution is
        Phase 2s territory. Without the positive baseline a stub
        passes trivially."""
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'app.yaml').write_text(
            'name: scalar-name\n'
            'tags:\n'
            '  - alpha\n'
            '  - beta\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('name') == 'scalar-name'
        # The list entries don't get individual rows
        assert 'tags' not in kv
        assert 'tags.0' not in kv

    def test_kustomize_yaml(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``kustomization.yaml`` is just YAML — no special
        treatment in v1, but the extractor must still pick it up."""
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'kustomization.yaml').write_text(
            'namespace: production\n'
            'commonLabels:\n'
            '  app: my-app\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('namespace') == 'production'
        assert kv.get('commonLabels.app') == 'my-app'


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------


class TestDotenv:
    def test_simple_assignment(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / '.env').write_text(
            'API_KEY=secret\n'
            'API_URL=https://api.example.com\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('API_KEY') == 'secret'
        assert kv.get('API_URL') == 'https://api.example.com'

    def test_quoted_value_quotes_stripped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / '.env').write_text(
            'KEY1="quoted value"\n'
            "KEY2='single quoted'\n"
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('KEY1') == 'quoted value'
        assert kv.get('KEY2') == 'single quoted'

    def test_export_prefix_stripped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / '.env').write_text(
            'export FOO=bar\n'
            'BAZ=qux\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        # 'export' prefix doesn't pollute the key
        assert kv.get('FOO') == 'bar'
        assert kv.get('BAZ') == 'qux'
        # Negative half: confirm 'export FOO' is NOT a key
        assert 'export FOO' not in kv

    def test_comments_and_blanks_skipped_with_real_keys_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired — comment lines must not become keys, real
        assignments must. A stub fails on the positive half; a
        regex impl that grabs ``=`` from comments fails on the
        negative half."""
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / '.env').write_text(
            '# This is a comment with KEY=fake-value\n'
            '\n'
            'REAL=actual-value\n'
            '   \n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('REAL') == 'actual-value'
        assert 'KEY' not in kv
        assert 'fake-value' not in kv.values()


# ---------------------------------------------------------------------------
# Multiple files / file walking
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    def test_walks_subdirectories(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'top.conf').write_text('top = "v1"\n')
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'sub' / 'inner.yaml').write_text('nested: v2\n')
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        rows = _query_config_values(conn, 'myapi')
        kv = _kv(rows)
        # Both files contribute
        assert kv.get('top') == 'v1'
        assert kv.get('nested') == 'v2'
        # File attribution is per-file
        files = {r['file'] for r in rows}
        assert any(f.endswith('top.conf') for f in files)
        assert any(f.endswith('inner.yaml') for f in files)

    def test_non_config_files_not_scanned(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired — Python source contains a ``key=value`` shape
        that a regex impl might pick up. The real config file with
        the same shape must emit; the .py file must not."""
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'real.conf').write_text(
            'real_key = "real_value"\n'
        )
        (tmp_path / 'app.py').write_text(
            'fake_key = "fake_value"\n'
        )
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('real_key') == 'real_value'
        # The .py file's content is NOT scanned
        assert 'fake_key' not in kv
        assert 'fake_value' not in kv.values()


# ---------------------------------------------------------------------------
# Adversarial — error tolerance + format isolation
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_malformed_yaml_skips_only_that_file(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired — bad YAML in one file shouldn't kill the run for
        a healthy file in the same source."""
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        (tmp_path / 'broken.yaml').write_text(
            'key: : :: : not yaml\n'
            '  unindented: bad\n'
        )
        (tmp_path / 'good.conf').write_text('healthy = "ok"\n')
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('healthy') == 'ok'
        # Broken file contributed nothing — bad data didn't leak in

    def test_no_config_files_returns_zero(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Empty-source-root branch paired with a follow-up call
        that has files — the second branch is what bites a stub."""
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        # First branch — no config files
        rc1 = ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        assert rc1 == 0
        assert _query_config_values(conn, 'myapi') == []

        # Second branch — add a config file, re-run, expect rows
        (tmp_path / 'app.conf').write_text('present = "yes"\n')
        rc2 = ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        assert rc2 >= 1
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('present') == 'yes'


# ---------------------------------------------------------------------------
# Re-ingest semantics
# ---------------------------------------------------------------------------


class TestReIngest:
    def test_replaces_same_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        cfg = tmp_path / 'app.conf'
        cfg.write_text('key = "old"\n')
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        # Edit and re-ingest
        cfg.write_text('key = "new"\n')
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        kv = _kv(_query_config_values(conn, 'myapi'))
        assert kv.get('key') == 'new'
        # Old value gone — re-ingest cleaned it up
        assert 'old' not in kv.values()

    def test_preserves_other_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_config_value_extractor import (
            ingest_config_values,
        )

        # Pre-existing row from a different source
        conn.execute(
            '''INSERT INTO config_values
               (source_name, file, key, value, line_start)
               VALUES (?, ?, ?, ?, ?)''',
            ('other', '/x.conf', 'preserved', 'kept', 1),
        )
        conn.commit()

        (tmp_path / 'mine.conf').write_text('mine = "v"\n')
        ingest_config_values(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        my = _kv(_query_config_values(conn, 'myapi'))
        other = _kv(_query_config_values(conn, 'other'))
        assert my.get('mine') == 'v'
        # Other source untouched
        assert other.get('preserved') == 'kept'
