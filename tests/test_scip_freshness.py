"""Freshness: is the index behind the working tree?

The rebuilt module owns only that question. Its path functions delegate to
``docgen/scip_paths.py``, so the over-stripping rule it used to carry (longest matching
indexer ``cwd``, which turns ``spark/sql/core/...`` into ``core/...``) cannot come back in a
second copy.

Two behaviours are deliberate and easy to "fix" into bugs, so both are pinned: a file that
resolves under no indexer ``cwd`` stays unflagged, and a source with no manifest stays
silent. *Cannot determine* is not *fresh*, and a false alarm trains people to ignore the
warning.

Synthetic fixtures only.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from docgen.scip_freshness import (
    absolute_from_scip,
    changed_indexed_files,
    freshness_warning,
    resolve_scip_file,
    source_for_file,
    stale_report,
    stale_report_for_files,
)
from library.scip import init_scip_schema


def _iso(**delta):
    return (datetime.now(timezone.utc) - timedelta(**delta)).isoformat()


def _cfg(paths, *, ignore=None):
    """The three config methods this module uses, and nothing else."""
    return SimpleNamespace(
        get_all_source_paths=lambda: paths,
        source_ignore_staleness=lambda name: (ignore or {}).get(name, False),
        get_source_config=lambda name: (SimpleNamespace(path=str(paths[name]))
                                        if name in paths else None),
    )


def _source(tmp_path, name='src1', *, cwds=('.',), indexed_days_ago=1):
    root = tmp_path / name
    (root / '.ariadne').mkdir(parents=True)
    (root / '.ariadne' / 'manifest.json').write_text(json.dumps({
        'indexers': [{'cwd': cwd, 'indexed_at': _iso(days=indexed_days_ago)}
                     for cwd in cwds],
    }))
    return root


def _touch(root, relative, *, days_ago=0):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')
    when = time.time() - days_ago * 86400
    os.utime(path, (when, when))
    return path


class TestChangedFiles:
    def test_a_file_newer_than_the_index_is_changed(self, tmp_path):
        root = _source(tmp_path, indexed_days_ago=2)
        _touch(root, 'pkg/a.py', days_ago=0)

        indexed_at, changed = changed_indexed_files(
            root, [{'cwd': '.', 'indexed_at': _iso(days=2)}], ['pkg/a.py'])

        assert changed == ['pkg/a.py']
        assert indexed_at is not None

    def test_a_file_older_than_the_index_is_not_changed(self, tmp_path):
        root = _source(tmp_path)
        _touch(root, 'pkg/a.py', days_ago=5)

        _, changed = changed_indexed_files(
            root, [{'cwd': '.', 'indexed_at': _iso(days=1)}], ['pkg/a.py'])

        assert changed == []

    def test_a_file_under_no_indexer_cwd_stays_unflagged(self, tmp_path):
        """Conservative on purpose: a re-index is the remedy for adds and removals."""
        root = _source(tmp_path)

        _, changed = changed_indexed_files(
            root, [{'cwd': '.', 'indexed_at': _iso(days=1)}], ['gone/absent.py'])

        assert changed == []

    def test_an_indexer_with_no_timestamp_cannot_flag_anything(self, tmp_path):
        root = _source(tmp_path)
        _touch(root, 'pkg/a.py')

        _, changed = changed_indexed_files(root, [{'cwd': '.'}], ['pkg/a.py'])

        assert changed == []


class TestStaleReport:
    def test_a_source_behind_its_tree_is_reported(self, tmp_path):
        root = _source(tmp_path, indexed_days_ago=3)
        _touch(root, 'pkg/a.py')

        report = stale_report_for_files(_cfg({'src1': root}), {'src1': ['pkg/a.py']})

        assert 'src1' in report
        assert report['src1'][1] == ['pkg/a.py']

    def test_a_fully_exempt_source_is_skipped(self, tmp_path):
        root = _source(tmp_path, indexed_days_ago=3)
        _touch(root, 'pkg/a.py')

        report = stale_report_for_files(
            _cfg({'src1': root}, ignore={'src1': True}), {'src1': ['pkg/a.py']})

        assert report == {}

    def test_a_glob_exemption_filters_matching_files_only(self, tmp_path):
        root = _source(tmp_path, indexed_days_ago=3)
        _touch(root, 'vendor/v.py')
        _touch(root, 'pkg/a.py')

        report = stale_report_for_files(
            _cfg({'src1': root}, ignore={'src1': ['vendor/**']}),
            {'src1': ['vendor/v.py', 'pkg/a.py']})

        assert report['src1'][1] == ['pkg/a.py']

    def test_a_source_with_no_manifest_stays_silent(self, tmp_path):
        """Cannot determine is not fresh — and must not be reported as stale either."""
        root = tmp_path / 'src1'
        root.mkdir()
        _touch(root, 'pkg/a.py')

        report = stale_report_for_files(_cfg({'src1': root}), {'src1': ['pkg/a.py']})

        assert report == {}

    def test_stale_report_reads_the_indexed_files_from_the_store(self, tmp_path):
        root = _source(tmp_path, indexed_days_ago=3)
        _touch(root, 'pkg/a.py')
        conn = sqlite3.connect(':memory:')
        init_scip_schema(conn)
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
            'line_start, line_end, kind, display_name, qualified_name, '
            "parent_qualified_name) VALUES ('s','src1','python','pkg/a.py',1,2,"
            "'','','m.a','')")
        conn.commit()
        try:
            report = stale_report(conn, _cfg({'src1': root}), ['src1'])
        finally:
            conn.close()

        assert report['src1'][1] == ['pkg/a.py']


class TestPathDelegation:
    def test_the_owning_source_is_the_longest_matching_root(self, tmp_path):
        outer = tmp_path / 'outer'
        inner = outer / 'nested'
        inner.mkdir(parents=True)
        target = inner / 'pkg' / 'a.py'
        target.parent.mkdir(parents=True)
        target.write_text('x')

        cfg = _cfg({'outer': outer, 'inner': inner})

        assert source_for_file(cfg, target) == 'inner'

    def test_a_file_outside_every_source_resolves_to_nothing(self, tmp_path):
        cfg = _cfg({'src1': tmp_path / 'src1'})

        assert source_for_file(cfg, tmp_path / 'elsewhere' / 'x.py') is None
        assert resolve_scip_file(cfg, tmp_path / 'elsewhere' / 'x.py') == (None, None)

    def test_a_verified_resolve_prefers_the_candidate_the_index_confirms(self, tmp_path):
        """Where the old rule over-stripped: `spark` is right, `spark/sql` is not."""
        root = _source(tmp_path, cwds=('spark', 'spark/sql'))
        target = _touch(root, 'spark/sql/core/app.py')
        conn = sqlite3.connect(':memory:')
        init_scip_schema(conn)
        for file in ('sql/core/app.py', 'core/app.py'):
            conn.execute(
                'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
                'line_start, line_end, kind, display_name, qualified_name, '
                "parent_qualified_name) VALUES (?,'src1','python',?,1,2,'','','m','')",
                (f'sym:{file}', file))
        conn.commit()
        try:
            resolved = resolve_scip_file(_cfg({'src1': root}), target, conn)
        finally:
            conn.close()

        assert resolved == ('src1', 'sql/core/app.py')

    def test_an_unverified_resolve_strips_the_least_rather_than_the_most(self, tmp_path):
        root = _source(tmp_path, cwds=('spark', 'spark/sql'))
        target = _touch(root, 'spark/sql/core/app.py')

        assert resolve_scip_file(_cfg({'src1': root}), target) == (
            'src1', 'sql/core/app.py')

    def test_absolute_from_scip_inverts_the_strip(self, tmp_path):
        root = _source(tmp_path, cwds=('spark',))
        expected = _touch(root, 'spark/sql/core/app.py')

        assert absolute_from_scip(_cfg({'src1': root}), 'src1',
                                 'sql/core/app.py') == str(expected)


def test_the_warning_names_the_source_the_count_and_the_remedy():
    message = freshness_warning('src1', '2026-01-01', ['a.py', 'b.py', 'c.py', 'd.py'])

    assert 'src1' in message
    assert '4 indexed file(s)' in message
    assert '+1 more' in message
    assert 'ariadne index --source src1' in message
