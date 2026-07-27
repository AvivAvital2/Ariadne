"""`check_tests.py coverage` — a coverage mode so test runs (incl. coverage)
go through the single sanctioned entry point instead of a bare `pytest`.

It must run pytest UNDER coverage using the numpy/py3.14-safe flags
(`coverage run -m pytest --import-mode=prepend`), route `--source`/`--include`
to coverage rather than pytest, and forward everything else to pytest.
"""
from __future__ import annotations

import sys

import check_tests


def test_coverage_mode_parses_args_and_builds_commands():
    # --source / --include are consumed by coverage; the rest go to pytest.
    source, include, pytest_args = check_tests.parse_coverage_args(
        ['--source=docgen.catalog_extractor', 'tests/test_x.py', '-k', 'docker'],
    )
    assert source == 'docgen.catalog_extractor'
    assert include is None
    assert pytest_args == ['tests/test_x.py', '-k', 'docker']
    assert '--source=docgen.catalog_extractor' not in pytest_args

    # The run command must NOT carry --source: on py3.14 that double-loads
    # numpy's C extension and aborts collection. --source instead scopes the
    # REPORT, derived to a coverage --include glob (legacy interface preserved).
    run_cmd = check_tests.build_coverage_run_cmd(pytest_args)
    assert run_cmd == [
        sys.executable, '-m', 'coverage', 'run',
        '-m', 'pytest', '--import-mode=prepend',
        'tests/test_x.py', '-k', 'docker',
    ]
    assert '--source=docgen.catalog_extractor' not in run_cmd
    assert check_tests._include_from_source(source) == '*/docgen/catalog_extractor.py'

    # report command omits --include unless one was given...
    assert check_tests.build_coverage_report_cmd(None) == [
        sys.executable, '-m', 'coverage', 'report', '-m',
    ]

    # ...and includes it when present (same behavior, grown demand). Also
    # supports the space-separated `--source X` / `--include PAT` forms.
    source2, include2, rest2 = check_tests.parse_coverage_args(
        ['--source', 'cli.sync', '--include', '*/cli/sync.py', 'tests/test_sync.py'],
    )
    assert source2 == 'cli.sync'
    assert include2 == '*/cli/sync.py'
    assert rest2 == ['tests/test_sync.py']
    assert check_tests.build_coverage_report_cmd(include2) == [
        sys.executable, '-m', 'coverage', 'report', '-m', '--include=*/cli/sync.py',
    ]


def test_green_mode_accepts_xfailed_flags_strict_xpass():
    # An xfail-marked test that fails is an EXPECTED failure (the repo's
    # security-guard convention: record the finding, keep the gate green).
    # A strict xpass arrives from pytest-json-report as outcome='failed'
    # and must still gate.
    from check_tests import report_green

    assert report_green({'tests': [
        {'nodeid': 't::guard', 'outcome': 'xfailed'},
        {'nodeid': 't::ok', 'outcome': 'passed'},
    ]}) is True
    assert report_green({'tests': [
        {'nodeid': 't::broke', 'outcome': 'failed'},
    ]}) is False
