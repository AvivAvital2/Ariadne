"""Tests for the `scip merge` step (``cli/index.py`` ``_SubprocessMerger``).

The multi-language index path shells out to the vendored ``scip merge`` command
(``scripts/scip/merge.go``, built into the Docker image) as
``scip merge --output <out> <inputs...>``. These lock that CLI contract on
Ariadne's side — the exact shape ``merge.go`` implements. Two neighbours cover
the rest: the multi-vs-single *invocation* wiring lives in ``test_cmd_index.py``
(``FakeMerger``: single-language copies, 2+ merges, merge-failure halts), and the
real merge *binary* is exercised end-to-end by the multi-language container
journey (``web/e2e/cypress/e2e/container-journey.cy.js``). Synthetic data only.
"""
from __future__ import annotations

import subprocess

from cli.index import _SubprocessMerger


def test_subprocess_merger_builds_scip_merge_command(tmp_path, monkeypatch):
    """Success path: builds ``scip merge --output <out> <in1> <in2>`` → True,
    and creates the output directory first."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        captured['check'] = kwargs.get('check')
        return subprocess.CompletedProcess(cmd, 0, stdout=b'', stderr=b'')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    out = tmp_path / 'merged' / 'index.scip'
    inputs = [tmp_path / 'python.scip', tmp_path / 'typescript.scip']

    ok = _SubprocessMerger().merge(inputs, out)

    assert ok is True
    assert captured['cmd'] == [
        'scip', 'merge', '--output', str(out), str(inputs[0]), str(inputs[1])]
    assert captured['check'] is True          # a nonzero exit must raise, not pass silently
    assert out.parent.is_dir()                # output dir is ensured before the merge


def test_subprocess_merger_missing_binary_returns_false(tmp_path, monkeypatch):
    """`scip` not on PATH → False (a clean signal to hint + halt), not a crash."""
    def boom(cmd, **kwargs):
        raise FileNotFoundError('scip')

    monkeypatch.setattr(subprocess, 'run', boom)
    ok = _SubprocessMerger().merge([tmp_path / 'a.scip'], tmp_path / 'out.scip')
    assert ok is False


def test_subprocess_merger_nonzero_exit_returns_false(tmp_path, monkeypatch):
    """A nonzero ``scip merge`` exit is surfaced as False, not swallowed."""
    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(2, cmd, stderr=b'merge exploded')

    monkeypatch.setattr(subprocess, 'run', fail)
    ok = _SubprocessMerger().merge([tmp_path / 'a.scip'], tmp_path / 'out.scip')
    assert ok is False
