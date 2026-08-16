"""A clean checkout must carry everything its own tests need.

Committed suites once depended on untracked helper scripts, so a fresh
`git archive HEAD` failed tests the dirty checkout passed. This smoke
test collects an archive of HEAD and runs the file-dependent suites
inside it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_git_archive_head_runs_the_file_dependent_suites(tmp_path):
    probe = subprocess.run(
        ["git", "rev-parse", "--git-dir"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if probe.returncode != 0:
        pytest.skip("not a git checkout (archive snapshots skip this)")

    archive = subprocess.run(
        ["git", "archive", "HEAD"], cwd=ROOT, check=True,
        stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x"], cwd=tmp_path, input=archive.stdout,
                   check=True)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_item_lifecycle_audit.py",
         "tests/test_ask_synthesis_config.py::"
         "test_graph_inspector_bootstraps_repository_import_path"],
        cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)

    assert result.returncode == 0, result.stdout[-2000:]
