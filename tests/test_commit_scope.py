"""Commit-diff incremental scope: generate/onboard regenerate only files
changed since the last synced commit, then promote the commit.

These cover the git plumbing (git_ops) and the scope resolver
(cli.generate._commit_scope) that turns a source's stored sync_state into the
set passed to the orchestrator as ``restrict_to_files``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> None:
    subprocess.run(['git', *args], cwd=repo, check=True,
                   capture_output=True, text=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.email', 't@t')
    _git(repo, 'config', 'user.name', 't')


def _commit(repo: Path, msg: str) -> str:
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', msg)
    out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo,
                         check=True, capture_output=True, text=True)
    return out.stdout.strip()


class TestGitHelpers:
    def test_changed_files_since_and_head(self, tmp_path: Path) -> None:
        from git_ops import get_changed_files_since, get_head_commit

        repo = tmp_path / 'repo'
        _init_repo(repo)
        (repo / 'a.py').write_text('a = 1\n')
        (repo / 'b.py').write_text('b = 1\n')
        c1 = _commit(repo, 'init')

        (repo / 'a.py').write_text('a = 2\n')   # modified
        (repo / 'c.py').write_text('c = 1\n')   # added
        c2 = _commit(repo, 'change')

        assert get_head_commit(repo) == c2
        changed = set(get_changed_files_since(c1, repo))
        assert changed == {'a.py', 'c.py'}       # b.py unchanged → excluded

    def test_non_git_dir_returns_none(self, tmp_path: Path) -> None:
        from git_ops import get_changed_files_since, get_head_commit

        plain = tmp_path / 'plain'
        plain.mkdir()
        assert get_head_commit(plain) is None
        assert get_changed_files_since('deadbeef', plain) is None


class TestCommitScope:
    """``_commit_scope`` turns sync_state into (restrict_to_files, head):
    - never synced → (None, head): full pass, but head is returned to set the
      baseline after success;
    - synced → (frozenset(changed-since-last), head);
    - --force → (None, head): full pass, still promote;
    - non-git → (None, None): legacy staleness, nothing to promote."""

    def _setup(self, tmp_path: Path):
        from library import Library
        repo = tmp_path / 'repo'
        _init_repo(repo)
        (repo / 'a.py').write_text('a = 1\n')
        (repo / 'b.py').write_text('b = 1\n')
        c1 = _commit(repo, 'init')
        db = tmp_path / 'lib.db'
        return repo, c1, db, Library

    def test_never_synced_returns_none_scope_with_head(self, tmp_path: Path) -> None:
        from cli.generate import _commit_scope
        repo, c1, db, _ = self._setup(tmp_path)
        scope, head = _commit_scope(db, 'src', repo, force=False)
        assert scope is None and head == c1   # full first pass; baseline to set

    def test_synced_returns_changed_set(self, tmp_path: Path) -> None:
        from cli.generate import _commit_scope
        from library import Library
        repo, c1, db, _ = self._setup(tmp_path)
        with Library(db) as lib:
            lib.set_sync_state('src', c1)
        (repo / 'a.py').write_text('a = 2\n')
        c2 = _commit(repo, 'change')
        scope, head = _commit_scope(db, 'src', repo, force=False)
        assert scope == frozenset({'a.py'}) and head == c2

    def test_force_returns_none_scope_but_head(self, tmp_path: Path) -> None:
        from cli.generate import _commit_scope
        from library import Library
        repo, c1, db, _ = self._setup(tmp_path)
        with Library(db) as lib:
            lib.set_sync_state('src', c1)
        scope, head = _commit_scope(db, 'src', repo, force=True)
        assert scope is None and head == c1   # full pass, still promotable

    def test_non_git_returns_none_none(self, tmp_path: Path) -> None:
        from cli.generate import _commit_scope
        plain = tmp_path / 'plain'
        plain.mkdir()
        db = tmp_path / 'lib.db'
        assert _commit_scope(db, 'src', plain, force=False) == (None, None)
