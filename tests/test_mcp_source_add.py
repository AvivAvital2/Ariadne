"""Contract test for the ``ariadne_source_add`` MCP tool.

A single test that GROWS one demand at a time, in lockstep with the
implementation (mirrors ``tests/test_source_cli.py`` for the CLI):

  D1 — adding the first source persists ``sources.<name>.path``, sets
       ``default_source`` (first source), and the response reflects all of
       that. A plain directory reports ``git.is_repo == False``.
  D2 — a second source persists list options (depends_on, exclude,
       exclude_dirs) without stealing the default or clobbering the first.
  D3 — re-adding an existing source updates only the provided field
       (parent) and preserves the rest (idempotent); ``created == False``.
  D4 — a real git working tree is detected: ``git.is_repo`` with a branch,
       a positive ``file_count``, and a ``last_commit_relative``.

The test drives the registered tool function directly, so the in-process
service path (config bootstrap + read-back) is covered end to end.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ariadne_mcp.server_admin import ariadne_list_sources, ariadne_source_add
from config import Config


def _fresh_config(path: Path) -> Config:
    """Read the on-disk yaml back through a brand-new Config, bypassing
    the cached singleton so assertions see persisted state."""
    return Config(config_path=path)


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Point Ariadne at an isolated ariadne.yaml and reset both the cached
    global Config and the MCP service singleton, so each call starts from a
    blank slate.

    The tmp config is **pre-created empty** so ``config_search_paths()``
    selects it first. Without that, an ``ARIADNE_CONFIG`` that points at a
    not-yet-created file is skipped and the package-root ``ariadne.yaml``
    (a dev's real config) shadows it — the tests would then read/write the
    developer's actual config. (This is the documented config-resolution
    quirk; isolating at the test level avoids changing prod behavior.)
    """
    cfg_path = tmp_path / 'ariadne.yaml'
    cfg_path.write_text('sources: {}\n')
    monkeypatch.setenv('ARIADNE_CONFIG', str(cfg_path))
    monkeypatch.chdir(tmp_path)

    import config as config_module
    monkeypatch.setattr(config_module, '_global_config', None, raising=False)

    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, '_instance', None, raising=False)
    return cfg_path


def _git(args: list[str], cwd: Path) -> None:
    """Run a git command in ``cwd`` with identity flags so commits work in CI."""
    subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args],
        cwd=cwd, check=True, capture_output=True, text=True,
    )


async def test_source_add_tool_evolves_through_contract(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'ariadne.yaml'
    src1 = tmp_path / 'src1'
    src1.mkdir()
    src2 = tmp_path / 'src2'
    src2.mkdir()

    # ---- D1: first source persists, becomes default ------------------
    resp = await ariadne_source_add('alpha', path=str(src1))
    assert resp.source == 'alpha'
    assert resp.path == str(src1)
    assert resp.created is True
    assert resp.is_default is True
    assert resp.git is not None and resp.git.is_repo is False

    cfg = _fresh_config(cfg_path)
    sc = cfg.get_source_config('alpha')
    assert sc is not None and sc.path == str(src1)
    assert cfg.default_source == 'alpha', 'first source becomes default'

    # ---- D2: second source with options; first untouched -------------
    resp = await ariadne_source_add(
        'beta',
        path=str(src2),
        depends_on=['alpha'],
        exclude=['**/.env'],
        exclude_dirs=['build', 'dist'],
    )
    assert resp.created is True
    assert resp.is_default is False, 'second source must not steal default'
    assert resp.depends_on == ['alpha']
    assert resp.exclude == ['**/.env']
    assert resp.exclude_dirs == ['build', 'dist']

    cfg = _fresh_config(cfg_path)
    sb = cfg.get_source_config('beta')
    assert tuple(sb.depends_on) == ('alpha',)
    assert tuple(sb.exclude) == ('**/.env',)
    assert tuple(sb.exclude_dirs) == ('build', 'dist')
    assert cfg.default_source == 'alpha'
    assert cfg.get_source_config('alpha').path == str(src1)

    # ---- D3: idempotent update preserves untouched fields ------------
    resp = await ariadne_source_add('beta', parent='alpha')
    assert resp.created is False, 'updating an existing source is not a create'
    assert resp.parent == 'alpha'
    assert resp.path == str(src2), 'update must not drop existing path'

    cfg = _fresh_config(cfg_path)
    sb = cfg.get_source_config('beta')
    assert sb.parent == 'alpha'
    assert sb.path == str(src2)
    assert tuple(sb.depends_on) == ('alpha',), 'update must keep deps'

    # ---- D4: a real git working tree is detected ---------------------
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(['init'], repo)
    (repo / 'a.py').write_text('x = 1\n')
    _git(['add', '.'], repo)
    _git(['commit', '-m', 'init'], repo)

    resp = await ariadne_source_add('gamma', path=str(repo))
    assert resp.git is not None
    assert resp.git.is_repo is True
    assert resp.git.branch, 'a git tree reports its branch'
    assert resp.git.file_count and resp.git.file_count >= 1
    assert resp.git.last_commit_relative, 'committed repo reports last-commit age'

    # ---- D5: a non-existent path is rejected up front (not 3 steps later) ----
    with pytest.raises(ValueError):
        await ariadne_source_add('epsilon', path=str(tmp_path / 'no_such_dir'))

    # ---- D6: list_sources returns every configured source (the deps picker) --
    listing = ariadne_list_sources()
    names = {s.name for s in listing.sources}
    assert {'alpha', 'beta', 'gamma'} <= names
    assert 'epsilon' not in names  # the rejected add never persisted
    assert listing.default_source == 'alpha'

    # ---- D7: exempt_dirs overrides a default-excluded directory ------------
    # 'vendor' is in DEFAULT_EXCLUDE_POLICY; exempting it must remove it from
    # the effective excludes (the "include something the default skips" knob),
    # while other defaults stay excluded.
    resp = await ariadne_source_add('alpha', exempt_dirs=['vendor'])
    assert resp.exempt_dirs == ['vendor']
    cfg = _fresh_config(cfg_path)
    assert tuple(cfg.get_source_config('alpha').exempt_dirs) == ('vendor',)
    effective = cfg.resolve_excluded_dirs('alpha')
    assert 'vendor' not in effective       # overridden back in
    assert 'node_modules' in effective     # other defaults still excluded
    # ---- D8: doc_types_by_language persists + the response echoes it -------
    # The doc-type screen's per-format excludes, written via source add.
    resp = await ariadne_source_add(
        'alpha',
        doc_types_by_language={
            'python': ['explanation'],
            'yaml': ['explanation', 'architecture'],
        },
    )
    assert resp.doc_types_by_language == {
        'python': ['explanation'],
        'yaml': ['explanation', 'architecture'],
    }, resp.doc_types_by_language
    cfg = _fresh_config(cfg_path)
    assert cfg.get_source_config('alpha').doc_types_by_language == {
        'python': ('explanation',),
        'yaml': ('explanation', 'architecture'),
    }
