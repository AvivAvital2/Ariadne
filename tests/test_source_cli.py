"""Contract test for the ``ariadne source`` command group.

A single test that GROWS one demand at a time, in lockstep with the
implementation — each demand below was added only after the previous
one was green:

  D1 — `source add` in a project with NO ariadne.yaml creates the file,
       writes ``sources.<name>.path``, and (since there was no default)
       sets ``default_source`` to the new source.
  D2 — `source add` for a second source persists list/scalar options
       (depends_on, exclude, exclude_dirs) and leaves the existing
       source + default_source untouched.
  D3 — re-running `source add` for an existing source UPDATES only the
       provided field (parent) and preserves the rest (idempotent).
  D4 — `source list` exits 0 and names every configured source.
  D5 — `source remove --yes` deletes a source, preserving the others.
  D6 — with required args omitted on a TTY, `source add` PROMPTS for
       name and path (flag values are optional fallbacks).
  D7 — branch-aware fields: `--branches` (comma list) and `--ref`
       persist for branch-scoped / ref-pinned sources.
  D8 — `--skip-dependency-detection` opts the source out of the
       cross-source import scan; omitting it keeps the default (off).

The test drives the real parser (``cli.create_parser``) and dispatches
through the registered handler, so command registration is covered too.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _run(argv: list[str]) -> int:
    """Parse ``argv`` with the real CLI parser and dispatch its handler."""
    from cli.main import create_parser
    from cli.integration import HANDLERS

    args = create_parser().parse_args(argv)
    return HANDLERS[args.command](args)


def _fresh_config(path: Path):
    """Read the on-disk yaml back through a brand-new Config instance,
    bypassing the cached singleton so assertions see persisted state."""
    from config import Config

    return Config(config_path=path)


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    """Point Ariadne at a not-yet-existing ariadne.yaml inside tmp_path
    and reset the cached global so each `source` command starts from the
    same blank slate a new project would have."""
    cfg_path = tmp_path / 'ariadne.yaml'
    monkeypatch.setenv('ARIADNE_CONFIG', str(cfg_path))
    monkeypatch.chdir(tmp_path)
    import config as config_module

    monkeypatch.setattr(config_module, '_global_config', None, raising=False)
    return cfg_path


def test_source_group_evolves_through_contract(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'ariadne.yaml'
    src_a = tmp_path / 'repo_a'
    src_a.mkdir()
    src_b = tmp_path / 'repo_b'
    src_b.mkdir()

    # Non-interactive: required values come from flags, never a prompt.
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)

    # ---- D1: add into a project with no ariadne.yaml ----------------
    assert not cfg_path.exists()
    rc = _run(['source', 'add', 'alpha', '--path', str(src_a)])
    assert rc == 0
    assert cfg_path.exists(), 'source add must bootstrap ariadne.yaml'

    cfg = _fresh_config(cfg_path)
    sc = cfg.get_source_config('alpha')
    assert sc is not None and sc.path == str(src_a)
    assert cfg.default_source == 'alpha', 'first source becomes default'

    # ---- D2: second source with options; first one untouched --------
    rc = _run([
        'source', 'add', 'beta', '--path', str(src_b),
        '--depends-on', 'alpha',
        '--exclude', '**/.env',
        '--exclude-dirs', 'build,dist',
    ])
    assert rc == 0

    cfg = _fresh_config(cfg_path)
    sb = cfg.get_source_config('beta')
    assert sb is not None
    assert sb.path == str(src_b)
    assert tuple(sb.depends_on) == ('alpha',)
    assert tuple(sb.exclude) == ('**/.env',)
    assert tuple(sb.exclude_dirs) == ('build', 'dist')
    # Adding a second source must not steal default or clobber the first.
    assert cfg.default_source == 'alpha'
    assert cfg.get_source_config('alpha').path == str(src_a)

    # ---- D3: idempotent update preserves untouched fields -----------
    rc = _run(['source', 'add', 'beta', '--parent', 'alpha'])
    assert rc == 0

    cfg = _fresh_config(cfg_path)
    sb = cfg.get_source_config('beta')
    assert sb.parent == 'alpha'
    assert sb.path == str(src_b), 'update must not drop existing path'
    assert tuple(sb.depends_on) == ('alpha',), 'update must keep deps'

    # ---- D4: list names every configured source ---------------------
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = _run(['source', 'list'])
    assert rc == 0
    out = buf.getvalue()
    assert 'alpha' in out and 'beta' in out

    # ---- D5: remove deletes one, preserves the rest -----------------
    rc = _run(['source', 'remove', 'beta', '--yes'])
    assert rc == 0
    cfg = _fresh_config(cfg_path)
    assert cfg.get_source_config('beta') is None
    assert cfg.get_source_config('alpha') is not None

    # ---- D6: prompt fallback when required args omitted on a TTY ----
    src_c = tmp_path / 'repo_c'
    src_c.mkdir()
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    answers = iter(['gamma', str(src_c)])
    monkeypatch.setattr('builtins.input', lambda *a, **k: next(answers))

    rc = _run(['source', 'add'])
    assert rc == 0
    cfg = _fresh_config(cfg_path)
    sg = cfg.get_source_config('gamma')
    assert sg is not None and sg.path == str(src_c)

    # ---- D7: branch-aware fields (branches + ref) ------------------
    src_d = tmp_path / 'repo_d'
    src_d.mkdir()
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    rc = _run([
        'source', 'add', 'delta', '--path', str(src_d),
        '--branches', 'feature/*,main',
        '--ref', 'main',
    ])
    assert rc == 0
    cfg = _fresh_config(cfg_path)
    sd = cfg.get_source_config('delta')
    assert tuple(sd.branches) == ('feature/*', 'main')
    assert sd.ref == 'main'

    # ---- D8: --skip-dependency-detection opts the source out of the
    # cross-source import scan and persists to ariadne.yaml ----------
    src_e = tmp_path / 'repo_e'
    src_e.mkdir()
    rc = _run([
        'source', 'add', 'epsilon', '--path', str(src_e),
        '--skip-dependency-detection',
    ])
    assert rc == 0
    cfg = _fresh_config(cfg_path)
    se = cfg.get_source_config('epsilon')
    assert se.skip_dependency_detection is True
    assert cfg.source_skip_dependency_detection('epsilon') is True
    # Omitting the flag leaves the default (off), not None-clobbered.
    assert cfg.get_source_config('delta').skip_dependency_detection is False
