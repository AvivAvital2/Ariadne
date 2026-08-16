"""`ariadne spools` tells you when a spool has no declared scope.

Scope gating (designs/spool-scope.md) is inert until someone populates
``spools.<name>.projects``: an empty list means "not declared yet" and admits
every project, so an existing config keeps working on upgrade instead of
silently losing its spool. That back-compat is only safe if the undeclared
state is *visible* — otherwise every project keeps ranking against an
environment most of them don't run on, and nobody is told.

Reported as an ADVISORY, not a gap. The existing convention in `_status` is
explicit about the distinction — an older-extraction-coverage pack prints ``⚠``
and does not affect the exit code, because the pack is still usable. An
undeclared scope is the same shape: the spool is registered and serving. Gaps
mean the spool did NOT load, `ariadne spools` exits 1 on them, and
``spools enable`` refuses to reconcile against one — so filing this as a gap
would both break CI on upgrade and block the very command that fixes it.

Synthetic fixtures only: fake spool name, fake runtime, tmp store.
"""
import argparse
import os

import pytest

import config as config_module
from cli.spools_cmd import _status
from config import Config


def _cache_manifest(cfg_dir, name, runtime='rt'):
    d = cfg_dir / '.ariadne' / 'spools' / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'manifest.yaml').write_text(
        f'environment: {name}\nversion: "1.0.0"\n'
        f'target_runtime: {runtime}\nchecksum: abc123\n'
    )


@pytest.fixture
def project(tmp_path):
    """Two configured sources and one registered spool; scope set by the test."""
    for name in ('alpha', 'beta'):
        (tmp_path / name).mkdir()
    _cache_manifest(tmp_path, 'fakebricks')

    def _write(projects_line: str) -> Config:
        (tmp_path / 'ariadne.yaml').write_text(
            'default_source: alpha\n'
            'sources:\n'
            f'  alpha:\n    path: {tmp_path / "alpha"}\n'
            f'  beta:\n    path: {tmp_path / "beta"}\n'
            'spools:\n'
            '  fakebricks:\n'
            '    runtime: rt\n'
            f'{projects_line}'
        )
        cfg = Config(tmp_path / 'ariadne.yaml')
        os.environ['ARIADNE_CONFIG'] = str(tmp_path / 'ariadne.yaml')
        config_module._global_config = cfg
        return cfg

    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    try:
        yield _write
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env


class TestUndeclaredScopeIsAdvised:
    def test_undeclared_scope_is_reported_without_failing(
        self, project, tmp_path, capsys,
    ) -> None:
        project('')                       # no `projects:` key at all
        rc = _status(argparse.Namespace(cache_dir=None))
        out = capsys.readouterr().out

        assert 'registered' in out, 'the spool still loads and serves'
        assert rc == 0, (
            'an undeclared scope is advisory — the spool is usable, so this '
            'must not fail a health check the way a real gap does'
        )
        assert 'scope-undeclared' in out
        # names the projects that are silently in scope, and how to fix it
        assert 'alpha' in out and 'beta' in out
        assert 'spools enable fakebricks --project' in out

    def test_declared_scope_is_silent(self, project, capsys) -> None:
        project('    projects: [alpha]\n')
        rc = _status(argparse.Namespace(cache_dir=None))
        out = capsys.readouterr().out

        assert rc == 0
        assert 'scope-undeclared' not in out, (
            'scope is declared — nothing to advise about'
        )
