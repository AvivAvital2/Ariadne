"""The dry-run cost preview must discover files the same way the real
``generate`` does — honoring the source's ``ariadne.yaml`` excludes — so it
doesn't price test suites and deploy configs the run never documents.
"""
from __future__ import annotations

from types import SimpleNamespace


class _Cfg:
    """Minimal config double exposing only what the discovery helper needs."""

    def __init__(self, exclude, excluded_dirs):
        self._exclude = tuple(exclude)
        self._excluded_dirs = set(excluded_dirs)

    def get_source_config(self, name):
        return SimpleNamespace(exclude=self._exclude)

    def resolve_excluded_dirs(self, name):
        return self._excluded_dirs


def test_dry_run_discovery_honors_config_excludes(tmp_path):
    from cli.generation import _discover_files_for_estimate

    (tmp_path / 'app.py').write_text('x = 1\n')
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg' / 'core.py').write_text('y = 2\n')
    (tmp_path / 'pkg' / 'legacy.py').write_text('z = 3\n')   # dropped by pattern
    (tmp_path / 'test').mkdir()
    (tmp_path / 'test' / 'helper.py').write_text('t = 4\n')  # dropped by dir name

    cfg = _Cfg(exclude=('**/legacy.py',), excluded_dirs={'test'})
    files = _discover_files_for_estimate(cfg, 'mylib', tmp_path)

    rels = {p.relative_to(tmp_path).as_posix() for p, _ in files}
    assert rels == {'app.py', 'pkg/core.py'}
    # sizes are carried through for the estimator
    assert all(isinstance(size, int) and size > 0 for _, size in files)
