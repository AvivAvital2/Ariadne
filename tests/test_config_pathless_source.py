from __future__ import annotations

from pathlib import Path

import pytest

from config import Config, ConfigError


def _yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body)
    return p


def test_pathless_source_is_serve_only_not_an_error(tmp_path: Path) -> None:
    """A serve-only box lists source NAMES with no path — the dependency graph
    and the docs both come from the database. Such a config must:

    - load without error (no more `path: .` hack to satisfy a required field),
    - expose the source as FOUND but path-less (``path is None``), and
    - resolve to no path — never silently walking cwd.
    """
    cfg = Config(config_path=_yaml(tmp_path, '''
default_source: app
sources:
  app:
    depends_on: [core]
  core:
    path: /x/core
'''))

    sc = cfg.get_source_config('app')
    assert sc is not None and sc.path is None        # found, but serve-only
    assert sc.depends_on == ('core',)
    assert cfg.get_source_path('app') is None         # NOT cwd/app
    assert cfg.resolve_source('app') is None          # build fails closed, no cwd walk
    assert cfg.get_source_config('nope') is None       # genuinely-unknown still None


def test_typo_in_a_source_key_still_fails_loud(tmp_path: Path) -> None:
    """Dropping the hard path-required check must NOT weaken the typo guard:
    an unknown key (the actual footgun) still raises ConfigError."""
    with pytest.raises(ConfigError):
        Config(config_path=_yaml(tmp_path, '''
sources:
  app:
    pth: /x/app
'''))
