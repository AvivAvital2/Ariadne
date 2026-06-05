"""Tests for Ariadne configuration module."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

import config as config_mod
from config import Config


class TestConfigResolution:
    """Where Ariadne finds ``ariadne.yaml`` — grown rung by rung.

    The load-bearing rung is the **package/repo-root** fallback: ``ariadne
    mcp`` is launched via ``uv run --directory <repo>``, which does NOT chdir
    the spawned child, so a cwd-only search misses and the server silently
    serves an empty source list. Config must resolve its yaml with no
    ``ARIADNE_CONFIG`` and no manual setup.
    """

    @staticmethod
    def _write_cfg(path: Path, source: str) -> None:
        path.write_text(
            f'default_source: {source}\nsources:\n  {source}:\n    path: .\n'
        )

    def test_resolution_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir()
        self._write_cfg(repo / 'ariadne.yaml', 'fromrepo')
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()

        monkeypatch.delenv('ARIADNE_CONFIG', raising=False)
        monkeypatch.chdir(elsewhere)
        monkeypatch.setenv('HOME', str(elsewhere))   # home has no ariadne.yaml
        monkeypatch.setattr(config_mod, '_PACKAGE_ROOT', repo, raising=False)

        # 1) cwd has none → fall back to the package/repo root (the real fix).
        assert sorted(config_mod.Config().sources) == ['fromrepo']

        # 2) cwd wins over the package root when it has its own config.
        self._write_cfg(elsewhere / 'ariadne.yaml', 'fromcwd')
        assert sorted(config_mod.Config().sources) == ['fromcwd']

        # 3) ARIADNE_CONFIG wins over everything.
        env_cfg = tmp_path / 'env.yaml'
        self._write_cfg(env_cfg, 'fromenv')
        monkeypatch.setenv('ARIADNE_CONFIG', str(env_cfg))
        assert sorted(config_mod.Config().sources) == ['fromenv']


class TestMcpStartupWarning:
    """The MCP server must fail LOUD when it starts with no sources, instead of
    silently serving ``configured sources: []`` (the failure that made this
    impossible to diagnose from the outside)."""

    def test_warns_loudly_when_no_sources(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import ariadne_mcp.server as server

        empty = tmp_path / 'empty'
        empty.mkdir()
        monkeypatch.delenv('ARIADNE_CONFIG', raising=False)
        monkeypatch.chdir(empty)
        monkeypatch.setenv('HOME', str(empty))
        monkeypatch.setattr(config_mod, '_PACKAGE_ROOT', empty, raising=False)
        monkeypatch.setattr(config_mod, '_global_config', None, raising=False)

        with caplog.at_level(logging.WARNING):
            server._warn_if_no_sources()

        assert any('no sources' in r.getMessage().lower() for r in caplog.records)
        assert 'ARIADNE_CONFIG' in caplog.text

        # ...and it stays SILENT once sources resolve (no false alarm).
        caplog.clear()
        (empty / 'ariadne.yaml').write_text(
            'default_source: ok\nsources:\n  ok:\n    path: .\n'
        )
        monkeypatch.setattr(config_mod, '_global_config', None, raising=False)
        with caplog.at_level(logging.WARNING):
            server._warn_if_no_sources()
        assert not caplog.records


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with a config file."""
    config = {
        'default_source': 'mylib',
        'sources': {
            'mylib': str(tmp_path / 'mylib'),
            'other': {
                'path': str(tmp_path / 'other'),
                'depends_on': ['mylib'],
            },
        },
        'docs_base': './docs',
        'defaults': {
            'model': 'gpt-5.2',
            'db_path': 'test.db',
        },
    }
    config_path = tmp_path / 'ariadne.yaml'
    config_path.write_text(yaml.dump(config))
    return tmp_path


class TestConfig:
    """Tests for the Config class."""

    def test_load_from_file(self, config_dir: Path) -> None:
        """Config should load from a YAML file."""
        cfg = Config(str(config_dir / 'ariadne.yaml'))
        assert cfg.default_source == 'mylib'
        assert cfg.model == 'gpt-5.2'

    def test_missing_file_uses_defaults(self, tmp_path: Path) -> None:
        """Missing config file should use defaults without error."""
        cfg = Config(str(tmp_path / 'nonexistent.yaml'))
        assert cfg.default_source is None
        assert cfg.model is not None  # Has a default

    def test_invalid_yaml_uses_defaults(self, tmp_path: Path) -> None:
        """Invalid YAML should use defaults without error."""
        bad = tmp_path / 'bad.yaml'
        bad.write_text('invalid: [yaml: broken')
        cfg = Config(str(bad))
        assert cfg.default_source is None

    def test_get_source_path(self, config_dir: Path) -> None:
        """Should resolve source names to paths."""
        cfg = Config(str(config_dir / 'ariadne.yaml'))
        path = cfg.get_source_path('mylib')
        assert path is not None
        assert str(path).endswith('mylib')

    def test_get_source_path_not_found(self, config_dir: Path) -> None:
        """Unknown source name should return None."""
        cfg = Config(str(config_dir / 'ariadne.yaml'))
        assert cfg.get_source_path('nonexistent') is None

    def test_sources_property(self, config_dir: Path) -> None:
        """Sources property should return configured source names."""
        cfg = Config(str(config_dir / 'ariadne.yaml'))
        assert 'mylib' in cfg.sources
        assert 'other' in cfg.sources

    def test_resolve_source(self, config_dir: Path) -> None:
        """resolve_source should handle names and paths."""
        cfg = Config(str(config_dir / 'ariadne.yaml'))
        # Resolve by name
        path = cfg.resolve_source('mylib')
        assert path is not None

        # Resolve None -> default source
        path2 = cfg.resolve_source(None)
        assert path2 is not None

    def test_config_dir_property(self, config_dir: Path) -> None:
        """config_dir should return the directory containing the config file."""
        cfg = Config(str(config_dir / 'ariadne.yaml'))
        assert cfg.config_dir == config_dir

    def test_db_path(self, config_dir: Path) -> None:
        """db_path should resolve relative to config directory."""
        cfg = Config(str(config_dir / 'ariadne.yaml'))
        db = cfg.db_path
        assert db.endswith('test.db')
