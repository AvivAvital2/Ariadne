"""Tests for Ariadne configuration module."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config import Config


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
