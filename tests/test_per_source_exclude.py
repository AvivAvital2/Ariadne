"""Tests for per-source exclude patterns.

Sources can declare ``exclude:`` glob patterns in ``ariadne.yaml``. The
patterns thread through to file discovery (``find_catalog_files``,
``find_python_files``) so sensitive files never reach the LLM, the
embedding service, or the docs DB.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Config layer: SourceConfig.exclude + Config.get_source_config parsing
# ---------------------------------------------------------------------------


def test_source_config_has_exclude_defaulting_empty():
    from config import SourceConfig

    sc = SourceConfig(path='/tmp/x')
    assert sc.exclude == ()


def test_source_config_accepts_exclude_tuple():
    from config import SourceConfig

    sc = SourceConfig(
        path='/tmp/x',
        exclude=('**/.env*', '**/secrets/**'),
    )
    assert sc.exclude == ('**/.env*', '**/secrets/**')


def test_config_reads_exclude_from_yaml(tmp_path):
    """``Config.get_source_config(name).exclude`` returns the patterns
    declared under ``sources.<name>.exclude:`` in ariadne.yaml.
    """
    from config import Config

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        "sources:\n"
        "  myapp:\n"
        "    path: /tmp/myapp\n"
        "    exclude:\n"
        "      - '**/.env*'\n"
        "      - '**/secrets/**'\n"
        "      - '**/credentials.json'\n",
        encoding='utf-8',
    )
    cfg = Config(config_path=yaml_path)
    sc = cfg.get_source_config('myapp')
    assert sc is not None
    assert sc.exclude == (
        '**/.env*', '**/secrets/**', '**/credentials.json',
    )


def test_config_exclude_absent_yields_empty_tuple(tmp_path):
    """Sources without an ``exclude:`` block produce an empty tuple."""
    from config import Config

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  myapp:\n'
        '    path: /tmp/myapp\n',
        encoding='utf-8',
    )
    cfg = Config(config_path=yaml_path)
    sc = cfg.get_source_config('myapp')
    assert sc is not None
    assert sc.exclude == ()


# ---------------------------------------------------------------------------
# Orchestrator layer: exclude_patterns is honored during file discovery
# ---------------------------------------------------------------------------


def test_orchestrator_config_has_exclude_patterns_default_empty():
    from docgen.orchestrator import OrchestratorConfig

    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
    )
    assert cfg.exclude_patterns == ()


def test_orchestrator_config_accepts_exclude_patterns():
    from docgen.orchestrator import OrchestratorConfig

    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
        exclude_patterns=('**/.env*',),
    )
    assert cfg.exclude_patterns == ('**/.env*',)


# ---------------------------------------------------------------------------
# Discovery layer: find_catalog_files honors caller patterns + new defaults
# ---------------------------------------------------------------------------


def test_find_catalog_files_excludes_user_provided_pattern(tmp_path):
    """Patterns passed via ``exclude_patterns`` filter matching files."""
    from docgen.staleness import find_catalog_files

    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'main.py').write_text('x = 1', encoding='utf-8')
    (tmp_path / 'src' / 'secrets.py').write_text(
        "API_KEY = 'sk-real-thing'", encoding='utf-8',
    )

    files = find_catalog_files(
        tmp_path, exclude_patterns=('**/secrets.py',),
    )
    rel = sorted(str(f.relative_to(tmp_path)) for f in files)
    assert 'src/main.py' in rel
    assert 'src/secrets.py' not in rel
