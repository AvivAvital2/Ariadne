"""Tests for ``SourceConfig.env_hints`` — Phase 2e.

Auto-discovery (Phase 2j) infers most of the indexer config from the
filesystem, but a few facts can't be auto-derived — most notably the
Python interpreter path for projects with non-standard env layouts
(conda envs, pipenv venvs with hashed names, custom virtualenvs). For
those, the user supplies a hint in ``ariadne.yaml`` and discovery
respects it.

``env_hints`` is a free-form dict so we can extend it later (e.g.,
``node_path``, ``jdk_path``) without schema migrations. Per design
decision #6, it's the *only* SCIP-related field users may need to
touch in ``ariadne.yaml`` — everything else is auto-derived.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Field defaults + construction
# ---------------------------------------------------------------------------


def test_source_config_has_env_hints_default_empty():
    from config import SourceConfig
    sc = SourceConfig(path='/tmp/x')
    assert sc.env_hints == {}


def test_source_config_accepts_env_hints_dict():
    from config import SourceConfig
    hints = {'python_path': '/path/to/.venv/bin/python'}
    sc = SourceConfig(path='/tmp/x', env_hints=hints)
    assert sc.env_hints == hints


def test_source_config_env_hints_independent_per_instance():
    """attrs.field(factory=dict) creates a fresh dict per instance.
    A mutation on one instance must not leak into another instance's
    default. Without ``factory``, all defaults share the same mutable
    {} which is a classic Python gotcha.
    """
    from config import SourceConfig
    a = SourceConfig(path='/tmp/a')
    b = SourceConfig(path='/tmp/b')
    a.env_hints['leaked'] = 'yes'
    assert 'leaked' not in b.env_hints


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def test_config_reads_env_hints_from_yaml(tmp_path):
    from config import Config

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  pyproject:\n'
        '    path: /path/to/biggerproject/be\n'
        '    env_hints:\n'
        '      python_path: /path/to/virtualenvs/pyproject/bin/python\n',
        encoding='utf-8',
    )
    cfg = Config(config_path=yaml_path)
    sc = cfg.get_source_config('pyproject')
    assert sc is not None
    assert sc.env_hints == {
        'python_path': '/path/to/virtualenvs/pyproject/bin/python',
    }


def test_config_yaml_without_env_hints_yields_empty_dict(tmp_path):
    """Backwards compat: a source declared without env_hints loads as
    an empty dict, not None or KeyError."""
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
    assert sc.env_hints == {}


def test_config_yaml_string_source_yields_empty_env_hints(tmp_path):
    """A source declared as a bare string (no dict) — preserved
    backwards-compat shape — has empty env_hints."""
    from config import Config

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  myapp: /tmp/myapp\n',
        encoding='utf-8',
    )
    cfg = Config(config_path=yaml_path)
    sc = cfg.get_source_config('myapp')
    assert sc is not None
    assert sc.env_hints == {}


# ---------------------------------------------------------------------------
# Phase 2e retro-audit — malformed env_hints handling
# ---------------------------------------------------------------------------


def test_malformed_env_hints_string_raises_config_error_at_load(tmp_path):
    """env_hints must be a dict (mapping) in ariadne.yaml. A string,
    list, or scalar should raise ConfigError at LOAD time so the user
    fixes their config immediately — not later when get_source_config
    surfaces some opaque Python TypeError.
    """
    from config import Config, ConfigError

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  myapp:\n'
        '    path: /tmp/x\n'
        '    env_hints: "not a dict"\n',
        encoding='utf-8',
    )
    import pytest
    with pytest.raises(ConfigError) as exc_info:
        Config(config_path=yaml_path)
    # Error message should name the offending source and field
    msg = str(exc_info.value)
    assert 'myapp' in msg
    assert 'env_hints' in msg


def test_malformed_env_hints_list_raises_config_error_at_load(tmp_path):
    """A list value for env_hints is also invalid — same fail-loud
    contract."""
    from config import Config, ConfigError

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  myapp:\n'
        '    path: /tmp/x\n'
        '    env_hints:\n'
        '      - python_path\n'
        '      - /tmp/python\n',
        encoding='utf-8',
    )
    import pytest
    with pytest.raises(ConfigError):
        Config(config_path=yaml_path)


def test_env_hints_with_non_string_values_rejected(tmp_path):
    """env_hints must be ``dict[str, str]``. A value of ``True`` or
    ``42`` should raise ConfigError, not silently coerce to string."""
    from config import Config, ConfigError

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  myapp:\n'
        '    path: /tmp/x\n'
        '    env_hints:\n'
        '      python_path: 42\n',
        encoding='utf-8',
    )
    import pytest
    with pytest.raises(ConfigError):
        Config(config_path=yaml_path)


# ---------------------------------------------------------------------------
# swagger_paths — Phase 7b (Wave 4, API surface tracking)
# ---------------------------------------------------------------------------


def test_source_config_has_swagger_paths_default_empty():
    """swagger_paths defaults to an empty tuple — sources without
    OpenAPI specs declared work as before. The field is the entry
    point for Phase 7c's Swagger ingestion."""
    from config import SourceConfig
    sc = SourceConfig(path='/tmp/x')
    assert sc.swagger_paths == ()


def test_source_config_accepts_swagger_paths_tuple():
    from config import SourceConfig
    sc = SourceConfig(
        path='/tmp/x',
        swagger_paths=('docs/openapi.yaml', 'api/swagger.json'),
    )
    assert sc.swagger_paths == ('docs/openapi.yaml', 'api/swagger.json')


def test_config_reads_swagger_paths_from_yaml(tmp_path):
    from config import Config

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  scalaproject:\n'
        '    path: /tmp/scalaproject\n'
        '    swagger_paths:\n'
        '      - api/openapi.yaml\n'
        '      - docs/legacy-swagger.json\n',
        encoding='utf-8',
    )
    cfg = Config(config_path=yaml_path)
    sc = cfg.get_source_config('scalaproject')
    assert sc is not None
    assert sc.swagger_paths == (
        'api/openapi.yaml', 'docs/legacy-swagger.json',
    )


def test_yaml_without_swagger_paths_yields_empty_tuple(tmp_path):
    """Backwards compat: a source declared without swagger_paths
    loads as an empty tuple, not None or KeyError."""
    from config import Config

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n  myapp:\n    path: /tmp/myapp\n',
        encoding='utf-8',
    )
    cfg = Config(config_path=yaml_path)
    sc = cfg.get_source_config('myapp')
    assert sc is not None
    assert sc.swagger_paths == ()


def test_malformed_swagger_paths_raises_config_error(tmp_path):
    """swagger_paths must be a list of strings. A scalar/dict raises
    ConfigError at load time — same fail-loud principle as env_hints."""
    from config import Config, ConfigError

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  myapp:\n'
        '    path: /tmp/x\n'
        '    swagger_paths: "not a list"\n',
        encoding='utf-8',
    )
    import pytest
    with pytest.raises(ConfigError):
        Config(config_path=yaml_path)

