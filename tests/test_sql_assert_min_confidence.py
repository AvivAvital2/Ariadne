"""``sql_assert_min_confidence`` (design §3a) — the SQL data-model read-boundary
floor as a config key (default ``resolved``), settable top-level or under
``defaults:``, fail-loud on an invalid value."""
from __future__ import annotations

from pathlib import Path

import pytest

from config import Config, ConfigError


def _cfg(tmp_path, body: str) -> Config:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body, encoding='utf-8')
    return Config(config_path=p)


def test_defaults_to_resolved(tmp_path) -> None:
    assert _cfg(tmp_path, 'sources: {}\n').sql_assert_min_confidence == 'resolved'


def test_top_level_override(tmp_path) -> None:
    cfg = _cfg(tmp_path, 'sql_assert_min_confidence: exact\nsources: {}\n')
    assert cfg.sql_assert_min_confidence == 'exact'


def test_under_defaults_block(tmp_path) -> None:
    cfg = _cfg(
        tmp_path, 'defaults:\n  sql_assert_min_confidence: derived\nsources: {}\n')
    assert cfg.sql_assert_min_confidence == 'derived'


def test_invalid_value_fails_loud(tmp_path) -> None:
    with pytest.raises(ConfigError):
        _cfg(tmp_path, 'sql_assert_min_confidence: bogus\nsources: {}\n')


def test_floor_rank_resolves_from_config(tmp_path, monkeypatch) -> None:
    """``_floor_rank(None)`` reads the configured floor (the one place the key
    is consulted); an explicit value wins over config."""
    import config as config_module
    from docgen.scip_cross_source import _CONFIDENCE_RANK, _floor_rank
    monkeypatch.setattr(
        config_module, 'get_config',
        lambda: _cfg(tmp_path, 'sql_assert_min_confidence: exact\nsources: {}\n'))
    assert _floor_rank(None) == _CONFIDENCE_RANK['exact']      # unset → config
    assert _floor_rank('derived') == _CONFIDENCE_RANK['derived']  # explicit wins


def test_floor_rank_falls_back_when_config_lacks_key(monkeypatch) -> None:
    """A config object that doesn't define the key (e.g. a partial test double)
    resolves to the built-in default floor, not an AttributeError — an unset key
    means the same as the default."""
    import config as config_module
    from types import SimpleNamespace
    from docgen.scip_cross_source import (
        _CONFIDENCE_RANK, _floor_rank, DEFAULT_ASSERT_MIN_CONFIDENCE)
    monkeypatch.setattr(config_module, 'get_config', lambda: SimpleNamespace())
    assert _floor_rank(None) == _CONFIDENCE_RANK[DEFAULT_ASSERT_MIN_CONFIDENCE]


def test_query_view_default_floor_follows_config(tmp_path, monkeypatch) -> None:
    """A query view's default floor (min_confidence unset) follows the config
    key: with the floor at ``exact``, a ``resolved`` access is held back."""
    import config as config_module
    from docgen.sql_query_views import data_access_for
    from library import Library

    library = Library(tmp_path / 'db.sqlite')
    try:
        with library._conn_provider.acquire() as conn:
            conn.executemany(
                'INSERT INTO data_access (source_name, consumer_symbol_id, '
                'schema_symbol_id, role, witness, confidence) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                [('s', 'c_exact', 'tbl#c', 'filter', 'rawsql', 'exact'),
                 ('s', 'c_resolved', 'tbl#c', 'filter', 'rawsql', 'resolved')])
            conn.commit()
            monkeypatch.setattr(
                config_module, 'get_config',
                lambda: _cfg(
                    tmp_path, 'sql_assert_min_confidence: exact\nsources: {}\n'))
            # min_confidence unset → resolves to the config floor 'exact'
            assert data_access_for(conn, 'tbl#c')['reads'] == ['c_exact']
    finally:
        library.close()


def test_mcp_data_tool_default_follows_config(tmp_path, monkeypatch) -> None:
    """The ``ariadne_data`` MCP tool, with ``min_confidence`` unset, honors the
    configured floor: at ``exact`` a ``resolved`` access is held back."""
    import config as config_module
    from ariadne_mcp.server_knowledge import ariadne_data
    from library import Library

    cfg = _cfg(tmp_path, 'sql_assert_min_confidence: exact\nsources: {}\n')
    library = Library(Path(cfg.db_path))
    try:
        with library._conn_provider.acquire() as conn:
            conn.executemany(
                'INSERT INTO data_access (source_name, consumer_symbol_id, '
                'schema_symbol_id, role, witness, confidence) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                [('s', 'c1', 't#exact', 'filter', 'rawsql', 'exact'),
                 ('s', 'c1', 't#resolved', 'filter', 'rawsql', 'resolved')])
            conn.commit()
    finally:
        library.close()
    monkeypatch.setattr(config_module, 'get_config', lambda: cfg)
    touched = ariadne_data(symbol='c1')['touches']  # min_confidence unset → config
    assert {t['schema_id'] for t in touched} == {'t#exact'}
