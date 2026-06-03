"""Tests for SCIP config + error hierarchy (SCIP plan, Phase A.3).

Pure-data tests — no protobuf, no scip-java. Pins:
- ``SourceScipConfig`` shape and defaults.
- The ``ScipError`` hierarchy with isinstance relationships.
- ``Config.get_source_scip_config(name)`` reads ``ariadne.yaml`` correctly.

Tests must FAIL until ``docgen.scip_config`` exists and ``Config`` gains
``get_source_scip_config``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# SourceScipConfig
# ---------------------------------------------------------------------------


class TestSourceScipConfig:
    def test_required_fields(self) -> None:
        from docgen.scip_config import SourceScipConfig

        c = SourceScipConfig(
            repo='scalaproject',
            artifact_path=Path('/tmp/index.scip'),
        )
        assert c.repo == 'scalaproject'
        assert c.artifact_path == Path('/tmp/index.scip')

    def test_defaults(self) -> None:
        from docgen.scip_config import SourceScipConfig

        c = SourceScipConfig(
            repo='scalaproject',
            artifact_path=Path('/tmp/index.scip'),
        )
        # Per plan §4.4 — sane defaults so callers can omit the obvious cases.
        assert c.max_staleness_days == 7
        assert c.allow_degraded is False
        assert c.index_kinds == {}

    def test_index_kinds_is_dict(self) -> None:
        from docgen.scip_config import SourceScipConfig

        c = SourceScipConfig(
            repo='scalaproject',
            artifact_path=Path('/tmp/index.scip'),
            index_kinds={'scala': 'scip', 'java': 'scip'},
        )
        assert c.index_kinds['scala'] == 'scip'
        assert c.index_kinds['java'] == 'scip'

    def test_is_frozen(self) -> None:
        from docgen.scip_config import SourceScipConfig

        c = SourceScipConfig(
            repo='scalaproject',
            artifact_path=Path('/tmp/index.scip'),
        )
        with pytest.raises(Exception):  # noqa: B017
            c.repo = 'other'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_scip_error_carries_repo_and_reason(self) -> None:
        from docgen.scip_config import ScipError

        err = ScipError(repo='scalaproject', reason='index_missing')
        assert err.repo == 'scalaproject'
        assert err.reason == 'index_missing'

    def test_unavailable_is_scip_error(self) -> None:
        from docgen.scip_config import ScipError, ScipUnavailableError

        err = ScipUnavailableError(repo='scalaproject', reason='index_missing')
        assert isinstance(err, ScipError)
        assert isinstance(err, Exception)

    def test_too_stale_is_scip_error(self) -> None:
        from docgen.scip_config import ScipError, ScipTooStaleError

        err = ScipTooStaleError(
            repo='scalaproject',
            reason='index_too_stale',
            last_good_age_days=14,
        )
        assert isinstance(err, ScipError)
        assert err.last_good_age_days == 14

    def test_corrupt_is_scip_error(self) -> None:
        from docgen.scip_config import ScipCorruptError, ScipError

        err = ScipCorruptError(repo='scalaproject', reason='index_corrupt')
        assert isinstance(err, ScipError)

    def test_subclasses_are_distinguishable(self) -> None:
        """The whole point of subclassing is so callers can dispatch on
        specific failure modes — pin that distinct types are distinct.
        """
        from docgen.scip_config import (
            ScipCorruptError,
            ScipTooStaleError,
            ScipUnavailableError,
        )

        u = ScipUnavailableError(repo='r', reason='index_missing')
        s = ScipTooStaleError(repo='r', reason='index_too_stale')
        c = ScipCorruptError(repo='r', reason='index_corrupt')

        assert not isinstance(u, ScipTooStaleError)
        assert not isinstance(u, ScipCorruptError)
        assert not isinstance(s, ScipUnavailableError)
        assert not isinstance(s, ScipCorruptError)
        assert not isinstance(c, ScipUnavailableError)
        assert not isinstance(c, ScipTooStaleError)


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body, encoding='utf-8')
    return p


class TestGetSourceScipConfig:
    def test_returns_none_when_no_scip_block(self, tmp_path: Path) -> None:
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  scalaproject:
    path: /tmp/scalaproject
''')
        cfg = Config(cfg_path)
        assert cfg.get_source_scip_config('scalaproject') is None

    def test_returns_none_for_unknown_source(self, tmp_path: Path) -> None:
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  scalaproject:
    path: /tmp/scalaproject
    scip:
      artifact_path: /tmp/idx.scip
''')
        cfg = Config(cfg_path)
        assert cfg.get_source_scip_config('nonexistent') is None

    def test_returns_scip_config_when_configured(self, tmp_path: Path) -> None:
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  scalaproject:
    path: /tmp/scalaproject
    index_kinds:
      scala: scip
      java: scip
    scip:
      artifact_path: /tmp/idx.scip
      max_staleness_days: 14
''')
        cfg = Config(cfg_path)
        scip_cfg = cfg.get_source_scip_config('scalaproject')

        assert scip_cfg is not None
        assert scip_cfg.repo == 'scalaproject'
        assert scip_cfg.artifact_path == Path('/tmp/idx.scip')
        assert scip_cfg.max_staleness_days == 14
        assert scip_cfg.index_kinds == {'scala': 'scip', 'java': 'scip'}

    def test_default_max_staleness_when_unspecified(
        self, tmp_path: Path,
    ) -> None:
        """If max_staleness_days is omitted, default to 7 — matches the
        attrs default on SourceScipConfig itself.
        """
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  scalaproject:
    path: /tmp/scalaproject
    scip:
      artifact_path: /tmp/idx.scip
''')
        cfg = Config(cfg_path)
        scip_cfg = cfg.get_source_scip_config('scalaproject')
        assert scip_cfg is not None
        assert scip_cfg.max_staleness_days == 7

    def test_allow_degraded_starts_false(self, tmp_path: Path) -> None:
        """Even if YAML somehow has allow_degraded=true (which would be
        unusual), per-call CLI flag is the gate. Defaults must be safe.
        """
        from config import Config

        cfg_path = _write_config(tmp_path, '''\
sources:
  scalaproject:
    path: /tmp/scalaproject
    scip:
      artifact_path: /tmp/idx.scip
''')
        cfg = Config(cfg_path)
        scip_cfg = cfg.get_source_scip_config('scalaproject')
        assert scip_cfg is not None
        assert scip_cfg.allow_degraded is False
