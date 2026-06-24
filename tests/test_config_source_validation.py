"""Contract for SourceConfig key validation in ariadne.yaml loading.

When a user typos a source key (e.g., ``purh: /foo`` instead of
``path: /foo``), config loading must fail loudly with a ConfigError
naming the unknown key and suggesting the closest match via difflib.

``path`` itself is OPTIONAL — a source without one is serve-only (its
docs + dependency graph live in the database). The old "empty path →
``Path('').resolve()`` == cwd → walk thousands of files" footgun is
closed at RESOLUTION instead (``get_source_path`` / ``resolve_source``
return None), not by rejecting the config at load. A typo'd key is
still caught loudly here. Footgun: a source whose path was typo'd as
``purh`` — catalog-sync then attempted to walk thousands of files
instead of failing fast.

Tests follow the project's discipline: each "should fail" case is
paired with a "should load" baseline so a stub validator that always
raises (or never raises) fails one half.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config import Config, ConfigError


@pytest.fixture
def yaml_factory(tmp_path: Path):
    """Write a temp ariadne.yaml and return its Path."""
    def _write(body: str) -> Path:
        p = tmp_path / 'ariadne.yaml'
        p.write_text(body, encoding='utf-8')
        return p
    return _write


# ---------------------------------------------------------------------------
# Required ``path`` field
# ---------------------------------------------------------------------------


class TestPathOptionalServeOnly:
    """``path`` is optional: a dict source without one is a SERVE-ONLY source
    (docs + dependency graph live in the DB). It loads cleanly, and the old
    cwd-walk footgun is closed at RESOLUTION (get_source_path / resolve_source
    return None) rather than by rejecting the config at load. Typo'd keys still
    fail loud — see TestUnknownKeys.
    """

    def test_dict_source_missing_path_loads_as_serve_only(
        self, yaml_factory,
    ) -> None:
        """Dict-form source without a ``path`` key loads as serve-only:
        found (not None) but path-less, and resolution yields None so a
        build fails closed instead of walking cwd."""
        yaml = yaml_factory(
            'sources:\n'
            '  src1:\n'
            '    branches: [main]\n',
        )
        cfg = Config(config_path=yaml)
        sc = cfg.get_source_config('src1')
        assert sc is not None and sc.path is None
        assert cfg.get_source_path('src1') is None
        assert cfg.resolve_source('src1') is None

    def test_dict_source_empty_path_resolves_to_none(
        self, yaml_factory,
    ) -> None:
        """``path: ""`` is treated as path-less, same as omitting it — the
        ``Path('').resolve()`` == cwd footgun is closed at resolution."""
        yaml = yaml_factory(
            'sources:\n'
            '  src1:\n'
            '    path: ""\n',
        )
        cfg = Config(config_path=yaml)
        assert cfg.get_source_path('src1') is None
        assert cfg.resolve_source('src1') is None

    def test_string_form_source_loads_cleanly(
        self, yaml_factory, tmp_path: Path,
    ) -> None:
        """Paired baseline: legacy ``sources: name: /path`` form
        bypasses dict validation and loads cleanly. Bites a too-broad
        validator that chokes on the simple form."""
        yaml = yaml_factory(
            'sources:\n'
            f'  src1: {tmp_path}\n',
        )
        cfg = Config(config_path=yaml)
        sc = cfg.get_source_config('src1')
        assert sc is not None
        assert sc.path == str(tmp_path)

    def test_dict_source_with_path_loads_cleanly(
        self, yaml_factory, tmp_path: Path,
    ) -> None:
        """Paired baseline: a proper dict source with a path loads
        without error. Confirms the validator only fires on the
        actual misconfiguration."""
        yaml = yaml_factory(
            'sources:\n'
            '  src1:\n'
            f'    path: {tmp_path}\n',
        )
        cfg = Config(config_path=yaml)
        sc = cfg.get_source_config('src1')
        assert sc is not None
        assert sc.path == str(tmp_path)


# ---------------------------------------------------------------------------
# Unknown keys + typo suggestion
# ---------------------------------------------------------------------------


class TestUnknownKeys:
    def test_unknown_key_typo_suggests_close_match(
        self, yaml_factory, tmp_path: Path,
    ) -> None:
        """The canonical case: ``purh: /foo`` is a typo of ``path``.
        Loader must raise with both ``'purh'`` and ``'path'`` in
        the message so the user sees the exact mistake and the
        intended key.

        This was the bug surfaced 2026-05-09. Without this guard,
        the loader silently dropped 'purh' as an unknown extra and
        proceeded with empty path."""
        yaml = yaml_factory(
            'sources:\n'
            '  src1:\n'
            f'    purh: {tmp_path}\n',
        )
        with pytest.raises(ConfigError) as exc:
            Config(config_path=yaml)
        msg = str(exc.value)
        assert 'src1' in msg
        assert 'purh' in msg
        assert 'path' in msg

    def test_unknown_key_far_from_any_match_no_false_suggestion(
        self, yaml_factory, tmp_path: Path,
    ) -> None:
        """Paired with the typo test: a key with no close match
        still raises (it's still unknown), but the message doesn't
        fabricate a suggestion. Bites a too-eager difflib that
        returns spurious matches for nonsense input."""
        yaml = yaml_factory(
            'sources:\n'
            '  src1:\n'
            f'    path: {tmp_path}\n'
            '    qzxqzx: 1\n',
        )
        with pytest.raises(ConfigError) as exc:
            Config(config_path=yaml)
        msg = str(exc.value)
        assert 'qzxqzx' in msg
        # No "did you mean" phrasing when difflib returns nothing —
        # otherwise the message lies about the user's intent.
        assert 'did you mean' not in msg.lower()

    def test_all_known_keys_load_cleanly(
        self, yaml_factory, tmp_path: Path,
    ) -> None:
        """Paired baseline: a config using every recognized
        SourceConfig key loads without error. Pins the recognized
        set so a future field added to SourceConfig that isn't
        also added to the validator's allowlist fails this test."""
        yaml = yaml_factory(
            'sources:\n'
            '  src1:\n'
            f'    path: {tmp_path}\n'
            '    depends_on: []\n'
            '    branches: ["main"]\n'
            '    ref: main\n'
            '    exclude: []\n'
            '    exclude_dirs: []\n'
            '    exempt_dirs: []\n'
            '    env_hints: {}\n'
            '    swagger_paths: []\n',
        )
        cfg = Config(config_path=yaml)
        sc = cfg.get_source_config('src1')
        assert sc is not None

    def test_scip_and_index_kinds_recognized(
        self, yaml_factory, tmp_path: Path,
    ) -> None:
        """SCIP-related keys (``scip``, ``index_kinds``) are
        consumed by ``get_source_scip_config`` rather than
        SourceConfig itself, but they're still legal at the source
        level. The validator must allow them."""
        yaml = yaml_factory(
            'sources:\n'
            '  src1:\n'
            f'    path: {tmp_path}\n'
            '    scip:\n'
            '      artifact_path: index.scip\n'
            '      max_staleness_days: 7\n'
            '    index_kinds:\n'
            '      scala: scip\n',
        )
        cfg = Config(config_path=yaml)
        sc = cfg.get_source_config('src1')
        assert sc is not None


# ---------------------------------------------------------------------------
# Combination case: typo'd path AND no other valid path
# ---------------------------------------------------------------------------


class TestCombinedFailure:
    def test_typo_only_for_path_surfaces_typo_first(
        self, yaml_factory, tmp_path: Path,
    ) -> None:
        """The canonical user case: ``purh: /tmp/foo`` and no
        ``path:``. The unknown-key error must fire first (and
        surface the typo) — that's the most actionable message.
        After the user fixes the typo, the next load succeeds.

        If the missing-path check fired first, the user would see
        'missing required field path' and be confused — they have
        purh in the file. The unknown-key message points at the
        actual mistake."""
        yaml = yaml_factory(
            'sources:\n'
            '  src1:\n'
            f'    purh: {tmp_path}\n',
        )
        with pytest.raises(ConfigError) as exc:
            Config(config_path=yaml)
        msg = str(exc.value)
        # Surfaces the typo, not a generic missing-field error
        assert 'purh' in msg
        assert 'path' in msg
