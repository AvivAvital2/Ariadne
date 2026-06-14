"""Opt-in staleness exemption via the per-source ``ignore_staleness`` field.

A source can opt out of staleness checks entirely (``ignore_staleness: true``)
or for specific files (a glob list). The flag suppresses the
content-changed -> stale signal; never-documented files still surface as
coverage gaps. ``true`` also exempts the source's SCIP index from the
mtime-based age check; a glob list does not (the index is source-level).
"""
from __future__ import annotations

import pytest

from cli.dry_run import _apply_explorer_staleness
from config import Config, ConfigError
from docgen.staleness import StalenessTracker


def _write_cfg(tmp_path, sources_body: str) -> Config:
    src = tmp_path / "src1"
    src.mkdir(exist_ok=True)
    cfg_path = tmp_path / "ariadne.yaml"
    cfg_path.write_text(sources_body.replace("__SRC1__", str(src)))
    return Config(cfg_path)


def test_true_exempts_whole_source(tmp_path):
    cfg = _write_cfg(
        tmp_path,
        "sources:\n"
        "  src1:\n"
        "    path: __SRC1__\n"
        "    ignore_staleness: true\n",
    )
    assert cfg.get_source_config("src1").ignore_staleness is True
    assert cfg.source_staleness_exempt("src1") is True
    # A whole-source exemption covers every path.
    assert cfg.path_staleness_exempt("src1", "pkg/anything.py") is True


def test_glob_list_exempts_matching_files_only(tmp_path):
    cfg = _write_cfg(
        tmp_path,
        "sources:\n"
        "  src1:\n"
        "    path: __SRC1__\n"
        "    ignore_staleness:\n"
        "      - 'vendor/**'\n"
        "      - 'legacy/*.py'\n",
    )
    sc = cfg.get_source_config("src1")
    assert sc.ignore_staleness == ("vendor/**", "legacy/*.py")
    # A file-level list does NOT make the whole source (or its SCIP index) exempt.
    assert cfg.source_staleness_exempt("src1") is False
    assert cfg.path_staleness_exempt("src1", "vendor/lib/deep/x.py") is True
    assert cfg.path_staleness_exempt("src1", "legacy/old.py") is True
    assert cfg.path_staleness_exempt("src1", "pkg/main.py") is False


def test_rejects_scalar_type(tmp_path):
    with pytest.raises(ConfigError):
        _write_cfg(
            tmp_path,
            "sources:\n  src1:\n    path: __SRC1__\n    ignore_staleness: 7\n",
        )


def test_rejects_non_string_glob(tmp_path):
    with pytest.raises(ConfigError):
        _write_cfg(
            tmp_path,
            "sources:\n  src1:\n    path: __SRC1__\n    ignore_staleness:\n      - 5\n",
        )


def test_defaults_off(tmp_path):
    cfg = _write_cfg(
        tmp_path,
        "sources:\n  src1:\n    path: __SRC1__\n",
    )
    assert cfg.get_source_config("src1").ignore_staleness is False
    assert cfg.source_staleness_exempt("src1") is False
    assert cfg.path_staleness_exempt("src1", "pkg/main.py") is False
    # An unknown source is never exempt (no config to read).
    assert cfg.path_staleness_exempt("ghost", "pkg/main.py") is False


# --- doc-staleness suppression (StalenessTracker honors an is_exempt predicate) ---

def _tracker(tmp_path):
    src = tmp_path / "src1"
    src.mkdir(exist_ok=True)
    return StalenessTracker(db_path=tmp_path / "staleness.db"), src


def test_exempt_predicate_suppresses_changed_file(tmp_path):
    tracker, src = _tracker(tmp_path)
    f = src / "service.py"
    f.write_text("value = 1\n")
    tracker.record_documentation(f, ["doc1"], base_path=src)
    f.write_text("value = 2\n")  # content changed -> normally stale

    assert tracker.is_stale(f, base_path=src) is True
    assert tracker.is_stale(f, base_path=src, is_exempt=lambda rel: True) is False
    assert tracker.get_stale_files([f], base_path=src) == [f]
    assert (
        tracker.get_stale_files(
            [f], base_path=src, is_exempt=lambda rel: rel == "service.py"
        )
        == []
    )
    tracker.close()


def test_exempt_does_not_hide_never_documented_file(tmp_path):
    tracker, src = _tracker(tmp_path)
    f = src / "new.py"
    f.write_text("y = 1\n")  # never recorded -> coverage gap, not staleness

    assert tracker.is_stale(f, base_path=src, is_exempt=lambda rel: True) is True
    assert tracker.get_stale_files([f], base_path=src, is_exempt=lambda rel: True) == [f]
    tracker.close()


# --- SCIP index-age suppression (max_staleness_days=None skips the mtime gate) ---

def test_scip_load_none_skips_age_check(tmp_path):
    import os
    import time

    from docgen.scip_config import ScipCorruptError, ScipTooStaleError
    from docgen.scip_extractor import ScipIndex

    idx = tmp_path / "index.scip"
    idx.write_bytes(b"not-real-protobuf")
    old = time.time() - 100 * 86400  # 100 days old
    os.utime(idx, (old, old))

    # Finite threshold: an old index is rejected as too stale.
    with pytest.raises(ScipTooStaleError):
        ScipIndex.load(idx, repo="src1", max_staleness_days=7)

    # None (exempt source): the age gate is skipped, so the load gets PAST
    # staleness and fails at parse instead (ScipCorruptError) — proving the
    # mtime check no longer fires.
    with pytest.raises(ScipCorruptError):
        ScipIndex.load(idx, repo="src1", max_staleness_days=None)


# --- shared matcher reused by config + orchestrator (DRY, single source of truth) ---

def test_ignore_staleness_matches_helper():
    from config import ignore_staleness_matches

    assert ignore_staleness_matches(True, "any/file.py") is True
    assert ignore_staleness_matches(False, "any/file.py") is False
    assert ignore_staleness_matches((), "any/file.py") is False
    assert ignore_staleness_matches(("vendor/**",), "vendor/deep/a.py") is True
    assert ignore_staleness_matches(("legacy/*.py",), "legacy/old.py") is True
    assert ignore_staleness_matches(("vendor/**",), "pkg/main.py") is False


# --- orchestrator wiring: OrchestratorConfig field + exemption predicate ---

def _orch_config(tmp_path, **kw):
    from docgen.orchestrator import OrchestratorConfig

    return OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / "docs.db",
        staleness_db_path=tmp_path / "stale.db",
        **kw,
    )


def test_orchestrator_config_ignore_staleness_default_and_set(tmp_path):
    assert _orch_config(tmp_path).ignore_staleness is False
    assert _orch_config(tmp_path, ignore_staleness=True).ignore_staleness is True


def test_orchestrator_staleness_exempt_predicate(tmp_path):
    from docgen.orchestrator import DocGenOrchestrator

    def predicate(ignore):
        return DocGenOrchestrator(
            _orch_config(tmp_path, ignore_staleness=ignore)
        )._staleness_exempt()

    assert predicate(False) is None  # default -> legacy path, no predicate
    assert predicate(True)("anything/x.py") is True
    glob_pred = predicate(("vendor/**",))
    assert glob_pred("vendor/a/b.py") is True
    assert glob_pred("pkg/main.py") is False


@pytest.mark.asyncio
async def test_check_staleness_honors_ignore_staleness(tmp_path):
    """End-to-end design intent: a documented-then-changed file is reported
    stale by default, but NOT when the source is staleness-exempt."""
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    async def stale_count(ignore: bool, tag: str) -> int:
        src = tmp_path / tag
        src.mkdir()
        f = src / "service.py"
        f.write_text("value = 1\n")
        config = OrchestratorConfig(
            source_path=src,
            db_path=tmp_path / f"{tag}.db",
            staleness_db_path=tmp_path / f"{tag}-stale.db",
            dry_run=True,
            ignore_staleness=ignore,
        )
        async with DocGenOrchestrator(config) as orch:
            orch._staleness.record_documentation(f, ["doc1"], base_path=src)
            f.write_text("value = 2\n")  # source content changed
            return (await orch.check_staleness())["stale_files"]

    assert await stale_count(True, "exempt") == 0
    assert await stale_count(False, "tracked") == 1


# --- SCIP index-age origin: effective max-staleness-days collapses to None when exempt ---

def test_effective_scip_staleness_days(tmp_path):
    # Exempt source -> None disables the age gate.
    exempt = _write_cfg(
        tmp_path,
        "sources:\n  src1:\n    path: __SRC1__\n    ignore_staleness: true\n",
    )
    assert exempt.effective_scip_staleness_days("src1") is None

    # Not exempt, no scip block -> legacy default of 7.
    plain = _write_cfg(tmp_path, "sources:\n  src1:\n    path: __SRC1__\n")
    assert plain.effective_scip_staleness_days("src1") == 7

    # Not exempt, with a scip block -> that block's max_staleness_days.
    scoped = _write_cfg(
        tmp_path,
        "sources:\n"
        "  src1:\n"
        "    path: __SRC1__\n"
        "    index_kinds:\n"
        "      python: scip\n"
        "    scip:\n"
        "      artifact_path: /tmp/x.scip\n"
        "      max_staleness_days: 30\n",
    )
    assert scoped.effective_scip_staleness_days("src1") == 30


def test_source_ignore_staleness_raw_value(tmp_path):
    """The raw value the CLI hands to OrchestratorConfig: the glob tuple / True,
    or False for an unknown/None source."""
    cfg = _write_cfg(
        tmp_path,
        "sources:\n"
        "  src1:\n"
        "    path: __SRC1__\n"
        "    ignore_staleness:\n"
        "      - 'vendor/**'\n",
    )
    assert cfg.source_ignore_staleness("src1") == ("vendor/**",)
    assert cfg.source_ignore_staleness("unknown") is False
    assert cfg.source_ignore_staleness(None) is False


# --- CLI: `source add --ignore-staleness` persists the flag to ariadne.yaml ---

def test_source_add_ignore_staleness_flag(monkeypatch, tmp_path):
    cfg_path = tmp_path / "ariadne.yaml"
    cfg_path.write_text("sources: {}\n")  # isolate from the repo's own ariadne.yaml
    monkeypatch.setenv("ARIADNE_CONFIG", str(cfg_path))
    monkeypatch.chdir(tmp_path)
    import config as config_module

    monkeypatch.setattr(config_module, "_global_config", None, raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    from cli.integration import HANDLERS
    from cli.main import create_parser
    from config import Config

    def run(argv):
        args = create_parser().parse_args(argv)
        return HANDLERS[args.command](args)

    src = tmp_path / "repo"
    src.mkdir()

    # Without the flag the source defaults to staleness-tracked.
    assert run(["source", "add", "plain", "--path", str(src)]) == 0
    assert Config(config_path=cfg_path).get_source_config("plain").ignore_staleness is False

    # With the flag the source is persisted as exempt.
    assert run(["source", "add", "rare", "--path", str(src), "--ignore-staleness"]) == 0
    assert Config(config_path=cfg_path).get_source_config("rare").ignore_staleness is True

    # Idempotent update of another field leaves the exemption intact.
    assert run(["source", "add", "rare", "--ref", "main"]) == 0
    assert Config(config_path=cfg_path).get_source_config("rare").ignore_staleness is True


# --- Onboarding: the explorer's Apply-time staleness modal persists the choice ---
# The CLI y/N prompt was replaced by a pop-up in the file browser (the browser
# owns ariadne.yaml); ``_apply_explorer_staleness`` persists its result.

def test_apply_explorer_staleness_persists_on_yes(tmp_path):
    cfg = _write_cfg(tmp_path, "sources:\n  src1:\n    path: __SRC1__\n")

    # Modal accepted + not already exempt → persists ignore_staleness: true.
    assert _apply_explorer_staleness(
        cfg, "src1", chosen=True, currently_exempt=False) is True
    assert Config(cfg.config_path).get_source_config(
        "src1").ignore_staleness is True


def test_apply_explorer_staleness_noop_when_declined_or_already_exempt(tmp_path):
    cfg = _write_cfg(tmp_path, "sources:\n  src1:\n    path: __SRC1__\n")

    # Declined in the modal → nothing written.
    assert _apply_explorer_staleness(
        cfg, "src1", chosen=False, currently_exempt=False) is False
    assert Config(cfg.config_path).get_source_config(
        "src1").ignore_staleness in (None, False)

    # Already exempt → no redundant write.
    assert _apply_explorer_staleness(
        cfg, "src1", chosen=True, currently_exempt=True) is False
