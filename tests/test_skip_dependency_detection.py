"""Opt-in switch to skip cross-source dependency detection per source.

Dependency detection scans the onboarded repo's Python files for imports
that reference the project's OTHER configured sources, to infer a hidden
``depends_on`` relationship. On a large repo that full-tree walk is pure
wait, so a source can opt out with ``skip_dependency_detection: true``.

When opted out:
- the generate phase's import scan (the expensive ``rglob('*.py')`` walk)
  is never run for that source;
- the interactive onboard dependency offer is suppressed (pinned in
  ``test_onboard_dependency_prompt.py``).

The switch is also settable via ``ariadne source add --skip-dependency-detection``
(pinned in ``test_source_cli.py``).
"""
from __future__ import annotations

from pathlib import Path

import docgen.dependency as dependency_module
from cli.generate_cost import _check_and_prompt_dependencies
from config import Config


def _write_cfg(tmp_path: Path, sources_body: str) -> Config:
    src = tmp_path / "src1"
    src.mkdir(exist_ok=True)
    cfg_path = tmp_path / "ariadne.yaml"
    cfg_path.write_text(sources_body.replace("__SRC1__", str(src)))
    return Config(cfg_path)


def test_field_defaults_false_and_parses_true(tmp_path):
    cfg = _write_cfg(
        tmp_path,
        "sources:\n  src1:\n    path: __SRC1__\n",
    )
    assert cfg.get_source_config("src1").skip_dependency_detection is False
    assert cfg.source_skip_dependency_detection("src1") is False

    cfg = _write_cfg(
        tmp_path,
        "sources:\n  src1:\n    path: __SRC1__\n"
        "    skip_dependency_detection: true\n",
    )
    assert cfg.get_source_config("src1").skip_dependency_detection is True
    assert cfg.source_skip_dependency_detection("src1") is True


def _two_source_cfg(tmp_path: Path, *, skip: bool) -> Config:
    """``svc_main`` plus one eligible dependency target ``lib_a``; only the
    presence of ``skip_dependency_detection`` on ``svc_main`` differs."""
    for d in ("svc_main", "lib_a"):
        (tmp_path / d).mkdir()
    line = "    skip_dependency_detection: true\n" if skip else ""
    cfg_path = tmp_path / "ariadne.yaml"
    cfg_path.write_text(
        "sources:\n"
        "  svc_main:\n"
        f"    path: {tmp_path / 'svc_main'}\n"
        f"{line}"
        "  lib_a:\n"
        f"    path: {tmp_path / 'lib_a'}\n"
    )
    return Config(cfg_path)


def test_generate_scan_runs_when_not_opted_out(tmp_path, monkeypatch):
    """Baseline: with the switch off, the import scan is invoked."""
    calls: list = []
    monkeypatch.setattr(
        dependency_module,
        "detect_dependencies",
        lambda *a, **k: calls.append((a, k)) or [],
    )
    cfg = _two_source_cfg(tmp_path, skip=False)

    _check_and_prompt_dependencies("svc_main", tmp_path / "svc_main", cfg)

    assert len(calls) == 1, "detection must run when not opted out"


def test_generate_scan_skipped_when_opted_out(tmp_path, monkeypatch):
    """With the switch on, the expensive file walk is never reached."""
    def boom(*a, **k):
        raise AssertionError("detect_dependencies must not run when opted out")

    monkeypatch.setattr(dependency_module, "detect_dependencies", boom)
    cfg = _two_source_cfg(tmp_path, skip=True)

    # Must return without walking the tree (and without raising).
    _check_and_prompt_dependencies("svc_main", tmp_path / "svc_main", cfg)
