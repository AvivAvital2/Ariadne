"""Onboarding: manually mark which configured sources a project depends on.

During an interactive ``ariadne onboard`` the user is shown the OTHER
configured sources and checks the ones the onboarded project depends on; the
selection is persisted as ``depends_on`` in ariadne.yaml so the paid generate
phase loads those sources' docs as context. This complements the import-based
auto-detector (which is Python-only) — e.g. a Scala service depending on a
Java library can only be expressed this way.

Pins:
- candidate list = the other configured sources, excluding the onboarded
  source itself and Ariadne's own repo (a source whose path is _PACKAGE_ROOT,
  which collides with packages and is never a real dependency target).
- the picker pre-checks the source's current ``depends_on`` and persists the
  new selection to ariadne.yaml.
- a non-TTY context, or a source with no eligible candidates, is a silent
  no-op (the interactive widget is never invoked).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def _config(tmp_path: Path, body: str):
    from config import Config

    p = tmp_path / 'ariadne.yaml'
    p.write_text(textwrap.dedent(body))
    return Config(config_path=p)


def _config_with_candidates(tmp_path: Path):
    """``svc_main`` plus two eligible deps and one source pointing at
    Ariadne's own repo, which must never be offered."""
    from config import _PACKAGE_ROOT

    for d in ('svc_main', 'lib_a', 'lib_b'):
        (tmp_path / d).mkdir()
    return _config(tmp_path, f"""
        default_source: svc_main
        sources:
          svc_main: {tmp_path / 'svc_main'}
          lib_b: {tmp_path / 'lib_b'}
          lib_a: {tmp_path / 'lib_a'}
          the_tool: {_PACKAGE_ROOT}
    """)


def _force_tty(monkeypatch, value: bool = True) -> None:
    monkeypatch.setattr('sys.stdin.isatty', lambda: value)
    monkeypatch.setattr('sys.stdout.isatty', lambda: value)


def test_candidates_exclude_self_and_ariadne_repo(tmp_path):
    from cli.onboard import _dependency_candidates

    cfg = _config_with_candidates(tmp_path)
    # sorted; the onboarded source and Ariadne's own repo are filtered out
    assert _dependency_candidates(cfg, 'svc_main') == ['lib_a', 'lib_b']


def test_picker_prechecks_current_and_persists_selection(tmp_path, monkeypatch):
    from config import Config
    from cli import onboard

    cfg = _config_with_candidates(tmp_path)
    cfg.set_source_dependencies('svc_main', ['lib_a'])          # an existing dep
    cfg = Config(config_path=tmp_path / 'ariadne.yaml')         # reload so it's visible
    _force_tty(monkeypatch)

    seen: dict = {}

    def fake_widget(options, *, title, selected):
        seen['options'] = [v for v, _ in options]
        seen['selected'] = set(selected)
        return ['lib_a', 'lib_b']                              # user also checks lib_b

    monkeypatch.setattr(onboard, '_arrow_key_multiselect', fake_widget)

    onboard._select_onboard_dependencies(cfg, 'svc_main')

    # the existing dep arrives pre-checked, by index into the option list
    assert seen['options'] == ['lib_a', 'lib_b']
    assert seen['selected'] == {0}
    # the new selection is persisted to ariadne.yaml
    persisted = Config(
        config_path=tmp_path / 'ariadne.yaml',
    ).get_source_dependencies('svc_main')
    assert sorted(persisted) == ['lib_a', 'lib_b']


def test_non_tty_is_silent_noop(tmp_path, monkeypatch):
    from config import Config
    from cli import onboard

    cfg = _config_with_candidates(tmp_path)
    _force_tty(monkeypatch, value=False)

    def boom(*a, **k):
        raise AssertionError('widget must not run without a TTY')

    monkeypatch.setattr(onboard, '_arrow_key_multiselect', boom)
    onboard._select_onboard_dependencies(cfg, 'svc_main')   # must not raise

    assert Config(
        config_path=tmp_path / 'ariadne.yaml',
    ).get_source_dependencies('svc_main') == []


def test_no_candidates_is_silent_noop(tmp_path, monkeypatch):
    from config import _PACKAGE_ROOT
    from cli import onboard

    (tmp_path / 'svc_main').mkdir()
    cfg = _config(tmp_path, f"""
        default_source: svc_main
        sources:
          svc_main: {tmp_path / 'svc_main'}
          the_tool: {_PACKAGE_ROOT}
    """)
    _force_tty(monkeypatch)

    def boom(*a, **k):
        raise AssertionError('widget must not run with no candidates')

    monkeypatch.setattr(onboard, '_arrow_key_multiselect', boom)
    onboard._select_onboard_dependencies(cfg, 'svc_main')   # must not raise


# ---------------------------------------------------------------------------
# Making detection optional in interactive onboard: a y/n gate offered before
# the picker. A 'no' persists ``skip_dependency_detection: true`` so the later
# generate phase's import scan is skipped too; a 'yes' opens the picker.
# ---------------------------------------------------------------------------


def _spy_picker(monkeypatch):
    """Replace the dependency picker with a spy recording whether it ran."""
    from cli import onboard

    calls: list = []
    monkeypatch.setattr(
        onboard, '_select_onboard_dependencies',
        lambda *a, **k: calls.append((a, k)),
    )
    return calls


def test_offer_runs_picker_on_yes(tmp_path, monkeypatch):
    from cli import onboard

    cfg = _config_with_candidates(tmp_path)
    _force_tty(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda *a, **k: 'y')
    picked = _spy_picker(monkeypatch)

    onboard._offer_dependency_detection(cfg, 'svc_main')

    assert len(picked) == 1, 'a yes must open the dependency picker'


def test_offer_persists_skip_and_bypasses_picker_on_no(tmp_path, monkeypatch):
    from config import Config
    from cli import onboard

    cfg = _config_with_candidates(tmp_path)
    _force_tty(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda *a, **k: 'n')
    picked = _spy_picker(monkeypatch)

    onboard._offer_dependency_detection(cfg, 'svc_main')

    assert picked == [], 'a no must NOT open the picker'
    # The opt-out is persisted so generate's import scan is skipped too.
    reloaded = Config(config_path=tmp_path / 'ariadne.yaml')
    assert reloaded.source_skip_dependency_detection('svc_main') is True


def test_offer_is_noop_when_already_opted_out(tmp_path, monkeypatch):
    from config import Config
    from cli import onboard

    cfg = _config_with_candidates(tmp_path)
    cfg.set_source_config('svc_main', skip_dependency_detection=True)
    cfg = Config(config_path=tmp_path / 'ariadne.yaml')
    _force_tty(monkeypatch)

    def no_prompt(*a, **k):
        raise AssertionError('must not prompt when already opted out')

    monkeypatch.setattr('builtins.input', no_prompt)
    picked = _spy_picker(monkeypatch)

    onboard._offer_dependency_detection(cfg, 'svc_main')   # must not raise

    assert picked == [], 'opted-out source must not open the picker'
