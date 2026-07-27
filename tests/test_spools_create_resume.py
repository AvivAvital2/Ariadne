"""`ariadne spools create --resume` — CLI half.

Resume reuses the existing spools.yaml + already-fetched corpus and skips
the interactive setup, so an interrupted build continues without
re-answering which-spool/which-versions. (The fetch-consent skip lands with
the create_spool `resume` param.)
"""
from __future__ import annotations

from pathlib import Path

import yaml

import cli.spools_cmd as spools_cmd
import spool_acquire
from cli.main import create_parser
from spool_acquire import CreateResult

_RECIPE = {
    'name': 'databricks', 'runtime': 'dbr17.3-lts', 'version': '1.0.0',
    'corpus': {'spark': {'url': 'u', 'tag': 'v4.0.0', 'sha': 'deadbeef'}},
}


def _run(argv: list[str]) -> int:
    return spools_cmd.HANDLERS['spools'](create_parser().parse_args(argv))


def test_resume_without_spoolfile_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    setup_calls: list = []
    monkeypatch.setattr(spool_acquire, 'setup_recipe',
                        lambda *a, **k: setup_calls.append(a))
    monkeypatch.setattr(spool_acquire, 'create_spool',
                        lambda *a, **k: CreateResult(accepted=True, pack_path='p'))

    rc = _run(['spools', 'create', '--resume'])
    assert rc == 1                 # nothing to resume
    assert not setup_calls          # never falls back to setup


def test_resume_skips_setup_and_builds(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'spools.yaml').write_text(yaml.safe_dump(_RECIPE))

    setup_calls: list = []
    monkeypatch.setattr(spool_acquire, 'setup_recipe',
                        lambda *a, **k: setup_calls.append(a))
    created: list = []
    monkeypatch.setattr(
        spool_acquire, 'create_spool',
        lambda *a, **k: created.append(k) or CreateResult(
            accepted=True, pack_path='databricks.zip'))

    rc = _run(['spools', 'create', '--resume'])
    assert rc == 0
    assert not setup_calls, 'resume must reuse the recipe, not re-run setup'
    assert created, 'resume must proceed to the build'
    assert 'Resuming' in capsys.readouterr().out


def test_create_spool_resume_skips_fetch_consent(tmp_path, monkeypatch):
    """resume=True approves the fetch (clones reuse at the pinned sha), so the
    consent prompt is skipped; without resume the consent still gates it."""
    from spool_acquire import AcquireResult, create_spool

    sf = tmp_path / 'spools.yaml'
    sf.write_text(yaml.safe_dump(_RECIPE))
    fake_phases = {
        'source_add': lambda *a: None, 'index': lambda *a: None,
        'onboard': lambda name: None, 'build': lambda **k: None,
    }
    seen: dict = {}

    def _fake_acquire(packfile, *, dest_dir, approve, confirm=input):
        seen['approve'] = approve
        return AcquireResult(accepted=True, cloned=())
    monkeypatch.setattr(spool_acquire, 'acquire', _fake_acquire)

    res = create_spool(
        str(sf), dest_dir=tmp_path / 'd1', out_path=str(tmp_path / 'p1.zip'),
        approve=False, resume=True, allow_ungrounded=True, phases=fake_phases,
    )
    assert res.accepted and seen['approve'] is True   # resume → consent skipped

    seen.clear()
    create_spool(
        str(sf), dest_dir=tmp_path / 'd2', out_path=str(tmp_path / 'p2.zip'),
        approve=False, resume=False, allow_ungrounded=True, phases=fake_phases,
    )
    assert seen['approve'] is False                   # no resume → consent gates


def test_plain_create_still_runs_setup(tmp_path, monkeypatch):
    """Without --resume/--yes, setup still runs (guards against the resume
    branch swallowing the normal path)."""
    monkeypatch.chdir(tmp_path)
    setup_calls: list = []
    monkeypatch.setattr(
        spool_acquire, 'setup_recipe',
        lambda *a, **k: setup_calls.append(a) or (tmp_path / 'spools.yaml')
        .write_text(yaml.safe_dump(_RECIPE)))
    monkeypatch.setattr(spool_acquire, 'create_spool',
                        lambda *a, **k: CreateResult(accepted=True, pack_path='p'))

    _run(['spools', 'create', 'databricks'])
    assert setup_calls, 'plain create must run interactive setup'
