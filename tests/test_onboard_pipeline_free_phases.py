"""The web ``ariadne_onboard`` tool must run the free phases
(discover → index → catalog-sync) before the paid phases, matching the CLI
blueprint (``cmd_onboard`` runs them via dry-run). Without them a SCIP-routed
source crashes in generate (no index built) and a Python source silently gets an
empty catalog + cross-source graph.

``run_onboard_pipeline`` gains an ``include_free_phases`` flag: the tool sets it,
while the CLI (which already ran the free phases in its dry-run preview) leaves
it off so it doesn't re-index.
"""
from __future__ import annotations

import pytest

import cli.catalog as cat
import cli.index as idx
import cli.onboard_pipeline as op
from cli.onboard_pipeline import OnboardError


def _recorder(calls, name, *, is_async, rc=0):
    if is_async:
        async def _f(ns, *a, **k):
            calls.append(name)
            return rc
    else:
        def _f(ns, *a, **k):
            calls.append(name)
            return rc
    return _f


class _FakeLib:
    def count_documents(self):
        return 0

    def list_themes(self, coherent_only=False):
        return []


@pytest.fixture
def calls(monkeypatch):
    recorded: list[str] = []
    # Free phases are imported locally from their source modules inside the
    # pipeline, so patch them there.
    monkeypatch.setattr(idx, 'cmd_discover', _recorder(recorded, 'discover', is_async=False))
    monkeypatch.setattr(idx, 'cmd_index', _recorder(recorded, 'index', is_async=False))
    monkeypatch.setattr(cat, 'cmd_catalog_sync', _recorder(recorded, 'catalog_sync', is_async=True))
    # Paid phases are imported at module top.
    monkeypatch.setattr(op, 'cmd_catalog_describe', _recorder(recorded, 'describe', is_async=True))
    monkeypatch.setattr(op, 'cmd_generate', _recorder(recorded, 'generate', is_async=True))
    monkeypatch.setattr(op, 'cmd_themes_build', _recorder(recorded, 'themes', is_async=True))
    monkeypatch.setattr(op, 'get_library', lambda db_path=None: _FakeLib())
    return recorded


async def test_free_phases_run_before_paid_when_requested(calls, tmp_path):
    await op.run_onboard_pipeline(
        'webapp', 'claude-opus-4-8', ('explanation',),
        include_free_phases=True, db_path=str(tmp_path / 'x.db'),
    )
    assert calls == ['discover', 'index', 'catalog_sync',
                     'describe', 'generate', 'themes']


async def test_free_phases_skipped_by_default(calls, tmp_path):
    # The CLI (cmd_onboard) already ran the free phases in its dry-run preview,
    # so the default must NOT re-run them — only the paid phases.
    await op.run_onboard_pipeline(
        'webapp', 'claude-opus-4-8', ('explanation',),
        db_path=str(tmp_path / 'x.db'),
    )
    assert calls == ['describe', 'generate', 'themes']


async def test_free_phase_failure_is_fatal_before_paid(calls, tmp_path, monkeypatch):
    # A failing free phase (index rc != 0) must stop the pipeline before any
    # paid (LLM) phase runs — no budget spent on a broken index.
    monkeypatch.setattr(idx, 'cmd_index', _recorder(calls, 'index', is_async=False, rc=2))
    with pytest.raises(OnboardError):
        await op.run_onboard_pipeline(
            'webapp', 'claude-opus-4-8', ('explanation',),
            include_free_phases=True, db_path=str(tmp_path / 'x.db'),
        )
    assert calls == ['discover', 'index']
