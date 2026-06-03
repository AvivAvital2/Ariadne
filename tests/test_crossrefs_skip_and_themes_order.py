"""Tests for crossref skipping + post-processing order.

The orchestrator's post-process used to run crossrefs first, then
themes. With ~5000 docs, crossrefs is O(N²) and never completes, so
themes never built. Two fixes:

1. ``refresh_themes`` runs *before* ``_inject_crossrefs_scoped`` — themes
   are useful on their own and shouldn't gate on crossrefs finishing.
2. Crossrefs can be disabled per-run via ``inject_crossrefs=False``,
   and auto-skip when scoped doc count exceeds a threshold to prevent
   hangs on large repos.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_themes_run_before_crossrefs(tmp_path, monkeypatch):
    """``refresh_themes`` must be called BEFORE ``_inject_crossrefs_scoped``
    so that even if crossrefs hang or are skipped, themes still build.
    """
    import numpy as np

    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    async def fake_embed(self, text):
        return np.zeros(3072, dtype=np.float32)

    async def fake_embed_batch(self, texts):
        return [np.zeros(3072, dtype=np.float32) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr('embedding.EmbeddingService.embed_batch', fake_embed_batch)
    monkeypatch.setattr('embedding.EmbeddingService._get_client', fake_get_client)
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)

    cfg = OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'x.db',
        staleness_db_path=tmp_path / 's.db',
        dry_run=False,
        inject_crossrefs=True,
        themes_enabled=True,
    )

    call_order: list[str] = []

    async def fake_refresh_themes(*args, **kwargs):
        call_order.append('themes')
        return {'path': 'noop'}

    async def fake_crossrefs(self, progress_callback=None):
        call_order.append('crossrefs')

    monkeypatch.setattr(
        'docgen.orchestrator.DocGenOrchestrator._inject_crossrefs_scoped',
        fake_crossrefs,
    )
    monkeypatch.setattr(
        'docgen.themes.refresh_themes', fake_refresh_themes,
    )

    async with DocGenOrchestrator(cfg) as orch:
        from docgen.orchestrator import GenerationResult
        results = [GenerationResult(
            source_path=tmp_path / 'fake.py',
            docs_generated=1, docs_failed=0,
        )]
        await orch._post_process(results)

    assert call_order == ['themes', 'crossrefs'], (
        f'expected themes before crossrefs, got {call_order}'
    )


def test_orchestrator_has_inject_crossrefs_field_defaulting_true():
    """Existing field; default stays True for backwards-compat."""
    from docgen.orchestrator import OrchestratorConfig

    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
    )
    assert cfg.inject_crossrefs is True


def test_orchestrator_inject_crossrefs_can_be_false():
    from docgen.orchestrator import OrchestratorConfig

    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
        inject_crossrefs=False,
    )
    assert cfg.inject_crossrefs is False


# Threshold-guard tests removed: crossref injection now uses the
# precomputed doc_graph (O(N×K)) instead of brute-force regex (O(N²)),
# so the threshold is unnecessary. See test_crossrefs_graph_based.py for
# the graph-based behavior tests.
