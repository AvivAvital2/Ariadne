"""A pinned/immutable spool corpus that is already fully generated at its sha
must skip discovery + generation entirely on a re-run — the file set is frozen,
so re-walking it looks for changes that cannot exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config as config_module
import docgen.orchestrator
import docgen.themes
from docgen.generation_marker import current_corpus_shas, write_marker
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


def _spool_config(
    tmp_path: Path, src: Path, *, dry_run: bool = True, themes_enabled: bool = False,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        source_path=src,
        source_name='demo',
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider='anthropic',
        model='claude-3-5-sonnet',
        doc_types=('explanation',),
        validate=False,
        dry_run=dry_run,
        batch_mode='always',
        auto_batch_threshold=200,
        inject_crossrefs=False,
        themes_enabled=themes_enabled,
    )


@pytest.fixture
def exempt_spool(tmp_path: Path):
    """A pinned spool corpus (clone dir with a corpus-sha marker) whose source is
    registered staleness-exempt in the active config."""
    src = tmp_path / 'corpus'
    (src / 'repo').mkdir(parents=True)
    (src / 'repo' / '.ariadne-corpus-sha').write_text('deadbeef\n', encoding='utf-8')
    (src / 'repo' / 'a.py').write_text('x = 1\n', encoding='utf-8')

    yaml = tmp_path / 'ariadne.yaml'
    yaml.write_text(
        f'sources:\n  demo:\n    path: {src}\n    ignore_staleness: true\n',
        encoding='utf-8',
    )
    saved = config_module._global_config
    config_module._global_config = config_module.Config(config_path=yaml)
    yield src
    config_module._global_config = saved


def _spy_discovery(monkeypatch) -> list:
    calls: list[str] = []
    monkeypatch.setattr(
        docgen.orchestrator, 'find_catalog_files',
        lambda *a, **k: calls.append('catalog') or [],
    )
    monkeypatch.setattr(
        docgen.orchestrator, 'find_python_files',
        lambda *a, **k: calls.append('python') or [],
    )
    return calls


@pytest.mark.asyncio
async def test_skips_discovery_when_pinned_corpus_fully_generated(
    exempt_spool: Path, tmp_path: Path, monkeypatch,
) -> None:
    src = exempt_spool
    # a prior completed build recorded the generation marker at this sha
    write_marker(src / '.ariadne', corpus_shas=current_corpus_shas(src), doc_types=('explanation',))
    calls = _spy_discovery(monkeypatch)

    async with DocGenOrchestrator(_spool_config(tmp_path, src)) as orch:
        result = await orch.run()

    assert calls == []                       # NO discovery walk happened
    assert result.files_processed == 0
    assert result.docs_created == 0


@pytest.mark.asyncio
async def test_walks_when_no_completion_marker(
    exempt_spool: Path, tmp_path: Path, monkeypatch,
) -> None:
    """No marker (fresh or interrupted build) => must still walk to find the
    not-yet-generated files."""
    src = exempt_spool
    calls = _spy_discovery(monkeypatch)

    async with DocGenOrchestrator(_spool_config(tmp_path, src)) as orch:
        await orch.run()

    assert calls != []                       # discovery ran


@pytest.mark.asyncio
async def test_generation_marker_written_before_themes(
    exempt_spool: Path, tmp_path: Path, monkeypatch,
) -> None:
    """The marker is written at the START of _post_process — BEFORE theme
    summarization — so an abort/failure during the long themes phase still
    leaves it, and the next run skips discovery. (The bug: it used to be written
    only AFTER themes, so aborting mid-themes lost it forever and re-walked.)"""
    src = exempt_spool
    marker = src / '.ariadne' / 'generation.ok'
    assert not marker.exists()

    # Simulate themes being interrupted: refresh_themes raises. _post_process
    # catches it (non-fatal), but the marker must already be on disk by then.
    async def _boom(*args, **kwargs):
        raise RuntimeError('themes interrupted')

    monkeypatch.setattr(docgen.themes, 'refresh_themes', _boom)

    cfg = _spool_config(tmp_path, src, dry_run=False, themes_enabled=True)
    async with DocGenOrchestrator(cfg) as orch:
        await orch._post_process([], crossref_progress=None)

    assert marker.exists()  # written before themes → survives the abort
