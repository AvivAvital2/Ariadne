"""Per-doc-type staleness — Tier 2 (orchestrator generates only missing types).

When a file's source is unchanged but some requested doc types are missing, the
run generates ONLY the missing types — reusing the present ones — instead of
regenerating the full set. See ``designs/per-doctype-staleness.md``.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from docgen.generator import DocGenerator
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig
from docgen.staleness import StalenessTracker
from library import Library

_DOC_CONTENT = (
    '# Generated\n\nA sufficiently long generated document body to clear any '
    'minimum content length floor in the generator pipeline for the test.\n'
)
_ALL = ('explanation', 'architecture', 'qa', 'gotcha', 'diagram')


def _make_source_tree(root: Path, files: dict[str, str]) -> Path:
    src = root / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for rel, code in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(dedent(code).lstrip('\n'), encoding='utf-8')
    return src


def _config(tmp_path: Path, source: Path, **over) -> OrchestratorConfig:
    base = dict(
        source_path=source,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        dry_run=True,
        api_key='test-not-used',
        provider='openai',
        model='gpt-5.2',
        doc_types=_ALL,
        concurrency=2,
        validate=False,
        inject_crossrefs=False,
        themes_enabled=False,
        catalog_only_generator=False,
    )
    base.update(over)
    return OrchestratorConfig(**base)


def _seed_existing(db_path, stale_path, source, rel, content_types):
    """Document ``rel`` with ``content_types`` (in the same DBs the run uses)
    so a later run sees them present on unchanged source."""
    src_file = source / rel
    lib = Library(db_path)
    try:
        ids = []
        for ct in content_types:
            doc = lib.add_document(
                content_type=ct, title=f'doc-{ct}', content='body',
                source_files=[str(src_file)],
                embedding=np.zeros(3072, dtype=np.float32),
                metadata={}, source_name='test',
            )
            ids.append(doc.id)
    finally:
        lib.close()
    tracker = StalenessTracker(stale_path)
    try:
        tracker.record_documentation(src_file, ids, base_path=source)
    finally:
        tracker.close()


@pytest.mark.asyncio
async def test_run_generates_only_missing_doc_types(tmp_path):
    """a.py already has explanation+architecture (unchanged source); a run
    requesting all five generates ONLY qa/gotcha/diagram → 3 LLM calls, not 5."""
    source = _make_source_tree(tmp_path, {'a.py': '"""a."""\ndef foo(): pass\n'})
    db = tmp_path / 'ariadne.db'
    stale = tmp_path / 'staleness.db'
    _seed_existing(db, stale, source, 'a.py', ['explanation', 'architecture'])

    config = _config(tmp_path, source, doc_types=_ALL)
    with patch.object(
        DocGenerator, '_call_llm', new_callable=AsyncMock,
        return_value=_DOC_CONTENT,
    ) as mock_llm:
        async with DocGenOrchestrator(config) as orch:
            await orch.run()

    assert mock_llm.call_count == 3, (
        'expected only the 3 missing types (qa/gotcha/diagram) to generate; '
        f'got {mock_llm.call_count} LLM calls'
    )


@pytest.mark.asyncio
async def test_batch_prompt_builder_narrows_to_missing_types(tmp_path):
    """The batch path's prompt builder (_build_prompts_for_file) narrows to the
    per-file missing types too — so a batch run prices/submits only those."""
    source = _make_source_tree(tmp_path, {'a.py': '"""a."""\ndef foo(): pass\n'})
    config = _config(tmp_path, source, doc_types=_ALL)  # legacy path

    captured: dict = {}

    async def fake_build(self, metadata, doc_types):
        captured['doc_types'] = tuple(doc_types)
        return []

    with patch.object(DocGenerator, 'build_prompts_for_module', new=fake_build):
        async with DocGenOrchestrator(config) as orch:
            orch._doc_types_by_file = {source / 'a.py': ('qa', 'gotcha', 'diagram')}
            await orch._build_prompts_for_file(source / 'a.py')

    assert captured['doc_types'] == ('qa', 'gotcha', 'diagram'), (
        f'batch builder must use the per-file missing set; got {captured.get("doc_types")}'
    )


@pytest.mark.asyncio
async def test_force_regenerate_bypasses_narrowing(tmp_path):
    """``--force`` regenerates ALL requested types even for a partially-
    documented file — the per-doc-type narrowing is bypassed (escape hatch)."""
    source = _make_source_tree(tmp_path, {'a.py': '"""a."""\ndef foo(): pass\n'})
    db = tmp_path / 'ariadne.db'
    stale = tmp_path / 'staleness.db'
    _seed_existing(db, stale, source, 'a.py', ['explanation', 'architecture'])

    config = _config(tmp_path, source, doc_types=_ALL, force_regenerate=True)
    with patch.object(
        DocGenerator, '_call_llm', new_callable=AsyncMock,
        return_value=_DOC_CONTENT,
    ) as mock_llm:
        async with DocGenOrchestrator(config) as orch:
            await orch.run()

    assert mock_llm.call_count == 5, (
        f'--force must regenerate all 5 types, not narrow to the missing; '
        f'got {mock_llm.call_count}'
    )
@pytest.mark.asyncio
async def test_commit_gate_still_fills_missing_types_on_unchanged_source(tmp_path):
    """Sampleproj fix: with the commit-diff gate active (restrict_to_files set) and
    NO source file changed since sync (empty restrict), a run requesting
    architecture for a file that only has explanation STILL generates the missing
    architecture — instead of skipping everything. Only the missing type runs
    (1 call), not the full requested set."""
    source = _make_source_tree(tmp_path, {'a.py': '"""a."""\ndef foo(): pass\n'})
    db = tmp_path / 'ariadne.db'
    stale = tmp_path / 'staleness.db'
    _seed_existing(db, stale, source, 'a.py', ['explanation'])  # missing architecture

    config = _config(
        tmp_path, source, doc_types=('explanation', 'architecture'),
        restrict_to_files=frozenset(),  # synced, nothing changed since baseline
    )
    with patch.object(
        DocGenerator, '_call_llm', new_callable=AsyncMock,
        return_value=_DOC_CONTENT,
    ) as mock_llm:
        async with DocGenOrchestrator(config) as orch:
            await orch.run()

    assert mock_llm.call_count == 1, (
        'unchanged synced source must still generate the missing architecture '
        f'doc (1 call), not skip everything; got {mock_llm.call_count}'
    )
@pytest.mark.asyncio
async def test_per_language_doc_type_override_skips_excluded_format(tmp_path):
    """A source override capping python to ('explanation',) means a python file
    generates only explanation — architecture is skipped — even though python
    supports it and the run requests both. This is the doc-type screen's
    per-format exclude, enforced at generation."""
    source = _make_source_tree(tmp_path, {'a.py': '"""a."""\ndef foo(): pass\n'})
    config = _config(
        tmp_path, source, doc_types=('explanation', 'architecture'),
        doc_types_by_language={'python': ('explanation',)},
    )
    with patch.object(
        DocGenerator, '_call_llm', new_callable=AsyncMock, return_value=_DOC_CONTENT,
    ) as mock_llm:
        async with DocGenOrchestrator(config) as orch:
            await orch.run()
    assert mock_llm.call_count == 1, (
        'python is capped to explanation by the override; architecture must be '
        f'skipped → 1 call, got {mock_llm.call_count}'
    )
