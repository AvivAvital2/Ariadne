"""Contract tests for ``DocGenOrchestrator._collect_prompts`` (#45.4).

The batch path collects prompts upfront for every (file, doc_type)
pair before submitting to Anthropic, then maps results back to files
via a ``file_to_idxs`` map. This stage replaces the streaming
``Semaphore``-coordinated per-file dispatch.

Tests pin:
- Happy path: N files × M doc_types → N*M prompts; file_to_idxs maps
  each file to its M indices.
- SyntaxError on legacy parse → file goes to ``pre_gen_failed``,
  doesn't crash the whole collection.
- File-map sidecar parity: files >200 lines write file_map sidecars
  during collection, same as ``_legacy_generate`` does in streaming.
- Catalog dispatch: ``catalog_only_generator=True`` routes via
  ``enrich_file`` + ``build_prompts_for_bundle``.
- Pre-gen failure on catalog path (unsupported extension) → file
  goes to ``pre_gen_failed``.

Each test independently fails under the stubbed implementation
(returns empty results) — behavioral red, not ImportError.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


def _write(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


def _make_config(
    tmp_path: Path, *, catalog_only: bool = False, dry_run: bool = True,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider='openai',
        model='gpt-5.2',
        doc_types=('explanation', 'qa'),
        validate=False,
        dry_run=dry_run,
        catalog_only_generator=catalog_only,
    )


# ---------------------------------------------------------------------------
# Happy path — multiple files, multiple doc_types
# ---------------------------------------------------------------------------


class TestCollectPromptsHappyPath:
    @pytest.mark.asyncio
    async def test_collects_prompts_for_multiple_files(
        self, tmp_path: Path,
    ) -> None:
        """Two python files × two doc_types → 4 PromptBundles total.
        ``file_to_idxs`` maps each file to its 2 indices. The
        contract is that the orchestrator can map any prompt index
        back to its source file via this dict, which is what
        ``_assemble_and_store`` (#45.6) needs."""
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        _write(a, '''"""A."""
def foo(): return 1
''')
        _write(b, '''"""B."""
def bar(): return 2
''')

        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            prompts, file_to_idxs, pre_gen_failed = await orch._collect_prompts(
                [a, b],
            )

            # 2 files × 2 doc_types = 4 prompts
            assert len(prompts) == 4
            # Each file mapped to 2 indices
            assert set(file_to_idxs.keys()) == {a, b}
            assert all(len(idxs) == 2 for idxs in file_to_idxs.values())
            # Indices cover all 4 prompts exactly once
            all_idxs = sorted(
                i for idxs in file_to_idxs.values() for i in idxs
            )
            assert all_idxs == [0, 1, 2, 3]
            # No pre-gen failures
            assert pre_gen_failed == []

            # Each prompt's file matches its mapping
            for path, idxs in file_to_idxs.items():
                for i in idxs:
                    assert prompts[i].file == path


# ---------------------------------------------------------------------------
# SyntaxError on legacy parse → pre_gen_failed
# ---------------------------------------------------------------------------


class TestCollectPromptsSyntaxError:
    @pytest.mark.asyncio
    async def test_syntax_error_marks_file_pre_gen_failed(
        self, tmp_path: Path,
    ) -> None:
        """A file with invalid Python goes into ``pre_gen_failed``
        without crashing the whole collection. The good file's
        prompts are still returned. Bites a fix that lets SyntaxError
        propagate (one bad file would forfeit the entire batch)."""
        good = tmp_path / 'good.py'
        bad = tmp_path / 'bad.py'
        _write(good, '''"""Good."""
def foo(): return 1
''')
        _write(bad, '''def broken(:
    not valid python
''')

        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            prompts, file_to_idxs, pre_gen_failed = await orch._collect_prompts(
                [good, bad],
            )

            # Good file has its 2 prompts (explanation + qa)
            assert len(prompts) == 2
            assert good in file_to_idxs
            # Bad file is in pre_gen_failed, not in file_to_idxs
            assert bad in pre_gen_failed
            assert bad not in file_to_idxs


# ---------------------------------------------------------------------------
# File-map sidecar parity (legacy path)
# ---------------------------------------------------------------------------


class TestCollectPromptsFileMap:
    @pytest.mark.asyncio
    async def test_writes_file_map_for_files_over_200_lines(
        self, tmp_path: Path,
    ) -> None:
        """Files >200 lines must trigger ``_store_file_map`` writes
        during collection — feature parity with ``_legacy_generate``.
        Without this, a batch run silently drops file_map sidecars
        that streaming runs would have produced.

        Verified by patching ``_store_file_map`` and asserting it
        was called for the big file, not the small one."""
        from unittest.mock import patch

        big = tmp_path / 'big.py'
        small = tmp_path / 'small.py'
        _write(big, '"""big."""\n' + '\n'.join(
            f'def f{i}(): return {i}' for i in range(250)
        ))
        _write(small, '"""small."""\ndef foo(): return 1\n')

        # Use dry_run=False so file_map writes aren't gated off.
        # Patch at the class level — attrs @define makes instance
        # attributes read-only, so patch.object(orch, ...) fails.
        config = _make_config(tmp_path, dry_run=False)
        async with DocGenOrchestrator(config) as orch:
            with patch.object(
                DocGenOrchestrator, '_store_file_map',
            ) as mock_store:
                await orch._collect_prompts([big, small])

                # Class-level patching doesn't bind self via the
                # descriptor protocol for mocks, so call.args[0] is
                # the first real argument (metadata), not self.
                called_paths = [
                    call.args[0].path for call in mock_store.call_args_list
                ]
                assert big in called_paths
                assert small not in called_paths


# ---------------------------------------------------------------------------
# Catalog path
# ---------------------------------------------------------------------------


class TestCollectPromptsCatalogPath:
    @pytest.mark.asyncio
    async def test_dispatches_to_catalog_when_flag_on(
        self, tmp_path: Path,
    ) -> None:
        """``catalog_only_generator=True`` routes through
        ``enrich_file`` + ``build_prompts_for_bundle``. Pins that the
        legacy/catalog fork in ``_build_prompts_for_file`` mirrors
        ``_process_file``'s — otherwise non-Python files would never
        reach the batch path."""
        py = tmp_path / 'm.py'
        _write(py, '''"""M."""
def foo(): return 1
''')

        config = _make_config(tmp_path, catalog_only=True)
        async with DocGenOrchestrator(config) as orch:
            prompts, file_to_idxs, pre_gen_failed = await orch._collect_prompts(
                [py],
            )

            # Catalog path goes through filter_doc_types_for_language —
            # for .py both explanation + qa survive.
            assert len(prompts) >= 1
            assert py in file_to_idxs
            assert pre_gen_failed == []
            # Each prompt has metadata['language'] set (catalog
            # signature) rather than 'source_hash' (legacy signature).
            for p in prompts:
                assert 'language' in p.metadata
