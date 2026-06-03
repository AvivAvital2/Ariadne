"""Contract tests for DocGenerator prompt-building helpers (#45.2).

The orchestrator's batch path needs to:
1. Collect prompts upfront for every (file, doc_type) pair.
2. Dispatch the prompt list to ``provider.submit_batch``.
3. Reassemble responses into ``GeneratedDoc`` instances that
   downstream validation + storage treat identically to streaming.

These three helpers expose that build/assemble surface:

- ``PromptBundle`` — pre-built prompt + everything assemble_doc
  needs to wrap the response (title, metadata, source path).
- ``build_prompts_for_module`` / ``build_prompts_for_bundle`` —
  parallel of ``generate_for_module`` / ``generate_from_elements``
  but stop short of ``_call_llm``.
- ``assemble_doc`` — wrap a batch response into a ``GeneratedDoc``
  using the pre-computed title + metadata.

Tests use ``DocGenOrchestrator`` as a factory so the
``DocGenerator`` is wired with the same config the production
runtime sees (consistent with
``tests/test_generator_quota_propagation.py``).

Each test is paired with a contract explanation: a real bug that
the test would catch.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from docgen.generator import (
    DocGenerator,
    GeneratedDoc,
    GeneratorConfig,
    PromptBundle,
)
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


def _write(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


@pytest.fixture
def python_file(tmp_path: Path) -> Path:
    """Sample Python module with public symbols the prompt formatter
    will surface — needed to assert ``user_prompt`` carries content."""
    p = tmp_path / 'm.py'
    _write(p, '''"""Sample module for prompt-building tests."""
def foo():
    """Hello."""
    return 1

class Bar:
    """A class."""
    def baz(self):
        return 2
''')
    return p


@pytest.fixture
def json_file(tmp_path: Path) -> Path:
    """JSON file: only ``explanation`` survives the
    ``filter_doc_types_for_language`` filter."""
    p = tmp_path / 'config.json'
    p.write_text('{"key": "value"}', encoding='utf-8')
    return p


def _make_config(
    tmp_path: Path, *, catalog_only: bool = False,
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
        dry_run=True,
        catalog_only_generator=catalog_only,
    )


# ---------------------------------------------------------------------------
# PromptBundle dataclass
# ---------------------------------------------------------------------------


class TestPromptBundle:
    def test_fields_carry_through_construction(self) -> None:
        """PromptBundle must expose the six fields the batch path
        consumes: file, doc_type, system_prompt, user_prompt, title,
        metadata. Bites a refactor that drops or renames any."""
        b = PromptBundle(
            file=Path('/x/y.py'),
            doc_type='explanation',
            system_prompt='SYS',
            user_prompt='USR',
            title='Title',
            metadata={'k': 'v'},
        )
        assert b.file == Path('/x/y.py')
        assert b.doc_type == 'explanation'
        assert b.system_prompt == 'SYS'
        assert b.user_prompt == 'USR'
        assert b.title == 'Title'
        assert b.metadata == {'k': 'v'}


# ---------------------------------------------------------------------------
# build_prompts_for_module — legacy ModuleMetadata path
# ---------------------------------------------------------------------------


class TestBuildPromptsForModule:
    @pytest.mark.asyncio
    async def test_returns_one_bundle_per_doc_type(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """Two doc_types → two PromptBundles, each with its own
        doc_type. Bites a fix that returns a flat list collapsing
        types or ignores the doc_types argument."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None
            metadata = orch._analyzer.analyze_file(python_file)

            bundles = await gen.build_prompts_for_module(
                metadata, ('explanation', 'qa'),
            )

            assert len(bundles) == 2
            assert {b.doc_type for b in bundles} == {'explanation', 'qa'}
            for b in bundles:
                assert b.file == python_file

    @pytest.mark.asyncio
    async def test_user_prompt_carries_module_content(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """User prompt must reference the module's public symbols.
        Bites a fix that builds a prompt skeleton without actually
        filling in metadata, or one that uses a stale/empty
        ``ModuleMetadata`` argument."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None
            metadata = orch._analyzer.analyze_file(python_file)

            bundles = await gen.build_prompts_for_module(
                metadata, ('explanation',),
            )
            user_prompt = bundles[0].user_prompt

            # 'foo' (function) and/or 'Bar' (class) — both are public
            # symbols passed into format_module_info.
            assert 'foo' in user_prompt or 'Bar' in user_prompt
            # System prompt non-empty (template's system text).
            assert bundles[0].system_prompt

    @pytest.mark.asyncio
    async def test_metadata_includes_module_name_and_source_hash(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """``assemble_doc`` downstream uses these fields to wrap
        a batch response into a ``GeneratedDoc``. Bites a fix that
        drops them or substitutes hard-coded placeholders."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None
            metadata = orch._analyzer.analyze_file(python_file)

            bundles = await gen.build_prompts_for_module(
                metadata, ('explanation',),
            )

            assert bundles[0].metadata['module_name'] == metadata.module_name
            assert bundles[0].metadata['source_hash'] == metadata.source_hash

    @pytest.mark.asyncio
    async def test_extra_prompt_context_appended_to_user_prompt(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """Phase 3 (reverse-augment) injects consumer-context via
        ``extra_prompt_context``. Streaming path appends it to the
        user prompt; build path must do the same so batch runs
        receive the same context. Bites a fix that ignores the kwarg."""
        config = _make_config(tmp_path)
        sentinel = '## SENTINEL_CONTEXT_BLOCK'
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None
            metadata = orch._analyzer.analyze_file(python_file)

            bundles = await gen.build_prompts_for_module(
                metadata, ('explanation',),
                extra_prompt_context=sentinel,
            )

            assert sentinel in bundles[0].user_prompt


# ---------------------------------------------------------------------------
# build_prompts_for_bundle — catalog path
# ---------------------------------------------------------------------------


class TestBuildPromptsForBundle:
    @pytest.mark.asyncio
    async def test_returns_one_bundle_per_doc_type(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """Same shape as the module-path test, against the catalog
        ``EnrichedFileBundle`` entry point. Pins parity for non-Python
        languages that go through this path exclusively."""
        from docgen.catalog_enrich import enrich_file

        bundle = enrich_file(python_file, source_root=python_file.parent)
        assert bundle is not None

        config = _make_config(tmp_path, catalog_only=True)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None

            bundles = await gen.build_prompts_for_bundle(
                bundle, ('explanation', 'qa'),
            )

            assert len(bundles) == 2
            assert {b.doc_type for b in bundles} == {'explanation', 'qa'}
            for b in bundles:
                assert b.file == python_file

    @pytest.mark.asyncio
    async def test_filters_doc_types_by_language(
        self, json_file: Path, tmp_path: Path,
    ) -> None:
        """JSON files only receive ``explanation`` (per
        ``filter_doc_types_for_language``). build_prompts_for_bundle
        must apply the same filter as ``generate_from_elements``,
        otherwise non-Python files burn batch tokens on prompts whose
        doc_type doesn't apply."""
        from docgen.catalog_enrich import enrich_file

        bundle = enrich_file(json_file, source_root=json_file.parent)
        assert bundle is not None

        config = _make_config(tmp_path, catalog_only=True)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None

            bundles = await gen.build_prompts_for_bundle(
                bundle,
                ('explanation', 'architecture', 'qa', 'diagram'),
            )

            doc_types = {b.doc_type for b in bundles}
            assert doc_types == {'explanation'}

    @pytest.mark.asyncio
    async def test_metadata_includes_language_and_module_name(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """Bundle path's metadata carries language (legacy path
        carries source_hash). Pins the schema downstream code reads."""
        from docgen.catalog_enrich import enrich_file

        bundle = enrich_file(python_file, source_root=python_file.parent)
        assert bundle is not None

        config = _make_config(tmp_path, catalog_only=True)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None

            bundles = await gen.build_prompts_for_bundle(
                bundle, ('explanation',),
            )

            assert bundles[0].metadata['language'] == bundle.language
            assert bundles[0].metadata['module_name'] == bundle.module_name


# ---------------------------------------------------------------------------
# assemble_doc — wrap batch response into GeneratedDoc
# ---------------------------------------------------------------------------


class TestAssembleDoc:
    def test_assembles_from_prompt_and_content(self) -> None:
        """``assemble_doc(prompt, content)`` must produce a
        ``GeneratedDoc`` whose title/doc_type/source_files/metadata
        come from the prompt and whose content comes from the LLM
        response. This is the symmetry point with
        ``build_prompts_for_module`` — downstream validation and
        storage cannot tell whether a doc came through batch or
        streaming."""
        gen = DocGenerator(config=GeneratorConfig(provider='openai'))
        prompt = PromptBundle(
            file=Path('/x/y.py'),
            doc_type='explanation',
            system_prompt='SYS',
            user_prompt='USR',
            title='My Module',
            metadata={'module_name': 'x.y', 'source_hash': 'abc123'},
        )
        doc = gen.assemble_doc(prompt, 'GENERATED CONTENT')

        assert isinstance(doc, GeneratedDoc)
        assert doc.title == 'My Module'
        assert doc.content == 'GENERATED CONTENT'
        assert doc.doc_type == 'explanation'
        assert doc.source_files == ('/x/y.py',)
        assert doc.metadata == {'module_name': 'x.y', 'source_hash': 'abc123'}


# ---------------------------------------------------------------------------
# Refactor parity — build matches what generate sends
# ---------------------------------------------------------------------------


class TestPromptBuildingMatchesGenerationFlow:
    """The build path must produce identical ``(system, user)`` prompts
    to what ``generate_for_module`` sends through ``_call_llm``.
    Otherwise batch runs and streaming runs would feed different inputs
    to the LLM for the same source — silent quality regression on
    every batch run.
    """

    @pytest.mark.asyncio
    async def test_build_matches_generate_for_module(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """Patch ``_call_llm`` in ``generate_for_module`` to capture
        the (system, user) it receives; verify ``build_prompts_for_module``
        produces the same pair."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None
            metadata = orch._analyzer.analyze_file(python_file)

            captured: list[tuple[str, str]] = []

            async def fake_llm(system: str, user: str) -> str:
                captured.append((system, user))
                return 'fake content'

            with patch.object(
                DocGenerator, '_call_llm',
                new=AsyncMock(side_effect=fake_llm),
            ):
                await gen.generate_for_module(
                    metadata, ('explanation',),
                )

            bundles = await gen.build_prompts_for_module(
                metadata, ('explanation',),
            )

            assert len(captured) == 1
            assert len(bundles) == 1
            assert bundles[0].system_prompt == captured[0][0]
            assert bundles[0].user_prompt == captured[0][1]
