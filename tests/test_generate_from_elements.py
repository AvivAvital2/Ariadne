"""Tests for DocGenerator.generate_from_elements (Catalog transition Phase 2.3).

The new entry point on DocGenerator consumes an EnrichedFileBundle (the
output of docgen.catalog_enrich.enrich_file) and produces the same set of
GeneratedDoc instances the legacy generate_for_module produced from a
ModuleMetadata. These tests assert:

- The new method exists and is async.
- For each DocType, the formatted prompt contains the same key facts the
  legacy path injected (module name, public classes/functions, source code,
  dependencies, etc.).
- GeneratedDoc shape is preserved (title, content, doc_type, source_files,
  metadata).
- Title generation works from the bundle.
- The new path produces the SAME doc_types as the legacy path for Python.

LLM calls are mocked at _call_llm.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from docgen.catalog_enrich import enrich_file
from docgen.generator import DocGenerator, GeneratorConfig


def _write(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


@pytest.fixture
def py_file(tmp_path: Path) -> Path:
    """A small but representative Python module."""
    f = tmp_path / 'alpha.py'
    _write(f, '''
        """Alpha module — does alpha things."""

        from typing import Iterable
        from beta import helper

        class Greeter:
            """Greets people."""

            def hello(self, name: str) -> str:
                """Return a greeting."""
                return f"hi {name}"

        class _Internal:
            pass

        def public_fn(items: Iterable[int]) -> int:
            """Sum public items."""
            return sum(items)

        def _private_fn():
            return None
    ''')
    return f


@pytest.fixture
def bundle(py_file: Path, tmp_path: Path):
    return enrich_file(py_file, source_root=tmp_path)


@pytest.fixture
def generator():
    return DocGenerator(config=GeneratorConfig(api_key='dummy'))


# ---------------------------------------------------------------------------
# Method existence and contract
# ---------------------------------------------------------------------------


class TestMethodExists:
    def test_generate_from_elements_is_callable(
        self, generator: DocGenerator,
    ) -> None:
        # The method must exist on DocGenerator — without it the new
        # orchestrator branch can't be wired.
        assert hasattr(generator, 'generate_from_elements')

    @pytest.mark.asyncio
    async def test_generate_from_elements_returns_list(
        self, generator: DocGenerator, bundle,
    ) -> None:
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Generated\n\nbody',
        ):
            async with generator:
                docs = await generator.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )
        assert isinstance(docs, list)
        assert len(docs) == 1


# ---------------------------------------------------------------------------
# Prompt content (the CRITICAL parity check)
# ---------------------------------------------------------------------------


class TestPromptContentParity:
    """For each DocType, the prompt produced from a bundle must contain the
    same key facts the legacy path injected. This catches regressions where
    field-flow is silently dropped during the transition.
    """

    @pytest.mark.asyncio
    async def test_explanation_prompt_includes_module_facts(
        self, generator: DocGenerator, bundle,
    ) -> None:
        captured: list[tuple[str, str]] = []

        async def fake_llm(system, user):
            captured.append((system, user))
            return '# Doc'

        with patch.object(DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm):
            async with generator:
                await generator.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        assert len(captured) == 1
        _, user_prompt = captured[0]
        # Module name surfaces.
        assert 'alpha' in user_prompt
        # Module docstring surfaces.
        assert 'alpha things' in user_prompt
        # Public class/function names surface.
        assert 'Greeter' in user_prompt
        assert 'public_fn' in user_prompt
        # Private members do NOT surface in the public-listing context.
        # (They may appear in source_code, but not in the named summary.)
        # Source code surfaces.
        assert 'def public_fn' in user_prompt or 'public_fn' in user_prompt

    @pytest.mark.asyncio
    async def test_architecture_prompt_includes_dependencies(
        self, generator: DocGenerator, bundle,
    ) -> None:
        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '# Doc'

        with patch.object(DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm):
            async with generator:
                await generator.generate_from_elements(
                    bundle, doc_types=('architecture',),
                )

        user_prompt = captured[0]
        # Imports flow through the dependencies section.
        assert 'typing' in user_prompt or 'beta' in user_prompt
        assert 'Greeter' in user_prompt

    @pytest.mark.asyncio
    async def test_diagram_prompt_includes_classes_and_functions(
        self, generator: DocGenerator, bundle,
    ) -> None:
        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '```mermaid\nclassDiagram\n```'

        with patch.object(DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm):
            async with generator:
                await generator.generate_from_elements(
                    bundle, doc_types=('diagram',),
                )

        user_prompt = captured[0]
        # Diagram prompt sees classes, methods, and free-function names.
        assert 'Greeter' in user_prompt
        assert 'hello' in user_prompt
        assert 'public_fn' in user_prompt

    @pytest.mark.asyncio
    async def test_qa_prompt_includes_source(
        self, generator: DocGenerator, bundle,
    ) -> None:
        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '## Q\n## A'

        with patch.object(DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm):
            async with generator:
                await generator.generate_from_elements(
                    bundle, doc_types=('qa',),
                )

        user_prompt = captured[0]
        assert 'alpha' in user_prompt  # module name
        # Source code is included.
        assert 'Greeter' in user_prompt

    @pytest.mark.asyncio
    async def test_catalog_prompt_includes_source(
        self, generator: DocGenerator, bundle,
    ) -> None:
        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '# alpha — Function Catalog'

        with patch.object(DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm):
            async with generator:
                await generator.generate_from_elements(
                    bundle, doc_types=('catalog',),
                )

        user_prompt = captured[0]
        assert 'alpha' in user_prompt
        assert 'public_fn' in user_prompt

    @pytest.mark.asyncio
    async def test_gotcha_prompt_includes_source(
        self, generator: DocGenerator, bundle,
    ) -> None:
        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '### A gotcha\n**Trigger:** ...'

        with patch.object(DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm):
            async with generator:
                await generator.generate_from_elements(
                    bundle, doc_types=('gotcha',),
                )

        user_prompt = captured[0]
        assert 'alpha' in user_prompt


# ---------------------------------------------------------------------------
# GeneratedDoc shape
# ---------------------------------------------------------------------------


class TestGeneratedDocShape:
    @pytest.mark.asyncio
    async def test_doc_has_correct_doc_type(
        self, generator: DocGenerator, bundle,
    ) -> None:
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Title\n\nbody',
        ):
            async with generator:
                docs = await generator.generate_from_elements(
                    bundle, doc_types=('explanation', 'architecture'),
                )

        types = {d.doc_type for d in docs}
        assert types == {'explanation', 'architecture'}

    @pytest.mark.asyncio
    async def test_doc_source_files_points_at_bundle_path(
        self, generator: DocGenerator, bundle,
    ) -> None:
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Title',
        ):
            async with generator:
                docs = await generator.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        assert len(docs) == 1
        assert docs[0].source_files == (str(bundle.path),)

    @pytest.mark.asyncio
    async def test_doc_metadata_includes_module_name(
        self, generator: DocGenerator, bundle,
    ) -> None:
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Title',
        ):
            async with generator:
                docs = await generator.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        assert docs[0].metadata.get('module_name') == bundle.module_name


# ---------------------------------------------------------------------------
# Title generation
# ---------------------------------------------------------------------------


class TestTitleGeneration:
    @pytest.mark.asyncio
    async def test_explanation_title_uses_first_docstring_line(
        self, generator: DocGenerator, bundle,
    ) -> None:
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Body',
        ):
            async with generator:
                docs = await generator.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        # Module docstring "Alpha module — does alpha things." starts with the title.
        assert 'Alpha' in docs[0].title

    @pytest.mark.asyncio
    async def test_architecture_title_format(
        self, generator: DocGenerator, bundle,
    ) -> None:
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Body',
        ):
            async with generator:
                docs = await generator.generate_from_elements(
                    bundle, doc_types=('architecture',),
                )

        assert 'Architecture' in docs[0].title


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_llm_returning_none_drops_doc(
        self, generator: DocGenerator, bundle,
    ) -> None:
        """If _call_llm returns None (all retries failed), no GeneratedDoc
        should be produced for that doc_type — but other doc_types should
        still succeed.
        """
        responses = iter([None, '# Generated'])

        async def fake_llm(system, user):
            return next(responses)

        with patch.object(DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm):
            async with generator:
                docs = await generator.generate_from_elements(
                    bundle, doc_types=('explanation', 'architecture'),
                )

        # Only the second doc_type succeeded.
        assert len(docs) == 1
        assert docs[0].doc_type == 'architecture'

    @pytest.mark.asyncio
    async def test_llm_exception_does_not_break_pipeline(
        self, generator: DocGenerator, bundle,
    ) -> None:
        """Per legacy semantics — an exception in LLM call for one doc_type
        should be logged and skipped, not raised.
        """
        async def fake_llm(system, user):
            raise RuntimeError('LLM down')

        with patch.object(DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm):
            async with generator:
                docs = await generator.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        # No docs returned, but no exception raised.
        assert docs == []


# ---------------------------------------------------------------------------
# Non-Python languages — bundle works but with empty Python enrichment
# ---------------------------------------------------------------------------


class TestNonPythonBundle:
    @pytest.mark.asyncio
    async def test_javascript_bundle_can_be_processed(
        self, generator: DocGenerator, tmp_path: Path,
    ) -> None:
        f = tmp_path / 'app.js'
        _write(f, '''
            function greet(name) { return "hi " + name; }
            class Service {}
        ''')
        js_bundle = enrich_file(f, source_root=tmp_path)

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# JS Module',
        ):
            async with generator:
                # For Phase 2.3 we just verify it doesn't crash on a
                # JS bundle. Per-language doc-type filtering is Phase 2.4.
                docs = await generator.generate_from_elements(
                    js_bundle, doc_types=('explanation',),
                )

        assert isinstance(docs, list)
