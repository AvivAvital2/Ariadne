"""Tests for per-language prompt parameterization (Catalog transition Phase 2.4).

The catalog-driven path serves multiple languages; the prompts must adapt:
- Code-fence label matches the language (```python vs ```javascript vs ```json)
- System framing references the language ("Python module" vs "JSON config")
- Doc-type set is curated per language (e.g. JSON/YAML/MD only get 'explanation')

These tests pin down LANGUAGE_DOC_TYPES and the language-aware rendering
without breaking the legacy Python-only path.
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


# ---------------------------------------------------------------------------
# LANGUAGE_DOC_TYPES — data contract
# ---------------------------------------------------------------------------


class TestLanguageDocTypes:
    def test_python_supports_full_set(self) -> None:
        from docgen.prompts import LANGUAGE_DOC_TYPES

        assert 'python' in LANGUAGE_DOC_TYPES
        py_types = set(LANGUAGE_DOC_TYPES['python'])
        # Python keeps full coverage — explanation/architecture/qa/catalog/gotcha/diagram.
        assert {'explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram'}.issubset(py_types)

    def test_data_languages_only_get_explanation(self) -> None:
        """JSON/YAML/Markdown are config/doc formats — only 'explanation' makes
        sense (architecture/qa/etc. are nonsense for a JSON dict).
        """
        from docgen.prompts import LANGUAGE_DOC_TYPES

        for lang in ('json', 'yaml', 'markdown'):
            assert LANGUAGE_DOC_TYPES[lang] == ('explanation',)

    def test_html_and_javascript_get_subset(self) -> None:
        from docgen.prompts import LANGUAGE_DOC_TYPES

        # HTML has structure but no Q&A/gotcha pattern that fits.
        # JS does have those patterns.
        for lang in ('html', 'javascript'):
            assert 'explanation' in LANGUAGE_DOC_TYPES[lang]
            assert 'architecture' in LANGUAGE_DOC_TYPES[lang]


# ---------------------------------------------------------------------------
# Code-fence label adapts per language
# ---------------------------------------------------------------------------


class TestCodeFenceAdaptsToLanguage:
    @pytest.mark.asyncio
    async def test_python_bundle_uses_python_fence(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / 'm.py'
        _write(f, '''
            """A python module."""
            def foo(): pass
        ''')
        bundle = enrich_file(f, source_root=tmp_path)
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '# Doc'

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm,
        ):
            async with gen:
                await gen.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        assert '```python' in captured[0]

    @pytest.mark.asyncio
    async def test_javascript_bundle_uses_javascript_fence(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / 'app.js'
        _write(f, '''
            function greet() { return 1; }
        ''')
        bundle = enrich_file(f, source_root=tmp_path)
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '# Doc'

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm,
        ):
            async with gen:
                await gen.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        # The fence in the prompt's source-code block matches the language.
        assert '```javascript' in captured[0]
        # And the python fence isn't used.
        assert '```python\n' not in captured[0]

    @pytest.mark.asyncio
    async def test_json_bundle_uses_json_fence(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / 'config.json'
        f.write_text('{"name": "ariadne"}', encoding='utf-8')
        bundle = enrich_file(f, source_root=tmp_path)
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '# Doc'

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm,
        ):
            async with gen:
                await gen.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        assert '```json' in captured[0]


# ---------------------------------------------------------------------------
# Language framing in user prompt
# ---------------------------------------------------------------------------


class TestLanguageFraming:
    @pytest.mark.asyncio
    async def test_javascript_prompt_mentions_javascript_not_python(
        self, tmp_path: Path,
    ) -> None:
        """The user prompt's framing line should describe JavaScript code
        when the bundle is JS — not still say 'Python code'.
        """
        f = tmp_path / 'app.js'
        _write(f, '''
            function greet() { return 1; }
        ''')
        bundle = enrich_file(f, source_root=tmp_path)
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '# Doc'

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm,
        ):
            async with gen:
                await gen.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )

        prompt = captured[0]
        # The framing should say JavaScript (case-insensitive check).
        assert 'javascript' in prompt.lower() or 'JavaScript' in prompt
        # And shouldn't claim it's Python.
        assert 'Python code' not in prompt


# ---------------------------------------------------------------------------
# Doc-type filtering when no doc_types argument is passed
# ---------------------------------------------------------------------------


class TestPerLanguageDefaultDocTypes:
    @pytest.mark.asyncio
    async def test_json_bundle_only_generates_explanation_by_default(
        self, tmp_path: Path,
    ) -> None:
        """If the caller doesn't override doc_types, JSON gets only
        'explanation' — not all six default Python doc types.
        """
        f = tmp_path / 'config.json'
        f.write_text('{"name": "ariadne"}', encoding='utf-8')
        bundle = enrich_file(f, source_root=tmp_path)
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# JSON config',
        ):
            async with gen:
                docs = await gen.generate_from_elements(bundle)

        types = {d.doc_type for d in docs}
        # JSON should only get 'explanation' even though the generator's
        # config.doc_types defaults to all six.
        assert types == {'explanation'}

    @pytest.mark.asyncio
    async def test_python_bundle_uses_full_doc_types_by_default(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / 'm.py'
        _write(f, '''
            """py."""
            def f(): pass
        ''')
        bundle = enrich_file(f, source_root=tmp_path)
        gen = DocGenerator(config=GeneratorConfig(
            api_key='dummy',
            doc_types=('explanation', 'architecture', 'catalog'),
        ))

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Doc',
        ):
            async with gen:
                docs = await gen.generate_from_elements(bundle)

        types = {d.doc_type for d in docs}
        # Python uses the configured/default doc_types (intersected with what's
        # supported for python — which is everything).
        assert types == {'explanation', 'architecture', 'catalog'}

    @pytest.mark.asyncio
    async def test_explicit_doc_types_overrides_language_default(
        self, tmp_path: Path,
    ) -> None:
        """When the caller passes doc_types explicitly, that wins — but
        the per-language filter is still applied so we don't try to
        generate 'qa' for JSON.
        """
        f = tmp_path / 'config.json'
        f.write_text('{}', encoding='utf-8')
        bundle = enrich_file(f, source_root=tmp_path)
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Doc',
        ):
            async with gen:
                # User asks for explanation + qa, but qa isn't in
                # LANGUAGE_DOC_TYPES['json']. The qa doc should be filtered out.
                docs = await gen.generate_from_elements(
                    bundle, doc_types=('explanation', 'qa'),
                )

        types = {d.doc_type for d in docs}
        assert types == {'explanation'}


# ---------------------------------------------------------------------------
# Legacy path is not regressed
# ---------------------------------------------------------------------------


class TestLegacyPathUnchanged:
    """The legacy generate_for_module uses the same templates. Adding the new
    placeholders must not break it (the legacy path passes language='python').
    """

    def test_format_module_info_unchanged(self) -> None:
        # Sanity: format_module_info is the legacy formatter and shouldn't
        # depend on language. (If we accidentally moved the language plumbing
        # into format_module_info, it would error here without the kwarg.)
        from docgen.prompts import format_module_info

        out = format_module_info('foo.bar', 'Doc.', ['A'], ['b'])
        assert 'foo.bar' in out
        assert 'A' in out
