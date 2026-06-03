"""Tests for catalog-driven group architecture (Catalog transition Phase 2.6).

Replaces ``DocGenerator._generate_group_architecture`` (which walks
``ModuleGroup.all_modules``) with a version that walks ``iter_catalog_files``
and builds ``EnrichedFileBundle`` objects per file. The new entry point —
``DocGenerator.generate_for_directory(path)`` — produces a single group-level
architecture doc summarizing every catalog file in the directory.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from docgen.generator import DocGenerator, GeneratorConfig


def _write(path: Path, src: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    """A two-file Python package."""
    pkg = tmp_path / 'demo'
    _write(pkg / 'alpha.py', '''
        """Alpha — does alpha things."""
        class AlphaService:
            """The alpha service."""
            def run(self): pass
    ''')
    _write(pkg / 'beta.py', '''
        """Beta — does beta things."""
        def beta_helper():
            """Beta helper function."""
            return 42
    ''')
    return pkg


# ---------------------------------------------------------------------------
# Method existence
# ---------------------------------------------------------------------------


class TestMethodExists:
    def test_generate_for_directory_is_callable(self) -> None:
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))
        assert hasattr(gen, 'generate_for_directory')


# ---------------------------------------------------------------------------
# Group arch doc content
# ---------------------------------------------------------------------------


class TestGroupArchContent:
    @pytest.mark.asyncio
    async def test_directory_produces_arch_doc(
        self, package_dir: Path,
    ) -> None:
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# demo Architecture\n\n## Overview\n...',
        ):
            async with gen:
                docs = await gen.generate_for_directory(
                    package_dir, doc_types=('architecture',),
                )

        # At least one doc — the group arch.
        assert any(d.doc_type == 'architecture' for d in docs)

    @pytest.mark.asyncio
    async def test_group_arch_prompt_includes_all_modules(
        self, package_dir: Path,
    ) -> None:
        """The directory-level prompt should reference every module in the
        package, so the LLM sees the full surface.
        """
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        captured: list[str] = []

        async def fake_llm(system, user):
            captured.append(user)
            return '# Arch'

        with patch.object(
            DocGenerator, '_call_llm',
            new_callable=AsyncMock, side_effect=fake_llm,
        ):
            async with gen:
                await gen.generate_for_directory(
                    package_dir, doc_types=('architecture',),
                )

        # Find the prompt for the group-level arch doc (not the per-file ones).
        # Per-file arch prompts mention "Architecture" too — the group prompt
        # is the one that mentions both modules side-by-side.
        prompt_with_both = [
            p for p in captured if 'alpha' in p and 'beta' in p
        ]
        assert prompt_with_both, (
            'no prompt referenced both alpha and beta — group-level arch '
            'doc must aggregate all modules in the directory'
        )
        prompt = prompt_with_both[0]
        # Module summaries surface.
        assert 'AlphaService' in prompt or 'alpha' in prompt
        assert 'beta_helper' in prompt or 'beta' in prompt

    @pytest.mark.asyncio
    async def test_group_arch_title_uses_directory_name(
        self, package_dir: Path,
    ) -> None:
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Arch',
        ):
            async with gen:
                docs = await gen.generate_for_directory(
                    package_dir, doc_types=('architecture',),
                )

        arch_docs = [d for d in docs if d.doc_type == 'architecture']
        # The group-level doc's title should mention the directory name.
        # Per-file arch docs already mention their own module — we want
        # at least one doc whose title is the package name.
        group_titles = [d.title for d in arch_docs if 'demo' in d.title.lower()]
        assert group_titles, (
            f'no arch doc titled after the directory; got titles: '
            f'{[d.title for d in arch_docs]}'
        )


# ---------------------------------------------------------------------------
# Empty directory
# ---------------------------------------------------------------------------


class TestEmptyDirectory:
    @pytest.mark.asyncio
    async def test_empty_directory_produces_no_docs(
        self, tmp_path: Path,
    ) -> None:
        empty = tmp_path / 'empty'
        empty.mkdir()
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Doc',
        ):
            async with gen:
                docs = await gen.generate_for_directory(
                    empty, doc_types=('architecture',),
                )

        # No catalog files → no group arch doc.
        assert not any(d.doc_type == 'architecture' for d in docs)

    @pytest.mark.asyncio
    async def test_single_file_directory_does_not_emit_group_arch(
        self, tmp_path: Path,
    ) -> None:
        """One file is just a module — the group-level summary doesn't add
        anything beyond the per-module arch. Mirror the legacy behavior
        (``len(group.all_modules) > 1``).
        """
        pkg = tmp_path / 'lone'
        _write(pkg / 'only.py', '''"""Only one."""
def f(): pass
''')
        gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value='# Doc',
        ):
            async with gen:
                docs = await gen.generate_for_directory(
                    pkg, doc_types=('architecture',),
                )

        # The group-level architecture doc is suppressed; the per-file one
        # may still exist if doc_types includes "architecture".
        group_arch = [
            d for d in docs
            if d.doc_type == 'architecture'
            and 'lone' in d.title.lower()
            and d.metadata.get('group') is True
        ]
        assert not group_arch
