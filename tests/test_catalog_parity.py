"""Prompt-equivalence tests for the catalog transition (Phase 3.1).

The legacy ``generate_for_module`` path (SourceAnalyzer → ModuleMetadata) and
the new ``generate_from_elements`` path (catalog_extractor → enrich_file →
EnrichedFileBundle) should produce semantically equivalent prompts for the
same Python file. These tests capture both pipelines' prompts via mocked
_call_llm and assert both contain the same key facts: module name, public
classes, public functions, top-level dependencies, source code.

The tests are deterministic — no LLM calls. They catch the "field-flow
silently dropped" class of regression that an LLM-output-based parity check
might miss because the LLM smooths over absent context.

Per Phase 3, real-LLM parity (via tests/parity_judge.py) and frozen-fixture
regression are user-driven and gated on captured fixtures.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from docgen.catalog_enrich import enrich_file
from docgen.generator import DocGenerator, GeneratorConfig


def _write(path: Path, src: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


# ---------------------------------------------------------------------------
# Fixtures — ~6 representative Python shapes
# ---------------------------------------------------------------------------


def _fixture_dataclass(tmp_path: Path) -> Path:
    f = tmp_path / 'dc_module.py'
    _write(f, '''
        """Dataclass module."""
        from dataclasses import dataclass

        @dataclass
        class Point:
            """A 2D point."""
            x: int
            y: int

        def make_origin() -> Point:
            """Return the origin."""
            return Point(0, 0)
    ''')
    return f


def _fixture_attrs_frozen(tmp_path: Path) -> Path:
    f = tmp_path / 'attrs_module.py'
    _write(f, '''
        """attrs frozen module."""
        from attrs import frozen

        @frozen
        class Config:
            """Immutable config."""
            host: str
            port: int = 80

        def default_config() -> Config:
            return Config(host="localhost")
    ''')
    return f


def _fixture_abc(tmp_path: Path) -> Path:
    f = tmp_path / 'abc_module.py'
    _write(f, '''
        """Abstract base."""
        from abc import ABC, abstractmethod

        class Provider(ABC):
            """Abstract provider."""

            @abstractmethod
            def fetch(self, key: str) -> str: ...

            def cached(self, key: str) -> str:
                """Cache wrapper."""
                return self.fetch(key)
    ''')
    return f


def _fixture_async(tmp_path: Path) -> Path:
    f = tmp_path / 'async_module.py'
    _write(f, '''
        """Async module."""
        import asyncio
        from typing import AsyncIterator

        async def fetch(url: str) -> bytes:
            """Fetch URL contents."""
            await asyncio.sleep(0)
            return b""

        async def stream() -> AsyncIterator[bytes]:
            """Stream chunks."""
            for _ in range(3):
                yield b""
    ''')
    return f


def _fixture_nested_classes(tmp_path: Path) -> Path:
    f = tmp_path / 'nested_module.py'
    _write(f, '''
        """Nested classes."""

        class Outer:
            """Outer class."""

            class Inner:
                """Inner class."""

                def deep(self) -> int:
                    """Deep method."""
                    return 1

            def shallow(self) -> int:
                return 0
    ''')
    return f


def _fixture_package_init(tmp_path: Path) -> Path:
    pkg = tmp_path / 'demopkg'
    f = pkg / '__init__.py'
    _write(f, '''
        """Package init."""
        from demopkg.alpha import alpha_fn
        from demopkg.beta import BetaService

        __all__ = ["alpha_fn", "BetaService"]
    ''')
    # Add the referenced sibling files so the package looks real.
    _write(pkg / 'alpha.py', 'def alpha_fn(): pass\n')
    _write(pkg / 'beta.py', 'class BetaService: pass\n')
    return f


FIXTURES = [
    ('dataclass', _fixture_dataclass),
    ('attrs_frozen', _fixture_attrs_frozen),
    ('abc', _fixture_abc),
    ('async', _fixture_async),
    ('nested_classes', _fixture_nested_classes),
    ('package_init', _fixture_package_init),
]


# ---------------------------------------------------------------------------
# Helpers — capture prompts from both paths
# ---------------------------------------------------------------------------


async def _capture_legacy_prompt(
    file: Path, doc_type: str,
) -> tuple[str, str]:
    from docgen._legacy_analyzer import SourceAnalyzer

    analyzer = SourceAnalyzer()
    metadata = analyzer.analyze_file(file)
    gen = DocGenerator(config=GeneratorConfig(api_key='dummy'), analyzer=analyzer)

    captured: list[tuple[str, str]] = []

    async def fake_llm(system, user):
        captured.append((system, user))
        return '# stub'

    with patch.object(
        DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm,
    ):
        async with gen:
            await gen.generate_for_module(metadata, doc_types=(doc_type,))

    assert captured, f'legacy path produced no prompt for {doc_type}'
    return captured[0]


async def _capture_new_prompt(
    file: Path, source_root: Path, doc_type: str,
) -> tuple[str, str]:
    bundle = enrich_file(file, source_root=source_root)
    assert bundle is not None
    gen = DocGenerator(config=GeneratorConfig(api_key='dummy'))

    captured: list[tuple[str, str]] = []

    async def fake_llm(system, user):
        captured.append((system, user))
        return '# stub'

    with patch.object(
        DocGenerator, '_call_llm', new_callable=AsyncMock, side_effect=fake_llm,
    ):
        async with gen:
            await gen.generate_from_elements(bundle, doc_types=(doc_type,))

    assert captured, f'new path produced no prompt for {doc_type}'
    return captured[0]


# ---------------------------------------------------------------------------
# Per-doctype parity
# ---------------------------------------------------------------------------


_DOC_TYPES_TO_TEST = ('explanation', 'architecture', 'qa', 'catalog', 'diagram')


@pytest.mark.parametrize('name,builder', FIXTURES)
@pytest.mark.parametrize('doc_type', _DOC_TYPES_TO_TEST)
@pytest.mark.asyncio
async def test_prompt_parity(
    name: str, builder, doc_type: str, tmp_path: Path,
) -> None:
    """Both prompt paths retain module facts and comparable length for each template."""
    f = builder(tmp_path)
    legacy = await _capture_legacy_prompt(f, doc_type)
    new = await _capture_new_prompt(f, tmp_path, doc_type)

    legacy_user = legacy[1]
    new_user = new[1]

    # Both should mention some form of the module name. For most fixtures
    # the bare stem appears (e.g. "dc_module"); the package-init fixture
    # has module_name "demopkg".
    expected_token = f.stem if f.name != '__init__.py' else f.parent.name
    assert expected_token in legacy_user, (
        f'[{name}/{doc_type}] legacy prompt missing module name {expected_token!r}'
    )
    assert expected_token in new_user, (
        f'[{name}/{doc_type}] new prompt missing module name {expected_token!r}'
    )
    legacy_len = len(legacy_user)
    new_len = len(new_user)
    ratio = new_len / max(legacy_len, 1)
    assert 0.5 <= ratio <= 1.5, (
        f'[{name}/{doc_type}] prompt length divergence: '
        f'legacy={legacy_len}, new={new_len}, ratio={ratio:.2f}'
    )

    if doc_type in ('explanation', 'architecture'):
        from docgen._legacy_analyzer import SourceAnalyzer

        metadata = SourceAnalyzer().analyze_file(f)
        if doc_type == 'explanation':
            for cls in {c.name for c in metadata.public_classes}:
                assert cls in legacy_user, f'[{name}] legacy prompt missing class {cls}'
                assert cls in new_user, f'[{name}] new prompt missing class {cls}'
            for fn in {fn.name for fn in metadata.public_functions}:
                assert fn in legacy_user, f'[{name}] legacy prompt missing function {fn}'
                assert fn in new_user, f'[{name}] new prompt missing function {fn}'
            src = f.read_text(encoding='utf-8')
            lines = [ln for ln in src.splitlines() if ln.strip() and not ln.startswith('#!')]
            if lines:
                marker = lines[0]
                assert marker in legacy_user, f'[{name}] legacy prompt missing src marker {marker!r}'
                assert marker in new_user, f'[{name}] new prompt missing src marker {marker!r}'
        else:
            for dep in {d for d in metadata.dependencies if d}:
                assert dep in legacy_user, f'[{name}] legacy arch prompt missing dep {dep}'
                assert dep in new_user, f'[{name}] new arch prompt missing dep {dep}'
