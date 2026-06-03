"""Tests for per-file doc-type parallelism in DocGenerator.

The catalog-driven generator path runs multiple doc types per file
(explanation, architecture, qa, etc.). These types are independent and
must run concurrently — otherwise walltime is sum-of-types instead of
max-of-types, doubling the run length without reason.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_generate_from_elements_runs_doc_types_concurrently(tmp_path):
    """``generate_from_elements`` must dispatch all doc_types via
    ``asyncio.gather`` — not in a serial ``for`` loop. We verify this by
    making the underlying call slow enough that serial execution would
    take noticeably longer than parallel.
    """
    from docgen.catalog_enrich import EnrichedFileBundle
    from docgen.generator import DocGenerator, GeneratorConfig

    src = tmp_path / 'x.py'
    src.write_text('class Foo:\n    pass\n', encoding='utf-8')

    bundle = EnrichedFileBundle(
        path=src,
        language='python',
        module_name='x',
        module_docstring=None,
        imports=(),
        elements=(),
        line_count=2,
    )

    in_flight = 0
    max_concurrent = 0
    DELAY = 0.10

    async def slow_generate(self, b, src_text, dt):
        nonlocal in_flight, max_concurrent
        in_flight += 1
        max_concurrent = max(max_concurrent, in_flight)
        await asyncio.sleep(DELAY)
        in_flight -= 1
        return f'# doc body for {dt}'

    gen = DocGenerator(config=GeneratorConfig(model='gpt-5.4'))
    # Skip context-manager setup — we patch the only method that calls
    # the provider, so a real provider isn't needed. Setting the slot
    # to a sentinel keeps any "must be used as async context manager"
    # guards happy.
    gen._provider = object()

    with patch.object(
        DocGenerator, '_generate_doc_from_bundle', slow_generate,
    ):
        start = asyncio.get_event_loop().time()
        docs = await gen.generate_from_elements(
            bundle, doc_types=('explanation', 'architecture', 'qa'),
        )
        elapsed = asyncio.get_event_loop().time() - start

    assert len(docs) == 3, f'expected 3 docs, got {len(docs)}'
    assert max_concurrent == 3, (
        f'doc_types ran serially (max {max_concurrent} concurrent); '
        f'expected all 3 to overlap via asyncio.gather'
    )
    # Serial would be ~3 * DELAY; parallel is ~1 * DELAY. Allow a 2x
    # margin — well below the 3x serial floor.
    assert elapsed < DELAY * 2, (
        f'elapsed {elapsed:.3f}s suggests serial execution; '
        f'expected close to {DELAY:.2f}s for parallel'
    )
