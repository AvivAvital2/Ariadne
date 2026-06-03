"""Tests for batched embedding in catalog-sync.

Pre-batch behavior: each new element triggers a separate
``EmbeddingService.embed`` call. For scalaproject's thousands of Scala
elements that's thousands of API calls — enough to trigger 502s from
OpenAI's edge under load.

Post-batch: ``sync_file_catalog`` collects all new-element texts and
calls ``embed_batch`` once per file. This drops API calls by the average
elements-per-file factor (often 10-30×).

These tests catch regressions: if someone reverts to per-element embed,
``embed`` calls > 1 and the test fires.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _test_config(tmp_path: Path, monkeypatch):
    """Provide a Config that knows about the 'src' source."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'src')


def _unit(dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 1.0
    return v


class TestSubBatchSizeCap:
    """OpenAI's per-request input cap is 2048 items; total tokens cap
    around 300k. A single file with 1500+ elements would 400-error
    without sub-batching. catalog_writer must chunk the embedding call
    into sub-batches under the cap.
    """

    @pytest.mark.asyncio
    async def test_large_file_chunks_into_sub_batches(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import catalog_writer
        from docgen.catalog_extractor import ElementInfo

        f = tmp_path / 'huge.scala'
        f.write_text('// placeholder', encoding='utf-8')

        # 1500 synthetic elements — bigger than what one embed request
        # would safely accept.
        elements = [
            ElementInfo(
                language='scala', subtype='scala_def', file=str(f),
                qualified_name=f'com.example.Foo.method{i}',
                signature=f'def method{i}',
                line_start=i, line_end=i, col_start=0, col_end=10,
                parent_qualified_name='com.example.Foo',
            )
            for i in range(1500)
        ]
        monkeypatch.setattr(
            'docgen.catalog_writer.extract_elements',
            lambda *args, **kwargs: elements,
        )
        monkeypatch.setattr(
            'docgen.catalog_writer._file_sha', lambda p: 'sha-current',
        )

        embed_batch_sizes: list[int] = []

        async def fake_embed_batch(self, texts):
            embed_batch_sizes.append(len(texts))
            return [_unit() for _ in texts]

        async def fake_embed(self, text):
            return _unit()

        async def fake_get_client(self):
            return None

        async def fake_close(self):
            return None

        monkeypatch.setattr(
            'embedding.EmbeddingService.embed', fake_embed,
        )
        monkeypatch.setattr(
            'embedding.EmbeddingService.embed_batch', fake_embed_batch,
        )
        monkeypatch.setattr(
            'embedding.EmbeddingService._get_client', fake_get_client,
        )
        monkeypatch.setattr(
            'embedding.EmbeddingService.close', fake_close,
        )

        library = MagicMock()
        library.get_document.return_value = None

        from writer import LibraryWriter
        async with LibraryWriter(library) as writer:
            await catalog_writer.sync_file_catalog(
                library, writer, 'src', tmp_path, f,
            )

        # Every batch must be under a safe cap (e.g. 1000).
        assert embed_batch_sizes, 'no embed_batch calls were made'
        assert all(n <= 1000 for n in embed_batch_sizes), (
            f'a sub-batch exceeded the safe cap; sizes: {embed_batch_sizes}'
        )
        # Sum of all batch sizes equals the input count (no items dropped).
        assert sum(embed_batch_sizes) == 1500, (
            f'expected 1500 total items embedded; got {sum(embed_batch_sizes)}'
        )


class TestSyncFileCatalogBatchesEmbeddings:
    @pytest.mark.asyncio
    async def test_three_new_elements_one_embed_batch_call(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A file with 3 new elements should produce exactly 1
        ``embed_batch`` call (containing all 3 texts) — NOT 3
        ``embed`` calls.
        """
        from docgen import catalog_writer
        from docgen.catalog_extractor import ElementInfo

        f = tmp_path / 'm.py'
        f.write_text('# placeholder', encoding='utf-8')

        elements = [
            ElementInfo(
                language='python', subtype='function', file=str(f),
                qualified_name=f'm.fn{i}',
                signature=f'def fn{i}()',
                line_start=i, line_end=i, col_start=0, col_end=10,
            )
            for i in range(3)
        ]
        monkeypatch.setattr(
            'docgen.catalog_writer.extract_elements',
            lambda *args, **kwargs: elements,
        )
        monkeypatch.setattr(
            'docgen.catalog_writer._file_sha',
            lambda p: 'sha-current',
        )

        # Track every embedding service call.
        embed_calls: list[str] = []
        embed_batch_calls: list[list[str]] = []

        async def fake_embed(self, text: str):
            embed_calls.append(text)
            return _unit()

        async def fake_embed_batch(self, texts: list[str]):
            embed_batch_calls.append(list(texts))
            return [_unit() for _ in texts]

        async def fake_get_client(self):
            return None

        async def fake_close(self):
            return None

        monkeypatch.setattr(
            'embedding.EmbeddingService.embed', fake_embed,
        )
        monkeypatch.setattr(
            'embedding.EmbeddingService.embed_batch', fake_embed_batch,
        )
        monkeypatch.setattr(
            'embedding.EmbeddingService._get_client', fake_get_client,
        )
        monkeypatch.setattr(
            'embedding.EmbeddingService.close', fake_close,
        )

        # Library: nothing pre-existing.
        library = MagicMock()
        library.get_document.return_value = None

        from writer import LibraryWriter
        async with LibraryWriter(library) as writer:
            await catalog_writer.sync_file_catalog(
                library, writer, 'src', tmp_path, f,
            )

        # The element texts are what should be in embed_batch's argument.
        # Allow up to 1 single ``embed`` call for the file_index doc itself —
        # or zero, if batched too.
        assert len(embed_batch_calls) >= 1, (
            f'expected at least one embed_batch call; got {embed_batch_calls=}, {embed_calls=}'
        )
        # The 3 new element texts should all appear in batch calls.
        all_batched_texts = [t for batch in embed_batch_calls for t in batch]
        elements_batched = sum(
            1 for t in all_batched_texts if 'fn' in t and 'function' in t
        )
        assert elements_batched == 3, (
            f'expected 3 element texts batched; got {elements_batched} '
            f'(batches: {embed_batch_calls!r})'
        )
        # And critically: at most 1 single-text embed call (for file_index).
        # If we see 3+ embed calls, we're back to per-element.
        assert len(embed_calls) <= 1, (
            f'expected ≤1 single-text embed call (file_index only); '
            f'got {len(embed_calls)} — implementation is back to per-element'
        )
