from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library
from schema import EMBEDDING_DIM
from writer import LibraryWriter


@pytest.fixture
def mocked_embedding(monkeypatch):
    """Patch EmbeddingService.embed/embed_batch so add_document makes no API calls."""

    async def fake_embed(self, text):
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    async def fake_embed_batch(self, texts):
        return [np.zeros(EMBEDDING_DIM, dtype=np.float32) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr('embedding.EmbeddingService.embed_batch', fake_embed_batch)
    monkeypatch.setattr('embedding.EmbeddingService._get_client', fake_get_client)
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)


async def test_add_diagram_stores_a_dot_block(tmp_path: Path, mocked_embedding) -> None:
    """add_diagram persists the diagram as a ```dot block (so the bridge can render
    it to PNG with `dot`) and keeps the raw DOT in metadata — not Mermaid."""
    lib = Library(tmp_path / 'd.db')
    writer = LibraryWriter(lib)

    dot = 'digraph G { source -> render -> upload }'
    doc = await writer.add_diagram(
        title='Export Flow',
        description='How the export pipeline runs.',
        dot_code=dot,
        source_name='mylib',
    )

    assert doc.content_type == 'diagram'
    assert '```dot' in doc.content
    assert dot in doc.content
    assert '```mermaid' not in doc.content
    assert doc.metadata.get('dot_code') == dot
    lib.close()
