"""Tests for deterministic doc IDs.

UUID4 doc IDs caused duplicates when the same conceptual doc was
generated twice (e.g., re-running with --types architecture after
adding architecture). Deterministic IDs derived from
``(source_name, content_type, primary_key)`` make re-runs idempotent
and let library.add_document UPDATE in place.

The collision case the user hit: two files with the same module-name
leaf produced the same title ("Llm Architecture") and got two random
UUID4 IDs. With deterministic-on-file-path IDs, the two files get
different IDs even though their titles collide.
"""
from __future__ import annotations


def test_doc_id_for_is_deterministic():
    """Same args → same UUID, every call."""
    from schema import doc_id_for

    a = doc_id_for('ariadne', 'explanation', 'docgen/llm/__init__.py')
    b = doc_id_for('ariadne', 'explanation', 'docgen/llm/__init__.py')
    assert a == b


def test_doc_id_for_distinguishes_paths():
    """Two files with the same title-leaf must produce different IDs."""
    from schema import doc_id_for

    # Both files would produce title "Llm Architecture" via the title
    # generator (their module_name leaves are both "llm").
    root_llm = doc_id_for('ariadne', 'architecture', 'llm.py')
    nested_llm = doc_id_for('ariadne', 'architecture', 'docgen/llm/__init__.py')
    assert root_llm != nested_llm


def test_doc_id_for_distinguishes_content_types():
    """Same file, different doc types → different IDs."""
    from schema import doc_id_for

    expl = doc_id_for('ariadne', 'explanation', 'x.py')
    arch = doc_id_for('ariadne', 'architecture', 'x.py')
    qa = doc_id_for('ariadne', 'qa', 'x.py')
    assert len({expl, arch, qa}) == 3


def test_doc_id_for_distinguishes_sources():
    """Same file path under different sources → different IDs."""
    from schema import doc_id_for

    a = doc_id_for('ariadne', 'explanation', 'x.py')
    b = doc_id_for('myproject', 'explanation', 'x.py')
    assert a != b


def test_doc_id_for_returns_string_uuid():
    """Output must be a UUID-format string (so it slots into existing
    documents.id TEXT PRIMARY KEY column).
    """
    import uuid as _uuid

    from schema import doc_id_for

    result = doc_id_for('ariadne', 'explanation', 'x.py')
    parsed = _uuid.UUID(result)  # raises if not a UUID
    assert str(parsed) == result


def test_writer_add_explanation_accepts_doc_id(tmp_path, monkeypatch):
    """``LibraryWriter.add_explanation`` must accept ``doc_id`` and forward
    it to the underlying ``add_document`` so the orchestrator can pass
    deterministic IDs through.
    """
    import asyncio

    import numpy as np

    from library import Library
    from writer import LibraryWriter

    async def fake_embed(self, text):
        return np.zeros(3072, dtype=np.float32)

    async def fake_embed_batch(self, texts):
        return [np.zeros(3072, dtype=np.float32) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr(
        'embedding.EmbeddingService.embed_batch', fake_embed_batch,
    )
    monkeypatch.setattr(
        'embedding.EmbeddingService._get_client', fake_get_client,
    )
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)

    lib = Library(tmp_path / 'test.db')
    try:
        async def run():
            async with LibraryWriter(lib) as writer:
                fixed_id = '11111111-1111-1111-1111-111111111111'
                doc = await writer.add_explanation(
                    title='x',
                    content='body',
                    source_files=['x.py'],
                    source_name='ariadne',
                    doc_id=fixed_id,
                )
                assert doc.id == fixed_id

                # Re-running with same doc_id updates in place (no duplicate row).
                doc2 = await writer.add_explanation(
                    title='x',
                    content='body v2',
                    source_files=['x.py'],
                    source_name='ariadne',
                    doc_id=fixed_id,
                )
                assert doc2.id == fixed_id

                with lib._conn_provider.acquire() as conn:
                    n = conn.execute(
                        'SELECT COUNT(*) FROM documents WHERE id=?',
                        (fixed_id,),
                    ).fetchone()[0]
                assert n == 1, 'deterministic doc_id must update in place'

        asyncio.run(run())
    finally:
        lib.close()
