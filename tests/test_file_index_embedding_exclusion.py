"""file_index docs are pure derived index data — never worth an embedding.

They exist so catalog-sync can short-circuit on file_sha and track which
element docs belong to a file (both live in metadata, looked up by id), and
so a file stays present in the catalog. Their content
("Catalog index for X -- N elements.") carries no semantic signal, yet they
were ~28% of all embeddings. This module pins the design intent:

  1. file_index docs are created WITHOUT an embedding (element docs keep theirs).
  2. They never enter the "needs (re)embedding" set, so an only_missing
     rebuild leaves them NULL instead of "fixing" them.
  3. export omits them; import skips any stray file_index markdown and
     regenerates the per-file index from list_documents() alone — no source
     tree, no API calls.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np
import pytest

from docgen.catalog_writer import (
    regenerate_file_index_docs,
    sync_file_catalog,
    sync_source_catalog,
)
from export import LibraryExporter, import_from_markdown
from library import Library
from schema import CATALOG_KIND_ELEMENT, CATALOG_KIND_FILE_INDEX
from writer import LibraryWriter


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


@pytest.fixture(autouse=True)
def _test_config(tmp_path: Path, monkeypatch):
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'test')


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'catalog-test.db')
    yield lib
    lib.close()


@pytest.fixture
def mocked_embedding(monkeypatch):
    """Patch the embedding service so tests make no API calls.

    Returns a non-zero, normalized vector so "has an embedding" is
    distinguishable from "NULL embedding" — a zero vector would be a valid
    stored embedding, which is exactly what we must NOT see on file_index docs.
    """
    async def fake_embed(self, text):
        return np.ones(8, dtype=np.float32) / np.sqrt(8)

    async def fake_embed_batch(self, texts):
        return [np.ones(8, dtype=np.float32) / np.sqrt(8) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr('embedding.EmbeddingService.embed_batch', fake_embed_batch)
    monkeypatch.setattr('embedding.EmbeddingService._get_client', fake_get_client)
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)


def _run_sync_file(library, source_root, file):
    async def go():
        async with LibraryWriter(library) as writer:
            return await sync_file_catalog(library, writer, 'test', source_root, file)
    return asyncio.run(go())


def _run_sync_source(library, source_root):
    async def go():
        async with LibraryWriter(library) as writer:
            return await sync_source_catalog(library, writer, 'test', source_root)
    return asyncio.run(go())


def _run_rebuild(library, only_missing):
    async def go():
        async with LibraryWriter(library) as writer:
            return await writer.rebuild_all_embeddings(only_missing=only_missing)
    return asyncio.run(go())


def _by_kind(library, kind):
    return [d for d in library.list_documents() if d.metadata.get('kind') == kind]


def _exported_kinds(docs_dir: Path) -> list[str]:
    """Every metadata.kind value found across the exported markdown frontmatter."""
    kinds: list[str] = []
    for md in docs_dir.rglob('*.md'):
        for match in re.finditer(r'(?m)^\s*kind:\s*"?(\w+)"?\s*$', md.read_text()):
            kinds.append(match.group(1))
    return kinds


class TestCreatedWithoutEmbedding:
    def test_file_index_has_no_embedding_but_elements_do(
        self, tmp_path: Path, library, mocked_embedding,
    ) -> None:
        f = _write(tmp_path, 'mod.py', 'def foo(): pass\nclass Bar: pass\n')
        _run_sync_file(library, tmp_path, f)

        indexes = _by_kind(library, CATALOG_KIND_FILE_INDEX)
        elements = _by_kind(library, CATALOG_KIND_ELEMENT)
        assert len(indexes) == 1
        assert len(elements) == 2
        # The whole point: the index doc costs no embedding.
        assert indexes[0].embedding is None
        # Elements remain the searchable, embedded units.
        assert all(e.embedding is not None for e in elements)

    def test_symbolless_file_still_has_an_index_doc_just_unembedded(
        self, tmp_path: Path, library, mocked_embedding,
    ) -> None:
        # A CSS file has no extractable symbols, so its file_index doc is its
        # only catalog representation. It must still exist (so the file is
        # catalogued) but, like every index doc, carries no embedding.
        f = _write(tmp_path, 'styles.css', '.btn { color: red; }\n')
        _run_sync_file(library, tmp_path, f)
        assert _by_kind(library, CATALOG_KIND_ELEMENT) == []  # genuinely symbol-less

        indexes = _by_kind(library, CATALOG_KIND_FILE_INDEX)
        assert len(indexes) == 1
        assert indexes[0].embedding is None


class TestNeverInNeedsEmbeddingSet:
    """An unembedded file_index doc must not look like a doc 'missing' its
    embedding — otherwise every only_missing rebuild (which `import` runs)
    would re-embed it, undoing the saving."""

    def test_unembedded_index_is_not_counted_as_missing(
        self, tmp_path: Path, library, mocked_embedding,
    ) -> None:
        f = _write(tmp_path, 'mod.py', 'def foo(): pass\nclass Bar: pass\n')
        _run_sync_file(library, tmp_path, f)
        # Elements are embedded; the only NULL-embedding doc is the file_index,
        # and it is deliberately exempt — so nothing is "missing".
        assert library.count_missing_embeddings() == 0
        assert library.list_documents_without_embedding() == []

    def test_only_missing_rebuild_leaves_index_unembedded(
        self, tmp_path: Path, library, mocked_embedding,
    ) -> None:
        f = _write(tmp_path, 'mod.py', 'def foo(): pass\n')
        _run_sync_file(library, tmp_path, f)
        _run_rebuild(library, only_missing=True)
        assert _by_kind(library, CATALOG_KIND_FILE_INDEX)[0].embedding is None

    def test_full_rebuild_leaves_index_unembedded(
        self, tmp_path: Path, library, mocked_embedding,
    ) -> None:
        f = _write(tmp_path, 'mod.py', 'def foo(): pass\n')
        _run_sync_file(library, tmp_path, f)
        _run_rebuild(library, only_missing=False)
        index = _by_kind(library, CATALOG_KIND_FILE_INDEX)[0]
        assert index.embedding is None
        # Elements are still (re)embedded by a full rebuild.
        assert all(e.embedding is not None for e in _by_kind(library, CATALOG_KIND_ELEMENT))


class TestExportExcludeImportRegenerate:
    """The shareable artifact carries elements but not the derived index; a
    consumer rebuilds the per-file index from the element docs on import."""

    def test_roundtrip_excludes_index_then_regenerates_from_documents(
        self, tmp_path: Path, library, mocked_embedding,
    ) -> None:
        _write(tmp_path, 'mod.py', 'def foo(): pass\nclass Bar: pass\n')
        _write(tmp_path, 'sub/other.py', 'def baz(): pass\n')
        _write(tmp_path, 'styles.css', '.btn { color: red; }\n')  # symbol-less
        _run_sync_source(library, tmp_path)

        # export omits file_index docs but keeps the element docs.
        docs_dir = tmp_path / '_exported'
        LibraryExporter(library).export_all(docs_dir)
        kinds = _exported_kinds(docs_dir)
        assert CATALOG_KIND_FILE_INDEX not in kinds
        assert CATALOG_KIND_ELEMENT in kinds

        # The element ids for mod.py — what the regenerated index must point to.
        mod_element_ids = {
            d.id for d in _by_kind(library, CATALOG_KIND_ELEMENT)
            if d.source_files and d.source_files[0].endswith('mod.py')
        }
        assert len(mod_element_ids) == 2

        fresh = Library(tmp_path / 'fresh.db')
        try:
            import_from_markdown(fresh, docs_dir)
            # Nothing index-shaped arrives from the export.
            assert _by_kind(fresh, CATALOG_KIND_FILE_INDEX) == []
            assert len(_by_kind(fresh, CATALOG_KIND_ELEMENT)) == 3

            # Regenerate purely from the imported documents.
            rebuilt = regenerate_file_index_docs(fresh)
            # One index per element-bearing file (mod.py, sub/other.py). The
            # symbol-less styles.css left no elements and cannot be recovered.
            assert rebuilt == 2
            indexes = _by_kind(fresh, CATALOG_KIND_FILE_INDEX)
            assert len(indexes) == 2
            assert all(i.embedding is None for i in indexes)

            mod_index = next(i for i in indexes if i.title.endswith('mod.py'))
            assert set(mod_index.metadata['element_ids']) == mod_element_ids
            assert mod_index.content == 'Catalog index for mod.py -- 2 elements.'
        finally:
            fresh.close()


class TestImportSkipsStrayIndex:
    def test_import_skips_file_index_markdown_when_present(
        self, tmp_path: Path,
    ) -> None:
        # An older export may still hold a file_index doc on disk; import must
        # skip it — index docs only ever come back via regeneration.
        other = tmp_path / 'docs' / 'other'
        other.mkdir(parents=True)
        (other / 'file-index.md').write_text(
            '---\n'
            'id: idx-1\n'
            'type: catalog\n'
            'title: "file_index:test:mod.py"\n'
            'metadata:\n'
            '  kind: "file_index"\n'
            '---\n'
            '# file_index:test:mod.py\n\nCatalog index for mod.py -- 1 elements.\n'
        )
        (other / 'element.md').write_text(
            '---\n'
            'id: el-1\n'
            'type: catalog\n'
            'title: "mod.foo"\n'
            'metadata:\n'
            '  kind: "element"\n'
            '---\n'
            '# mod.foo\n\nfunction mod.foo ...\n'
        )

        fresh = Library(tmp_path / 'fresh2.db')
        try:
            count = import_from_markdown(fresh, tmp_path / 'docs')
            assert count == 1  # only the element doc was imported
            assert _by_kind(fresh, CATALOG_KIND_FILE_INDEX) == []
            assert len(_by_kind(fresh, CATALOG_KIND_ELEMENT)) == 1
        finally:
            fresh.close()
