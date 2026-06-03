"""Tests for generated-doc attribution and uniqueness.

Two real bugs observed in production runs:

1. Generated explanation/architecture docs had ``source_name=NULL`` —
   they were orphaned from their source. Per-source queries silently
   missed them.

2. For Scala/Java, ``EnrichedFileBundle.module_name`` was set to the
   package, so multiple files in the same package produced the same
   doc title. Combined with non-deterministic UUID doc IDs, this
   created duplicate explanation docs (one per file in the package,
   all with the same title).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Bug 1: source_name set on generated docs
# ---------------------------------------------------------------------------


class TestLibraryAddDocumentSetsSourceNameAtomically:
    """``library.add_document(source_name=...)`` must set the column in
    the same INSERT — no separate UPDATE. This eliminates the race window
    where a reader between the INSERT and a follow-up UPDATE would see
    ``source_name=NULL`` momentarily.
    """

    def test_add_document_with_source_name_kwarg_sets_column(
        self, tmp_path,
    ) -> None:
        import numpy as np

        from library import Library

        lib = Library(tmp_path / 'test.db')
        try:
            doc = lib.add_document(
                content_type='explanation',
                title='x',
                content='x',
                source_files=[],
                embedding=np.zeros(3072, dtype=np.float32),
                metadata={},
                source_name='scalaproject',
            )

            with lib._conn_provider.acquire() as conn:
                row = conn.execute(
                    'SELECT source_name FROM documents WHERE id = ?',
                    (doc.id,),
                ).fetchone()
            assert row[0] == 'scalaproject'
        finally:
            lib.close()


class TestSourceNameSetOnGeneratedDocs:
    @pytest.mark.asyncio
    async def test_orchestrator_writes_source_name(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When the orchestrator stores a generated doc, the
        ``source_name`` column on the documents row must be populated.
        Without this, per-source filtering misses everything.
        """
        import numpy as np

        from docgen.orchestrator import (
            DocGenOrchestrator,
            GeneratedDoc,
            OrchestratorConfig,
        )

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

        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            source_name='scalaproject',
            dry_run=False,
        )

        gen_doc = GeneratedDoc(
            title='Some Module',
            content='# explanation body',
            doc_type='explanation',
            source_files=('/path/to/X.scala',),
            metadata={'module_name': 'com.example.X'},
        )

        async with DocGenOrchestrator(config) as orch:
            doc = await orch._store_document(gen_doc)

            assert doc is not None
            with orch._library._conn_provider.acquire() as conn:
                row = conn.execute(
                    'SELECT source_name FROM documents WHERE id = ?',
                    (doc.id,),
                ).fetchone()

        assert row is not None
        assert row[0] == 'scalaproject', (
            f'source_name not set on generated doc; got {row[0]!r}'
        )


class TestCatalogWriterSetsSourceNameAtomically:
    """Catalog docs (per-element + file_index) must populate the
    ``source_name`` *column* in the same INSERT, not only the metadata
    JSON. Otherwise SQL filters on ``WHERE source_name = ?`` miss every
    catalog row even though metadata says the right thing.
    """

    @pytest.fixture(autouse=True)
    def _test_config(self, monkeypatch, tmp_path):
        """Configure ``'scalaproject'`` so the chokepoint admits the
        catalog writer's docs. The contract under test is column
        attribution; source naming is environmental."""
        from tests._scoped_config_fixture import install_test_config
        install_test_config(monkeypatch, tmp_path, 'scalaproject')

    @pytest.mark.asyncio
    async def test_sync_file_catalog_sets_source_name_on_element_and_index(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import numpy as np

        from docgen.catalog_writer import sync_file_catalog
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

        src = tmp_path / 'module.py'
        src.write_text(
            'class Foo:\n    def bar(self):\n        return 1\n',
            encoding='utf-8',
        )

        lib = Library(tmp_path / 'test.db')
        try:
            async with LibraryWriter(lib) as writer:
                await sync_file_catalog(
                    library=lib,
                    writer=writer,
                    source_name='scalaproject',
                    source_root=tmp_path,
                    file=src,
                )

            with lib._conn_provider.acquire() as conn:
                rows = conn.execute(
                    "SELECT id, source_name, "
                    "json_extract(metadata, '$.kind') AS kind "
                    "FROM documents WHERE content_type = 'catalog'",
                ).fetchall()

            assert len(rows) >= 2, (
                f'expected >=2 catalog rows (element + file_index), got {len(rows)}'
            )
            offenders = [
                (r[0], r[2]) for r in rows if r[1] != 'scalaproject'
            ]
            assert not offenders, (
                f'catalog rows missing source_name in column '
                f'(metadata may be set, but column is NULL): {offenders}'
            )
        finally:
            lib.close()


# ---------------------------------------------------------------------------
# Bug 2: module_name disambiguates across files in the same package
# ---------------------------------------------------------------------------


class TestModuleNameUniquePerFile:
    def test_two_scala_files_in_same_package_get_distinct_module_names(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Multiple Scala files in the same package must produce distinct
        ``module_name`` values — otherwise generated docs collide on the
        same title and the library accumulates duplicates.
        """
        from docgen.catalog_enrich import enrich_file
        from docgen.scip_config import SourceScipConfig
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
        )

        # Two files, same package.
        f1 = tmp_path / 'Foo.scala'
        f2 = tmp_path / 'Bar.scala'
        f1.write_text('class Foo\n', encoding='utf-8')
        f2.write_text('class Bar\n', encoding='utf-8')

        sym_foo = 'scip-java maven g a 1 com/example/Foo#'
        sym_bar = 'scip-java maven g a 1 com/example/Bar#'
        synthetic = ScipIndex(
            documents=(
                _ScipDoc(
                    relative_path='Foo.scala',
                    occurrences=(_ScipOccurrence(
                        symbol=sym_foo, range=(0, 0, 0, 5),
                        is_definition=True,
                    ),),
                    symbols=(_ScipSymbol(symbol=sym_foo, kind='Class'),),
                ),
                _ScipDoc(
                    relative_path='Bar.scala',
                    occurrences=(_ScipOccurrence(
                        symbol=sym_bar, range=(0, 0, 0, 5),
                        is_definition=True,
                    ),),
                    symbols=(_ScipSymbol(symbol=sym_bar, kind='Class'),),
                ),
            ),
        )

        with patch(
            'docgen.scip_config.resolve_index', lambda cfg, lang: synthetic,
        ):
            cfg = SourceScipConfig(
                repo='r', artifact_path=tmp_path / 'idx.scip',
                index_kinds={'scala': 'scip'},
            )
            bundle1 = enrich_file(f1, source_root=tmp_path, source_config=cfg)
            bundle2 = enrich_file(f2, source_root=tmp_path, source_config=cfg)

        assert bundle1 is not None and bundle2 is not None
        assert bundle1.module_name != bundle2.module_name, (
            f'both files got module_name={bundle1.module_name!r} — '
            f'duplicate docs will be created on generate'
        )

    def test_scala_module_name_includes_file_stem_when_scip(
        self, tmp_path: Path,
    ) -> None:
        """Scala file ``Foo.scala`` in package ``com.example`` should
        produce module_name like ``com.example.Foo`` — package + file
        stem — so it's unique per file.
        """
        from docgen.catalog_enrich import enrich_file
        from docgen.scip_config import SourceScipConfig
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
        )

        f = tmp_path / 'Foo.scala'
        f.write_text('class Foo\n', encoding='utf-8')

        sym = 'scip-java maven g a 1 com/example/Foo#'
        synthetic = ScipIndex(
            documents=(_ScipDoc(
                relative_path='Foo.scala',
                occurrences=(_ScipOccurrence(
                    symbol=sym, range=(0, 0, 0, 5), is_definition=True,
                ),),
                symbols=(_ScipSymbol(symbol=sym, kind='Class'),),
            ),),
        )

        with patch(
            'docgen.scip_config.resolve_index', lambda cfg, lang: synthetic,
        ):
            cfg = SourceScipConfig(
                repo='r', artifact_path=tmp_path / 'idx.scip',
                index_kinds={'scala': 'scip'},
            )
            bundle = enrich_file(f, source_root=tmp_path, source_config=cfg)

        assert bundle is not None
        # Either "com.example.Foo" (package + stem) or just including
        # the file stem at the end is acceptable; the critical thing is
        # that it's unique per file in the package.
        assert 'Foo' in bundle.module_name, (
            f"module_name {bundle.module_name!r} doesn't carry file "
            f"identity; duplicate docs likely"
        )
