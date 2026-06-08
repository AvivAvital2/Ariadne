"""End-to-end tests for the Themes pipeline.

These exercise the full chain: catalog elements → semantic edges → Leiden
clustering → LLM summarization → searchable theme docs. Mocked at the LLM
and embedding boundaries; everything else is real wiring.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library
from writer import LibraryWriter


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _add_catalog(
    library: Library,
    doc_id: str,
    embedding: list[float],
    *,
    description: str | None = None,
    source_name: str = 'test',
) -> None:
    metadata: dict[str, object] = {
        'kind': 'element',
        'source_name': source_name,
        'qualified_name': doc_id,
        'subtype': 'function',
    }
    if description is not None:
        metadata['description'] = description
    library.add_document(
        content_type='catalog',
        title=doc_id,
        content=f'function {doc_id}',
        source_files=[],
        embedding=_unit(embedding),
        metadata=metadata,
        doc_id=doc_id,
    )
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE documents SET source_name = ? WHERE id = ?',
            (source_name, doc_id),
        )


def _populate_two_clusters(library: Library) -> None:
    for i, v in enumerate([
        [1.0, 0.05, 0, 0, 0, 0, 0, 0],
        [1.0, 0.04, 0, 0, 0, 0, 0, 0],
        [1.0, 0.06, 0, 0, 0, 0, 0, 0],
    ]):
        _add_catalog(library, f'A{i}', v, description=f'A-thing {i}')
    for i, v in enumerate([
        [0, 0, 1.0, 0.05, 0, 0, 0, 0],
        [0, 0, 1.0, 0.04, 0, 0, 0, 0],
        [0, 0, 1.0, 0.06, 0, 0, 0, 0],
    ]):
        _add_catalog(library, f'B{i}', v, description=f'B-thing {i}')


@pytest.fixture(autouse=True)
def _test_config(monkeypatch, tmp_path):
    """Configure the ``'test'`` source so the chokepoint admits these
    fixture docs. The contract under test is end-to-end theme
    summarization; source naming is environmental."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'test')


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'themes-e2e.db')
    yield lib
    lib.close()


@pytest.fixture
def mocked_embedding(monkeypatch):
    async def fake_embed(self, text):
        return np.zeros(3072, dtype=np.float32)

    async def fake_embed_batch(self, texts):
        return [np.zeros(3072, dtype=np.float32) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr('embedding.EmbeddingService.embed_batch', fake_embed_batch)
    monkeypatch.setattr('embedding.EmbeddingService._get_client', fake_get_client)
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)


def _coherent_response(label: str = 'Theme') -> str:
    return (
        f'# {label}: Cluster Theme\n\n'
        '## What this is\nA cluster.\n\n'
        '## Why this is a coherent theme\nShared concern.\n\n'
        '## Key participants\n- **m1** — does m1 things\n\n'
        '## Cross-cutting concerns\nNone.\n\n'
        '## Caveats\nNone apparent.\n'
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_populate_to_searchable_themes(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        """The headline e2e: populate catalog → build semantic edges → cluster
        → summarize → themes appear as content_type='theme' docs in the library
        with real LLM-written content (not '(pending summarization)').
        """
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges
        from docgen.themes import generate_themes
        from ariadne_mcp.service_themes import themes_action

        async def fake_chat(messages, *, model=None, **kwargs):
            return _coherent_response()

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        # 1. Populate
        _populate_two_clusters(library)

        # 2. Build semantic edges
        n_edges = build_semantic_edges(library, k=5, min_sim=0.6)
        assert n_edges > 0

        # 3. Cluster
        run = cluster_themes(library, min_cluster_size=3)
        assert len(run.clusters) == 2

        # 4. Summarize all dirty themes
        async with LibraryWriter(library) as writer:
            summary = await generate_themes(library, writer)
        assert summary['summarized'] == 2
        assert summary['incoherent'] == 0
        assert summary['failed'] == 0

        # 5. Themes are coherent and have real content (not placeholder)
        themes = library.list_themes(coherent_only=True)
        assert len(themes) == 2
        for theme in themes:
            doc = library.get_document(theme.doc_id)
            assert doc is not None
            assert doc.content.startswith('# '), (
                f'theme {theme.cluster_id} content should be markdown, '
                f'got: {doc.content[:100]!r}'
            )
            assert '(pending summarization)' not in doc.content
            assert theme.dirty is False
            assert theme.summary_hash != ''

        # 6. The MCP tool surface returns them.
        result = themes_action(library, action='list')
        assert len(result['themes']) == 2
        assert result['total'] == 2

    @pytest.mark.asyncio
    async def test_drift_triggers_full_recluster_e2e(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        """After an initial build + summarize, adding many new catalog
        elements should make refresh_themes (re-called) trigger a full
        recluster — and the new themes should also get real content.
        """
        from docgen.graph_builder import build_semantic_edges
        from docgen.themes import refresh_themes

        async def fake_chat(messages, *, model=None, **kwargs):
            return _coherent_response()

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)

        # Initial build: refresh_themes recognizes no prior cluster_history
        # and does the first-run path internally.
        async with LibraryWriter(library) as writer:
            initial = await refresh_themes(library, writer)
        assert initial['path'] == 'initial_build'

        # Add 4 new elements → 4/10 = 40% drift, well above default threshold 0.05.
        for i in range(4):
            _add_catalog(library, f'C{i}', [0, 1.0, 0, 0, 0.05 * i, 0, 0, 0])

        async with LibraryWriter(library) as writer:
            summary = await refresh_themes(library, writer)

        assert summary['recluster_full'] is True
        assert summary['path'] == 'rebuilt'

        # Themes are coherent and content is real (not placeholder).
        themes = library.list_themes(coherent_only=True)
        for theme in themes:
            doc = library.get_document(theme.doc_id)
            assert doc is not None and doc.content.startswith('# ')


# ---------------------------------------------------------------------------
# Search returns themes
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    """The integration tests that catch the original 'wired-but-orphaned'
    bug. If `refresh_themes` isn't actually called from `orchestrator.run`,
    these fail.
    """

    @pytest.mark.asyncio
    async def test_orchestrator_run_invokes_refresh_themes(
        self, tmp_path, mocked_embedding, monkeypatch,
    ) -> None:
        """Driving DocGenOrchestrator.run() must result in refresh_themes
        being called exactly once. Without this, the themes pipeline never
        runs during ariadne generate.
        """
        from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

        refresh_calls: list[dict] = []

        async def tracking(library, writer, **kwargs):
            refresh_calls.append(kwargs)
            return {'path': 'noop'}

        monkeypatch.setattr('docgen.themes.refresh_themes', tracking)

        # Empty source dir → run() finds no files, no LLM work needed.
        src = tmp_path / 'src'
        src.mkdir()

        config = OrchestratorConfig(
            source_path=src,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'staleness.db',
        )

        async with DocGenOrchestrator(config) as orch:
            await orch.run()

        assert len(refresh_calls) == 1, (
            f'refresh_themes must be called exactly once from orchestrator.run; '
            f'got {len(refresh_calls)} calls — wiring is missing or duplicated'
        )
        # Default themes_enabled=True is forwarded.
        assert refresh_calls[0].get('enabled') is True

    @pytest.mark.asyncio
    async def test_orchestrator_forwards_themes_enabled_false(
        self, tmp_path, mocked_embedding, monkeypatch,
    ) -> None:
        """When OrchestratorConfig.themes_enabled is False, refresh_themes
        should still be called (so its 'disabled' code path runs and the
        config is honoured at the function boundary, not via dead-code
        removal in the caller).
        """
        from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

        refresh_calls: list[dict] = []

        async def tracking(library, writer, **kwargs):
            refresh_calls.append(kwargs)
            return {'path': 'disabled'}

        monkeypatch.setattr('docgen.themes.refresh_themes', tracking)

        src = tmp_path / 'src'
        src.mkdir()

        config = OrchestratorConfig(
            source_path=src,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'staleness.db',
            themes_enabled=False,
        )

        async with DocGenOrchestrator(config) as orch:
            await orch.run()

        assert len(refresh_calls) == 1
        assert refresh_calls[0].get('enabled') is False

    @pytest.mark.asyncio
    async def test_orchestrator_skips_refresh_themes_on_dry_run(
        self, tmp_path, mocked_embedding, monkeypatch,
    ) -> None:
        """Dry-run shouldn't write to DB; refresh_themes should NOT be called."""
        from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

        refresh_calls: list[dict] = []

        async def tracking(library, writer, **kwargs):
            refresh_calls.append(kwargs)
            return {'path': 'noop'}

        monkeypatch.setattr('docgen.themes.refresh_themes', tracking)

        src = tmp_path / 'src'
        src.mkdir()

        config = OrchestratorConfig(
            source_path=src,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'staleness.db',
            dry_run=True,
        )

        async with DocGenOrchestrator(config) as orch:
            await orch.run()

        assert refresh_calls == [], 'dry_run must not invoke refresh_themes'


class TestSearchReturnsTheme:
    @pytest.mark.asyncio
    async def test_text_fallback_search_surfaces_theme_doc(
        self, library: Library, monkeypatch,
    ) -> None:
        """Plan §5.6 acceptance: ariadne_search('retry') surfaces the theme.

        Drives the actual SearchMixin._search_uncached path with embedding
        ranking forced to fail (so the title-text fallback runs), and asserts
        a theme doc with 'retry' in its title comes back in the results.
        """
        from ariadne_mcp.service_search import SearchMixin

        # Theme docs are cross-source by design (source_name=NULL) —
        # the chokepoint admits them regardless of closure. No
        # ``source_name=...`` here; that's the production shape.
        library.add_document(
            content_type='theme',
            title='Retry Logic with Exponential Backoff',
            content='# Retry Logic\n\nRetry-related explanation.\n',
            source_files=[],
            embedding=_unit([1, 0, 0, 0, 0, 0, 0, 0]),
            metadata={},
            doc_id='theme-retry',
        )

        # Add some other theme docs that should NOT match.
        library.add_document(
            content_type='theme',
            title='Database Migrations',
            content='# DB Migrations\n',
            source_files=[],
            embedding=_unit([0, 1, 0, 0, 0, 0, 0, 0]),
            metadata={},
            doc_id='theme-db',
        )

        # Build a minimal search service via the mixin.
        class _FailEmbedder:
            async def embed(self, text):
                raise RuntimeError('force text fallback')

        class _Svc(SearchMixin):
            @staticmethod
            def _cache_key(*args, **kwargs):
                return hash((args, tuple(sorted(kwargs.items()))))

            def get_branch(self):
                return None

            def _resolve_scope(self, source):
                from scope_resolution import make_scoped_library
                return make_scoped_library(self.config, self.library, source)

        from config import get_config
        svc = _Svc()
        svc.library = library
        svc.config = get_config()
        svc._query_cache = {}
        svc.embedding_service = _FailEmbedder()

        response = await svc._search_uncached(
            query='retry', limit=10, source='test',
        )

        titles = [d.title for d in response.documents]
        assert any('Retry' in t for t in titles), (
            f"theme doc not surfaced for 'retry'; got titles: {titles}"
        )
