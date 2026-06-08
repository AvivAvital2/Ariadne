"""Tests for docgen.themes (Themes plan, Phase 4).

Covers theme summarization: coherent-response writes the placeholder theme
document, INCOHERENT response marks the theme accordingly, and
generate_themes batches dirty themes.
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
        content=f'function {doc_id}() — body',
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
    fixture docs. The contract under test is theme summarization;
    source naming is environmental."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'test')


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'themes-phase4-test.db')
    yield lib
    lib.close()


@pytest.fixture
def mocked_embedding(monkeypatch):
    """Stub OpenAI embedding API so writer.update_document_embedding makes no network call."""
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


def _coherent_response_for(cluster_label: str) -> str:
    return f'''# Theme: {cluster_label}

## What this is
A cluster of related code elements that share a common theme.

## Why this is a coherent theme
They all participate in the {cluster_label} concern.

## Key participants
- **member1** — does part of the {cluster_label} work
- **member2** — does another part

## Cross-cutting concerns
None apparent.

## Caveats
None apparent.
'''


def _incoherent_response() -> str:
    return 'INCOHERENT\n\nThe cluster is algorithmic noise — no shared concern.'


# ---------------------------------------------------------------------------
# Single theme summarization
# ---------------------------------------------------------------------------


class TestSummarizeTheme:
    @pytest.mark.asyncio
    async def test_coherent_response_writes_doc_and_clears_dirty(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        # Set up clusters via Phase 2/3.
        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        async def fake_chat(messages, *, model=None, **kwargs):
            return _coherent_response_for('A')

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        # Pick one cluster to summarize.
        all_themes = library.list_themes(coherent_only=False)
        cluster_id = all_themes[0].cluster_id
        original_doc_id = all_themes[0].doc_id

        async with LibraryWriter(library) as writer:
            result = await themes.summarize_theme(library, writer, cluster_id)

        assert result is True
        # Theme is now clean and coherent.
        theme = library.get_theme(cluster_id)
        assert theme is not None
        assert theme.coherent is True
        assert theme.dirty is False
        # The placeholder doc was updated with real content.
        doc = library.get_document(original_doc_id)
        assert doc is not None
        assert doc.content.startswith('# Theme')
        assert '(pending summarization)' not in doc.content
        # summary_hash was set to non-empty.
        assert theme.summary_hash != ''

    @pytest.mark.asyncio
    async def test_incoherent_response_marks_theme_and_skips_doc(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        async def fake_chat(messages, *, model=None, **kwargs):
            return _incoherent_response()

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        all_themes = library.list_themes(coherent_only=False)
        cluster_id = all_themes[0].cluster_id

        async with LibraryWriter(library) as writer:
            result = await themes.summarize_theme(library, writer, cluster_id)

        assert result is False
        theme = library.get_theme(cluster_id)
        assert theme is not None
        assert theme.coherent is False
        # No longer "dirty" because we processed it (just judged it INCOHERENT).
        assert theme.dirty is False


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestGenerateThemes:
    @pytest.mark.asyncio
    async def test_processes_only_dirty_themes(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        # Mark one theme clean to simulate "already-summarized".
        all_themes = library.list_themes(coherent_only=False)
        assert len(all_themes) == 2
        library.mark_themes_clean([all_themes[0].cluster_id])

        call_count = 0

        async def fake_chat(messages, *, model=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return _coherent_response_for('X')

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        async with LibraryWriter(library) as writer:
            summary = await themes.generate_themes(library, writer)

        # Only the still-dirty theme triggered an LLM call.
        assert call_count == 1
        assert summary['summarized'] == 1
        # Both should be clean now.
        assert library.get_dirty_themes() == []

    @pytest.mark.asyncio
    async def test_no_dirty_themes_means_no_llm_calls(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes

        call_count = 0

        async def fake_chat(messages, *, model=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return _coherent_response_for('X')

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        async with LibraryWriter(library) as writer:
            summary = await themes.generate_themes(library, writer)

        assert call_count == 0
        assert summary['summarized'] == 0

    @pytest.mark.asyncio
    async def test_api_usage_cap_stops_themes_gracefully(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        """When the Anthropic API cap is hit mid-phase, themes stop
        immediately (no point hammering a maxed cap), surface it via
        ``quota_exhausted`` + a message, and do NOT record per-cluster
        failures or spew tracebacks."""
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges
        from docgen.llm.anthropic import WorkspaceUsageLimitError

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)
        assert len(library.get_dirty_themes()) == 2

        calls = 0

        async def fake_chat(messages, *, model=None, **kwargs):
            nonlocal calls
            calls += 1
            raise WorkspaceUsageLimitError(
                'You have reached your specified workspace API usage limits. '
                'You will regain access on 2026-07-01 at 00:00 UTC.'
            )

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        async with LibraryWriter(library) as writer:
            summary = await themes.generate_themes(
                library, writer, concurrency=1,
            )

        assert calls == 1                        # stopped at the cap, didn't hammer
        assert summary.get('quota_exhausted') is True
        assert 'usage limit' in (summary.get('quota_message') or '').lower()
        assert summary['summarized'] == 0
        assert summary['failed'] == 0            # skipped due to cap, not a failure


# ---------------------------------------------------------------------------
# Prompt construction & summary_hash
# ---------------------------------------------------------------------------


class TestPromptContent:
    @pytest.mark.asyncio
    async def test_summarize_theme_passes_member_ids_to_llm(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        captured: list[str] = []

        async def fake_chat(messages, *, model=None, **kwargs):
            # Capture the user-message content (the prompt body).
            captured.append(messages[1]['content'])
            return _coherent_response_for('X')

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        all_themes = library.list_themes(coherent_only=False)
        cluster_id = all_themes[0].cluster_id
        member_ids = [
            eid for eid, _ in library.get_theme_members(cluster_id)
        ]

        async with LibraryWriter(library) as writer:
            await themes.summarize_theme(library, writer, cluster_id)

        assert captured, 'chat_complete was not called'
        prompt = captured[0]
        for mid in member_ids:
            assert mid in prompt, f'member id {mid!r} not in prompt body'


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_generate_themes_counts_chat_failures(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        async def failing_chat(messages, *, model=None, **kwargs):
            raise RuntimeError('LLM unavailable')

        monkeypatch.setattr('docgen.themes.chat_complete', failing_chat)

        async with LibraryWriter(library) as writer:
            summary = await themes.generate_themes(library, writer)

        # Two dirty themes, both throw → both counted as failed, none summarized.
        assert summary['failed'] == summary['total_dirty']
        assert summary['summarized'] == 0
        assert summary['incoherent'] == 0


class TestCoherenceTransitions:
    @pytest.mark.asyncio
    async def test_incoherent_then_coherent_flips_back(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        cluster_id = library.list_themes(coherent_only=False)[0].cluster_id

        async def fake_incoherent(messages, *, model=None, **kwargs):
            return _incoherent_response()

        async def fake_coherent(messages, *, model=None, **kwargs):
            return _coherent_response_for('X')

        # First pass: LLM judges incoherent.
        monkeypatch.setattr('docgen.themes.chat_complete', fake_incoherent)
        library.mark_theme_dirty(cluster_id)
        async with LibraryWriter(library) as writer:
            await themes.summarize_theme(library, writer, cluster_id)
        theme = library.get_theme(cluster_id)
        assert theme is not None and theme.coherent is False

        # Second pass: LLM judges coherent.
        monkeypatch.setattr('docgen.themes.chat_complete', fake_coherent)
        library.mark_theme_dirty(cluster_id)
        async with LibraryWriter(library) as writer:
            await themes.summarize_theme(library, writer, cluster_id)
        theme = library.get_theme(cluster_id)
        assert theme is not None and theme.coherent is True


class TestMaxCallsAndSummaryHashIntegrity:
    @pytest.mark.asyncio
    async def test_max_calls_limits_chat_invocations(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        call_count = 0

        async def fake_chat(messages, *, model=None, **kwargs):
            nonlocal call_count
            call_count += 1
            return _coherent_response_for('X')

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        async with LibraryWriter(library) as writer:
            summary = await themes.generate_themes(library, writer, max_calls=1)

        assert call_count == 1
        assert summary['summarized'] == 1
        # The unprocessed dirty theme remains dirty.
        assert len(library.get_dirty_themes()) == 1

    @pytest.mark.asyncio
    async def test_summary_hash_matches_compute_after_summarize(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        """The hash stored on the theme after summarize_theme should equal
        compute_summary_hash(members, summaries, resolution).
        """
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)

        async def fake_chat(messages, *, model=None, **kwargs):
            return _coherent_response_for('X')

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        cluster_id = library.list_themes(coherent_only=False)[0].cluster_id

        # Pre-compute what the hash should be using the same gather logic.
        members = library.get_theme_members(cluster_id)
        member_ids = sorted(eid for eid, _ in members)
        summaries: list[str] = []
        for mid in member_ids:
            doc = library.get_document(mid)
            if doc is None:
                summaries.append('')
                continue
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            desc = meta.get('description')
            summaries.append(
                str(desc)[:300] if desc else (doc.content or '')[:300]
            )

        async with LibraryWriter(library) as writer:
            await themes.summarize_theme(library, writer, cluster_id)

        theme_after = library.get_theme(cluster_id)
        expected = themes.compute_summary_hash(
            member_ids, summaries, theme_after.resolution,
        )
        assert theme_after.summary_hash == expected


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_summarize_theme_missing_cluster_returns_false(
        self, library: Library, mocked_embedding, monkeypatch,
    ) -> None:
        from docgen import themes

        async def fake_chat(messages, **kwargs):
            return _coherent_response_for('X')

        monkeypatch.setattr('docgen.themes.chat_complete', fake_chat)

        async with LibraryWriter(library) as writer:
            result = await themes.summarize_theme(library, writer, 'no-such-cluster')

        assert result is False


class TestSummaryHash:
    def test_summary_hash_stable_for_same_inputs(self) -> None:
        from docgen.themes import compute_summary_hash

        h1 = compute_summary_hash(
            members=['a', 'b', 'c'],
            summaries=['s_a', 's_b', 's_c'],
            resolution=1.0,
        )
        h2 = compute_summary_hash(
            members=['c', 'b', 'a'],  # different order, same content
            summaries=['s_c', 's_b', 's_a'],
            resolution=1.0,
        )
        assert h1 == h2

    def test_summary_hash_changes_when_summary_changes(self) -> None:
        from docgen.themes import compute_summary_hash

        h1 = compute_summary_hash(
            members=['a'], summaries=['original'], resolution=1.0,
        )
        h2 = compute_summary_hash(
            members=['a'], summaries=['changed'], resolution=1.0,
        )
        assert h1 != h2

    def test_summary_hash_changes_when_resolution_changes(self) -> None:
        from docgen.themes import compute_summary_hash

        h1 = compute_summary_hash(members=['a'], summaries=['s'], resolution=1.0)
        h2 = compute_summary_hash(members=['a'], summaries=['s'], resolution=1.5)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Batched theme summarization (cost fix: live chat_complete per theme was an
# unestimated, full-price, post-completion spike — route through the batch
# API instead, like catalog-describe).
# ---------------------------------------------------------------------------


class _FakeBatchProvider:
    """Records a single batch submission and returns coherent responses,
    standing in for AnthropicProvider's batch surface."""

    def __init__(self, responder):
        self._responder = responder
        self.submitted: list[list] = []
        self._last: dict = {}

    async def submit_batch(self, requests):
        from docgen.llm.anthropic import BatchSubmission
        reqs = list(requests)
        self.submitted.append(reqs)
        self._last = {r.custom_id: r for r in reqs}
        return BatchSubmission(batch_id='thm_batch_1')

    async def poll_batch(self, batch_id, *, on_progress=None, **kwargs):
        if on_progress is not None:
            on_progress(0, len(self._last), 0)
        return None

    async def fetch_batch_results(self, batch_id):
        return {cid: self._responder(req) for cid, req in self._last.items()}


class TestGenerateThemesBatched:
    @pytest.mark.asyncio
    async def test_batched_summarizes_all_dirty_in_one_submit(
        self, library: Library, mocked_embedding,
    ) -> None:
        from docgen import themes
        from docgen.cluster import cluster_themes
        from docgen.graph_builder import build_semantic_edges

        _populate_two_clusters(library)
        build_semantic_edges(library, k=5, min_sim=0.6)
        cluster_themes(library, min_cluster_size=3)
        dirty_before = library.get_dirty_themes()
        assert len(dirty_before) >= 1, 'fixture should produce dirty themes'

        provider = _FakeBatchProvider(lambda req: _coherent_response_for('X'))
        async with LibraryWriter(library) as writer:
            result = await themes.generate_themes_batched(
                library, writer, provider,
            )

        # One batch submission covering every dirty theme — not N live calls.
        assert len(provider.submitted) == 1
        assert len(provider.submitted[0]) == len(dirty_before)
        assert result['summarized'] == len(dirty_before)
        assert result['total_dirty'] == len(dirty_before)
        # All themes now clean + coherent, just like the live path.
        assert library.get_dirty_themes() == []
