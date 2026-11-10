"""Section fallback must provide prose, not just a Markdown title."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from ariadne_mcp.models import DocumentResult, SearchResponse
from ariadne_mcp.service_search import SearchMixin
from schema import Document, Section


@pytest.mark.parametrize(
    ('introduction', 'score', 'expected'),
    [
        ('# A manual\n\n', 0.2, 'Dependencies are loaded transitively.'),
        ('# A manual\n\nA useful overview.', 0.2, 'A useful overview.'),
        ('# A manual\n\nA useful overview.', 0.8, 'Dependencies are loaded transitively.'),
    ],
    ids=['skip-title-only', 'retain-overview-fallback', 'retain-threshold-selection'],
)
@pytest.mark.asyncio
async def test_section_fallback_contains_body(introduction, score, expected):
    sections = [
        Section(document_id='doc', index=0, heading='Introduction', content=introduction),
        Section(
            document_id='doc',
            index=1,
            heading='Configuration',
            content='## Configuration\n\nDependencies are loaded transitively.',
        ),
    ]
    vectors = [np.array([value, np.sqrt(1 - value**2)], dtype=np.float32) for value in (0.1, score)]
    service = SearchMixin()
    service.library = Mock()
    service.library.get_sections.return_value = sections
    service.library.get_section_embeddings_for_doc.return_value = list(enumerate(vectors))
    service.embedding_service = SimpleNamespace(
        embed=AsyncMock(return_value=np.array([1, 0], dtype=np.float32))
    )
    result = await service._select_relevant_sections(['doc'], 'Are dependencies recursive?')
    assert len(result['doc']) == 1
    assert expected in result['doc'][0].content
    assert result['doc'][0].score == pytest.approx(score if score >= 0.3 else 0)


@pytest.mark.asyncio
async def test_repeated_headings_count_every_unreturned_section(tmp_path):
    sections = [
        Section(document_id='doc', index=i, heading=heading, content=f'## {heading}\n{body}')
        for i, (heading, body) in enumerate([
            ('Question', 'First question?'), ('Answer', 'First answer.'),
            ('Question', 'Second question?'), ('Answer', 'Second answer.'),
        ])
    ]
    document = Document(id='doc', title='FAQ', content_type='qa', source_name='app',
                        content='\n\n'.join(section.content for section in sections))
    service = SearchMixin()
    service.config = SimpleNamespace(to_dict=dict, config_dir=tmp_path)
    service.library = Mock()
    service.library.list_documents_lite.return_value = [document]
    service.library.get_documents_batch.return_value = [document]
    service.library.get_sections.return_value = sections
    service.library.get_section_embeddings_for_doc.return_value = [
        (section.index, np.array([0.1, np.sqrt(0.99)], dtype=np.float32)) for section in sections
    ]
    service.library.log_usage.return_value = 1
    service._resolve_scope = Mock(return_value=service.library)
    service._rank_ids_by_embedding = AsyncMock(return_value=[('doc', 0.9)])
    service.embedding_service = SimpleNamespace(
        embed=AsyncMock(return_value=np.array([1, 0], dtype=np.float32))
    )

    result = await service._search_uncached(query='question', sections_only=True, source='app')

    assert len(result.documents) == 1
    assert [section.content for section in result.documents[0].sections] == [sections[0].content]
    assert result.documents[0].content == sections[0].content + '\n\n[... 3 sections omitted ...]'


@pytest.mark.asyncio
async def test_old_persistent_section_result_is_not_reused(tmp_path):
    from spools import resolve_spools

    service = SearchMixin()
    service._query_cache = {}
    service.config = SimpleNamespace(to_dict=dict, config_dir=tmp_path)
    service._cache_key = lambda *args: args
    service.library = Mock()
    spool_fingerprint = resolve_spools(service.config).fingerprint()
    legacy_key = service._persistent_cache_key(
        'search', None, 'question', None, None, 10, None, True,
        'developer', 'app', None,
    )
    legacy_payload = SearchResponse(event_id=1).model_dump_json()
    old_key = service._persistent_cache_key(
        'search-v2', None, 'question', None, None, 10, None, True,
        'developer', 'app', spool_fingerprint, None,
    )
    old = SearchResponse(documents=[DocumentResult(
        id='doc', title='FAQ', content_type='qa',
        content='## Question\nFirst question?\n\n[... 2 sections omitted ...]',
    )], event_id=2)
    cached = {legacy_key: legacy_payload, old_key: old.model_dump_json()}
    service.library.cache_get.side_effect = cached.get
    service.library.cache_put.side_effect = (
        lambda key, _branch, _query, value: cached.__setitem__(key, value)
    )
    corrected = old.model_copy(deep=True)
    corrected.event_id = 3
    corrected.documents[0].content = '## Question\nFirst question?\n\n[... 3 sections omitted ...]'
    service._search_uncached = AsyncMock(return_value=corrected)

    result = await service.search(query='question', sections_only=True, source='app')

    assert result == corrected
    assert cached[legacy_key] == legacy_payload
    assert cached[old_key] == old.model_dump_json()
    assert len(cached) == 3
    # Discard only the in-memory result so the second call exercises persistence.
    service._query_cache.clear()
    reused = await service.search(query='question', sections_only=True, source='app')
    assert reused == corrected
    service._search_uncached.assert_awaited_once()
    service.library.cache_put.assert_called_once()
