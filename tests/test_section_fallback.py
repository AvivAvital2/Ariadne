"""Section fallback must provide prose, not just a Markdown title."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from ariadne_mcp.models import SearchResponse
from ariadne_mcp.service_search import SearchMixin
from schema import Section


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
async def test_old_persistent_section_result_is_not_reused(tmp_path):
    service = SearchMixin()
    service._query_cache = {}
    service.config = SimpleNamespace(to_dict=lambda: {}, config_dir=tmp_path)
    service._cache_key = lambda *args: args
    service.library = Mock()
    old_key = service._persistent_cache_key(
        'search',
        None,
        'question',
        None,
        None,
        10,
        None,
        True,
        'developer',
        'synthetic',
        None,
    )
    cached = {old_key: SearchResponse(event_id=1).model_dump_json()}
    service.library.cache_get.side_effect = cached.get
    fresh = SearchResponse(event_id=2)
    service._search_uncached = AsyncMock(return_value=fresh)
    result = await service.search(query='question', sections_only=True, source='synthetic')
    assert result is fresh
    service._search_uncached.assert_awaited_once()
    service.library.cache_put.assert_called_once()
