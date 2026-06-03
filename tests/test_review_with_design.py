"""Tests for design_doc param on AnalysisMixin.review (T1)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from ariadne_mcp.models import ReviewResponse  # noqa: F401


def _make_host():
    """Build a minimal AnalysisMixin host with mocked deps."""
    from ariadne_mcp.service_analysis import AnalysisMixin
    host = AnalysisMixin()
    host._query_cache = {}
    host._cache_key = lambda *args: str(args)
    host.library = MagicMock()
    host.library.log_usage.return_value = 1
    host.search = AsyncMock(return_value=MagicMock(documents=[]))
    host.review_checklist = MagicMock(return_value=[])
    return host


class TestReviewWithDesignDoc:
    def test_no_design_doc_returns_no_alignment_fields(self):
        host = _make_host()
        result = asyncio.run(host.review(
            task_description='test task',
            changed_files=[],
            design_doc=None,
        ))
        assert result.design_alignment_verdict is None
        assert result.divergence_reasons == []

    def test_design_doc_invokes_arbiter_and_parses_verdict(self):
        host = _make_host()
        llm_response = '{"verdict": "aligned", "reasons": []}'
        with patch('llm.chat_complete', new=AsyncMock(return_value=llm_response)):
            result = asyncio.run(host.review(
                task_description='test task',
                changed_files=[],
                design_doc='## Design\nDo X',
            ))
        assert result.design_alignment_verdict == 'aligned'
        assert result.divergence_reasons == []

    def test_divergent_verdict_captures_reasons(self):
        host = _make_host()
        llm_response = '{"verdict": "divergent", "reasons": ["Missing X", "Wrong Y"]}'
        with patch('llm.chat_complete', new=AsyncMock(return_value=llm_response)):
            result = asyncio.run(host.review(
                task_description='test',
                changed_files=[],
                design_doc='design',
            ))
        assert result.design_alignment_verdict == 'divergent'
        assert 'Missing X' in result.divergence_reasons

    def test_invalid_llm_response_falls_back_to_partial(self):
        host = _make_host()
        with patch('llm.chat_complete', new=AsyncMock(return_value='not json at all')):
            result = asyncio.run(host.review(
                task_description='test',
                changed_files=[],
                design_doc='design',
            ))
        assert result.design_alignment_verdict == 'partial'
        assert result.divergence_reasons
        assert 'invalid JSON' in result.divergence_reasons[0]

    def test_cache_keys_differ_per_design(self):
        """Different design_docs must not share a cache entry."""
        host = _make_host()
        llm_reply = '{"verdict":"aligned","reasons":[]}'
        with patch('llm.chat_complete', new=AsyncMock(return_value=llm_reply)):
            asyncio.run(host.review(task_description='t', changed_files=[], design_doc='A'))
            asyncio.run(host.review(task_description='t', changed_files=[], design_doc='B'))
        assert len(host._query_cache) == 2
