"""OpenAI quota exhaustion must abort the run, not silently return None.

OpenAI uses HTTP 429 for both per-minute rate limits (transient) and
monthly/credit quota exhaustion (fatal). Treating the latter as transient
retries through it on every file, burning the retry budget for a
guaranteed-empty run. The Anthropic provider already distinguishes these and
raises QuotaExhaustedError so the orchestrator aborts; OpenAI must match.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from docgen.llm.anthropic import QuotaExhaustedError
from docgen.llm.openai import OpenAIProvider


def _response(message: str) -> httpx.Response:
    req = httpx.Request('POST', 'https://api.openai.com/v1/chat/completions')
    return httpx.Response(429, json={'error': {'message': message}}, request=req)


def _provider_with(resp: httpx.Response) -> OpenAIProvider:
    class _FakeClient:
        async def post(self, url, json=None):
            return resp

    prov = OpenAIProvider(model='gpt-4o', api_key='x', retry_delay=0.0, max_retries=2)
    prov._client = _FakeClient()
    return prov


def test_openai_quota_429_raises_quota_exhausted():
    # insufficient_quota message → fatal → abort (not retried into a None).
    prov = _provider_with(_response(
        'You exceeded your current quota, please check your plan and billing details.'
    ))
    with pytest.raises(QuotaExhaustedError):
        asyncio.run(prov.call('sys', 'user'))


def test_openai_rate_limit_429_is_still_transient():
    # A per-minute rate limit has no quota keywords → must NOT be treated as
    # fatal; it retries and (here) exhausts retries to None, never aborting.
    prov = _provider_with(_response('Rate limit reached for requests'))
    assert asyncio.run(prov.call('sys', 'user')) is None
