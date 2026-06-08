"""Tests for graceful quota-exhaustion handling.

Anthropic returns HTTP 429 for both per-minute rate limits AND monthly/credit
quota exhaustion. The provider must distinguish them: rate-limits are
transient (keep retrying with exponential backoff); quota exhaustion is
fatal for this run (abort and let the orchestrator surface a resume hint).

The signal we use is the error message + error type fields in the 429 body.
Anthropic's wording: rate-limit errors mention "rate limit" / "tokens per
minute"; quota-exhausted errors mention "credit", "monthly", or "billing".
"""
from __future__ import annotations

import json

import httpx
import pytest

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body or {}
        self.text = json.dumps(self._body)

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            request = httpx.Request('POST', 'https://api.anthropic.com/v1/messages')
            response = httpx.Response(
                status_code=self.status_code,
                request=request,
                content=self.text.encode(),
            )
            raise httpx.HTTPStatusError(
                'error', request=request, response=response,
            )

    def json(self) -> dict:
        return self._body


class _ScriptedClient:
    """Returns the next response from ``responses`` per call."""

    def __init__(self, responses: list[_FakeResponse]):
        self.responses = list(responses)
        self.calls = 0
        self.headers: dict[str, str] = {}

    async def post(self, url: str, json: dict):
        self.calls += 1
        return self.responses.pop(0) if self.responses else _FakeResponse(500)

    async def get(self, url: str):
        self.calls += 1
        return self.responses.pop(0) if self.responses else _FakeResponse(500)

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Detection — quota vs rate-limit
# ---------------------------------------------------------------------------


async def test_quota_exhausted_message_raises_quota_error(monkeypatch):
    """A 429 with a quota/credit/billing message must raise QuotaExhaustedError
    immediately — no retry loop, since further requests will fail identically.
    """
    from docgen.llm.anthropic import AnthropicProvider, QuotaExhaustedError

    quota_body = {
        'type': 'error',
        'error': {
            'type': 'rate_limit_error',
            'message': (
                'You have exceeded your monthly token limit for this organization. '
                'Please upgrade your plan or wait until next billing cycle.'
            ),
        },
    }
    sc = _ScriptedClient([_FakeResponse(429, quota_body)])
    monkeypatch.setattr(AnthropicProvider, '_get_client', lambda self: sc)

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    with pytest.raises(QuotaExhaustedError) as excinfo:
        await provider.call(system_prompt='s', user_prompt='u')

    assert 'monthly' in str(excinfo.value).lower() or 'quota' in str(excinfo.value).lower()
    # Must NOT have retried — quota errors are fatal for the run.
    assert sc.calls == 1


async def test_workspace_usage_limit_400_raises_quota_error(monkeypatch):
    """A maxed workspace usage cap comes back as HTTP 400
    ``invalid_request_error`` (NOT 429). It's still a hard, non-retryable
    account cap, so the provider must classify it as QuotaExhaustedError —
    letting callers fail gracefully instead of dumping a raw httpx traceback.
    """
    from docgen.llm.anthropic import AnthropicProvider, QuotaExhaustedError

    body = {
        'type': 'error',
        'error': {
            'type': 'invalid_request_error',
            'message': (
                'You have reached your specified workspace API usage limits. '
                'You will regain access on 2026-07-01 at 00:00 UTC.'
            ),
        },
    }
    sc = _ScriptedClient([_FakeResponse(400, body)])
    monkeypatch.setattr(AnthropicProvider, '_get_client', lambda self: sc)

    provider = AnthropicProvider(model='claude-opus-4-8', api_key='test')
    with pytest.raises(QuotaExhaustedError) as excinfo:
        await provider.call(system_prompt='s', user_prompt='u')

    m = str(excinfo.value).lower()
    assert 'usage limit' in m or 'workspace' in m
    assert sc.calls == 1   # hard cap → no retry


async def test_genuine_400_is_not_misclassified_as_quota(monkeypatch):
    """A real bad-payload 400 must still surface as HTTPStatusError (not
    QuotaExhaustedError) so the request bug stays diagnosable."""
    from docgen.llm.anthropic import AnthropicProvider, QuotaExhaustedError

    body = {
        'type': 'error',
        'error': {
            'type': 'invalid_request_error',
            'message': 'max_tokens: must be greater than 0',
        },
    }
    sc = _ScriptedClient([_FakeResponse(400, body)])
    monkeypatch.setattr(AnthropicProvider, '_get_client', lambda self: sc)

    provider = AnthropicProvider(
        model='claude-opus-4-8', api_key='test', retry_delay=0,
    )
    with pytest.raises(httpx.HTTPStatusError):
        await provider.call(system_prompt='s', user_prompt='u')
    assert sc.calls == 1   # 400 is non-retryable


async def test_rate_limit_per_minute_retries_not_aborts(monkeypatch):
    """A 429 with a transient rate-limit message must trigger normal retry
    backoff (no QuotaExhaustedError raised).
    """
    from docgen.llm.anthropic import AnthropicProvider

    transient_body = {
        'type': 'error',
        'error': {
            'type': 'rate_limit_error',
            'message': (
                'Number of request tokens has exceeded your per-minute rate limit. '
                'Please retry your request.'
            ),
        },
    }
    success_body = {
        'content': [{'type': 'text', 'text': 'OK'}],
        'usage': {
            'input_tokens': 5, 'output_tokens': 5,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        },
    }
    sc = _ScriptedClient([
        _FakeResponse(429, transient_body),
        _FakeResponse(200, success_body),
    ])
    monkeypatch.setattr(AnthropicProvider, '_get_client', lambda self: sc)

    provider = AnthropicProvider(
        model='claude-opus-4-6', api_key='test', retry_delay=0,
    )
    # Should NOT raise QuotaExhaustedError; should succeed via retry.
    result = await provider.call(system_prompt='s', user_prompt='u')

    assert result == 'OK'
    assert sc.calls == 2


async def test_quota_error_after_retry_window_still_classified(monkeypatch):
    """Even after exhausting all retry attempts, if the final error is a
    quota exhaustion, raise QuotaExhaustedError (not a generic error).
    """
    from docgen.llm.anthropic import AnthropicProvider, QuotaExhaustedError

    body = {
        'type': 'error',
        'error': {
            'type': 'rate_limit_error',
            'message': (
                'Your credit balance is too low to use this model. '
                'Please add credits to continue.'
            ),
        },
    }
    sc = _ScriptedClient([
        _FakeResponse(429, body), _FakeResponse(429, body), _FakeResponse(429, body),
    ])
    monkeypatch.setattr(AnthropicProvider, '_get_client', lambda self: sc)

    provider = AnthropicProvider(
        model='claude-opus-4-6', api_key='test', retry_delay=0,
    )
    with pytest.raises(QuotaExhaustedError):
        await provider.call(system_prompt='s', user_prompt='u')

    # Even though the credit-low message is unambiguous, the provider should
    # short-circuit on first detection rather than retry. So calls == 1.
    assert sc.calls == 1
