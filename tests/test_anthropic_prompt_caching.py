"""Tests for Anthropic prompt-caching wiring.

The provider must mark the (static, per-doc-type) system prompt as cacheable
so consecutive calls within a generation run reuse the cached prefix at ~10%
of the base input cost.

What we verify here:
  - The HTTP payload sent to ``/v1/messages`` puts ``system`` as a list of
    content blocks (not a flat string), with ``cache_control={"type": "ephemeral"}``
    on the static block.
  - User content stays in ``messages`` and is NOT marked cacheable.
  - The ``anthropic-beta: prompt-caching-2024-07-31`` header is present so the
    pinned ``anthropic-version: 2023-06-01`` API accepts the marker.
  - When the system prompt is empty, no cache marker is emitted (the cache
    block disappears so we don't send a malformed empty content block).
  - When ``cache_system_prompt=False`` is passed, the payload reverts to the
    legacy flat-string ``system`` shape (rollback path if caching misbehaves).
"""
from __future__ import annotations

from typing import Any

import pytest


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    """Captures the kwargs of the last ``.post`` call so tests can inspect them."""

    def __init__(self):
        self.last_url: str | None = None
        self.last_json: dict[str, Any] | None = None
        self.headers: dict[str, str] = {}

    async def post(self, url: str, json: dict[str, Any]):
        self.last_url = url
        self.last_json = json
        return _FakeResponse({
            'content': [{'type': 'text', 'text': 'OK'}],
            'usage': {
                'input_tokens': 50,
                'output_tokens': 10,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
            },
        })

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_client(monkeypatch):
    """Replace AnthropicProvider's httpx client with our capture fake."""
    from docgen.llm.anthropic import AnthropicProvider

    fc = _FakeClient()
    monkeypatch.setattr(
        AnthropicProvider,
        '_get_client',
        lambda self: fc,
    )
    return fc


# ---------------------------------------------------------------------------
# Default behavior: caching enabled
# ---------------------------------------------------------------------------


async def test_system_prompt_sent_as_cacheable_block(fake_client):
    """A non-empty system prompt is sent as one content block with cache_control."""
    from docgen.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    await provider.call(
        system_prompt='You are an expert documentation writer.',
        user_prompt='Document this file.',
    )

    payload = fake_client.last_json
    assert payload is not None
    sys_field = payload['system']
    assert isinstance(sys_field, list), 'system must be a list of blocks for caching'
    assert len(sys_field) == 1
    block = sys_field[0]
    assert block['type'] == 'text'
    assert block['text'] == 'You are an expert documentation writer.'
    assert block.get('cache_control') == {'type': 'ephemeral'}


async def test_user_prompt_not_marked_cacheable(fake_client):
    """The per-call user message must NOT carry cache_control."""
    from docgen.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    await provider.call(
        system_prompt='static',
        user_prompt='dynamic user content per-file',
    )

    payload = fake_client.last_json
    msgs = payload['messages']
    assert len(msgs) == 1
    msg = msgs[0]
    assert msg['role'] == 'user'
    # Either a plain string (legacy shape) or content blocks without cache_control.
    if isinstance(msg['content'], list):
        for block in msg['content']:
            assert 'cache_control' not in block, (
                'user content must not be marked cacheable — it changes every call'
            )
    else:
        assert isinstance(msg['content'], str)


async def test_prompt_caching_beta_header_present():
    """The provider must advertise the prompt-caching beta so the pinned
    ``anthropic-version: 2023-06-01`` accepts cache_control markers.
    """
    from docgen.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    client = provider._get_client()
    headers = dict(client.headers)
    assert 'anthropic-beta' in headers, (
        'cache_control on system content requires the prompt-caching beta header'
    )
    assert 'prompt-caching' in headers['anthropic-beta']


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_empty_system_prompt_omits_cache_block(fake_client):
    """If system_prompt is empty, we shouldn't send an empty cache block."""
    from docgen.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    await provider.call(system_prompt='', user_prompt='user')

    payload = fake_client.last_json
    sys_field = payload.get('system')
    # Either omitted entirely, or an empty list — never a [{"text": "", ...}] block.
    if sys_field is not None:
        if isinstance(sys_field, list):
            assert sys_field == [], 'empty system prompt must not produce a cache block'
        else:
            assert sys_field == ''


async def test_cache_system_prompt_false_uses_flat_string(fake_client):
    """``cache_system_prompt=False`` is the rollback path: legacy flat-string system."""
    from docgen.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    await provider.call(
        system_prompt='static',
        user_prompt='user',
        cache_system_prompt=False,
    )

    payload = fake_client.last_json
    assert payload['system'] == 'static', (
        'with caching disabled, system must revert to a flat string'
    )


# ---------------------------------------------------------------------------
# Usage telemetry
# ---------------------------------------------------------------------------


async def test_usage_logging_captures_cache_token_counts(fake_client, caplog):
    """Cache creation / read counts from the response must be logged so we can
    verify caching actually happened in production runs.
    """
    import logging

    from docgen.llm.anthropic import AnthropicProvider

    # Override fake response to simulate a cache hit.
    async def post_with_cache_read(url, json):
        return _FakeResponse({
            'content': [{'type': 'text', 'text': 'OK'}],
            'usage': {
                'input_tokens': 5,
                'output_tokens': 10,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 1024,
            },
        })

    fake_client.post = post_with_cache_read

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    with caplog.at_level(logging.INFO, logger='docgen.llm.anthropic'):
        await provider.call(system_prompt='s', user_prompt='u')

    log_text = ' '.join(r.message for r in caplog.records)
    assert 'cache_read' in log_text or 'cache_read_input_tokens' in log_text, (
        f'cache token counts must appear in logs; got: {log_text!r}'
    )
