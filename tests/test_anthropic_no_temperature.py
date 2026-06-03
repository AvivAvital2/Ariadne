"""Tests pinning the removal of ``temperature`` from the LLM stack.

Background: ``claude-opus-4-7`` (and the modern Claude line generally) reject
the ``temperature`` parameter on ``/v1/messages`` with::

    400 invalid_request_error: `temperature` is deprecated for this model.

Ariadne previously sent ``temperature`` unconditionally, so every doc-gen
request 400'd. ``temperature`` is removed completely — not made optional —
because no currently-targeted model accepts it.

These tests guardrail that removal across the surface that touched it:
  - the sync ``/v1/messages`` payload carries no ``temperature`` key
  - ``AnthropicProvider.call`` no longer accepts a ``temperature`` kwarg
  - ``BatchRequest`` has no ``temperature`` field
  - ``_build_batch_params`` emits no ``temperature`` in the batch ``params``

They are RED against the current code (temperature is present everywhere)
and go GREEN once the parameter is fully removed.
"""
from __future__ import annotations

import inspect
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
    """Captures the JSON body of the last ``.post`` so tests can inspect it."""

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
    from docgen.llm.anthropic import AnthropicProvider

    fc = _FakeClient()
    monkeypatch.setattr(AnthropicProvider, '_get_client', lambda self: fc)
    return fc


async def test_call_payload_omits_temperature(fake_client):
    """The /v1/messages body must not contain a ``temperature`` key —
    the model 400s on it."""
    from docgen.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider(model='claude-opus-4-7', api_key='test')
    await provider.call(system_prompt='static', user_prompt='document this')

    payload = fake_client.last_json
    assert payload is not None
    assert 'temperature' not in payload, (
        f'temperature must not be sent — it 400s the request; payload keys: '
        f'{sorted(payload)}'
    )


def test_call_signature_has_no_temperature_param():
    """``AnthropicProvider.call`` must not expose a ``temperature`` kwarg —
    callers that still pass it should fail loudly, not silently re-introduce
    the rejected field."""
    from docgen.llm.anthropic import AnthropicProvider

    params = inspect.signature(AnthropicProvider.call).parameters
    assert 'temperature' not in params, (
        f'call() still accepts temperature: {list(params)}'
    )


def test_batch_request_has_no_temperature_field():
    """``BatchRequest`` must not carry a ``temperature`` field."""
    from docgen.llm.anthropic import BatchRequest

    req = BatchRequest(
        custom_id='c1', system_prompt='sys', user_prompt='usr',
    )
    assert not hasattr(req, 'temperature'), (
        'BatchRequest still has a temperature field'
    )


def test_build_batch_params_omits_temperature():
    """The per-request ``params`` block in a batch submission must not
    contain ``temperature`` — batch rows 400 on it just like sync calls."""
    from docgen.llm.anthropic import AnthropicProvider, BatchRequest
    from docgen.llm.anthropic_batch import AnthropicBatchStrategy

    provider = AnthropicProvider(model='claude-opus-4-7', api_key='test')
    req = BatchRequest(custom_id='c1', system_prompt='sys', user_prompt='usr')
    # _build_batch_params moved onto the strategy in the batch-strategy refactor.
    params = AnthropicBatchStrategy(provider)._build_batch_params(req)

    assert 'temperature' not in params, (
        f'batch params must not send temperature; keys: {sorted(params)}'
    )
