"""Offline safeguards for a bounded replay of actual application call prompts."""

import asyncio
import json

import httpx
import pytest

from tests import live_mac_call_probe as probe
from tests.live_model_smoke import Budget


def case(name='selector', cap=64, timeout=30):
    return {'name': name, 'system_prompt': '', 'user_prompt': 'Select family alpha.',
            'max_tokens': cap, 'timeout_seconds': timeout,
            'expected': {'private_oracle': 'NEVER SEND THIS EXPECTATION'}}


def response(*, finish='stop', content='["alpha"]', completion=20):
    return {'model': 'Qwen/Qwen3.8-27B', 'choices': [{'index': 0,
            'finish_reason': finish, 'message': {'role': 'assistant', 'content': content,
            'reasoning_content': 'PRIVATE REASONING'}}], 'usage': {
            'prompt_tokens': 50, 'completion_tokens': completion, 'total_tokens': 50 + completion}}


async def test_frozen_payload_timeout_empty_system_and_final_only_archive(tmp_path):
    output = tmp_path / 'run'
    calls = []
    frozen = []

    def serve(request):
        frozen.append((output / 'protocol.json').read_bytes())
        payload = json.loads(request.content)
        assert payload['max_tokens'] == 64
        assert payload['messages'][0] == {'role': 'system', 'content': ''}
        assert payload['reasoning_effort'] == 'xhigh'
        assert payload['chat_template_kwargs'] == {'enable_thinking': True}
        assert payload['repetition_penalty'] == 1
        assert 'reasoning_format' not in payload and 'repeat_penalty' not in payload
        assert 'NEVER SEND' not in json.dumps(payload)
        assert set(request.extensions['timeout'].values()) == {30}
        calls.append(payload)
        return httpx.Response(200, json=response())

    result = await probe.execute({'cases': [case()]}, output, Budget(.5), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert result['status'] == 'completed_pending_review'
    assert result['quality_verdict'] == 'unreviewed'
    assert result['cases'][0]['status'] == 'completed'
    assert result['cases'][0]['content'] == '["alpha"]'
    assert len(calls) == 1 and (output / 'protocol.json').read_bytes() == frozen[0]
    for path in output.rglob('*'):
        if path.is_file() and 'provenance' not in path.parts:
            assert 'PRIVATE REASONING' not in path.read_text()
            assert 'test-secret' not in path.read_text()


async def test_truncation_is_recorded_without_retry_and_next_case_is_examined(tmp_path):
    calls = []

    def serve(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=response(finish='length', content=None, completion=64)
                              if len(calls) == 1 else response())

    result = await probe.execute({'cases': [case(), case('catalog', 2048, 600)]},
                                 tmp_path / 'run', Budget(.5), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert len(calls) == 2
    assert result['status'] == 'completed_with_failures'
    assert [c['status'] for c in result['cases']] == ['failed', 'completed']
    assert result['cases'][0]['finish_reason'] == 'length'
    assert result['cases'][0]['usage']['completion_tokens'] == 64
    assert 'content' not in result['cases'][0]
    assert result['accounted_upper_bound_usd'] > 0


async def test_timeout_preserves_reservation_and_does_not_repeat_content(tmp_path):
    calls = []

    def serve(request):
        calls.append(request)
        raise httpx.ReadTimeout('secret provider error', request=request)

    result = await probe.execute({'cases': [case()]}, tmp_path / 'run', Budget(.5),
                                 api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert len(calls) == 1
    assert result['cases'][0]['error_type'] == 'ReadTimeout'
    assert result['accounted_upper_bound_usd'] == result['requests'][0]['reserved_usd']
    assert 'secret provider error' not in json.dumps(result)


async def test_cancel_is_terminal_and_preserves_paid_reservation(tmp_path):
    def serve(_request):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await probe.execute({'cases': [case()]}, tmp_path / 'run', Budget(.5),
                            api_key='test-secret',
                            transport_factory=lambda: httpx.MockTransport(serve))
    result = json.loads((tmp_path / 'run/report.json').read_text())
    assert result['status'] == 'cancelled'
    assert result['accounted_upper_bound_usd'] > 0


@pytest.mark.parametrize('changes', [
    {'max_tokens': 8193}, {'max_tokens': True}, {'timeout_seconds': 601},
    {'timeout_seconds': 0}, {'name': '../outside'}, {'user_prompt': ''},
    {'user_prompt': 'x' * 33000}, {'native_request': {'model': 'other'}},
    {'hosted_request': {'model': 'other'}},
])
async def test_invalid_case_rejected_before_any_paid_call(tmp_path, changes):
    with pytest.raises(ValueError, match=r'Invalid captured|Captured payload|Conservative context'):
        await probe.execute({'cases': [{**case(), **changes}]}, tmp_path / 'run',
                            Budget(.5), api_key='test-secret',
                            transport_factory=lambda: pytest.fail('Network instantiated'))


async def test_budget_refuses_call_before_network(tmp_path):
    result = await probe.execute({'cases': [case()]}, tmp_path / 'run', Budget(.0000001),
                                 api_key='test-secret', transport_factory=lambda: httpx.MockTransport(
                                     lambda _request: pytest.fail('Unaffordable request sent')))
    assert result['status'] == 'budget_exhausted'
    assert result['requests'] == []


async def test_larger_probe_budget_is_not_allowed(tmp_path):
    with pytest.raises(ValueError, match='At most'):
        await probe.execute({'cases': [case()]}, tmp_path / 'run', Budget(1.01),
                            api_key='test-secret')
