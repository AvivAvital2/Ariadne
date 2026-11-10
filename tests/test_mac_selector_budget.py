"""Task-scoped thinking allowance without changing caller or replay contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from docgen.llm.local import ContextOverflow, build_chat_request
from library.selection_policy import signal_from_usage
from llm import capture_completion_trace, chat_complete
from local_models import mac_accuracy_profile, mac_model_profile
from model_config import resolve_models


@pytest.mark.parametrize('native_pair', ['accuracy'], indirect=True)
@pytest.mark.parametrize(('phase', 'requested'), [
    ('scip-obligation-plan', 384), ('scip-route-family-select', 64),
    ('scip-component-select', 256), ('scip-body-select', 128),
])
async def test_selected_thinking_profile_reserves_extra_tokens_for_selectors(native_pair, phase, requested):
    context, chat, embed = native_pair
    sink = []
    chat.overrides['/native/v1/chat/completions'] = {
        'choices': [{'message': {'content': 'Documented arithmetic.'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 4, 'completion_tokens': requested + 100},
    }
    try:
        with context.activate(), capture_completion_trace() as trace:
            result = await chat_complete(
                system_prompt='Choose only listed IDs.', user_prompt='B1: relevant source body',
                phase=phase, max_tokens=requested, timeout=12.5, usage_sink=sink)
        assert result == 'Documented arithmetic.'
        requests = dict(chat.requests)
        payload = requests['/native/v1/chat/completions']
        assert payload['max_tokens'] == requested + 2048
        assert trace[0]['max_tokens'] == sink[0]['max_tokens'] == requested + 2048
        assert sink[0]['completion_tokens'] == requested + 100
        assert not signal_from_usage(sink[0], max_tokens=requested).truncated
        assert trace[0]['timeout'] == 12.5
        assert payload['chat_template_kwargs'] == {'enable_thinking': True}
        assert payload['reasoning_effort'] == 'xhigh'
        template = requests['/native/apply-template']
        assert template['messages'] == payload['messages']
        assert template['reasoning_effort'] == payload['reasoning_effort']
        assert template['reasoning_format'] == payload['reasoning_format']
        assert embed.requests == []
    finally:
        await context.aclose()


async def _recorded_call(settings, phase, requested):
    calls = []

    class Provider:
        def __init__(self):
            self.last_usage = {'prompt_tokens': 17, 'completion_tokens': 300}

        async def call(self, _system, _user, **kwargs):
            calls.append(kwargs)
            return 'B1'

    context = SimpleNamespace(
        local=settings.local, settings=settings, chat=Provider(),
        validate_producer=lambda **_kwargs: None)
    sink = []
    with capture_completion_trace() as trace:
        assert await chat_complete(user_prompt='Choose a body.', phase=phase,
                                   max_tokens=requested, context=context, usage_sink=sink) == 'B1'
    return calls[0], trace[0], sink[0]


@pytest.mark.parametrize('phase', [
    'scip-family-select', 'scip-route-select', 'scip-symbol-select', 'scip-exact-route-select',
])
async def test_other_explicit_selector_phases_receive_same_allowance(phase):
    settings = resolve_models(mac_accuracy_profile())
    call, trace, usage = await _recorded_call(settings, phase, 256)
    assert call['max_tokens'] == trace['max_tokens'] == usage['max_tokens'] == 2304


@pytest.mark.parametrize('changed', ['legacy', 'hosted', 'generation_revision', 'tokenizer', 'embedding_revision'])
async def test_allowance_requires_exact_selected_local_model_pair_identity(changed):
    settings = resolve_models(mac_accuracy_profile())
    if changed == 'legacy':
        settings = resolve_models(mac_model_profile())
    elif changed == 'hosted':
        settings = replace(settings, chat=replace(settings.chat, provider='deepinfra'))
    elif changed == 'generation_revision':
        settings = replace(settings, chat=replace(settings.chat, model_revision='a' * 40))
    elif changed == 'tokenizer':
        settings = replace(settings, chat=replace(settings.chat, tokenizer_revision='a' * 40))
    else:
        settings = replace(settings, embeddings=replace(settings.embeddings, model_revision='a' * 40))
    call, trace, usage = await _recorded_call(settings, 'scip-body-select', 128)
    assert call['max_tokens'] == trace['max_tokens'] == usage['max_tokens'] == 128


@pytest.mark.parametrize(('phase', 'cap'), [
    ('scip-future-select', 64), ('completion', 4096), ('catalog', 2048),
    ('themes', 2048), ('role-adaptation', 1024), ('trace-flow', 1024),
])
async def test_unknown_and_prose_phases_preserve_their_limits(phase, cap):
    call, trace, usage = await _recorded_call(resolve_models(mac_accuracy_profile()), phase, cap)
    assert call['max_tokens'] == trace['max_tokens'] == usage['max_tokens'] == cap


@pytest.mark.parametrize(('ceiling', 'requested', 'expected'), [
    (8192, 7800, 8192), (8192, 8192, 8192), (1600, 64, 1600),
    (8192, 8193, 8193), (8192, True, True), (8192, 0, 0),
])
async def test_profile_ceiling_and_invalid_requested_caps_are_preserved(ceiling, requested, expected):
    settings = resolve_models(mac_accuracy_profile())
    settings = replace(settings, chat=replace(settings.chat, max_output_tokens=ceiling))
    call, _trace, _usage = await _recorded_call(settings, 'scip-body-select', requested)
    assert call['max_tokens'] == expected
    assert type(call['max_tokens']) is type(expected)


@pytest.mark.parametrize('native_pair', ['accuracy'], indirect=True)
@pytest.mark.parametrize('limit_owner', ['server', 'profile'])
async def test_enlarged_selector_reservation_rejects_small_context_before_generation(native_pair, limit_owner):
    context, chat, embed = native_pair
    if limit_owner == 'server':
        chat.context = 1000  # Original128 fits; new2176 plus prompt/safety does not.
    else:
        context.settings = replace(context.settings, chat=replace(
            context.settings.chat, context_window=3000, max_output_tokens=2500))
        chat.tokens = list(range(900))
    before = context.db_path.read_bytes()
    try:
        with capture_completion_trace() as trace, pytest.raises(ContextOverflow):
            await chat_complete(user_prompt='Choose B1.', max_tokens=128,
                                phase='scip-body-select', context=context)
        assert trace[0]['max_tokens'] == 2176 and trace[0]['status'] == 'error'
        assert all(not path.endswith('/chat/completions') for path, _ in chat.requests)
        assert context.db_path.read_bytes() == before
        assert embed.requests == []
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['accuracy'], indirect=True)
async def test_direct_calls_and_frozen_payloads_keep_original_cap(native_pair):
    context, chat, _embed = native_pair
    payload = build_chat_request(context.settings.chat, 'system', 'source', 128)
    try:
        assert await context.chat.call_payload(payload) == 'Documented arithmetic.'
        assert context.chat.call_sync('system', 'source', max_tokens=1024) == 'Documented arithmetic.'
        caps = [body['max_tokens'] for path, body in chat.requests if path.endswith('/chat/completions')]
        assert caps == [128, 1024]
        assert payload['max_tokens'] == 128
    finally:
        await context.aclose()
