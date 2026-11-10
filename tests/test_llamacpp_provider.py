from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from docgen.llm.local import (
    ContextOverflow,
    LocalInferenceError,
    TransientInferenceError,
    build_chat_request,
    chat_token_request,
    check_endpoints,
)
from model_config import ModelConfigurationError, canonical_json


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.asyncio
async def test_native_chat_counts_server_rendered_template(native_pair):
    context, chat, embed = native_pair
    try:
        result = await context.chat.call('system', 'user', max_tokens=8192)
        assert result == 'Documented arithmetic.'
        requests = dict(chat.requests)
        assert requests['/native/apply-template']['messages'] == [
            {'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'user'}]
        assert requests['/native/tokenize'] == {
            'content': chat.rendered, 'add_special': True,
            'parse_special': True, 'with_pieces': False,
        }
        assert requests['/native/v1/chat/completions']['max_tokens'] == 8192
        assert context.chat.last_usage == {'prompt_tokens': 4, 'completion_tokens': 5}
        assert embed.requests == []
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.asyncio
async def test_native_sync_chat_uses_the_same_prompt_count(native_pair):
    context, chat, _embed = native_pair
    try:
        assert context.chat.call_sync('system', 'user') == 'Documented arithmetic.'
        assert dict(chat.requests)['/native/tokenize']['content'] == chat.rendered
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.asyncio
async def test_doctor_probes_both_native_endpoints_without_generation(native_pair):
    context, chat, embed = native_pair
    try:
        before = context.db_path.read_bytes()
        report = await check_endpoints(context.settings)
        assert report['chat']['probe_tokens'] == report['embeddings']['probe_tokens'] == 4
        assert report['chat']['server_context'] == 32768
        assert all(not route.endswith('/chat/completions') for route, _ in chat.requests)
        assert all(not route.endswith('/apply-template') for route, _ in embed.requests)
        assert dict(embed.requests)['/native/tokenize']['content'].startswith('Instruct: ')
        assert context.db_path.read_bytes() == before
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.asyncio
async def test_native_embedding_tokenization_uses_framed_input_without_chat_template(native_pair):
    context, chat, embed = native_pair
    try:
        vectors = await context.embeddings.embed_documents(['first document', 'second document'])
        query = await context.embeddings.embed_query('How does arithmetic work?')
        assert len(vectors) == 2 and query.shape == (3072,)
        token_inputs = [payload['content'] for path, payload in embed.requests
                        if path.endswith('/tokenize')]
        assert token_inputs[:2] == ['first document', 'second document']
        assert token_inputs[2] == (
            'Instruct: Given a question about a codebase, retrieve relevant code documentation '
            'that answers the question.\nQuery:How does arithmetic work?')
        embedded_inputs = [payload['input'] for path, payload in embed.requests
                           if path.endswith('/embeddings')]
        assert embedded_inputs == [token_inputs[:2], token_inputs[2:]]
        assert all(not path.endswith('/apply-template') for path, _ in embed.requests)
        assert chat.requests == []
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize('extra_input', [0, 1])
@pytest.mark.asyncio
async def test_native_context_boundary_includes_output_and_safety_margin(native_pair, extra_input):
    context, chat, _embed = native_pair
    chat.context = 8260  # Four input tokens + 8192 output + 64 safety margin.
    chat.tokens = list(range(4 + extra_input))
    try:
        if extra_input:
            with pytest.raises(ContextOverflow):
                await context.chat.call('system', 'user', max_tokens=8192)
            assert all(not path.endswith('/chat/completions') for path, _ in chat.requests)
        else:
            assert await context.chat.call('system', 'user', max_tokens=8192)
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize(('route', 'response'), [
    ('/native/apply-template', {'prompt': ''}),
    ('/native/apply-template', {'prompt': ['unexpected']}),
    ('/native/tokenize', {'tokens': []}),
    ('/native/tokenize', {'tokens': [True]}),
    ('/native/tokenize', {'tokens': [{'id': 42, 'piece': 'text'}]}),
    ('/native/props', {}),
    ('/native/props', {'default_generation_settings': {'n_ctx': 0}}),
    ('/native/props', {'default_generation_settings': {'n_ctx': True}}),
])
@pytest.mark.asyncio
async def test_invalid_native_token_contract_fails_before_generation(native_pair, route, response):
    context, chat, _embed = native_pair
    chat.overrides[route] = response
    try:
        with pytest.raises(LocalInferenceError):
            await context.chat.call('system', 'user')
        assert all(not path.endswith('/chat/completions') for path, _ in chat.requests)
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.asyncio
async def test_native_embedding_rejects_overflow_and_batches_from_native_counts(native_pair):
    context, _chat, embed = native_pair
    context.settings = replace(context.settings, embeddings=replace(
        context.settings.embeddings, batch_size=3, batch_max_tokens=8))
    try:
        assert len(await context.embeddings.embed_documents(['a', 'b', 'c', 'd', 'e'])) == 5
        assert [body['input'] for path, body in embed.requests if path.endswith('/embeddings')] == [
            ['a', 'b'], ['c', 'd'], ['e']]
        embed.requests.clear()
        embed.context = 3
        with pytest.raises(ContextOverflow):
            await context.embeddings.embed_documents(['oversized'])
        assert all(not path.endswith('/embeddings') for path, _ in embed.requests)
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.asyncio
async def test_native_sync_respects_configured_limit_and_rejects_bad_tokens(native_pair):
    context, chat, _embed = native_pair
    context.settings = replace(context.settings, chat=replace(context.settings.chat, context_window=8260))
    try:
        assert context.chat.call_sync('system', 'user', max_tokens=8192)
        chat.tokens.append(105)
        chat.requests.clear()
        with pytest.raises(ContextOverflow):
            context.chat.call_sync('system', 'user', max_tokens=8192)
        chat.tokens = [True]
        with pytest.raises(LocalInferenceError, match='token IDs'):
            context.chat.call_sync('system', 'user')
        assert all(not path.endswith('/chat/completions') for path, _ in chat.requests)
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('route', ['/native/apply-template', '/native/props'])
@pytest.mark.asyncio
async def test_native_tokenization_honors_retry_budget(native_pair, route, synchronous):
    context, chat, _embed = native_pair
    chat.failures[route] = [503]
    try:
        context.chat.transport.attempts = 2
        assert (context.chat.call_sync('system', 'user') if synchronous else
                await context.chat.call('system', 'user', attempts=2))
        chat.requests.clear()
        chat.failures[route] = [503]
        context.chat.transport.attempts = 1
        with pytest.raises(LocalInferenceError):
            if synchronous:
                context.chat.call_sync('system', 'user')
            else:
                await context.chat.call('system', 'user', attempts=1)
        assert len([path for path, _ in chat.requests if path == route]) == 1
        assert all(not path.endswith('/chat/completions') for path, _ in chat.requests)
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['accuracy'], indirect=True)
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.asyncio
async def test_thinking_preflight_preserves_native_template_controls(native_pair, synchronous):
    """b11669 applies top-level effort/format in both native chat routes."""
    context, chat, _embed = native_pair
    settings = replace(context.settings.chat, request_options_json=canonical_json({
        'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
        'presence_penalty': 0.0, 'repeat_penalty': 1.0,
        'chat_template_kwargs': {'enable_thinking': True},
        'reasoning_effort': 'xhigh', 'reasoning_format': 'deepseek',
    }))
    payload = build_chat_request(settings, 'Document the source.', 'def add(a, b): return a + b', 8192)
    rendered = '<start>system\nDocument the source.<assistant><think>\nxhigh\n'
    chat.overrides['/native/apply-template'] = {'prompt': rendered}
    try:
        if synchronous:
            counted = context.chat.transport.tokenize_sync(chat_token_request(payload))
        else:
            counted = await context.chat.transport.tokenize(chat_token_request(payload))
        requests = dict(chat.requests)
        assert requests['/native/apply-template'] == {
            'model': payload['model'],
            'messages': [
                {'role': 'system', 'content': 'Document the source.'},
                {'role': 'user', 'content': 'def add(a, b): return a + b'},
            ],
            'chat_template_kwargs': {'enable_thinking': True},
            'reasoning_effort': 'xhigh',
            'reasoning_format': 'deepseek',
            'add_generation_prompt': True,
            'add_special_tokens': False,
        }
        assert requests['/native/tokenize']['content'] == rendered
        assert counted == {'tokens': [101, 102, 103, 104], 'count': 4, 'max_model_len': 32768}
        assert payload['repeat_penalty'] == 1.0
        assert 'repetition_penalty' not in payload
        assert payload['max_tokens'] == 8192
    finally:
        await context.aclose()


@pytest.fixture
def native_final_pair(native_pair):
    """Final content stays separate from server-side reasoning for either model."""
    context, chat, embed = native_pair
    chat.overrides['/native/v1/chat/completions'] = {
        'choices': [{
            'message': {
                'role': 'assistant',
                'reasoning_content': 'PRIVATE REASONING: inspect the addition.',
                'content': '## add\n\nReturns the sum of its two arguments.\n',
            },
            'finish_reason': 'stop',
        }],
        # Completion usage includes thinking, not merely the final Markdown.
        'usage': {'prompt_tokens': 4, 'completion_tokens': 113},
    }
    return context, chat, embed


@pytest.mark.parametrize('native_pair', ['accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.asyncio
async def test_native_profile_returns_only_server_separated_final_answer(native_final_pair, synchronous):
    context, chat, embed = native_final_pair
    try:
        if synchronous:
            result = context.chat.call_sync('Document the source.', 'def add(a, b): return a + b',
                                            max_tokens=8192)
        else:
            result = await context.chat.call('Document the source.', 'def add(a, b): return a + b',
                                             max_tokens=8192)
        assert result == '## add\n\nReturns the sum of its two arguments.\n'
        assert context.chat.last_usage == {'prompt_tokens': 4, 'completion_tokens': 113}
        requests = dict(chat.requests)
        generated = requests['/native/v1/chat/completions']
        rendered = requests['/native/apply-template']
        for key in ('model', 'messages', 'chat_template_kwargs', 'reasoning_format'):
            assert generated[key] == rendered[key]
        if chat.thinking:
            assert generated['model'] == 'Qwen/Qwen3.8-27B'
            assert generated['chat_template_kwargs'] == {'enable_thinking': True}
            assert generated['reasoning_effort'] == rendered['reasoning_effort'] == 'xhigh'
            assert generated['reasoning_format'] == 'deepseek' and generated['top_k'] == 20
        else:
            assert generated['model'] == 'google/gemma-4-12b-it'
            assert generated['chat_template_kwargs'] == {'enable_thinking': False}
            assert 'reasoning_effort' not in generated and 'reasoning_effort' not in rendered
            assert generated['reasoning_format'] == 'auto' and generated['top_k'] == 64
        assert generated['min_p'] == generated['presence_penalty'] == 0.0
        assert generated['repeat_penalty'] == generated['temperature'] == 1.0
        assert generated['top_p'] == 0.95
        assert generated['max_tokens'] == 8192 and generated['stream'] is False
        assert 'repetition_penalty' not in generated
        assert requests['/native/tokenize']['content'] == chat.rendered
        assert embed.requests == []
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize(('message', 'finish_reason'), [
    ({'reasoning_content': 'PRIVATE REASONING: no final answer.', 'content': None}, 'stop'),
    ({'reasoning_content': 'PRIVATE REASONING: no final answer.'}, 'stop'),
    ({'reasoning_content': 'PRIVATE REASONING', 'content': [{'text': 'not final text'}]}, 'stop'),
    ({'content': '<think>PRIVATE REASONING</think>Unseparated final.'}, 'stop'),
    ({'content': '<|channel>thought\nPRIVATE REASONING'}, 'stop'),
    ({'content': '<channel|>PRIVATE REASONING'}, 'stop'),
    ({'content': '<|think|>PRIVATE REASONING'}, 'stop'),
    ({'reasoning_content': 'PRIVATE REASONING', 'content': 'Truncated final.'}, 'length'),
])
@pytest.mark.asyncio
async def test_native_profile_rejects_incomplete_or_malformed_final(
    native_final_pair, synchronous, message, finish_reason,
):
    context, chat, _embed = native_final_pair
    chat.overrides['/native/v1/chat/completions'] = {
        'choices': [{'message': message, 'finish_reason': finish_reason}],
    }
    before = context.db_path.read_bytes()
    try:
        with pytest.raises(LocalInferenceError) as error:
            if synchronous:
                context.chat.call_sync('Document the source.', 'def add(a, b): return a + b')
            else:
                await context.chat.call('Document the source.', 'def add(a, b): return a + b')
        assert 'PRIVATE REASONING' not in str(error.value)
        assert len([path for path, _ in chat.requests if path.endswith('/chat/completions')]) == 1
        assert context.db_path.read_bytes() == before
        assert context.chat.last_usage == {}
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize('mutation', ['model', 'stream', 'role', 'output_cap', 'empty_input', 'thinking'])
async def test_changed_persisted_request_is_rejected_before_any_inference(native_pair, mutation):
    context, chat, embed = native_pair
    payload = build_chat_request(context.settings.chat, 'system', 'user', 8192)
    if mutation == 'model':
        payload['model'] = 'different/model'
    elif mutation == 'stream':
        payload['stream'] = True
    elif mutation == 'role':
        payload['messages'][0]['role'] = 'assistant'
    elif mutation == 'output_cap':
        payload['max_tokens'] = 8193
    elif mutation == 'empty_input':
        payload['messages'][1]['content'] = ' '
    else:
        payload['chat_template_kwargs']['enable_thinking'] = not chat.thinking
    before = context.db_path.read_bytes()
    try:
        with pytest.raises((LocalInferenceError, ModelConfigurationError)):
            await context.chat.call_payload(payload)
        assert chat.requests == embed.requests == []
        assert context.db_path.read_bytes() == before
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('response', [[], {'data': [{'id': 'different/model'}]}])
async def test_wrong_model_or_malformed_readiness_never_generates(native_pair, synchronous, response):
    context, chat, embed = native_pair
    chat.overrides['/native/v1/models'] = response
    try:
        with pytest.raises(LocalInferenceError):
            if synchronous:
                context.chat.call_sync('system', 'user')
            else:
                await context.chat.call('system', 'user')
        assert [route for route, _ in chat.requests] == ['/native/v1/models']
        assert embed.requests == []
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize('synchronous', [False, True])
@pytest.mark.parametrize('failure', [httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError])
async def test_connection_failures_exhaust_budget_without_fallback(native_pair, synchronous, failure, monkeypatch):
    context, chat, embed = native_pair
    transport = context.chat.transport
    transport.attempts = 2
    client = transport.sync_client() if synchronous else transport.client()
    request = (Mock if synchronous else AsyncMock)(side_effect=failure('PRIVATE transport detail'))
    monkeypatch.setattr(client, 'request', request)
    monkeypatch.setattr('docgen.llm.local.retry_delay', lambda *_args: 0)
    before = context.db_path.read_bytes()
    try:
        with pytest.raises(TransientInferenceError, match='connection failed') as error:
            if synchronous:
                context.chat.call_sync('system', 'user')
            else:
                await context.chat.call('system', 'user', attempts=2)
        assert 'PRIVATE' not in str(error.value)
        assert request.call_count == 2
        assert all(call.args == ('GET', context.settings.chat.base_url + '/models')
                   for call in request.call_args_list)
        assert chat.requests == embed.requests == []
        assert context.db_path.read_bytes() == before
    finally:
        await context.aclose()
