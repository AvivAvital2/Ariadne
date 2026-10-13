import httpx
import pytest

from docgen.llm.local import ContextOverflow, LocalInferenceError


@pytest.mark.asyncio
async def test_chat_exact_template_auth_and_final_output(local_deployment, monkeypatch):
    ctx, chat, embed = local_deployment
    monkeypatch.setenv('OPENAI_API_KEY', 'hosted-secret')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'hosted-secret')
    monkeypatch.setenv('HTTP_PROXY', 'http://forbidden.example:9000')
    result = await ctx.chat.call('system', 'Explain arithmetic', max_tokens=8192)
    assert 'Core arithmetic' in result
    request = chat.requests[-1]
    assert request[1]['max_tokens'] == 8192
    assert 'Authorization' not in request[2]
    assert chat.requests[-2][0] == '/chat/tokenize'
    assert embed.requests == []
    assert ctx.chat.last_usage == {'prompt_tokens': 32, 'completion_tokens': 200}
    await ctx.aclose()


@pytest.mark.parametrize(
    'choice',
    [
        {'message': {'content': 'partial'}, 'finish_reason': 'length'},
        {
            'message': {'content': '', 'reasoning_content': 'private reasoning'},
            'finish_reason': 'stop',
        },
        {'message': {'content': '<think>reasoning</think>'}, 'finish_reason': 'stop'},
        {'message': {'content': []}, 'finish_reason': 'stop'},
    ],
)
@pytest.mark.asyncio
async def test_reject_incomplete_final_output(local_deployment, choice):
    ctx, chat, _ = local_deployment
    chat.chat_response = {'choices': [choice]}
    with pytest.raises(LocalInferenceError):
        await ctx.chat.call('system', 'user')
    await ctx.aclose()


@pytest.mark.asyncio
async def test_context_overflow_fails_before_chat(local_deployment):
    ctx, chat, _ = local_deployment
    chat.token_count = 32760
    with pytest.raises(ContextOverflow):
        await ctx.chat.call('system', 'user')
    assert all(not route.endswith('/chat/completions') for route, _, _ in chat.requests)
    await ctx.aclose()


@pytest.mark.asyncio
async def test_transient_retry_and_permanent_failure(local_deployment):
    ctx, chat, _ = local_deployment
    chat.failures = [429, 503]
    assert await ctx.chat.call('system', 'user')
    chat.failures = [400]
    before = len(chat.requests)
    with pytest.raises(LocalInferenceError):
        await ctx.chat.call('system', 'user')
    assert len(chat.requests) == before + 1
    await ctx.aclose()


@pytest.mark.asyncio
@pytest.mark.usefixtures('local_deployment')
async def test_network_trap_detects_forbidden_host():
    async with httpx.AsyncClient() as client:
        with pytest.raises(AssertionError, match='Forbidden inference host'):
            await client.get('https://api.openai.com/v1/models')
