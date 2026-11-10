"""Shared Ask/clew completions honor local deployment timing without changing hosted defaults."""
from types import SimpleNamespace

import pytest

from llm import capture_completion_trace, chat_complete


@pytest.mark.parametrize('explicit_context', [False, True])
@pytest.mark.parametrize('override', [None, 12.5])
async def test_native_shared_completion_uses_configured_timeout(native_pair, explicit_context, override):
    context, chat, _embed = native_pair
    requests = []

    async def record(request):
        requests.append((request.url.path, request.extensions['timeout']))

    context.chat.transport.client().event_hooks['request'].append(record)
    options = {'context': context} if explicit_context else {}
    if override is not None:
        options['timeout'] = override
    expected = override if override is not None else context.settings.chat.timeout_seconds
    try:
        with context.activate(), capture_completion_trace() as trace:
            assert await chat_complete(user_prompt='Explain arithmetic.', **options) == 'Documented arithmetic.'
        inference = [(path, timeouts) for path, timeouts in requests if not path.endswith('/models')]
        assert {path for path, _ in inference} == {
            '/native/apply-template', '/native/tokenize', '/native/props', '/native/v1/chat/completions'}
        assert all(timeouts['read'] == expected for _, timeouts in inference), inference
        assert trace[0]['timeout'] == expected
        assert any(path.endswith('/chat/completions') for path, _ in chat.requests)
    finally:
        await context.aclose()


@pytest.mark.parametrize(('provider', 'model', 'key'), [
    ('openai', 'gpt-5', 'OPENAI_API_KEY'),
    ('anthropic', 'claude-opus-4-8', 'ANTHROPIC_API_KEY'),
])
@pytest.mark.parametrize('override', [None, 12.5])
async def test_hosted_shared_completion_preserves_default_and_explicit_timeout(
        monkeypatch, provider, model, key, override):
    import config
    import docgen.llm.factory

    options = []

    class Provider:
        async def call(self, _system, _user, *, max_tokens):
            assert max_tokens == 2048
            return 'answer'

        async def aclose(self):
            pass

    def factory(**kwargs):
        options.append(kwargs)
        return Provider()

    monkeypatch.setattr(config, 'get_config', lambda: SimpleNamespace(
        model=model, configured_provider=provider))
    monkeypatch.setenv(key, 'synthetic-key')
    monkeypatch.setattr(docgen.llm.factory, 'make_llm_provider', factory)
    with capture_completion_trace() as trace:
        assert await chat_complete(user_prompt='Question', **(
            {'timeout': override} if override is not None else {})) == 'answer'
    expected = override if override is not None else 60.0
    assert options[0]['timeout'] == trace[0]['timeout'] == expected
