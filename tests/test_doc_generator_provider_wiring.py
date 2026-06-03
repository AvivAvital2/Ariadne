"""Tests for DocGenerator routing through the LLM provider factory.

DocGenerator used to call the OpenAI client directly via ``httpx``.
After the provider abstraction (Phase 2.x), it must construct a
provider via the factory and delegate the LLM call.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_doc_generator_constructs_openai_provider_by_default():
    """When ``provider="openai"`` (default), DocGenerator's underlying
    LLM client is an OpenAIProvider built via the factory.
    """
    from docgen.generator import DocGenerator, GeneratorConfig
    from docgen.llm.openai import OpenAIProvider

    cfg = GeneratorConfig(
        model='gpt-5.4',
        api_key='sk-test',
        provider='openai',
    )

    gen = DocGenerator(config=cfg)
    await gen.__aenter__()
    try:
        assert isinstance(gen._provider, OpenAIProvider)
        assert gen._provider.model == 'gpt-5.4'
    finally:
        await gen.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_doc_generator_constructs_anthropic_provider_when_configured():
    """``provider="anthropic"`` produces an AnthropicProvider with the
    Anthropic default base_url and the provided api_key.
    """
    from docgen.generator import DocGenerator, GeneratorConfig
    from docgen.llm.anthropic import AnthropicProvider

    cfg = GeneratorConfig(
        model='claude-opus-4-6',
        api_key='sk-ant-test',
        provider='anthropic',
        base_url='https://api.anthropic.com/v1',
    )

    gen = DocGenerator(config=cfg)
    await gen.__aenter__()
    try:
        assert isinstance(gen._provider, AnthropicProvider)
        assert gen._provider.model == 'claude-opus-4-6'
        assert gen._provider.api_key == 'sk-ant-test'
    finally:
        await gen.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_call_llm_delegates_to_provider():
    """``_call_llm`` must invoke the provider's ``call`` method instead
    of running its own httpx request.
    """
    from docgen.generator import DocGenerator, GeneratorConfig

    cfg = GeneratorConfig(model='gpt-5.4', api_key='sk-test', provider='openai')
    gen = DocGenerator(config=cfg)
    await gen.__aenter__()
    try:
        # Replace the whole _provider slot with a mock — slot reassignment
        # works on attrs @define classes; modifying methods on a slotted
        # provider instance does not.
        mock_provider = MagicMock()
        mock_provider.call = AsyncMock(return_value='hello world')
        mock_provider.aclose = AsyncMock()
        gen._provider = mock_provider

        result = await gen._call_llm('system msg', 'user msg')
        assert result == 'hello world'
        mock_provider.call.assert_awaited_once()
        args, kwargs = mock_provider.call.call_args
        assert args[0] == 'system msg'
        assert args[1] == 'user msg'
    finally:
        await gen.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_doc_generator_aexit_closes_provider():
    """``__aexit__`` must call ``provider.aclose`` so HTTP connections
    are released.
    """
    from docgen.generator import DocGenerator, GeneratorConfig

    cfg = GeneratorConfig(model='gpt-5.4', api_key='sk-test', provider='openai')
    gen = DocGenerator(config=cfg)
    await gen.__aenter__()

    aclose_mock = AsyncMock()
    mock_provider = MagicMock()
    mock_provider.aclose = aclose_mock
    gen._provider = mock_provider

    await gen.__aexit__(None, None, None)

    aclose_mock.assert_awaited_once()
