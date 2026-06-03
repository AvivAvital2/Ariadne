"""Tests for the LLM provider factory and Protocol.

The provider abstraction lets DocGenerator call OpenAI or Anthropic
through one interface. The factory selects the concrete provider by
name. Tests here cover only the Protocol shape and factory dispatch —
the per-provider request/response handling is tested in their own
test files.
"""
from __future__ import annotations

import pytest


class TestLLMProviderProtocol:
    """``LLMProvider`` protocol must declare ``call`` and ``aclose``."""

    def test_protocol_is_importable(self):
        from docgen.llm.base import LLMProvider  # noqa: F401

    def test_protocol_has_call_method(self):
        from docgen.llm.base import LLMProvider
        assert hasattr(LLMProvider, 'call')

    def test_protocol_has_aclose_method(self):
        from docgen.llm.base import LLMProvider
        assert hasattr(LLMProvider, 'aclose')


class TestFactoryDispatch:
    """``make_llm_provider`` must dispatch to the right concrete class."""

    def test_factory_is_importable(self):
        from docgen.llm.factory import make_llm_provider  # noqa: F401

    def test_openai_provider_dispatch(self):
        from docgen.llm.factory import make_llm_provider
        from docgen.llm.openai import OpenAIProvider

        p = make_llm_provider(
            provider='openai', model='gpt-5.4', api_key='sk-test',
        )
        assert isinstance(p, OpenAIProvider)

    def test_anthropic_provider_dispatch(self):
        from docgen.llm.anthropic import AnthropicProvider
        from docgen.llm.factory import make_llm_provider

        p = make_llm_provider(
            provider='anthropic',
            model='claude-opus-4-6',
            api_key='sk-ant-test',
        )
        assert isinstance(p, AnthropicProvider)

    def test_unknown_provider_raises(self):
        from docgen.llm.factory import make_llm_provider
        with pytest.raises(ValueError, match='provider'):
            make_llm_provider(
                provider='grok', model='grok-9', api_key='x',
            )

    def test_factory_passes_model_through(self):
        from docgen.llm.factory import make_llm_provider
        p = make_llm_provider(
            provider='openai', model='gpt-5.4', api_key='sk-test',
        )
        assert p.model == 'gpt-5.4'

    def test_factory_passes_api_key_through(self):
        from docgen.llm.factory import make_llm_provider
        p = make_llm_provider(
            provider='anthropic',
            model='claude-opus-4-6',
            api_key='sk-ant-test',
        )
        assert p.api_key == 'sk-ant-test'

    def test_factory_default_base_url_openai(self):
        from docgen.llm.factory import make_llm_provider
        p = make_llm_provider(
            provider='openai', model='gpt-5.4', api_key='sk-test',
        )
        assert p.base_url == 'https://api.openai.com/v1'

    def test_factory_default_base_url_anthropic(self):
        from docgen.llm.factory import make_llm_provider
        p = make_llm_provider(
            provider='anthropic',
            model='claude-opus-4-6',
            api_key='sk-ant-test',
        )
        assert p.base_url == 'https://api.anthropic.com/v1'

    def test_factory_respects_explicit_base_url(self):
        from docgen.llm.factory import make_llm_provider
        p = make_llm_provider(
            provider='openai', model='gpt-5.4', api_key='sk-test',
            base_url='https://my-proxy.local/v1',
        )
        assert p.base_url == 'https://my-proxy.local/v1'


class TestOrchestratorAndGeneratorConfigsHaveProviderField:
    """Both configs must expose ``provider``, defaulting to ``"openai"``
    for backwards-compatibility.
    """

    def test_orchestrator_config_has_provider_field(self):
        from pathlib import Path

        from docgen.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig(
            source_path=Path('/tmp'),
            db_path=Path('/tmp/x.db'),
            staleness_db_path=Path('/tmp/s.db'),
        )
        assert cfg.provider == 'openai'

    def test_orchestrator_config_provider_can_be_anthropic(self):
        from pathlib import Path

        from docgen.orchestrator import OrchestratorConfig
        cfg = OrchestratorConfig(
            source_path=Path('/tmp'),
            db_path=Path('/tmp/x.db'),
            staleness_db_path=Path('/tmp/s.db'),
            provider='anthropic',
        )
        assert cfg.provider == 'anthropic'

    def test_generator_config_has_provider_field(self):
        from docgen.generator import GeneratorConfig
        gc = GeneratorConfig()
        assert gc.provider == 'openai'

    def test_generator_config_provider_anthropic(self):
        from docgen.generator import GeneratorConfig
        gc = GeneratorConfig(provider='anthropic')
        assert gc.provider == 'anthropic'
