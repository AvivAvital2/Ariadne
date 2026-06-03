"""Tests for model→provider inference and mismatch detection.

The rule: gpt-* models go to OpenAI; claude-* models go to Anthropic.
The CLI infers provider when only the model is set, and rejects
configurations where an explicit provider doesn't match the model
family.
"""
from __future__ import annotations

import pytest


def test_infer_provider_for_gpt_models():
    from cli.generate import infer_provider_from_model
    assert infer_provider_from_model('gpt-5.4') == 'openai'
    assert infer_provider_from_model('gpt-5.5') == 'openai'
    assert infer_provider_from_model('gpt-5-mini') == 'openai'
    assert infer_provider_from_model('gpt-4o') == 'openai'


def test_infer_provider_for_claude_models():
    from cli.generate import infer_provider_from_model
    assert infer_provider_from_model('claude-opus-4-6') == 'anthropic'
    assert infer_provider_from_model('claude-opus-4-7') == 'anthropic'
    assert infer_provider_from_model('claude-sonnet-4-6') == 'anthropic'
    assert infer_provider_from_model('claude-haiku-4-5') == 'anthropic'


def test_infer_provider_for_unknown_returns_none():
    from cli.generate import infer_provider_from_model
    assert infer_provider_from_model('grok-9') is None
    assert infer_provider_from_model('llama-3-70b') is None


def test_resolve_provider_inferred_from_model():
    """No explicit provider → inferred from model family."""
    from cli.generate import resolve_provider
    assert resolve_provider(
        cli_provider=None, cfg_provider=None, model='gpt-5.4',
    ) == 'openai'
    assert resolve_provider(
        cli_provider=None, cfg_provider=None, model='claude-opus-4-6',
    ) == 'anthropic'


def test_resolve_provider_cli_overrides_config():
    """CLI flag wins over yaml config."""
    from cli.generate import resolve_provider
    assert resolve_provider(
        cli_provider='anthropic',
        cfg_provider='openai',
        model='claude-opus-4-6',
    ) == 'anthropic'


def test_resolve_provider_mismatch_raises():
    """Explicit provider that contradicts the model family must fail loudly."""
    from cli.generate import resolve_provider
    with pytest.raises(ValueError, match='provider'):
        resolve_provider(
            cli_provider='anthropic', cfg_provider=None, model='gpt-5.4',
        )
    with pytest.raises(ValueError, match='provider'):
        resolve_provider(
            cli_provider=None, cfg_provider='openai', model='claude-opus-4-6',
        )


def test_resolve_provider_matched_explicit_passes():
    """Explicit provider that agrees with model family is accepted."""
    from cli.generate import resolve_provider
    assert resolve_provider(
        cli_provider='openai', cfg_provider=None, model='gpt-5.4',
    ) == 'openai'
    assert resolve_provider(
        cli_provider='anthropic', cfg_provider=None, model='claude-opus-4-6',
    ) == 'anthropic'


def test_resolve_provider_unknown_model_defaults_to_openai():
    """For unknown model patterns, fall back to "openai" (current default).
    Users with custom proxies / fine-tunes set provider explicitly.
    """
    from cli.generate import resolve_provider
    assert resolve_provider(
        cli_provider=None, cfg_provider=None, model='custom-model-xyz',
    ) == 'openai'
