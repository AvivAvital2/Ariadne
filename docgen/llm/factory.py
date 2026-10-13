"""Factory for LLM providers.

Selects the concrete provider class by name and constructs it with the
provided model + api_key + base_url. Use this from configuration code;
don't import provider classes directly elsewhere.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docgen.spool import SpoolJobSpec
    from inference import InferenceContext
    from library import Library

from typing import Literal

from docgen.llm.base import LLMProvider
from docgen.llm.batch import BatchStrategy

ProviderName = Literal['openai', 'anthropic', 'local']


def make_llm_provider(
    provider: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    max_retries: int = 3,
    timeout: float = 120.0,
    context: InferenceContext | None = None,
) -> LLMProvider:
    """Construct an LLM provider by name.

    Args:
        provider: ``"openai"`` or ``"anthropic"``. Raises ``ValueError``
            on unknown names.
        model: Model identifier as the provider expects it
            (e.g. ``"gpt-5.4"`` or ``"claude-opus-4-6"``).
        api_key: Auth credential; passed through unchanged.
        base_url: Optional override (default depends on provider).
        max_retries: Retry budget for transient HTTP errors.
        timeout: Per-request timeout in seconds.

    Returns:
        A concrete provider implementing ``LLMProvider``.
    """
    if context is None:
        from inference import current_context

        context = current_context()
    if context is not None:
        context.validate_producer(model=model, provider=provider)
    if provider == 'local':
        if context is None or not context.local:
            raise ValueError('Local inference requires a bound InferenceContext')
        if base_url is not None and base_url.rstrip('/') != context.settings.chat.base_url:
            raise ValueError('Explicit local endpoint differs from the operation configuration')
        if api_key and api_key != context.settings.chat.credential():
            raise ValueError(
                'Explicit local credential differs from its configured environment reference'
            )
        from docgen.llm.local import LocalOpenAICompatibleProvider

        return LocalOpenAICompatibleProvider(context)
    if provider == 'openai':
        from docgen.llm.openai import OpenAIProvider
        return OpenAIProvider(
            model=model,
            api_key=api_key,
            base_url=base_url or 'https://api.openai.com/v1',
            max_retries=max_retries,
            timeout=timeout,
        )
    if provider == 'anthropic':
        from docgen.llm.anthropic import AnthropicProvider
        return AnthropicProvider(
            model=model,
            api_key=api_key,
            base_url=base_url or 'https://api.anthropic.com/v1',
            max_retries=max_retries,
            timeout=timeout,
        )
    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        f"Supported: 'openai', 'anthropic'."
    )


def make_batch_strategy(
    provider: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    context: InferenceContext | None = None,
    library: Library | None = None,
    job_specs: list[SpoolJobSpec] | None = None,
    operation: str = 'generate',
    scope: dict | None = None,
) -> BatchStrategy:
    """Construct a batch dispatch strategy by provider name.

    Mirrors :func:`make_llm_provider`. Anthropic's Message Batches API and
    OpenAI's Batch API expose the same submit → poll → fetch → cancel
    lifecycle (24h window, ~50% off), so callers depend only on
    ``BatchStrategy`` and pick the concrete one by the resolved provider.

    Args:
        provider: ``"openai"`` or ``"anthropic"``. Raises ``ValueError`` on
            unknown names — there is no batch backend for it.
        model: Model identifier as the provider expects it.
        api_key: Auth credential for the underlying transport.
        base_url: Optional override (default depends on provider).

    Returns:
        A concrete strategy implementing ``BatchStrategy``.
    """
    if provider == 'local':
        if context is None or library is None or job_specs is None:
            raise ValueError(
                'Local batches require a bound context, library and complete job specifications'
            )
        context.validate_producer(model=model, provider=provider)
        from docgen.llm.local_batch import LocalBatchStrategy

        return LocalBatchStrategy(
            context, library=library, job_specs=job_specs, operation=operation, scope=scope
        )
    return batch_strategy_for(make_llm_provider(
        provider, model=model, api_key=api_key, base_url=base_url,
    ))


def batch_strategy_for(
    provider: LLMProvider,
    *,
    context: InferenceContext | None = None,
    library: Library | None = None,
    job_specs: list[SpoolJobSpec] | None = None,
    operation: str = 'generate',
    scope: dict | None = None,
) -> BatchStrategy:
    """Wrap an already-constructed provider in its batch strategy.

    The strategy reuses the provider's authenticated transport (HTTP client,
    retry budget, cache-stats accounting), so callers that already hold a
    provider — the orchestrator's generator, say — dispatch batch through the
    strategy rather than the provider itself. Anthropic → Message Batches,
    OpenAI → Batch API; both a 24h window at ~50% off.

    Raises ``ValueError`` for a provider type with no batch backend.
    """
    from docgen.llm.anthropic import AnthropicProvider
    from docgen.llm.local import LocalOpenAICompatibleProvider
    from docgen.llm.openai import OpenAIProvider

    if isinstance(provider, LocalOpenAICompatibleProvider):
        return make_batch_strategy(
            'local',
            model=provider.context.settings.chat.model,
            api_key='',
            context=context or provider.context,
            library=library,
            job_specs=job_specs,
            operation=operation,
            scope=scope,
        )

    if isinstance(provider, OpenAIProvider):
        from docgen.llm.openai_batch import OpenAIBatchStrategy
        return OpenAIBatchStrategy(provider)
    if isinstance(provider, AnthropicProvider):
        from docgen.llm.anthropic_batch import AnthropicBatchStrategy
        return AnthropicBatchStrategy(provider)
    raise ValueError(
        f'No batch strategy for provider type {type(provider).__name__!r}. '
        f'Supported: OpenAIProvider, AnthropicProvider.'
    )
