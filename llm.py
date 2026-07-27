"""Shared LLM completion helper for Ariadne.

Consolidates the chat completion pattern used across gap_analysis,
mcp_service, themes labeling, and other modules that need one-shot
LLM calls. Routes through the provider factory so the configured
provider (openai / anthropic) is honored — same model selection rules
as ``ariadne generate``.
"""
from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)


async def close() -> None:
    """Backwards-compat no-op.

    Earlier versions used a module-level ``httpx.AsyncClient`` cache;
    the provider factory now owns its own client lifecycle, so callers
    no longer need to close a shared one. Kept as a stub so existing
    shutdown hooks don't fail.
    """
    return None


async def chat_complete(
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    timeout: float = 60.0,
) -> str:
    """One-shot chat completion via the configured provider.

    Honors ``provider:`` in ``ariadne.yaml`` — Anthropic models route to
    ``/v1/messages``, OpenAI models to ``/chat/completions``. The model
    name is inferred when ``provider:`` is omitted (gpt-* → openai,
    claude-* → anthropic).

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts. The
            ``"system"`` role becomes the provider's system prompt; all
            ``"user"`` roles are concatenated into the user prompt
            (current callers pass exactly one of each).
        model: Model name (defaults to config model).
        max_tokens: Maximum output tokens.
        timeout: Request timeout in seconds.

    Returns:
        The assistant's response text. Empty string if the provider
        returned None (after retries).

    Raises:
        ValueError: If the required API key for the provider isn't set.
    """
    from cli.generate import resolve_provider
    from config import get_config
    from docgen.llm.factory import make_llm_provider

    system_parts: list[str] = []
    user_parts: list[str] = []
    for m in messages:
        role = m.get('role')
        content = m.get('content', '')
        if role == 'system':
            system_parts.append(content)
        elif role == 'user':
            user_parts.append(content)
    system_prompt = '\n\n'.join(system_parts)
    user_prompt = '\n\n'.join(user_parts)

    cfg = get_config()
    if model is None:
        model = cfg.model

    provider_name = resolve_provider(
        cli_provider=None,
        cfg_provider=getattr(cfg, 'configured_provider', None),
        model=model,
    )

    if provider_name == 'anthropic':
        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            raise ValueError(
                'ANTHROPIC_API_KEY environment variable is required '
                'when provider=anthropic.'
            )
        base_url = os.environ.get(
            'ANTHROPIC_BASE_URL', 'https://api.anthropic.com/v1',
        )
    else:
        api_key = os.environ.get('OPENAI_API_KEY', '')
        if not api_key:
            raise ValueError(
                'OPENAI_API_KEY environment variable is required '
                'when provider=openai.'
            )
        base_url = os.environ.get(
            'OPENAI_BASE_URL', 'https://api.openai.com/v1',
        )

    provider = make_llm_provider(
        provider=provider_name,
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
    try:
        result = await provider.call(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
        )
        return result or ''
    finally:
        await provider.aclose()
