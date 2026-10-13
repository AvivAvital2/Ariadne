"""Shared LLM completion helper for Ariadne.

Consolidates the chat completion pattern used across gap_analysis,
mcp_service, themes labeling, and other modules that need one-shot
LLM calls. Routes through the provider factory so the configured
provider (openai / anthropic) is honored — same model selection rules
as ``ariadne generate``.
"""
from __future__ import annotations
import contextvars
from contextlib import contextmanager

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference import InferenceContext

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
    messages: list[dict] | None = None,
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    timeout: float = 60.0,
    phase: str = "completion",
    usage_sink: list | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    context: InferenceContext | None = None,
) -> str:
    """One-shot completion with optional exact request/response diagnostics.

    ``usage_sink``, when given, receives one row per completion carrying
    the recorded provider usage (token counts and, when reported, the
    provider stop reason) together with the request's ``max_tokens`` cap —
    the signal a caller needs to detect a truncated selection reply
    instead of mistaking it for a deliberately short one.
    """
    from config import get_config
    from docgen.llm.factory import make_llm_provider
    from inference import current_context
    from model_config import resolve_provider

    if messages is not None and (system_prompt is not None or user_prompt is not None):
        raise ValueError('Pass messages or system_prompt/user_prompt, not both')
    if messages is None:
        messages = [
            {'role': 'system', 'content': system_prompt or ''},
            {'role': 'user', 'content': user_prompt or ''},
        ]

    system_parts: list[str] = []
    user_parts: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    system_prompt = "\n\n".join(system_parts)
    user_prompt = "\n\n".join(user_parts)

    context = context or current_context()
    cfg = get_config()
    if context is None and getattr(cfg, 'provider', None) == 'local':
        from inference import InferenceContext

        owned = InferenceContext.from_config(cfg)
        try:
            with owned.activate():
                return await chat_complete(
                    messages, model=model, max_tokens=max_tokens, timeout=timeout,
                    phase=phase, usage_sink=usage_sink, context=owned,
                )
        finally:
            await owned.aclose()
    if context is not None:
        model = model or context.settings.chat.model
        context.validate_producer(model=model, provider=context.settings.chat.provider)
        provider = context.chat
    else:
        if model is None:
            model = cfg.model
        provider_name = resolve_provider(
            cli_provider=None,
            cfg_provider=getattr(cfg, "configured_provider", None),
            model=model,
        )
        if provider_name == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is required "
                    "when provider=anthropic.")
            base_url = os.environ.get(
                "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required "
                    "when provider=openai.")
            base_url = os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1")

        provider = make_llm_provider(
            provider=provider_name,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
    trace = _completion_trace.get()
    request = {
        "phase": phase,
        "model": model,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "messages": [dict(message) for message in messages],
    }
    try:
        call_options = {'max_tokens': max_tokens}
        if context is not None and context.local:
            call_options['timeout'] = timeout
        result = await provider.call(system_prompt, user_prompt, **call_options)
        usage = dict(getattr(provider, "last_usage", None) or {})
        if usage_sink is not None:
            usage_sink.append({
                "phase": phase, "model": model,
                "max_tokens": max_tokens, **usage})
        collector = _completion_usage.get()
        if collector is not None and usage:
            collector.append({"phase": phase, "model": model, **usage})
        if trace is not None:
            trace.append({
                **request,
                "response": result or "",
                "usage": usage,
                "status": "ok",
            })
        return result or ""
    except Exception as error:
        if trace is not None:
            trace.append({
                **request,
                "response": "",
                "usage": {},
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        raise
    finally:
        if context is None:
            await provider.aclose()


def provider_key_env(model: str | None = None) -> str:
    """Name of the env var the CONFIGURED provider needs — 'ANTHROPIC_API_KEY'
    or 'OPENAI_API_KEY'.

    Callers that want to skip an LLM step when unconfigured have to ask which
    key matters, not assume. ``ask`` assumed OPENAI_API_KEY and so skipped
    synthesis on an ``provider: anthropic`` install holding a valid Anthropic
    key -- the check and the call disagreed about who was being called.
    """
    from config import get_config
    from cli.generate import resolve_provider
    cfg = get_config()
    provider_name = resolve_provider(
        cli_provider=None,
        cfg_provider=getattr(cfg, 'configured_provider', None),
        model=model or cfg.model,
    )
    return ('ANTHROPIC_API_KEY' if provider_name == 'anthropic'
            else 'OPENAI_API_KEY')


def has_provider_key(model: str | None = None) -> bool:
    """Whether the configured provider's key is present in the environment."""
    return bool(os.environ.get(provider_key_env(model), ''))


_completion_usage = contextvars.ContextVar("ariadne_completion_usage", default=None)


@contextmanager
def capture_completion_usage():
    rows = []
    token = _completion_usage.set(rows)
    try:
        yield rows
    finally:
        _completion_usage.reset(token)
_completion_trace = contextvars.ContextVar(
    "ariadne_completion_trace", default=None)


@contextmanager
def capture_completion_trace(*, enabled: bool = True):
    """Capture exact completion requests and raw replies for offline diagnosis."""
    rows = []
    if not enabled:
        yield rows
        return
    token = _completion_trace.set(rows)
    try:
        yield rows
    finally:
        _completion_trace.reset(token)
