"""Protocol for LLM provider implementations.

Each provider (OpenAI, Anthropic) implements ``call`` to perform a
single chat-completion-equivalent request and ``aclose`` to release
the HTTP client. Output-shaping params (max_tokens) are
passed per-call; the model, base_url, and api_key are bound at
construction.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from attrs import define


@define
class CacheStats:
    """Per-provider running counters for prompt caching outcomes.

    Anthropic populates this from the ``usage.cache_creation_input_tokens``
    / ``cache_read_input_tokens`` fields on each ``/messages`` response.
    OpenAI leaves all counters at 0 (no caching equivalent today).

    Surfaces in ``PipelineResult.cache_stats`` so the CLI can print a
    one-line summary regardless of log level — log-grep verification
    isn't a great UX.
    """
    cache_writes: int = 0
    cache_reads: int = 0
    cache_misses: int = 0
    total_calls: int = 0
    total_create_tokens: int = 0
    total_read_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of calls served from cache (excludes warm-up writes)."""
        return self.cache_reads / self.total_calls if self.total_calls else 0.0

    def record(self, *, create_tokens: int, read_tokens: int) -> None:
        """Update counters from one response's usage block.

        Classification: ``create_tokens > 0`` means the call wrote a new
        cache entry (the warm-up call for that prefix); ``read_tokens > 0``
        means a hit; both 0 means caching wasn't engaged (e.g., prompt
        below the 1024-token minimum).
        """
        self.total_calls += 1
        self.total_create_tokens += create_tokens
        self.total_read_tokens += read_tokens
        if create_tokens > 0:
            self.cache_writes += 1
        elif read_tokens > 0:
            self.cache_reads += 1
        else:
            self.cache_misses += 1


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-specific LLM completion interface.

    Implementations encapsulate the request/response shape of their
    backend (OpenAI ``/chat/completions`` vs Anthropic ``/v1/messages``).
    Callers see a uniform ``call`` signature.
    """

    model: str
    api_key: str
    base_url: str

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        cache_system_prompt: bool = True,
    ) -> str | None:
        """Run a chat completion and return the assistant's text.

        Args:
            system_prompt: System role content.
            user_prompt: User role content.
            max_tokens: Cap on generated tokens.
            cache_system_prompt: Provider-specific hint to cache the static
                system prompt. Anthropic uses ``cache_control`` markers;
                OpenAI ignores this (no equivalent today). Default True.

        Returns:
            Generated text, or None if all retries failed.
        """
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        ...
