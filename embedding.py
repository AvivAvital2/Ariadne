"""Embedding generation for the Claude Library.

This module provides functions to generate vector embeddings using OpenAI's API.
"""
from __future__ import annotations

__all__ = ['EmbeddingConfig', 'EmbeddingService', 'embed_batch_sync', 'embed_sync']

import logging
import os
from typing import TYPE_CHECKING

import httpx
import numpy as np
from attrs import frozen

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_MODEL = 'text-embedding-3-large'
EMBEDDING_DIM = 3072
def _parse_retry_after(response: httpx.Response) -> float | None:
    """Seconds to wait per the server hint (Retry-After header or body note), else None."""
    import re
    header = response.headers.get('retry-after')
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r'try again in ([\d.]+)\s*(ms|s)', response.text or '')
    if match:
        value = float(match.group(1))
        return value / 1000 if match.group(2) == 'ms' else value
    return None


def _retry_delay_for(response: httpx.Response, attempt: int) -> float:
    """Honor the server hint (+ small buffer, capped); else exponential backoff."""
    hint = _parse_retry_after(response)
    if hint is not None:
        return min(hint + 0.1, 30.0)
    return 2 ** attempt


@frozen
class EmbeddingConfig:
    """Configuration for the embedding service.

    Attributes:
        model: The embedding model to use.
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        base_url: Base URL for the API. Defaults to OpenAI's API.
        dimensions: Number of dimensions for the embedding. None uses model default.
    """
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None  # None = read from OPENAI_BASE_URL env or default
    dimensions: int | None = EMBEDDING_DIM

    def get_base_url(self) -> str:
        """Get API base URL from config, env var, or default."""
        if self.base_url is not None:
            return self.base_url
        return os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')

    def get_api_key(self) -> str:
        """Get the API key, falling back to environment variable."""
        if self.api_key is not None:
            return self.api_key
        key = os.environ.get('OPENAI_API_KEY')
        if key is None:
            raise ValueError('No API key provided and OPENAI_API_KEY environment variable not set')
        return key


class EmbeddingService:
    """Service for generating embeddings using OpenAI's API.

    This class uses httpx for HTTP requests, following the codebase patterns.

    Example:
        >>> service = EmbeddingService()
        >>> embedding = await service.embed("Hello world")
        >>> embeddings = await service.embed_batch(["Hello", "World"])
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize the embedding service.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        self.config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.get_base_url(),
                headers={
                    'Authorization': f'Bearer {self.config.get_api_key()}',
                    'Content-Type': 'application/json',
                },
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> EmbeddingService:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def embed(self, text: str) -> NDArray[np.float32]:
        """Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            A numpy array of shape (dimensions,) containing the embedding.
        """
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimensions,).
        """
        if not texts:
            return []

        # Filter out empty/whitespace-only texts (waste of API calls)
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            _logger.warning('All texts were empty/whitespace — skipping embedding API call')
            return [np.zeros(self.config.dimensions or 1536, dtype=np.float32)] * len(texts)

        texts = valid_texts

        import asyncio

        client = await self._get_client()

        payload: dict[str, object] = {
            'model': self.config.model,
            'input': texts,
        }
        if self.config.dimensions is not None:
            payload['dimensions'] = self.config.dimensions

        import httpx

        max_retries = 6
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = await client.post('/embeddings', json=payload)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                # Surface the response body — OpenAI's 4xx errors carry a
                # structured JSON message that explains *why* the request
                # was rejected (token-limit, malformed input, etc.). Without
                # this, callers see only "400 Bad Request" and have nothing
                # to act on.
                body = (e.response.text or '').strip()
                detail = f'{e} body={body[:500]}' if body else str(e)
                last_error = RuntimeError(
                    f'Embedding API HTTP {e.response.status_code}: {detail}'
                )
                # Most 4xx are permanent client errors — bail out. But 429
                # (rate limit) and 408 (timeout) are transient: fall through
                if 400 <= e.response.status_code < 500 and e.response.status_code not in (408, 429):
                    raise last_error from e
                if attempt < max_retries - 1:
                    delay = _retry_delay_for(e.response, attempt)  # 1s, 2s, 4s
                    _logger.debug(
                        'Embedding API request failed (attempt %d/%d), retrying in %ds: %s',
                        attempt + 1, max_retries, delay, detail,
                    )
                    await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # 1s, 2s, 4s
                    _logger.debug(
                        'Embedding API request failed (attempt %d/%d), retrying in %ds: %s',
                        attempt + 1, max_retries, delay, e,
                    )
                    await asyncio.sleep(delay)
        else:
            raise last_error  # type: ignore[misc]

        data = response.json()
        embeddings: list[NDArray[np.float32]] = []

        # Sort by index to ensure order matches input
        sorted_data = sorted(data['data'], key=lambda x: x['index'])
        for item in sorted_data:
            embedding = np.array(item['embedding'], dtype=np.float32)
            # Pre-normalize to unit vector so search can use dot product
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding)

        return embeddings


# Synchronous wrapper for convenience
def embed_sync(
    text: str,
    config: EmbeddingConfig | None = None,
) -> NDArray[np.float32]:
    """Generate an embedding synchronously.

    This is a convenience function for simple use cases. For batch processing,
    use EmbeddingService with async/await.

    Args:
        text: The text to embed.
        config: Optional embedding configuration.

    Returns:
        A numpy array of shape (dimensions,) containing the embedding.
    """
    import asyncio
    return asyncio.run(_embed_sync_impl(text, config))


async def _embed_sync_impl(
    text: str,
    config: EmbeddingConfig | None = None,
) -> NDArray[np.float32]:
    """Implementation of sync embedding."""
    async with EmbeddingService(config) as service:
        return await service.embed(text)


def embed_batch_sync(
    texts: list[str],
    config: EmbeddingConfig | None = None,
) -> list[NDArray[np.float32]]:
    """Generate embeddings for multiple texts synchronously.

    Args:
        texts: List of texts to embed.
        config: Optional embedding configuration.

    Returns:
        List of numpy arrays.
    """
    import asyncio
    return asyncio.run(_embed_batch_sync_impl(texts, config))


async def _embed_batch_sync_impl(
    texts: list[str],
    config: EmbeddingConfig | None = None,
) -> list[NDArray[np.float32]]:
    """Implementation of sync batch embedding."""
    async with EmbeddingService(config) as service:
        return await service.embed_batch(texts)
