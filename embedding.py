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

from model_config import EMBEDDING_DIM, EmbeddingSettings

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from inference import InferenceContext

_logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_MODEL = 'text-embedding-3-large'
VECTOR_NORM_TOLERANCE = 1e-5

# Max chars of a document's content folded into its embedding text.
_DOC_EMBED_CONTENT_CHARS = 2000


def doc_embedding_text(title: str, content: str) -> str:
    """The canonical text a document is embedded over: title + truncated
    content. The ONE source of truth for doc-embedding input, shared by the
    embedding writer and the spool pack's re-embed fallback so their vectors
    live in the same space (two hand-rolled copies had already drifted —
    see the spool CRIT-10 re-embed regression)."""
    return f'{title}\n\n{(content or "")[:_DOC_EMBED_CONTENT_CHARS]}'
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
    provider: str = 'openai'
    local_settings: EmbeddingSettings | None = None

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings) -> EmbeddingConfig:
        return cls(
            model=settings.request_model,
            base_url=settings.base_url,
            dimensions=settings.dimensions,
            provider=settings.provider,
            local_settings=settings if settings.provider == 'local' else None,
            api_key=settings.credential(),
        )

    def get_base_url(self) -> str:
        """Get API base URL from config, env var, or default."""
        if self.provider == 'local' and not self.base_url:
            raise ValueError('Local embedding base_url is required')
        if self.base_url is not None:
            return self.base_url
        if self.provider == 'deepinfra':
            from model_config import DEEPINFRA_URL

            return DEEPINFRA_URL
        return os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')

    def get_api_key(self) -> str:
        """Get the API key, falling back to environment variable."""
        if self.provider == 'local':
            return self.local_settings.credential() or ''
        if self.api_key is not None:
            return self.api_key
        from model_config import credential_env

        name = credential_env(self.provider)
        key = os.environ.get(name)
        if key is None:
            raise ValueError(f'No API key provided and {name} environment variable not set')
        return key


class EmbeddingService:
    """Service for generating embeddings using OpenAI's API.

    This class uses httpx for HTTP requests, following the codebase patterns.

    Example:
        >>> service = EmbeddingService()
        >>> embedding = await service.embed("Hello world")
        >>> embeddings = await service.embed_batch(["Hello", "World"])
    """

    def __init__(self, config: EmbeddingConfig | None = None, *, context: InferenceContext | None = None) -> None:
        """Initialize the embedding service.

        Args:
            config: Embedding configuration. Uses defaults if not provided.
        """
        from inference import current_context
        from library.model_binding import ModelBindingMismatch

        self.context = context or current_context()
        if config is None and self.context is not None:
            config = EmbeddingConfig.from_settings(self.context.settings.embeddings)
        self.config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._local_transport = None
        self.embedding_space_id = None
        if self.config.provider == 'local':
            if self.context is None or self.config.local_settings is None:
                raise ModelBindingMismatch('Local embeddings require a bound InferenceContext')
            self.embedding_space_id = self.config.local_settings.embedding_space_id
            self.context.validate_producer(
                model=self.config.model,
                provider='local',
                embedding_space_id=self.embedding_space_id,
            )
            from docgen.llm.local import LocalTransport

            self._local_transport = LocalTransport(
                self.config.local_settings, attempts=self.context.settings.execution.max_attempts
            )
        elif self.context is not None and self.context.local:
            raise ModelBindingMismatch(
                'A hosted embedding service cannot be used in a local context'
            )
        elif self.context is not None and self.context.binding is not None:
            expected = self.context.settings.embeddings
            if (self.config.provider, self.config.model, self.config.dimensions) != (
                expected.provider,
                expected.request_model,
                expected.dimensions,
            ):
                raise ModelBindingMismatch(
                    'Embedding configuration differs from the bound model or dimensions'
                )
            self.embedding_space_id = expected.embedding_space_id

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.get_base_url(),
                headers={'Authorization': f'Bearer {self.config.get_api_key()}'},
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._local_transport is not None:
            await self._local_transport.aclose()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> EmbeddingService:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def embed(self, text: str, *, purpose: str | None = None) -> NDArray[np.float32]:
        """Generate an embedding for a single text.

    Identical texts within one service lifetime resolve to one
    provider call: the ask-path stages (document search, catalog
    positioning, clew recall, compact dispatch) all embed the same
    question, and only the first may spend.
    """
        memo = getattr(self, "_embed_memo", None)
        if memo is None:
            memo = self._embed_memo = {}
        cached = memo.get((purpose, text))
        if cached is not None:
            return cached
        embeddings = await self.embed_batch([text], **({'purpose': purpose} if purpose is not None else {}))
        memo[purpose, text] = embeddings[0]
        if len(memo) > 32:
            memo.pop(next(iter(memo)))
        return embeddings[0]

    async def embed_batch(self, texts: list[str], *, purpose: str | None = None) -> list[NDArray[np.float32]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of numpy arrays, each of shape (dimensions,).
        """
        if self.config.provider == 'local':
            if purpose == 'query':
                return [await self.embed_query(text) for text in texts]
            if purpose == 'document':
                return await self.embed_documents(texts)
            raise ValueError(
                'Local embed/embed_batch requires an explicit query or document purpose'
            )
        if self.config.provider == 'deepinfra':
            from model_config import query_input

            purpose = purpose or 'document'
            if purpose not in ('query', 'document'):
                raise ValueError('Qwen embeddings require an explicit query or document purpose')
            if any(not isinstance(text, str) or not text.strip() for text in texts):
                raise ValueError('Qwen embedding inputs must not be empty')
            if purpose == 'query':
                texts = [query_input(text) for text in texts]
            batch_size = self.context.settings.embeddings.batch_size if self.context else 16
            results = []
            for start in range(0, len(texts), batch_size):
                results.extend(await self._embed_cloud(texts[start:start + batch_size]))
            return results
        return await self._embed_cloud(texts)

    async def _embed_cloud(self, texts: list[str]) -> list[NDArray[np.float32]]:
        if not texts:
            return []
        # Empty/whitespace texts get zero vectors IN PLACE — the API would
        # reject them, and the result must stay positionally aligned with the
        # input (callers zip vectors onto their doc/chunk lists, so a shifted
        # result silently assigns wrong embeddings).
        dim = self.config.dimensions or 1536
        valid_positions = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_positions:
            _logger.warning('All texts were empty/whitespace — skipping embedding API call')
            return [np.zeros(dim, dtype=np.float32)] * len(texts)
        original_count = len(texts)
        texts = [texts[i] for i in valid_positions]

        import asyncio

        client = await self._get_client()

        payload: dict[str, object] = {
            'model': self.config.model,
            'input': texts,
        }
        if self.config.provider == 'deepinfra':
            payload['encoding_format'] = 'float'
        elif self.config.dimensions is not None:
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
        if self.config.provider == 'deepinfra':
            return reduce_local_embeddings(data, len(texts))
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
        if len(valid_positions) != original_count:
            aligned = [np.zeros(dim, dtype=np.float32) for _ in range(original_count)]
            for position, vector in zip(valid_positions, embeddings):
                aligned[position] = vector
            return aligned
        return embeddings

    async def embed_query(self, text: str) -> NDArray[np.float32]:
        if self.config.provider == 'deepinfra':
            return await self.embed(text, purpose='query')
        if self.config.provider != 'local':
            return await self.embed(text)
        from model_config import query_input

        return (await self._embed_local([query_input(text)]))[0]

    async def embed_documents(self, texts: list[str]) -> list[NDArray[np.float32]]:
        if self.config.provider != 'local':
            return await self.embed_batch(list(texts))
        return await self._embed_local(list(texts))

    async def _embed_local(
        self, texts: list[str], *, attempts: int | None = None
    ) -> list[NDArray[np.float32]]:
        from docgen.llm.local import ContextOverflow, token_count
        from model_config import tokenize_url

        invalid = [
            i for i, text in enumerate(texts) if not isinstance(text, str) or not text.strip()
        ]
        if invalid:
            raise ValueError(f'Empty embedding inputs at indices {invalid}')
        if not texts:
            return []
        self.context.validate(credentials=True)
        attempts = attempts or self.context.transport_attempts
        self.context.validate_producer(
            model=self.config.model, provider='local', embedding_space_id=self.embedding_space_id
        )
        settings = self.config.local_settings
        transport = self._local_transport
        results = []
        async with self.context.embedding_limit:
            await transport.ready(attempts=attempts)
            counts = []
            for text in texts:
                data = await transport.request(
                    'POST',
                    tokenize_url(settings.base_url),
                    payload={
                        'model': settings.request_model,
                        'prompt': text,
                        'add_special_tokens': True,
                    },
                    attempts=attempts,
                )
                count, server_limit = token_count(data)
                if count > min(settings.max_input_tokens, settings.batch_max_tokens, server_limit):
                    raise ContextOverflow(
                        f'Embedding input {len(counts)} exceeds the configured token limit'
                    )
                counts.append(count)
            start = 0
            while start < len(texts):
                end, total = start, 0
                while (
                    end < len(texts)
                    and end - start < settings.batch_size
                    and total + counts[end] <= settings.batch_max_tokens
                ):
                    total += counts[end]
                    end += 1
                data = await transport.request(
                    'POST',
                    settings.base_url + '/embeddings',
                    payload={
                        'model': settings.request_model,
                        'input': texts[start:end],
                        'encoding_format': 'float',
                    },
                    attempts=attempts,
                )
                results.extend(reduce_local_embeddings(data, end - start))
                start = end
        return results


def reduce_local_embeddings(data: dict, expected: int) -> list[NDArray[np.float32]]:
    """Validate native Qwen output, retain its prefix, then normalize in float64."""
    from docgen.llm.local import EmbeddingContractError
    from model_config import NATIVE_DIM

    rows = data.get('data')
    if not isinstance(rows, list) or len(rows) != expected:
        raise EmbeddingContractError('Wrong embedding response cardinality')
    if any(not isinstance(row, dict) or type(row.get('index')) is not int for row in rows):
        raise EmbeddingContractError('Missing or invalid embedding indices')
    rows = sorted(rows, key=lambda row: row['index'])
    if any(row['index'] != index for index, row in enumerate(rows)):
        raise EmbeddingContractError('Duplicate, missing or out-of-range embedding indices')
    result = []
    for row in rows:
        values = row.get('embedding')
        if (
            not isinstance(values, list)
            or len(values) != NATIVE_DIM
            or any(type(value) not in (float, int) for value in values)
        ):
            raise EmbeddingContractError(
                f'Local endpoint must return {NATIVE_DIM} numeric components'
            )
        try:
            native = np.asarray(values, dtype=np.float64)
        except (ValueError, OverflowError) as exc:
            raise EmbeddingContractError('Invalid embedding components') from exc
        if not np.isfinite(native).all():
            raise EmbeddingContractError('Nonfinite native embedding')
        prefix = native[:EMBEDDING_DIM]
        scale = np.max(np.abs(prefix))
        if scale == 0:
            raise EmbeddingContractError('Zero retained embedding norm')
        scaled = prefix / scale
        normalized = (scaled / np.linalg.norm(scaled)).astype(np.float32)
        if (
            not np.isfinite(normalized).all()
            or abs(float(np.linalg.norm(normalized.astype(np.float64))) - 1) > VECTOR_NORM_TOLERANCE
        ):
            raise EmbeddingContractError('Invalid reduced embedding norm')
        result.append(normalized)
    return result


# Synchronous wrapper for convenience
def embed_sync(
    text: str,
    config: EmbeddingConfig | None = None,
    *,
    purpose: str | None = None,
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
    return asyncio.run(_embed_sync_impl(text, config, purpose=purpose))


async def _embed_sync_impl(
    text: str,
    config: EmbeddingConfig | None = None,
    *,
    purpose: str | None = None,
) -> NDArray[np.float32]:
    """Implementation of sync embedding."""
    async with EmbeddingService(config) as service:
        return await service.embed(text, purpose=purpose)


def embed_batch_sync(
    texts: list[str],
    config: EmbeddingConfig | None = None,
    *,
    purpose: str | None = None,
) -> list[NDArray[np.float32]]:
    """Generate embeddings for multiple texts synchronously.

    Args:
        texts: List of texts to embed.
        config: Optional embedding configuration.

    Returns:
        List of numpy arrays.
    """
    import asyncio
    return asyncio.run(_embed_batch_sync_impl(texts, config, purpose=purpose))


async def _embed_batch_sync_impl(
    texts: list[str],
    config: EmbeddingConfig | None = None,
    *,
    purpose: str | None = None,
) -> list[NDArray[np.float32]]:
    """Implementation of sync batch embedding."""
    async with EmbeddingService(config) as service:
        return await service.embed_batch(texts, purpose=purpose)
