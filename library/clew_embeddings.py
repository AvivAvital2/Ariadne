"""Embed clew routes using their library's model pair and write ownership."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from library.model_binding import ModelBindingMismatch
from library.prepared import ObsoleteInput, validate_vector

if TYPE_CHECKING:
    from library import Library


def clew_embedding_surface(steps: str, question: str | None) -> str:
    route = ' -> '.join(json.loads(steps))
    question = (question or '').strip()
    return f'{question}\n{route}' if question else route


def unembedded_clews(library: Library, source_name: str, limit: int | None = None) -> list:
    if limit is not None and limit <= 0:
        raise ValueError('Clew limit must be positive')
    with library._conn_provider.acquire() as conn:
        return conn.execute(
            'SELECT id, steps, question FROM clews '
            'WHERE source_name=? AND embedding IS NULL ORDER BY id LIMIT ?',
            (source_name, limit if limit is not None else -1),
        ).fetchall()


async def embed_clews(
    library: Library,
    *,
    source_name: str,
    batch_size: int = 256,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Commit complete embedding batches; completed rows are skipped on retry."""
    if batch_size <= 0:
        raise ValueError('Clew batch size must be positive')
    context = library.inference_context
    if context is None:
        raise ModelBindingMismatch('Embedding clews requires the library inference context')
    context.require_writer()
    context.validate()
    rows = unembedded_clews(library, source_name, limit)
    done = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        vectors = await context.embeddings.embed_documents([
            clew_embedding_surface(steps, question) for _, steps, question in batch
        ])
        if len(vectors) != len(batch):
            raise ValueError('Embedding response count does not match the clew batch')
        for vector in vectors:
            if vector is None:
                raise ValueError('Missing clew embedding')
            validate_vector(vector)
        context.validate()
        with library._conn_provider.acquire() as conn:
            for (clew_id, steps, question), vector in zip(batch, vectors, strict=True):
                updated = conn.execute(
                    'UPDATE clews SET embedding=? WHERE id=? AND steps=? '
                    'AND question IS ? AND embedding IS NULL',
                    (vector.tobytes(), clew_id, steps, question),
                )
                if updated.rowcount != 1:
                    raise ObsoleteInput(f'Clew {clew_id} changed during embedding')
        from library.clews import clear_clew_embedding_cache

        clear_clew_embedding_cache()
        done += len(batch)
        if progress:
            progress(done, len(rows))
    return done
