#!/usr/bin/env python
"""Embed stored clews so a question can match a route. Uses the configured embedding model.

Route generation performs no inference. Embedding uses the model pair selected by
``ARIADNE_CONFIG``; local endpoints do not require a cloud provider key. Completed
batches are retained if a later batch fails.

What gets embedded is the clew's ``question`` **and** its route together. The question does
the topical matching — question-to-question is what embeddings are good at — and the route
breaks ties between clews that share a theme's question. A clew with no question embeds its
route alone; that still works, a route being a sequence of verbs and nouns from the domain,
but it is the weaker surface.

A clew with no vector is skipped by ``nearest_clews`` rather than treated as distance zero, so
running this over part of the set is safe and resumable: only unembedded rows are fetched.

Usage
-----
    .venv/bin/python evaluation/answer-path/embed_clews.py --source databricks --dry-run
    OPENAI_API_KEY=... .venv/bin/python evaluation/answer-path/embed_clews.py --yes
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def surface(row) -> str:
    """The text a question is compared against: the generated question *and* the route.

    Both, not one. The question carries the topic and is shared across a theme's clews, so on
    its own it cannot distinguish two routes inside the same theme; the route distinguishes
    them but is symbol names rather than prose. Concatenated, the question does the topical
    matching and the route breaks the tie.

    A clew with no question falls back to its route alone — entry-point clews sit outside every
    theme and are deliberately left unquestioned rather than given an invented one.
    """
    from library.clew_embeddings import clew_embedding_surface

    return clew_embedding_surface(row['steps'], row['question'])


async def run(path: str | None, source: str, batch: int, limit: int | None, write: bool,
              *, staleness_db: str | None = None) -> int:
    from config import get_config
    from docgen.pricing import CHARS_PER_TOKEN
    from inference import InferenceContext
    from library.clew_embeddings import clew_embedding_surface, embed_clews, unembedded_clews

    if batch <= 0 or (limit is not None and limit <= 0):
        raise ValueError('Batch size and limit must be positive')
    context = InferenceContext.from_config(get_config(), db=path, staleness_db=staleness_db)
    try:
        with context.write_session() if write else nullcontext():
            library = context.open_library(operation='write' if write else 'plan')
            rows = unembedded_clews(library, source, limit)
            chars = sum(len(clew_embedding_surface(steps, question))
                        for _, steps, question in rows)
            with_question = sum(bool((question or '').strip()) for _, _, question in rows)
            print(f'{len(rows):,} clews to embed for {source}')
            print(f'  {with_question:,} carry a generated question, '
                  f'{len(rows) - with_question:,} embed their route instead')
            print(f'  {chars:,} chars ~= {chars / CHARS_PER_TOKEN:,.0f} tokens')
            if not write:
                print('\ndry run: nothing embedded, nothing written')
                return 0
            started = time.time()

            def progress(done: int, total: int) -> None:
                print(f'  embedded {done:,}/{total:,} '
                      f'({time.time() - started:.0f}s)', flush=True)

            await embed_clews(library, source_name=source, batch_size=batch,
                             limit=limit, progress=progress)
            with library._conn_provider.acquire() as conn:
                remaining = conn.execute(
                    'SELECT COUNT(*) FROM clews WHERE source_name=? AND embedding IS NULL',
                    (source,),
                ).fetchone()[0]
            print(f'\ndone in {time.time() - started:.0f}s; {remaining:,} still unembedded')
            return 0
    finally:
        await context.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--db', default=None, help='store to update; defaults to the configured library')
    parser.add_argument('--staleness-db', default=None, help='Explicit staleness database')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--yes', action='store_true', help='required to write vectors')
    args = parser.parse_args()
    if not args.dry_run and not args.yes:
        print('refusing to spend without --yes (or use --dry-run)')
        return 2
    return asyncio.run(run(args.db, args.source, args.batch, args.limit,
                           not args.dry_run, staleness_db=args.staleness_db))


if __name__ == '__main__':
    raise SystemExit(main())
