#!/usr/bin/env python
"""Embed stored clews so a question can match a route. Needs an embedding provider key.

Generation is local and free; this is the one paid step, and it is small — measured, ~20,000
clews average 238 characters, about 2.3M tokens, roughly $0.30 with ``text-embedding-3-large``
(half that batched). One time, at pack build.

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
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

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
    steps = ' -> '.join(json.loads(row['steps']))
    question = (row['question'] or '').strip()
    return f'{question}\n{steps}' if question else steps


async def run(path: str, source: str, batch: int, limit: int | None, write: bool) -> int:
    from docgen.pricing import CHARS_PER_TOKEN
    from embedding import EmbeddingConfig, EmbeddingService

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT id, steps, question FROM clews '
        'WHERE source_name = ? AND embedding IS NULL', (source,)).fetchall()
    if limit:
        rows = rows[:limit]
    texts = [surface(row) for row in rows]
    chars = sum(len(text) for text in texts)
    with_question = sum(1 for row in rows if (row['question'] or '').strip())
    print(f'{len(rows):,} clews to embed for {source}')
    print(f'  {with_question:,} carry a generated question, '
          f'{len(rows) - with_question:,} embed their route instead')
    print(f'  {chars:,} chars ~= {chars / CHARS_PER_TOKEN:,.0f} tokens')
    if not rows:
        conn.close()
        return 0
    if not write:
        print('\ndry run: nothing embedded, nothing written')
        conn.close()
        return 0

    service = EmbeddingService(EmbeddingConfig())
    started = time.time()
    done = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        vectors = await service.embed_batch([surface(row) for row in chunk])
        for row, vector in zip(chunk, vectors):
            conn.execute('UPDATE clews SET embedding = ? WHERE id = ?',
                         (np.asarray(vector, dtype=np.float32).tobytes(), row['id']))
        conn.commit()
        done += len(chunk)
        print(f'  embedded {done:,}/{len(rows):,} '
              f'({time.time() - started:.0f}s)', flush=True)
    remaining = conn.execute(
        'SELECT COUNT(*) FROM clews WHERE source_name = ? AND embedding IS NULL',
        (source,)).fetchone()[0]
    conn.close()
    print(f'\ndone in {time.time() - started:.0f}s; {remaining:,} still unembedded')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--db', default=None, help='store to update; default ./ariadne.db')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--yes', action='store_true', help='required to write vectors')
    args = parser.parse_args()
    if not args.dry_run and not args.yes:
        print('refusing to spend without --yes (or use --dry-run)')
        return 2
    return asyncio.run(run(args.db or str(ROOT / 'ariadne.db'), args.source,
                           args.batch, args.limit, not args.dry_run))


if __name__ == '__main__':
    raise SystemExit(main())
