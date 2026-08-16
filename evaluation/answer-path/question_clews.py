#!/usr/bin/env python
"""Give each clew the question it answers, so retrieval matches question-to-question.

A clew's default embedding surface is its route — ``fit -> withTransformEvent -> listenerBus``
— and a user's question is prose. Matching prose against symbol names is the weaker of the two
comparisons available; matching it against *another question* is what embeddings are good at.

So one question per theme, assigned to every clew anchored inside that theme. Themes are the
right unit for three measured reasons: the number a chain touches correlates with per-question
recall at **r = 0.80**; theme-anchored routes are the strongest single strategy at 78.4%; and
there are only 155 of them, so the whole corpus of questions costs ~155 short calls rather
than one per clew.

Many clews share a theme's question, which is intended. The question supplies topical match
and the route disambiguates within the topic — ``embed_clews`` embeds both together for
exactly that reason.

Entry-point clews get no question: they sit outside every theme, and inventing one from a
route would be generating the thing we are trying to retrieve by.

Usage
-----
    .venv/bin/python evaluation/answer-path/question_clews.py --dry-run
    ANTHROPIC_API_KEY=... .venv/bin/python evaluation/answer-path/question_clews.py --yes
"""
from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

#: Ask for the question, not a summary. The text becomes an embedding surface, so it has to
#: look like something a developer types — naming concrete things from the cluster.
PROMPT = (
    'Below is a cluster of related code from one codebase: its title, and the names of '
    'symbols it contains.\n\n'
    'Write the single question a developer is most likely to ask that this cluster answers. '
    'Reply with the question only — one sentence, no preamble, phrased the way a developer '
    'types it, naming concrete things from the cluster.\n\n'
    'Title: {title}\n\nSymbols: {symbols}'
)


async def ask_one(title: str, symbols: list) -> str:
    from llm import chat_complete
    reply = await chat_complete(
        messages=[
            {'role': 'system', 'content': 'You write the question a developer would ask.'},
            {'role': 'user', 'content': PROMPT.format(
                title=title, symbols=', '.join(symbols[:40]))},
        ],
        max_tokens=80)
    text = (reply or '').strip()
    return text.splitlines()[0].strip() if text else ''


def themes_with_members(conn, source: str):
    """``cluster_id -> (title, qualified names)`` via the deterministic catalog doc id."""
    from docgen.catalog_writer import _element_doc_id
    name = {}
    for cid, qn, owner in conn.execute(
            'SELECT canonical_id, qualified_name, source_name FROM scip_symbols'):
        if owner == source and not cid.startswith('local '):
            name[cid] = qn
    doc_of = {_element_doc_id(source, qn): cid for cid, qn in name.items()}
    titles = {cluster_id: title for cluster_id, title in conn.execute(
        'SELECT t.cluster_id, d.title FROM themes t JOIN documents d ON d.id = t.doc_id')}
    members = defaultdict(set)
    for cluster_id, element_id in conn.execute(
            'SELECT cluster_id, element_id FROM theme_members'):
        cid = doc_of.get(element_id)
        if cid is not None:
            members[cluster_id].add(name[cid])
    return {c: (titles.get(c, ''), names) for c, names in members.items()}


async def run(path: str, source: str, write: bool, limit: int | None) -> int:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    total = conn.execute('SELECT COUNT(*) FROM clews WHERE source_name = ?',
                         (source,)).fetchone()[0]
    if not total:
        print(f'no clews stored for {source} — run write_clews.py first')
        conn.close()
        return 1
    themes = themes_with_members(conn, source)
    entries = {row['entry_symbol'] for row in conn.execute(
        'SELECT DISTINCT entry_symbol FROM clews WHERE source_name = ?', (source,))}
    # Which themes actually own a clew's entry point; the rest cannot be assigned a question.
    owned = {c: (title, names & entries) for c, (title, names) in themes.items()}
    useful = {c: v for c, v in owned.items() if v[1]}
    reachable = len({e for _, names in useful.values() for e in names})
    print(f'{total:,} clews for {source}; {len(themes)} themes, '
          f'{len(useful)} of them own at least one clew entry point')
    print(f'  {reachable:,} of {len(entries):,} distinct entry points sit inside a theme')
    if not write:
        print(f'\ndry run: {len(useful)} questions would be generated, '
              f'no call made and nothing written')
        conn.close()
        return 0

    started = time.time()
    updated = 0
    for position, (cluster_id, (title, names)) in enumerate(
            sorted(useful.items())[:limit] if limit else sorted(useful.items()), start=1):
        question = await ask_one(title, sorted(n.split('.')[-1] for n in names))
        if not question:
            continue
        placeholders = ','.join('?' * len(names))
        cursor = conn.execute(
            f'UPDATE clews SET question = ? WHERE source_name = ? '
            f'AND entry_symbol IN ({placeholders})',
            [question, source, *sorted(names)])
        updated += cursor.rowcount
        conn.commit()
        if position % 20 == 0 or position == len(useful):
            print(f'  {position}/{len(useful)} themes  {updated:,} clews carry a question  '
                  f'({time.time() - started:.0f}s)', flush=True)
    with_question = conn.execute(
        'SELECT COUNT(*) FROM clews WHERE source_name = ? AND question IS NOT NULL',
        (source,)).fetchone()[0]
    conn.close()
    print(f'\n{with_question:,} of {total:,} clews now carry a question '
          f'in {time.time() - started:.0f}s')
    print('next: embed_clews.py — it embeds the question and the route together')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--db', default=None, help='store to update; default ./ariadne.db')
    parser.add_argument('--limit', type=int, default=None, help='first N themes only')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--yes', action='store_true', help='required to spend and write')
    args = parser.parse_args()
    if not args.dry_run and not args.yes:
        print('refusing to spend without --yes (or use --dry-run)')
        return 2
    return asyncio.run(run(args.db or str(ROOT / 'ariadne.db'), args.source,
                           not args.dry_run, args.limit))


if __name__ == '__main__':
    raise SystemExit(main())
