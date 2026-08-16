#!/usr/bin/env python
"""Do many narrow walks, pooled, cover what one question-seeded walk cannot? Free by default.

Measured this session: a walk seeded by a real question reaches 66.7% of its own required
symbols, and **pooling the 24 walks reaches 79.2%** — different questions start in different
places, so together they cover what none reaches alone. That is the entire case for
pre-generating clews from questions rather than from graph shape, and it is the only measured
result that beats the live walk.

The unproven link is whether *generated* questions position walks as well as real ones. This
tests the half that needs no LLM: a theme is a semantic cluster of symbols, so a walk seeded
from one theme's members is what a question about that theme would plausibly reach. Pool 155
such walks and score against answer keys fixed in advance:

    lands near 79%   generated anchors are as good as real questions, and the loop holds
    lands near 57%   generated anchors are not a substitute, and the idea stops here

``--live`` additionally asks the model for one question per theme, which is the text a clew
would embed so that retrieval matches question-to-question instead of question-to-route. That
part spends and needs a provider key; the coverage number above does not.

Usage
-----
    .venv/bin/python evaluation/answer-path/pool_from_themes.py
    .venv/bin/python evaluation/answer-path/pool_from_themes.py --limit 40
    .venv/bin/python evaluation/answer-path/pool_from_themes.py --live --json pool.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from recall_matching import hits  # noqa: E402

KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'

#: Ask for the question a theme answers, not a summary of it. The text is the clew's embedding
#: surface, so it has to look like something a user would type.
QUESTION_PROMPT = (
    'Below is a cluster of code from one codebase, described by its title and the names of '
    'symbols it contains. Write the single question a developer would most likely ask that '
    'this cluster answers.\n\n'
    'Reply with the question only — no preamble, one sentence, phrased the way a developer '
    'types it, naming concrete things from the cluster.\n\n'
    'Title: {title}\n\nSymbols: {symbols}'
)


def theme_members(conn, source: str):
    """``cluster_id -> (title, member symbol ids)``, resolved through the catalog doc id.

    A theme records members as document ids. ``_element_doc_id`` is deterministic, so the map
    back to symbols is built by hashing every symbol's name rather than by parsing documents.
    """
    from docgen.catalog_writer import _element_doc_id
    name: dict = {}
    for cid, qn, owner in conn.execute(
            'SELECT canonical_id, qualified_name, source_name FROM scip_symbols'):
        if owner == source and not cid.startswith('local '):
            name[cid] = qn
    doc_of = {_element_doc_id(source, qn): cid for cid, qn in name.items()}
    titles = {cluster_id: title for cluster_id, title in conn.execute(
        'SELECT t.cluster_id, d.title FROM themes t JOIN documents d ON d.id = t.doc_id')}
    clusters: dict = defaultdict(set)
    for cluster_id, element_id in conn.execute(
            'SELECT cluster_id, element_id FROM theme_members'):
        cid = doc_of.get(element_id)
        if cid is not None:
            clusters[cluster_id].add(cid)
    return name, {c: (titles.get(c, ''), members) for c, members in clusters.items()}


async def ask_question(title: str, symbols: list) -> str:
    from llm import chat_complete
    reply = await chat_complete(
        messages=[
            {'role': 'system', 'content': 'You write the question a developer would ask.'},
            {'role': 'user', 'content': QUESTION_PROMPT.format(
                title=title, symbols=', '.join(symbols[:40]))},
        ],
        max_tokens=80)
    return (reply or '').strip().splitlines()[0] if reply else ''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--seeds-per-theme', type=int, default=200,
                        help='cap seeds so one broad theme cannot become the whole graph')
    parser.add_argument('--limit', type=int, default=None, help='first N themes only')
    parser.add_argument('--live', action='store_true',
                        help='also generate one question per theme (spends)')
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    from library.structural_assembly import chain_from_seeds

    wanted_by_q = {}
    for entry in json.loads(KEY.read_text(encoding='utf-8')):
        symbols = [r['symbol'] for r in entry.get('required', [])]
        if symbols:
            wanted_by_q[entry['id']] = symbols
    wanted_all = [s for symbols in wanted_by_q.values() for s in symbols]

    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    started = time.time()
    name, clusters = theme_members(conn, args.source)
    print(f'{len(clusters)} themes over {len(name):,} symbols '
          f'in {time.time() - started:.1f}s')
    ordered = sorted(clusters.items(), key=lambda kv: (-len(kv[1][1]), kv[0]))
    if args.limit:
        ordered = ordered[:args.limit]

    pooled: set = set()
    rows: list = []
    started = time.time()
    for position, (cluster_id, (title, members)) in enumerate(ordered, start=1):
        seeds = sorted(members)[:args.seeds_per_theme]
        citations, _ = chain_from_seeds(conn, seeds, source=args.source, depth=args.depth)
        reached = {c.qualified_name for c in citations}
        pooled |= reached
        question = ''
        if args.live:
            question = asyncio.run(ask_question(
                title, [name[m].split('.')[-1] for m in seeds]))
        rows.append({'cluster_id': cluster_id, 'title': title, 'members': len(members),
                     'seeds': len(seeds), 'symbols': len(reached),
                     'hops': len(citations), 'question': question})
        if position % 10 == 0 or position == len(ordered):
            covered = sum(len(hits(symbols, pooled)) for symbols in wanted_by_q.values())
            print(f'  {position:>4}/{len(ordered)} themes  pooled '
                  f'{len(pooled):>7,} symbols  required '
                  f'{covered:>3}/{len(wanted_all)} ({covered/len(wanted_all):.1%})',
                  flush=True)
    conn.close()

    covered = sum(len(hits(symbols, pooled)) for symbols in wanted_by_q.values())
    print('\n' + '=' * 82)
    print(f'POOLED THEME-SEEDED WALKS — {len(rows)} themes'.center(82))
    print('=' * 82)
    print(f'  distinct symbols pooled        {len(pooled):>8,}')
    print(f'  required symbols covered       {covered:>4}/{len(wanted_all)}  '
          f'({covered/len(wanted_all):.1%})')
    print(f'  walked in {time.time() - started:.0f}s\n')
    print('  for comparison, on the same key:')
    print('    one question-seeded walk       66.7%')
    print('    24 question-seeded walks pooled 79.2%')
    print('    theme-anchored clews (no walk)  57.7%')
    if args.live:
        asked = [r for r in rows if r['question']]
        print(f'\n  questions generated: {len(asked)} of {len(rows)}')
        for row in asked[:6]:
            print(f'    {row["title"][:34]:<34} {row["question"][:60]}')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'\n  written: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
