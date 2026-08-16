#!/usr/bin/env python
"""How much of the answer key is code at all? Free, and it bounds every other figure here.

Every recall percentage in this directory is *reached / required*, and ``required`` comes from
``chain_requirements.json``. A caller-rooted walk left 22 of 96 unreached, and the list opens
``ALTER, ALWAYS, ANSI, COLUMN, CREATE`` — SQL grammar keywords. A call-graph walk cannot reach
a keyword at any depth or direction, because a keyword is not a node in the graph. Those
entries are not a deficiency in the walk; they are a property of the key.

So the denominator has to be split before anything else is judged:

    indexed       the symbol exists in ``scip_symbols`` for this source, so a walk could in
                  principle reach it — this is the honest denominator for walk recall
    not indexed   nothing by that name is in the index. Either a grammar keyword, a symbol in
                  a repo outside the corpus, or an ingest gap; the three are distinguished by
                  eye from the printed list, deliberately, rather than by a guess in code

This audits the evaluation, not the product, and it is reported separately for exactly that
reason. Reaching "100% of what is reachable" is the ceiling worth chasing; reaching 100% of a
key that asks for ``CREATE`` is not a goal any call graph can hold.

Usage
-----
    .venv/bin/python evaluation/answer-path/audit_answer_key.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'
DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'

#: Reserved words that appear in the key as required "symbols". Listed, not pattern-matched:
#: an all-caps test would also catch legitimate constants.
SQL_KEYWORDS = frozenset({
    'ALTER', 'ALWAYS', 'ANSI', 'BY', 'COLUMN', 'CREATE', 'DEFAULT', 'DELETE', 'DROP',
    'GENERATED', 'IDENTITY', 'INSERT', 'MERGE', 'REPLACE', 'SELECT', 'SET', 'START',
    'TABLE', 'UPDATE', 'USING', 'VALUES', 'WHEN', 'WHERE', 'WITH',
})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    args = parser.parse_args()

    entries = json.loads(KEY.read_text(encoding='utf-8'))
    dump = json.loads(DUMP.read_text(encoding='utf-8'))
    walked = {r['id'] for r in dump['results']
              if r['widths']['8']['documents']}

    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    segments: set = set()
    for (name,) in conn.execute(
            'SELECT DISTINCT qualified_name FROM scip_symbols WHERE source_name = ? '
            "AND canonical_id NOT LIKE 'local %'", (args.source,)):
        segments.update(name.split('.'))
    conn.close()

    verdict: Counter = Counter()
    missing: list[tuple] = []
    per_question: list[dict] = []
    for entry in entries:
        wanted = [r['symbol'] for r in entry.get('required', [])]
        if not wanted or entry['id'] not in walked:
            continue
        indexed = [s for s in wanted if s in segments]
        absent = [s for s in wanted if s not in segments]
        verdict['indexed'] += len(indexed)
        verdict['not indexed'] += len(absent)
        for symbol in absent:
            kind = 'sql keyword' if symbol in SQL_KEYWORDS else 'not in the index'
            verdict[kind] += 1
            missing.append((entry['id'], entry.get('family', ''), symbol, kind))
        per_question.append({'id': entry['id'], 'family': entry.get('family', ''),
                             'required': len(wanted), 'indexed': len(indexed)})

    total = verdict['indexed'] + verdict['not indexed']
    print('\n' + '=' * 84)
    print(f'ANSWER KEY COMPOSITION — source {args.source}'.center(84))
    print('=' * 84)
    print(f'required symbols across {len(per_question)} answered questions: {total}')
    print(f'  in the index (a walk could reach these): {verdict["indexed"]:>3} '
          f'({verdict["indexed"] / max(total, 1):.1%})')
    print(f'  not in the index                       : {verdict["not indexed"]:>3} '
          f'({verdict["not indexed"] / max(total, 1):.1%})')
    print(f'      of which SQL grammar keywords      : {verdict["sql keyword"]:>3}')
    print(f'      of which named but absent          : {verdict["not in the index"]:>3}')
    if missing:
        print(f'\n{"q":>4} {"family":<16} {"symbol":<34} why')
        print('-' * 84)
        for qid, family, symbol, kind in missing:
            print(f'{qid:>4} {family[:16]:<16} {symbol:<34} {kind}')
    print(f'\nQuestions whose key is entirely unreachable by a call graph:')
    dead = [q for q in per_question if not q['indexed']]
    for q in dead:
        print(f'  q{q["id"]} ({q["family"]}) — 0 of {q["required"]} required symbols indexed')
    if not dead:
        print('  none')
    print(f'\nWalk recall should be read against {verdict["indexed"]}, not {total}. '
          f'The difference is\nthe evaluation asking for things that are not nodes in a '
          f'call graph.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
