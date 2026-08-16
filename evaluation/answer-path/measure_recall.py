#!/usr/bin/env python
"""Re-measure every recall claim with boundary-aware matching. Deterministic, spends nothing.

Every recall figure produced earlier in this work used substring containment — the eval
harness's own rule — and it inflated all of them. ``DataSource`` matched
``DataSourceV2Relation``; ``MergeIntoTable`` matched 117 protobuf builder lines. So the
numbers below replace, and are not comparable with, these earlier claims:

    the chain reaches 74% of required symbols        the spine holds 52%
    rule G keeps 64.8%                              rule H keeps 81.7%

Input is recorded production retrieval (``evaluation/measurements/retrieval_dump.json``,
``ariadne_search`` at width 8, 2026-08-04) against answer keys fixed in advance by the eval
harness (``evaluation/spool-clean-room/chain_requirements.json``). Matching comes from
``recall_matching`` so no script carries its own copy again.

Two ceilings are reported because they answer different questions:

    of required   how much of the answer key the chain carries at all — the honest headline
    of reached    how much of what the chain found a rule keeps — how lossy the rule is

Usage
-----
    .venv/bin/python evaluation/answer-path/measure_recall.py
    .venv/bin/python evaluation/answer-path/measure_recall.py --source databricks --json r.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from recall_matching import ambiguous, hits, rule_sets  # noqa: E402

DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'
KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'
RULES = ('spine', 'G', 'H', 'chain')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    from config import get_config
    from docgen.scip_paths import indexer_cwds
    from library.structural_assembly import chain_from_seeds, seeds_from_documents

    dump = json.loads(DUMP.read_text(encoding='utf-8'))
    key = {entry['id']: entry for entry in json.loads(KEY.read_text(encoding='utf-8'))}
    root = str(get_config().get_all_source_paths().get(args.source) or '') or None
    cwds = indexer_cwds(root) if root else ()
    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)

    results = dump['results'][:args.limit] if args.limit else dump['results']
    rows: list[dict] = []
    for result in results:
        entry = key.get(result['id'], {})
        wanted = [r['symbol'] for r in entry.get('required', [])]
        documents = [{'content': d.get('content') or '',
                      'source_files': d.get('source_files') or []}
                     for d in result['widths']['8']['documents']]
        row = {'id': result['id'], 'family': result.get('family', ''),
               'required': len(wanted), 'chain_hops': 0, 'reached': 0, 'ambiguous': 0,
               'kept': {name: 0 for name in RULES},
               'symbols': {name: 0 for name in RULES}, 'missed_by_H': []}
        rows.append(row)
        if not wanted or not documents:
            continue
        seeds = seeds_from_documents(conn, documents, source=args.source,
                                     indexer_cwds=cwds, source_root=root)
        citations, _ = chain_from_seeds(conn, seeds.seeds, source=args.source,
                                       depth=args.depth)
        if not citations:
            continue
        sets = rule_sets(citations)
        reached = hits(wanted, sets['chain'])
        row['chain_hops'] = len(citations)
        row['reached'] = len(reached)
        row['ambiguous'] = len(ambiguous(reached, sets['chain']))
        for name in RULES:
            row['kept'][name] = len(hits(wanted, sets[name]))
            row['symbols'][name] = len(sets[name])
        row['missed_by_H'] = sorted(reached - hits(wanted, sets['H']))
    conn.close()

    print('\n' + '=' * 100)
    print(f'RECALL WITH BOUNDARY MATCHING — source {args.source}, depth {args.depth}'
          .center(100))
    print('=' * 100)
    print(f'{"q":>4} {"family":<16} {"hops":>6} {"req":>4} {"reach":>6} {"amb":>4} '
          f'{"spine":>6} {"G":>4} {"H":>4} {"chain":>6}  H missed')
    print('-' * 100)
    for row in rows:
        k = row['kept']
        print(f'{row["id"]:>4} {row["family"][:16]:<16} {row["chain_hops"]:>6,} '
              f'{row["required"]:>4} {row["reached"]:>6} {row["ambiguous"]:>4} '
              f'{k["spine"]:>6} {k["G"]:>4} {k["H"]:>4} {k["chain"]:>6}  '
              f'{",".join(row["missed_by_H"][:2])}')
    print('-' * 100)
    walked = [r for r in rows if r['chain_hops']]
    required = sum(r['required'] for r in walked)
    reached = sum(r['reached'] for r in walked)
    print(f'questions with a chain: {len(walked)} of {len(rows)}   '
          f'required symbols: {required}   ambiguous credits: '
          f'{sum(r["ambiguous"] for r in walked)}')
    print(f'\n{"rule":<8} {"symbols/q":>10} {"kept":>6} {"of required":>12} '
          f'{"of reached":>11}')
    print(f'{"chain":<8} {sum(r["symbols"]["chain"] for r in walked)/max(len(walked),1):>10.0f} '
          f'{reached:>6} {reached/max(required,1):>11.1%} {1.0:>11.1%}   <- the ceiling')
    for name in ('H', 'G', 'spine'):
        kept = sum(r['kept'][name] for r in walked)
        print(f'{name:<8} {sum(r["symbols"][name] for r in walked)/max(len(walked),1):>10.0f} '
              f'{kept:>6} {kept/max(required,1):>11.1%} {kept/max(reached,1):>11.1%}')
    print('\nThe chain row is what an unpruned menu offers. A rule below it trades recall for '
          'menu size,\nand the menu is already the cheap half of the two-call path.')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'\nwritten: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
