#!/usr/bin/env python
"""How far should the walk descend *from a caller*? The middle of the sweep, never measured.

Two points were measured and they are 32x apart in cost:

    callers cited, not walked (depth 0)   +5.2pp   +24 symbols/q
    callers rooted (depth 3)              +9.3pp  +758 symbols/q

Depth 0 buys 56% of the gain for 3% of the cost, so the interesting question is what sits
between. A dispatcher's branches — the three executors ``MergeIntoCommand`` chooses among —
are *one* hop from the dispatcher, not three, and a caller's deeper subtree is mostly sibling
work unrelated to the question. That predicts depth 1 captures nearly all of it.

Seeds keep their own depth throughout; only the callers' depth varies. The two walks are
unioned rather than merged into one call because ``chain_from_seeds`` takes a single depth for
all roots, and changing that contract to run a measurement would be the wrong way round.

Usage
-----
    .venv/bin/python evaluation/answer-path/measure_caller_depth.py
    .venv/bin/python evaluation/answer-path/measure_caller_depth.py --limit 8
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

from measure_backward import _chunked, callers_of  # noqa: E402
from recall_matching import hits  # noqa: E402

DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'
KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--depth', type=int, default=3, help='depth for the seed walk')
    parser.add_argument('--caller-depths', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    from config import get_config
    from docgen.scip_paths import indexer_cwds
    from library.structural_assembly import (DEFAULT_EXPAND_FAN_IN_MAX, chain_from_seeds,
                                             seeds_from_documents)

    dump = json.loads(DUMP.read_text(encoding='utf-8'))
    key = {e['id']: e for e in json.loads(KEY.read_text(encoding='utf-8'))}
    root = str(get_config().get_all_source_paths().get(args.source) or '') or None
    cwds = indexer_cwds(root) if root else ()
    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    results = dump['results'][:args.limit] if args.limit else dump['results']

    labels = ['forward'] + [f'callers@{d}' for d in args.caller_depths]
    totals = {label: {'reached': 0, 'symbols': 0, 'hops': 0} for label in labels}
    required_total = 0
    rows: list[dict] = []
    for result in results:
        wanted = [r['symbol'] for r in key.get(result['id'], {}).get('required', [])]
        documents = [{'content': d.get('content') or '',
                      'source_files': d.get('source_files') or []}
                     for d in result['widths']['8']['documents']]
        if not wanted or not documents:
            continue
        print(f'  q{result["id"]} ...', end='', flush=True)
        seeds = seeds_from_documents(conn, documents, source=args.source,
                                     indexer_cwds=cwds, source_root=root)
        forward, _ = chain_from_seeds(conn, seeds.seeds, source=args.source,
                                      depth=args.depth)
        if not forward:
            print(' no chain', flush=True)
            continue
        required_total += len(wanted)
        base_names = {c.qualified_name for c in forward}
        totals['forward']['reached'] += len(hits(wanted, base_names))
        totals['forward']['symbols'] += len(base_names)
        totals['forward']['hops'] += len(forward)
        row = {'id': result['id'], 'family': result.get('family', ''),
               'required': len(wanted),
               'forward': {'reached': len(hits(wanted, base_names)),
                           'symbols': len(base_names)}}

        added, _ = callers_of(conn, seeds.seeds, fan_in_max=DEFAULT_EXPAND_FAN_IN_MAX)
        for caller_depth in args.caller_depths:
            label = f'callers@{caller_depth}'
            if caller_depth == 0:
                # named only: the caller is offered, nothing runs from it
                extra: set = set()
                for chunk in _chunked(added):
                    placeholders = ','.join('?' * len(chunk))
                    for name, owner in conn.execute(
                            f'SELECT qualified_name, source_name FROM scip_symbols '
                            f'WHERE canonical_id IN ({placeholders})', chunk):
                        if owner == args.source:
                            extra.add(name)
                hops = 0
            else:
                up, _unused = chain_from_seeds(conn, list(added), source=args.source,
                                               depth=caller_depth)
                extra = {c.qualified_name for c in up}
                hops = len(up)
            names = base_names | extra
            totals[label]['reached'] += len(hits(wanted, names))
            totals[label]['symbols'] += len(names)
            totals[label]['hops'] += len(forward) + hops
            row[label] = {'reached': len(hits(wanted, names)), 'symbols': len(names)}
        rows.append(row)
        print(f' {row["forward"]["reached"]}/{len(wanted)} -> '
              f'{row[labels[-1]]["reached"]}/{len(wanted)}', flush=True)
    conn.close()

    n = max(len(rows), 1)
    base = totals['forward']
    print('\n' + '=' * 96)
    print(f'HOW FAR TO WALK FROM A CALLER — {len(rows)} questions'.center(96))
    print('=' * 96)
    print(f'{"caller depth":<14} {"reached":>8} {"of required":>12} {"symbols/q":>11} '
          f'{"added/q":>9} {"hops/q":>8} {"pp per +100":>12}')
    for label in labels:
        stat = totals[label]
        added_symbols = (stat['symbols'] - base['symbols']) / n
        gain = (stat['reached'] - base['reached']) / max(required_total, 1) * 100
        rate = (f'{gain / (added_symbols / 100):>11.2f}' if added_symbols > 0
                else f'{"-":>11}')
        print(f'{label:<14} {stat["reached"]:>8} '
              f'{stat["reached"] / max(required_total, 1):>11.1%} '
              f'{stat["symbols"] / n:>11.0f} {added_symbols:>9.0f} '
              f'{stat["hops"] / n:>8.0f} {rate:>12}')
    print(f'\nrequired symbols: {required_total}')
    print('The seed walk is unchanged in every row. Only the callers\' depth varies, so the\n'
          'difference between rows is the price of descending from a dispatcher.')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'written: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
