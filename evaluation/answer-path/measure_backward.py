#!/usr/bin/env python
"""Would one hop **up** from the seeds reach what the forward walk cannot? Free to run.

``measure_headroom`` established that depth is not the lever: 3 -> 4 reached 77.1% both
times. Classifying the symbols the chain missed says why — of seven, **six are callers** of
symbols the chain did reach and one is disconnected. None is downstream. A forward-only walk
cannot reach a caller at any depth, so no amount of depth was ever going to close the gap.

That is a walk-direction gap, and it has a shape: retrieval lands mid-funnel (the file that
defines the thing asked about), the walk descends, and the dispatchers that *invoke* it —
``PreprocessTableUpdate``, ``UpdateTable``, ``ClassicMergeExecutor``, ``DataSource`` — are
above the entry point, never visited. A question like "how does an update work end to end"
requires exactly those: the top of the funnel is part of the mechanism.

Two ways to spend a backward hop, measured side by side because they cost very differently:

    cited    callers of the seeds are added to the menu and **not** walked. Bounded by the
             number of seeds; a caller is offered as evidence, nothing descends through it.
    roots    callers of the seeds become additional roots and the forward walk runs again
             from there. Reaches their subtrees too, and pays for them.

Both gate on the seed's own caller count (``DEFAULT_EXPAND_FAN_IN_MAX``, the number already
measured for descent): walking up from ``LogicalPlan.<init>`` would add 375 roots and say
nothing about the question. The gate is symmetric because the reasoning is — fan-in separates
a step in a mechanism from framework plumbing in either direction.

Reported against answer keys fixed in advance (``chain_requirements.json``) with the shared
boundary matcher, so these figures are comparable with every other number in this directory.

Usage
-----
    .venv/bin/python evaluation/answer-path/measure_backward.py
    .venv/bin/python evaluation/answer-path/measure_backward.py --limit 10 --json back.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from recall_matching import hits  # noqa: E402

DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'
KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'


def _chunked(items, size: int = 300):
    listed = sorted(items)
    for start in range(0, len(listed), size):
        yield listed[start:start + size]


def caller_counts(conn, ids) -> dict:
    """``id -> number of call edges into it``. The same measure the descent gate uses."""
    counts: dict = {}
    for chunk in _chunked(ids):
        placeholders = ','.join('?' * len(chunk))
        for callee, total in conn.execute(
                f'SELECT callee_canonical_id, COUNT(*) FROM scip_edges '
                f'WHERE callee_canonical_id IN ({placeholders}) '
                f"AND edge_type = 'call' GROUP BY callee_canonical_id", chunk):
            counts[callee] = total
    return counts


def callers_of(conn, ids, *, fan_in_max: int) -> tuple[set, int]:
    """Callers of every id under the gate, plus how many ids the gate held back.

    ``source_name`` is deliberately absent from the query. It has three distinct values
    across 727,967 rows, so naming it makes SQLite prefer the low-selectivity index and scan
    — the mistake ``_locate``'s docstring records, and one this script made once already
    (two ten-minute timeouts). Ownership is enforced by ``_locate`` when the walk resolves
    these ids, exactly as it is for a forward hop.
    """
    counts = caller_counts(conn, ids)
    eligible = [i for i in ids if counts.get(i, 0) < fan_in_max]
    found: set = set()
    for chunk in _chunked(eligible):
        placeholders = ','.join('?' * len(chunk))
        for (caller,) in conn.execute(
                f'SELECT DISTINCT caller_canonical_id FROM scip_edges '
                f"WHERE callee_canonical_id IN ({placeholders}) AND edge_type = 'call'",
                chunk):
            if not caller.startswith('local '):
                found.add(caller)
    return found, len(ids) - len(eligible)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width', default='8', choices=('8', '40'),
                        help='retrieval width to replay from the dump')
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

    rows: list[dict] = []
    for result in results:
        wanted = [r['symbol'] for r in key.get(result['id'], {}).get('required', [])]
        documents = [{'content': d.get('content') or '',
                      'source_files': d.get('source_files') or []}
                     for d in result['widths'][args.width]['documents']]
        if not wanted or not documents:
            continue
        print(f'  q{result["id"]} ...', end='', flush=True)
        seeds = seeds_from_documents(conn, documents, source=args.source,
                                     indexer_cwds=cwds, source_root=root)
        started = time.time()
        forward, _ = chain_from_seeds(conn, seeds.seeds, source=args.source,
                                      depth=args.depth)
        forward_secs = time.time() - started
        if not forward:
            print(' no chain', flush=True)
            continue
        forward_names = {c.qualified_name for c in forward}

        added, gated = callers_of(conn, seeds.seeds,
                                  fan_in_max=DEFAULT_EXPAND_FAN_IN_MAX)
        # cited: the callers are offered, nothing is walked through them
        cited_names = set(forward_names)
        for chunk in _chunked(added):
            placeholders = ','.join('?' * len(chunk))
            for name, owner in conn.execute(
                    f'SELECT qualified_name, source_name FROM scip_symbols '
                    f'WHERE canonical_id IN ({placeholders})', chunk):
                if owner == args.source:
                    cited_names.add(name)
        # roots: the callers seed a second forward walk
        started = time.time()
        rooted, _ = chain_from_seeds(conn, list(set(seeds.seeds) | added),
                                     source=args.source, depth=args.depth)
        rooted_secs = time.time() - started
        rooted_names = {c.qualified_name for c in rooted}

        reached = hits(wanted, forward_names)
        rows.append({
            'id': result['id'], 'family': result.get('family', ''),
            'required': len(wanted), 'seeds': len(seeds.seeds),
            'added': len(added), 'gated': gated,
            'forward': len(reached),
            'cited': len(hits(wanted, cited_names)),
            'roots': len(hits(wanted, rooted_names)),
            'symbols_forward': len(forward_names),
            'symbols_cited': len(cited_names),
            'symbols_roots': len(rooted_names),
            'hops_forward': len(forward), 'hops_roots': len(rooted),
            'secs_forward': forward_secs, 'secs_roots': rooted_secs,
            'won_by_cited': sorted(hits(wanted, cited_names) - reached),
            'won_by_roots': sorted(hits(wanted, rooted_names) - hits(wanted, cited_names)),
            'still_missing': sorted(set(wanted) - hits(wanted, rooted_names)),
        })
        print(f' fwd {len(reached)}/{len(wanted)} -> cited {rows[-1]["cited"]} '
              f'-> roots {rows[-1]["roots"]}', flush=True)
    conn.close()

    print('\n' + '=' * 104)
    print(f'ONE HOP UP FROM THE SEEDS — source {args.source}, depth {args.depth}'
          .center(104))
    print('=' * 104)
    print(f'{"q":>4} {"family":<15} {"req":>4} {"seeds":>6} {"+up":>5} {"gate":>5} '
          f'{"fwd":>4} {"cite":>5} {"root":>5} {"symFwd":>7} {"symCit":>7} {"symRoot":>8}'
          f'  won')
    print('-' * 104)
    for row in rows:
        won = ','.join(row['won_by_cited'] + row['won_by_roots'])
        print(f'{row["id"]:>4} {row["family"][:15]:<15} {row["required"]:>4} '
              f'{row["seeds"]:>6} {row["added"]:>5} {row["gated"]:>5} '
              f'{row["forward"]:>4} {row["cited"]:>5} {row["roots"]:>5} '
              f'{row["symbols_forward"]:>7,} {row["symbols_cited"]:>7,} '
              f'{row["symbols_roots"]:>8,}  {won[:22]}')
    print('-' * 104)
    n = max(len(rows), 1)
    required = sum(r['required'] for r in rows)
    print(f'questions: {len(rows)}   required symbols: {required}\n')
    print(f'{"walk":<10} {"reached":>8} {"of required":>12} {"symbols/q":>11} '
          f'{"hops/q":>9} {"secs/q":>8}')
    fwd = sum(r['forward'] for r in rows)
    cit = sum(r['cited'] for r in rows)
    rts = sum(r['roots'] for r in rows)
    print(f'{"forward":<10} {fwd:>8} {fwd / max(required, 1):>11.1%} '
          f'{sum(r["symbols_forward"] for r in rows) / n:>11.0f} '
          f'{sum(r["hops_forward"] for r in rows) / n:>9.0f} '
          f'{sum(r["secs_forward"] for r in rows) / n:>8.1f}')
    print(f'{"+ cited":<10} {cit:>8} {cit / max(required, 1):>11.1%} '
          f'{sum(r["symbols_cited"] for r in rows) / n:>11.0f} '
          f'{sum(r["hops_forward"] for r in rows) / n:>9.0f} '
          f'{sum(r["secs_forward"] for r in rows) / n:>8.1f}')
    print(f'{"+ roots":<10} {rts:>8} {rts / max(required, 1):>11.1%} '
          f'{sum(r["symbols_roots"] for r in rows) / n:>11.0f} '
          f'{sum(r["hops_roots"] for r in rows) / n:>9.0f} '
          f'{sum(r["secs_roots"] for r in rows) / n:>8.1f}')
    print(f'\ncallers added per question: {sum(r["added"] for r in rows) / n:.0f} '
          f'(seeds held back by the fan-in gate: {sum(r["gated"] for r in rows) / n:.0f} '
          f'of {sum(r["seeds"] for r in rows) / n:.0f})')
    missing = sorted({s for r in rows for s in r['still_missing']})
    print(f'still missing after a backward hop: {len(missing)} '
          f'{", ".join(missing[:8])}')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'\nwritten: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
