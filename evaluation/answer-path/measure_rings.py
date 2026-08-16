#!/usr/bin/env python
"""Cheap ways to reach more of the answer key, priced per symbol added. Free to run.

Measured at full scale, the two ways of spending a backward hop differ by a factor of 39:

    + callers cited   66.7% -> 71.9%   674 -> 698 symbols/q   +21.7pp per 100 symbols
    + callers rooted  71.9% -> 76.0%   698 -> 1,432           +0.56pp per 100 symbols

Walking is what costs; *offering* is nearly free. So this prices three more rings that only
offer — each names symbols the chain has already proven adjacent to, and none of them
descends, so none multiplies the chain the way rooting does:

    callers-2   callers of every symbol the caller-rooted chain reached, cited not walked.
                One more level up, bounded by the same fan-in gate.
    callees-1   callees of every reached symbol that the walk *declined* to descend into —
                the ``plumbing`` and ``depth`` terminals. Their coordinates are already
                cited; this asks what naming their callees as well would add.
    types       symbols reached through ``type_ref`` edges from anything in the chain. A
                type the code touches is part of what it does, and today those are cited
                only when a body the walk entered references them directly.

Every ring is reported with what it costs, because a ring that adds 4,000 symbols to gain a
point is not cheap however free the walk is — the menu and the prompt both grow with it.

Usage
-----
    .venv/bin/python evaluation/answer-path/measure_rings.py
    .venv/bin/python evaluation/answer-path/measure_rings.py --limit 8
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

from measure_backward import _chunked, caller_counts, callers_of  # noqa: E402
from recall_matching import hits  # noqa: E402

DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'
KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'


def names_for(conn, ids, source: str) -> set:
    """Qualified names for ``ids`` owned by ``source``. Scoped in Python, never in SQL."""
    out: set = set()
    for chunk in _chunked(ids):
        placeholders = ','.join('?' * len(chunk))
        for name, owner in conn.execute(
                f'SELECT qualified_name, source_name FROM scip_symbols '
                f'WHERE canonical_id IN ({placeholders})', chunk):
            if owner == source:
                out.add(name)
    return out


def ids_for(conn, names, source: str) -> list:
    """Canonical ids for ``names``. ``source_name`` is indexed badly; filter in Python."""
    out: list = []
    for chunk in _chunked(names):
        placeholders = ','.join('?' * len(chunk))
        out += [cid for cid, owner in conn.execute(
            f'SELECT canonical_id, source_name FROM scip_symbols '
            f'WHERE qualified_name IN ({placeholders}) '
            f"AND canonical_id NOT LIKE 'local %'", chunk) if owner == source]
    return out


def edges_from(conn, ids, *, edge_type: str, gate: int | None = None) -> set:
    """Callees of ``ids`` over ``edge_type``, optionally skipping high-fan-in callers."""
    listed = list(ids)
    if gate is not None:
        counts = caller_counts(conn, listed)
        listed = [i for i in listed if counts.get(i, 0) < gate]
    found: set = set()
    for chunk in _chunked(listed):
        placeholders = ','.join('?' * len(chunk))
        for (callee,) in conn.execute(
                f'SELECT DISTINCT callee_canonical_id FROM scip_edges '
                f'WHERE caller_canonical_id IN ({placeholders}) AND edge_type = ?',
                [*chunk, edge_type]):
            if not callee.startswith('local '):
                found.add(callee)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--width', default='8', choices=('8', '40'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    from config import get_config
    from docgen.scip_paths import indexer_cwds
    from library.structural_assembly import (DEFAULT_EXPAND_FAN_IN_MAX, chain_from_seeds,
                                             seeds_from_documents)

    gate = DEFAULT_EXPAND_FAN_IN_MAX
    dump = json.loads(DUMP.read_text(encoding='utf-8'))
    key = {e['id']: e for e in json.loads(KEY.read_text(encoding='utf-8'))}
    root = str(get_config().get_all_source_paths().get(args.source) or '') or None
    cwds = indexer_cwds(root) if root else ()
    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    results = dump['results'][:args.limit] if args.limit else dump['results']

    RINGS = ('rooted', '+ callers-2', '+ callees-1', '+ types', '+ all three')
    totals = {name: {'reached': 0, 'symbols': 0} for name in RINGS}
    required_total = 0
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
        up, _ = callers_of(conn, seeds.seeds, fan_in_max=gate)
        citations, _ = chain_from_seeds(conn, list(set(seeds.seeds) | up),
                                        source=args.source, depth=args.depth)
        if not citations:
            print(' no chain', flush=True)
            continue
        required_total += len(wanted)
        reached = {c.qualified_name for c in citations}
        chain_ids = ids_for(conn, reached, args.source)

        callers2, _ = callers_of(conn, chain_ids, fan_in_max=gate)
        ring_callers2 = names_for(conn, callers2, args.source) - reached
        # callees of the terminals the walk declined to open
        declined = ids_for(conn, {c.qualified_name for c in citations
                                  if c.stop_reason in ('plumbing', 'depth')}, args.source)
        ring_callees1 = names_for(
            conn, edges_from(conn, declined, edge_type='call'), args.source) - reached
        ring_types = names_for(
            conn, edges_from(conn, chain_ids, edge_type='type_ref', gate=gate),
            args.source) - reached

        sets = {
            'rooted': reached,
            '+ callers-2': reached | ring_callers2,
            '+ callees-1': reached | ring_callees1,
            '+ types': reached | ring_types,
            '+ all three': reached | ring_callers2 | ring_callees1 | ring_types,
        }
        row = {'id': result['id'], 'family': result.get('family', ''),
               'required': len(wanted)}
        for name in RINGS:
            totals[name]['reached'] += len(hits(wanted, sets[name]))
            totals[name]['symbols'] += len(sets[name])
            row[name] = {'reached': len(hits(wanted, sets[name])),
                         'symbols': len(sets[name])}
        rows.append(row)
        print(f' {row["rooted"]["reached"]}/{len(wanted)} -> '
              f'{row["+ all three"]["reached"]}/{len(wanted)}', flush=True)
    conn.close()

    n = max(len(rows), 1)
    base = totals['rooted']
    print('\n' + '=' * 92)
    print(f'RINGS ON TOP OF THE CALLER-ROOTED CHAIN — width {args.width}, '
          f'{len(rows)} questions'.center(92))
    print('=' * 92)
    print(f'{"ring":<14} {"reached":>8} {"of required":>12} {"symbols/q":>11} '
          f'{"added/q":>9} {"pp per +100":>12}')
    for name in RINGS:
        stat = totals[name]
        added = (stat['symbols'] - base['symbols']) / n
        gain = (stat['reached'] - base['reached']) / max(required_total, 1) * 100
        rate = f'{gain / (added / 100):>11.2f}' if added > 0 else f'{"-":>11}'
        print(f'{name:<14} {stat["reached"]:>8} '
              f'{stat["reached"] / max(required_total, 1):>11.1%} '
              f'{stat["symbols"] / n:>11.0f} {added:>9.0f} {rate:>12}')
    print(f'\nrequired symbols: {required_total}')
    print('A ring only offers; nothing descends through it, so the chain does not multiply.\n'
          'Judge by the last column — points gained per hundred symbols the menu grows.')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'written: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
