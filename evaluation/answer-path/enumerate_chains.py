#!/usr/bin/env python
"""Can the call graph's paths be enumerated and deduplicated into an embeddable set?

Retrieval today can match a **symbol** — every symbol has an embedded catalog document —
but never a **path**, because ``scip_edges`` has no embedding column: 2,775,837 edges with
no semantic surface. So a question selects where the walk starts and nothing more, and the
component that builds the path has never seen the question.

Embedding paths would remove that, and the objection is combinatorial. This tests the
counter-argument: a short path is a contiguous window inside a longer one, so storing only
**maximal** paths — entry point to terminal — covers every shorter path for free. Storing
``A->B->C->D->E`` gives you ``A->B->C``, ``B->C->D`` and ``C->D->E`` at no extra cost.

What that argument does not bound is how many maximal paths exist, because branching
multiplies them. So this enumerates real ones and reports:

    maximal paths found        how many distinct routes entry -> terminal
    windows they contain       distinct shorter sub-paths covered for free
    dedup ratio               windows per stored path -- the compression the idea rests on
    symbol coverage           how much of the graph the stored set touches

An entry point is a symbol with outbound call edges and no inbound ones; a terminal has no
outbound call edges. Both come from the index, not from a heuristic about names. The fan-in
gate that governs the live walk is applied here too, so a hub with hundreds of callers does
not generate hundreds of near-identical routes.

Usage
-----
    .venv/bin/python evaluation/answer-path/enumerate_chains.py
    .venv/bin/python evaluation/answer-path/enumerate_chains.py --target 1000 --min-hops 4
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def load_call_graph(conn, source: str):
    """``id -> [callee ids]`` plus name and fan-in maps, in one pass each.

    ``source_name`` never appears in the edge query: it holds three distinct values across
    727,967 symbol rows, so naming it makes SQLite prefer that index and scan. Ownership is
    applied in Python against the name map, which is built from the source's rows only.
    """
    name: dict = {}
    for cid, qn, owner in conn.execute(
            'SELECT canonical_id, qualified_name, source_name FROM scip_symbols'):
        if owner == source and not cid.startswith('local '):
            name[cid] = qn
    out: dict = defaultdict(list)
    fan_in: Counter = Counter()
    for caller, callee in conn.execute(
            "SELECT caller_canonical_id, callee_canonical_id FROM scip_edges "
            "WHERE edge_type = 'call'"):
        if caller in name and callee in name and caller != callee:
            out[caller].append(callee)
            fan_in[callee] += 1
    return name, out, fan_in
def enumerate_paths(out, fan_in, entries, *, target: int, min_hops: int,
                    max_hops: int, fan_in_max: int, per_entry: int = 0):
    """Maximal routes from ``entries``, depth-first, bounded by ``target``.

    ``per_entry`` caps how many routes one entry point may contribute, and it is the
    difference between a feasible set and an explosion: uncapped, four entry points produced
    1,000 maximal paths through 463 symbols, because every branch point spawns another route
    through the same nodes. Capping trades enumerating one region exhaustively for covering
    many regions once, which is what an embeddable set needs.
    """
    paths: list = []
    for entry in entries:
        if len(paths) >= target:
            break
        found = 0
        stack = [(entry, (entry,))]
        while stack and len(paths) < target:
            if per_entry and found >= per_entry:
                break
            node, route = stack.pop()
            onward = [c for c in out.get(node, ())
                      if c not in route and fan_in[c] < fan_in_max]
            if not onward or len(route) > max_hops:
                if len(route) - 1 >= min_hops:
                    paths.append(route)
                    found += 1
                continue
            for callee in onward:
                stack.append((callee, route + (callee,)))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--min-hops', type=int, default=4)
    parser.add_argument('--max-hops', type=int, default=12)
    parser.add_argument('--fan-in-max', type=int, default=16)
    parser.add_argument('--per-entry', type=int, default=0,
                        help='cap routes contributed by one entry point (0 = uncapped)')
    parser.add_argument('--window', type=int, default=3,
                        help='length in hops of the shorter paths covered for free')
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    started = time.time()
    name, out, fan_in = load_call_graph(conn, args.source)
    conn.close()
    edges = sum(len(v) for v in out.values())
    print(f'call graph: {len(name):,} symbols, {edges:,} internal call edges '
          f'in {time.time() - started:.1f}s')

    entries = sorted(
        (n for n in out if fan_in[n] == 0),
        key=lambda n: (-len(out[n]), name[n]))
    print(f'entry points (outbound calls, no inbound): {len(entries):,}')

    started = time.time()
    paths = enumerate_paths(out, fan_in, entries, target=args.target,
                            min_hops=args.min_hops, max_hops=args.max_hops,
                            fan_in_max=args.fan_in_max,
                            per_entry=args.per_entry)
    print(f'enumerated {len(paths):,} maximal paths of >= {args.min_hops} hops '
          f'in {time.time() - started:.1f}s')
    if not paths:
        print('none found — the entry/terminal definition admits nothing')
        return 1

    lengths = Counter(len(p) - 1 for p in paths)
    windows: set = set()
    for route in paths:
        span = args.window + 1
        for start in range(0, max(len(route) - span + 1, 0)):
            windows.add(route[start:start + span])
    covered = {n for route in paths for n in route}

    print(f'\n{"hops":>6} {"paths":>8}')
    for hops in sorted(lengths):
        print(f'{hops:>6} {lengths[hops]:>8,}')
    print(f'\nstored paths                  {len(paths):>9,}')
    print(f'distinct {args.window}-hop windows inside them  {len(windows):>9,}')
    print(f'dedup ratio                   {len(windows) / len(paths):>9.1f}x  '
          f'(shorter paths covered per stored path)')
    print(f'symbols touched               {len(covered):>9,} '
          f'({len(covered) / max(len(name), 1):.1%} of the source)')
    print(f'entry points consumed         '
          f'{len({p[0] for p in paths}):>9,} of {len(entries):,}')
    print(f'\nA stored path is one embeddable unit. Every window inside it is a shorter '
          f'path a\nquestion could match, retrieved without being stored separately.')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {'paths': [[name[n] for n in p] for p in paths[:200]],
             'stored': len(paths), 'windows': len(windows),
             'symbols_touched': len(covered)}, indent=2), encoding='utf-8')
        print(f'written: {args.json_out} (first 200 paths, as names)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
