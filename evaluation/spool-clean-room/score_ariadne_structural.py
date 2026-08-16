#!/usr/bin/env python3
"""Does Ariadne's SCIP graph CONTAIN each question's required chain?

Ariadne is not a search agent, and scoring it as one imposes the bare arm's
modality on a tool that does not work that way. It extracts a call graph with a
compiler -- the README's central claim -- so for any chain question the answer
is already determined before a model is involved: the symbols are in the index
or they are not, and the edges connect them or they do not. Re-asking cannot
change it, which is also why the bare arm's retry guard is meaningless here.

So this measures the thing Ariadne actually is, deterministically and for free:

  resolution  -- each required symbol resolves to a SCIP symbol in the corpus
  adjacency   -- consecutive required symbols are joined by a call edge
  reachability-- or joined by a path of at most --max-hops edges

Comparable to the bare arm because the DENOMINATOR is identical: the same fixed
``chain_requirements.json``, decided before either arm ran. What differs is the
evidence mechanism -- quotes the bare arm had to find by reading, versus edges
Ariadne either holds or lacks.

What it does NOT measure is correctness: containing the chain is necessary for
answering, not sufficient. Read it as the completeness pillar only.

No API key, no LLM, no network.

    python evaluation/spool-clean-room/score_ariadne_structural.py --source databricks
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REQS = HERE / 'chain_requirements.json'


def resolve(conn, source: str, symbol: str) -> list[str]:
    """Canonical ids a required symbol maps to (display_name, then qualified)."""
    rows = conn.execute(
        'SELECT canonical_id FROM scip_symbols '
        'WHERE source_name = ? AND display_name = ?', (source, symbol)).fetchall()
    if not rows:
        rows = conn.execute(
            'SELECT canonical_id FROM scip_symbols '
            'WHERE source_name = ? AND qualified_name LIKE ?',
            (source, f'%.{symbol}')).fetchall()
    return [r[0] for r in rows]


def reachable(conn, starts: list[str], goals: set[str], max_hops: int) -> int | None:
    """Fewest call-edge hops from any start to any goal, or None.

    Undirected on purpose: "A relates to B" in a mechanism chain does not commit
    to which side calls which, and the answer key names mechanisms rather than a
    direction. Counting only forward edges would understate what the graph can
    actually connect.
    """
    seen, frontier = set(starts), deque((s, 0) for s in starts)
    while frontier:
        node, d = frontier.popleft()
        if node in goals and d > 0:
            return d
        if d >= max_hops:
            continue
        rows = conn.execute(
            'SELECT callee_canonical_id FROM scip_edges WHERE caller_canonical_id = ? '
            'UNION SELECT caller_canonical_id FROM scip_edges WHERE callee_canonical_id = ?',
            (node, node)).fetchall()
        for (nxt,) in rows:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append((nxt, d + 1))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='databricks')
    ap.add_argument('--max-hops', type=int, default=3,
                    help='path length allowed between consecutive hops (default 3)')
    ap.add_argument('--db', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))
    from config import get_config
    from library import Library

    reqs = [r for r in json.loads(REQS.read_text()) if not r['flag']]
    lib = Library(Path(args.db) if args.db else get_config().db_path)
    rows = []
    try:
        with lib._conn_provider.acquire() as conn:
            for q in reqs:
                syms = [r['symbol'] for r in q['required']]
                ids = {s: resolve(conn, args.source, s) for s in syms}
                unresolved = [s for s, v in ids.items() if not v]
                links, gaps = [], []
                for a, b in zip(syms, syms[1:]):
                    if not ids[a] or not ids[b]:
                        gaps.append(f'{a}->{b} (unresolved)')
                        continue
                    d = reachable(conn, ids[a], set(ids[b]), args.max_hops)
                    (links if d is not None else gaps).append(
                        f'{a}->{b}' + (f' ({d})' if d else ''))
                pairs = max(len(syms) - 1, 0)
                rows.append({
                    'id': q['id'], 'family': q.get('family'), 'hops': len(syms),
                    'resolved': len(syms) - len(unresolved), 'unresolved': unresolved,
                    'linked': len(links), 'pairs': pairs, 'gaps': gaps,
                    'resolution': (len(syms) - len(unresolved)) / len(syms) if syms else None,
                    'connectivity': (len(links) / pairs) if pairs else None,
                })
    finally:
        lib.close()

    print(f'{"id":>5} {"hops":>5} {"resolved":>9} {"linked":>12}  unresolved')
    for r in sorted(rows, key=lambda r: -(r['connectivity'] or 0)):
        print(f'{r["id"]:>5} {r["hops"]:>5} '
              f'{r["resolved"]}/{r["hops"]:<7} '
              f'{r["linked"]}/{r["pairs"]:<10} '
              f'{",".join(r["unresolved"][:3])}')
    res = [r['resolution'] for r in rows if r['resolution'] is not None]
    con = [r['connectivity'] for r in rows if r['connectivity'] is not None]
    print(f'\n  symbol resolution : mean {st.mean(res):.0%}  '
          f'fully resolved {sum(1 for x in res if x == 1.0)}/{len(res)}')
    print(f'  chain connectivity: mean {st.mean(con):.0%}  '
          f'fully connected {sum(1 for x in con if x == 1.0)}/{len(con)}  '
          f'(<= {args.max_hops} call-edge hops between consecutive mechanisms)')
    out = Path(args.out) if args.out else HERE / 'ariadne-structural.json'
    # Record the bound with the data: a hop count carried separately
    # from the numbers it produced is a mislabel waiting to happen.
    payload = {'max_hops': args.max_hops, 'source': args.source,
               'rows': rows}
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(f'  detail -> {out.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
