#!/usr/bin/env python
"""Generate the pooled clew set and store it, so the answer path can be measured with it.

Three anchoring strategies, pooled, because pooling is what the measurement says matters — on
the databricks pack, per-question required-slot coverage:

    theme-seeded walks      73.2%   walk outward from each theme's members
    theme-anchored routes   78.4%   routes among theme members, hubs allowed
    entry-point routes      66.0%   routes from symbols with no inbound call edge
    ALL THREE POOLED        92.8%   against 66.0% for one live document-seeded walk

No single strategy exceeds 78.4%; together they reach 92.8%, because different anchors find
different routes. Dedup happens in the store (:func:`library.clews.add_clew` keys on the
route), so overlap between strategies costs nothing.

Writing is additive — a new table, no existing row touched — but it writes to a real store, so
``--yes`` is required. Embedding is a separate step and needs a provider key; a clew with no
vector is stored and skipped by ``nearest_clews`` rather than ranked as distant.

Usage
-----
    .venv/bin/python evaluation/answer-path/write_clews.py --dry-run
    .venv/bin/python evaluation/answer-path/write_clews.py --yes
    .venv/bin/python evaluation/answer-path/write_clews.py --yes --db /tmp/copy.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from spool_clews import collapse, load, routes_from  # noqa: E402


def theme_members(conn, name: dict, source: str) -> dict:
    """``cluster_id -> member symbol ids``, mapped through the deterministic catalog doc id."""
    from docgen.catalog_writer import _element_doc_id
    doc_of = {_element_doc_id(source, qn): cid for cid, qn in name.items()}
    clusters: dict = defaultdict(set)
    for cluster_id, element_id in conn.execute(
            'SELECT cluster_id, element_id FROM theme_members'):
        cid = doc_of.get(element_id)
        if cid is not None:
            clusters[cluster_id].add(cid)
    return clusters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--db', default=None, help='store to write; default ./ariadne.db')
    parser.add_argument('--seeds-per-theme', type=int, default=200)
    parser.add_argument('--per-anchor', type=int, default=1)
    parser.add_argument('--min-steps', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true',
                        help='generate and count, write nothing')
    parser.add_argument('--yes', action='store_true', help='required to write')
    args = parser.parse_args()

    if not args.dry_run and not args.yes:
        print('refusing to write without --yes (or use --dry-run)')
        return 2

    from library.clews import add_clew, init_clews_schema
    from library.structural_assembly import chain_from_seeds

    path = args.db or str(ROOT / 'ariadne.db')
    conn = sqlite3.connect(path)
    started = time.time()
    name, by_name, out, fan_in, dropped = load(conn, args.source)
    clusters = theme_members(conn, name, args.source)
    print(f'{args.source}: {len(name):,} symbols, {len(clusters)} themes, '
          f'{dropped:,} edges dropped as outside the caller body '
          f'in {time.time() - started:.1f}s')

    #: route (tuple of ids) -> strategy that found it first
    routes: dict = {}

    def keep(route, strategy: str) -> None:
        if len(collapse(route, name)) >= args.min_steps:
            routes.setdefault(tuple(route), strategy)

    started = time.time()
    for members in clusters.values():
        citations, _ = chain_from_seeds(
            conn, sorted(members)[:args.seeds_per_theme], source=args.source, depth=3)
        # A walk yields hops, not routes; the parent link turns it back into paths.
        parent = {c.qualified_name: c.parent_qualified_name for c in citations}
        ids = {qn: by_name.get(qn) for qn in parent}
        for reached in parent:
            chain, node = [], reached
            while node and node not in chain:
                chain.append(node)
                node = parent.get(node)
            resolved = [ids[qn] for qn in reversed(chain) if ids.get(qn)]
            if len(resolved) >= 2:
                keep(resolved, 'theme-walk')
    print(f'  theme-seeded walks   {len(routes):>7,} routes '
          f'in {time.time() - started:.0f}s', flush=True)

    started = time.time()
    for label, anchors in (
            ('theme-anchored', sorted({m for v in clusters.values() for m in v},
                                      key=lambda m: (fan_in[m], name[m]))),
            ('entry-point', sorted((n for n in out if fan_in[n] == 0),
                                   key=lambda n: (-len(out[n]), name[n])))):
        before = len(routes)
        for anchor in anchors:
            for route in routes_from(out, fan_in, anchor, per_anchor=args.per_anchor,
                                     min_hops=2, max_hops=8, fan_in_max=10**9):
                keep(route, label)
        print(f'  {label:<20} {len(routes) - before:>7,} new routes '
              f'({len(routes):,} total)', flush=True)
    print(f'  route generation in {time.time() - started:.0f}s')

    if args.dry_run:
        print(f'\ndry run: {len(routes):,} routes would be stored')
        conn.close()
        return 0

    init_clews_schema(conn)
    started = time.time()
    written = 0
    for route, strategy in routes.items():
        qualified = [name[n] for n in route]
        add_clew(
            conn, source_name=args.source, entry_symbol=qualified[0],
            steps=collapse(route, name), route=qualified,
            files=sorted({f for f in (
                row[0] for row in conn.execute(
                    'SELECT file FROM scip_symbols WHERE canonical_id IN '
                    f'({",".join("?" * len(route))})', list(route))) if f}),
            strategy=strategy)
        written += 1
        if written % 5000 == 0:
            conn.commit()
            print(f'    stored {written:,}', flush=True)
    conn.commit()
    stored = conn.execute('SELECT COUNT(*) FROM clews WHERE source_name = ?',
                          (args.source,)).fetchone()[0]
    conn.close()
    print(f'\nstored {written:,} routes in {time.time() - started:.0f}s; '
          f'{stored:,} clews now in {path}')
    print('next: embed them (needs a provider key), then run the arm')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
