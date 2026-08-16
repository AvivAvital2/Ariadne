#!/usr/bin/env python
"""How many clews can a spool ship? Retroactive extraction from an already-indexed pack.

A spool is a versioned pack of prebuilt docs and embeddings for a runtime. If clews travel
with the pack, every project that enables the spool gets path-level retrieval without paying
to generate anything — which is the point of a pack.

The request archetype does not transfer. A spool has no ``api_endpoints``, no ``data_access``
and no ``schema_symbols`` (all zero for databricks), so "endpoint reaches table" has nothing
to anchor on. What a spool *does* have is different and richer:

    version_facts   3,195 rows carrying a ``qualified_name`` — a symbol the pack has already
                    judged user-facing enough to record a fact about (deprecated at 3.5.0,
                    added in, and so on). These join to SCIP by name, no path resolution.
    surface_tags    1,806 docs tagged with an interaction vocabulary (``parallelism``, ...)
    entry points    symbols with no inbound call edge — 74,864 of them
    terminals       symbols with no outbound call edge, where work lands

So a spool clew is a route from something the pack considers an interface to somewhere work
lands. Deduplicated by route, since many interfaces share tails.

Every lookup runs against Python dicts. ``source_name`` in a WHERE or JOIN clause makes
SQLite prefer that low-selectivity index over the primary key and scan — three queries in
this session died to it, two of them after ten minutes.

Usage
-----
    .venv/bin/python evaluation/answer-path/spool_clews.py
    .venv/bin/python evaluation/answer-path/spool_clews.py --per-anchor 4 --show 12
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


def load(conn, source: str):
    """Names, extents, forward adjacency and fan-in — four dicts, two passes, no joins."""
    name: dict = {}
    extent: dict = {}
    by_name: dict = {}
    for cid, qn, owner, start, end in conn.execute(
            'SELECT canonical_id, qualified_name, source_name, line_start, line_end '
            'FROM scip_symbols'):
        if owner == source and not cid.startswith('local '):
            name[cid] = qn
            extent[cid] = (start or 0, end or 0)
            by_name.setdefault(qn, cid)
    out: dict = defaultdict(list)
    fan_in: Counter = Counter()
    dropped = 0
    for caller, callee, line in conn.execute(
            "SELECT caller_canonical_id, callee_canonical_id, line FROM scip_edges "
            "WHERE edge_type = 'call'"):
        if caller not in name or callee not in name or caller == callee:
            continue
        start, end = extent[caller]
        if end > start and not (start <= (line or 0) <= end):
            dropped += 1
            continue
        out[caller].append(callee)
        fan_in[callee] += 1
    return name, by_name, out, fan_in, dropped


def routes_from(out, fan_in, start, *, per_anchor: int, min_hops: int, max_hops: int,
                fan_in_max: int):
    """Up to ``per_anchor`` routes from ``start`` to a terminal, depth-first."""
    found: list = []
    stack = [(start, (start,))]
    while stack and len(found) < per_anchor:
        node, route = stack.pop()
        onward = [c for c in out.get(node, ())
                  if c not in route and fan_in[c] < fan_in_max]
        if not onward or len(route) > max_hops:
            if len(route) - 1 >= min_hops:
                found.append(route)
            continue
        for callee in onward:
            stack.append((callee, route + (callee,)))
    return found
_UNINFORMATIVE = frozenset({'<init>', '<clinit>'})


def informative(step: str) -> bool:
    """Whether a step is worth saying. ``apply`` stays -- in Scala it is the mechanism."""
    return step not in _UNINFORMATIVE and any(c.isalnum() for c in step)


def collapse(route, name) -> list:
    """The route as distinct steps: consecutive hops sharing a display name become one.

    A Java wrapper delegating to its Scala implementation, an overload forwarding to the real
    body, a bridge method -- each renders the same name twice and none of them says anything
    about the mechanism. Measured on the databricks pack, collapsing these runs and then
    requiring three distinct steps drops 17,660 of 39,072 raw routes (45%) as not routes.

    The graph is untouched. This is what the clew asserts, and a clew that asserts
    ``popStdev -> popStdev`` asserts nothing.
    """
    steps: list = []
    for cid in route:
        leaf = name[cid].split('.')[-1]
        if steps and steps[-1] == leaf:
            continue
        if not informative(leaf):
            continue
        steps.append(leaf)
    return steps
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--db', default=None)
    parser.add_argument('--per-anchor', type=int, default=3)
    parser.add_argument('--min-hops', type=int, default=2)
    parser.add_argument('--max-hops', type=int, default=8)
    parser.add_argument('--min-steps', type=int, default=3,
                        help='distinct steps a route needs to explain anything')
    parser.add_argument('--fan-in-max', type=int, default=16)
    parser.add_argument('--show', type=int, default=8)
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(f'file:{args.db or (ROOT / "ariadne.db")}?mode=ro', uri=True)
    started = time.time()
    name, by_name, out, fan_in, dropped = load(conn, args.source)
    facts: dict = defaultdict(list)
    for qn, fact, version, component in conn.execute(
            'SELECT qualified_name, fact, version, component FROM version_facts'):
        if qn in by_name:
            facts[by_name[qn]].append({'fact': fact, 'version': version,
                                       'component': component})
    conn.close()
    print(f'{args.source}: {len(name):,} named symbols, '
          f'{sum(len(v) for v in out.values()):,} call edges '
          f'({dropped:,} dropped as outside the caller body) '
          f'in {time.time() - started:.1f}s')

    interfaces = sorted(facts)
    entries = sorted((n for n in out if fan_in[n] == 0),
                     key=lambda n: (-len(out[n]), name[n]))
    print(f'anchors: {len(interfaces):,} symbols carrying a version fact, '
          f'{len(entries):,} entry points')

    started = time.time()
    clews: dict = {}
    raw = 0
    for label, anchors in (('version-fact', interfaces), ('entry-point', entries)):
        made = 0
        for anchor in anchors:
            for route in routes_from(out, fan_in, anchor, per_anchor=args.per_anchor,
                                     min_hops=args.min_hops, max_hops=args.max_hops,
                                     fan_in_max=args.fan_in_max):
                raw += 1
                steps = collapse(route, name)
                key = tuple(steps)
                if len(steps) < args.min_steps or key in clews:
                    continue
                clews[key] = {
                    'anchor_kind': label,
                    'entry': name[route[0]],
                    'steps': steps,
                    'route': [name[n] for n in route],
                    'hops': len(route) - 1,
                    'facts': facts.get(route[0], []),
                }
                made += 1
        print(f'  {label:<14} {made:>8,} clews')
    print(f'\nraw routes {raw:,} -> {len(clews):,} clews '
          f'({1 - len(clews) / max(raw, 1):.0%} dropped as not a route) '
          f'in {time.time() - started:.1f}s')
    steps_hist = Counter(len(c['steps']) for c in clews.values())
    print(f'{"steps":>6} {"clews":>9}')
    for k in sorted(steps_hist):
        print(f'{k:>6} {steps_hist[k]:>9,}')
    covered = {n for c in clews.values() for n in c['route']}
    print(f'\nsymbols covered {len(covered):,} ({len(covered)/max(len(name),1):.1%})')

    print(f'\nsamples anchored at a recorded interface:')
    shown = 0
    for clew in clews.values():
        if clew['anchor_kind'] != 'version-fact' or len(clew['steps']) < 4:
            continue
        fact = clew['facts'][0] if clew['facts'] else {}
        print(f'  [{fact.get("fact", "")} {fact.get("version", "")}] '
              f'{" -> ".join(clew["steps"])[:92]}')
        shown += 1
        if shown >= args.show:
            break
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            [c for c in list(clews.values())[:500]], indent=2), encoding='utf-8')
        print(f'\nwritten: {args.json_out} (first 500)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
