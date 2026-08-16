#!/usr/bin/env python
"""Clews for a request-shaped corpus: HTTP entry -> handler route -> data touched.

A clew is a pre-generated path through the call graph, anchored at both ends by something a
question is actually about. This module implements the **request archetype**, which applies to
a corpus that serves HTTP:

    an endpoint has a handler; that handler reaches a table
    a client call reaches an endpoint; that endpoint reaches a table

Both ends are already recorded by the HTTP and data-model tiers. ``api_endpoints`` maps a
method and path template to a ``producer_symbol_id``; ``data_access`` maps a
``consumer_symbol_id`` to a ``schema_symbol_id`` with a read/write/filter/order role;
``api_calls`` maps a client-side ``consumer_symbol_id`` to an endpoint. What is missing between
them is the SCIP route, which is free to compute.

The archetype is domain-specific, and that is the point: a data engine has no endpoints, so a
spool anchors elsewhere — see :mod:`spool_clews`, which uses ``version_facts`` and entry points
to yield 39,072 clews for one pack. Same object, different anchors.

``--source`` is required rather than defaulted, because which corpora have these tiers
populated changes; the run reports plainly when the anchors are empty.

Two defects worth knowing before trusting the output, both observed on real data:

* ``api_calls`` resolves **method-blind** where several methods share a path template, so a
  ``GET`` client call can be attributed to the ``DELETE`` endpoint on the same path. That is
  wrong in the stored tier, not here.
* a route is only as good as the caller extents. Where ``line_end == line_start`` the decorator
  filter in :func:`load_graph` cannot fire, and a route can escape through a decorator edge
  into application startup.

Usage
-----
    .venv/bin/python evaluation/answer-path/generate_clews.py --source NAME
    .venv/bin/python evaluation/answer-path/generate_clews.py --source NAME --db store.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

MAX_HOPS = 10


def load_graph(conn, source: str):
    """Forward adjacency and the name map, scoped in Python not in the WHERE clause.

    An edge is kept only when its call site lies inside the caller's definition extent. That is
    what separates a call the body makes from a **decorator**: a handler's edge to the
    application object is emitted at the ``@app.get(...)`` line, and following it walks out of
    request handling into application startup, yielding the claim that a health check writes
    audit rows.

    Where the extent is unknown the edge is kept rather than guessed at. A store whose symbols
    carry identifier spans instead of body extents (``line_end == line_start``, which
    ``docgen.scip_wiring._definition_extents_present`` exists to catch) cannot support the test,
    and silently dropping every edge there would be worse than not filtering.

    ``source_name`` never appears in a WHERE or JOIN clause: it holds few distinct values over
    hundreds of thousands of rows, so naming it makes SQLite prefer that index and scan.
    """
    name: dict = {}
    extent: dict = {}
    for cid, qn, owner, start, end in conn.execute(
            'SELECT canonical_id, qualified_name, source_name, line_start, line_end '
            'FROM scip_symbols'):
        if owner == source and not cid.startswith('local '):
            name[cid] = qn
            extent[cid] = (start or 0, end or 0)
    out: dict = defaultdict(list)
    dropped = 0
    for caller, callee, line in conn.execute(
            "SELECT caller_canonical_id, callee_canonical_id, line FROM scip_edges "
            "WHERE edge_type = 'call'"):
        if caller not in name or callee not in name or caller == callee:
            continue
        start, end = extent.get(caller, (0, 0))
        if end > start and not (start <= (line or 0) <= end):
            dropped += 1
            continue
        out[caller].append(callee)
    return name, out, dropped


def anchors(conn, source: str):
    """The two ends a clew joins: where a request enters, and where data is touched."""
    endpoints = [
        {'symbol': row[0], 'method': row[1] or '', 'path': row[2] or '',
         'endpoint_id': row[3]}
        for row in conn.execute(
            'SELECT producer_symbol_id, http_method, path_template, endpoint_id '
            'FROM api_endpoints WHERE source_name = ? AND producer_symbol_id IS NOT NULL',
            (source,))]
    sinks: dict = defaultdict(list)
    for consumer, schema_symbol, role, table, column in conn.execute(
            'SELECT da.consumer_symbol_id, da.schema_symbol_id, da.role, '
            '       s.table_name, s.column_name '
            'FROM data_access da '
            'LEFT JOIN schema_symbols s ON s.canonical_id = da.schema_symbol_id '
            'WHERE da.source_name = ?', (source,)):
        if consumer:
            sinks[consumer].append({'role': role or '', 'table': table or '',
                                    'column': column or '', 'schema': schema_symbol})
    callers_of_endpoint: dict = defaultdict(list)
    for consumer, endpoint_id in conn.execute(
            'SELECT consumer_symbol_id, endpoint_id FROM api_calls WHERE endpoint_id '
            'IS NOT NULL'):
        if consumer:
            callers_of_endpoint[endpoint_id].append(consumer)
    return endpoints, sinks, callers_of_endpoint


def route_to_any(out, start: str, targets: set, *, max_hops: int = MAX_HOPS):
    """Shortest call route from ``start`` to any target — breadth-first, so it is shortest.

    Shortest matters: a clew is a claim about how a request reaches data, and the shortest route
    is the one least likely to have wandered. Longer alternatives exist and say no more about
    *whether* the path exists.
    """
    if start in targets:
        return (start,)
    seen = {start}
    queue = deque([(start, (start,))])
    while queue:
        node, route = queue.popleft()
        if len(route) > max_hops:
            continue
        for callee in out.get(node, ()):
            if callee in seen:
                continue
            seen.add(callee)
            extended = route + (callee,)
            if callee in targets:
                return extended
            queue.append((callee, extended))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True,
                        help='indexed corpus whose HTTP and data-model tiers are populated')
    parser.add_argument('--show', type=int, default=8)
    parser.add_argument('--db', default=None,
                        help='store to read; defaults to ./ariadne.db')
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(
        f'file:{args.db or (ROOT / "ariadne.db")}?mode=ro', uri=True)
    name, out, dropped = load_graph(conn, args.source)
    endpoints, sinks, callers_of_endpoint = anchors(conn, args.source)
    conn.close()
    print(f'{args.source}: {len(name):,} symbols, '
          f'{sum(len(v) for v in out.values()):,} call edges '
          f'({dropped:,} edges dropped as outside the caller body)')
    print(f'anchors: {len(endpoints)} endpoints, {len(sinks)} data-touching symbols, '
          f'{sum(len(v) for v in callers_of_endpoint.values())} client calls')
    if not endpoints or not sinks:
        print('\nThis archetype needs both endpoints and data access. One or both is empty '
              'for this\nsource, so no clew can be anchored — see spool_clews for a corpus '
              'without endpoints.')
        return 1

    targets = set(sinks)
    clews: dict = {}
    unreached = []
    for endpoint in endpoints:
        route = route_to_any(out, endpoint['symbol'], targets)
        if route is None:
            unreached.append(endpoint)
            continue
        touched = sinks[route[-1]]
        clews[(endpoint['method'], endpoint['path'], route)] = {
            'entry': f'{endpoint["method"]} {endpoint["path"]}'.strip(),
            'route': [name[n] for n in route],
            'hops': len(route) - 1,
            'touches': sorted({f'{t["table"]}{"." + t["column"] if t["column"] else ""}'
                               f' ({t["role"]})' for t in touched if t['table']}),
            'client_callers': sorted({name[c] for c
                                      in callers_of_endpoint.get(endpoint['endpoint_id'], [])
                                      if c in name}),
        }
    print(f'\nclews generated: {len(clews)}  (endpoints with no route to data: '
          f'{len(unreached)})')
    for endpoint in unreached[:4]:
        print(f'   no route: {endpoint["method"]} {endpoint["path"]}')

    print()
    for clew in list(clews.values())[:args.show]:
        arrow = ' -> '.join(part.split('.')[-1] for part in clew['route'])
        print(f'  {clew["entry"]:<28} {arrow}')
        if clew['touches']:
            print(f'  {"":<28} touches: {", ".join(clew["touches"])}')
        if clew['client_callers']:
            print(f'  {"":<28} called from: '
                  f'{", ".join(c.split(".")[-1] for c in clew["client_callers"][:3])}')
    if clews:
        hops = [c['hops'] for c in clews.values()]
        print(f'\nroute length: min {min(hops)}, max {max(hops)}, '
              f'mean {sum(hops)/len(hops):.1f} hops')
        print(f'distinct routes after dedup: '
              f'{len({tuple(c["route"]) for c in clews.values()})}')
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(list(clews.values()), indent=2), encoding='utf-8')
        print(f'written: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
