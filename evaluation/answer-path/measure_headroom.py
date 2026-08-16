#!/usr/bin/env python
"""Where the missing half lives, and whether depth or menu hygiene reaches it. Free to run.

End to end the path carries about 52% of what the answer keys require — 66.7% reached by the
chain, of which the model selects 77.8%. Reaching 90% needs roughly 95% x 95%, so this asks
three deterministic questions about the headroom:

A. **Depth.** Depth 3 was chosen when hops drove prompt cost. They no longer do: the menu is
   priced per distinct symbol and the answer call carries only what was picked. So does depth
   4 reach more of the key, and what does it cost in menu size?

B. **Where the missing symbols are.** For every required symbol the chain fails to reach,
   which is it:
     absent        — not in ``scip_symbols`` at all (an ingest gap)
     one edge out  — indexed, and an edge connects it to a symbol the chain did reach
                     (a walk-shape gap: depth, breadth or seeds)
     unconnected   — indexed, but no edge joins it to anything in the chain (the connectivity
                     gap; the eval's own notes say cross-project pairs join by shared literal,
                     not by a call, so a call-graph walk cannot reach these at all)

C. **Generated code.** One required symbol had 114 of 116 menu lines under
   ``io.delta.connect.proto``. How much of a menu is protobuf scaffolding, and would filtering
   it lose anything the key requires?

Usage
-----
    .venv/bin/python evaluation/answer-path/measure_headroom.py
    .venv/bin/python evaluation/answer-path/measure_headroom.py --depths 3 4 --limit 8
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from recall_matching import hits  # noqa: E402

DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'
KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'

#: Marks of generated code in a qualified name. A heuristic, and named as one: it is used to
#: measure how much of a menu is scaffolding, never to decide what an answer may cite.
GENERATED = ('.proto.', 'Builder', 'DEFAULT_INSTANCE', 'newBuilder', 'parseFrom',
             'getDescriptor', 'FIELD_NUMBER', 'internalGet', 'toBuilder', 'mergeFrom')


def is_generated(qualified_name: str) -> bool:
    return any(mark in qualified_name for mark in GENERATED)


def segment_index(conn, source: str) -> dict:
    """``segment -> qualified names containing it``, built in one pass over the source.

    The first version asked ``qualified_name LIKE '%.Symbol'`` per missing symbol. A leading
    wildcard cannot use an index, so each lookup scanned 306,471 rows and two runs of this
    script had to be killed. One pass and a dict does the same work once.
    """
    index: dict = {}
    for (name,) in conn.execute(
            'SELECT DISTINCT qualified_name FROM scip_symbols WHERE source_name = ? '
            "AND canonical_id NOT LIKE 'local %'", (source,)):
        for segment in name.split('.'):
            index.setdefault(segment, set()).add(name)
    return index


def edges_touching(conn, source: str, names: set) -> set:
    """Qualified names one edge away from ``names``, in either direction."""
    if not names:
        return set()
    ids = []
    listed = sorted(names)
    for start in range(0, len(listed), 300):
        chunk = listed[start:start + 300]
        placeholders = ','.join('?' * len(chunk))
        ids += [row[0] for row in conn.execute(
            f'SELECT canonical_id FROM scip_symbols WHERE source_name = ? '
            f'AND qualified_name IN ({placeholders})', [source, *chunk])]
    neighbours: set = set()
    for start in range(0, len(ids), 300):
        chunk = ids[start:start + 300]
        placeholders = ','.join('?' * len(chunk))
        for row in conn.execute(
                f'SELECT s.qualified_name FROM scip_edges e '
                f'JOIN scip_symbols s ON s.canonical_id = e.callee_canonical_id '
                f'WHERE e.caller_canonical_id IN ({placeholders}) AND s.source_name = ?',
                [*chunk, source]):
            neighbours.add(row[0])
        for row in conn.execute(
                f'SELECT s.qualified_name FROM scip_edges e '
                f'JOIN scip_symbols s ON s.canonical_id = e.caller_canonical_id '
                f'WHERE e.callee_canonical_id IN ({placeholders}) AND s.source_name = ?',
                [*chunk, source]):
            neighbours.add(row[0])
    return neighbours


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--depths', type=int, nargs='+', default=[3, 4])
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    from config import get_config
    from docgen.scip_paths import indexer_cwds
    from library.structural_assembly import chain_from_seeds, seeds_from_documents

    dump = json.loads(DUMP.read_text(encoding='utf-8'))
    key = {e['id']: e for e in json.loads(KEY.read_text(encoding='utf-8'))}
    root = str(get_config().get_all_source_paths().get(args.source) or '') or None
    cwds = indexer_cwds(root) if root else ()
    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    started = time.time()
    by_segment = segment_index(conn, args.source)
    print(f'segment index: {len(by_segment):,} distinct name segments '
          f'in {time.time() - started:.1f}s', flush=True)
    results = dump['results'][:args.limit] if args.limit else dump['results']

    per_depth = {d: {'reached': 0, 'symbols': 0, 'hops': 0, 'seconds': 0.0}
                 for d in args.depths}
    required_total = 0
    attribution = Counter()
    unreached_names: list = []
    generated_share = {'menu': 0, 'generated': 0, 'required_generated': 0}

    for result in results:
        wanted = [r['symbol'] for r in key.get(result['id'], {}).get('required', [])]
        documents = [{'content': d.get('content') or '',
                      'source_files': d.get('source_files') or []}
                     for d in result['widths']['8']['documents']]
        if not wanted or not documents:
            continue
        required_total += len(wanted)
        print(f'  q{result["id"]} ...', end='', flush=True)
        seeds = seeds_from_documents(conn, documents, source=args.source,
                                     indexer_cwds=cwds, source_root=root)
        deepest_names: set = set()
        for depth in args.depths:
            started = time.time()
            citations, _ = chain_from_seeds(conn, seeds.seeds, source=args.source,
                                           depth=depth)
            elapsed = time.time() - started
            names = {c.qualified_name for c in citations}
            per_depth[depth]['reached'] += len(hits(wanted, names))
            per_depth[depth]['symbols'] += len(names)
            per_depth[depth]['hops'] += len(citations)
            per_depth[depth]['seconds'] += elapsed
            if depth == args.depths[0]:
                deepest_names = names
        # B: where do the symbols the shallowest depth missed actually live?
        missed = [w for w in wanted if w not in hits(wanted, deepest_names)]
        if missed:
            indexed = {symbol for symbol in missed if symbol in by_segment}
            neighbours = edges_touching(conn, args.source, deepest_names)
            for symbol in missed:
                if symbol not in indexed:
                    attribution['absent from the index'] += 1
                elif hits([symbol], neighbours):
                    attribution['one edge out of the chain'] += 1
                else:
                    attribution['no edge to anything in the chain'] += 1
                unreached_names.append(symbol)
        print(f' done', flush=True)
        # C: generated share of what the menu would offer
        generated_share['menu'] += len(deepest_names)
        generated_share['generated'] += sum(1 for n in deepest_names if is_generated(n))
        generated_share['required_generated'] += len(
            hits(wanted, {n for n in deepest_names if is_generated(n)}))
    conn.close()

    print('\n' + '=' * 92)
    print(f'HEADROOM — source {args.source}, {len(results)} questions'.center(92))
    print('=' * 92)
    print('A. DEPTH')
    print(f'   {"depth":>6} {"reached":>8} {"of required":>12} {"symbols/q":>11} '
          f'{"hops/q":>9} {"secs/q":>8}')
    n = max(len([r for r in results if key.get(r['id'], {}).get('required')]), 1)
    for depth in args.depths:
        stat = per_depth[depth]
        print(f'   {depth:>6} {stat["reached"]:>8} '
              f'{stat["reached"] / max(required_total, 1):>11.1%} '
              f'{stat["symbols"] / n:>11.0f} {stat["hops"] / n:>9.0f} '
              f'{stat["seconds"] / n:>8.1f}')
    print(f'\nB. WHERE THE MISSING SYMBOLS ARE (at depth {args.depths[0]}, '
          f'{sum(attribution.values())} unreached)')
    for cause, count in attribution.most_common():
        print(f'   {count:>4}  {cause}')
    print(f'\nC. GENERATED CODE IN WHAT THE MENU WOULD OFFER')
    menu = generated_share['menu']
    print(f'   {generated_share["generated"]:,} of {menu:,} symbols look generated '
          f'({generated_share["generated"] / max(menu, 1):.1%})')
    print(f'   required symbols that ONLY match generated names: '
          f'{generated_share["required_generated"]} — filtering would lose these')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {'depth': per_depth, 'attribution': dict(attribution),
             'generated': generated_share, 'required_total': required_total,
             'unreached': unreached_names}, indent=2), encoding='utf-8')
        print(f'\nwritten: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
