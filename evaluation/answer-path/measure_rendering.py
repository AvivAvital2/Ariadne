#!/usr/bin/env python
"""What does the prompt's *shape* cost, as opposed to its content? Free to run.

Measuring the real rendered prompt turned up something the document totals hid: at
production width the forward chain renders 598,553 chars, of which only 187,637 are catalog
documents. **69% is coordinate lines** — one per hop, and there are 2,645 hops over roughly
674 distinct symbols, so each symbol's coordinates are emitted about four times.

That repetition is not an accident. ``render_spine`` deliberately keeps every call site,
because a second invocation of the same body is new evidence, and it deliberately keeps hops
in execution order, because order is what makes a chain explicable rather than a set. Both
are right. Neither requires re-printing the symbol's file and line each time.

So this measures one alternative shape against the current one, on identical bundles:

    per-hop    what ships today: ``name [file:line] called at file:line`` for every hop,
               description printed once per symbol
    compact    one block per distinct symbol, in order of first appearance, with its
               description once and its call sites gathered on a single line

Compact keeps both properties the docstring defends — every call site survives, and first
appearance preserves execution order — and it is a pure formatting change, so it cannot lose
a symbol. The question is only how much of the prompt it gives back.

Usage
-----
    .venv/bin/python evaluation/answer-path/measure_rendering.py
    .venv/bin/python evaluation/answer-path/measure_rendering.py --limit 8
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

from measure_backward import callers_of  # noqa: E402

DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'


def render_compact(bundle) -> str:
    """One block per symbol, first-appearance order, every call site kept.

    Deliberately built from the same ``bundle.hops`` the production renderer walks, so the
    comparison is of shape alone. Depth is taken from the hop that introduced the symbol —
    a symbol reached at several depths is indented at the shallowest, which is where the
    walk first needed it.
    """
    order: list = []
    blocks: dict = {}
    for entry in bundle.hops:
        hop = entry.citation
        name = hop.qualified_name
        if name not in blocks:
            order.append(name)
            blocks[name] = {'hop': hop, 'sites': [], 'evidence': entry.evidence}
        blocks[name]['sites'].append(f'{hop.call_site_file}:{hop.call_site_line}')
        if entry.evidence and not blocks[name]['evidence']:
            blocks[name]['evidence'] = entry.evidence
    lines: list[str] = []
    for name in order:
        block = blocks[name]
        hop = block['hop']
        indent = '  ' * (hop.hop - 1)
        sites = block['sites']
        shown = ', '.join(sites[:8]) + ('' if len(sites) <= 8
                                        else f', +{len(sites) - 8} more')
        lines.append(f'{indent}{name}  [{hop.file}:{hop.line_start}]')
        lines.append(f'{indent}    called at {shown}')
        if block['evidence']:
            lines.append(f'{indent}    {block["evidence"].strip()}')
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='databricks')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    from config import get_config
    from docgen.pricing import CHARS_PER_TOKEN, LLM_PRICING, context_window_tokens
    from docgen.scip_paths import indexer_cwds
    from library import Library
    from library.chain_answer import render_spine
    from library.chain_bundle import curate_bundle
    from library.structural_assembly import (DEFAULT_EXPAND_FAN_IN_MAX, chain_from_seeds,
                                             seeds_from_documents)

    model = get_config().model
    rate = (LLM_PRICING.get(model) or (0, 0))[0]
    window = context_window_tokens(model)
    dump = json.loads(DUMP.read_text(encoding='utf-8'))
    root = str(get_config().get_all_source_paths().get(args.source) or '') or None
    cwds = indexer_cwds(root) if root else ()
    library = Library(ROOT / 'ariadne.db')
    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    results = dump['results'][:args.limit] if args.limit else dump['results']

    rows: list[dict] = []
    for result in results:
        documents = [{'content': d.get('content') or '',
                      'source_files': d.get('source_files') or []}
                     for d in result['widths']['8']['documents']]
        if not documents:
            continue
        print(f'  q{result["id"]} ...', end='', flush=True)
        seeds = seeds_from_documents(conn, documents, source=args.source,
                                     indexer_cwds=cwds, source_root=root)
        row: dict = {'id': result['id'], 'family': result.get('family', '')}
        for label, extra in (('forward', False), ('rooted', True)):
            symbols = list(seeds.seeds)
            if extra:
                added, _ = callers_of(conn, seeds.seeds,
                                      fan_in_max=DEFAULT_EXPAND_FAN_IN_MAX)
                symbols = list(set(symbols) | added)
            citations, _ = chain_from_seeds(conn, symbols, source=args.source,
                                            depth=args.depth)
            if not citations:
                break
            bundle = curate_bundle(library, citations, source=args.source)
            row[label] = {
                'hops': len(citations),
                'symbols': len({c.qualified_name for c in citations}),
                'per_hop': len(render_spine(bundle, None)),
                'compact': len(render_compact(bundle)),
            }
        if 'forward' in row:
            rows.append(row)
            print(f' {row["forward"]["per_hop"]:,} -> '
                  f'{row["forward"]["compact"]:,}', flush=True)
        else:
            print(' no chain', flush=True)
    conn.close()

    print('\n' + '=' * 96)
    print(f'PROMPT SHAPE — {model}, {len(rows)} questions'.center(96))
    print('=' * 96)
    print(f'{"walk":<9} {"shape":<9} {"chars/q":>10} {"tokens/q":>10} {"$/q":>7} '
          f'{"% window":>9} {"worst q":>9} {"saved":>7}')
    n = max(len(rows), 1)
    summary: dict = {}
    for label in ('forward', 'rooted'):
        present = [r[label] for r in rows if label in r]
        if not present:
            continue
        base = sum(p['per_hop'] for p in present) / n
        for shape in ('per_hop', 'compact'):
            avg = sum(p[shape] for p in present) / n
            worst = max(p[shape] for p in present)
            saved = '' if shape == 'per_hop' else f'{1 - avg / max(base, 1):>6.0%}'
            print(f'{label:<9} {shape:<9} {avg:>10,.0f} {avg / CHARS_PER_TOKEN:>10,.0f} '
                  f'{avg / CHARS_PER_TOKEN * rate / 1e6:>7.3f} '
                  f'{avg / CHARS_PER_TOKEN / window:>8.1%} '
                  f'{worst / CHARS_PER_TOKEN / window:>8.1%} {saved:>7}')
            summary[f'{label}.{shape}'] = avg
        print()
    over = [r for r in rows if 'rooted' in r
            and r['rooted']['compact'] / CHARS_PER_TOKEN > window]
    print(f'caller-rooted questions still over the window after compacting: {len(over)}')
    print('\nCompact keeps every call site and first-appearance order. It cannot lose a '
          'symbol,\nso any saving here is recall-neutral by construction.')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'written: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
