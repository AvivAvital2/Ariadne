#!/usr/bin/env python
"""What would the answer call cost if nothing were selected? Deterministic, spends nothing.

Selection exists to cut the answer call down, and it was introduced against a payload of
**quoted source**: 953,190 chars for one chain, $1.19 a question. Stage two no longer sends
source. It sends the catalog document for each hop — a signature and a paragraph — and the
premise that the whole chain is unaffordable has not been re-checked since.

It matters because selection is now the recall bottleneck. The chain reaches more of the
answer key than the model picks out of the menu, so every point selection drops is a point
the path had and threw away. If the whole bundle fits in a 1M-token context at a price worth
paying, the cheapest route to end-to-end recall is not a better selection prompt — it is not
selecting.

Measured per question, deduplicated by ``document_id`` because one document serves every hop
into the same symbol:

    menu        what the selection call reads (the price of asking)
    whole       every distinct document the chain reached (the price of not asking)
    spine only  documents on the structural spine (today's fallback when selection fails)

Reported for the forward walk and for the caller-rooted walk from ``measure_backward``, since
that walk reaches 91.2% of the answer key and is the one worth affording.

Usage
-----
    .venv/bin/python evaluation/answer-path/measure_payload.py
    .venv/bin/python evaluation/answer-path/measure_payload.py --limit 10
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
from recall_matching import spine_of  # noqa: E402

DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'
KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'


def payload(bundle, *, only: set | None = None) -> tuple[int, int]:
    """``(distinct documents, chars)`` — one document per symbol, however many hops.

    Deduplication is the whole point of measuring here rather than summing hops: the
    ``runMerge`` chain enters some bodies dozens of times, and a payload that repeated the
    document each time would overstate the cost of not selecting by an order of magnitude.
    """
    seen: dict[str, int] = {}
    for hop in bundle.hops:
        if not hop.document_id or not hop.evidence:
            continue
        if only is not None and hop.citation.qualified_name not in only:
            continue
        seen[hop.document_id] = len(hop.evidence)
    return len(seen), sum(seen.values())


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
    from library.chain_menu import menu_for
    from library.structural_assembly import (DEFAULT_EXPAND_FAN_IN_MAX, chain_from_seeds,
                                             seeds_from_documents)

    model = get_config().model
    rates = LLM_PRICING.get(model)
    window = context_window_tokens(model)
    dump = json.loads(DUMP.read_text(encoding='utf-8'))
    key = {e['id']: e for e in json.loads(KEY.read_text(encoding='utf-8'))}
    root = str(get_config().get_all_source_paths().get(args.source) or '') or None
    cwds = indexer_cwds(root) if root else ()
    library = Library(ROOT / 'ariadne.db')
    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    results = dump['results'][:args.limit] if args.limit else dump['results']

    rows: list[dict] = []
    for result in results:
        wanted = key.get(result['id'], {}).get('required', [])
        documents = [{'content': d.get('content') or '',
                      'source_files': d.get('source_files') or []}
                     for d in result['widths']['8']['documents']]
        if not wanted or not documents:
            continue
        print(f'  q{result["id"]} ...', end='', flush=True)
        seeds = seeds_from_documents(conn, documents, source=args.source,
                                     indexer_cwds=cwds, source_root=root)
        row: dict = {'id': result['id'], 'family': result.get('family', '')}
        for label, symbols in (('forward', seeds.seeds), ('rooted', None)):
            if symbols is None:
                added, _ = callers_of(conn, seeds.seeds,
                                      fan_in_max=DEFAULT_EXPAND_FAN_IN_MAX)
                symbols = list(set(seeds.seeds) | added)
            citations, _ = chain_from_seeds(conn, symbols, source=args.source,
                                            depth=args.depth)
            if not citations:
                break
            bundle = curate_bundle(library, citations, source=args.source)
            menu = menu_for(library, bundle.hops, source=args.source)
            whole_docs, whole_chars = payload(bundle)
            spine_docs, spine_chars = payload(bundle, only=spine_of(citations))
            row[label] = {
                'hops': len(citations),
                'symbols': len({c.qualified_name for c in citations}),
                'menu_chars': len(menu.text),
                'menu_lines': len(menu.symbols),
                'whole_docs': whole_docs, 'whole_chars': whole_chars,
                'spine_docs': spine_docs, 'spine_chars': spine_chars,'prompt_chars': len(render_spine(bundle, None))
            }
        if 'forward' in row:
            rows.append(row)
            print(f' whole {row.get("rooted", row["forward"])["whole_chars"]:,} chars',
                  flush=True)
        else:
            print(' no chain', flush=True)
    conn.close()

    def price(chars: int) -> float:
        return (chars / CHARS_PER_TOKEN) * (rates[0] if rates else 0) / 1_000_000

    print('\n' + '=' * 100)
    print(f'THE PRICE OF NOT SELECTING — {model}'.center(100))
    print('=' * 100)
    print(f'{"q":>4} {"family":<14} {"menu":>8} {"whole":>9} {"spine":>8}  |  '
          f'{"menu":>8} {"whole":>9} {"spine":>8}   (chars; forward | caller-rooted)')
    print('-' * 100)
    for row in rows:
        f, r = row['forward'], row.get('rooted', row['forward'])
        print(f'{row["id"]:>4} {row["family"][:14]:<14} {f["menu_chars"]:>8,} '
              f'{f["whole_chars"]:>9,} {f["spine_chars"]:>8,}  |  '
              f'{r["menu_chars"]:>8,} {r["whole_chars"]:>9,} {r["spine_chars"]:>8,}')
    print('-' * 100)
    n = max(len(rows), 1)
    print(f'{"walk":<14} {"docs/q":>7} {"chars/q":>10} {"tokens/q":>10} {"$/question":>11} '
          f'{"% of window":>12}')
    for label in ('forward', 'rooted'):
        present = [r[label] for r in rows if label in r]
        if not present:
            continue
        for kind in ('whole', 'spine'):
            chars = sum(p[f'{kind}_chars'] for p in present) / n
            docs = sum(p[f'{kind}_docs'] for p in present) / n
            share = (f'{(chars / CHARS_PER_TOKEN) / (window * 1.0):>11.1%}'
                     if window else f'{"unknown":>11}')
            print(f'{label + " / " + kind:<14} {docs:>7.0f} {chars:>10,.0f} '
                  f'{chars / CHARS_PER_TOKEN:>10,.0f} {price(chars):>11.3f} {share:>12}')
        menu_chars = sum(p['menu_chars'] for p in present) / n
        print(f'{label + " / menu":<14} {"-":>7} {menu_chars:>10,.0f} '
              f'{menu_chars / CHARS_PER_TOKEN:>10,.0f} {price(menu_chars):>11.3f} '
              f'{(menu_chars / CHARS_PER_TOKEN) / window if window else 0:>11.1%}')
    print(f'\nA selection call costs its menu whether or not it picks well. Sending the '
          f'whole bundle\ninstead costs the "whole" row and cannot lose a symbol the chain '
          f'reached.')
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'written: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
