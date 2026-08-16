#!/usr/bin/env python
"""Does the model pick the right lines out of the menu? The one thing still unmeasured.

The two-call path offers the chain as a numbered menu and sends only what comes back. Every
measurement so far has been about what the menu *contains* — 100% of what the chain reached,
if no selection rule prunes it. This measures what the model *takes*, against answer keys it
has never seen: ``evaluation/spool-clean-room/chain_requirements.json``, 25 questions whose
required symbols were fixed in advance by the eval harness, not by this script.

Three numbers per question, and the third is the one that matters:

    in menu   required symbols the menu offered  (the ceiling — selection cannot beat it)
    selected  required symbols the model asked for
    covered   required symbols the answer call would carry: selected OR on the spine

``covered`` is the composition under test: the structural spine travels regardless, so a
selection that misses something the spine already holds costs nothing. That is what makes
selection safe to be imperfect, and it is exactly what has never been checked.

Retrieval is replayed from ``evaluation/measurements/retrieval_dump.json`` (real
``ariadne_search`` output, width 8, captured 2026-08-04), so the input is production-shaped
and needs no embedding key. The selection prompt is imported from the service, not restated
here — measuring a paraphrase of the prompt would measure nothing.

Usage
-----
    # deterministic: menu sizes and the ceiling, nothing spent
    .venv/bin/python evaluation/answer-path/measure_selection.py

    # the real selection call per question (needs a provider key; prints the cost first)
    .venv/bin/python evaluation/answer-path/measure_selection.py --live --limit 4
    .venv/bin/python evaluation/answer-path/measure_selection.py --live --json sel.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# One matcher for every recall figure in this directory — it drifted once and the
# duplicate credited symbols that were not there.
from measure_backward import callers_of  # noqa: E402

# One matcher for every recall figure in this directory — it drifted once and the
# duplicate credited symbols that were not there.
from recall_matching import ambiguous, hits, spine_of  # noqa: E402

DUMP = ROOT / 'evaluation/measurements/retrieval_dump.json'
KEY = ROOT / 'evaluation/spool-clean-room/chain_requirements.json'
SOURCE = 'databricks'


@dataclass
class Row:
    """One question's result. ``None`` fields mean not measured, never zero."""

    id: int
    family: str = ''
    menu_lines: int = 0
    menu_chars: int = 0
    required: int = 0
    reached: int = 0
    in_menu: int = 0
    #: required symbols whose menu match spans several packages
    ambiguous: int = 0
    spine: int = 0
    selected: 'int | None' = None
    covered: 'int | None' = None
    picks: 'int | None' = None
    unknown: 'int | None' = None
    reply: str = ''
    missed: list = field(default_factory=list)
def build(conn, cwds, root, documents, depth: int, caller_depth: int = 0):
    """The chain the menu is built from.

    ``caller_depth`` roots the walk at the seeds' callers as well and walks them that far.
    A forward-only walk cannot reach a caller at any depth, and six of seven symbols the
    forward chain missed were callers, which is what takes reach from 66.7% to 76.0%.
    Depth 2 is the cheapest point that reaches it: depth 3 found the same 73 symbols for
    197 more per question.
    """
    from library.structural_assembly import (DEFAULT_EXPAND_FAN_IN_MAX, chain_from_seeds,
                                             seeds_from_documents)
    seeds = seeds_from_documents(conn, documents, source=SOURCE, indexer_cwds=cwds,
                                 source_root=root)
    citations, _ = chain_from_seeds(conn, seeds.seeds, source=SOURCE, depth=depth)
    if caller_depth:
        added, _gated = callers_of(conn, seeds.seeds,
                                   fan_in_max=DEFAULT_EXPAND_FAN_IN_MAX)
        if added:
            up, _truncation = chain_from_seeds(conn, list(added), source=SOURCE,
                                               depth=caller_depth)
            # A hop is the same hop only if it is the same symbol reached from the same
            # call site; the two walks legitimately share symbols and each call site is
            # separate evidence.
            seen = {(c.qualified_name, c.call_site_file, c.call_site_line)
                    for c in citations}
            citations = citations + [
                c for c in up
                if (c.qualified_name, c.call_site_file, c.call_site_line) not in seen]
    return citations


#: The wording production used BEFORE this measurement. Kept so the comparison that changed
#: production stays reproducible: on identical menus it found 7 of 10 required symbols and took
#: 1.47% of the lines, against 9 of 10 and 5.01% for the end-to-end wording that shipped.
#: ``--prompt minimal`` re-runs the losing arm.
MINIMAL_PROMPT = (
    'A compiler-derived index found the code below while tracing the call chain for a '
    'question. Say which entries you need in order to answer it.\n\n'
    'Reply with numbers only, comma-separated — plain numbers for definitions, S-prefixed '
    'for sections (for example: 3, 7, 12, S2). Choose only what the question needs; '
    'choose nothing if the answer does not require any of them.\n\n'
    'Question: {question}\n\n{menu}'
)

#: Superseded by production, retained for the record.
COMPLETE_PROMPT = (
    'A compiler-derived index found the code below while tracing the call chain for a '
    'question. Choose the entries needed to explain the mechanism END TO END.\n\n'
    'Include every definition on the path: where it starts, each step that carries it '
    'forward, the conditions that decide between paths, and where the work lands. Include the '
    'types the answer would have to name. Err toward including a step you are unsure about — '
    'a missing step cannot be recovered later, while an extra one only costs a little '
    'reading.\n\n'
    'Reply with numbers only, comma-separated — plain numbers for definitions, S-prefixed for '
    'sections (for example: 3, 7, 12, S2).\n\n'
    'Question: {question}\n\n{menu}'
)


async def select(question: str, menu_text: str, *, wording: str = 'production') -> str:
    """The selection call. ``production`` uses the service's own prompt, unmodified."""
    from ariadne_mcp.service_analysis import MENU_MAX_TOKENS, _menu_prompt
    from llm import chat_complete
    template = {'minimal': MINIMAL_PROMPT, 'complete': COMPLETE_PROMPT}.get(wording)
    prompt = (_menu_prompt(question, menu_text) if template is None
              else template.format(question=question, menu=menu_text))
    return await chat_complete(
        messages=[
            {'role': 'system',
             'content': 'You select evidence. Reply with numbers only.'},
            {'role': 'user', 'content': prompt},
        ],
        max_tokens=MENU_MAX_TOKENS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true',
                        help='make one selection call per question (spends)')
    parser.add_argument('--limit', type=int, default=None,
                        help='only the first N questions — start small')
    parser.add_argument('--prompt', default='production',
                        choices=('production', 'complete', 'minimal'),
                        help="'production' uses the service's own prompt; 'minimal' re-runs "
                             "the wording it replaced")
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--caller-depth', type=int, default=0,
                        help='also root the walk at the seeds callers '
                             'and walk them this far (2 is the measured sweet spot)')
    parser.add_argument('--json', dest='json_out', default=None)
    args = parser.parse_args()

    from config import get_config
    from docgen.pricing import CHARS_PER_TOKEN, LLM_PRICING
    from docgen.scip_paths import indexer_cwds
    from library import Library
    from library.chain_bundle import curate_bundle
    from library.chain_menu import menu_for, resolve_selection

    model = get_config().model
    rates = LLM_PRICING.get(model)
    dump = json.loads(DUMP.read_text(encoding='utf-8'))
    key = {entry['id']: entry for entry in json.loads(KEY.read_text(encoding='utf-8'))}
    root = str(get_config().get_all_source_paths().get(SOURCE) or '') or None
    cwds = indexer_cwds(root) if root else ()
    library = Library(ROOT / 'ariadne.db')

    results = dump['results'][:args.limit] if args.limit else dump['results']
    rows: list[Row] = []
    conn = sqlite3.connect(f'file:{ROOT / "ariadne.db"}?mode=ro', uri=True)
    for result in results:
        wanted = [r['symbol'] for r in key.get(result['id'], {}).get('required', [])]
        documents = [{'content': d.get('content') or '',
                      'source_files': d.get('source_files') or []}
                     for d in result['widths']['8']['documents']]
        row = Row(id=result['id'], family=result.get('family', ''), required=len(wanted))
        if not wanted or not documents:
            rows.append(row)
            continue
        citations = build(conn, cwds, root, documents, args.depth, args.caller_depth)
        if not citations:
            rows.append(row)
            continue
        bundle = curate_bundle(library, citations, source=SOURCE)
        menu = menu_for(library, bundle.hops, source=SOURCE)
        names = {c.qualified_name for c in citations}
        spine = spine_of(citations)
        row.menu_lines = len(menu.symbols)
        row.menu_chars = len(menu.text)
        row.reached = len(hits(wanted, names))
        offered = set(menu.symbols.values())
        row.in_menu = len(hits(wanted, offered))
        row.ambiguous = len(ambiguous(hits(wanted, offered), offered))
        row.spine = len(hits(wanted, spine))
        if args.live:
            reply = asyncio.run(select(
                key[result['id']].get('question') or result['question'], menu.text,
                wording=args.prompt))
            chosen = resolve_selection(menu, reply)
            row.reply = reply.strip()[:120]
            row.picks = len(chosen.symbols)
            row.unknown = len(chosen.unknown)
            selected_names = set(chosen.symbols)
            row.selected = len(hits(wanted, selected_names))
            covered = hits(wanted, selected_names | spine)
            row.covered = len(covered)
            row.missed = sorted(set(hits(wanted, names)) - covered)
        rows.append(row)
    conn.close()

    # ---------------------------------------------------------------- one table
    print('\n' + '=' * 104)
    print('MENU SELECTION vs ANSWER KEYS'.center(104))
    print('=' * 104)
    print(f'{"q":>4} {"family":<16} {"menu":>6} {"req":>4} {"reach":>6} {"inMenu":>7} '
          f'{"amb":>4} {"spine":>6} {"picks":>6} {"sel":>4} {"cover":>6}  missed')
    print('-' * 104)
    def cell(value):
        return 'n/m' if value is None else str(value)
    for row in rows:
        print(f'{row.id:>4} {row.family[:16]:<16} {row.menu_lines:>6} {row.required:>4} '
              f'{row.reached:>6} {row.in_menu:>7} {row.ambiguous:>4} {row.spine:>6} '
              f'{cell(row.picks):>6} '
              f'{cell(row.selected):>4} {cell(row.covered):>6}  '
              f'{",".join(row.missed[:3])}')
    print('-' * 104)
    walked = [r for r in rows if r.reached]
    reach = sum(r.reached for r in walked)
    print(f'questions with a chain: {len(walked)} of {len(rows)}')
    print(f'required symbols reached by the chain: {reach}')
    offered = sum(r.in_menu for r in walked)
    amb = sum(r.ambiguous for r in walked)
    print(f'  offered by the menu : {offered} '
          f'({offered / max(reach, 1):.1%} — loose containment)')
    print(f'    of which ambiguous: {amb} — matched >1 package; strict ceiling '
          f'{offered - amb} ({(offered - amb) / max(reach, 1):.1%})')
    print(f'  held by the spine   : {sum(r.spine for r in walked)} '
          f'({sum(r.spine for r in walked) / max(reach, 1):.1%}, travels regardless)')
    if args.live:
        sel = sum(r.selected or 0 for r in walked)
        cov = sum(r.covered or 0 for r in walked)
        picks = sum(r.picks or 0 for r in walked)
        lines = sum(r.menu_lines for r in walked)
        print(f'  chosen by the model : {sel} ({sel / max(reach, 1):.1%})')
        print(f'  COVERED (chosen or on the spine): {cov} ({cov / max(reach, 1):.1%})')
        print(f'  the model picked {picks} lines of {lines} offered '
              f'({picks / max(lines, 1):.2%}), '
              f'{sum(r.unknown or 0 for r in walked)} unresolvable')
        chars = sum(r.menu_chars for r in walked)
        cost = (chars / CHARS_PER_TOKEN) * (rates[0] if rates else 0) / 1_000_000
        print(f'  input spent: ~${cost:.2f} across {len(walked)} questions '
              f'(~${cost / max(len(walked), 1):.3f} each) on {model}')
    else:
        chars = sum(r.menu_chars for r in walked)
        cost = (chars / CHARS_PER_TOKEN) * (rates[0] if rates else 0) / 1_000_000
        print(f'\nNo selection call was made, so "picks", "sel" and "cover" are NOT MEASURED.')
        print(f'--live would spend about ${cost:.2f} in input across {len(walked)} '
              f'questions (~${cost / max(len(walked), 1):.3f} each) on {model}.')
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps([vars(r) for r in rows], indent=2, default=str), encoding='utf-8')
        print(f'written: {args.json_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
