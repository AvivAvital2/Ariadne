#!/usr/bin/env python3
"""The Ariadne arm: answer the same questions through ``ask``, scored the same way.

Until now the two arms were never comparable. The bare arm is graded on
mechanical grounding — a quote that sits at the line it cites — while Ariadne's
answering path cited document titles and nothing else, so the grounding gate
would have returned 0/21 for every answer regardless of quality. Not a
measurement of Ariadne, a measurement of its output format.

With the code tier wired into ``ask`` (``VERIFIED CODE LOCATIONS``, resolved from
``scip_symbols``), an Ariadne answer can carry ``file:line`` and can therefore be
put through the identical gate: same de-breadcrumbed questions, same fixed
``chain_requirements.json``, same ``score_chain.py``.

What is deliberately NOT equalized: the bare arm reads the corpus and Ariadne
does not. Ariadne answers from documents plus the SCIP index. That is the
comparison — an index against a reader — not a handicap to correct for.

Needs BOTH keys: Anthropic synthesizes the answer, OpenAI embeds the query for
retrieval (Anthropic has no embeddings API). Runs 4-wide by default.

    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \\
        python evaluation/spool-clean-room/ariadne_arm.py
    python evaluation/spool-clean-room/score_chain.py \\
        --answers evaluation/spool-clean-room/answers-ariadne-arm.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

QUESTIONS = HERE / 'questions_debcrumb_ask.json'
REQS = HERE / 'chain_requirements.json'
async def _one(service, qid: int, question: str, source: str) -> dict:
    """Ask once and record what Ariadne surfaced, including where it looked."""
    t0 = time.time()
    resp = await service.ask(question, source=source)

    # `chain_files` is every file the assembled chain put in front of synthesis. It replaces
    # a re-derivation through `_candidate_symbols`, which this script imported and which has
    # never existed in the repository — so the arm raised ImportError for all 22 questions
    # and wrote an empty answers file. Reading it off the response is also the honest set:
    # the grounding gate checks a citation against where the arm actually looked, and using
    # only what the answer cited would let the gate satisfy itself.
    files = list(getattr(resp, 'chain_files', None) or [])

    return {
        'id': qid,
        'question': question,
        'answer': resp.answer or '',
        'sources': list(getattr(resp, 'sources', None) or []),
        'confidence': getattr(resp, 'confidence', None),
        'files_read': files,
        'tool_calls': [],          # Ariadne does not explore; search_order is n/a
        'elapsed_s': round(time.time() - t0, 1),
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='databricks')
    ap.add_argument('--questions', default=str(QUESTIONS))
    ap.add_argument('--out', default=str(HERE / 'answers-ariadne-arm.json'))
    ap.add_argument('--only', default=None,
                    help='comma-separated ids, for a cheap smoke run')
    ap.add_argument('--concurrency', type=int, default=4,
                    help='questions in flight (default 4, matching the judge)')
    args = ap.parse_args()

    # Score only what the bare arm was scored on: `flag` marks questions whose
    # required chain the rewrite judged flawed.
    scored = {r['id'] for r in json.loads(REQS.read_text()) if not r['flag']}
    qs = [q for q in json.loads(Path(args.questions).read_text())
          if q['id'] in scored]
    if args.only:
        keep = {int(x) for x in args.only.split(',')}
        qs = [q for q in qs if q['id'] in keep]
    print(f'Ariadne arm: {len(qs)} question(s), source={args.source}, '
          f'{args.concurrency}-wide')

    from ariadne_mcp.service import AriadneService
    service = AriadneService.get()

    # Bounded concurrency. Each question is an independent retrieval plus one
    # synthesis call, so they overlap freely; the cap exists because these are
    # LLM round-trips, not CPU work, and an unbounded fan-out would just
    # collect rate-limit errors.
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def guarded(q: dict) -> dict:
        text = q.get('after') or q.get('question') or ''
        async with sem:
            row = await _one(service, q['id'], text, args.source)
        print(f'  id {q["id"]:>3}  {row["elapsed_s"]:>6}s  '
              f'{len(row["answer"]):>5} chars  '
              f'{len(row["files_read"])} code ref(s)  conf={row["confidence"]}')
        return row

    settled = await asyncio.gather(*(guarded(q) for q in qs),
                                   return_exceptions=True)
    rows, failed = [], []
    for q, res in zip(qs, settled):
        if isinstance(res, BaseException):
            # Surface it rather than silently shrinking the denominator -- a
            # missing answer is not an unevidenced answer, and conflating them
            # would flatter the arm.
            failed.append((q['id'], f'{type(res).__name__}: {res}'))
            continue
        rows.append(res)
    # Deterministic on disk regardless of completion order, so a re-run diffs.
    rows.sort(key=lambda r: r['id'])

    out = Path(args.out)
    out.write_text(json.dumps(rows, indent=2) + '\n')
    if failed:
        print(f'\n{len(failed)} question(s) FAILED and are absent from the '
              f'output (not scored as zero):', file=sys.stderr)
        for qid, why in failed:
            print(f'  id {qid}: {why}', file=sys.stderr)
    print(f'\nanswers ({len(rows)}/{len(qs)}) -> {out}')
    print('score with:\n'
          f'  python evaluation/spool-clean-room/score_chain.py --answers {out}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
