#!/usr/bin/env python3
"""Capture the retrieval half of designs/answer-path.md §5.1 — the one retracted
measurement that cannot be redone offline.

§5.1 recorded **"call only: SCIP adds 0.0pp"**, seeded from *"the symbols defined in the
files retrieved at production width (k=8)"*. The §2.6 correction (2026-08-04) showed that
measurement family had been seeded on CamelCase **type** symbols, which carry no outgoing
`call` edge, so the walk terminated at hop 0 and the figure is unverified. Re-running it
needs the retrieval side at production width — and a query embedding is the only part of
the pipeline behind an API key.

This script does ONLY that part. For each de-breadcrumbed question it runs the production
retrieval path — ``AriadneService.search``, the same call ``ask`` makes at
``ariadne_mcp/service_analysis.py:207`` — and writes what came back to JSON. No synthesis,
no judge, no grading: 25 questions x len(widths) query embeddings, fractions of a cent.

**Needs OPENAI_API_KEY only.** Embeddings are OpenAI (`text-embedding-3-large`); no
Anthropic call is made, because nothing here synthesizes an answer.

    OPENAI_API_KEY=... .venv/bin/python evaluation/measurements/capture_retrieval.py

Writes ``evaluation/measurements/retrieval_dump.json`` (refuses to clobber an existing one
without ``--force``). The graph half then runs offline from that file: seed from the symbols
defined in the retrieved files, expand to declared members via ``parent_qualified_name``,
walk ``call`` edges source-guarded with ``local N`` nodes excluded, and score required-slot
reach against the docs-alone baseline.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

CLEANROOM = REPO / 'evaluation' / 'spool-clean-room'
QUESTIONS = CLEANROOM / 'questions_debcrumb_25.json'
REQS = CLEANROOM / 'chain_requirements.json'

# Everything except `content` and `sections` — a retrieved explanation averages 8.4k
# chars, and 25 questions x 40 docs of body text would be an 8 MB dump for context the
# graph half never reads.
KEEP = ('id', 'title', 'content_type', 'source_name', 'score', 'source_files')
META_KEEP = ('location', 'file', 'path', 'subtype', 'language', 'qualified_name',
             'provenance', 'element_kind')


def _slim(doc) -> dict:
    """One retrieved document, reduced to what the graph half joins on."""
    out: dict = {}
    for key in KEEP:
        val = getattr(doc, key, None)
        if isinstance(val, (str, int, float, bool)) or val is None or isinstance(val, list):
            out[key] = val
        else:
            out[key] = str(val)
    meta = getattr(doc, 'metadata', None)
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            meta = {'_raw': meta[:200]}
    if isinstance(meta, dict):
        out['metadata'] = {k: meta[k] for k in META_KEEP if k in meta}
    else:
        out['metadata'] = {}
    return out


def _store_provenance(db: Path) -> dict:
    """Pin the dump to the store it was taken from, so the offline half cannot
    silently score retrieval from one graph against edges from another."""
    try:
        con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    except sqlite3.Error as exc:
        return {'error': str(exc)}
    try:
        state = [dict(zip(('source_name', 'scip_path', 'sha256', 'indexed_at'), r))
                 for r in con.execute(
                     'SELECT source_name, scip_path, file_sha256, indexed_at '
                     'FROM scip_index_state')]
        edges = dict(con.execute(
            'SELECT edge_type, COUNT(*) FROM scip_edges GROUP BY 1').fetchall())
        symbols = con.execute('SELECT COUNT(*) FROM scip_symbols').fetchone()[0]
    except sqlite3.Error as exc:
        return {'error': str(exc)}
    finally:
        con.close()
    return {'scip_index_state': state, 'edge_counts': edges, 'scip_symbols': symbols}


async def _capture(service, qid: int, text: str, source: str,
                   widths: list[int]) -> dict:
    rec: dict = {'id': qid, 'question': text, 'widths': {}}
    for width in widths:
        t0 = time.time()
        resp = await service.search(query=text, limit=width, source=source)
        docs = list(getattr(resp, 'documents', None) or [])
        rec['widths'][str(width)] = {
            'documents': [_slim(d) for d in docs],
            'lens_primary': getattr(resp, 'lens_primary', None),
            'truncated': getattr(resp, 'truncated', None),
            'event_id': getattr(resp, 'event_id', None),
            'spool_connections': bool(getattr(resp, 'spool_connections', None)),
            'elapsed_s': round(time.time() - t0, 1),
        }
    return rec


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default='databricks',
                    help='scope argument passed to search (default: databricks)')
    ap.add_argument('--questions', default=str(QUESTIONS))
    ap.add_argument('--widths', default='8,40',
                    help='retrieval widths to capture; 8 is production width, 40 gives '
                         'the offline half a width sweep for free (default: 8,40)')
    ap.add_argument('--out', default=str(HERE / 'retrieval_dump.json'))
    ap.add_argument('--only', default=None, help='comma-separated question ids')
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--db', default=str(REPO / 'ariadne.db'))
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing dump (refused by default)')
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f'{out} already exists — pass --force to replace it, or --out to write '
              f'somewhere new. Refusing to clobber a previous capture.', file=sys.stderr)
        return 2

    widths = [int(w) for w in args.widths.split(',') if w.strip()]
    flagged = {r['id']: bool(r['flag']) for r in json.loads(REQS.read_text())}
    questions = json.loads(Path(args.questions).read_text())
    # `after` is the de-breadcrumbed phrasing (it names 9 of 97 required symbols;
    # `before` names 97 of 97, which is why it cannot be the measured input).
    rows = [{'id': q['id'],
             'text': (q.get('after') or q.get('question') or q.get('before') or '').strip(),
             'family': q.get('family')}
            for q in questions]
    if args.only:
        keep = {int(x) for x in args.only.split(',')}
        rows = [r for r in rows if r['id'] in keep]
    rows = [r for r in rows if r['text']]
    if not rows:
        print('no questions to capture', file=sys.stderr)
        return 1

    print(f'capturing retrieval for {len(rows)} question(s) at width(s) {widths}, '
          f'source={args.source}, {args.concurrency}-wide')
    print('OpenAI embeddings only — no synthesis, no judge.')

    from ariadne_mcp.service import AriadneService
    service = AriadneService.get()
    try:
        closure = sorted(service._resolve_scope(args.source).closure)
    except Exception as exc:  # noqa: BLE001 — provenance only
        closure = [f'unresolved: {exc}']
    print(f'scope closure: {closure}')
    # Ranking has a cliff, not a slope: the mmap'd matrix is ~66 ms, the SQLite
    # fallback ~100x that. Say which one is about to run so a slow capture is
    # explained rather than mysterious.
    try:
        from library.embedding_matrix import EmbeddingMatrix, matrix_dir_for
        matrix = EmbeddingMatrix.load(matrix_dir_for(service.library))
        print('embedding matrix: ' + ('loaded' if matrix is not None else
              'NOT BUILT for this store — expect the ~6.8 s SQLite fallback per query '
              '(build it with `ariadne rebuild` if you would rather not wait)'))
    except Exception as exc:  # noqa: BLE001 — diagnostic only, never fatal
        print(f'embedding matrix: check failed ({type(exc).__name__}: {exc})')

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def guarded(row: dict) -> dict:
        async with sem:
            rec = await _capture(service, row['id'], row['text'], args.source, widths)
        rec['family'] = row['family']
        rec['flag'] = flagged.get(row['id'])
        first = rec['widths'][str(widths[0])]
        print(f'  id {row["id"]:>3}  {len(first["documents"]):>2} docs  '
              f'{first["elapsed_s"]:>5}s  lens={first["lens_primary"]}')
        return rec

    settled = await asyncio.gather(*(guarded(r) for r in rows),
                                   return_exceptions=True)
    captured, failed = [], []
    for row, res in zip(rows, settled):
        if isinstance(res, BaseException):
            # Report rather than shrink the denominator: a failed capture is not an
            # empty retrieval, and conflating them would flatter the docs-alone baseline.
            failed.append((row['id'], f'{type(res).__name__}: {res}'))
        else:
            captured.append(res)
    captured.sort(key=lambda r: r['id'])

    payload = {
        'captured_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'purpose': 'designs/answer-path.md §5.1 re-run — retrieval half',
        'source_arg': args.source,
        'scope_closure': closure,
        'widths': widths,
        'questions_file': str(Path(args.questions).name),
        'store': _store_provenance(Path(args.db)),
        'failed': [{'id': i, 'error': e} for i, e in failed],
        'results': captured,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + '\n')

    if failed:
        print(f'\n{len(failed)} question(s) FAILED and are recorded as failures, not as '
              f'empty retrievals:', file=sys.stderr)
        for qid, why in failed:
            print(f'  id {qid}: {why}', file=sys.stderr)
    total_docs = sum(len(r['widths'][str(widths[0])]['documents']) for r in captured)
    print(f'\ncaptured {len(captured)}/{len(rows)} question(s), '
          f'{total_docs} docs at width {widths[0]} -> {out}')
    print('the offline graph half reads only this file plus ariadne.db.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
