#!/usr/bin/env python3
"""designs/answer-path.md §5.1 — the graph half, run offline from the retrieval dump.

§5.1 published: seeded from "the symbols defined in the files retrieved at production
width (k=8)", **call only adds 0.0pp** over docs alone (37.1% -> 37.1%), while type_ref
at depth 2 adds +27.4pp. §2.6's correction showed that family of measurement seeded on
CamelCase TYPE symbols, which carry no outgoing call edge.

Re-run here with the corrections §2.6 established:
  * seeds = every named databricks symbol DEFINED IN a retrieved file, then expanded to
    declared members via parent_qualified_name (never via `kind` — blank on scip-python);
  * `call` edges only for the chain, source-guarded, `local N` nodes excluded from
    traversal (19.5% of the call graph fuses through them);
  * depth 4 — the answer key's `hops` median — not 1-2;
  * absolute, not marginal-only: docs-alone and docs+SCIP both reported;
  * stratified by `lens_primary`, because a repo-primary route ranks against the half of
    the databricks closure that holds ZERO documents and returns 3 docs instead of 8.

PATH NORMALIZATION (the join seam, exact by construction): `documents.source_files` for a
spool carry a repo-name prefix (`delta/spark/src/...`); `scip_symbols.file` is
repo-relative (`spark/src/...`). The leading segment is stripped only when it equals one
of the three corpus repo roots, which are exactly the directories present in
spool-corpus/. Suffix-free, substring-free: one segment, one allow-list.

    .venv/bin/python evaluation/measurements/remeasure_5_1.py
"""
from __future__ import annotations

import collections
import json
import re
import sqlite3
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DB = REPO / 'ariadne.db'
DUMP = Path(__file__).resolve().parent / 'retrieval_dump.json'
REQS = REPO / 'evaluation' / 'spool-clean-room' / 'chain_requirements.json'
SRC = 'databricks'
REPO_ROOTS = ('spark', 'delta', 'databricks-sdk-py')
DEPTH = 4
VISIT_CAP = 250_000

con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
TEST_RE = re.compile(r'(^|/)(test|tests|it)/|Suite\.scala$|(^|/)test_[^/]+\.py$|_test\.py$')
is_test = lambda f: bool(TEST_RE.search(f))


def strip_repo(path: str) -> str:
    head, _, rest = path.partition('/')
    return rest if head in REPO_ROOTS and rest else path


NAMED = {r[0] for r in con.execute(
    "SELECT canonical_id FROM scip_symbols WHERE source_name=? "
    "AND canonical_id NOT LIKE 'local %'", (SRC,))}

_fc: dict[str, str] = {}


def files_of(ids):
    out, todo = set(), []
    for i in ids:
        (out.add(_fc[i]) if i in _fc else todo.append(i))
    for i in range(0, len(todo), 800):
        c = todo[i:i + 800]
        qs = ','.join('?' * len(c))
        for cid, f in con.execute(
                f'SELECT canonical_id, file FROM scip_symbols WHERE canonical_id IN ({qs})', c):
            _fc[cid] = f
            out.add(f)
    return out


def symbols_in_files(files):
    """Named databricks symbols defined in these files — the §5.1 seed rule."""
    out = set()
    fl = list(files)
    for i in range(0, len(fl), 400):
        c = fl[i:i + 400]
        qs = ','.join('?' * len(c))
        out |= {r[0] for r in con.execute(
            f'SELECT canonical_id, qualified_name FROM scip_symbols '
            f'WHERE source_name=? AND file IN ({qs}) '
            f"AND canonical_id NOT LIKE 'local %'", [SRC] + c)}
    return out


def members(ids):
    """Expand to declared members via parent_qualified_name, to fixpoint (2 levels)."""
    if not ids:
        return set()
    qns, il = set(), list(ids)
    for i in range(0, len(il), 800):
        c = il[i:i + 800]
        qs = ','.join('?' * len(c))
        qns |= {r[0] for r in con.execute(
            f'SELECT qualified_name FROM scip_symbols WHERE canonical_id IN ({qs})', c)}
    out, frontier = set(), qns
    for _ in range(2):
        if not frontier:
            break
        got, fl = [], list(frontier)
        for i in range(0, len(fl), 800):
            c = fl[i:i + 800]
            qs = ','.join('?' * len(c))
            got += con.execute(
                f'SELECT canonical_id, qualified_name FROM scip_symbols '
                f'WHERE source_name=? AND parent_qualified_name IN ({qs}) '
                f"AND canonical_id NOT LIKE 'local %'", [SRC] + c).fetchall()
        new = {g[0] for g in got} - out
        out |= new
        frontier = {g[1] for g in got}
    return out


def walk(seeds, depth, types=('call',)):
    tq = ','.join('?' * len(types))
    seeds = {s for s in seeds if s in NAMED}
    seen, frontier, capped = set(seeds), set(seeds), False
    for _ in range(depth):
        nxt = set()
        fl = list(frontier)
        for i in range(0, len(fl), 400):
            c = fl[i:i + 400]
            qs = ','.join('?' * len(c))
            for (n,) in con.execute(
                    f'SELECT DISTINCT callee_canonical_id FROM scip_edges '
                    f'WHERE caller_canonical_id IN ({qs}) AND edge_type IN ({tq})',
                    c + list(types)):
                if n in NAMED:
                    nxt.add(n)
        frontier = nxt - seen
        seen |= frontier
        if len(seen) > VISIT_CAP:
            capped = True
            break
        if not frontier:
            break
    return seen, capped


# ------------------------------------------------------------------ slots
def slot_files(sym):
    rows = con.execute('SELECT file FROM scip_symbols WHERE source_name=? AND display_name=?',
                       (SRC, sym)).fetchall()
    prod = [r[0] for r in rows if not is_test(r[0])]
    return set(prod or [r[0] for r in rows])


reqs = {r['id']: r for r in json.loads(REQS.read_text())}
slots = {}
for q in reqs.values():
    for s in q['required']:
        slots.setdefault(s['symbol'], slot_files(s['symbol']))

dump = json.loads(DUMP.read_text())
print(f"dump {dump['captured_at']} · store call edges "
      f"{dump['store']['edge_counts'].get('call'):,} · widths {dump['widths']}")

# ------------------------------------------------------------------ join-seam report
allf, matched, unmatched = set(), set(), set()
for rec in dump['results']:
    for d in rec['widths']['8']['documents']:
        for f in (d.get('source_files') or []):
            allf.add(f)
for f in allf:
    n = strip_repo(f)
    if con.execute("SELECT 1 FROM scip_symbols WHERE source_name=? AND file=? LIMIT 1",
                   (SRC, n)).fetchone():
        matched.add(n)
    else:
        unmatched.add(n)
ext = collections.Counter(('.' + f.rsplit('.', 1)[-1]) if '.' in f else '(none)'
                          for f in unmatched)
print(f'\njoin seam: {len(allf)} retrieved files · {len(matched)} join to scip_symbols '
      f'after stripping the repo prefix · {len(unmatched)} do not')
print(f'  unmatched by extension: {dict(ext.most_common())}')
code_ext = {'.scala', '.java', '.py', '.js', '.ts', '.jsx'}
code_unmatched = sorted(f for f in unmatched
                        if '.' in f and '.' + f.rsplit('.', 1)[-1] in code_ext)
print(f'  of those, CODE files that should have been indexed: {len(code_unmatched)}')
for f in code_unmatched[:5]:
    print(f'     {f}')

# ------------------------------------------------------------------ the measurement
def measure(width, types, depth):
    rows = []
    for rec in dump['results']:
        qid = rec['id']
        if qid not in reqs:
            continue
        want = [s['symbol'] for s in reqs[qid]['required']]
        want = [s for s in want if slots.get(s)]
        if not want:
            continue
        got_files = set()
        for d in rec['widths'][str(width)]['documents']:
            for f in (d.get('source_files') or []):
                got_files.add(strip_repo(f))
        docs_hit = {s for s in want if slots[s] & got_files}
        seeds = symbols_in_files(got_files)
        seeds |= members(seeds)
        reached, capped = walk(seeds, depth, types)
        rf = files_of(reached)
        scip_hit = {s for s in want if slots[s] & rf}
        rows.append({
            'id': qid, 'lens': rec['widths'][str(width)]['lens_primary'],
            'n_docs': len(rec['widths'][str(width)]['documents']),
            'slots': len(want), 'docs': len(docs_hit),
            'union': len(docs_hit | scip_hit), 'added': len(scip_hit - docs_hit),
            'seeds': len(seeds), 'reached': len(reached), 'capped': capped,
        })
    return rows


def report(rows, label):
    def agg(sel):
        sl = sum(r['slots'] for r in sel)
        if not sl:
            return None
        return (len(sel), sl, sum(r['docs'] for r in sel), sum(r['union'] for r in sel),
                sum(r['added'] for r in sel))
    print(f'\n=== {label} ===')
    print(f'   {"stratum":16s} {"qs":>3} {"slots":>6} {"docs alone":>11} '
          f'{"docs+SCIP":>10} {"SCIP adds":>10}')
    for name, sel in (('ALL', rows),
                      ('lens=spool', [r for r in rows if r['lens'] == 'spool']),
                      ('lens=repo', [r for r in rows if r['lens'] == 'repo'])):
        a = agg(sel)
        if not a:
            continue
        nq, sl, dh, un, ad = a
        print(f'   {name:16s} {nq:>3} {sl:>6} {dh:>6} ({100*dh/sl:3.0f}%) '
              f'{un:>5} ({100*un/sl:3.0f}%) {ad:>4} (+{100*ad/sl:.1f}pp)')
    cap = sum(1 for r in rows if r['capped'])
    if cap:
        print(f'   [{cap} question(s) hit the {VISIT_CAP:,} visit cap — lower bound]')
    return rows


base = report(measure(8, ('call',), DEPTH), f'width 8 (production) · call only · depth {DEPTH}')
report(measure(8, ('call',), 2), 'width 8 · call only · depth 2 (the original depth)')
report(measure(8, ('call', 'type_ref'), DEPTH), f'width 8 · call+type_ref · depth {DEPTH}')
report(measure(40, ('call',), DEPTH), f'width 40 · call only · depth {DEPTH}')

print('\nper-question (width 8, call only, depth 4):')
print(f'   {"id":>4} {"lens":>6} {"docs":>5} {"slots":>6} {"d":>3} {"d+S":>4} {"+":>3} '
      f'{"seeds":>7} {"reached":>8}')
for r in base:
    print(f'   {r["id"]:>4} {r["lens"]:>6} {r["n_docs"]:>5} {r["slots"]:>6} '
          f'{r["docs"]:>3} {r["union"]:>4} {r["added"]:>3} {r["seeds"]:>7} {r["reached"]:>8}')
con.close()
