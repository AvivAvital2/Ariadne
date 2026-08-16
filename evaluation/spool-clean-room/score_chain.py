#!/usr/bin/env python3
"""Mechanical completeness: did the answer show how A leads to B?

Correctness is graded by an LLM judge elsewhere. This decides, without judgement,
whether an answer is admitted for scoring at all -- because the judge cannot hold
this line. Measured on the first grounded run, the derivation rubric awarded 8/10
to nine answers that quoted no code whatsoever, and scored answers with a fully
traced chain 9.83 against 9.50 for answers with no evidence at all: fluent prose
with citations nearby reads as demonstration.

An answer is COMPLETE only when every required mechanism, fixed in advance in
causal order, carries a verbatim quote from a file that the arm opened. Every
substantive quoted line is checked at the exact cited location. Endpoint-only
evidence is incomplete: it cannot distinguish a derived middle from recall.

An answer is ADMISSIBLE when it is complete, and additionally

  * for a cross-repo question, verified quotes exist in >=2 repositories; and
  * for an SDK-spanning question only, those quotes share a literal that
    provably appears on both sides. delta<->spark are code-coupled, so their
    link is a call or an override; the SDK has no call path into either, so a
    shared table property or config key is the only thing that can join them.

Correct-but-inadmissible scores nothing. A correct answer with no traced chain
cannot be told apart from one recalled and then confirmed with a single lookup.
Pass --grades to fold an LLM judge's correctness onto this gate, which is the
only form of it worth quoting: correctness earned over every question asked.

Hashes recorded in the container are compared against the local corpus, so a
mismatch means the oracle is not the bytes the arm read, and the answer is not
scored rather than scored wrongly.

    python evaluation/spool-clean-room/score_chain.py \\
        --answers evaluation/cleanroom-claude/answers-cleanroom-questions_debcrumb_ask.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REQS = HERE / 'chain_requirements.json'
KEYFILE = HERE / 'questions_debcrumb_25.json'
SYMBOL = re.compile(r'\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]{2,}\b')
CITE = re.compile(
    r'(/corpus/(?:[\w.+-]+/)*[A-Za-z_][\w+.-]*\.(?:scala|java|py|g4))'
    r'\s*:\s*(\d+)(?:\s*-\s*(\d+))?')
FENCE = re.compile(r'```[a-zA-Z0-9]*\n(.*?)```', re.S)
ANNOT = re.compile(r'//\s*:?\d*\s*<--.*$|//\s*:\d+\s*$')
# A fence may open with the path it was taken from, and may prefix each line
# with that line's number. Both are line claims and both are checked.
PATHHDR = re.compile(
    r'^\s*(/corpus/(?:[\w.+-]+/)*[A-Za-z_][\w+.-]*\.(?:scala|java|py|g4))'
    r'(?::(\d+))?\s*$')
LEADNUM = re.compile(r'^\s*(\d{1,5})[:|\s]\s*(\S.*)$')
# Candidate joining artifacts: dotted config/table-property style constants.
LITERAL = re.compile(r'\b(?:delta|spark)\.[a-zA-Z][a-zA-Z0-9.]{4,60}\b')


def norm(s: str) -> str:
    return ' '.join(ANNOT.sub('', s).split())


def repo_of(path: Path, corpus: Path) -> str:
    try:
        return path.relative_to(corpus).parts[0]
    except ValueError:
        return '?'
def index(corpus: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    for p in corpus.rglob('*'):
        if p.suffix in ('.scala', '.java', '.py', '.g4') and p.is_file():
            idx.setdefault(p.name, []).append(p)
    return idx


def resolve(fname, idx, read_paths):
    """Which copy of a same-named file the answer meant (per its own tool log)."""
    paths = idx.get(fname) or []
    if len(paths) > 1 and read_paths:
        pref = [p for p in paths if any(str(p).endswith(rp) for rp in read_paths)]
        if pref:
            return pref
    return paths
def fence_claim(raw: str, *, min_chars: int = 15):
    """(filename, line, [(inline_no, stripped, as_written)]) parsed from a fence.

    Answers state where a quote came from in three ways: a path header opening
    the fence, a line number prefixed to every line, or neither -- leaving the
    citation in the surrounding prose. Measured on the archived run, 5 of 22
    answers used one of the first two and scored zero verified quotes against a
    verifier that only knew the third, so a formatting choice was being read as
    fabricated evidence. Whichever way it is stated, the number is the claim.
    """
    lines = raw.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    fname = hdr_line = None
    if lines:
        m = PATHHDR.match(lines[0])
        if m:
            fname = m.group(1)
            hdr_line = int(m.group(2)) if m.group(2) else None
            lines.pop(0)
    keep = []
    for relative, x in enumerate(lines):
        m = LEADNUM.match(x)
        n, body = (int(m.group(1)), m.group(2)) if m else (None, x)
        written = norm(x)
        # A source line may legitimately begin with a digit, so the unstripped
        # form is kept alongside and either may satisfy the match.
        if len(written) > min_chars and not written.startswith(('...', '//')):
            keep.append((n, norm(body), written, relative))
    return fname, hdr_line, keep
def _local_path(corpus: Path, claimed: str) -> Path:
    return corpus / claimed.removeprefix('/corpus/')


def provenance_ok(rec: dict, corpus: Path) -> bool:
    """Every recorded source file has a hash matching the scoring corpus.

    Historical tool logs include directory paths from Glob in ``files_read``.
    They are discovery events, not source files and cannot be hashed. A quoted
    path is separately required to occur in ``file_hashes`` by
    :func:`verified_quotes`, so an unhashed source can never become evidence.
    """
    hashes = rec.get('file_hashes') or {}
    if not hashes:
        return False
    return all(
        (p := _local_path(corpus, path)).is_file()
        and hashlib.sha256(p.read_bytes()).hexdigest()[:16] == hashes[path]
        for path in hashes
    )


def verified_quotes(rec, idx, corpus, cache, tol=0, *, min_chars: int = 15):
    """Code blocks whose full contents match an exact, opened path and line."""
    text = rec.get('answer') or ''
    cites = [(m.start(), m.group(1), int(m.group(2)), int(m.group(3) or m.group(2)))
             for m in CITE.finditer(text)]
    read = set(rec.get('files_read') or [])
    hashed = set((rec.get('file_hashes') or {}))
    out = []
    for fm in FENCE.finditer(text):
        hdr_name, hdr_line, keep = fence_claim(fm.group(1), min_chars = min_chars)
        if not keep:
            continue
        near = sorted(cites, key=lambda c: abs(c[0] - fm.start()))
        near = [c for c in near if c[1] in read]
        claimed = hdr_name or (near[0][1] if near else None)
        if not claimed or claimed not in read or claimed not in hashed:
            continue
        p = _local_path(corpus, claimed)
        if not p.is_file():
            continue
        if p not in cache:
            cache[p] = [norm(x) for x in p.read_text(errors='ignore').splitlines()]
        body = cache[p]
        start = hdr_line or (near[0][2] if near else None)
        if start is None:
            continue
        matched = True
        for inline, stripped, written, relative in keep:
            line = inline if inline is not None else start + relative
            want = stripped if inline is not None else written
            if line < 1 or line > len(body) or body[line - 1] != want:
                matched = False
                break
        if matched:
            out.append({'file': p, 'repo': repo_of(p, corpus), 'line': start,
                        'text': ' '.join(t for _, t, _, _ in keep)})
    return out
def completeness(rec, req, quotes):
    """Does the answer show how the first mechanism reaches the last one?

    Every required mechanism must occur in verified source. Prose mentions do
    not count because they can be recalled and decorated with endpoint quotes.
    """
    syms = [r['symbol'] for r in req['required']]
    blob = ' '.join(q['text'] for q in quotes)
    stems = {q['file'].stem for q in quotes}

    def quoted(s):
        return s in blob or s in stems

    missing = [s for s in syms if not quoted(s)]
    full = len(quotes) >= len(syms) and not missing
    return {'complete_full': full, 'complete_endpoints': False,
            'complete': full, 'missing_middle': missing,
            'endpoint_repos': []}
def score_answer(rec, req, idx, corpus, cache, tol):
    quotes = verified_quotes(rec, idx, corpus, cache, tol)
    blob = ' '.join(q['text'] for q in quotes)
    stems = {q['file'].stem for q in quotes}

    evidenced, missing = [], []
    for r in req['required']:
        s = r['symbol']
        (evidenced if (s in blob or s in stems) else missing).append(s)
    hops = len(req['required'])
    chain = (len(evidenced) / hops) if hops else None

    repos = sorted({q['repo'] for q in quotes})
    cross_needed = len(req['repos_spanned']) >= 2
    # A shared literal is the link ONLY where no call path exists. delta<->spark
    # are genuinely code-coupled (delta forks and extends Spark classes), so
    # demanding a shared config constant there would fail correct answers. The
    # SDK is the case with no call edge, so that is where a literal is the only
    # thing that can join the sides.
    literal_needed = 'databricks-sdk-py' in req['repos_spanned']
    # A joining literal must appear in verified quotes from two DIFFERENT repos.
    per_repo = {}
    for q in quotes:
        per_repo.setdefault(q['repo'], set()).update(LITERAL.findall(q['text']))
    joiners = sorted({l for a in per_repo for b in per_repo if a < b
                      for l in per_repo[a] & per_repo[b]})

    comp = completeness(rec, req, quotes)
    provenance = provenance_ok(rec, corpus)
    admissible = (bool(hops) and provenance and comp['complete']
                  and (not cross_needed or len(repos) >= 2)
                  and (not literal_needed or bool(joiners)))
    return {'id': req['id'], 'family': req.get('family'), 'hops': hops,
            'verified_quotes': len(quotes), 'evidenced': len(evidenced),
            'missing': missing, 'chain_score': chain, 'repos': repos,
            'cross_repo_required': cross_needed,
            'literal_required': literal_needed, 'joining_literals': joiners,
            **comp, 'provenance_ok': provenance, 'admissible': admissible}


def search_order(rec: dict, q: dict) -> dict:
    """Did the arm go straight to the answer, or find it by exploring?

    Reported, never gated. Tool calls are recorded in order, and the
    de-breadcrumbed question deliberately withholds the mechanism's name -- so an
    arm whose FIRST move is to open the file the answer key names did not derive
    that name from the corpus. It knew it, and the chain was assembled backwards
    around a known conclusion.

    Not a gate, because the signal is suggestive rather than conclusive: a
    question describing "forked file-writing machinery" makes globbing
    *FileFormatWriter* a competent inference, not necessarily recall. It also
    UNDER-counts -- an arm can open a file named in neither question nor key and
    still be working from memory.
    """
    key_only = ({s for s in SYMBOL.findall(q.get('before', ''))}
                - {s for s in SYMBOL.findall(q.get('after', ''))})
    calls = rec.get('tool_calls') or []
    first = None
    for i, c in enumerate(calls):
        blob = f"{c.get('pattern', '')} {c.get('path', '')}"
        if any(s in blob for s in key_only):
            first = i
            break
    opener = (calls[0].get('pattern') or calls[0].get('path') or '') if calls else ''
    return {'first_key_hit': first, 'tool_calls': len(calls),
            'answer_first': first is not None and first <= 1,
            'opening_search': opener[:60]}
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--answers', required=True)
    ap.add_argument('--corpus', default=str(REPO / 'spool-corpus'))
    ap.add_argument('--tol', type=int, default=0,
                    help='deprecated; exact line matching is always enforced')
    ap.add_argument('--grades', default=None,
                    help='jsonl of LLM correctness grades to gate on this result')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    reqs = {r['id']: r for r in json.loads(REQS.read_text())}
    answers = {a['id']: a for a in json.loads(Path(args.answers).read_text())}
    idx, cache = index(corpus), {}

    # Hash provenance: are we checking the bytes the container actually read?
    checked = mismatched = 0
    for a in answers.values():
        for path, want in (a.get('file_hashes') or {}).items():
            p = corpus / path.lstrip('/').removeprefix('corpus/')
            if p.exists():
                checked += 1
                if hashlib.sha256(p.read_bytes()).hexdigest()[:16] != want:
                    mismatched += 1
    if checked:
        print(f'hash provenance: {checked} files checked, {mismatched} mismatched')
        if mismatched:
            print('ERROR: the local corpus is not byte-identical to what the '
                  'container read; refusing to score.')
            return 2
    else:
        print('ERROR: no hash provenance recorded; refusing to score.')
        return 2

    qs = {q['id']: q for q in json.loads(KEYFILE.read_text())}
    rows = [score_answer(answers[i], r, idx, corpus, cache, args.tol)
            for i, r in sorted(reqs.items())
            if not r['flag'] and i in answers]
    for r in rows:
        r.update(search_order(answers[r['id']], qs[r['id']]))

    print(f'\n{"id":>5} {"hops":>5} {"evid":>5} {"chain":>6} {"repos":>5} '
          f'{"join":>5} {"shape":>9} {"admiss":>7} {"1st-key":>8}  derivation')
    for r in sorted(rows, key=lambda r: -(r['chain_score'] or 0)):
        fk = r['first_key_hit']
        shape = 'full' if r['complete_full'] else '-'
        print(f'{r["id"]:>5} {r["hops"]:>5} {r["evidenced"]:>5} '
              f'{(r["chain_score"] or 0):>6.0%} {len(r["repos"]):>5} '
              f'{len(r["joining_literals"]):>5} {shape:>9} '
              f'{("YES" if r["admissible"] else "no"):>7} '
              f'{(str(fk) if fk is not None else "never"):>8}  '
              f'{"answer-first" if r["answer_first"] else "explored"}')

    adm = [r for r in rows if r['admissible']]
    cs = [r['chain_score'] for r in rows if r['chain_score'] is not None]
    nfull = sum(1 for r in rows if r['complete_full'])
    print(f'\n  ADMISSIBLE: {len(adm)}/{len(rows)}  (full chain {nfull})')
    print(f'  chain_score: mean {sum(cs)/len(cs):.0%}  median {st.median(cs):.0%}')
    print(f'  answers with zero verified quotes: '
          f'{sum(1 for r in rows if r["verified_quotes"] == 0)}/{len(rows)}')
    print(f'  answers with only one verified quote: '
          f'{sum(1 for r in rows if r["verified_quotes"] == 1)}/{len(rows)}')
    incomplete = sum(1 for r in rows if r['missing_middle'])
    print(f'  missing required source evidence: {incomplete}/{len(rows)}')
    af = sum(1 for r in rows if r['answer_first'])
    print(f'  ANSWER-FIRST (key symbol sought in the first two calls): '
          f'{af}/{len(rows)}')
    print(f'  admissible AND answer-first: '
          f'{sum(1 for r in rows if r["admissible"] and r["answer_first"])} '
          f'-- a verified chain can still be assembled backwards')

    if args.grades:
        grades = {}
        for line in Path(args.grades).read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                if g.get('score') is not None:
                    grades[g['id']] = g['score']
        scored = [r for r in rows if r['id'] in grades]
        if scored:
            earned = sum(grades[r['id']] for r in scored if r['admissible'])
            gated = [grades[r['id']] for r in scored if r['admissible']]
            ungated = [grades[r['id']] for r in scored if not r['admissible']]
            print(f'\n  correctness, admissible   : '
                  f'{(st.mean(gated) if gated else 0):.2f}  (n={len(gated)})')
            print(f'  correctness, inadmissible : '
                  f'{(st.mean(ungated) if ungated else 0):.2f}  (n={len(ungated)})'
                  f'  <- a small gap here means the judge cannot see grounding')
            print(f'  EARNED correctness        : {earned / len(scored):.2f}/10'
                  f'  (inadmissible scores nothing, over all {len(scored)})')
            for r in rows:
                r['correctness'] = grades.get(r['id'])

    out = Path(args.out) if args.out else HERE / f'chain-{Path(args.answers).stem}.json'
    out.write_text(json.dumps(rows, indent=2) + '\n')
    print(f'  detail -> {out.name}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
