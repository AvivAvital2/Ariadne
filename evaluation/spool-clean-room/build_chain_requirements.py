#!/usr/bin/env python3
"""Fix each question's required chain in advance, from the answer key.

Admissibility asks whether an answer evidenced the whole chain. That is only
meaningful if the chain is decided BEFORE the answer exists: if the requirement
were read off what an answer claims, the winning strategy would be to claim less
-- collapse the explanation to one step, quote it, and score a perfect ratio.

So the required evidence points come from ``before``, the expert phrasing that
names the real mechanisms, and are frozen in a file both arms are scored against.

A symbol is required only if it RESOLVES in the corpus -- a file of that name, or
the token appearing in some source file. Prose in the key ("IDENTITY", "writer
version 6") names a concept rather than a thing that can be quoted, and demanding
code evidence for it would fail an honest answer.

    python evaluation/spool-clean-room/build_chain_requirements.py
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CORPUS = REPO / 'spool-corpus'
KEY = HERE / 'questions_debcrumb_25.json'
OUT = HERE / 'chain_requirements.json'

SYM = re.compile(r'\b[A-Z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]{2,}\b')
# All-caps words are SQL keywords or protocol nouns in this key, not symbols.
STOP = {'INTO', 'SELECT', 'MERGE', 'DELETE', 'UPDATE', 'INSERT', 'NULL', 'JSON',
        'IDENTITY', 'DEFAULT', 'CDC', 'DML', 'API', 'SQL', 'URI'}
REPOS = ('spark', 'delta', 'databricks-sdk-py')


def symbols_in_order(text: str) -> list[str]:
    """Resolvable-looking mechanisms in expert-authored causal order."""
    return list(dict.fromkeys(
        s for s in SYM.findall(text) if s.upper() not in STOP))


def file_index() -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for r in REPOS:
        for p in (CORPUS / r).rglob('*'):
            if p.suffix in ('.scala', '.java', '.py') and p.is_file():
                idx.setdefault(p.stem, []).append(r)
    return idx


def token_repos(sym: str) -> list[str]:
    """Which repos contain this token at all (grep -l, bounded)."""
    found = []
    for r in REPOS:
        try:
            res = subprocess.run(
                ['rg', '-lF', '-g', '*.scala', '-g', '*.java', '-g', '*.py',
                 sym, str(CORPUS / r)],
                capture_output=True, text=True, timeout=120)
            if res.stdout.strip():
                found.append(r)
        except (subprocess.SubprocessError, OSError):
            continue
    return found


def main() -> int:
    spec = json.loads(KEY.read_text())
    idx = file_index()
    out = []
    for q in spec:
        # Preserve the expert key's first-appearance order. Alphabetical sorting
        # turns a causal chain into a bag of names and can reward an explanation
        # that never establishes the actual transition order.
        syms = symbols_in_order(q['before'])
        required, unresolved, repos = [], [], set()
        for s in syms:
            in_files = idx.get(s)
            rs = in_files if in_files else token_repos(s)
            if rs:
                required.append({'symbol': s, 'repos': sorted(set(rs)),
                                 'defines_file': bool(in_files)})
                repos |= set(rs)
            else:
                unresolved.append(s)
        out.append({
            'id': q['id'], 'family': q.get('family'), 'flag': bool(q.get('flag')),
            'required': required, 'unresolved': unresolved,
            'repos_spanned': sorted(repos), 'hops': len(required),
        })
        print(f"  id {q['id']:>3}  hops={len(required):<2} repos={','.join(sorted(repos)) or '-':<22}"
              f"{' UNRESOLVED: ' + ','.join(unresolved) if unresolved else ''}")
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    live = [r for r in out if not r['flag']]
    print(f'\n  {len(out)} questions ({len(live)} unflagged) -> {OUT.name}')
    print(f'  hops per question: min {min(r["hops"] for r in live)}  '
          f'max {max(r["hops"] for r in live)}')
    print(f'  cross-repo (>=2): {sum(1 for r in live if len(r["repos_spanned"]) >= 2)}/{len(live)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
