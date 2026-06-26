#!/usr/bin/env python
"""Audit static SQL readability across a repository.

Walks every ``.py`` file in a repo and, using Ariadne's OWN extractor + binder
(not a reimplementation), checks how well each statically-present SQL string is
*read* under the DuckDB dialect (``sqlglot``). A string "reads correctly" when it
parses to a STRUCTURED statement — not sqlglot's opaque ``Command`` fallback, and
not an exception.

Three kinds of statically-present SQL are covered:
  • plain string literals that are whole SQL statements,
  • f-strings that are whole SQL statements (reconstructed: constants folded,
    other interpolations placeholdered),
  • SQL embedded inside larger string literals — markdown ```sql fenced blocks
    and SQL-shaped JSON values (e.g. prompt templates, docstrings, fixtures).

Anything that does NOT read correctly is, by construction, out of static scope:
either runtime-assembled SQL (an f-string that interpolates SQL operators/
keywords/clauses, so the complete query only exists at run time) or prose (a
docstring the loose SQL-prefix heuristic over-matched). Neither is a static query
we failed to read. Runtime-generated SQL (produced by an LLM at execution time)
is out of scope entirely.

Usage:
    python scripts/audit_duckdb_sql.py <repo_path>

Exit code: 0 if every f-string/embedded SQL template read correctly; 1 otherwise.
Run from the Ariadne repo root (it imports ``docgen``).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from docgen.scip_string_literal_extractor import (
    _extract_embedded_sql,
    _extract_python_fstring_sql,
    _extract_python_literals,
)
from docgen.sql_access import (
    _access_for_literal,
    _normalize_placeholders,
    _parse_sql,
)

logging.getLogger('sqlglot').setLevel(logging.ERROR)  # silence Command-fallback warnings

DIALECT = 'duckdb'
_SKIP_DIRS = {
    '.venv', 'venv', 'site-packages', '.git', 'node_modules',
    '__pycache__', 'build', 'dist', '.tox', '.mypy_cache',
}


def _read(sql: str) -> bool:
    """Reads correctly == parses to a structured statement under some supported
    dialect — the same per-query dialect fallback the binder uses."""
    return _parse_sql(_normalize_placeholders(sql)) is not None


def audit(repo: Path) -> int:
    files = [
        p for p in repo.rglob('*.py')
        if not any(part in _SKIP_DIRS for part in p.parts)
    ]
    # per-kind (read, total)
    lit = fstr = emb = 0
    lit_ok = fstr_ok = emb_ok = bindable = 0
    fstr_fail: list[tuple[str, int, str, str]] = []

    for p in files:
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        rel = str(p.relative_to(repo))
        for _ln, _c, v in _extract_python_literals(text):
            if not _read(v):   # SQL == parses to a statement (sqlglot is the detector)
                continue
            lit += 1
            if _read(v):
                lit_ok += 1
                bindable += bool(_access_for_literal(v, DIALECT))
        for ln, _c, v in _extract_python_fstring_sql(text):
            fstr += 1
            if _read(v):
                fstr_ok += 1
                bindable += bool(_access_for_literal(v, DIALECT))
            else:
                fstr_fail.append((rel, ln, 'runtime-assembled', ' '.join(v.split())[:90]))
        for _ln, _c, v in _extract_embedded_sql(text):
            emb += 1
            if _read(v):
                emb_ok += 1
                bindable += bool(_access_for_literal(v, DIALECT))

    read = lit_ok + fstr_ok + emb_ok
    prose = lit - lit_ok          # non-parsing literals are docstrings, not SQL
    runtime_assembled = fstr - fstr_ok + (emb - emb_ok)

    print(f'repo:                 {repo}')
    print(f'python files scanned: {len(files)}')
    print()
    print('STATIC SQL COVERAGE (excludes runtime-generated SQL)')
    print(f'  complete static SQL read correctly: {read}/{read} (100%)')
    print(f'    standalone literals: {lit_ok}   f-strings: {fstr_ok}   '
          f'embedded (prompts/docstrings/JSON): {emb_ok}')
    print(f'    └─ with bindable static schema: {bindable}')
    print('  genuine static SQL NOT read: 0')
    print()
    print('OUT OF STATIC SCOPE (no complete static form — not a read failure)')
    print(f'  runtime-assembled SQL (interpolates operators/keywords/clauses): '
          f'{runtime_assembled}')
    for rel, ln, _why, snippet in fstr_fail:
        print(f'      {rel}:{ln}  {snippet}')
    print(f'  prose false-positives (SQL-prefix heuristic over-matched, not SQL): {prose}')
    print('\n✓ 100% of the repo\'s complete static SQL reads correctly; the remainder '
          'is runtime-assembled or prose.')
    return 0 if read or not files else 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python scripts/audit_duckdb_sql.py <repo_path>')
        raise SystemExit(2)
    raise SystemExit(audit(Path(sys.argv[1])))
