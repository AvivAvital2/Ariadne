"""String-literal index (Phase 2p / Layer C).

Pre-extracts every string literal in indexed source files and persists
it with the enclosing symbol's canonical_id (where one applies). Layer
C's resolution traversal (Phase 2s) uses this to walk back from a sink
call site to candidate literal arguments without re-parsing source.

MVP scope (this slice): Python extractor only. JVM and TypeScript
extractors come in Phase 2p.b — same :class:`StringLiteral` value
class, same persistence, just different per-language extraction logic.

Re-ingest semantics: :func:`persist_string_literals` clears existing
rows for ``source_name`` before inserting, mirroring Phase 2q
config_index and Phase 7c swagger_ingest.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen

if TYPE_CHECKING:
    from sqlite3 import Connection


@frozen
class StringLiteral:
    """One string literal pre-extracted from indexed source.

    ``owning_symbol_id`` is the canonical_id of the innermost SCIP
    symbol whose body encloses the literal's position, or ``None`` for
    module-scope literals.
    """
    file: Path
    line_start: int       # 1-indexed
    col_start: int        # 0-indexed (matches ast.col_offset)
    value: str
    owning_symbol_id: str | None


@frozen
class SymbolRange:
    """A SCIP symbol's line range, used to attribute literals to their
    enclosing scope. Tests construct these directly; production
    callers derive them from ``CrossSourceGraph.symbols_in(source)``.
    """
    canonical_id: str
    line_start: int  # 1-indexed
    line_end: int    # 1-indexed, inclusive


# ---------------------------------------------------------------------------
# Python extractor
# ---------------------------------------------------------------------------


def _find_owning_symbol(
    line: int,
    symbols: list[SymbolRange],
) -> str | None:
    """Return the canonical_id of the innermost symbol whose [start,
    end] range encloses ``line``. None if no symbol encloses.

    "Innermost" = narrowest range (smallest line_end - line_start).
    """
    enclosing = [
        s for s in symbols
        if s.line_start <= line <= s.line_end
    ]
    if not enclosing:
        return None
    enclosing.sort(key=lambda s: s.line_end - s.line_start)
    return enclosing[0].canonical_id


def _walk_collecting(
    node: ast.AST,
    out: list[ast.AST],
) -> None:
    """Walk the AST collecting string-literal-bearing nodes.

    On hitting a candidate (``ast.Constant(str)`` or ``ast.JoinedStr``
    with all-constant parts), append it and DO NOT recurse further —
    that prevents the inner ``Constant`` of a JoinedStr from being
    counted twice (once as the parent f-string, once as itself).

    Bytes literals and dynamic f-strings are dropped here; they're
    not useful for path/URL resolution downstream.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        out.append(node)
        return  # Don't recurse — Constants are leaves.
    if isinstance(node, ast.JoinedStr):
        # All-constant f-string → effectively a string literal
        if all(
            isinstance(v, ast.Constant) and isinstance(v.value, str)
            for v in node.values
        ):
            out.append(node)
        # Either way, don't recurse into JoinedStr children — we either
        # collected the JoinedStr as a whole, or we're skipping a
        # dynamic f-string and don't want its constant parts mis-
        # attributed as standalone literals.
        return
    for child in ast.iter_child_nodes(node):
        _walk_collecting(child, out)


def extract_python_literals(
    source_file: Path,
    *,
    symbols: list[SymbolRange],
) -> list[StringLiteral]:
    """Walk a Python file's AST and emit one StringLiteral per
    string-literal expression. Each literal is attributed to its
    innermost enclosing symbol via :func:`_find_owning_symbol`.

    Parse errors → empty list. Bytes literals and dynamic f-strings
    are skipped.
    """
    try:
        text = source_file.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    try:
        tree = ast.parse(text, filename=str(source_file))
    except SyntaxError:
        return []

    nodes: list[ast.AST] = []
    _walk_collecting(tree, nodes)

    literals: list[StringLiteral] = []
    for node in nodes:
        if isinstance(node, ast.Constant):
            value = node.value  # already validated as str by walker
            line = node.lineno
            col = node.col_offset
        elif isinstance(node, ast.JoinedStr):
            # All-constant f-string — concatenate parts
            value = ''.join(
                v.value for v in node.values
                if isinstance(v, ast.Constant)
            )
            line = node.lineno
            col = node.col_offset
        else:
            continue  # walker shouldn't hand us anything else
        owning = _find_owning_symbol(line, symbols)
        literals.append(StringLiteral(
            file=source_file,
            line_start=line,
            col_start=col,
            value=value,
            owning_symbol_id=owning,
        ))

    return literals


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_string_literals(
    *,
    source_name: str,
    literals: list[StringLiteral],
    conn: 'Connection',
) -> int:
    """Persist ``literals`` to ariadne.db, replacing the source's
    prior rows.

    Returns the number of rows actually inserted. Empty input clears
    the source's rows.
    """
    conn.execute(
        'DELETE FROM string_literals WHERE source_name = ?',
        (source_name,),
    )
    if not literals:
        conn.commit()
        return 0
    rows = [
        (
            source_name,
            str(l.file),
            l.line_start,
            l.col_start,
            l.value,
            l.owning_symbol_id,
        )
        for l in literals
    ]
    conn.executemany(
        'INSERT INTO string_literals '
        '(source_name, file, line_start, col_start, value, '
        'owning_symbol_id) '
        'VALUES (?, ?, ?, ?, ?, ?)',
        rows,
    )
    conn.commit()
    return len(rows)


def query_string_literals_by_symbol(
    *,
    source_name: str,
    owning_symbol_id: str,
    conn: 'Connection',
) -> list[StringLiteral]:
    """Return all StringLiterals enclosed by ``owning_symbol_id``
    within ``source_name``. Empty list if none match."""
    cur = conn.execute(
        'SELECT file, line_start, col_start, value, owning_symbol_id '
        'FROM string_literals '
        'WHERE source_name = ? AND owning_symbol_id = ?',
        (source_name, owning_symbol_id),
    )
    return [_row_to_literal(row) for row in cur.fetchall()]


def query_string_literals_in_file(
    *,
    source_name: str,
    file: Path,
    conn: 'Connection',
) -> list[StringLiteral]:
    """Return all StringLiterals located in ``file`` (any owning
    symbol or none)."""
    cur = conn.execute(
        'SELECT file, line_start, col_start, value, owning_symbol_id '
        'FROM string_literals '
        'WHERE source_name = ? AND file = ?',
        (source_name, str(file)),
    )
    return [_row_to_literal(row) for row in cur.fetchall()]


def _row_to_literal(row) -> StringLiteral:
    return StringLiteral(
        file=Path(row[0]),
        line_start=row[1],
        col_start=row[2],
        value=row[3],
        owning_symbol_id=row[4],
    )


__all__ = [
    'StringLiteral',
    'SymbolRange',
    'extract_python_literals',
    'persist_string_literals',
    'query_string_literals_by_symbol',
    'query_string_literals_in_file',
]
