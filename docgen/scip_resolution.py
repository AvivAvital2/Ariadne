"""Phase 2s — sink-site argument resolution.

Library function called by sink-extracting consumers (Phase 2t,
8a, 8b) to upgrade their resolution capability beyond direct
literal lookup. Three branches:

1. **Direct literal** — Phase 2p ``string_literals`` lookup at the
   exact ``(file, line, col)``. Confidence ``'literal'``.
2. **Variable → config getter** (Phase 2s.b) — when the arg is an
   identifier and the unique scip_symbol's def line classifies as
   a getter call (``config.getString("k")`` / ``config["k"]``),
   resolve ``k`` against Phase 2q ``config_values``. Confidence
   ``'config-resolved'``.
3. **Variable → literal** — when the def line classifies as a
   literal RHS (or the inspector can't classify but the def line
   carries exactly one literal), return that literal. Confidence
   ``'resolved-constant'``.

The Phase 2s.b inspector runs only when the def file is readable
on disk and its language is supported (python / javascript /
scala). For unreadable files or unsupported languages the resolver
falls back to the v1 "single literal in def-line range" heuristic
to preserve behavior on languages without a per-language inspector.

What this does NOT do:

- Transitive variable chains (``A = B; B = "x"``).
- Sequence resolution (``A = ["a", "b"]``; subprocess.run(A)).

Resolution priority: direct literal at the call's position wins
over variable resolution. The caller passes both ``(line, col)``
and optional ``identifier_name``; if the position has a literal,
that value is returned without consulting the variable branch.

Source isolation: every query filters by ``source_name``. A var
named ``URL`` in source A is independent of one in source B.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlite3 import Connection


# Languages for which Phase 2s.b can classify the def-line RHS.
# Files in other languages skip the inspector and use the v1
# literal-in-range heuristic.
_INSPECTOR_LANGUAGES: frozenset[str] = frozenset({
    'python', 'javascript', 'scala',
})


def _lookup_literal_at_position(
    conn: 'Connection',
    *,
    source_name: str,
    file: str,
    line: int,
    col: int,
) -> str | None:
    cursor = conn.execute(
        '''SELECT value FROM string_literals
           WHERE source_name = ? AND file = ?
             AND line_start = ? AND col_start = ?
           LIMIT 1''',
        (source_name, file, line, col),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def _find_unique_symbol_def(
    conn: 'Connection',
    *,
    source_name: str,
    identifier_name: str,
) -> tuple[str, int, int, str] | None:
    """Find the unique scip_symbol whose ``qualified_name`` ends in
    ``.<identifier_name>`` (or equals it bare for top-level no-module
    cases). Returns ``(file, line_start, line_end, language)`` if
    exactly one matches; ``None`` for zero or multiple matches.
    """
    cursor = conn.execute(
        '''SELECT file, line_start, line_end, language FROM scip_symbols
           WHERE source_name = ?
             AND (qualified_name = ?
                  OR qualified_name LIKE ?)''',
        (source_name, identifier_name, f'%.{identifier_name}'),
    )
    rows = cursor.fetchall()
    if len(rows) != 1:
        return None
    return rows[0]


def _lookup_config_value(
    conn: 'Connection',
    *,
    source_name: str,
    key: str,
) -> str | None:
    cursor = conn.execute(
        '''SELECT value FROM config_values
           WHERE source_name = ? AND key = ?
           LIMIT 1''',
        (source_name, key),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def _classify_def_rhs(
    *, def_file: str, def_line_start: int, language: str,
):
    """Run the Phase 2s.b inspector on the def line. Returns the
    ``InspectionResult`` or ``None`` when the language is unsupported,
    the file is unreadable, or any error occurs — caller falls back
    to the v1 literal-in-range heuristic in those cases.
    """
    if language not in _INSPECTOR_LANGUAGES:
        return None
    try:
        source_text = Path(def_file).read_text(
            encoding='utf-8', errors='replace',
        )
    except OSError:
        return None
    from docgen.scip_definition_inspector import inspect_definition_rhs
    try:
        return inspect_definition_rhs(
            source_text=source_text,
            line=def_line_start,
            language=language,
        )
    except Exception:
        return None


def _lookup_literals_in_range(
    conn: 'Connection',
    *,
    source_name: str,
    file: str,
    line_start: int,
    line_end: int,
) -> list[str]:
    cursor = conn.execute(
        '''SELECT value FROM string_literals
           WHERE source_name = ? AND file = ?
             AND line_start >= ? AND line_start <= ?''',
        (source_name, file, line_start, line_end),
    )
    return [r[0] for r in cursor.fetchall()]


def resolve_arg_value(
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
    line: int,
    col: int,
    identifier_name: str | None = None,
) -> tuple[str | None, str]:
    """Resolve the value of an arg expression at the given position.

    Returns ``(value, confidence)`` where ``confidence`` is one of:

    - ``'literal'`` — direct hit on string_literals at (line, col)
    - ``'config-resolved'`` — variable whose def is a config-getter
      call; the key resolved against Phase 2q config_values
    - ``'resolved-constant'`` — variable reference resolved through
      a scip_symbol's def line carrying a literal
    - ``'unresolved'`` — no branch produced a value; ``value`` is
      ``None``
    """
    # Branch 1: direct literal at the call's position
    direct = _lookup_literal_at_position(
        conn,
        source_name=source_name,
        file=file,
        line=line,
        col=col,
    )
    if direct is not None:
        return (direct, 'literal')

    # Branch 2: variable reference (only if name is provided)
    if identifier_name is None:
        return (None, 'unresolved')

    sym_def = _find_unique_symbol_def(
        conn,
        source_name=source_name,
        identifier_name=identifier_name,
    )
    if sym_def is None:
        return (None, 'unresolved')

    def_file, def_line_start, def_line_end, def_language = sym_def

    # Phase 2s.b: inspect the def-line RHS to distinguish literal
    # from config-getter from opaque expression. Unsupported language
    # or unreadable file → None, fall through to v1 literal-in-range.
    inspection = _classify_def_rhs(
        def_file=def_file,
        def_line_start=def_line_start,
        language=def_language,
    )
    if inspection is not None:
        if inspection.kind == 'getter_call':
            cv = _lookup_config_value(
                conn,
                source_name=source_name,
                key=inspection.config_key or '',
            )
            if cv is not None:
                return (cv, 'config-resolved')
            return (None, 'unresolved')
        if inspection.kind == 'other':
            # Opaque expression on the def line — don't misattribute
            # an incidental literal hidden inside the call.
            return (None, 'unresolved')
        # kind == 'literal' falls through to literal-in-range below

    literals = _lookup_literals_in_range(
        conn,
        source_name=source_name,
        file=def_file,
        line_start=def_line_start,
        line_end=def_line_end,
    )
    if len(literals) != 1:
        return (None, 'unresolved')

    return (literals[0], 'resolved-constant')


__all__ = ['resolve_arg_value']
