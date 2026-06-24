"""SCIP-driven string-literal extractor (Phase 2p).

Walks the SCIP index for a source, parses each indexed file with the
language-appropriate AST tool (``ast`` for Python; ``ast-grep`` /
tree-sitter for JS/TS/Scala), and persists every string literal's
``(file, line_start, col_start, value)`` along with its enclosing
function/method ``canonical_id`` (if any) into the ``string_literals``
table.

Architecture note: this is the literal index that route extractors
(Phase 8a) and resolution traversal (Phase 2s) query when they need a
literal value at a known position. Built once per source via this
module; queried many times. Replacing per-extractor literal parsing
with a single index here is the architectural goal called out in
``designs/scip-everywhere-remaining.md``.

Re-ingest semantics: clears prior rows for ``source_name`` before
insert. Other sources' rows are preserved.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ast_grep_py import SgRoot

from docgen.scip_extractor import ScipIndex
from ast_utils import safe_ast_parse
from progress_util import iter_with_progress

if TYPE_CHECKING:
    from sqlite3 import Connection


# Extension → handler. ``.ts`` / ``.tsx`` / ``.mjs`` route through the
# ``javascript`` ast-grep grammar (matching ``catalog_extractor.py``'s
# convention) — TypeScript-specific syntax outside our area of interest
# is parsed leniently by tree-sitter-javascript.
_PY_EXTS: tuple[str, ...] = ('.py',)
_JS_EXTS: tuple[str, ...] = (
    '.js', '.jsx', '.ts', '.tsx', '.mjs',
)
_SCALA_EXTS: tuple[str, ...] = ('.scala', '.sbt')


def _detect_lang(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _PY_EXTS:
        return 'python'
    if ext in _JS_EXTS:
        return 'javascript'
    if ext in _SCALA_EXTS:
        return 'scala'
    return None


def _extract_python_literals(text: str) -> list[tuple[int, int, str]]:
    """Return ``(line_1indexed, col_0indexed, value)`` for every plain
    string literal in a Python source file.

    Skips ``ast.JoinedStr`` (f-strings) ENTIRELY — both the f-string
    itself and any literal ``Constant`` children it contains. Bytes
    literals are excluded by checking ``isinstance(value, str)``.

    Adjacent literals (``'a' 'b'``) are concatenated by Python's parser
    into a single ``ast.Constant`` and emitted as one row.
    """
    try:
        tree = safe_ast_parse(text)
    except SyntaxError:
        return []
    out: list[tuple[int, int, str]] = []
    # Manual stack walk so we can prune entire JoinedStr subtrees.
    # ``ast.walk`` would still descend into them.
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.JoinedStr):
            continue
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.lineno is not None
            and node.col_offset is not None
        ):
            out.append((node.lineno, node.col_offset, node.value))
        for child in ast.iter_child_nodes(node):
            stack.append(child)
    return out


def _strip_outer_quotes(
    text: str, *, single_kinds: tuple[str, ...],
    triple_kinds: tuple[str, ...] = (),
) -> str | None:
    # Try triple-quote forms first so a triple-quoted string isn't
    # mis-stripped as a single quote plus the rest. Returns None if
    # the text isn't properly enclosed by any of the supplied quote
    # kinds.
    for triple in triple_kinds:
        marker = triple * 3
        if (
            len(text) >= 2 * len(marker)
            and text.startswith(marker)
            and text.endswith(marker)
        ):
            return text[len(marker):-len(marker)]
    if len(text) >= 2 and text[0] in single_kinds and text[-1] == text[0]:
        return text[1:-1]
    return None


def _has_ancestor_kind(node, kinds: tuple[str, ...]) -> bool:
    """True if any ancestor of ``node`` has a kind in ``kinds``.
    Used to skip ``string`` nodes that live inside an interpolated
    container (Scala ``interpolated_string_expression`` is the case in
    practice)."""
    parent = node.parent()
    while parent is not None:
        if parent.kind() in kinds:
            return True
        parent = parent.parent()
    return False


def _extract_javascript_literals(
    text: str,
) -> list[tuple[int, int, str]]:
    """Return ``(line_1indexed, col_0indexed, value)`` for every plain
    string and template-literal-without-interpolation in a JS/TS
    source file."""
    try:
        root = SgRoot(text, 'javascript').root()
    except Exception:
        return []
    out: list[tuple[int, int, str]] = []
    for node in root.find_all(kind='string'):
        node_text = node.text()
        value = _strip_outer_quotes(node_text, single_kinds=('"', "'"))
        if value is None:
            continue
        r = node.range()
        out.append((r.start.line + 1, r.start.column, value))
    for node in root.find_all(kind='template_string'):
        node_text = node.text()
        if '${' in node_text:
            continue
        value = _strip_outer_quotes(node_text, single_kinds=('`',))
        if value is None:
            continue
        r = node.range()
        out.append((r.start.line + 1, r.start.column, value))
    return out


def _extract_scala_literals(text: str) -> list[tuple[int, int, str]]:
    """Return ``(line_1indexed, col_0indexed, value)`` for every plain
    string literal in a Scala source file. ``s"..."`` / ``f"..."``
    interpolated forms are different kinds in tree-sitter-scala
    (``interpolated_string_expression``) and are skipped. We also
    defensively skip any ``string`` node whose ancestor is an
    interpolated container, in case a grammar version emits the inner
    text as a ``string`` child."""
    try:
        root = SgRoot(text, 'scala').root()
    except Exception:
        return []
    out: list[tuple[int, int, str]] = []
    interp_kinds = (
        'interpolated_string_expression',
        'interpolated_string',
    )
    for node in root.find_all(kind='string'):
        if _has_ancestor_kind(node, interp_kinds):
            continue
        node_text = node.text()
        value = _strip_outer_quotes(
            node_text,
            single_kinds=('"',),
            triple_kinds=('"',),
        )
        if value is None:
            continue
        r = node.range()
        out.append((r.start.line + 1, r.start.column, value))
    return out


def lookup_literal_at_position(
    conn: 'Connection',
    *,
    source_name: str,
    file: str,
    line: int,
    col: int,
) -> str | None:
    """Return the literal value recorded at ``(file, line, col)``, or
    ``None`` if no literal is indexed at that exact position.

    ``line`` is 1-indexed; ``col`` is 0-indexed — same convention as
    ``ingest_string_literals`` writes. Callers reading positions from
    ``ast-grep`` ranges should add 1 to the line; ``ast`` Constant
    ``lineno`` is already 1-indexed.

    This is the public query API the route extractors (Phase 8a) and
    the resolution traversal (Phase 2s) call when they have a known
    syntactic position and want the literal value without re-parsing
    the source.
    """
    cursor = conn.execute(
        '''SELECT value FROM string_literals
           WHERE source_name = ? AND file = ?
             AND line_start = ? AND col_start = ?
           LIMIT 1''',
        (source_name, file, line, col),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def _find_owning_symbol(
    conn: 'Connection',
    *,
    source_name: str,
    file: str,
    line: int,
) -> str | None:
    """Return ``canonical_id`` of the smallest-range ``Method`` /
    ``Function`` SCIP symbol whose line range covers ``line`` in
    ``file``. ``None`` when the table has no qualifying entry — that's
    a clean signal, not an error.

    ``Class`` / ``Object`` / ``Trait`` / ``Field`` etc. are excluded by
    the ``kind`` filter — the schema's ``owning_symbol_id`` field is
    documented as the enclosing function/method, not the enclosing
    container.
    """
    cursor = conn.execute(
        '''SELECT canonical_id FROM scip_symbols
           WHERE source_name = ? AND file = ?
             AND line_start <= ? AND line_end >= ?
             AND kind IN ('Method', 'Function')
           ORDER BY (line_end - line_start) ASC
           LIMIT 1''',
        (source_name, file, line, line),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def ingest_string_literals(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root``, extract every string
    literal from each indexed file, and persist to ``string_literals``.

    ``index_factory`` is the test-injection point. Production passes
    None; the function loads ``<source_root>/.ariadne/index.scip``.
    Missing index → return 0 cleanly (no SCIP, nothing to index).

    Returns the number of literal rows inserted.
    """
    if index_factory is None:
        scip_path = source_root / '.ariadne' / 'index.scip'
        if not scip_path.exists():
            return 0
        try:
            index = ScipIndex.load(
                scip_path, repo='', max_staleness_days=999,
            )
        except Exception:
            return 0
    else:
        index = index_factory()

    conn.execute(
        'DELETE FROM string_literals WHERE source_name = ?',
        (source_name,),
    )

    rows: list[tuple] = []
    for doc in iter_with_progress(index.documents, progress_callback, source_name):
        path = source_root / doc.relative_path
        lang = _detect_lang(path)
        if lang is None:
            continue
        try:
            text = path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        if lang == 'python':
            literals = _extract_python_literals(text)
        elif lang == 'javascript':
            literals = _extract_javascript_literals(text)
        elif lang == 'scala':
            literals = _extract_scala_literals(text)
        else:
            continue

        file_str = str(path.resolve())
        for line_start, col_start, value in literals:
            owner = _find_owning_symbol(
                conn,
                source_name=source_name,
                file=file_str,
                line=line_start,
            )
            rows.append((
                source_name, file_str, line_start, col_start, value, owner,
            ))

    if rows:
        conn.executemany(
            '''INSERT INTO string_literals
               (source_name, file, line_start, col_start, value,
                owning_symbol_id)
               VALUES (?, ?, ?, ?, ?, ?)''',
            rows,
        )
    conn.commit()
    return len(rows)


__all__ = ['ingest_string_literals', 'lookup_literal_at_position']
