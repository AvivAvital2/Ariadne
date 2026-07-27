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
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ast_grep_py import SgRoot

from docgen.scip_extractor import ScipIndex
from docgen.scip_owning import build_owning_resolver
from ast_utils import safe_ast_parse
from progress_util import iter_with_progress

if TYPE_CHECKING:
    from sqlite3 import Connection


# Extension → handler. ``.ts`` / ``.tsx`` / ``.mjs`` route through the
# ``javascript`` ast-grep grammar (matching ``catalog_extractor.py``'s
# convention) — TypeScript-specific syntax outside our area of interest
# is parsed leniently by tree-sitter-javascript.
# Grammar file-filters come from the one authoritative source in
# docgen/scip_languages.py — no per-extractor list to drift or misroute.
from docgen.scip_languages import (  # noqa: E402
    GO_GRAMMAR_EXTS as _GO_EXTS,
    JS_GRAMMAR_EXTS as _JS_EXTS,
    PY_GRAMMAR_EXTS as _PY_EXTS,
    SCALA_GRAMMAR_EXTS as _SCALA_EXTS,
)


def _detect_lang(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in _PY_EXTS:
        return 'python'
    if ext in _JS_EXTS:
        return 'javascript'
    if ext in _SCALA_EXTS:
        return 'scala'
    if ext in _GO_EXTS:
        return 'go'
    return None
def _docstring_constants(tree: ast.AST) -> set:
    """The string-literal ``Constant`` nodes that are docstrings — the bare
    string first in a module/class/function body. Documentation, never a data
    value, so excluded from the literal index (and never parsed as SQL)."""
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docs.add(body[0].value)
    return docs


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
    docstrings = _docstring_constants(tree)
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
            if node in docstrings:
                continue
            out.append((node.lineno, node.col_offset, node.value))
        for child in ast.iter_child_nodes(node):
            stack.append(child)
    return out


def _extract_python_fstring_sql(text: str) -> list[tuple[int, int, str]]:
    """Reconstruct SQL-shaped f-strings into parseable templates for the raw-SQL
    binder (design: f-string SQL capture).

    Each ``ast.JoinedStr`` becomes its literal parts verbatim. An interpolation
    (``{...}``) that names a module-level ``NAME = '<str>'`` constant is folded to
    that literal — so ``FROM {PRIMARY_TABLE}`` -> ``FROM primary_table``, a real
    table the schema witness can bind; any other interpolation becomes the
    reserved placeholder identifier ``FSTRING_PLACEHOLDER`` (later dropped by the
    binder). Only templates that look like SQL are returned — these rows feed
    only the SQL binder. Returns ``(line_1indexed, col_0indexed, template)``.
    """
    from docgen.sql_access import FSTRING_PLACEHOLDER, is_sql
    try:
        tree = safe_ast_parse(text)
    except SyntaxError:
        return []
    consts: dict[str, str] = {}
    for stmt in tree.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            consts[stmt.targets[0].id] = stmt.value.value

    def _fold(v: ast.AST) -> str:
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value
        if (isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name)
                and v.value.id in consts):
            return consts[v.value.id]
        return FSTRING_PLACEHOLDER

    out: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        template = ''.join(_fold(v) for v in node.values)
        if is_sql(template):
            out.append((node.lineno, node.col_offset, template))
    return out


_SQL_FENCE_RE = re.compile(r"```sql\b[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_SQL_VALUE_RE = re.compile(r':\s*"([^"\\]*(?:\\.[^"\\]*)*)"')


def _extract_embedded_sql(text: str) -> list[tuple[int, int, str]]:
    """Extract SQL embedded inside a larger string literal — markdown ```sql
    fenced blocks and SQL-shaped JSON values (``"...": "SELECT ..."``).

    Many codebases embed example or spec SQL inside prose: prompt templates,
    docstrings, and JSON fixtures. The whole literal is not a SQL statement (it
    begins with prose/JSON), so the standalone SQL extractors skip it; this pulls
    the embedded queries out so the raw-SQL binder can read them. Interpolation
    placeholders (``{...}``) are left intact for the binder to normalize. Returns
    ``(line_1indexed, col_0indexed, sql)`` using the enclosing literal's position.
    """
    from docgen.sql_access import is_sql
    try:
        tree = safe_ast_parse(text)
    except SyntaxError:
        return []
    out: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        value = node.value
        candidates = [m.group(1).strip() for m in _SQL_FENCE_RE.finditer(value)]
        candidates += [
            m.group(1) for m in _JSON_SQL_VALUE_RE.finditer(value)
            if is_sql(m.group(1))
        ]
        for sql in candidates:
            if sql:
                out.append((node.lineno, node.col_offset, sql))
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


def _extract_go_literals(text: str) -> list[tuple[int, int, str]]:
    """Return ``(line_1indexed, col_0indexed, value)`` for every Go string
    literal: ``interpreted_string_literal`` (``"..."``) and
    ``raw_string_literal`` (`` `...` ``). Go has no string interpolation, so
    both forms are plain values."""
    try:
        root = SgRoot(text, 'go').root()
    except Exception:
        return []
    out: list[tuple[int, int, str]] = []
    for node in root.find_all(kind='interpreted_string_literal'):
        value = _strip_outer_quotes(node.text(), single_kinds=('"',))
        if value is None:
            continue
        r = node.range()
        out.append((r.start.line + 1, r.start.column, value))
    for node in root.find_all(kind='raw_string_literal'):
        value = _strip_outer_quotes(node.text(), single_kinds=('`',))
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
    """Return the plain literal value recorded at ``(file, line, col)``, or
    ``None`` if no plain literal is indexed at that exact position.

    ``line`` is 1-indexed; ``col`` is 0-indexed — same convention as
    ``ingest_string_literals`` writes. Restricted to ``kind='plain'`` so the
    route extractors (Phase 8a) and resolution traversal (Phase 2s) never see a
    reconstructed f-string template (those are the raw-SQL binder's, slice 1).
    """
    cursor = conn.execute(
        '''SELECT value FROM string_literals
           WHERE source_name = ? AND file = ?
             AND line_start = ? AND col_start = ?
             AND kind = 'plain'
           LIMIT 1''',
        (source_name, file, line, col),
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

    Plain literals are stored with ``kind='plain'`` (the behaviour every
    route/HTTP/resolution reader relies on). Python SQL-shaped f-strings are
    additionally reconstructed into parseable templates and stored with
    ``kind='fstring'`` — consumed only by the raw-SQL binder (slice 1).

    ``index_factory`` is the test-injection point. Production passes
    None; the function loads ``<source_root>/.ariadne/index.scip``.
    Missing index -> return 0 cleanly (no SCIP, nothing to index).

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
    owning = build_owning_resolver(index)

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
            tagged = [(ln, cs, v, 'plain') for (ln, cs, v) in _extract_python_literals(text)]
            tagged += [(ln, cs, v, 'fstring') for (ln, cs, v) in _extract_python_fstring_sql(text)]
            tagged += [(ln, cs, v, 'embedded') for (ln, cs, v) in _extract_embedded_sql(text)]
        elif lang == 'javascript':
            tagged = [(ln, cs, v, 'plain') for (ln, cs, v) in _extract_javascript_literals(text)]
        elif lang == 'scala':
            tagged = [(ln, cs, v, 'plain') for (ln, cs, v) in _extract_scala_literals(text)]
        elif lang == 'go':
            tagged = [(ln, cs, v, 'plain') for (ln, cs, v) in _extract_go_literals(text)]

        file_str = str(path.resolve())
        for line_start, col_start, value, kind in tagged:
            owner = owning(doc.relative_path, line_start - 1)
            rows.append((
                source_name, file_str, line_start, col_start, value, owner, kind,
            ))

    if rows:
        conn.executemany(
            '''INSERT INTO string_literals
               (source_name, file, line_start, col_start, value,
                owning_symbol_id, kind)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            rows,
        )
    conn.commit()
    return len(rows)


__all__ = ['ingest_string_literals', 'lookup_literal_at_position']
