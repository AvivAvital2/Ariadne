"""Phase 2t — multi-language process invocation extractor.

Architecture mirrors Phase 8b (HTTP client extractors) but writes to
``process_invocations`` instead of ``http_client_calls`` and matches
sinks of ``kind='process_invocation'`` instead of ``'http_client'``.

- **SCIP** filters call sites via the Phase 2r registry.
- **Per-language AST tools** (``ast`` for Python, ``ast-grep`` for
  JS/Scala) walk call structure to find argument positions.
- **Phase 2p ``string_literals``** supplies the literal command/script
  value at the first arg's position. List args (Python
  ``subprocess.run(["python", "x"])``) are deferred to Phase 2s.
- **Phase 2d ``scip_symbols``** supplies ``caller_symbol_id``.
  ``process_invocations.caller_symbol_id`` is NOT NULL by schema, so
  module-level subprocess calls are skipped (no caller side to
  attribute the edge to).

Output: rows in ``process_invocations``. ``target_path`` is the
literal command/script string at the call site; ``target_symbol_id``
is reserved for Phase 2t.b fuzzy file-matching (deferred — v1 leaves
it ``NULL``).

Re-ingest semantics: clears prior rows for ``source_name``. Other
sources' rows are preserved.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ast_grep_py import SgRoot

from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_owning import build_owning_resolver
from docgen.scip_sink_registry import DEFAULT_SINK_REGISTRY, SinkSpec
from ast_utils import safe_ast_parse

if TYPE_CHECKING:
    from sqlite3 import Connection


# One authoritative source (docgen/scip_languages.py) — no drift, no misroute.
from docgen.scip_languages import (  # noqa: E402
    JS_GRAMMAR_EXTS as _JS_EXTS,
    PY_GRAMMAR_EXTS as _PY_EXTS,
    SCALA_GRAMMAR_EXTS as _SCALA_EXTS,
)


def _detect_language(path: Path) -> str | None:
    """Map file extension to the SCIP language name (matches the
    sink registry's ``language`` field). ``.py`` → ``python``;
    JS/TS variants → ``typescript``; Scala → ``jvm``."""
    ext = path.suffix.lower()
    if ext in _PY_EXTS:
        return 'python'
    if ext in _JS_EXTS:
        return 'typescript'
    if ext in _SCALA_EXTS:
        return 'jvm'
    return None


def _occ_position(occ: _ScipOccurrence) -> tuple[int, int]:
    return (occ.range[0], occ.range[1])


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _python_callee_position(call: ast.Call) -> tuple[int, int] | None:
    """For ``obj.method(args)`` (``ast.Attribute``), the position of
    the ``method`` token. For bare ``func(args)`` (``ast.Name``), the
    name's start. Same as Phase 8b's Python position finder."""
    func = call.func
    if isinstance(func, ast.Attribute):
        if (
            func.end_lineno is not None
            and func.end_col_offset is not None
        ):
            return (
                func.end_lineno - 1,
                func.end_col_offset - len(func.attr),
            )
        return None
    if isinstance(func, ast.Name):
        if func.lineno is not None and func.col_offset is not None:
            return (func.lineno - 1, func.col_offset)
        return None
    return None


def _python_arg_info(
    call: ast.Call, arg_index: int,
) -> tuple[int, int, str | None] | None:
    """Return ``(line_1idx, col_0idx, identifier_name)`` for the N-th
    positional arg.

    - String literal → ``(line, col, None)`` — Phase 2s direct branch
      will resolve via string_literals.
    - Identifier (``ast.Name``) → ``(line, col, name)`` — Phase 2s
      variable branch follows the symbol to its definition.
    - Anything else (list, expression, f-string, attribute access) →
      ``None``; the caller skips.
    """
    if arg_index >= len(call.args):
        return None
    arg = call.args[arg_index]
    if (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, str)
        and arg.lineno is not None
        and arg.col_offset is not None
    ):
        return (arg.lineno, arg.col_offset, None)
    if (
        isinstance(arg, ast.Name)
        and arg.lineno is not None
        and arg.col_offset is not None
    ):
        return (arg.lineno, arg.col_offset, arg.id)
    return None


def _extract_python_invocations(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
owning) -> list[tuple]:
    from docgen.scip_resolution import resolve_arg_value

    classified: list[tuple[_ScipOccurrence, SinkSpec]] = []
    for occ in doc.occurrences:
        spec = DEFAULT_SINK_REGISTRY.matching_symbol(
            occ.symbol, language='python',
        )
        if spec is not None and spec.kind == 'process_invocation':
            classified.append((occ, spec))
    if not classified:
        return []

    try:
        tree = safe_ast_parse(source_text)
    except SyntaxError:
        return []

    call_index: dict[tuple[int, int], list[ast.Call]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            pos = _python_callee_position(node)
            if pos is not None:
                call_index.setdefault(pos, []).append(node)

    rows: list[tuple] = []
    for occ, spec in classified:
        pos = _occ_position(occ)
        calls = call_index.get(pos, [])
        if not calls:
            continue
        call = calls[0]
        arg_info = _python_arg_info(call, spec.arg_index)
        if arg_info is None:
            continue
        arg_line, arg_col, arg_name = arg_info
        target, confidence = resolve_arg_value(
            conn=conn,
            source_name=source_name,
            file=file,
            line=arg_line,
            col=arg_col,
            identifier_name=arg_name,
        )
        if target is None:
            continue
        call_line = pos[0] + 1
        caller = owning(doc.relative_path, call_line - 1)
        if caller is None:
            # Schema NOT NULL — skip module-level calls.
            continue
        rows.append((
            source_name,
            caller,
            target,
            None,  # target_symbol_id deferred to Phase 2t.b
            confidence,
            file,
            call_line,
            call_line,
        ))
    return rows


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------


def _node_start(node) -> tuple[int, int]:
    r = node.range()
    return (r.start.line, r.start.column)


def _js_callee_position(call_node) -> tuple[int, int] | None:
    """For ``obj.method(args)`` (member_expression callee), position
    of the rightmost ``property_identifier``. For bare identifier
    callee, the identifier's start position."""
    children = list(call_node.children())
    if not children:
        return None
    func = children[0]
    kind = func.kind()
    if kind == 'member_expression':
        for c in reversed(list(func.children())):
            if c.kind() == 'property_identifier':
                return _node_start(c)
        return None
    if kind == 'identifier':
        return _node_start(func)
    return None


def _js_arguments_node(call_node):
    for c in call_node.children():
        if c.kind() == 'arguments':
            return c
    return None


def _js_argument_expressions(args_node) -> list:
    return [
        c for c in args_node.children()
        if c.kind() not in ('(', ')', ',')
    ]


def _js_arg_info(
    call, arg_index: int,
) -> tuple[int, int, str | None] | None:
    """Return ``(line_1idx, col_0idx, identifier_name)`` for the
    N-th positional arg, or ``None`` for non-resolvable shapes.

    String / template_string → literal-position tuple with ``None``
    name. Bare identifier → position + ``.text()`` name. Anything
    else (object literal, array, expression, interpolated template)
    skips.
    """
    args = _js_arguments_node(call)
    if args is None:
        return None
    exprs = _js_argument_expressions(args)
    if arg_index >= len(exprs):
        return None
    arg = exprs[arg_index]
    kind = arg.kind()
    r = arg.range()
    line = r.start.line + 1
    col = r.start.column
    if kind in ('string', 'template_string'):
        return (line, col, None)
    if kind == 'identifier':
        return (line, col, arg.text())
    return None


def _extract_js_invocations(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
owning) -> list[tuple]:
    from docgen.scip_resolution import resolve_arg_value

    classified: list[tuple[_ScipOccurrence, SinkSpec]] = []
    for occ in doc.occurrences:
        spec = DEFAULT_SINK_REGISTRY.matching_symbol(
            occ.symbol, language='typescript',
        )
        if spec is not None and spec.kind == 'process_invocation':
            classified.append((occ, spec))
    if not classified:
        return []

    try:
        root = SgRoot(source_text, 'javascript').root()
    except Exception:
        return []

    call_index: dict[tuple[int, int], list] = {}
    for call in root.find_all(kind='call_expression'):
        pos = _js_callee_position(call)
        if pos is not None:
            call_index.setdefault(pos, []).append(call)

    rows: list[tuple] = []
    for occ, spec in classified:
        pos = _occ_position(occ)
        calls = call_index.get(pos, [])
        if not calls:
            continue
        call = next(
            (c for c in calls if _js_arguments_node(c) is not None),
            None,
        )
        if call is None:
            continue
        arg_info = _js_arg_info(call, spec.arg_index)
        if arg_info is None:
            continue
        arg_line, arg_col, arg_name = arg_info
        target, confidence = resolve_arg_value(
            conn=conn,
            source_name=source_name,
            file=file,
            line=arg_line,
            col=arg_col,
            identifier_name=arg_name,
        )
        if target is None:
            continue
        call_line = pos[0] + 1
        caller = owning(doc.relative_path, call_line - 1)
        if caller is None:
            continue
        rows.append((
            source_name,
            caller,
            target,
            None,
            confidence,
            file,
            call_line,
            call_line,
        ))
    return rows


# ---------------------------------------------------------------------------
# Scala (JVM)
# ---------------------------------------------------------------------------


def _scala_callee_identifier_position(
    call_node,
) -> tuple[int, int] | None:
    """Walk down a Scala ``call_expression`` to find its callee
    identifier's position. Same logic as the Akka HTTP and Scala
    HTTP client extractors — handles dotted-call (field_expression)
    and chained calls."""
    current = call_node
    for _ in range(8):
        kind = current.kind()
        if kind in ('identifier', 'simple_identifier', 'name'):
            return _node_start(current)
        if kind in ('field_expression', 'select_expression'):
            sub = list(current.children())
            for c in reversed(sub):
                if c.kind() in (
                    'identifier', 'simple_identifier', 'name',
                ):
                    return _node_start(c)
            return _node_start(current)
        if kind == 'call_expression':
            children = list(current.children())
            if not children:
                return None
            current = children[0]
            continue
        return _node_start(current)
    return None


def _scala_direct_args(call_node):
    for c in call_node.children():
        if c.kind() in (
            'arguments', 'arguments_list', 'argument_list',
        ):
            return c
    return None


def _scala_arg_expressions(args_node) -> list:
    return [
        c for c in args_node.children()
        if c.kind() not in ('(', ')', ',')
    ]


def _scala_arg_info(
    call, arg_index: int,
) -> tuple[int, int, str | None] | None:
    """Return ``(line_1idx, col_0idx, identifier_name)`` for the
    N-th positional arg.

    String literal → ``(line, col, None)``. Bare identifier
    (``simple_identifier`` / ``identifier``) → ``(line, col,
    name)``. Interpolated string / array / other shapes → ``None``.
    """
    args = _scala_direct_args(call)
    if args is None:
        return None
    exprs = _scala_arg_expressions(args)
    if arg_index >= len(exprs):
        return None
    arg = exprs[arg_index]
    kind = arg.kind()
    r = arg.range()
    line = r.start.line + 1
    col = r.start.column
    if kind in ('string', 'string_literal'):
        return (line, col, None)
    if kind in ('identifier', 'simple_identifier', 'name'):
        return (line, col, arg.text())
    return None


def _extract_scala_invocations(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
owning) -> list[tuple]:
    from docgen.scip_resolution import resolve_arg_value

    classified: list[tuple[_ScipOccurrence, SinkSpec]] = []
    for occ in doc.occurrences:
        spec = DEFAULT_SINK_REGISTRY.matching_symbol(
            occ.symbol, language='jvm',
        )
        if spec is not None and spec.kind == 'process_invocation':
            classified.append((occ, spec))
    if not classified:
        return []

    try:
        root = SgRoot(source_text, 'scala').root()
    except Exception:
        return []

    call_index: dict[tuple[int, int], list] = {}
    for call in root.find_all(kind='call_expression'):
        pos = _scala_callee_identifier_position(call)
        if pos is not None:
            call_index.setdefault(pos, []).append(call)

    rows: list[tuple] = []
    for occ, spec in classified:
        pos = _occ_position(occ)
        calls = call_index.get(pos, [])
        if not calls:
            continue
        call = next(
            (c for c in calls if _scala_direct_args(c) is not None),
            None,
        )
        if call is None:
            continue
        arg_info = _scala_arg_info(call, spec.arg_index)
        if arg_info is None:
            continue
        arg_line, arg_col, arg_name = arg_info
        target, confidence = resolve_arg_value(
            conn=conn,
            source_name=source_name,
            file=file,
            line=arg_line,
            col=arg_col,
            identifier_name=arg_name,
        )
        if target is None:
            continue
        call_line = pos[0] + 1
        caller = owning(doc.relative_path, call_line - 1)
        if caller is None:
            continue
        rows.append((
            source_name,
            caller,
            target,
            None,
            confidence,
            file,
            call_line,
            call_line,
        ))
    return rows


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def ingest_process_invocations(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root``, find process-invocation
    sinks across Python / JS / Scala, resolve target paths from
    ``string_literals``, persist to ``process_invocations``.

    ``index_factory`` is the test-injection point. Production passes
    None; the function loads ``<source_root>/.ariadne/index.scip``.
    Missing index → return 0 cleanly.

    Re-ingest semantics: clears prior rows for ``source_name``.
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
        'DELETE FROM process_invocations WHERE source_name = ?',
        (source_name,),
    )

    rows: list[tuple] = []
    owning = build_owning_resolver(index)
    for doc in index.documents:
        path = source_root / doc.relative_path
        lang = _detect_language(path)
        if lang is None:
            continue
        try:
            text = path.read_text(
                encoding='utf-8', errors='replace',
            )
        except OSError:
            continue
        try:
            if lang == 'python':
                file_rows = _extract_python_invocations(
                    doc, text, conn=conn, source_name=source_name, file=str(path.resolve()), owning = owning)
            elif lang == 'typescript':
                file_rows = _extract_js_invocations(
                    doc, text, conn=conn, source_name=source_name, file=str(path.resolve()), owning = owning)
            elif lang == 'jvm':
                file_rows = _extract_scala_invocations(
                    doc, text, conn=conn, source_name=source_name, file=str(path.resolve()), owning = owning)
            else:
                continue
        except Exception:
            continue
        rows.extend(file_rows)

    if rows:
        conn.executemany(
            '''INSERT INTO process_invocations
               (source_name, caller_symbol_id, target_path,
                target_symbol_id, confidence, file,
                line_start, line_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            rows,
        )
    conn.commit()
    return len(rows)


__all__ = ['ingest_process_invocations']
