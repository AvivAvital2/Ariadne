"""SCIP-driven Python HTTP client extractor (Phase 8b.1).

Architecture mirrors Phase 8a refactored route extractors:

- **SCIP** filters which call sites are real Python HTTP-client
  primitives. The match goes through Phase 2r's
  ``DEFAULT_SINK_REGISTRY`` filtered to ``language='python'`` —
  adding a new client library is a registry edit, not a new code
  branch here.
- **ast.parse** walks the call STRUCTURE: which ``ast.Call`` sits at
  each SCIP-matched callee position; what's the position of the URL
  argument expression.
- **Phase 2p ``string_literals``** supplies the literal URL value.
  All quote handling, f-string exclusion, and escape unfolding live
  in the Phase 2p pre-pass; this module just queries by position.
- **Phase 2d ``scip_symbols``** supplies the consumer symbol — the
  enclosing function/method whose body contains the call. ``NULL``
  when the call sits at module level.

Output: rows in ``http_client_calls``. Phase 8c will read these and
match URLs against ``api_endpoints.path_template`` to populate the
resolved ``api_calls``.

Precondition: ``ingest_string_literals`` must have run for the same
``(source_name, source_root)`` first. Tests drive both via a single
``_run_pipeline`` helper.

Re-ingest: clears prior rows for ``source_name``. Other sources'
rows are preserved.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_owning import build_owning_resolver
from docgen.scip_sink_registry import (
    DEFAULT_SINK_REGISTRY,
    SinkSpec,
)
from ast_utils import safe_ast_parse

if TYPE_CHECKING:
    from sqlite3 import Connection


def _occ_position(occ: _ScipOccurrence) -> tuple[int, int]:
    """SCIP occurrence's (line_0idx, col_0idx) start position."""
    return (occ.range[0], occ.range[1])


def _callee_position(call: ast.Call) -> tuple[int, int] | None:
    """Return the (line_0idx, col_0idx) position of the call's callee
    identifier — the position scip-python records for the symbol.

    For ``obj.method(...)`` (``ast.Attribute``), that's the position
    of the ``method`` name token. For bare ``func(...)`` (``ast.Name``),
    it's the start of the name.
    """
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


def _build_call_index(
    tree: ast.AST,
) -> dict[tuple[int, int], list[ast.Call]]:
    out: dict[tuple[int, int], list[ast.Call]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            pos = _callee_position(node)
            if pos is not None:
                out.setdefault(pos, []).append(node)
    return out


def _classify_node(
    node,
) -> tuple[int, int, str | None] | None:
    """Return ``(line_1idx, col_0idx, identifier_name)`` for a node
    that's either a string literal or a bare identifier; ``None``
    for anything else (f-strings, expressions, list literals)."""
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.lineno is not None
        and node.col_offset is not None
    ):
        return (node.lineno, node.col_offset, None)
    if (
        isinstance(node, ast.Name)
        and node.lineno is not None
        and node.col_offset is not None
    ):
        return (node.lineno, node.col_offset, node.id)
    return None


def _arg_info(
    call: ast.Call, arg_index: int,
) -> tuple[int, int, str | None] | None:
    """Find the URL arg's position + identifier name (if applicable).

    Tries positional ``arg_index`` first, then ``url=`` kwarg.
    Accepts string literals (returns ``identifier_name=None``) and
    bare identifiers (returns ``identifier_name=<name>``). Phase 2s
    resolves either form via ``resolve_arg_value``.
    """
    if arg_index < len(call.args):
        info = _classify_node(call.args[arg_index])
        if info is not None:
            return info
    for kw in call.keywords:
        if kw.arg != 'url':
            continue
        info = _classify_node(kw.value)
        if info is not None:
            return info
    return None


def _extract_calls_from_doc(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
owning) -> list[tuple]:
    """For one Python file, return tuples ready to insert into
    ``http_client_calls``."""
    from docgen.scip_resolution import resolve_arg_value

    classified: list[tuple[_ScipOccurrence, SinkSpec]] = []
    for occ in doc.occurrences:
        spec = DEFAULT_SINK_REGISTRY.matching_symbol(
            occ.symbol, language='python',
        )
        if spec is not None and spec.kind == 'http_client':
            classified.append((occ, spec))
    if not classified:
        return []

    try:
        tree = safe_ast_parse(source_text)
    except SyntaxError:
        return []

    call_index = _build_call_index(tree)
    rows: list[tuple] = []
    for occ, spec in classified:
        pos = _occ_position(occ)
        calls = call_index.get(pos, [])
        if not calls:
            continue
        call = calls[0]
        info = _arg_info(call, spec.arg_index)
        if info is None:
            continue
        arg_line, arg_col, arg_name = info
        url_value, confidence = resolve_arg_value(
            conn=conn,
            source_name=source_name,
            file=file,
            line=arg_line,
            col=arg_col,
            identifier_name=arg_name,
        )
        if url_value is None:
            continue
        call_line = pos[0] + 1
        consumer = owning(doc.relative_path, call_line - 1)
        rows.append((
            source_name,
            consumer,
            url_value,
            spec.http_method,
            file,
            call_line,
            spec.name,
            confidence,
        ))
    return rows


def ingest_python_http_clients(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root``, find Python HTTP-client
    sinks, resolve URLs from ``string_literals``, persist to
    ``http_client_calls``.

    ``index_factory`` is the test-injection point. Production passes
    None; the function loads ``<source_root>/.ariadne/index.scip``.
    Missing index → return 0 cleanly.

    Re-ingest semantics: clears prior rows for ``source_name``.
    Other sources' rows are preserved.

    Returns the number of rows inserted.
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
        'DELETE FROM http_client_calls WHERE source_name = ?',
        (source_name,),
    )

    rows: list[tuple] = []
    owning = build_owning_resolver(index)
    for doc in index.documents:
        py_path = source_root / doc.relative_path
        if py_path.suffix.lower() != '.py':
            continue
        try:
            text = py_path.read_text(
                encoding='utf-8', errors='replace',
            )
        except OSError:
            continue
        try:
            file_rows = _extract_calls_from_doc(
                doc, text, conn=conn, source_name=source_name, file=str(py_path.resolve()), owning = owning)
        except Exception:
            continue
        rows.extend(file_rows)

    if rows:
        conn.executemany(
            '''INSERT INTO http_client_calls
               (source_name, consumer_symbol_id, raw_url, http_method,
                call_site_file, call_site_line, sink_name, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            rows,
        )
    conn.commit()
    return len(rows)


__all__ = ['ingest_python_http_clients']
