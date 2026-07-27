"""SCIP-driven Go net/http client extractor.

Mirrors the JS/TS (8b.2) and Python (8b.1) client extractors:

- **SCIP** classifies which call sites are Go HTTP-client primitives via
  ``DEFAULT_SINK_REGISTRY`` filtered to ``language='go'`` (suffix match on
  the occurrence symbol). Adding a client library is a registry edit.
- **ast-grep / tree-sitter-go** walks the call STRUCTURE: the
  ``call_expression`` → ``selector_expression`` (``field_identifier`` =
  method name scip-go anchors) and the ``argument_list``; the URL sits at
  the sink's ``arg_index``.
- The URL literal is read DIRECTLY from the AST node
  (``interpreted_string_literal`` / ``raw_string_literal``). Go isn't in
  ``string_literals`` and net/http URLs are literals in practice;
  non-literal (variable) URLs are skipped, as with a template-string URL
  in the JS extractor.

Output: rows in ``http_client_calls`` (``confidence='literal'``). Method
resolution for ``NewRequest`` (its verb is a separate arg) is future work,
so those rows carry ``http_method=NULL`` — the same shape as the JVM
play-ws sink whose verb comes from a chained call.

Re-ingest: clears prior rows for ``source_name``; other sources preserved.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ast_grep_py import SgRoot

from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_go_ast import (
    _GO_EXTS,
    _GO_STRING_KINDS,
    _argument_expressions,
    _argument_list,
    _build_call_index,
    _select_call_with_args,
)
from docgen.scip_owning import build_owning_resolver
from docgen.scip_resolution import resolve_arg_value
from docgen.scip_sink_registry import DEFAULT_SINK_REGISTRY, SinkSpec

if TYPE_CHECKING:
    from sqlite3 import Connection


def _arg_url_info(call_node, arg_index: int) -> tuple[int, int, str | None] | None:
    """``(line_1idx, col_0idx, identifier_name)`` for the URL arg at
    ``arg_index``. String literal → name ``None`` (direct resolution);
    ``identifier`` → its name (variable resolution follows the SCIP symbol
    to its definition); other shapes → ``None`` (caller skips)."""
    exprs = _argument_expressions(_argument_list(call_node))
    if len(exprs) <= arg_index:
        return None
    node = exprs[arg_index]
    kind = node.kind()
    r = node.range()
    line = r.start.line + 1
    col = r.start.column
    if kind in _GO_STRING_KINDS:
        return (line, col, None)
    if kind == 'identifier':
        return (line, col, node.text())
    return None


def _extract_calls_from_doc(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
    owning,
) -> list[tuple]:
    """Return ``http_client_calls`` rows for one Go file. URLs resolve via
    ``resolve_arg_value`` — a direct literal (``'literal'``), or a variable
    followed through its SCIP symbol to a defining literal / config value."""
    classified: list[tuple[_ScipOccurrence, SinkSpec]] = []
    for occ in doc.occurrences:
        spec = DEFAULT_SINK_REGISTRY.matching_symbol(occ.symbol, language='go')
        if spec is not None and spec.kind == 'http_client':
            classified.append((occ, spec))
    if not classified:
        return []

    try:
        root = SgRoot(source_text, 'go').root()
    except Exception:
        return []
    call_index = _build_call_index(root)

    rows: list[tuple] = []
    for occ, spec in classified:
        pos = (occ.range[0], occ.range[1])
        call = _select_call_with_args(call_index.get(pos, []))
        if call is None:
            continue
        info = _arg_url_info(call, spec.arg_index)
        if info is None:
            continue
        arg_line, arg_col, arg_name = info
        url, confidence = resolve_arg_value(
            conn=conn,
            source_name=source_name,
            file=file,
            line=arg_line,
            col=arg_col,
            identifier_name=arg_name,
        )
        if url is None:
            continue
        call_line = pos[0] + 1
        consumer = owning(doc.relative_path, call_line - 1)
        rows.append((
            source_name, consumer, url, spec.http_method, file, call_line,
            spec.name, confidence,
        ))
    return rows


def ingest_go_http_clients(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root``, find Go net/http client
    sinks, read literal URLs, persist to ``http_client_calls``.

    ``index_factory`` is the test-injection point; production passes None
    and reads ``<source_root>/.ariadne/index.scip`` (missing → 0). Returns
    the number of rows inserted.
    """
    if index_factory is None:
        scip_path = source_root / '.ariadne' / 'index.scip'
        if not scip_path.exists():
            return 0
        try:
            index = ScipIndex.load(scip_path, repo='', max_staleness_days=999)
        except Exception:
            return 0
    else:
        index = index_factory()

    conn.execute(
        'DELETE FROM http_client_calls WHERE source_name = ?', (source_name,),
    )

    rows: list[tuple] = []
    owning = build_owning_resolver(index)
    for doc in index.documents:
        go_path = source_root / doc.relative_path
        if go_path.suffix.lower() not in _GO_EXTS:
            continue
        try:
            text = go_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        try:
            file_rows = _extract_calls_from_doc(
                doc, text, conn=conn, source_name=source_name,
                file=str(go_path.resolve()), owning=owning,
            )
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


__all__ = ['ingest_go_http_clients']
