"""SCIP-driven JS/TS HTTP client extractor (Phase 8b.2).

Architecture mirrors Phase 8b.1 (Python clients) and Phase 8a.3
(Express routes):

- **SCIP** filters which call sites are real JS/TS HTTP-client
  primitives via the Phase 2r ``DEFAULT_SINK_REGISTRY`` filtered to
  ``language='typescript'``. Adding a new client library is a
  registry edit, not a new code branch here.
- **ast-grep / tree-sitter-javascript** walks the call STRUCTURE:
  ``call_expression`` and its first ``arguments`` child. Same
  ``'javascript'`` grammar handles ``.js``/``.jsx``/``.ts``/``.tsx``/
  ``.mjs``/``.cjs`` (TypeScript-specific syntax outside what we
  inspect — generic type args, type annotations — is handled
  permissively by the grammar's error tolerance).
- **Phase 2p ``string_literals``** supplies the URL value at the
  first arg's position. Template literals with ``${}`` interpolation
  aren't indexed by Phase 2p, so the lookup naturally returns
  ``None`` for those.
- **Phase 2d ``scip_symbols``** supplies ``consumer_symbol_id`` —
  the enclosing function/method.

Output: rows in ``http_client_calls``. Phase 8c will read these and
match URLs against ``api_endpoints.path_template``.

Re-ingest semantics: clears prior rows for ``source_name``. Other
sources' rows are preserved.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ast_grep_py import SgRoot

from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_owning import build_owning_resolver
from docgen.scip_sink_registry import DEFAULT_SINK_REGISTRY, SinkSpec

if TYPE_CHECKING:
    from sqlite3 import Connection


# Extensions scip-typescript indexes; matches the LANGUAGES registry's
# typescript entry.
# One authoritative source (docgen/scip_languages.py) — no drift, no misroute.
from docgen.scip_languages import JS_GRAMMAR_EXTS as _JS_EXTS  # noqa: E402


def _occ_position(occ: _ScipOccurrence) -> tuple[int, int]:
    return (occ.range[0], occ.range[1])


def _node_start_position(node) -> tuple[int, int]:
    r = node.range()
    return (r.start.line, r.start.column)


def _callee_position(call_node) -> tuple[int, int] | None:
    """For ``obj.method(args)`` (member_expression callee), return the
    position of the rightmost ``property_identifier``. For bare
    ``func(args)`` (identifier callee), return the identifier's start.
    Both correspond to where scip-typescript anchors the symbol
    occurrence."""
    children = list(call_node.children())
    if not children:
        return None
    func = children[0]
    kind = func.kind()
    if kind == 'member_expression':
        for c in reversed(list(func.children())):
            if c.kind() == 'property_identifier':
                return _node_start_position(c)
        return None
    if kind == 'identifier':
        return _node_start_position(func)
    return None


def _arguments_node(call_node):
    for c in call_node.children():
        if c.kind() == 'arguments':
            return c
    return None


def _argument_expressions(args_node) -> list:
    return [
        c for c in args_node.children()
        if c.kind() not in ('(', ')', ',')
    ]


def _build_call_index(root) -> dict[tuple[int, int], list]:
    out: dict[tuple[int, int], list] = {}
    for call in root.find_all(kind='call_expression'):
        pos = _callee_position(call)
        if pos is not None:
            out.setdefault(pos, []).append(call)
    return out


def _select_call_with_args(calls: list):
    for call in calls:
        if _arguments_node(call) is not None:
            return call
    return None


def _url_arg_info(call_node) -> tuple[int, int, str | None] | None:
    """Return ``(line_1idx, col_0idx, identifier_name)`` for the
    first arg.

    ``string`` / ``template_string`` → ``(line, col, None)`` —
    Phase 2s direct branch resolves via string_literals.
    ``identifier`` → ``(line, col, name)`` — Phase 2s variable
    branch follows the symbol to its definition.
    Other shapes → ``None`` (caller skips).
    """
    args = _arguments_node(call_node)
    if args is None:
        return None
    exprs = _argument_expressions(args)
    if not exprs:
        return None
    first = exprs[0]
    kind = first.kind()
    r = first.range()
    line = r.start.line + 1
    col = r.start.column
    if kind in ('string', 'template_string'):
        return (line, col, None)
    if kind == 'identifier':
        return (line, col, first.text())
    return None


def _extract_calls_from_doc(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
owning) -> list[tuple]:
    """For one JS/TS file, return tuples ready to insert into
    ``http_client_calls``."""
    from docgen.scip_resolution import resolve_arg_value

    classified: list[tuple[_ScipOccurrence, SinkSpec]] = []
    for occ in doc.occurrences:
        spec = DEFAULT_SINK_REGISTRY.matching_symbol(
            occ.symbol, language='typescript',
        )
        if spec is not None and spec.kind == 'http_client':
            classified.append((occ, spec))
    if not classified:
        return []

    try:
        root = SgRoot(source_text, 'javascript').root()
    except Exception:
        return []
    call_index = _build_call_index(root)

    rows: list[tuple] = []
    for occ, spec in classified:
        pos = _occ_position(occ)
        calls = call_index.get(pos, [])
        if not calls:
            continue
        call = _select_call_with_args(calls)
        if call is None:
            continue
        info = _url_arg_info(call)
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


def ingest_js_http_clients(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root``, find JS/TS HTTP-client
    sinks, resolve URLs from ``string_literals``, persist to
    ``http_client_calls``.

    ``index_factory`` is the test-injection point. Production passes
    None; the function loads ``<source_root>/.ariadne/index.scip``.
    Missing index → return 0 cleanly.

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
        js_path = source_root / doc.relative_path
        if js_path.suffix.lower() not in _JS_EXTS:
            continue
        try:
            text = js_path.read_text(
                encoding='utf-8', errors='replace',
            )
        except OSError:
            continue
        try:
            file_rows = _extract_calls_from_doc(
                doc, text, conn=conn, source_name=source_name, file=str(js_path.resolve()), owning = owning)
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


__all__ = ['ingest_js_http_clients']
