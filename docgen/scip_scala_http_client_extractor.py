"""SCIP-driven Scala HTTP client extractor (Phase 8b.3).

Architecture mirrors Phase 8b.1 / 8b.2 and reuses the tree-sitter-scala
call-walking idioms from Phase 8a.1 (Akka):

- **SCIP** filters call sites via the Phase 2r ``DEFAULT_SINK_REGISTRY``
  with ``language='jvm'``. v1 covers ``play.WSClient.url()``; sttp /
  Akka HTTP client / OkHttp builder require Phase 2s chained-call
  walking and are deferred.
- **ast-grep / tree-sitter-scala** walks the call STRUCTURE: the
  ``call_expression`` and its ``arguments`` child. Same grammar Phase
  8a.1 uses for Akka routes.
- **Phase 2p ``string_literals``** supplies the URL value at the
  first arg's position. ``s"..."`` / ``f"..."`` interpolated forms
  aren't indexed by Phase 2p, so the lookup naturally returns
  ``None`` for those.
- **Phase 2d ``scip_symbols``** supplies ``consumer_symbol_id`` —
  the enclosing ``def`` (kind ``Method``).

Output: rows in ``http_client_calls``. ``http_method`` is ``None`` for
play-ws (the verb comes from a chained call we don't track in v1).
Phase 8c reads these and matches URLs against
``api_endpoints.path_template``; ``http_method=None`` rows match any
verb at endpoint resolution time.

Scope: ``.scala`` and ``.sbt`` files only. ``.java`` / ``.kt``
sources share the ``'jvm'`` language tag in the registry but use
different sinks (HttpClient, OkHttp, etc.) and a different grammar;
those go in a future ``scip_java_http_client_extractor``.
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


# One authoritative source (docgen/scip_languages.py) — no drift, no misroute.
from docgen.scip_languages import SCALA_GRAMMAR_EXTS as _SCALA_EXTS  # noqa: E402


def _occ_position(occ: _ScipOccurrence) -> tuple[int, int]:
    """SCIP occurrence's (line_0idx, col_0idx) start position."""
    return (occ.range[0], occ.range[1])


def _node_start_position(node) -> tuple[int, int]:
    r = node.range()
    return (r.start.line, r.start.column)


def _callee_identifier_position(call_node) -> tuple[int, int] | None:
    """Walk down a Scala ``call_expression`` to find its callee
    identifier's position — the spot scip-java anchors the symbol
    occurrence.

    For a chained call like ``wsClient.url("/api").get()`` there are
    two nested ``call_expression`` nodes:

    - Outer: ``<inner>.get()`` — callee is ``get``
    - Inner: ``wsClient.url("/api")`` — callee is ``url``

    The walker descends into nested call_expressions, then through a
    ``field_expression`` (``obj.method``), and returns the position of
    the rightmost identifier — that's the method name in dotted-call
    syntax. Bound the descent to avoid pathological loops.
    """
    current = call_node
    for _ in range(8):
        kind = current.kind()
        if kind in ('identifier', 'simple_identifier', 'name'):
            return _node_start_position(current)
        if kind in ('field_expression', 'select_expression'):
            sub = list(current.children())
            for c in reversed(sub):
                if c.kind() in (
                    'identifier', 'simple_identifier', 'name',
                ):
                    return _node_start_position(c)
            return _node_start_position(current)
        if kind == 'call_expression':
            children = list(current.children())
            if not children:
                return None
            current = children[0]
            continue
        return _node_start_position(current)
    return None


def _direct_arguments_node(call_node):
    """Return the direct ``arguments`` child of a call_expression, or
    ``None``. Same convention as the Akka extractor."""
    for c in call_node.children():
        if c.kind() in (
            'arguments', 'arguments_list', 'argument_list',
        ):
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
        pos = _callee_identifier_position(call)
        if pos is not None:
            out.setdefault(pos, []).append(call)
    return out


def _select_call_with_args(calls: list):
    """Pick the call with direct ``arguments``. For chained
    ``.url(x).get()``, the SCIP occurrence on ``url`` matches the
    inner call (which has the args); the outer ``.get()`` has empty
    args at a different callee position so it doesn't collide here.
    """
    for call in calls:
        if _direct_arguments_node(call) is not None:
            return call
    return None


def _arg_expression_info(
    call, arg_index: int,
) -> tuple[int, int, str | None] | None:
    """Return ``(line_1idx, col_0idx, identifier_name)`` for arg N.

    String literal → ``(line, col, None)`` — Phase 2s direct branch
    resolves via string_literals. Identifier (``simple_identifier``
    / ``identifier``) → ``(line, col, name)`` — Phase 2s variable
    branch follows the symbol to its definition. Interpolated string
    / array / other shapes → ``None``.
    """
    args_node = _direct_arguments_node(call)
    if args_node is None:
        return None
    exprs = _argument_expressions(args_node)
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


def _extract_calls_from_doc(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
owning) -> list[tuple]:
    """For one Scala file, return tuples ready to insert into
    ``http_client_calls``."""
    from docgen.scip_resolution import resolve_arg_value

    classified: list[tuple[_ScipOccurrence, SinkSpec]] = []
    for occ in doc.occurrences:
        spec = DEFAULT_SINK_REGISTRY.matching_symbol(
            occ.symbol, language='jvm',
        )
        if spec is not None and spec.kind == 'http_client':
            classified.append((occ, spec))
    if not classified:
        return []

    try:
        root = SgRoot(source_text, 'scala').root()
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
        info = _arg_expression_info(call, spec.arg_index)
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


def ingest_scala_http_clients(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root``, find Scala HTTP-client
    sinks, resolve URLs from ``string_literals``, persist to
    ``http_client_calls``.

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
        'DELETE FROM http_client_calls WHERE source_name = ?',
        (source_name,),
    )

    rows: list[tuple] = []
    owning = build_owning_resolver(index)
    for doc in index.documents:
        scala_path = source_root / doc.relative_path
        if scala_path.suffix.lower() not in _SCALA_EXTS:
            continue
        try:
            text = scala_path.read_text(
                encoding='utf-8', errors='replace',
            )
        except OSError:
            continue
        try:
            file_rows = _extract_calls_from_doc(
                doc, text, conn=conn, source_name=source_name, file=str(scala_path.resolve()), owning = owning)
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


__all__ = ['ingest_scala_http_clients']
