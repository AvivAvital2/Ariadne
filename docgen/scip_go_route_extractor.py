"""SCIP-driven Go HTTP route extractor.

Architecture mirrors the Express/Koa (8a.3) and Flask/FastAPI (8a.2)
extractors:

- **SCIP** classifies which call sites are real router methods, via the
  trailing descriptor of the occurrence symbol. The matcher is
  suffix-based — the ``scip-go gomod <module> <version>`` preamble is
  variable and ignored, same convention as the other route extractors.
- **ast-grep / tree-sitter-go** walks the call STRUCTURE: the
  ``call_expression`` → ``selector_expression`` (whose ``field_identifier``
  is the method name scip-go anchors) and the ``argument_list`` whose
  first element holds the route path.
- The path literal is read DIRECTLY from the AST node
  (``interpreted_string_literal`` / ``raw_string_literal``). Go isn't in
  the ``string_literals`` table, and Go route paths are string literals in
  practice, so the position-indirection the JS/Python extractors use isn't
  needed here.

Frameworks: gin (``Engine``/``RouterGroup`` verb methods), echo
(``Echo``/``Group``), chi (``Mux``/``Router`` Title-case verbs), and
net/http (``http.HandleFunc`` / ``ServeMux.HandleFunc``/``Handle`` →
method ``ANY``, since the stdlib mux doesn't bind a verb at registration).

The symbol suffixes encode scip-go / SCIP descriptor conventions plus each
framework's exported type names; they're validated by synthetic tests and
should be confirmed against a real scip-go index.

Re-ingest: clears prior ``resolution_source='pattern'`` rows for
``source_name``; Swagger-resolved rows are preserved.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ast_grep_py import SgRoot

from docgen.scip_extractor import ScipIndex, _ScipDoc
from docgen.scip_go_ast import (
    _GO_EXTS,
    _GO_STRING_KINDS,
    _argument_expressions,
    _argument_list,
    _build_call_index,
    _select_call_with_args,
)
from docgen.scip_string_literal_extractor import lookup_literal_at_position

if TYPE_CHECKING:
    from sqlite3 import Connection


# HTTP verbs gin/echo expose as UPPERCASE method names and chi exposes as
# Title-case. Classification accepts both forms and normalizes to uppercase.
_HTTP_VERBS: tuple[str, ...] = (
    'GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS',
)

# Router receiver types across the supported frameworks. gin's verb methods
# live on RouterGroup (Engine embeds it); echo on Echo/Group; chi on
# Mux/Router. Includes interface forms scip-go may resolve embedded calls to.
_ROUTER_HOSTS: tuple[str, ...] = (
    'Engine', 'RouterGroup', 'IRoutes', 'IRouter',  # gin
    'Echo', 'Group',                                # echo
    'Mux', 'Router',                                # chi / net-http
)

# net/http registration primitives that bind no verb at the call site.
_ANY_METHOD_SUFFIXES: tuple[str, ...] = (
    'net/http/HandleFunc().', 'net/http/HandleFunc.',
    'ServeMux#HandleFunc().', 'ServeMux#HandleFunc.',
    'ServeMux#Handle().', 'ServeMux#Handle.',
)

# ``:name`` (gin/echo/httprouter) and ``*name`` (catch-all) → unified
# ``{name}`` template form. chi's native ``{name}`` is left untouched.
_GO_PARAM_RE = re.compile(r'[:*]([A-Za-z_][A-Za-z0-9_]*)')


def _classify_symbol(symbol: str) -> str | None:
    """Return the HTTP method (``'GET'``…/``'ANY'``) for a Go router
    registration symbol, or ``None`` if unrelated. Suffix-based match."""
    for suf in _ANY_METHOD_SUFFIXES:
        if symbol.endswith(suf):
            return 'ANY'
    for verb in _HTTP_VERBS:
        for host in _ROUTER_HOSTS:
            for form in (verb, verb.capitalize()):  # GET and Get
                if (symbol.endswith(f'{host}#{form}().')
                        or symbol.endswith(f'{host}#{form}.')):
                    return verb
    return None


def _path_literal(
    node, *, conn: 'Connection', source_name: str, file: str,
) -> str | None:
    """Route path for the first-arg node, read from the Phase 2p
    ``string_literals`` index by the node's position (same seam the
    Express/Python route extractors use). Non-literal args (identifiers,
    concatenations) return ``None`` — no position row exists for them."""
    if node.kind() not in _GO_STRING_KINDS:
        return None
    r = node.range()
    return lookup_literal_at_position(
        conn,
        source_name=source_name,
        file=file,
        line=r.start.line + 1,
        col=r.start.column,
    )


def _normalize_path(path: str) -> str:
    return _GO_PARAM_RE.sub(r'{\1}', path)


def _extract_routes_from_doc(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
) -> list[tuple[str, str]]:
    """Join SCIP classification (which calls are router methods) with
    tree-sitter-go parsing (the call structure) and the Phase 2p
    ``string_literals`` index (the path value)."""
    classified = [
        (occ, method)
        for occ in doc.occurrences
        if (method := _classify_symbol(occ.symbol)) is not None
    ]
    if not classified:
        return []

    try:
        root = SgRoot(source_text, 'go').root()
    except Exception:
        return []
    call_index = _build_call_index(root)

    routes: list[tuple[str, str]] = []
    for occ, method in classified:
        pos = (occ.range[0], occ.range[1])
        call = _select_call_with_args(call_index.get(pos, []))
        if call is None:
            continue
        exprs = _argument_expressions(_argument_list(call))
        if not exprs:
            continue
        path = _path_literal(
            exprs[0], conn=conn, source_name=source_name, file=file,
        )
        if path is None:
            continue
        routes.append((method, _normalize_path(path)))
    return routes


def ingest_go_routes(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root`` and persist Go HTTP route
    endpoints to ``api_endpoints`` (``resolution_source='pattern'``).

    ``index_factory`` is the test-injection point; production passes None
    and the loader reads ``<source_root>/.ariadne/index.scip`` (missing →
    0). Returns the number of endpoints inserted.
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
        'DELETE FROM api_endpoints '
        "WHERE source_name = ? AND resolution_source = 'pattern'",
        (source_name,),
    )

    rows: list[tuple] = []
    seen: set[tuple[str, str]] = set()
    for doc in index.documents:
        go_path = source_root / doc.relative_path
        if go_path.suffix.lower() not in _GO_EXTS:
            continue
        try:
            text = go_path.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        try:
            file_routes = _extract_routes_from_doc(
                doc, text, conn=conn, source_name=source_name,
                file=str(go_path.resolve()),
            )
        except Exception:
            continue
        for method, path_template in file_routes:
            key = (method, path_template)
            if key in seen:
                continue
            seen.add(key)
            endpoint_id = hashlib.sha256(
                f'{source_name}:{method}:{path_template}'.encode(),
            ).hexdigest()[:16]
            rows.append((
                endpoint_id, source_name, method, path_template, None,
                'pattern',
            ))

    if rows:
        conn.executemany(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)', rows,
        )
    conn.commit()
    return len(rows)


__all__ = ['ingest_go_routes']
