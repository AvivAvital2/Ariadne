"""SCIP-driven Python web framework route extractor (Phase 8a.2).

Architecture mirrors Phase 8a.1 (Akka HTTP):

- **SCIP** filters which decorators are *real* Flask/FastAPI calls (no
  false positives from any ``@x.route(...)`` decorator).
- **ast.parse** walks the decorator STRUCTURE: which arg is the path,
  which kwargs are present, where each ``ast.Constant`` lives.
- **Phase 2p ``string_literals``** supplies the literal VALUES for
  path args and ``methods=[...]`` elements. The extractor doesn't
  inspect ``ast.Constant.value`` directly — quote handling, f-string
  exclusion, and bytes filtering live in the Phase 2p pre-pass.

Precondition: ``ingest_string_literals`` must have run for the same
``(source_name, source_root)`` first. A query that returns ``None`` —
because the position holds an f-string, an interpolation, or simply
hasn't been indexed — is treated as "unresolvable arg" and the route
is skipped.

Pipeline:

1. Load SCIP index (production) or accept synthetic (tests).
2. For each Python file in the index, parse with ``ast.parse``.
3. For each ``FunctionDef`` / ``AsyncFunctionDef`` decorator that is
   an ``ast.Call`` with ``ast.Attribute`` func, look up its position
   in the SCIP occurrence map.
4. If the SCIP symbol matches Flask/FastAPI/APIRouter, classify as
   ``route`` (Flask classic) or ``<verb>`` (Flask 2 / FastAPI).
5. Read the path-string and ``methods=`` elements from
   ``string_literals`` by position.
6. Normalize path placeholders, persist to ``api_endpoints``.

Re-ingest: clears prior ``resolution_source='pattern'`` rows for
``source_name``. Swagger-resolved rows are preserved.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
)
from ast_utils import safe_ast_parse

if TYPE_CHECKING:
    from sqlite3 import Connection


# Web framework receiver classes scip-python emits in symbols.
# Blueprint covers Flask's blueprint pattern (``@bp.route(...)``).
_WEB_CLASSES: tuple[str, ...] = (
    'Flask', 'Blueprint', 'FastAPI', 'APIRouter',
)

# Single-method decorator names.
_VERB_NAMES = ('get', 'post', 'put', 'delete', 'patch', 'head', 'options')

# Path-normalization regexes.
_FLASK_PARAM_RE = re.compile(r'<(?:[^:>]+:)?([^>]+)>')
_FASTAPI_PARAM_TYPE_RE = re.compile(r'\{([^}:]+):[^}]+\}')


def _normalize_path(path: str) -> str:
    """Flask ``<id>`` / ``<int:id>`` and FastAPI ``{id:int}`` →
    unified ``{id}`` template form."""
    path = _FLASK_PARAM_RE.sub(r'{\1}', path)
    path = _FASTAPI_PARAM_TYPE_RE.sub(r'{\1}', path)
    return path


def _classify_symbol(symbol: str) -> tuple[str, str | None] | None:
    """Inspect a SCIP canonical_id; if it ends with a Flask/FastAPI
    web decorator descriptor, return ``(kind, http_method)``:

    - ``('route', None)`` for ``@app.route(...)`` (methods come from kwargs)
    - ``('verb', '<METHOD>')`` for ``@app.get/post/...``

    Otherwise None.
    """
    # Flask classic: Flask#route(). or Flask.route().
    for cls in _WEB_CLASSES:
        for sep in ('#', '.'):
            if symbol.endswith(f'{cls}{sep}route().'):
                return ('route', None)

    for verb in _VERB_NAMES:
        for cls in _WEB_CLASSES:
            for sep in ('#', '.'):
                if symbol.endswith(f'{cls}{sep}{verb}.'):
                    return ('verb', verb.upper())

    return None


def _occ_position(occ: _ScipOccurrence) -> tuple[int, int]:
    """Return (line, col) start position from an occurrence range."""
    return (occ.range[0], occ.range[1])


def _attr_position(dec: ast.Call) -> tuple[int, int] | None:
    """For an ``ast.Call`` decorator with ``ast.Attribute`` func,
    return the (line, col) position of the attribute name's start
    (0-indexed). Used to match against SCIP occurrence positions.
    """
    func = dec.func
    if not isinstance(func, ast.Attribute):
        return None
    if (
        func.end_lineno is None
        or func.end_col_offset is None
    ):
        return None
    line = func.end_lineno - 1  # ast is 1-indexed; SCIP is 0-indexed
    col = func.end_col_offset - len(func.attr)
    return (line, col)


def _extract_path_arg(
    dec: ast.Call,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
) -> str | None:
    """Return the path string from the decorator's first positional
    arg, or ``None`` if the arg isn't a string literal whose value is
    indexed in Phase 2p (f-strings, variables, expressions all return
    ``None``)."""
    from docgen.scip_string_literal_extractor import (
        lookup_literal_at_position,
    )

    if not dec.args:
        return None
    arg0 = dec.args[0]
    if not (
        isinstance(arg0, ast.Constant)
        and isinstance(arg0.value, str)
        and arg0.lineno is not None
        and arg0.col_offset is not None
    ):
        return None
    return lookup_literal_at_position(
        conn,
        source_name=source_name,
        file=file,
        line=arg0.lineno,
        col=arg0.col_offset,
    )


def _extract_methods_kwarg(
    dec: ast.Call,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
) -> list[str]:
    """For ``@app.route(...)``, parse the ``methods=[...]`` kwarg.
    Returns ``['GET']`` if not specified (Flask default). Element
    values are read from Phase 2p — non-literal entries
    (``methods=METHOD_LIST``) yield no methods, falling through to the
    default."""
    from docgen.scip_string_literal_extractor import (
        lookup_literal_at_position,
    )

    for kw in dec.keywords:
        if kw.arg == 'methods' and isinstance(kw.value, ast.List):
            methods: list[str] = []
            for elt in kw.value.elts:
                if not (
                    isinstance(elt, ast.Constant)
                    and isinstance(elt.value, str)
                    and elt.lineno is not None
                    and elt.col_offset is not None
                ):
                    continue
                val = lookup_literal_at_position(
                    conn,
                    source_name=source_name,
                    file=file,
                    line=elt.lineno,
                    col=elt.col_offset,
                )
                if val is not None:
                    methods.append(val.upper())
            if methods:
                return methods
    return ['GET']


def _extract_routes_from_doc(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
) -> list[tuple[str, str]]:
    """Find Flask/FastAPI route decorators by joining SCIP occurrences
    (which classify symbols) with the AST (which extracts arg shapes).
    Literal values are read from Phase 2p ``string_literals`` via the
    ``conn`` / ``source_name`` / ``file`` parameters."""
    # Build a position→classification map from SCIP occurrences.
    web_positions: dict[
        tuple[int, int], tuple[str, str | None],
    ] = {}
    for occ in doc.occurrences:
        cls = _classify_symbol(occ.symbol)
        if cls is not None:
            web_positions[_occ_position(occ)] = cls

    if not web_positions:
        return []

    try:
        tree = safe_ast_parse(source_text)
    except SyntaxError:
        return []

    routes: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            attr_pos = _attr_position(dec)
            if attr_pos is None:
                continue
            cls = web_positions.get(attr_pos)
            if cls is None:
                continue
            kind, verb_method = cls

            path = _extract_path_arg(
                dec, conn=conn, source_name=source_name, file=file,
            )
            if path is None:
                continue
            normalized = _normalize_path(path)

            if kind == 'route':
                methods = _extract_methods_kwarg(
                    dec, conn=conn, source_name=source_name, file=file,
                )
                for m in methods:
                    routes.append((m, normalized))
            elif kind == 'verb' and verb_method is not None:
                routes.append((verb_method, normalized))

    return routes


def ingest_python_routes(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root`` and persist Flask /
    FastAPI route endpoints.

    ``index_factory`` is the test-injection point. Production passes
    None; the function loads ``<source_root>/.ariadne/index.scip``.
    Missing index → return 0 cleanly.

    Re-ingest: clears prior ``resolution_source='pattern'`` rows for
    ``source_name``. Swagger-resolved rows preserved.
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
        'DELETE FROM api_endpoints '
        "WHERE source_name = ? AND resolution_source = 'pattern'",
        (source_name,),
    )

    rows: list[tuple] = []
    seen: set[tuple[str, str]] = set()

    for doc in index.documents:
        py_path = source_root / doc.relative_path
        try:
            text = py_path.read_text(
                encoding='utf-8', errors='replace',
            )
        except OSError:
            continue
        try:
            file_routes = _extract_routes_from_doc(
                doc, text,
                conn=conn,
                source_name=source_name,
                file=str(py_path.resolve()),
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
                endpoint_id,
                source_name,
                method,
                path_template,
                None,
                'pattern',
            ))

    if rows:
        conn.executemany(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            rows,
        )
    conn.commit()
    return len(rows)


__all__ = ['ingest_python_routes']
