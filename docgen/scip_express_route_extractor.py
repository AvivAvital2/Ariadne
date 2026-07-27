"""SCIP-driven Express/Koa route extractor (Phase 8a.3).

Architecture mirrors Phase 8a.1 (Akka HTTP) and 8a.2 (Flask/FastAPI):

- **SCIP** filters which call sites are real Express/Koa router methods.
  No false positives from any local function called ``get`` or ``post``.
- **ast-grep / tree-sitter-javascript** walks the call STRUCTURE: the
  ``call_expression`` and its ``arguments`` child, the position of the
  first-argument string literal. The same ``'javascript'`` grammar
  handles both ``.js`` and ``.ts`` (matching the catalog extractor's
  convention).
- **Phase 2p ``string_literals``** supplies literal VALUES for the
  path argument. Quote stripping, template-with-interpolation
  detection, and escape handling all live in the Phase 2p pre-pass.

Precondition: ``ingest_string_literals`` must have run for the same
``(source_name, source_root)`` first. A query that returns ``None``
for the first arg's position — because it's a template literal with
``${}`` interpolation, a variable reference, or simply not yet
indexed — is treated as "unresolvable" and the route is skipped.

Pipeline:

1. Load SCIP index (production) or accept synthetic (tests).
2. For each indexed file, parse with ast-grep using the ``javascript``
   grammar.
3. Bucket every ``call_expression`` by the position of its callee's
   ``property_identifier`` — the same position scip-typescript points at
   for ``obj.method(...)`` calls.
4. For each SCIP occurrence whose symbol classifies as Express/Koa,
   look up the bucketed call at that position; extract the first arg's
   literal string as the route path.
5. Normalize Express ``:name`` / ``:name?`` to unified ``{name}``;
   persist to ``api_endpoints``.

Re-ingest: clears prior ``resolution_source='pattern'`` rows for
``source_name``. Swagger-resolved rows are preserved.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ast_grep_py import SgRoot

from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
)

if TYPE_CHECKING:
    from sqlite3 import Connection


# JS/TS source extensions Express/Koa routes can live in. Mirrors the
# sibling HTTP-client extractors (scip_js_http_client_extractor et al.) so
# a non-JS corpus — e.g. a Databricks spool's Python/Scala source — isn't
# read and parsed file-by-file for routes it can't contain.
_JS_EXTS: tuple[str, ...] = (
    '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
)

# HTTP verbs Express/Koa expose as method names on Application/Router.
# `use` (middleware) and `all` (multi-method) are intentionally excluded
# from this initial contract.
_VERB_NAMES: tuple[str, ...] = (
    'get', 'post', 'put', 'delete', 'patch', 'head', 'options',
)

# Receiver classes that scip-typescript emits in canonical_ids when the
# method belongs to Express's app/router or Koa's router. Includes the
# callable-interface forms used by ``@types/express``.
_HOST_CLASSES: tuple[str, ...] = (
    'Application',
    'Router',
    'IRouter',
    'IRouterMatcher',
)

# Express path-param regex. Matches both ``:id`` and ``:id?`` (optional);
# the trailing ``?`` is consumed but discarded — we don't preserve
# optionality in path templates.
_EXPRESS_PARAM_RE = re.compile(r':([A-Za-z_][A-Za-z0-9_]*)\??')


def _classify_symbol(symbol: str) -> str | None:
    """Return the HTTP method (``'GET'``/``'POST'``/...) for symbols
    that resolve to an Express/Koa router verb method; ``None`` if the
    symbol is unrelated.

    The matcher is suffix-based — the project/version preamble that
    scip-typescript prepends is variable and ignored.
    """
    for verb in _VERB_NAMES:
        for cls in _HOST_CLASSES:
            for sep in ('#', '.'):
                if symbol.endswith(f'{cls}{sep}{verb}().'):
                    return verb.upper()
                if symbol.endswith(f'{cls}{sep}{verb}.'):
                    return verb.upper()
    return None


def _occ_position(occ: _ScipOccurrence) -> tuple[int, int]:
    """Start (line, col) of an occurrence range. SCIP wire format is
    0-indexed for both."""
    return (occ.range[0], occ.range[1])


def _node_start_position(node) -> tuple[int, int]:
    r = node.range()
    return (r.start.line, r.start.column)


def _callee_property_position(call_node) -> tuple[int, int] | None:
    """For an ``obj.method(args)`` call, return the (line, col) start
    position of the ``method`` ``property_identifier``. That's the
    position scip-typescript points at, and the position the test
    fixture's ``_occ_at`` resolves to.

    For a bare ``method(args)`` call (no receiver) we fall back to the
    callee identifier's position. That branch isn't used by the
    Express/Koa contract today (real calls go through ``app`` or
    ``router``), but it lets the call index stay uniform.
    """
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
    """Return the ``arguments`` child of a ``call_expression``, or None."""
    for c in call_node.children():
        if c.kind() == 'arguments':
            return c
    return None


def _argument_expressions(args_node) -> list:
    """Filter syntactic punctuation from an ``arguments`` node and
    return the meaningful expression children in source order."""
    return [
        c for c in args_node.children()
        if c.kind() not in ('(', ')', ',')
    ]


def _extract_path_arg(
    call_node, *, conn: 'Connection', source_name: str, file: str,
) -> str | None:
    """Return the first positional argument's literal string value, or
    ``None`` for non-literals.

    Looks up the value in Phase 2p ``string_literals`` by the first
    arg's syntactic position. ``string`` and ``template_string`` nodes
    are valid candidate kinds; everything else (identifier, call,
    interpolated template) returns ``None`` either at the kind check
    or at the lookup (Phase 2p doesn't index interpolated forms).
    """
    from docgen.scip_string_literal_extractor import (
        lookup_literal_at_position,
    )

    args = _arguments_node(call_node)
    if args is None:
        return None
    exprs = _argument_expressions(args)
    if not exprs:
        return None
    first = exprs[0]
    if first.kind() not in ('string', 'template_string'):
        return None
    r = first.range()
    return lookup_literal_at_position(
        conn,
        source_name=source_name,
        file=file,
        line=r.start.line + 1,
        col=r.start.column,
    )


def _normalize_path(path: str) -> str:
    """Express ``:name`` / ``:name?`` → unified ``{name}`` template
    form. The optional-marker ``?`` is dropped — we don't preserve
    optionality in path templates."""
    return _EXPRESS_PARAM_RE.sub(r'{\1}', path)


def _build_call_index(root) -> dict[tuple[int, int], list]:
    """Walk the AST and bucket every ``call_expression`` by the
    position of its callee's identifier. Buckets are lists for
    symmetry with the Akka extractor (which deals with nested
    apply-with-block calls); for JS we typically have one call per
    position but list semantics make the lookup uniform."""
    out: dict[tuple[int, int], list] = {}
    for call in root.find_all(kind='call_expression'):
        pos = _callee_property_position(call)
        if pos is not None:
            out.setdefault(pos, []).append(call)
    return out


def _select_call_with_args(calls: list):
    """From a list of calls sharing a callee position, return the
    first one with an ``arguments`` child. Returns ``None`` if none
    of them have args."""
    for call in calls:
        if _arguments_node(call) is not None:
            return call
    return None


def _extract_routes_from_doc(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
) -> list[tuple[str, str]]:
    """Find Express/Koa routes in one document by joining SCIP
    classification (which calls are real Express/Koa methods) with
    tree-sitter-javascript parsing (the actual call structure).
    Literal values are read from Phase 2p ``string_literals`` via
    ``conn`` / ``source_name`` / ``file``."""
    classified: list[tuple[_ScipOccurrence, str]] = []
    for occ in doc.occurrences:
        method = _classify_symbol(occ.symbol)
        if method is not None:
            classified.append((occ, method))
    if not classified:
        return []

    try:
        root = SgRoot(source_text, 'javascript').root()
    except Exception:
        return []
    call_index = _build_call_index(root)

    routes: list[tuple[str, str]] = []
    for occ, method in classified:
        pos = _occ_position(occ)
        calls = call_index.get(pos, [])
        if not calls:
            continue
        call = _select_call_with_args(calls)
        if call is None:
            continue
        path = _extract_path_arg(
            call, conn=conn, source_name=source_name, file=file,
        )
        if path is None:
            continue
        normalized = _normalize_path(path)
        routes.append((method, normalized))
    return routes


def ingest_express_routes(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root`` and persist Express /
    Koa route endpoints.

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
            file_routes = _extract_routes_from_doc(
                doc, text,
                conn=conn,
                source_name=source_name,
                file=str(js_path.resolve()),
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


__all__ = ['ingest_express_routes']
