"""SCIP-driven Akka HTTP route extractor (Phase 8a.1).

Architecture:

- **SCIP** filters which call sites are real Akka HTTP directives.
  No false positives from any local function called ``path`` or ``get``.
- **ast-grep / tree-sitter-scala** walks the call STRUCTURE: argument
  lists, block bodies, nesting, ``/``-chained path composition.
- **Phase 2p ``string_literals``** supplies literal VALUES for
  arguments at known positions. The extractor doesn't re-parse string
  contents — quote stripping, escape handling, and interpolation
  detection all live in the Phase 2p pre-pass. This module queries by
  position.

Precondition: ``ingest_string_literals`` must have run for the same
``(source_name, source_root)`` before this extractor runs. A query for
a literal at an unindexed position returns ``None`` — the same signal
as "interpolated / unresolvable arg" — and the surrounding route is
skipped.

Re-ingest semantics: clears prior ``resolution_source='pattern'`` rows
for ``source_name``. Swagger-resolved rows preserved.
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


# Suffix patterns that identify Akka HTTP directive symbols. SCIP
# canonical_ids end with these descriptors when the call resolves to
# the Akka HTTP DSL. The leading project/version preamble is variable
# and ignored.
_PATH_SUFFIXES: tuple[str, ...] = (
    'Directives#path().',
    'Directives.path().',
)
_PATH_PREFIX_SUFFIXES: tuple[str, ...] = (
    'Directives#pathPrefix().',
    'Directives.pathPrefix().',
)
_VERB_SUFFIXES: dict[tuple[str, ...], str] = {
    ('Directives#get.', 'Directives.get.'): 'GET',
    ('Directives#post.', 'Directives.post.'): 'POST',
    ('Directives#put.', 'Directives.put.'): 'PUT',
    ('Directives#delete.', 'Directives.delete.'): 'DELETE',
    ('Directives#patch.', 'Directives.patch.'): 'PATCH',
    ('Directives#head.', 'Directives.head.'): 'HEAD',
    ('Directives#options.', 'Directives.options.'): 'OPTIONS',
}

# Akka HTTP path-segment matchers — names that translate to a
# parameter placeholder in the resulting path template.
_PARAM_MATCHERS = frozenset({
    'Segment', 'IntNumber', 'LongNumber', 'JavaUUID',
    'DoubleNumber', 'HexLongNumber', 'HexIntNumber', 'Remaining',
})


def _classify_occurrence(symbol: str) -> str | None:
    """Return ``'path'`` / ``'pathPrefix'`` / a verb method name
    (``'GET'``/etc.) for Akka HTTP directives; ``None`` otherwise."""
    for suf in _PATH_SUFFIXES:
        if symbol.endswith(suf):
            return 'path'
    for suf in _PATH_PREFIX_SUFFIXES:
        if symbol.endswith(suf):
            return 'pathPrefix'
    for suffixes, method in _VERB_SUFFIXES.items():
        for suf in suffixes:
            if symbol.endswith(suf):
                return method
    return None


def _occ_position(occ: _ScipOccurrence) -> tuple[int, int]:
    """Start (line, col) of an occurrence range. SCIP wire format is
    0-indexed for both."""
    return (occ.range[0], occ.range[1])


def _node_start_position(node) -> tuple[int, int]:
    """Start (line, col) of an ast-grep node. Tree-sitter ranges are
    0-indexed; lines and columns are byte/grapheme offsets."""
    r = node.range()
    return (r.start.line, r.start.column)


def _callee_identifier_position(call_node) -> tuple[int, int] | None:
    """For a call_expression, walk down to find the position of the
    callee's identifier name.

    Tree-sitter-scala parses ``f(args) { block }`` as an *outer*
    call_expression whose first child is an *inner* call_expression
    containing the actual function identifier and arguments, and
    whose second child is the block. Both inner and outer share the
    same callee-name position; we recurse through nested
    call_expression layers until we hit an identifier or
    field_expression.

    For dotted access (``Directives.path``) parsed as field_expression,
    the method name is the rightmost identifier child — that's what
    SCIP points at.
    """
    current = call_node
    # Bound the descent to avoid pathological loops on weird grammars.
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
        # Unknown leaf-ish node — use its start position.
        return _node_start_position(current)
    return None


def _lookup_string_literal_at_node(
    node, *, conn, source_name: str, file: str,
) -> str | None:
    """Return the Phase 2p literal value at the start of ``node``, or
    ``None`` if the position isn't indexed.

    Interpolated strings (``s"..."``) aren't indexed by Phase 2p, so
    they correctly resolve to ``None`` here without us re-checking the
    node kind.
    """
    from docgen.scip_string_literal_extractor import (
        lookup_literal_at_position,
    )

    r = node.range()
    return lookup_literal_at_position(
        conn,
        source_name=source_name,
        file=file,
        line=r.start.line + 1,
        col=r.start.column,
    )


def _direct_arguments_node(call_node):
    """Return the immediate ``arguments`` child of a call_expression,
    or None.

    Per tree-sitter-scala, only the *inner* call_expression of an
    apply-with-block has an ``arguments`` child. The outer wrapper
    has the inner call as its first child and a block as its second.
    """
    for c in call_node.children():
        if c.kind() in (
            'arguments', 'arguments_list', 'argument_list',
        ):
            return c
    return None


def _argument_expressions(args_node) -> list:
    """Filter syntactic punctuation from an ``arguments`` node and
    return the meaningful expression children."""
    return [
        c for c in args_node.children()
        if c.kind() not in ('(', ')', ',')
    ]


def _block_body(call_node):
    """Return the block-argument node of an apply-with-block call.

    For ``f(args) { block }``, tree-sitter-scala makes ``block`` a
    direct child of the *outer* call_expression. Returns None if no
    block child exists (this call isn't an apply-with-block).
    """
    for c in call_node.children():
        if c.kind() in ('block', 'block_expression'):
            return c
    return None


def _inner_call_with_args(outer_call):
    """For an outer call_expression with a block child, return the
    inner call_expression that holds the function args.

    Layout (per AST dump):

    .. code-block:: text

        call_expression                 ← outer
        ├── call_expression             ← inner (function + arguments)
        └── block                       ← the body

    If ``outer_call`` itself has direct ``arguments`` (i.e., it IS the
    inner call — used e.g. for verb directives like ``post { ... }``
    that have no arguments-with-parens), return ``outer_call`` as the
    "inner" so callers can probe it uniformly.
    """
    if _direct_arguments_node(outer_call) is not None:
        return outer_call
    for c in outer_call.children():
        if c.kind() == 'call_expression':
            return c
    return None


def _resolve_path_fragment(
    arg_expressions: list,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
) -> str | None:
    """Build a path-template fragment from the argument expressions
    of a ``path()`` / ``pathPrefix()`` call.

    Common shapes:

    - A single string literal: ``"login"`` → ``login``
    - A ``/``-chained expression: ``"users" / Segment`` → ``users/{p}``
      (parsed as ``infix_expression`` or ``binary_expression``)

    Returns None if any component is unresolvable (interpolation,
    arbitrary expressions, unknown identifiers, or any literal not
    indexed in Phase 2p).
    """
    if not arg_expressions:
        return None
    parts: list[str] = []
    for arg in arg_expressions:
        if not _walk_path_chain(
            arg, parts,
            conn=conn, source_name=source_name, file=file,
        ):
            return None
    return '/'.join(parts)


def _walk_path_chain(
    node, parts_out: list[str], *,
    conn: 'Connection', source_name: str, file: str,
) -> bool:
    """Walk a possibly-``/``-chained path expression, appending
    fragment names to ``parts_out`` in source order.

    Returns False if any sub-expression is unresolvable; the caller
    discards partially-collected parts in that case.
    """
    kind = node.kind()
    text = node.text()

    # `/`-chained composition. Tree-sitter-scala may parse this as
    # binary_expression or infix_expression depending on grammar
    # version; both follow the lhs / op / rhs shape.
    if kind in ('binary_expression', 'infix_expression'):
        meaningful = [
            c for c in node.children()
            if c.kind() not in ('(', ')', ',')
        ]
        # Expect at least three children: lhs, op, rhs.
        if len(meaningful) >= 3:
            lhs, op = meaningful[0], meaningful[1]
            rhs_or_more = meaningful[2:]
            if op.text() != '/':
                return False
            if not _walk_path_chain(
                lhs, parts_out,
                conn=conn, source_name=source_name, file=file,
            ):
                return False
            for rhs in rhs_or_more:
                if not _walk_path_chain(
                    rhs, parts_out,
                    conn=conn, source_name=source_name, file=file,
                ):
                    return False
            return True
        # Unusual shape; bail.
        return False

    # Plain string literal — value comes from Phase 2p.
    if kind in ('string', 'string_literal'):
        val = _lookup_string_literal_at_node(
            node, conn=conn, source_name=source_name, file=file,
        )
        if val is not None:
            parts_out.append(val)
            return True
        # Position not indexed → unresolvable. Same outcome whether the
        # literal was an interpolation or simply not yet ingested.
        return False

    # Identifier — either a known param matcher or unresolvable.
    if kind in ('identifier', 'simple_identifier', 'name'):
        if text in _PARAM_MATCHERS:
            parts_out.append('{p}')
            return True
        return False

    # Interpolated string — unresolvable.
    if kind in (
        'interpolated_string_expression',
        'interpolated_string',
    ):
        return False

    return False


def _build_call_index(root) -> dict[tuple[int, int], list]:
    """Walk the AST and bucket every ``call_expression`` by the
    position of its callee identifier.

    Multiple calls share a position because tree-sitter-scala parses
    apply-with-block as nested calls — the inner has the args, the
    outer has the block. Both have the same callee identifier
    position. Callers pick which call they need (inner for args,
    outer for block).
    """
    out: dict[tuple[int, int], list] = {}
    for call in root.find_all(kind='call_expression'):
        pos = _callee_identifier_position(call)
        if pos is not None:
            out.setdefault(pos, []).append(call)
    return out


def _select_apply_with_block(calls: list):
    """From a list of calls sharing a callee position, return the one
    with a direct ``block`` child (the apply-with-block outer). None
    if no such call (e.g., a bare directive without a body)."""
    for call in calls:
        if _block_body(call) is not None:
            return call
    return None


def _select_call_with_args(calls: list):
    """From a list of calls sharing a callee position, return the one
    with a direct ``arguments`` child (the inner call holding the
    parenthesized args). None if none of them have args (e.g., a verb
    directive like ``post { ... }`` that has only a block)."""
    for call in calls:
        if _direct_arguments_node(call) is not None:
            return call
    return None


def _extract_routes_from_doc(
    doc: _ScipDoc,
    source_text: str,
    *,
    conn: 'Connection',
    source_name: str,
    file: str,
) -> list[tuple[str, str, int]]:
    """Find Akka HTTP routes in one document by joining SCIP
    classification (which calls are real directives) with tree-sitter
    parsing (the actual call structure). Literal values are read from
    Phase 2p ``string_literals`` via ``conn``/``source_name``/``file``."""
    classified: list[tuple[_ScipOccurrence, str]] = []
    for occ in doc.occurrences:
        kind = _classify_occurrence(occ.symbol)
        if kind is not None:
            classified.append((occ, kind))
    if not classified:
        return []

    # Parse Scala via tree-sitter; let exceptions propagate so any
    # parser issue surfaces in test output instead of producing an
    # empty result silently.
    root = SgRoot(source_text, 'scala').root()
    call_index = _build_call_index(root)

    # Phase 1: walk path / pathPrefix occurrences. For each, the
    # outer call_expression (which has the block child) gives us the
    # body span; the inner call_expression (which has the arguments
    # child) gives us the path-fragment text.
    resolved_scopes: list[
        tuple[str, tuple[int, int], tuple[int, int]]
    ] = []
    unresolvable_spans: list[
        tuple[tuple[int, int], tuple[int, int]]
    ] = []

    for occ, kind in classified:
        if kind not in ('path', 'pathPrefix'):
            continue
        pos = _occ_position(occ)
        calls = call_index.get(pos, [])
        outer = _select_apply_with_block(calls)
        if outer is None:
            # Directive without a block body — skip.
            continue
        block = _block_body(outer)
        if block is None:
            continue
        body_r = block.range()
        body_start = (body_r.start.line, body_r.start.column)
        body_end = (body_r.end.line, body_r.end.column)

        # Find the call that holds the arguments. For Scala
        # apply-with-block, that's the inner call; in degenerate
        # cases the outer may itself have args.
        inner = _inner_call_with_args(outer)
        args_node = (
            _direct_arguments_node(inner) if inner is not None else None
        )
        if args_node is None:
            unresolvable_spans.append((body_start, body_end))
            continue
        arg_exprs = _argument_expressions(args_node)
        fragment = _resolve_path_fragment(
            arg_exprs,
            conn=conn, source_name=source_name, file=file,
        )
        if fragment is None:
            unresolvable_spans.append((body_start, body_end))
            continue
        resolved_scopes.append((fragment, body_start, body_end))

    # Phase 2: emit one row per verb occurrence, attributed to its
    # enclosing path scopes.
    routes: list[tuple[str, str, int]] = []
    for occ, kind in classified:
        if kind in ('path', 'pathPrefix'):
            continue
        pos = _occ_position(occ)
        if _position_inside_any(pos, unresolvable_spans):
            continue
        enclosing = [
            (frag, start, end)
            for frag, start, end in resolved_scopes
            if _position_inside(pos, start, end)
        ]
        # Outer-to-inner ordering: outer scopes start earlier in the
        # source.
        enclosing.sort(key=lambda s: s[1])
        path = _compose_path([s[0] for s in enclosing])
        routes.append((kind, path, pos[0] + 1))

    return routes


def _position_inside(
    point: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    """``start <= point < end`` using lexicographic (line, col) order."""
    return start <= point < end


def _position_inside_any(
    point: tuple[int, int],
    spans: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    return any(
        _position_inside(point, start, end)
        for start, end in spans
    )


def _compose_path(fragments: list[str]) -> str:
    """Join non-empty fragments with `/`, prepend leading `/`,
    collapse duplicate slashes."""
    parts = [f for f in fragments if f]
    if not parts:
        return '/'
    joined = '/'.join(parts)
    if not joined.startswith('/'):
        joined = '/' + joined
    return re.sub(r'/+', '/', joined)


def ingest_akka_http_routes(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
    index_factory: Callable[[], ScipIndex] | None = None,
) -> int:
    """Walk the SCIP index for ``source_root``, extract Akka HTTP
    routes, persist to ``api_endpoints``.

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
        scala_path = source_root / doc.relative_path
        try:
            text = scala_path.read_text(
                encoding='utf-8', errors='replace',
            )
        except OSError:
            continue
        # Let exceptions propagate during development to surface real
        # parser issues. Production wrapping can be added once the
        # contract is settled.
        file_routes = _extract_routes_from_doc(
            doc, text,
            conn=conn,
            source_name=source_name,
            file=str(scala_path.resolve()),
        )
        for method, path_template, _line in file_routes:
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


__all__ = ['ingest_akka_http_routes']
