"""Shared tree-sitter-go call-structure helpers for the Go SCIP extractors.

The Go route extractor (:mod:`docgen.scip_go_route_extractor`) and the Go
HTTP-client extractor (:mod:`docgen.scip_go_http_client_extractor`) both join
SCIP occurrence positions to tree-sitter-go ``call_expression`` nodes and read
their argument list. That call-structure walk is identical for both, so it
lives here once instead of being copied into each extractor.
"""
from __future__ import annotations

# Go source extension — from the one authoritative registry (no drift).
from docgen.scip_languages import GO_GRAMMAR_EXTS as _GO_EXTS  # noqa: E402

_GO_STRING_KINDS: tuple[str, ...] = (
    'interpreted_string_literal', 'raw_string_literal',
)


def _node_start(node) -> tuple[int, int]:
    r = node.range()
    return (r.start.line, r.start.column)


def _callee_position(call_node) -> tuple[int, int] | None:
    """Position of the method-name token scip-go anchors: the
    ``field_identifier`` of a ``selector_expression`` callee
    (``obj.Method`` / ``pkg.Func``), or a bare ``identifier`` callee."""
    children = list(call_node.children())
    if not children:
        return None
    func = children[0]
    kind = func.kind()
    if kind == 'selector_expression':
        for c in reversed(list(func.children())):
            if c.kind() == 'field_identifier':
                return _node_start(c)
        return None
    if kind == 'identifier':
        return _node_start(func)
    return None


def _argument_list(call_node):
    for c in call_node.children():
        if c.kind() == 'argument_list':
            return c
    return None


def _argument_expressions(args_node) -> list:
    if args_node is None:
        return []
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
        if _argument_list(call) is not None:
            return call
    return None
