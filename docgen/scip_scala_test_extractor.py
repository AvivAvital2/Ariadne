"""SCIP-detected, AST-resolved ScalaTest (FunSuite/AsyncFunSuite) extractor.

ScalaTest registers tests via ``test("name") { … }`` / ``ignore("name") { … }`` —
runtime DSL calls, not named symbols. SCIP is used only to **detect** them:
scip-java resolves the call to
``org/scalatest/funsuite/{Any,Async}FunSuiteLike#{test,ignore}().`` — precise even
through custom base traits a syntactic ``extends AnyFunSuite`` check would miss.

Everything *structural* comes from the source AST (ast-grep / tree-sitter-scala),
because scip-java emits no ``enclosing_range`` for Scala definitions (dogfooding
a large real-world codebase: 0 of 143k `.scala`+`.java` defs carry it). From the AST we recover:

- the test **name** — a string-literal argument SCIP doesn't carry;
- the **enclosing suite** — the innermost ``class``/``object``/``trait`` whose node
  range covers the call; its qualified name is read from the SCIP class
  *definition* at the class-name position (canonical, matching the suite's own
  catalog element), else the bare class name;
- disabled-status **markers** — ``ignore(...)`` → ``skipped``, a ``pending`` test
  body → ``pending``.

Output: ``scala_test_case`` ``ElementInfo`` records (carrying ``markers``) parented
to the enclosing suite. Called by ``scip_extractor.extract`` for ``.scala`` files.
The tree-sitter-scala call helpers are reused from
``scip_scala_http_client_extractor``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ast_grep_py import SgRoot
from attrs import evolve

from docgen.catalog_extractor import ElementInfo
from docgen.scip_extractor import _ScipDoc, _qualified_name_from_symbol
from docgen.scip_scala_http_client_extractor import (
    _argument_expressions,
    _build_call_index,
    _direct_arguments_node,
    _select_call_with_args,
)

_SUITE_DEFINITION_KINDS = (
    'class_definition', 'object_definition', 'trait_definition',
)


def _is_scalatest_marker_symbol(symbol: str) -> bool:
    """True for ScalaTest's FunSuite/AsyncFunSuite ``test`` or ``ignore`` method
    symbol — both live in the ``*FunSuiteLike`` trait under
    ``org/scalatest/funsuite/``. SCIP resolves these through custom base traits,
    so detection catches suites a syntactic ``extends AnyFunSuite`` check misses."""
    tail = symbol.rstrip()
    return (
        'org/scalatest/funsuite/' in symbol
        and (tail.endswith('FunSuiteLike#test().')
             or tail.endswith('FunSuiteLike#ignore().'))
    )


def _first_string_literal(call) -> str | None:
    """Unquoted value of the call's first argument iff it is a plain string
    literal; ``None`` for an interpolated (``s"…"``) or non-string first arg
    (a dynamically-named test, which we skip) or no args. The call is guaranteed
    by the caller (``_select_call_with_args``) to have a direct ``arguments``
    node."""
    exprs = _argument_expressions(_direct_arguments_node(call))
    if not exprs or exprs[0].kind() not in ('string', 'string_literal'):
        return None
    return exprs[0].text()[1:-1]


def _suite_definition_nodes(root) -> list:
    """All class/object/trait definition nodes — candidate enclosing suites."""
    nodes: list = []
    for kind in _SUITE_DEFINITION_KINDS:
        nodes.extend(root.find_all(kind=kind))
    return nodes


def _enclosing_suite_qn(
    call, suite_nodes: list, class_def_index: dict,
) -> str | None:
    """Qualified name of the innermost class/object/trait whose node range covers
    the test ``call`` — its enclosing suite.

    The QN is read from the SCIP class *definition* occurrence at the class-name
    position (canonical — matches the suite's own catalog element), falling back
    to the bare class name when the index lacks it. ``None`` when no class
    encloses the call (a top-level test)."""
    call_line = call.range().start.line
    enclosing = None
    for node in suite_nodes:
        r = node.range()
        if (r.start.line <= call_line <= r.end.line
                and (enclosing is None
                     or node.range().start.line > enclosing.range().start.line)):
            enclosing = node
    if enclosing is None:
        return None
    name_node = next(c for c in enclosing.children()
                     if c.kind() in ('identifier', 'type_identifier'))
    name_pos = (name_node.range().start.line, name_node.range().start.column)
    symbol = class_def_index.get(name_pos)
    if symbol is not None:
        return _qualified_name_from_symbol(symbol, 'scala')[0]
    return name_node.text()


def _test_call_with_body(inner_call):
    """The outer ``call_expression`` wrapping ``test("…") { body }`` (so its span
    and text include the body); the inner call itself if it has no parent."""
    return inner_call.parent() or inner_call


def _markers_for(symbol: str, body_call) -> tuple[str, ...]:
    """Disabled-status markers for a test: ``ignore(...)`` → ``('skipped',)``;
    a bare ``pending`` in the test block → ``('pending',)``; an active ``test``
    → ``()``."""
    if symbol.rstrip().endswith('FunSuiteLike#ignore().'):
        return ('skipped',)
    for child in body_call.children():
        if child.kind() == 'block' and any(
                n.text() == 'pending' for n in child.find_all(kind='identifier')):
            return ('pending',)
    return ()


def extract_scalatest_cases(
    doc: _ScipDoc, *, file: Path, source_text: str,
) -> list[ElementInfo]:
    """``scala_test_case`` elements for the ScalaTest ``test``/``ignore`` calls in
    one Scala document. SCIP detects the calls; the AST supplies the name,
    enclosing suite, span, and markers. Empty when the doc has no ScalaTest
    calls."""
    test_occs = [
        o for o in doc.occurrences
        if not o.is_definition and _is_scalatest_marker_symbol(o.symbol)
    ]
    if not test_occs:
        return []
    root = SgRoot(source_text, 'scala').root()
    call_index = _build_call_index(root)
    suite_nodes = _suite_definition_nodes(root)
    class_def_index = {
        (o.range[0], o.range[1]): o.symbol
        for o in doc.occurrences if o.is_definition
    }

    file_str = str(file)
    out: list[ElementInfo] = []
    for occ in test_occs:
        inner = _select_call_with_args(
            call_index.get((occ.range[0], occ.range[1]), []))
        if inner is None:
            continue
        name = _first_string_literal(inner)
        if name is None:
            continue
        body_call = _test_call_with_body(inner)
        suite_qn = _enclosing_suite_qn(inner, suite_nodes, class_def_index)
        qn = f'{suite_qn} > {name}' if suite_qn else name
        r = body_call.range()
        out.append(ElementInfo(
            language='scala',
            subtype='scala_test_case',
            file=file_str,
            qualified_name=qn,
            signature=name,
            line_start=r.start.line + 1,
            line_end=r.end.line + 1,
            col_start=r.start.column,
            col_end=r.end.column,
            parent_qualified_name=suite_qn,
            markers=_markers_for(occ.symbol, body_call),
            body_sha=hashlib.sha256(
                body_call.text().encode('utf-8', errors='replace'),
            ).hexdigest(),
        ))
    return out


def relabel_suites(
    elements: list[ElementInfo], cases: list[ElementInfo],
) -> list[ElementInfo]:
    """Relabel each ``scala_class`` that encloses a ScalaTest case to
    ``scala_test_suite``.

    A test case's enclosing suite is its ``parent_qualified_name``; that class
    is already emitted as ``scala_class`` by the SCIP definition loop, so we
    promote that element in place — keeping its real range/signature/doc —
    instead of fabricating a duplicate. A suite whose class definition is absent
    from the index is simply left alone; no stub is invented."""
    suite_qns = {
        c.parent_qualified_name for c in cases if c.parent_qualified_name
    }
    if not suite_qns:
        return elements
    relabeled: list[ElementInfo] = []
    for e in elements:
        if e.subtype == 'scala_class' and e.qualified_name in suite_qns:
            relabeled.append(evolve(e, subtype='scala_test_suite'))
        else:
            relabeled.append(e)
    return relabeled


__all__ = ['extract_scalatest_cases', 'relabel_suites']
