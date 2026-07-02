"""SCIP-driven ScalaTest (FunSuite/AsyncFunSuite) test-case extractor.

ScalaTest registers tests via ``test("name") { … }`` — a runtime DSL call,
not a named symbol. So scip-java carries the *call occurrence* (resolved to
``org/scalatest/funsuite/{Any,Async}FunSuiteLike#test().``) and the enclosing
suite's range, but NOT the test name (a string-literal *argument*, which SCIP
does not index). This module pairs the two, exactly like the Scala HTTP-client
extractor: **SCIP** says which calls are real ScalaTest tests and what suite
encloses them; **ast-grep / tree-sitter-scala** reads the string-literal name
and range.

Output: ``scala_test_case`` ``ElementInfo`` records parented to the enclosing
suite's qualified name. Called by ``scip_extractor.extract`` for ``.scala``
files (reference occurrences, which ``extract``'s definition-only loop skips).

The call-walking helpers (``_build_call_index`` etc.) are reused from
``scip_scala_http_client_extractor`` — generic tree-sitter-scala utilities that
could later move to a shared module.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ast_grep_py import SgRoot

from docgen.catalog_extractor import ElementInfo
from docgen.scip_extractor import _ScipDoc, _qualified_name_from_symbol
from docgen.scip_scala_http_client_extractor import (
    _argument_expressions,
    _build_call_index,
    _direct_arguments_node,
    _select_call_with_args,
)


def _is_scalatest_test_symbol(symbol: str) -> bool:
    """True for ScalaTest's FunSuite / AsyncFunSuite ``test`` method symbol.

    Both suites define ``test`` in their ``*FunSuiteLike`` trait under
    ``org/scalatest/funsuite/`` — so a single suffix check covers the two
    styles in use (and excludes an unrelated ``test`` method elsewhere).
    """
    return (
        'org/scalatest/funsuite/' in symbol
        and symbol.rstrip().endswith('FunSuiteLike#test().')
    )


def _first_string_literal(call) -> str | None:
    """Unquoted value of the call's first argument iff it is a plain string
    literal; ``None`` for an interpolated (``s"…"``) or non-string first arg
    (a dynamically-named test, which we skip) or no args. The call is
    guaranteed by the caller (``_select_call_with_args``) to have a direct
    ``arguments`` node."""
    exprs = _argument_expressions(_direct_arguments_node(call))
    if not exprs or exprs[0].kind() not in ('string', 'string_literal'):
        return None
    return exprs[0].text()[1:-1]


def _enclosing_suite_qn(line0: int, suite_defs: list) -> str | None:
    """Qualified name of the first definition whose ``enclosing_range`` covers
    the 0-indexed ``line0`` — the suite the test call sits in (``suite_defs``
    is pre-filtered to definitions that carry an enclosing range)."""
    for d in suite_defs:
        er = d.enclosing_range
        if er[0] <= line0 <= er[2]:
            return _qualified_name_from_symbol(d.symbol, 'scala')[0]
    return None


def extract_scalatest_cases(
    doc: _ScipDoc, *, file: Path, source_text: str,
) -> list[ElementInfo]:
    """``scala_test_case`` elements for the ScalaTest ``test("…")`` calls in
    one Scala document. Empty when the doc has no ScalaTest ``test`` calls."""
    test_occs = [
        o for o in doc.occurrences
        if not o.is_definition and _is_scalatest_test_symbol(o.symbol)
    ]
    if not test_occs:
        return []
    call_index = _build_call_index(SgRoot(source_text, 'scala').root())
    suite_defs = [
        o for o in doc.occurrences
        if o.is_definition and o.enclosing_range is not None
    ]

    file_str = str(file)
    out: list[ElementInfo] = []
    for occ in test_occs:
        call = _select_call_with_args(call_index.get((occ.range[0], occ.range[1]), []))
        if call is None:
            continue
        name = _first_string_literal(call)
        if name is None:
            continue
        suite_qn = _enclosing_suite_qn(occ.range[0], suite_defs)
        qn = f'{suite_qn} > {name}' if suite_qn else name
        r = call.range()
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
            body_sha=hashlib.sha256(
                call.text().encode('utf-8', errors='replace'),
            ).hexdigest(),
        ))
    return out


__all__ = ['extract_scalatest_cases']
