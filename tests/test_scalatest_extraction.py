"""ScalaTest (FunSuite/AsyncFunSuite) test-structure extraction.

A `test("name") {…}` call resolves (via SCIP) to ScalaTest's `test` symbol;
its name is a string-literal argument SCIP does not carry, so it's recovered
from the source AST (ast-grep). The result is a `scala_test_case` catalog
element parented to its enclosing suite. See the ScalaTest plan in
designs/. Synthetic fixtures only.
"""
from __future__ import annotations

from pathlib import Path

from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
    _ScipSymbol,
    extract,
)

# Real ScalaTest `test` method symbols (FunSuite / AsyncFunSuite) that scip-java
# resolves a `test(...)` call to — the recognizer must match these.
_FUNSUITE_TEST_SYM = (
    'scip-java semanticdb maven org.scalatest scalatest_2.13 3.2.17 '
    'org/scalatest/funsuite/AnyFunSuiteLike#test().'
)
_ASYNC_TEST_SYM = (
    'scip-java semanticdb maven org.scalatest scalatest_2.13 3.2.17 '
    'org/scalatest/funsuite/AsyncFunSuiteLike#test().'
)
# A non-test reference under the same package (the base-class mention) — the
# recognizer must NOT treat this as a test.
_BASECLASS_SYM = (
    'scip-java semanticdb maven org.scalatest scalatest_2.13 3.2.17 '
    'org/scalatest/funsuite/AnyFunSuite#'
)
_SUITE_SYM = 'scip-java semanticdb maven . . . com/example/WidgetSuite#'
_ASYNC_SUITE_SYM = 'scip-java semanticdb maven . . . com/example/AsyncWidgetSuite#'
# A non-test class in the same file — its element must stay scala_class and
# never be relabeled to a test suite (slice-2 selectivity).
_HELPER_SYM = 'scip-java semanticdb maven . . . com/example/Helpers#'


def _make_index(src_file: Path, source_root: Path,
                occurrences: list[_ScipOccurrence],
                symbols: tuple[_ScipSymbol, ...] = ()) -> ScipIndex:
    rel = src_file.relative_to(source_root)
    return ScipIndex(
        documents=(_ScipDoc(
            relative_path=str(rel),
            occurrences=tuple(occurrences),
            symbols=tuple(symbols),
        ),),
        source_root=source_root,
    )


def _def_occ(symbol: str, line_start: int, line_end: int) -> _ScipOccurrence:
    """Definition occurrence with an enclosing body range (1-indexed lines) —
    the shape the owning resolver reads to find a test's enclosing suite."""
    return _ScipOccurrence(
        symbol=symbol, range=(line_start - 1, 6, 40), is_definition=True,
        enclosing_range=(line_start - 1, 0, line_end - 1, 0))


def _occ_at(text: str, marker: str, symbol: str, *, nth: int = 0) -> _ScipOccurrence:
    """A reference occurrence on the nth whole-word ``marker`` in ``text``."""
    found = 0
    pos = -1
    n = len(text)
    i = 0
    while i <= n - len(marker):
        if text.startswith(marker, i):
            left = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')
            j = i + len(marker)
            right = j >= n or not (text[j].isalnum() or text[j] == '_')
            if left and right:
                if found == nth:
                    pos = i
                    break
                found += 1
                i = j
                continue
        i += 1
    if pos < 0:
        raise ValueError(f'marker {marker!r} (nth={nth}) not found as a word')
    line = text.count('\n', 0, pos)
    col = pos - (text.rfind('\n', 0, pos) + 1)
    return _ScipOccurrence(
        symbol=symbol, range=(line, col, line, col + len(marker)),
        is_definition=False)


def test_funsuite_and_asyncfunsuite_test_blocks_extracted(tmp_path):
    """`test("...")` calls in FunSuite and AsyncFunSuite become located
    scala_test_case elements parented to their suite, and each enclosing suite
    class is relabeled from scala_class to scala_test_suite. A dynamically-named
    `test(x)`, a non-test base-class reference, and a plain helper class are
    none of those."""
    src = (
        'class WidgetSuite extends AnyFunSuite {\n'        # 1
        '  test("renders") {}\n'                            # 2: case
        '  test(dynamicName) {}\n'                          # 3: dynamic -> skip
        '}\n'                                               # 4
        'class AsyncWidgetSuite extends AsyncFunSuite {\n'  # 5
        '  test("updates") {}\n'                            # 6: case (async parity)
        '}\n'                                               # 7
        'class Helpers {\n'                                 # 8: non-test class
        '  def make(): Int = 0\n'                           # 9
        '}\n'                                               # 10
    )
    f = tmp_path / 'WidgetSuite.scala'
    f.write_text(src, encoding='utf-8')

    occs = [
        _def_occ(_SUITE_SYM, 1, 4),
        _def_occ(_ASYNC_SUITE_SYM, 5, 7),
        _def_occ(_HELPER_SYM, 8, 10),
        _occ_at(src, 'test', _FUNSUITE_TEST_SYM, nth=0),   # renders
        _occ_at(src, 'test', _FUNSUITE_TEST_SYM, nth=1),   # dynamicName
        _occ_at(src, 'test', _ASYNC_TEST_SYM, nth=2),      # updates
        _occ_at(src, 'AnyFunSuite', _BASECLASS_SYM, nth=0),  # non-test ref
    ]
    # Class symbols so the definition loop emits a scala_class per class —
    # the two suite ones are what slice 2 relabels.
    syms = (
        _ScipSymbol(symbol=_SUITE_SYM, kind='Class'),
        _ScipSymbol(symbol=_ASYNC_SUITE_SYM, kind='Class'),
        _ScipSymbol(symbol=_HELPER_SYM, kind='Class'),
    )
    elements = extract(
        f, source_root=tmp_path, index=_make_index(f, tmp_path, occs, syms))

    cases = {e.qualified_name: e for e in elements if e.subtype == 'scala_test_case'}
    leaf_names = {qn.split(' > ')[-1] for qn in cases}
    assert leaf_names == {'renders', 'updates'}  # dynamic + base-class ref excluded

    renders = next(c for qn, c in cases.items() if qn.endswith('renders'))
    assert renders.language == 'scala'
    assert renders.line_start == 2
    assert 'WidgetSuite' in (renders.parent_qualified_name or '')

    updates = next(c for qn, c in cases.items() if qn.endswith('updates'))
    assert updates.line_start == 6
    assert 'AsyncWidgetSuite' in (updates.parent_qualified_name or '')

    # Slice 2: each enclosing suite surfaces as a scala_test_suite, relabeled
    # from its scala_class definition — exactly the suites that own tests.
    suites = {e.qualified_name: e for e in elements
              if e.subtype == 'scala_test_suite'}
    assert set(suites) == {renders.parent_qualified_name,
                           updates.parent_qualified_name}
    # Relabel, not duplicate: no scala_class lingers for a suite with tests.
    assert not [e for e in elements
                if e.subtype == 'scala_class' and e.qualified_name in suites]
    # Derived from the real class def — keeps the class's range + signature,
    # not a fabricated stub.
    widget_suite = suites[renders.parent_qualified_name]
    assert widget_suite.line_start == 1
    assert 'WidgetSuite' in widget_suite.signature
    # Selectivity: a non-test class in the same file stays scala_class.
    helpers = [e for e in elements if e.qualified_name.endswith('Helpers')]
    assert len(helpers) == 1 and helpers[0].subtype == 'scala_class'


def test_scalatest_degrades_gracefully_on_partial_scip(tmp_path):
    """When the index doesn't carry the enclosing suite, a test still surfaces
    with no parent; a stale test occurrence pointing at no source call is
    skipped rather than crashing."""
    src = (
        'class S extends AnyFunSuite {\n'   # 1
        '  test("alone") {}\n'              # 2
        '}\n'                               # 3
    )
    f = tmp_path / 'S.scala'
    f.write_text(src, encoding='utf-8')

    occs = [
        # No def occurrence for `S` → its enclosing range is unknown.
        _occ_at(src, 'test', _FUNSUITE_TEST_SYM, nth=0),   # 'alone' @ line 2
        # A test occurrence pointing where there is no call (stale/skewed index).
        _ScipOccurrence(
            symbol=_FUNSUITE_TEST_SYM, range=(99, 0, 99, 4), is_definition=False),
    ]
    elements = extract(f, source_root=tmp_path, index=_make_index(f, tmp_path, occs))

    cases = [e for e in elements if e.subtype == 'scala_test_case']
    assert len(cases) == 1
    assert cases[0].qualified_name == 'alone'          # no suite prefix
    assert cases[0].parent_qualified_name is None
    # No class symbol in the index → the suite is not promoted; the case
    # surfaces parentless rather than us fabricating a scala_test_suite stub.
    assert not [e for e in elements if e.subtype == 'scala_test_suite']


def test_non_test_scala_file_yields_no_test_cases(tmp_path):
    """A Scala file with no ScalaTest `test` occurrences produces no
    scala_test_case elements (the common, non-test-file path)."""
    src = (
        'class Widget {\n'
        '  def render(): Unit = ()\n'
        '}\n'
    )
    f = tmp_path / 'Widget.scala'
    f.write_text(src, encoding='utf-8')
    occs = [_def_occ(_SUITE_SYM, 1, 3)]  # a definition, but no `test` reference
    elements = extract(f, source_root=tmp_path, index=_make_index(f, tmp_path, occs))
    assert not [e for e in elements if e.subtype == 'scala_test_case']
    assert not [e for e in elements if e.subtype == 'scala_test_suite']
