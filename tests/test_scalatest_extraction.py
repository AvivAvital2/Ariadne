"""ScalaTest (FunSuite/AsyncFunSuite) test-structure extraction.

A `test("name") {…}` / `ignore("name") {…}` call resolves (via SCIP) to
ScalaTest's `test`/`ignore` symbol — that's all SCIP is used for (detection,
precise even through custom base traits). Everything structural comes from the
source AST (ast-grep): the name (a string-literal argument SCIP does not carry),
the *enclosing suite* (the innermost class/object whose node range covers the
call — scip-java emits no `enclosing_range`, so this can't come from SCIP), and
the disabled-status markers. The suite's qualified name is recovered from the
SCIP class *definition* at the class-name position. See the ScalaTest design in
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

# Real ScalaTest method symbols (FunSuite / AsyncFunSuite) that scip-java
# resolves a `test(...)` / `ignore(...)` call to — the recognizer must match.
_FUNSUITE_TEST_SYM = (
    'scip-java semanticdb maven org.scalatest scalatest_2.13 3.2.17 '
    'org/scalatest/funsuite/AnyFunSuiteLike#test().'
)
_ASYNC_TEST_SYM = (
    'scip-java semanticdb maven org.scalatest scalatest_2.13 3.2.17 '
    'org/scalatest/funsuite/AsyncFunSuiteLike#test().'
)
# `ignore("name")` — a disabled test; same trait, `ignore` method.
_FUNSUITE_IGNORE_SYM = (
    'scip-java semanticdb maven org.scalatest scalatest_2.13 3.2.17 '
    'org/scalatest/funsuite/AnyFunSuiteLike#ignore().'
)
# A non-test reference under the same package (the base-class mention) — the
# recognizer must NOT treat this as a test.
_BASECLASS_SYM = (
    'scip-java semanticdb maven org.scalatest scalatest_2.13 3.2.17 '
    'org/scalatest/funsuite/AnyFunSuite#'
)
_SUITE_SYM = 'scip-java semanticdb maven . . . com/example/WidgetSuite#'
_ASYNC_SUITE_SYM = 'scip-java semanticdb maven . . . com/example/AsyncWidgetSuite#'
# A non-test class in the same file — stays scala_class, never a test suite.
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


def _occ_at(text: str, marker: str, symbol: str, *, nth: int = 0,
            is_definition: bool = False) -> _ScipOccurrence:
    """An occurrence on the nth whole-word ``marker`` in ``text``."""
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
        is_definition=is_definition)


def _def_occ_at(text: str, marker: str, symbol: str) -> _ScipOccurrence:
    """A class *definition* occurrence at the class-name token — the shape
    scip-java actually emits: a name-position range and **no enclosing_range**
    (its absence is exactly what broke the original suite-parenting)."""
    return _occ_at(text, marker, symbol, is_definition=True)


def test_funsuite_and_asyncfunsuite_test_blocks_extracted(tmp_path):
    """`test`/`ignore` calls in FunSuite and AsyncFunSuite become located
    scala_test_case elements, parented to their enclosing suite via the AST
    (not SCIP enclosing_range), and each suite class is relabeled to
    scala_test_suite. `ignore` → skipped, a `pending` body → pending, a plain
    test → no markers; a dynamic `test(x)`, a base-class reference, and a
    non-test helper class are none of those."""
    src = (
        'class WidgetSuite extends AnyFunSuite {\n'        # 1
        '  test("renders") {}\n'                            # 2: active case
        '  test(dynamicName) {}\n'                          # 3: dynamic -> skip
        '  ignore("flaky") {}\n'                            # 4: skipped marker
        '}\n'                                               # 5
        'class AsyncWidgetSuite extends AsyncFunSuite {\n'  # 6
        '  test("updates") { pending }\n'                   # 7: pending marker
        '}\n'                                               # 8
        'class Helpers {\n'                                 # 9: non-test class
        '  def make(): Int = 0\n'                           # 10
        '}\n'                                               # 11
    )
    f = tmp_path / 'WidgetSuite.scala'
    f.write_text(src, encoding='utf-8')

    occs = [
        # Class definitions at the name token — NO enclosing_range.
        _def_occ_at(src, 'WidgetSuite', _SUITE_SYM),
        _def_occ_at(src, 'AsyncWidgetSuite', _ASYNC_SUITE_SYM),
        _def_occ_at(src, 'Helpers', _HELPER_SYM),
        # test/ignore call references at the callee.
        _occ_at(src, 'test', _FUNSUITE_TEST_SYM, nth=0),     # renders
        _occ_at(src, 'test', _FUNSUITE_TEST_SYM, nth=1),     # dynamicName
        _occ_at(src, 'ignore', _FUNSUITE_IGNORE_SYM, nth=0),  # flaky (skipped)
        _occ_at(src, 'test', _ASYNC_TEST_SYM, nth=2),        # updates (pending)
        _occ_at(src, 'AnyFunSuite', _BASECLASS_SYM, nth=0),  # non-test ref
    ]
    # Class symbols so the definition loop emits a scala_class per class — the
    # two suite ones are what relabel_suites promotes.
    syms = (
        _ScipSymbol(symbol=_SUITE_SYM, kind='Class'),
        _ScipSymbol(symbol=_ASYNC_SUITE_SYM, kind='Class'),
        _ScipSymbol(symbol=_HELPER_SYM, kind='Class'),
    )
    elements = extract(
        f, source_root=tmp_path, index=_make_index(f, tmp_path, occs, syms))

    cases = {e.qualified_name: e for e in elements if e.subtype == 'scala_test_case'}
    leaf_names = {qn.split(' > ')[-1] for qn in cases}
    # dynamic + base-class ref excluded; ignore IS a (skipped) case.
    assert leaf_names == {'renders', 'updates', 'flaky'}

    renders = next(c for qn, c in cases.items() if qn.endswith('renders'))
    assert renders.language == 'scala'
    assert renders.line_start == 2
    assert 'WidgetSuite' in (renders.parent_qualified_name or '')
    assert renders.markers == ()                       # active

    updates = next(c for qn, c in cases.items() if qn.endswith('updates'))
    assert updates.line_start == 7
    assert 'AsyncWidgetSuite' in (updates.parent_qualified_name or '')
    assert updates.markers == ('pending',)             # pending body

    flaky = next(c for qn, c in cases.items() if qn.endswith('flaky'))
    assert 'WidgetSuite' in (flaky.parent_qualified_name or '')
    assert flaky.markers == ('skipped',)               # ignore(...)

    # Each enclosing suite surfaces as scala_test_suite, relabeled from its
    # SCIP class definition — the suite QN (recovered from the class-def symbol
    # at the class-name position) matches the class element's QN.
    suites = {e.qualified_name for e in elements if e.subtype == 'scala_test_suite'}
    assert suites == {renders.parent_qualified_name, updates.parent_qualified_name}
    assert not [e for e in elements
                if e.subtype == 'scala_class' and e.qualified_name in suites]
    # Selectivity: a non-test class in the same file stays scala_class.
    helpers = [e for e in elements if e.qualified_name.endswith('Helpers')]
    assert len(helpers) == 1 and helpers[0].subtype == 'scala_class'


def test_scalatest_suite_name_falls_back_to_ast_when_scip_lacks_class(tmp_path):
    """When the index doesn't carry the suite's class definition, the suite is
    still recovered from the AST as a bare class name (ast-grep enclosure does
    not depend on SCIP); a test outside any class is parentless; a stale test
    occurrence pointing at no source call is skipped rather than crashing."""
    src = (
        'class S extends AnyFunSuite {\n'   # 1
        '  test("alone") {}\n'              # 2
        '}\n'                               # 3
        'test("orphan") {}\n'              # 4: top-level — no enclosing class
    )
    f = tmp_path / 'S.scala'
    f.write_text(src, encoding='utf-8')

    occs = [
        # No def occurrence for `S` → no canonical QN, AST gives the bare name.
        _occ_at(src, 'test', _FUNSUITE_TEST_SYM, nth=0),   # 'alone' @ line 2
        _occ_at(src, 'test', _FUNSUITE_TEST_SYM, nth=1),   # 'orphan' @ line 4
        # A test occurrence pointing where there is no call (stale/skewed index).
        _ScipOccurrence(
            symbol=_FUNSUITE_TEST_SYM, range=(99, 0, 99, 4), is_definition=False),
    ]
    elements = extract(f, source_root=tmp_path, index=_make_index(f, tmp_path, occs))

    cases = {c.qualified_name: c for c in elements if c.subtype == 'scala_test_case'}
    assert set(cases) == {'S > alone', 'orphan'}
    # 'alone' is inside class S, but no SCIP class def → bare-name suite from AST.
    assert cases['S > alone'].parent_qualified_name == 'S'
    assert cases['S > alone'].markers == ()
    # 'orphan' has no enclosing class → parentless.
    assert cases['orphan'].parent_qualified_name is None
    # No scala_class to promote (no class symbol) → no suite element fabricated.
    assert not [e for e in elements if e.subtype == 'scala_test_suite']


def test_non_test_scala_file_yields_no_test_cases(tmp_path):
    """A Scala file with no ScalaTest `test`/`ignore` occurrences produces no
    scala_test_case / scala_test_suite elements (the common, non-test path)."""
    src = (
        'class Widget {\n'
        '  def render(): Unit = ()\n'
        '}\n'
    )
    f = tmp_path / 'Widget.scala'
    f.write_text(src, encoding='utf-8')
    occs = [_def_occ_at(src, 'Widget', _SUITE_SYM)]  # a definition, no `test` ref
    elements = extract(f, source_root=tmp_path, index=_make_index(f, tmp_path, occs))
    assert not [e for e in elements if e.subtype == 'scala_test_case']
    assert not [e for e in elements if e.subtype == 'scala_test_suite']
