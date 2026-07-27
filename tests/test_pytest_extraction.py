"""pytest / unittest test-structure extraction (ast-grep, _extract_python).

pytest tests are *named functions* — already extracted as function/method
elements — so recognition relabels them to `py_test_case` (and `Test*` classes
to `py_test_suite`) and reads skip/xfail **decorators** as disabled-status
markers. Detection follows pytest's collection rules: a `test_*` function is a
case only when module-level or a method of a `Test*` class; fixtures,
`testing_*` helpers, plain methods, methods of non-`Test*` classes, and nested
functions are not. See designs/pytest-test-extraction.md. Synthetic only.
"""
from __future__ import annotations

from docgen.catalog_extractor import extract_elements

_SRC = '''import pytest
import unittest


def test_module_level():
    assert True


@pytest.mark.asyncio
async def test_async():
    assert True


@pytest.mark.skipif(True, reason="env")
def test_skipped():
    pass


@pytest.mark.xfail
def test_xf():
    pass


@unittest.expectedFailure
def test_known_broken():
    pass


@pytest.fixture
def some_fixture():
    return 1


def testing_utils():
    pass


def outer_helper():
    def test_nested():
        pass
    return test_nested


class TestWidget:
    def test_method(self):
        assert True

    @pytest.mark.skip
    def test_skip_method(self):
        pass

    def helper(self):
        pass


class Helper:
    def test_not_collected(self):
        pass
'''


def test_pytest_cases_suites_and_markers(tmp_path):
    f = tmp_path / 'test_widget.py'
    f.write_text(_SRC, encoding='utf-8')
    elements = extract_elements(f, tmp_path)

    cases = {e.qualified_name.rsplit('.', 1)[-1]: e
             for e in elements if e.subtype == 'py_test_case'}
    suites = [e for e in elements if e.subtype == 'py_test_suite']
    by_leaf = {e.qualified_name.rsplit('.', 1)[-1]: e for e in elements}

    # Collectable tests: module-level test_* (incl. async + unittest-marked) and
    # test_* methods of a Test* class. Nothing else.
    assert set(cases) == {
        'test_module_level', 'test_async', 'test_skipped', 'test_xf',
        'test_known_broken', 'test_method', 'test_skip_method',
    }

    # The Test* class is a suite; its test methods parent to it.
    assert len(suites) == 1
    assert suites[0].qualified_name.endswith('TestWidget')
    assert cases['test_method'].parent_qualified_name == suites[0].qualified_name
    assert cases['test_module_level'].parent_qualified_name is None

    # Disabled-status markers from decorators.
    assert cases['test_skipped'].markers == ('skipped',)       # @pytest.mark.skipif
    assert cases['test_skip_method'].markers == ('skipped',)   # @pytest.mark.skip
    assert cases['test_xf'].markers == ('xfail',)              # @pytest.mark.xfail
    assert cases['test_known_broken'].markers == ('xfail',)    # @unittest.expectedFailure
    assert cases['test_module_level'].markers == ()            # active
    assert cases['test_async'].markers == ()                   # asyncio is not a status marker

    # Selectivity — none of these are tests/suites:
    assert by_leaf['some_fixture'].subtype == 'function'        # @pytest.fixture, not test_
    assert by_leaf['testing_utils'].subtype == 'function'       # testing_, not test_
    assert by_leaf['test_nested'].subtype == 'function'         # nested in a function
    assert by_leaf['helper'].subtype == 'method'                # not test_
    assert by_leaf['test_not_collected'].subtype == 'method'    # non-Test* class
    assert by_leaf['Helper'].subtype == 'class'                 # not Test*


def test_non_test_file_functions_are_not_relabeled(tmp_path):
    """A `test_*` function in a non-test file (not `test_*.py` / `*_test.py`) is
    something pytest never collects — so it stays a plain function, not a
    py_test_case."""
    f = tmp_path / 'helpers.py'
    f.write_text('def test_connection():\n    return True\n', encoding='utf-8')
    elements = extract_elements(f, tmp_path)
    assert not [e for e in elements if e.subtype == 'py_test_case']
    fn = next(e for e in elements if e.qualified_name.endswith('test_connection'))
    assert fn.subtype == 'function'
