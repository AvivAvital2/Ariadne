"""SCIP-backed dependency/test discovery.

Regression for the E5 finding: `find_tests_for` leaned on `doc_graph`
'imports' edges that are empty library-wide, so it degraded to a
cross-source filename heuristic (a coderbot test matched an ao-core file on
the word 'dataset'). The fix routes it through the SCIP call graph that
`impact_radius` already uses. This pins the shared helper: transitive
call-dependents of a file, source-scoped, optionally tests-only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import (
    ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
)
from library import Library


def _doc(rel, defs, refs):
    occ = [_ScipOccurrence(symbol=s, range=r, is_definition=True) for s, r, _ in defs]
    occ += [_ScipOccurrence(symbol=s, range=r, is_definition=False) for s, r in refs]
    syms = tuple(
        _ScipSymbol(symbol=s, kind='Function', display_name=name)
        for s, _, name in defs
    )
    return _ScipDoc(relative_path=rel, occurrences=tuple(occ), symbols=syms)


DATASET = 'scip-python . p 0.1 core/dataset/Dataset#'
TEST_FN = 'scip-python . p 0.1 tests/test_dataset/test_it().'
CONSUMER = 'scip-python . p 0.1 core/consumer/consume().'
OTHER_FN = 'scip-python . p 0.1 tests/test_dataset_utils/test_util().'


@pytest.fixture
def lib(tmp_path: Path):
    graph = CrossSourceGraph()
    # mysrc: dataset.py defines Dataset; a test AND a non-test both call it.
    graph.add_source('mysrc', language='python', index=ScipIndex(documents=(
        _doc('core/dataset.py', defs=[(DATASET, (0, 0, 10, 0), 'Dataset')], refs=[]),
        _doc('tests/test_dataset.py',
             defs=[(TEST_FN, (0, 0, 5, 0), 'test_it')],
             refs=[(DATASET, (3, 4, 3, 20))]),          # test calls Dataset
        _doc('core/consumer.py',
             defs=[(CONSUMER, (0, 0, 5, 0), 'consume')],
             refs=[(DATASET, (2, 4, 2, 20))]),          # non-test calls Dataset
    )))
    # otherlib: a same-named test that does NOT call mysrc's Dataset.
    graph.add_source('otherlib', language='python', index=ScipIndex(documents=(
        _doc('tests/test_dataset_utils.py',
             defs=[(OTHER_FN, (0, 0, 5, 0), 'test_util')], refs=[]),
    )))
    graph.materialize()
    library = Library(tmp_path / 'scip.db')
    with library._conn_provider.acquire() as conn:
        graph.save_to(conn)
    yield library
    library.close()


class TestScipDependentFiles:
    def test_tests_only_returns_the_calling_test_file(self, lib: Library) -> None:
        got = lib.scip_dependent_files('mysrc', 'core/dataset.py', tests_only=True)
        assert got == {'tests/test_dataset.py'}

    def test_without_tests_only_includes_nontest_callers(self, lib: Library) -> None:
        got = lib.scip_dependent_files('mysrc', 'core/dataset.py', tests_only=False)
        assert got == {'tests/test_dataset.py', 'core/consumer.py'}

    def test_unknown_file_is_empty(self, lib: Library) -> None:
        assert lib.scip_dependent_files('mysrc', 'core/ghost.py', tests_only=True) == set()
