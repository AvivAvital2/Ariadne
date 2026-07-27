"""SCIP extractor file-filters derive from one authoritative source in
``docgen.scip_languages`` — so they can't drift (the historical ``.cjs`` gap)
or misroute a language to the wrong tree-sitter grammar.

Each extractor binds its ``_*_EXTS`` name directly to the shared constant, so
the identity checks below fail the moment anyone re-hand-rolls a local list.
"""
from __future__ import annotations

from docgen.scip_languages import (
    GO_GRAMMAR_EXTS,
    JS_GRAMMAR_EXTS,
    PY_GRAMMAR_EXTS,
    SCALA_GRAMMAR_EXTS,
)


def test_js_grammar_exts_derived_has_cjs_excludes_vue():
    # Derived from the typescript indexer entry minus the Vue companion case:
    # the JS grammar parses TS leniently (deliberate), but a raw .vue isn't
    # valid JS (its .vue.script.{js,ts} companion is what gets parsed).
    assert JS_GRAMMAR_EXTS == {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}
    assert '.cjs' in JS_GRAMMAR_EXTS      # the historical drift/bug
    assert '.vue' not in JS_GRAMMAR_EXTS


def test_scala_grammar_exts_distinct_from_java():
    assert SCALA_GRAMMAR_EXTS == {'.scala', '.sbt'}
    assert '.java' not in SCALA_GRAMMAR_EXTS   # no conflation with Java
    assert '.kt' not in SCALA_GRAMMAR_EXTS      # nor Kotlin


def test_py_and_go_grammar_exts():
    assert PY_GRAMMAR_EXTS == {'.py'}
    assert GO_GRAMMAR_EXTS == {'.go'}


def test_extractors_bind_to_the_authoritative_sets():
    import docgen.scip_express_route_extractor as ex
    import docgen.scip_go_ast as ga
    import docgen.scip_js_http_client_extractor as jh
    import docgen.scip_process_extractor as pr
    import docgen.scip_scala_http_client_extractor as sh
    import docgen.scip_string_literal_extractor as sl

    # frozenset-using extractors bind the shared object itself (can't drift).
    for mod in (sl, ex, jh, pr):
        assert mod._JS_EXTS is JS_GRAMMAR_EXTS, mod.__name__
    for mod in (sl, pr, sh):
        assert mod._SCALA_EXTS is SCALA_GRAMMAR_EXTS, mod.__name__
    assert sl._PY_EXTS is PY_GRAMMAR_EXTS
    assert ga._GO_EXTS is GO_GRAMMAR_EXTS
    assert sl._GO_EXTS is GO_GRAMMAR_EXTS

    # slick keeps a tuple (str.endswith needs one) but sourced from the same set.
    from docgen.orm_bindings import slick
    assert frozenset(slick._SCALA_EXTS) == SCALA_GRAMMAR_EXTS
