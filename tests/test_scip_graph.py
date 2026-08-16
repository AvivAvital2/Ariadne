"""Building symbols and edges from the SCIP model.

Construction is a pure function of an index — no connection, no ordering, nothing a caller
can forget to invoke. That matters: the defect this replaces left ``doc_graph`` holding 769
imports and 491 ``scip_calls`` against 2.47M available edges, because a second step was
never wired up and the pipeline reported success anyway.

Synthetic fixtures only.
"""
from __future__ import annotations

import pytest

from docgen.scip_graph import build_rows, classify_edge
from docgen.scip_index import (
    ScipDocument,
    ScipIndex,
    ScipOccurrence,
    ScipRelationship,
    ScipSymbolInfo,
)

RUN = 'scip-python python src1 0.1 `m`/run().'
HELPER = 'scip-python python src1 0.1 `m`/helper().'
IMPL = 'scip-python python src1 0.1 `m`/Impl#'
PROTO = 'scip-python python src1 0.1 `m`/Proto#'


def _occ(symbol, line, *, definition=False, enclosing=None):
    return ScipOccurrence(symbol=symbol, range=(line, 4, line, 12),
                          is_definition=definition, enclosing_range=enclosing or ())


def _index(*docs):
    return ScipIndex(documents=tuple(docs)).scoped_to('src1')


def _rows(*docs):
    return build_rows(_index(*docs), source_name='src1', language='python')


class TestClassifyEdge:
    """An edge is typed by what it points at (`bff2dec`)."""

    @pytest.mark.parametrize('callee,expected', [
        ('p/foo().', 'call'),
        ('p/foo(+1).', 'call'),          # an overload keeps its disambiguator
        ('p/Cls#<init>(+2).', 'call'),   # Scala constructors are always `<init>(+N).`
        ('p/Cls#', 'type_ref'),
        ('p/module.', 'type_ref'),
        ('local src1:a.py:1', 'type_ref'),
        ('plain_name', 'call'),          # synthetic ids stay calls
    ])
    def test_the_moniker_grammar_decides(self, callee, expected):
        assert classify_edge(callee) == expected


class TestSymbols:
    def test_a_definition_carries_its_body_extent(self):
        rows = _rows(ScipDocument('m.py', occurrences=(
            _occ(RUN, 4, definition=True, enclosing=(4, 0, 12, 0)),
        )))

        symbol = rows.symbols[RUN]
        assert (symbol.line_start, symbol.line_end) == (5, 13)
        assert symbol.file == 'm.py'
        assert symbol.source_name == 'src1'

    def test_an_unspecified_kind_is_stored_as_empty_not_as_a_kind_name(self):
        rows = _rows(ScipDocument('m.py',
                                  occurrences=(_occ(RUN, 1, definition=True),),
                                  symbols=(ScipSymbolInfo(symbol=RUN,
                                                          kind='UnspecifiedKind'),)))

        assert rows.symbols[RUN].kind == ''
        assert rows.symbols[RUN].display_name == 'run'

    def test_a_parameter_definition_is_not_promoted_to_a_symbol(self):
        param = 'scip-python python src1 0.1 `m`/run().(self)'
        rows = _rows(ScipDocument('m.py', occurrences=(
            _occ(RUN, 1, definition=True),
            _occ(param, 1, definition=True),
        )))

        assert param not in rows.symbols
        assert RUN in rows.symbols


class TestEdges:
    def test_a_call_inside_a_body_attributes_to_the_enclosing_definition(self):
        rows = _rows(ScipDocument('m.py', occurrences=(
            _occ(RUN, 4, definition=True, enclosing=(4, 0, 20, 0)),
            _occ(HELPER, 30, definition=True, enclosing=(30, 0, 34, 0)),
            _occ(HELPER, 10),   # called from inside run()'s body
        )))

        assert [(e.caller.canonical_id, e.callee.canonical_id, e.edge_type)
                for e in rows.edges] == [(RUN, HELPER, 'call')]
        assert rows.edges[0].line == 11

    def test_the_innermost_definition_wins(self):
        inner = 'scip-python python src1 0.1 `m`/run().inner().'
        rows = _rows(ScipDocument('m.py', occurrences=(
            _occ(RUN, 1, definition=True, enclosing=(1, 0, 40, 0)),
            _occ(inner, 10, definition=True, enclosing=(10, 0, 20, 0)),
            _occ(HELPER, 30, definition=True, enclosing=(30, 0, 32, 0)),
            _occ(HELPER, 15),   # inside inner(), which is inside run()
        )))

        callers = {edge.caller.canonical_id for edge in rows.edges if edge.edge_type == 'call'}
        assert callers == {inner}

    def test_a_callee_defined_outside_the_corpus_is_counted_not_invented(self):
        rows = _rows(ScipDocument('m.py', occurrences=(
            _occ(RUN, 1, definition=True, enclosing=(1, 0, 9, 0)),
            _occ('semanticdb maven org.other lib 1.0 org/other/Absent#', 5),
        )))

        assert rows.edges == []
        assert rows.unresolved_callees == 1

    def test_a_call_site_outside_every_body_is_counted_not_dropped_silently(self):
        rows = _rows(ScipDocument('m.py', occurrences=(
            _occ(RUN, 20, definition=True, enclosing=(20, 0, 24, 0)),
            _occ(HELPER, 1, definition=True, enclosing=(1, 0, 3, 0)),
            _occ(HELPER, 10),   # module level: inside no definition
        )))

        assert rows.edges == []
        assert rows.unattributed_sites == 1


class TestImplements:
    """The relation the store never held."""

    def test_an_implementation_relationship_becomes_an_implements_edge(self):
        rows = _rows(ScipDocument('m.py',
            occurrences=(
                _occ(IMPL, 1, definition=True, enclosing=(1, 0, 9, 0)),
                _occ(PROTO, 20, definition=True, enclosing=(20, 0, 24, 0)),
            ),
            symbols=(ScipSymbolInfo(
                symbol=IMPL, kind='Class',
                relationships=(ScipRelationship(symbol=PROTO,
                                                is_implementation=True),)),),
        ))

        implements = [e for e in rows.edges if e.edge_type == 'implements']
        assert [(e.caller.canonical_id, e.callee.canonical_id) for e in implements] == [
            (IMPL, PROTO)]
        # sited at the implementor's own definition, since a relationship has no occurrence
        assert (implements[0].file, implements[0].line) == ('m.py', 2)

    def test_an_interface_outside_the_corpus_is_counted_not_invented(self):
        rows = _rows(ScipDocument('m.py',
            occurrences=(_occ(IMPL, 1, definition=True, enclosing=(1, 0, 9, 0)),),
            symbols=(ScipSymbolInfo(
                symbol=IMPL, kind='Class',
                relationships=(ScipRelationship(symbol='pydantic/BaseModel#',
                                                is_implementation=True),)),),
        ))

        assert [e for e in rows.edges if e.edge_type == 'implements'] == []
        assert rows.unresolved_callees == 1


class TestDocumentIsolation:
    def test_a_local_carries_only_its_own_document_s_edges(self):
        """The fusion, at the level that produced 499,249 cross-source edges.

        Cross-*file* calls are normal and must stay legal — `a.py` calling a helper
        defined in `b.py` is ordinary. What must not happen is one local row collecting
        both documents' call sites, so a caller that is a local must name the file whose
        site it was attributed from.
        """
        rows = _rows(
            ScipDocument('a.py', occurrences=(
                _occ('local 1', 5, definition=True),
                _occ(HELPER, 6),
            )),
            ScipDocument('b.py', occurrences=(
                _occ('local 1', 5, definition=True),
                _occ(HELPER, 40, definition=True, enclosing=(40, 0, 44, 0)),
                _occ(RUN, 41),
            )),
        )

        locals_ = sorted(cid for cid in rows.symbols if cid.startswith('local '))
        assert locals_ == ['local src1:a.py:1', 'local src1:b.py:1']
        for edge in rows.edges:
            if edge.caller.canonical_id.startswith('local '):
                assert edge.file in edge.caller.canonical_id, (
                    f'{edge.caller.canonical_id} took a site from {edge.file}')

    def test_building_is_pure_so_the_same_index_yields_the_same_rows(self):
        doc = ScipDocument('m.py', occurrences=(
            _occ(RUN, 1, definition=True, enclosing=(1, 0, 9, 0)),
            _occ(HELPER, 20, definition=True, enclosing=(20, 0, 24, 0)),
            _occ(HELPER, 5),
        ))

        first = _rows(doc)
        second = _rows(doc)
        assert first.symbols.keys() == second.symbols.keys()
        assert [(e.caller.canonical_id, e.callee.canonical_id, e.line)
                for e in first.edges] == [
            (e.caller.canonical_id, e.callee.canonical_id, e.line)
            for e in second.edges]


class TestLocalsAreNotCallers:
    """`dc138d3`'s rule, narrowed to what it should have been: a local is not a scope.

    Give a local a body span and it beats the enclosing method on every in-body call,
    because `_tightest_enclosing` prefers the innermost scope and the local's start is
    larger. scip-java emits locals densely, so this is the common case, not an edge case.

    A local is not a caller either, so it no longer competes for a call site that a
    callable encloses -- see `TestCallSitesAttributeToACallable`, which owns the
    same-line case and the module-level fallback.
    """

    def test_a_call_below_a_local_still_attributes_to_the_method(self):
        rows = _rows(ScipDocument('m.py', occurrences=(
            _occ(RUN, 4, definition=True, enclosing=(4, 0, 20, 0)),
            _occ('local 1', 6, definition=True),   # no body of its own
            _occ(HELPER, 30, definition=True, enclosing=(30, 0, 34, 0)),
            _occ(HELPER, 10),                      # below the local, inside run()
        )))

        assert [e.caller.canonical_id for e in rows.edges] == [RUN]


class TestCallSitesAttributeToACallable:
    """A call attributes to the function it sits in, never to a variable on its line.

    ``x = foo()`` is the most common call shape there is, and the local ``x`` is defined
    on the same line as the call. While locals compete as attribution candidates the
    local wins -- its start line is larger than the enclosing function's, and
    ``_tightest_enclosing`` prefers the innermost scope -- so the edge becomes
    ``x -> foo`` and ``callers_of(foo)`` never returns the function.

    Measured on the rebuilt graph: 160,012 of 777,182 call edges (20.6%) were owned by a
    local, and in an ariadne sample 84.4% of those were SEVERED -- the enclosing callable
    had no edge to that callee at all -- while every one of those sites sat inside a real
    function. ``check_tests.main`` calling ``run_pytest()`` was one of them.

    Stage one of the north star walks these edges. A hop owned by a variable cannot be
    rendered as "the function that calls this", and a local carries only its identifier
    line, so stage two has no body to quote either.

    A local stays the fallback for a site no named definition encloses -- module-level
    ``X = foo()`` -- which is the case that made locals candidates to begin with.
    """

    def test_a_call_beside_a_local_attributes_to_the_enclosing_function(self):
        """def run():  /  x = helper()  -- the edge is run -> helper, not x -> helper."""
        caller_doc = ScipDocument('m.py', occurrences=(
            _occ(RUN, 0, definition=True, enclosing=(0, 0, 4, 0)),
            _occ('local 1', 2, definition=True),
            _occ(HELPER, 2),
        ))
        callee_doc = ScipDocument('h.py', occurrences=(
            _occ(HELPER, 0, definition=True, enclosing=(0, 0, 2, 0)),
        ))

        rows = _rows(caller_doc, callee_doc)
        calls = [e for e in rows.edges if e.edge_type == 'call']

        assert {e.caller.canonical_id for e in calls} == {RUN}, (
            'the enclosing function owns the call; a local on the same line must not '
            f'take it: {[e.caller.canonical_id for e in calls]}'
        )

    def test_a_call_with_no_enclosing_callable_still_attributes_to_the_local(self):
        """Module-level ``X = helper()`` has no callable around it -- the local keeps the
        dependency rather than the reference being dropped."""
        caller_doc = ScipDocument('m.py', occurrences=(
            _occ('local 1', 0, definition=True),
            _occ(HELPER, 0),
        ))
        callee_doc = ScipDocument('h.py', occurrences=(
            _occ(HELPER, 0, definition=True, enclosing=(0, 0, 2, 0)),
        ))

        rows = _rows(caller_doc, callee_doc)
        calls = [e for e in rows.edges if e.edge_type == 'call']

        assert len(calls) == 1, f'the dependency must survive: {calls}'
        assert calls[0].caller.canonical_id.startswith('local '), (
            'with no callable enclosing the site the local is the only owner left, so '
            f'it keeps the edge: {calls[0].caller.canonical_id}'
        )
def test_canonical_owner_materializes_one_exact_contains_edge():
    owner_type = "semanticdb maven . . pkg/Owner#"
    owner_term = "semanticdb maven . . pkg/Owner."
    member = "semanticdb maven . . pkg/Owner#run()."
    rows = _rows(ScipDocument(
        "Owner.scala",
        occurrences=(
            _occ(owner_type, 1, definition=True, enclosing=(1, 0, 10, 0)),
            _occ(owner_term, 12, definition=True, enclosing=(12, 0, 16, 0)),
            _occ(member, 4, definition=True, enclosing=(4, 0, 8, 0))),
        symbols=(ScipSymbolInfo(
            symbol=member, kind="Method",
            enclosing_symbol=owner_type),)))

    contains = [edge for edge in rows.edges if edge.edge_type == "contains"]

    assert [
        (edge.caller.canonical_id, edge.callee.canonical_id)
        for edge in contains
    ] == [(owner_type, member)]
def test_global_descriptor_ownership_needs_no_symbol_information():
    owner_type = "semanticdb maven . . pkg/Owner#"
    owner_term = "semanticdb maven . . pkg/Owner."
    type_member = "semanticdb maven . . pkg/Owner#run()."
    term_member = "semanticdb maven . . pkg/Owner.make()."
    rows = _rows(ScipDocument(
        "Owner.scala",
        occurrences=(
            _occ(owner_type, 1, definition=True, enclosing=(1, 0, 10, 0)),
            _occ(type_member, 4, definition=True, enclosing=(4, 0, 8, 0)),
            _occ(owner_term, 12, definition=True, enclosing=(12, 0, 20, 0)),
            _occ(term_member, 15, definition=True, enclosing=(15, 0, 18, 0))),
    ))

    contains = {
        (edge.caller.canonical_id, edge.callee.canonical_id)
        for edge in rows.edges if edge.edge_type == "contains"
    }

    assert contains == {
        (owner_type, type_member),
        (owner_term, term_member),
    }
def test_package_descriptor_is_not_a_traversable_owner_hub():
    package = "semanticdb maven . . pkg/"
    left = "semanticdb maven . . pkg/left()."
    right = "semanticdb maven . . pkg/right()."
    rows = _rows(ScipDocument(
        "package.scala",
        occurrences=(
            _occ(package, 1, definition=True, enclosing=(1, 0, 30, 0)),
            _occ(left, 4, definition=True, enclosing=(4, 0, 8, 0)),
            _occ(right, 14, definition=True, enclosing=(14, 0, 18, 0))),
    ))

    assert [
        edge for edge in rows.edges
        if edge.edge_type == "contains" and edge.caller.canonical_id == package
    ] == []
def test_duplicate_definition_occurrences_emit_one_ownership_edge():
    owner = "semanticdb maven . . pkg/Owner#"
    member = "semanticdb maven . . pkg/Owner#run()."
    rows = _rows(ScipDocument(
        "Owner.scala",
        occurrences=(
            _occ(owner, 1, definition=True, enclosing=(1, 0, 12, 0)),
            _occ(member, 4, definition=True, enclosing=(4, 0, 8, 0)),
            _occ(member, 4, definition=True, enclosing=(4, 0, 8, 0))),
    ))

    assert [
        (edge.caller.canonical_id, edge.callee.canonical_id)
        for edge in rows.edges if edge.edge_type == "contains"
    ] == [(owner, member)]
