"""A SCIP edge must say whether it is a CALL or a type reference.

SCIP records every non-definition occurrence the same way, so ingest emitted
``edge_type='call'`` for all of them. Measured on the databricks spool: of
273,007 semanticdb edges, 145,817 pointed at a Class, 19,677 at a Trait and
9,633 at a TypeParameter -- none of which can be called. Only ~36k reached a
Method. `callers`, `callees`, `impact_radius` and the doc-graph `scip_calls`
enrichment all read that column and were being handed type annotations as calls.

``trace_flow`` already knew and worked around it locally by sniffing the moniker
suffix; this moves that knowledge to ingest so every consumer benefits and there
is one implementation rather than two.

The discriminator is the moniker's own grammar: a callable ends ``().``, a
class or trait in ``#``, an attribute or module value in a bare ``.``.
"""
from __future__ import annotations

from docgen.scip_cross_source import classify_edge

SCALA = 'semanticdb maven . . org/apache/spark/sql/delta/'


class TestClassifyEdge:
    def test_method_is_a_call(self):
        assert classify_edge(f'{SCALA}commands/merge/ClassicMergeExecutor#writeAllChanges().') == 'call'

    def test_overloaded_method_is_a_call(self):
        assert classify_edge(f'{SCALA}util/MetricDefinition#`<init>`(+1).') == 'call'

    def test_class_is_a_type_reference(self):
        assert classify_edge(f'{SCALA}util/UsageRecord#') == 'type_ref'

    def test_trait_is_a_type_reference(self):
        assert classify_edge(f'{SCALA}commands/merge/ClassicMergeExecutor#') == 'type_ref'

    def test_module_value_is_a_type_reference(self):
        """A bare trailing dot is an attribute or object value, not an invocation."""
        assert classify_edge(f'{SCALA}util/MetricDefinitions.EVENT_LOGGING_FAILURE.') == 'type_ref'

    def test_local_is_not_a_call(self):
        assert classify_edge('local 12') == 'type_ref'

    def test_python_callable(self):
        assert classify_edge('scip-python python databricks-sdk-py 0.1 `databricks.sdk`/_make_dbutil().') == 'call'

    def test_non_scip_synthetic_id_is_kept_as_a_call(self):
        """Plain-name graphs (no moniker grammar) must keep working."""
        assert classify_edge('audit.record') == 'call'


class TestEdgeConstructionUsesIt:
    def test_edges_are_typed_by_their_callee(self):
        """The classifier must be wired into construction, not just unit-tested.

        Pinned by BEHAVIOUR, not by grepping a method's source. The previous version read
        `inspect.getsource(CrossSourceGraph._collect_edges)` and asserted the text
        contained `classify_edge` — which broke the moment construction moved, while
        telling us nothing about whether edges are actually typed correctly. An ingest
        that computed the right label and then wrote a constant is exactly what this must
        catch, and a real edge check catches it.
        """
        from docgen.scip_graph import build_rows
        from docgen.scip_index import ScipDocument, ScipIndex, ScipOccurrence

        method = 'scip-python python src1 0.1 `m`/run().'
        cls = 'scip-python python src1 0.1 `m`/Config#'
        caller = 'scip-python python src1 0.1 `m`/caller().'

        rows = build_rows(
            ScipIndex(documents=(ScipDocument('m.py', occurrences=(
                ScipOccurrence(symbol=caller, range=(0, 0, 0, 6),
                               is_definition=True, enclosing_range=(0, 0, 20, 0)),
                ScipOccurrence(symbol=method, range=(30, 0, 30, 5),
                               is_definition=True, enclosing_range=(30, 0, 34, 0)),
                ScipOccurrence(symbol=cls, range=(40, 0, 40, 6),
                               is_definition=True, enclosing_range=(40, 0, 44, 0)),
                # both referenced from inside caller()'s body
                ScipOccurrence(symbol=method, range=(5, 4, 5, 9)),
                ScipOccurrence(symbol=cls, range=(6, 4, 6, 10)),
            )),)),
            source_name='src1', language='python')

        by_callee = {e.callee.canonical_id: e.edge_type for e in rows.edges}
        assert by_callee[method] == 'call', 'a callable target is a call'
        assert by_callee[cls] == 'type_ref', 'a class target is not a call'
