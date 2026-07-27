"""External-reference resolution — removing the cross-store SCIP "wall".

By default a reference to a symbol not *defined* in the indexed corpus is
dropped (``scip_cross_source`` decision #4), so a user repo that calls a spool's
API gets no cross-source edge into the spool — the two indexes assign the same
symbol different canonical ids (package/version), so ``self._symbols`` misses.

With ``materialize(resolve_external_to=<spool sources>)`` such a dropped
reference is resolved by *qualified name* to a unique definition in one of those
sources, and a cross-source edge is emitted (tagged ``confidence='resolved'`` so
it is distinguishable from an exact same-canonical-id SCIP edge). Resolution is
scoped to the named sources and unambiguous-match-only, so it can't invent edges
between unrelated same-named symbols.

Synthetic ScipIndex fixtures — no DB, no indexer run.
"""
from __future__ import annotations

from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
    _ScipSymbol,
)

# A spool defines pyspark.sql.SparkSession; the user repo references it from
# inside app.main.run via a moniker with the SAME descriptor but a DIFFERENT
# version — the canonical-id mismatch that is the wall.
SPARK_DEF = 'scip-python python pyspark 0.1 `pyspark.sql`/SparkSession#'
SPARK_REF = 'scip-python python pyspark 3.5 `pyspark.sql`/SparkSession#'
RUN_DEF = 'scip-python python userrepo 0.1 `app.main`/run().'


def _build_graph() -> CrossSourceGraph:
    spool_doc = _ScipDoc(
        relative_path='pyspark/sql/session.py',
        occurrences=(
            _ScipOccurrence(symbol=SPARK_DEF, range=(0, 0, 5, 0),
                            is_definition=True),
        ),
        symbols=(
            _ScipSymbol(symbol=SPARK_DEF, kind='Class',
                        display_name='SparkSession'),
        ),
    )
    user_doc = _ScipDoc(
        relative_path='app/main.py',
        occurrences=(
            _ScipOccurrence(symbol=RUN_DEF, range=(10, 0, 20, 0),
                            is_definition=True),
            # a reference to SparkSession INSIDE run() — a real call site
            _ScipOccurrence(symbol=SPARK_REF, range=(12, 4, 12, 30),
                            is_definition=False),
        ),
        symbols=(
            _ScipSymbol(symbol=RUN_DEF, kind='Function', display_name='run'),
        ),
    )
    graph = CrossSourceGraph()
    graph.add_source(
        'spool:databricks',
        index=ScipIndex(documents=(spool_doc,)), language='python')
    graph.add_source(
        'userrepo',
        index=ScipIndex(documents=(user_doc,)), language='python')
    return graph


class TestExternalRefResolution:
    def test_resolution_off_drops_external_ref(self) -> None:
        graph = _build_graph()
        graph.materialize()                       # default: decision #4 in force
        assert graph.callers_of(SPARK_DEF) == []

    def test_resolution_bridges_user_ref_to_spool_def(self) -> None:
        graph = _build_graph()
        graph.materialize(resolve_external_to={'spool:databricks'})
        callers = graph.callers_of(SPARK_DEF)
        assert len(callers) == 1
        edge = callers[0]
        assert edge.caller.canonical_id == RUN_DEF
        assert edge.callee.canonical_id == SPARK_DEF
        assert edge.confidence == 'resolved'      # not passed off as exact SCIP

    def test_resolution_scoped_to_named_sources(self) -> None:
        # The qn match exists, but SparkSession's source is not in the
        # resolvable set — so nothing is bridged (no promiscuous matching).
        graph = _build_graph()
        graph.materialize(resolve_external_to={'some-other-source'})
        assert graph.callers_of(SPARK_DEF) == []
