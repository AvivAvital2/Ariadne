"""Clews get their own vector surface, not a document content type.

A clew is a pre-generated route through the call graph — ``fit -> withTransformEvent ->
listenerBus -> getOrCreate`` — embedded so a question can match a *path* instead of a
starting symbol. ``scip_edges`` carries 2.78M edges and no embedding column, which is why the
walk has never seen the question: retrieval can only match a symbol.

Storing them as ``documents`` with a new ``content_type`` was the cheap route and it is wrong.
Twenty-two files enumerate content types, so a clew would inherit provenance weighting, gap
analysis, doc-type pickers and export — and any filter that lists types drops it silently. A
route is not prose and must not compete with prose in one ranked list.

So clews follow ``sections``: their own table, their own ``embedding`` column, queried
deliberately. That is the existing pattern for a non-document vector surface, not a new one.

Synthetic fixtures only: source ``src1``.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from library import Library; 

from library.chain_answer import evidence_for; 

from library.scip import init_scip_schema; 

from library.clews import Clew, add_clew, clews_for, init_clews_schema, nearest_clews, nearest_clew_matches, select_clew_matches

SOURCE = 'src1'


@pytest.fixture()
def conn():
    connection = sqlite3.connect(':memory:')
    connection.row_factory = sqlite3.Row
    init_clews_schema(connection)
    yield connection
    connection.close()


def _clew(entry: str, steps: list[str], files: list[str], vector=None) -> dict:
    return {
        'source_name': SOURCE,
        'entry_symbol': entry,
        'steps': steps,
        'route': [f'pkg.{step}' for step in steps],
        'files': files,
        'strategy': 'theme',
        'embedding': None if vector is None else np.asarray(vector, dtype=np.float32),
    }


class TestAClewIsItsOwnSurface:
    """Stored, scoped and retrieved without touching ``documents``."""

    def test_a_stored_clew_returns_its_route_and_the_files_to_seed_from(self, conn):
        add_clew(conn, **_clew('pkg.run', ['run', 'load', 'write'], ['a.py', 'b.py']))

        stored = clews_for(conn, source_name=SOURCE)

        assert len(stored) == 1
        clew = stored[0]
        assert isinstance(clew, Clew)
        assert clew.steps == ['run', 'load', 'write']
        assert clew.route == ['pkg.run', 'pkg.load', 'pkg.write']
        assert clew.files == ['a.py', 'b.py'], 'the files are what seeds a walk'
        assert clew.hops == 2, 'hops is derived from the route, never stored twice'

    def test_a_clew_belongs_to_one_source(self, conn):
        add_clew(conn, **_clew('pkg.run', ['run', 'load', 'write'], ['a.py']))
        other = _clew('pkg.other', ['other', 'load', 'write'], ['c.py'])
        other['source_name'] = 'src2'
        add_clew(conn, **other)

        assert len(clews_for(conn, source_name=SOURCE)) == 1
        assert len(clews_for(conn, source_name='src2')) == 1

    def test_the_same_route_stored_twice_is_one_clew(self, conn):
        """Generation pools several strategies, and they overlap by design.

        Measured on the databricks pack, pooling three strategies is worth +14 points of
        coverage over the best single one — so the same route arrives more than once and the
        store, not the generator, has to be the thing that dedupes.
        """
        add_clew(conn, **_clew('pkg.run', ['run', 'load', 'write'], ['a.py']))
        add_clew(conn, **_clew('pkg.run', ['run', 'load', 'write'], ['a.py']))

        assert len(clews_for(conn, source_name=SOURCE)) == 1

    def test_nearest_returns_the_closest_route_by_cosine(self, conn):
        add_clew(conn, **_clew('pkg.near', ['near', 'load', 'write'], ['a.py'],
                               vector=[1.0, 0.0, 0.0]))
        add_clew(conn, **_clew('pkg.far', ['far', 'load', 'write'], ['b.py'],
                               vector=[0.0, 1.0, 0.0]))

        found = nearest_clews(conn, np.asarray([0.9, 0.1, 0.0], dtype=np.float32),
                              source_name=SOURCE, top_k=1)

        assert [c.entry_symbol for c in found] == ['pkg.near']

    def test_a_clew_without_an_embedding_is_never_returned_as_nearest(self, conn):
        """An unembedded clew is not yet usable, and must not be silently ranked as distant.

        Embedding happens after generation and needs a provider key, so a pack can hold
        clews that are stored but not embedded. Treating a missing vector as a zero vector
        would make them the answer to every query.
        """
        add_clew(conn, **_clew('pkg.unembedded', ['unembedded', 'load', 'write'], ['a.py']))

        found = nearest_clews(conn, np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                              source_name=SOURCE, top_k=5)

        assert found == []
    def test_ranked_matches_keep_routes_scores_threshold_and_entry_diversity(self, conn):
        add_clew(conn, **_clew('pkg.entry', ['entry', 'best', 'done'], ['a.py'],
                               vector=[1.0, 0.0, 0.0]))
        add_clew(conn, **_clew('pkg.entry', ['entry', 'duplicate', 'done'], ['b.py'],
                               vector=[0.99, 0.01, 0.0]))
        add_clew(conn, **_clew('pkg.other', ['other', 'useful', 'done'], ['c.py'],
                               vector=[0.8, 0.2, 0.0]))
        add_clew(conn, **_clew('pkg.noise', ['noise', 'wrong', 'done'], ['d.py'],
                               vector=[0.0, 1.0, 0.0]))
        add_clew(conn, **_clew('pkg.zero', ['zero', 'empty', 'done'], ['e.py'],
                               vector=[0.0, 0.0, 0.0]))
        add_clew(conn, **_clew('pkg.wrongdim', ['wrongdim', 'bad', 'done'], ['f.py'],
                               vector=[1.0, 0.0]))

        matches = nearest_clew_matches(
            conn, np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            source_name=SOURCE, top_k=3, min_similarity=0.5)

        assert [match.clew.entry_symbol for match in matches] == ['pkg.entry', 'pkg.other']
        assert matches[0].similarity > matches[1].similarity >= 0.5
        assert matches[0].clew.route == ['pkg.entry', 'pkg.best', 'pkg.done']
        assert all(match.clew.entry_symbol != 'pkg.noise' for match in matches)
        assert nearest_clew_matches(
            conn, np.zeros(3, dtype=np.float32), source_name=SOURCE) == []
        assert nearest_clew_matches(
            conn, np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            source_name=SOURCE, top_k=0) == []
        one = nearest_clew_matches(
            conn, np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            source_name=SOURCE, top_k=1, min_similarity=0.5)
        assert [match.clew.entry_symbol for match in one] == ['pkg.entry']
        default_matches = nearest_clew_matches(
            conn, np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            source_name=SOURCE, top_k=5)
        assert all(match.clew.entry_symbol != 'pkg.noise' for match in default_matches)
        assert all(match.clew.entry_symbol not in {'pkg.zero', 'pkg.wrongdim'}
                   for match in default_matches)
        selection = select_clew_matches('How does pkg.useful commit?', matches)
        assert [match.clew.entry_symbol for match in selection.accepted] == ['pkg.other']
        assert [rejection.match.clew.entry_symbol for rejection in selection.rejected] == ['pkg.entry']
        assert selection.rejected[0].reason == 'question entity mismatch'
        anchorless = select_clew_matches('How does this work?', matches)
        assert anchorless.accepted == matches and anchorless.rejected == []
        unresolved = select_clew_matches('How does `missing.route` work?', matches)
        assert unresolved.accepted == matches and unresolved.rejected == []
RUN = 'scip-python python src1 0.1 `m`/run().'
HELPER = 'scip-python python src1 0.1 `m`/helper().'


def _symbol(conn, cid, *, file, qn, line_start, line_end, parent=''):
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
        'line_start, line_end, kind, display_name, qualified_name, '
        'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (cid, SOURCE, 'python', file, line_start, line_end, '', '', qn, parent))
def _edge(conn, caller, callee, *, line, file='m.py'):
    conn.execute(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
        "edge_type, file, line, confidence) VALUES (?,?,'call',?,?,'exact')",
        (caller, callee, file, line))
class TestAClewCanPositionTheWalk:
    """A matched route seeds the walk. That is the whole point of storing clews.

    Retrieval today hands ``evidence_for`` documents, and their files become seeds — so a
    question can only influence *where the walk starts* through whatever prose retrieval
    matched. A clew is a route the walk already found, so passing its symbols seeds the walk
    directly on the path. Measured on the databricks pack, pooling clew strategies contains
    92.8% of the symbols answer keys require against 66.0% for one document-seeded walk.

    This asserts the mechanism at its strongest: **no documents at all**, and the chain still
    exists because the clew positioned it.
    """

    def test_clew_symbols_seed_the_walk_when_retrieval_returned_nothing(self, tmp_path):
        library = Library(tmp_path / 'clew.db')
        with library._conn_provider.acquire() as conn:
            init_scip_schema(conn)
            _symbol(conn, RUN, file='m.py', qn='m.run', line_start=5, line_end=20)
            _symbol(conn, HELPER, file='h.py', qn='m.helper', line_start=3, line_end=9)
            _edge(conn, RUN, HELPER, line=8)
            conn.commit()

        without = evidence_for(library, [], source=SOURCE)
        with_clew = evidence_for(library, [], source=SOURCE, clew_symbols=['m.run'])

        assert without.spine == '', 'no documents and no clew means no chain'
        assert with_clew.spine, 'the clew alone must be able to seed the walk'
        assert 'm.helper' in with_clew.spine, (
            'the walk continues from where the clew put it')
def test_select_clews_rejects_nonproduction_routes_before_entity_matching():
    from library.clews import ClewMatch
    production = ClewMatch(clew=Clew(
        id="prod", source_name=SOURCE, entry_symbol="delta.DeltaSink.addBatch",
        route=["delta.DeltaSink.addBatch", "delta.OptimisticTransaction.commit"],
        files=["spark/src/main/scala/delta/DeltaSink.scala"]), similarity=0.8)
    suite = ClewMatch(clew=Clew(
        id="suite", source_name=SOURCE, entry_symbol="delta.DeltaSinkSuite.beforeAll",
        route=["delta.DeltaSinkSuite.beforeAll", "delta.DeltaSinkSuite.runQuery"],
        files=["spark/src/test/scala/delta/DeltaSinkSuite.scala"]), similarity=0.95)
    generated = ClewMatch(clew=Clew(
        id="generated", source_name=SOURCE, entry_symbol="proto.StreamingBatch",
        route=["proto.StreamingBatch", "proto.StreamingBatch.getBatchId"],
        files=["spark-connect/common/target/generated/StreamingBatch.java"]), similarity=0.9)

    selected = select_clew_matches(
        "How does DeltaSink preserve a streaming batch?",
        [suite, generated, production])

    assert selected.accepted == [production]
    assert {item.reason for item in selected.rejected} == {"non-production route"}
def test_clew_route_menu_keeps_semantic_symbols_on_each_route_card():
    from library.clews import ClewMatch, clew_route_menu, resolve_clew_routes
    first = ClewMatch(clew=Clew(
        id="a", source_name=SOURCE, entry_symbol="pkg.Entry.run",
        route=["pkg.Entry.run", "pkg.Store.commit"],
        files=["entry.py", "store.py"]), similarity=0.9)
    second = ClewMatch(clew=Clew(
        id="b", source_name=SOURCE, entry_symbol="pkg.Entry.run",
        route=["pkg.Entry.run", "pkg.Cache.read"],
        files=["entry.py", "cache.py"]), similarity=0.8)

    menu = clew_route_menu([first, second])
    assert menu.text.count("pkg.Entry.run") == 2
    assert menu.text.count("pkg.Store.commit") == 1
    assert menu.text.count("pkg.Cache.read") == 1
    assert "K1. pkg.Entry.run -> pkg.Store.commit" in menu.text
    assert "K2. pkg.Entry.run -> pkg.Cache.read" in menu.text
    assert resolve_clew_routes(menu, "K2, K99, K1", limit=1) == [second]
def test_lexical_clew_recall_works_without_embedding_provider(conn):
    from library.clews import lexical_clew_matches
    add_clew(conn, **_clew(
        "delta.sink", ["sink", "detectCommittedBatch", "skipReplay"],
        ["DeltaSink.scala"]),
        question="How does the sink skip a replayed committed batch?")
    add_clew(conn, **_clew(
        "delta.schema", ["schema", "castColumns", "updateMetadata"],
        ["Schema.scala"]),
        question="How are schema columns cast?")

    found = lexical_clew_matches(
        conn, "Why does a replayed batch get skipped after it was committed?",
        source_name=SOURCE, top_k=2)

    assert found[0].clew.entry_symbol == "delta.sink"
    assert found[0].similarity > found[1].similarity
def test_clew_family_menu_groups_routes_by_entry_owner_before_route_selection():
    from library.clews import ClewMatch, clew_family_menu, resolve_clew_families
    matches = [
        ClewMatch(clew=Clew(
            id="a", source_name=SOURCE, entry_symbol="pkg.Merge.run",
            route=["pkg.Merge.run", "pkg.Merge.write"], files=["a.py"]), similarity=0.8),
        ClewMatch(clew=Clew(
            id="b", source_name=SOURCE, entry_symbol="pkg.Merge.check",
            route=["pkg.Merge.check", "pkg.Merge.validate"], files=["a.py"]), similarity=0.7),
        ClewMatch(clew=Clew(
            id="c", source_name=SOURCE, entry_symbol="pkg.Reader.read",
            route=["pkg.Reader.read", "pkg.Reader.filter"], files=["b.py"]), similarity=0.6)]

    menu = clew_family_menu(matches, limit=10)
    selected = resolve_clew_families(menu, "F2, F99", limit=2)

    assert len(menu.labels) == 2
    assert "pkg.Merge" in menu.text and "2 route(s)" in menu.text
    assert [match.clew.id for match in selected] == ["c"]
def test_selected_clew_families_are_interleaved_before_bounding():
    from library.clews import ClewMatch, clew_family_menu, resolve_clew_families
    matches = []
    for index in range(70):
        matches.append(ClewMatch(clew=Clew(
            id=f"large-{index}", source_name=SOURCE, entry_symbol=f"pkg.Large.r{index}",
            route=[f"pkg.Large.r{index}"], files=["large.py"]), similarity=1.0))
    matches.append(ClewMatch(clew=Clew(
        id="small-0", source_name=SOURCE, entry_symbol="pkg.Small.read",
        route=["pkg.Small.read"], files=["small.py"]), similarity=0.5))

    menu = clew_family_menu(matches)
    selected = resolve_clew_families(menu, "F1 F2")

    assert [match.clew.id for match in selected[:3]] == ["large-0", "small-0", "large-1"]
    assert "small-0" in [match.clew.id for match in selected[:64]]
def test_obligation_coverage_requires_a_valid_route_on_the_same_line():
    from library.clews import ClewMatch, ClewRouteMenu, covered_route_obligations
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.run", route=["pkg.run"], files=["a.py"]), similarity=1.0)
    menu = ClewRouteMenu(text="", labels={"K1": match})

    covered = covered_route_obligations(menu, "C1: K1\nC2: no route\nC3: K99")

    assert covered == {1}
def test_obligation_coverage_rejects_one_route_reused_for_every_claim():
    from library.clews import ClewMatch, ClewRouteMenu, covered_route_obligations
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.run", route=["pkg.run"], files=["a.py"]), similarity=1.0)
    menu = ClewRouteMenu(labels={"K1": match})

    covered = covered_route_obligations(menu, "C1: K1\nC2: K1\nC3: K1")

    assert covered == set()
def test_family_selection_is_completed_to_the_obligation_count():
    from library.clews import (ClewMatch, clew_family_menu,
                               complete_clew_family_selection)
    matches = [ClewMatch(clew=Clew(id=str(i), source_name=SOURCE,
        entry_symbol=f"pkg.Owner{i}.run", route=[f"pkg.Owner{i}.run"],
        files=[f"{i}.py"]), similarity=1.0 - i / 10) for i in range(4)]
    menu = clew_family_menu(matches)

    selected = complete_clew_family_selection(menu, "F1", minimum=3)

    assert len({match.clew.entry_symbol.rsplit(".", 1)[0]
                for match in selected}) == 3
def test_route_selection_preserves_obligation_ids_on_each_clew():
    from library.clews import ClewMatch, ClewRouteMenu, resolve_clew_routes
    first = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.a", route=["pkg.a"], files=["a.py"]), similarity=1.0)
    second = ClewMatch(clew=Clew(id="b", source_name=SOURCE,
        entry_symbol="pkg.b", route=["pkg.b"], files=["b.py"]), similarity=0.9)
    menu = ClewRouteMenu(labels={"K1": first, "K2": second})

    selected = resolve_clew_routes(menu, "C1: K1\nC2: K1 K2", limit=4)

    assert selected[0].obligations == (1, 2)
    assert selected[1].obligations == (2,)
def test_owner_plan_text_is_attached_to_selected_obligation_routes():
    from library.clews import ClewMatch, attach_obligation_texts
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.a", route=["pkg.a"], files=["a.py"]),
        similarity=1.0, obligations=(2,))

    enriched = attach_obligation_texts([match], "C1 first stage: F1\nC2 stable identifier propagation: F2")

    assert enriched[0].obligation_texts == ((2, "C2 stable identifier propagation: F2"),)
def test_symbol_target_reply_is_attached_only_to_its_obligation():
    from library.clews import (ClewMatch, ClewSymbolMenu,
                               attach_symbol_targets)
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Entry", route=["pkg.Entry"], files=["a.py"]),
        similarity=1.0, obligations=(1, 2))
    menu = ClewSymbolMenu(labels={"S1": "pkg.Middle", "S2": "pkg.Terminal"})

    selected = attach_symbol_targets([match], menu, "C1: S1\nC2: S2")

    assert selected[0].target_symbols == ((1, "pkg.Middle"), (2, "pkg.Terminal"))


def test_symbol_target_menu_includes_route_owner_members(conn):
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    from library.clews import ClewMatch, clew_symbol_menu
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("entry", SOURCE, "python", "a.py", 1, 2, "", "run",
                  "pkg.Owner.run", "pkg.Owner"))
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("field", SOURCE, "python", "a.py", 3, 3, "", "SHARED_FIELD",
                  "pkg.Owner.SHARED_FIELD", "pkg.Owner"))
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Owner.run", route=["pkg.Owner.run"], files=["a.py"]),
        similarity=1.0)

    menu = clew_symbol_menu(conn, [match], "shared field", source_name=SOURCE)

    assert "pkg.Owner.SHARED_FIELD" in menu.labels.values()
    run_label = next(label for label, name in menu.labels.items()
                     if name == "pkg.Owner.run")
    field_label = next(label for label, name in menu.labels.items()
                       if name == "pkg.Owner.SHARED_FIELD")
    assert menu.text.count("pkg.Owner") >= 2
    assert f"{run_label}. pkg.Owner.run" in menu.text
    assert f"{run_label}. pkg.Owner.run" in menu.text
    assert f"{field_label}. pkg.Owner.SHARED_FIELD" in menu.text
def test_symbol_target_menu_reaches_a_second_hop_endpoint(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    symbols = [("entry", "pkg.Entry.run"), ("middle", "pkg.Middle.passThrough"),
               ("target", "pkg.Terminal.STABLE_ID")]
    for index, (canonical, qualified) in enumerate(symbols, 1):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", f"{canonical}.py", index, index,
                      "", qualified.rsplit(".", 1)[-1], qualified, "pkg"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("entry", "middle", "call", "entry.py", 1, "exact"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("middle", "target", "type_ref", "middle.py", 2, "exact"))
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Entry.run", route=["pkg.Entry.run"], files=["entry.py"]),
        similarity=1.0)

    menu = clew_symbol_menu(conn, [match], "stable identifier", source_name=SOURCE)

    assert "pkg.Terminal.STABLE_ID" in menu.labels.values()
def test_symbol_target_menu_resolves_edge_ids_without_symbol_join(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    for canonical, qualified in (("entry", "pkg.Entry.run"),
                                 ("target", "pkg.Target.finish")):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", "a.py", 1, 1, "",
                      qualified.rsplit(".", 1)[-1], qualified, "pkg"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("entry", "target", "call", "a.py", 1, "exact"))
    statements = []
    conn.set_trace_callback(statements.append)
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Entry.run", route=["pkg.Entry.run"], files=["a.py"]),
        similarity=1.0)

    menu = clew_symbol_menu(conn, [match], "target finish", source_name=SOURCE)

    assert "pkg.Target.finish" in menu.labels.values()
    assert not any("SCIP_EDGES E JOIN SCIP_SYMBOLS" in sql.upper()
                   for sql in statements)
def test_family_menu_examples_include_the_callable_owner_entry():
    from library.clews import ClewMatch, clew_family_menu
    matches = [ClewMatch(clew=Clew(id=f"helper-{i}", source_name=SOURCE,
        entry_symbol=f"pkg.Reconciler.helper{i}",
        route=[f"pkg.Reconciler.helper{i}", f"pkg.Dependency{i}.finish"],
        files=["a.scala"]), similarity=1.0) for i in range(4)]
    entry = ClewMatch(clew=Clew(id="entry", source_name=SOURCE,
        entry_symbol="pkg.Reconciler.apply",
        route=["pkg.Reconciler.apply", "pkg.Reconciler.complete"],
        files=["a.scala"]), similarity=0.1)

    menu = clew_family_menu(
        [*matches, entry], question="how does reconciliation complete", limit=1)

    assert "apply -> complete" in menu.text


def test_family_menu_prioritizes_strong_question_owner_evidence():
    from library.clews import ClewMatch, clew_family_menu
    generic = ClewMatch(clew=Clew(id="generic", source_name=SOURCE,
        entry_symbol="pkg.Utility.run", route=["pkg.Utility.run"], files=["a.py"]),
        similarity=0.99)
    owner = ClewMatch(clew=Clew(id="owner", source_name=SOURCE,
        entry_symbol="pkg.PaymentReconciliation.apply",
        route=["pkg.PaymentReconciliation.apply", "pkg.PaymentReconciliation.finish"],
        files=["b.py"]), similarity=0.2)

    menu = clew_family_menu(
        [generic, owner], question="how does payment reconciliation reach its final path", limit=1)

    assert list(menu.labels.values()) == [(owner,)]
    assert "apply -> finish" in menu.text


def test_pseudo_semantic_recall_expands_lexical_seeds_without_a_query_api(conn):
    import numpy as np
    from library.clews import add_clew, pseudo_semantic_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    add_clew(conn, source_name=SOURCE, entry_symbol="pkg.Payment.start",
             steps=[], route=["pkg.Payment.start"], files=["a.py"],
             question="payment reconciliation", embedding=np.array([1.0, 0.0]))
    add_clew(conn, source_name=SOURCE, entry_symbol="pkg.Ledger.finish",
             steps=[], route=["pkg.Ledger.finish"], files=["b.py"],
             question="ledger completion", embedding=np.array([0.9, 0.1]))
    add_clew(conn, source_name=SOURCE, entry_symbol="pkg.Unrelated.run",
             steps=[], route=["pkg.Unrelated.run"], files=["c.py"],
             question="unrelated operation", embedding=np.array([0.0, 1.0]))

    matches = pseudo_semantic_clew_matches(
        conn, "payment reconciliation", source_name=SOURCE, top_k=3, seed_k=1)

    assert [match.clew.entry_symbol for match in matches][:2] == [
        "pkg.Payment.start", "pkg.Ledger.finish"]


def test_document_files_feed_bounded_scip_routes(conn):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    for canonical, qualified, file in (
            ("root", "billing.Reconciler.apply", "billing.py"),
            ("target", "billing.Ledger.commit", "ledger.py")):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", file, 1, 5, "Method",
                      qualified.rsplit(".", 1)[-1], qualified,
                      qualified.rsplit(".", 1)[0]))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("root", "target", "call", "billing.py", 3, "exact"))

    matches = document_clew_matches(
        conn, [{"title": "billing.Reconciler.apply",
                "source_files": ["billing.py"], "content": ""}],
        "How does reconciliation commit?", source_name=SOURCE, limit=4)

    routes = [match.clew.route for match in matches]
    assert ["billing.Reconciler.apply"] in routes
    assert ["billing.Reconciler.apply", "billing.Ledger.commit"] in routes


def test_family_menu_ranks_semantic_similarity_before_catalog_order():
    from library.clews import ClewMatch, clew_family_menu
    weak = ClewMatch(clew=Clew(id="weak", source_name=SOURCE,
        entry_symbol="pkg.Generic.run", route=["pkg.Generic.run"], files=["a.py"]),
        similarity=0.1)
    strong = ClewMatch(clew=Clew(id="strong", source_name=SOURCE,
        entry_symbol="pkg.Replay.detect", route=["pkg.Replay.detect"], files=["b.py"]),
        similarity=0.9)

    menu = clew_family_menu([weak, strong], question="detect replay", limit=1)

    assert list(menu.labels.values()) == [(strong,)]
    assert "pkg.Replay" in menu.text
    assert "pkg.Generic" not in menu.text
def test_symbol_menu_never_drops_selected_route_symbols(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    rows = []
    for index in range(450):
        qualified = f"pkg.Noise{index}.update"
        rows.append((f"n{index}", SOURCE, "python", "noise.py", index, index,
                     "Method", "update", qualified, f"pkg.Noise{index}"))
    rows.append(("selected", SOURCE, "python", "selected.py", 1, 2,
                 "Method", "apply", "pkg.Selected.apply", "pkg.Selected"))
    conn.executemany("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Selected.apply", route=["pkg.Selected.apply"],
        files=["selected.py"]), similarity=1.0)

    menu = clew_symbol_menu(
        conn, [match], "How is update applied?", source_name=SOURCE, limit=20)

    assert "pkg.Selected.apply" in menu.labels.values()


def test_symbol_targets_are_bounded_to_the_four_structural_roles():
    from library.clews import ClewMatch, ClewSymbolMenu, attach_symbol_targets
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Entry", route=["pkg.Entry"], files=["a.py"]),
        similarity=1.0, obligations=(1,))
    menu = ClewSymbolMenu(labels={f"S{index}": f"pkg.Symbol{index}"
                                  for index in range(1, 8)})

    selected = attach_symbol_targets(
        [match], menu, "C1: S1 S2 S3 S4 S5 S6 S7 S2")

    assert selected[0].target_symbols == tuple(
        (1, f"pkg.Symbol{index}") for index in range(1, 5))
def test_symbol_menu_ranks_matching_owner_type_before_unrelated_members(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    for index, qualified, parent in (
            (1, "pkg.PaymentReconciliation", "pkg"),
            (2, "pkg.PaymentReconciliation.unrelatedHelper", "pkg.PaymentReconciliation"),
            (3, "pkg.PaymentReconciliation.paymentStatus", "pkg.PaymentReconciliation")):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (f"s{index}", SOURCE, "scala", "a.scala", index, index + 1,
                      "Method", qualified.rsplit(".", 1)[-1], qualified, parent))
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.PaymentReconciliation",
        route=["pkg.PaymentReconciliation",
               "pkg.PaymentReconciliation.unrelatedHelper"], files=["a.scala"]),
        similarity=1.0)

    menu = clew_symbol_menu(
        conn, [match], "explain payment reconciliation", source_name=SOURCE)
    ordered = list(menu.labels.values())

    assert ordered.index("pkg.PaymentReconciliation") < ordered.index(
        "pkg.PaymentReconciliation.unrelatedHelper")


def test_symbol_menu_expands_a_selected_owner_type_to_its_members(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    for canonical, qualified, parent in (
            ("owner", "billing.Reconciler", "billing"),
            ("member", "billing.Reconciler.commit", "billing.Reconciler")):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", "billing.py", 1, 5,
                      "Method", qualified.rsplit(".", 1)[-1], qualified, parent))
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="billing.Reconciler", route=["billing.Reconciler"],
        files=["billing.py"]), similarity=1.0)

    menu = clew_symbol_menu(
        conn, [match], "How is reconciliation committed?", source_name=SOURCE)

    assert "billing.Reconciler.commit" in menu.labels.values()


def test_symbol_menu_adds_members_of_entity_ranked_owners(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    for canonical, qualified, parent in (
            ("route", "pkg.Unrelated.run", "pkg.Unrelated"),
            ("ledger-owner", "billing.PaymentLedger", "billing"),
            ("ledger-apply", "billing.PaymentLedger.apply", "billing.PaymentLedger")):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", f"{canonical}.py", 1, 2,
                      "Method", qualified.rsplit(".", 1)[-1], qualified, parent))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("entry", "helper", "call", "rule.scala", 20, "exact"))
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Unrelated.run", route=["pkg.Unrelated.run"],
        files=["route.py"]), similarity=1.0)

    menu = clew_symbol_menu(
        conn, [match], "Explain the PaymentLedger reconciliation transition", source_name=SOURCE)

    assert "billing.PaymentLedger.apply" in menu.labels.values()
def test_selected_owner_type_promotes_its_own_callable_entry(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    for canonical, qualified, parent in (
            ("owner", "pkg.Reconciler", "pkg"),
            ("entry", "pkg.Reconciler.apply", "pkg.Reconciler")):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "scala", "a.scala", 1, 5, "Method",
                      qualified.rsplit(".", 1)[-1], qualified, parent))
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Reconciler", route=["pkg.Reconciler"], files=["a.scala"]),
        similarity=1.0)

    menu = clew_symbol_menu(conn, [match], "explain reconciliation",
                            source_name=SOURCE)

    assert menu.owner_entries["pkg.Reconciler"] == "pkg.Reconciler.apply"


def test_symbol_menu_uses_scala_callable_entry_when_dispatch_has_no_edge(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    for canonical, qualified, start, end in (
            ("helper", "pkg.Rule.helper", 30, 40),
            ("accessor", "pkg.Rule.setting", 5, 100),
            ("entry", "pkg.Rule.apply", 10, 20)):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "scala", "rule.scala", start, end,
                      "Method", qualified.rsplit(".", 1)[-1], qualified, "pkg.Rule"))
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Rule.helper", route=["pkg.Rule.helper"],
        files=["rule.scala"]), similarity=1.0)

    menu = clew_symbol_menu(conn, [match], "explain rule", source_name=SOURCE)

    assert menu.owner_entries["pkg.Rule.helper"] == "pkg.Rule.apply"


def test_symbol_menu_promotes_the_scip_calling_owner_executable(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    for canonical, qualified, parent, start, end in (
            ("helper", "pkg.Rule.helper", "pkg.Rule", 30, 40),
            ("accessor", "pkg.Rule.setting", "pkg.Rule", 5, 100),
            ("entry", "pkg.Rule.orchestrate", "pkg.Rule", 10, 80)):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "scala", "rule.scala", start, end,
                      "Method", qualified.rsplit(".", 1)[-1], qualified, parent))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("entry", "helper", "call", "rule.scala", 20, "exact"))
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Rule.helper", route=["pkg.Rule.helper"],
        files=["rule.scala"]), similarity=1.0)

    menu = clew_symbol_menu(conn, [match], "explain rule", source_name=SOURCE)

    assert menu.owner_entries["pkg.Rule.helper"] == "pkg.Rule.orchestrate"


def test_selected_internal_symbol_promotes_its_scip_owner_entry():
    from library.clews import ClewMatch, ClewSymbolMenu, attach_symbol_targets
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Start", route=["pkg.Start"], files=["a.py"]),
        similarity=1.0, obligations=(1,))
    menu = ClewSymbolMenu(
        labels={"S1": "pkg.Rule.deepHelper"},
        owner_entries={"pkg.Rule.deepHelper": "pkg.Rule.entry"})

    selected = attach_symbol_targets([match], menu, "C1: S1")

    assert selected[0].target_symbols == (
        (1, "pkg.Rule.entry"), (1, "pkg.Rule.deepHelper"))
def test_owner_promotion_preserves_four_selected_roles_and_their_entries():
    from library.clews import ClewMatch, ClewSymbolMenu, attach_symbol_targets
    match = ClewMatch(clew=Clew(id="a", source_name=SOURCE,
        entry_symbol="pkg.Start", route=["pkg.Start"], files=["a.py"]),
        similarity=1.0, obligations=(1,))
    menu = ClewSymbolMenu(
        labels={f"S{i}": f"pkg.Rule.helper{i}" for i in range(1, 6)},
        owner_entries={f"pkg.Rule.helper{i}": f"pkg.Rule.entry{i}"
                       for i in range(1, 6)})

    selected = attach_symbol_targets([match], menu, "C1: S1 S2 S3 S4 S5")

    assert selected[0].target_symbols == tuple(
        (1, f"pkg.Rule.{kind}{i}")
        for i in range(1, 5) for kind in ("entry", "helper"))


def test_symbol_menu_balances_symbols_across_selected_routes(tmp_path):
    from library.clews import ClewMatch, clew_symbol_menu
    conn = sqlite3.connect(tmp_path / "balanced.db")
    init_clews_schema(conn)
    conn.executescript("""
        CREATE TABLE scip_symbols (
            canonical_id TEXT, source_name TEXT, qualified_name TEXT,
            parent_qualified_name TEXT, kind TEXT, line_start INTEGER,
            line_end INTEGER, language TEXT, display_name TEXT, file TEXT);
        CREATE TABLE scip_edges (
            caller_canonical_id TEXT, callee_canonical_id TEXT, edge_type TEXT,
            file TEXT, line INTEGER);
        CREATE TABLE catalog (id INTEGER);
    """)
    matches = []
    for route_index in range(80):
        route = [f"pkg.Route{route_index}.step{step}" for step in range(10)]
        for step, name in enumerate(route):
            conn.execute(
                "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                (f"c{route_index}_{step}", "src", name, f"pkg.Route{route_index}",
                 "method", step, step, "scala", name.rsplit(".", 1)[-1], "F.scala"))
        matches.append(ClewMatch(clew=Clew(
            id=str(route_index), source_name="src", entry_symbol=route[0],
            route=route, files=["F.scala"]), similarity=1.0))
    menu = clew_symbol_menu(conn, matches, "unanchored question",
                            source_name="src", limit=500)
    values = set(menu.labels.values())
    assert all(f"pkg.Route{index}.step0" in values for index in range(80))
    assert all(f"pkg.Route{index}.step9" in values for index in range(80))



def test_document_routes_include_bounded_incoming_and_semantic_outgoing_neighbors(conn):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    rows = [
        ("root", "billing.UpdateTable.assignments", "logical.py", "UpdateTable"),
        ("caller", "billing.ResolveAssignments.apply", "analysis.py", "ResolveAssignments"),
        ("target", "billing.UpdateExpressions.generate", "update.py", "UpdateExpressions"),
    ]
    for canonical, qualified, file, parent in rows:
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", file, 1, 5, "Method",
                      qualified.rsplit(".", 1)[-1], qualified, f"billing.{parent}"))
    for index in range(6):
        canonical = f"noise-{index}"
        qualified = f"billing.Noise.helper{index}"
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", "noise.py", index, index + 1,
                      "Method", f"helper{index}", qualified, "billing.Noise"))
        conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                     ("root", canonical, "call", "logical.py", index, "exact"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("root", "target", "call", "logical.py", 20, "exact"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("caller", "root", "type_ref", "analysis.py", 8, "exact"))

    matches = document_clew_matches(
        conn, [{"title": "billing.UpdateTable.assignments",
                "metadata": {"qualified_name": "billing.UpdateTable.assignments"},
                "source_files": ["logical.py"], "content": ""}],
        "How do update assignments generate expressions?", source_name=SOURCE, limit=9)

    routes = [match.clew.route for match in matches]
    assert ["billing.ResolveAssignments.apply", "billing.UpdateTable.assignments"] in routes
    assert ["billing.UpdateTable.assignments", "billing.UpdateExpressions.generate"] in routes
def test_symbol_menu_keeps_route_local_endpoint_ahead_of_unrelated_owner_members(conn):
    from library.clews import ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    rows = [
        ("entry", SOURCE, "python", "route.py", 1, 2, "Method", "start",
         "pkg.Route.start", "pkg.Route"),
        ("finish", SOURCE, "python", "route.py", 3, 4, "Method", "finish",
         "pkg.Route.finish", "pkg.Route"),
        ("endpoint", SOURCE, "python", "route.py", 5, 6, "Method", "commit",
         "pkg.Route.zzCriticalEndpoint", "pkg.Route"),
    ]
    rows.extend(
        (f"noise-{number}", SOURCE, "python", "route.py", 10 + number,
         10 + number, "Method", f"aaaNoise{number}",
         f"pkg.Route.aaaNoise{number}", "pkg.Route")
        for number in range(20))
    conn.executemany("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("entry", "endpoint", "call", "route.py", 5, "exact"))
    match = ClewMatch(clew=Clew(
        id="route", source_name=SOURCE, entry_symbol="pkg.Route.start",
        route=["pkg.Route.start", "pkg.Route.finish"], files=["route.py"]),
        similarity=1.0)

    menu = clew_symbol_menu(
        conn, [match], "explain the route", source_name=SOURCE, limit=3)

    assert "pkg.Route.zzCriticalEndpoint" in menu.labels.values()
    assert not any("aaaNoise" in name for name in menu.labels.values())
def test_deterministic_clew_selection_covers_question_clauses_without_a_model():
    from library.clews import ClewMatch, deterministic_clew_matches

    matches = []
    for index in range(12):
        matches.append(ClewMatch(clew=Clew(
            id=f"noise-{index}", source_name=SOURCE,
            entry_symbol=f"noise.Worker{index}.run",
            route=[f"noise.Worker{index}.run", f"noise.Worker{index}.wait"],
            files=[f"noise{index}.py"], question="perform background maintenance"),
            similarity=0.95 - index / 1000))
    matches.extend((
        ClewMatch(clew=Clew(
            id="flush", source_name=SOURCE,
            entry_symbol="widget.Runner.run",
            route=["widget.Runner.run", "widget.Batch.flush"],
            files=["widget.py"], question="flush widget records"), similarity=0.82),
        ClewMatch(clew=Clew(
            id="report", source_name=SOURCE,
            entry_symbol="widget.Reporter.finish",
            route=["widget.Reporter.finish", "widget.Status.reportCompletion"],
            files=["report.py"], question="report completion status"), similarity=0.80),
    ))

    selected = deterministic_clew_matches(
        "How does the widget flush records and report completion?", matches, limit=6)

    assert len(selected) <= 6
    assert {match.clew.id for match in selected} >= {"flush", "report"}
    assert len({match.clew.entry_symbol.rsplit(".", 1)[0]
                for match in selected}) >= 2
def test_document_clew_matches_forwards_scip_path_context(conn, monkeypatch):
    from library.clews import document_clew_matches
    from library.structural_assembly import SeedSet
    seen = {}

    def fake_seeds(connection, documents, **kwargs):
        seen.update(kwargs)
        return SeedSet()

    monkeypatch.setattr("library.structural_assembly.seeds_from_documents", fake_seeds)

    result = document_clew_matches(
        conn, [], "how does this work?", source_name=SOURCE,
        indexer_cwds=("spark", "delta"), source_root="/corpus")

    assert result == []
    assert seen["indexer_cwds"] == ("spark", "delta")
    assert seen["source_root"] == "/corpus"
def test_deterministic_clew_precision_uses_structure_not_generated_question_text():
    from library.clews import ClewMatch, deterministic_clew_matches
    misleading = ClewMatch(clew=Clew(
        id="misleading", source_name=SOURCE,
        entry_symbol="pkg.Unrelated.delete",
        route=["pkg.Unrelated.delete", "pkg.Client.collect"], files=["wrong.py"],
        question="How does MergeRows emit resulting rows?"), similarity=0.99)
    structural = ClewMatch(clew=Clew(
        id="structural", source_name=SOURCE,
        entry_symbol="pkg.MergeRows.emitRows",
        route=["pkg.MergeRows.emitRows", "pkg.RowWriter.writeRows"],
        files=["right.py"], question="unrelated generated wording"), similarity=0.30)

    selected = deterministic_clew_matches(
        "How does MergeRows emit resulting rows?", [misleading, structural], limit=1)

    assert [match.clew.id for match in selected] == ["structural"]
def test_document_clew_routes_balance_incoming_and_outgoing_neighbors(conn):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    from schema import Document
    init_scip_schema(conn)
    symbols = [
        ("root", "pkg.Root.process", "root.py", "pkg.Root"),
        ("caller", "pkg.Entry.run", "entry.py", "pkg.Entry"),
        ("out1", "pkg.Target.first", "target.py", "pkg.Target"),
        ("out2", "pkg.Target.second", "target.py", "pkg.Target"),
        ("out3", "pkg.Target.third", "target.py", "pkg.Target"),
    ]
    for index, (canonical, qualified, file, parent) in enumerate(symbols, 1):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", file, index, index + 1,
                      "Method", qualified.rsplit(".", 1)[-1], qualified, parent))
    for index, target in enumerate(("out1", "out2", "out3"), 10):
        conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                     ("root", target, "call", "root.py", index, "exact"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("caller", "root", "call", "entry.py", 7, "exact"))
    document = Document(
        id="root-doc", content_type="catalog", title="Root.process", content="",
        source_files=["root.py"], metadata={}, source_name=SOURCE)

    matches = document_clew_matches(
        conn, [document], "How does Root process?", source_name=SOURCE, limit=3,
        indexer_cwds=(), source_root=None)
    routes = [match.clew.route for match in matches]

    assert ["pkg.Entry.run", "pkg.Root.process"] in routes
    assert routes[1] == ["pkg.Entry.run", "pkg.Root.process"]
    assert any(route[0] == "pkg.Root.process" and len(route) == 2 for route in routes)
def test_deterministic_clew_selection_spends_its_bounded_owner_budget():
    from library.clews import ClewMatch, deterministic_clew_matches
    matches = [ClewMatch(clew=Clew(
        id=str(index), source_name=SOURCE,
        entry_symbol=f"pkg.Owner{index}.processRows",
        route=[f"pkg.Owner{index}.processRows", f"pkg.Owner{index}.emitRows"],
        files=[f"owner{index}.py"]), similarity=1.0 - index / 100)
        for index in range(10)]

    selected = deterministic_clew_matches(
        "How are rows processed and emitted?", matches, limit=6)

    assert len(selected) == 6
    assert len({match.clew.entry_symbol.rsplit(".", 1)[0]
                for match in selected}) == 6
def test_deterministic_clew_selection_preserves_document_positioning_origin():
    from library.clews import ClewMatch, deterministic_clew_matches
    semantic = [ClewMatch(clew=Clew(
        id=f"semantic-{index}", source_name=SOURCE,
        entry_symbol=f"semantic.Owner{index}.processRows",
        route=[f"semantic.Owner{index}.processRows", f"semantic.Owner{index}.emitRows"],
        files=[f"semantic{index}.py"], strategy="theme-anchored"),
        similarity=0.99 - index / 100)
        for index in range(8)]
    positioned = ClewMatch(clew=Clew(
        id="positioned", source_name=SOURCE,
        entry_symbol="positioned.MergeRows.processPartition",
        route=["positioned.MergeRows.doExecute",
               "positioned.MergeRows.processPartition"],
        files=["merge.py"], strategy="document-scip"), similarity=0.20)

    selected = deterministic_clew_matches(
        "How are rows processed and emitted?", [*semantic, positioned], limit=4)

    assert positioned in selected
    assert any(match.clew.strategy != "document-scip" for match in selected)
    assert len(selected) == 4
def test_document_clew_uses_module_identity_and_owned_members(conn):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    symbols = [
        ("owner", "pkg.RowWorker", "Type", "pkg"),
        ("execute", "pkg.RowWorker.execute", "Method", "pkg.RowWorker"),
        ("process", "pkg.RowWorker.processPartition", "Method", "pkg.RowWorker"),
    ]
    for index, (canonical, qualified, kind, parent) in enumerate(symbols, 1):
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", "worker.py", index, index + 1,
                      kind, qualified.rsplit(".", 1)[-1], qualified, parent))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("execute", "process", "call", "worker.py", 2, "exact"))
    document = {
        "title": "Rowworker Module", "content": "", "source_files": [],
        "metadata": {"module_name": "pkg.RowWorker"},
    }

    matches = document_clew_matches(
        conn, [document], "How does the row worker process a partition?",
        source_name=SOURCE, limit=12, indexer_cwds=(), source_root=None)

    assert any(match.clew.route == ["pkg.RowWorker.execute",
                                    "pkg.RowWorker.processPartition"]
               for match in matches)
    assert {match.origin_rank for match in matches} == {0}
def test_deterministic_clew_selection_balances_positioned_document_ranks():
    from library.clews import ClewMatch, deterministic_clew_matches
    semantic = [ClewMatch(clew=Clew(
        id=f"semantic-{index}", source_name=SOURCE,
        entry_symbol=f"semantic.Owner{index}.processRows",
        route=[f"semantic.Owner{index}.processRows", f"semantic.Owner{index}.emitRows"],
        strategy="embedded"), similarity=0.99 - index / 100)
        for index in range(6)]
    first = ClewMatch(clew=Clew(
        id="doc-first", source_name=SOURCE, entry_symbol="docs.Entry.run",
        route=["docs.Entry.run", "docs.Rows.process"], strategy="document-scip"),
        similarity=0.4, origin_rank=1)
    second = ClewMatch(clew=Clew(
        id="doc-second", source_name=SOURCE, entry_symbol="docs.Writer.write",
        route=["docs.Writer.write", "docs.Result.emit"], strategy="document-scip"),
        similarity=0.3, origin_rank=4)

    selected = deterministic_clew_matches(
        "How are rows processed and emitted?", [*semantic, first, second], limit=4)

    assert first in selected
    assert second in selected
    assert any(match.clew.strategy != "document-scip" for match in selected)
def test_document_clew_prioritizes_call_edges_before_reference_fan_in(conn):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("root", SOURCE, "python", "worker.py", 1, 2, "Method",
                  "process", "pkg.Worker.process", "pkg.Worker"))
    for index in range(110):
        canonical = f"ref-{index:03d}"
        qualified = f"pkg.Reference{index}.value"
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", "refs.py", index + 3,
                      index + 3, "Field", "value", qualified,
                      f"pkg.Reference{index}"))
        conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                     (canonical, "root", "type_ref", "refs.py", index + 3, "exact"))
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("actual-caller", SOURCE, "python", "entry.py", 200, 201,
                  "Method", "run", "pkg.Entry.run", "pkg.Entry"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("actual-caller", "root", "call", "entry.py", 200, "exact"))
    document = {"title": "Worker.process", "content": "", "source_files": [],
                "metadata": {"qualified_name": "pkg.Worker.process"}}

    matches = document_clew_matches(
        conn, [document], "How is Worker.process called?", source_name=SOURCE,
        limit=4, indexer_cwds=(), source_root=None)

    assert any(match.clew.route == ["pkg.Entry.run", "pkg.Worker.process"]
               for match in matches)
def test_deterministic_clew_prefers_owner_coherent_document_transition():
    from library.clews import ClewMatch, deterministic_clew_matches
    nested = ClewMatch(clew=Clew(
        id="nested", source_name=SOURCE,
        entry_symbol="pkg.Worker.RowIterator.applyRows",
        route=["pkg.Worker.RowIterator.applyRows", "pkg.Worker.DiscardRow"],
        strategy="document-scip"), similarity=1.0, origin_rank=2,
        structure_score=2)
    direct = ClewMatch(clew=Clew(
        id="direct", source_name=SOURCE, entry_symbol="pkg.Worker.execute",
        route=["pkg.Worker.execute", "pkg.Worker.processPartition"],
        strategy="document-scip"), similarity=1.0, origin_rank=2,
        structure_score=3)

    selected = deterministic_clew_matches(
        "How does each worker process and emit rows?", [nested, direct], limit=1)

    assert selected == [direct]
def test_deterministic_clew_stitches_adjacent_document_transitions():
    from library.clews import ClewMatch, deterministic_clew_matches
    first = ClewMatch(clew=Clew(
        id="first", source_name=SOURCE, entry_symbol="pkg.Entry.run",
        route=["pkg.Entry.run", "pkg.Worker.process"], files=["entry.py"],
        strategy="document-scip"), similarity=1.0, origin_rank=2,
        structure_score=3)
    second = ClewMatch(clew=Clew(
        id="second", source_name=SOURCE, entry_symbol="pkg.Worker.process",
        route=["pkg.Worker.process", "pkg.Result.emit"], files=["worker.py"],
        strategy="document-scip"), similarity=1.0, origin_rank=2,
        structure_score=3)

    selected = deterministic_clew_matches(
        "How does the worker process and emit a result?", [first, second], limit=1)

    assert len(selected) == 1
    assert selected[0].clew.route == [
        "pkg.Entry.run", "pkg.Worker.process", "pkg.Result.emit"]
    assert selected[0].clew.files == ["entry.py", "worker.py"]
def test_document_clew_prioritizes_members_named_by_section_headings(conn):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("owner", SOURCE, "python", "worker.py", 1, 30, "Type",
                  "Worker", "pkg.Worker", "pkg"))
    for index in range(12):
        canonical = f"noise-{index}"
        qualified = f"pkg.Worker.aaaHelper{index}"
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", "worker.py", index + 2,
                      index + 2, "Method", f"aaaHelper{index}", qualified,
                      "pkg.Worker"))
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("process", SOURCE, "python", "worker.py", 20, 22, "Method",
                  "processPartition", "pkg.Worker.processPartition", "pkg.Worker"))
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("execute", SOURCE, "python", "worker.py", 18, 19, "Method",
                  "execute", "pkg.Worker.execute", "pkg.Worker"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("execute", "process", "call", "worker.py", 19, "exact"))
    document = {
        "title": "Worker Module", "source_files": [],
        "metadata": {"module_name": "pkg.Worker"},
        "content": "# Worker\n\n## Compilation Phase (`processPartition`)\n",
    }

    matches = document_clew_matches(
        conn, [document], "How is work completed?", source_name=SOURCE,
        limit=8, indexer_cwds=(), source_root=None)

    assert any(match.clew.route == ["pkg.Worker.execute",
                                    "pkg.Worker.processPartition"]
               for match in matches)
def test_document_clew_batches_neighbor_lookups_across_roots(conn):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("owner", SOURCE, "python", "worker.py", 1, 20, "Type",
                  "Worker", "pkg.Worker", "pkg"))
    for index in range(5):
        canonical = f"method-{index}"
        qualified = f"pkg.Worker.method{index}"
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "python", "worker.py", index + 2,
                      index + 2, "Method", f"method{index}", qualified,
                      "pkg.Worker"))
        if index:
            conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                         (f"method-{index - 1}", canonical, "call", "worker.py",
                          index + 2, "exact"))

    class RecordingConnection:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.edge_queries = []

        def execute(self, sql, parameters=()):
            if "FROM scip_edges" in sql:
                self.edge_queries.append(sql)
            return self.wrapped.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    recording = RecordingConnection(conn)
    document = {"title": "Worker Module", "content": "", "source_files": [],
                "metadata": {"module_name": "pkg.Worker"}}

    document_clew_matches(
        recording, [document], "How does Worker work?", source_name=SOURCE,
        limit=12, indexer_cwds=(), source_root=None)

    assert len(recording.edge_queries) == 2
def test_document_clew_pins_batched_edges_to_endpoint_indexes(conn):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                 ("owner", SOURCE, "python", "worker.py", 1, 2, "Type",
                  "Worker", "pkg.Worker", "pkg"))

    class RecordingConnection:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.edge_queries = []

        def execute(self, sql, parameters=()):
            if "FROM scip_edges" in sql:
                self.edge_queries.append(sql)
            return self.wrapped.execute(sql, parameters)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    recording = RecordingConnection(conn)
    document_clew_matches(
        recording, [{"title": "Worker", "content": "", "source_files": [],
                     "metadata": {"module_name": "pkg.Worker"}}],
        "How does Worker work?", source_name=SOURCE, limit=3,
        indexer_cwds=(), source_root=None)

    assert any("INDEXED BY idx_scip_edges_callee" in sql
               for sql in recording.edge_queries)
    assert any("INDEXED BY idx_scip_edges_caller" in sql
               for sql in recording.edge_queries)
def test_document_clew_recovers_same_owner_method_value_reference(conn, tmp_path):
    from library.clews import document_clew_matches
    from library.scip import init_scip_schema
    init_scip_schema(conn)
    source = tmp_path / "Worker.scala"
    source.write_text(
        "class Worker {\n"
        "  def execute() = input.map(processPartition)\n"
        "  def processPartition(row: Int) = row\n"
        "}\n")
    rows = [
        ("owner", "pkg.Worker", "Class", "pkg", 1),
        ("execute", "pkg.Worker.execute", "Method", "pkg.Worker", 2),
        ("process", "pkg.Worker.processPartition", "Method", "pkg.Worker", 3),
    ]
    for canonical, qualified, kind, parent, line in rows:
        conn.execute("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (canonical, SOURCE, "scala", "Worker.scala", line, line,
                      kind, qualified.rsplit(".", 1)[-1], qualified, parent))
    document = {
        "title": "Worker Module", "source_files": ["Worker.scala"],
        "metadata": {"module_name": "pkg.Worker"},
        "content": "## Execution (`processPartition`)",
    }

    matches = document_clew_matches(
        conn, [document], "How does the worker execute each partition?",
        source_name=SOURCE, limit=8, indexer_cwds=(), source_root=str(tmp_path))

    recovered = [match for match in matches if match.clew.route == [
        "pkg.Worker.execute", "pkg.Worker.processPartition"]]
    assert len(recovered) == 1
    assert recovered[0].structure_score == 3
def test_family_menu_keeps_full_owner_identity_on_each_card():
    from library.clews import ClewMatch, clew_family_menu
    first = ClewMatch(clew=Clew(
        id="first", source_name=SOURCE,
        entry_symbol="pkg.pipeline.Alpha.run",
        route=["pkg.pipeline.Alpha.run", "pkg.Output.write"],
        files=["alpha.py"]), similarity=0.9)
    second = ClewMatch(clew=Clew(
        id="second", source_name=SOURCE,
        entry_symbol="pkg.pipeline.Beta.run",
        route=["pkg.pipeline.Beta.run", "pkg.Output.write"],
        files=["beta.py"]), similarity=0.8)

    menu = clew_family_menu([first, second], question="run pipeline")
    assert menu.text.count("pkg.pipeline") == 2
    assert "pkg.pipeline.Alpha" in menu.text
    assert "pkg.pipeline.Alpha" in menu.text
    assert "pkg.pipeline.Beta" in menu.text
def test_nearest_matches_reuses_one_vector_index_per_unchanged_source(conn, monkeypatch):
    import library.clews as clews_module

    add_clew(conn, **_clew(
        "pkg.first", ["first", "save"], ["first.py"], vector=[1.0, 0.0, 0.0]))
    add_clew(conn, **_clew(
        "pkg.second", ["second", "save"], ["second.py"], vector=[0.0, 1.0, 0.0]))
    clews_module.clear_clew_embedding_cache()
    builds = []
    original = clews_module._build_clew_embedding_index

    def counted(connection, source_name, dimensions, stamp):
        builds.append((source_name, dimensions, stamp))
        return original(connection, source_name, dimensions, stamp)

    monkeypatch.setattr(clews_module, "_build_clew_embedding_index", counted)
    query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    first = clews_module.nearest_clew_matches(
        conn, query, source_name=SOURCE, top_k=2, min_similarity=-1.0)
    second = clews_module.nearest_clew_matches(
        conn, query, source_name=SOURCE, top_k=2, min_similarity=-1.0)

    assert [match.clew.id for match in first] == [match.clew.id for match in second]
    assert len(builds) == 1, builds


def test_nearest_matches_invalidates_vector_index_when_clews_change(conn):
    import library.clews as clews_module

    clews_module.clear_clew_embedding_cache()
    add_clew(conn, **_clew(
        "pkg.old", ["old", "save"], ["old.py"], vector=[0.8, 0.2, 0.0]))
    query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    assert clews_module.nearest_clew_matches(
        conn, query, source_name=SOURCE, top_k=1)[0].clew.entry_symbol == "pkg.old"

    add_clew(conn, **_clew(
        "pkg.new", ["new", "save"], ["new.py"], vector=[1.0, 0.0, 0.0]))

    assert clews_module.nearest_clew_matches(
        conn, query, source_name=SOURCE, top_k=1)[0].clew.entry_symbol == "pkg.new"
def test_symbol_target_entry_bfs_does_not_treat_ownership_as_execution(conn):
    from library.clews import Clew, ClewMatch, clew_symbol_menu
    from library.scip import init_scip_schema

    init_scip_schema(conn)
    conn.executemany("INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)", [
        ("owner", SOURCE, "java", "owner.java", 1, 80, "Class", "Owner",
         "pkg.Owner", "pkg"),
        ("target", SOURCE, "java", "owner.java", 10, 20, "Method", "target",
         "pkg.Owner.target", "pkg.Owner"),
        ("sibling", SOURCE, "java", "owner.java", 30, 40, "Method", "inspect",
         "pkg.Owner.inspect", "pkg.Owner"),
    ])
    conn.executemany("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)", [
        ("owner", "target", "contains", "owner.java", 10, "exact"),
        ("sibling", "owner", "type_ref", "owner.java", 34, "exact"),
    ])
    match = ClewMatch(clew=Clew(
        id="target", source_name=SOURCE, entry_symbol="pkg.Owner.target",
        route=["pkg.Owner.target"], files=["owner.java"], strategy="semantic"),
        similarity=1.0)

    menu = clew_symbol_menu(
        conn, [match], "which method executes target", source_name=SOURCE)

    assert menu.owner_entries.get("pkg.Owner.target") != "pkg.Owner.inspect"


def _plain_match(*route):
    clew = type("Clew", (), {
        "id": "-".join(route), "route": route, "steps": (),
        "question": "", "entry_symbol": route[0], "files": (),
        "strategy": "synthetic"})()
    return type("Match", (), {
        "clew": clew, "target_symbols": (), "similarity": 0.0,
        "structure_score": 0, "origin_rank": None})()


class TestDeterministicRarityWeighting:
    def test_rare_token_route_beats_common_token_ties(self):
        from library.clews import deterministic_clew_matches

        # "write" appears across three routes, "sink" in one; a raw
        # overlap count ties the writer route with the sink route and
        # alphabetical identity would pick the writer.
        matches = [
            _plain_match("pkg.BatchWriter.writeBatch"),
            _plain_match("pkg.LogWriter.writeLog"),
            _plain_match("pkg.FileWriter.writeFile"),
            _plain_match("pkg.Sink.addBatch"),
        ]

        chosen = deterministic_clew_matches(
            "how does the sink write each batch", matches, limit=1)

        assert chosen[0].clew.route == ("pkg.Sink.addBatch",)


class TestFullRankExposure:
    def test_full_ranking_orders_every_match_without_a_shortlist(self):
        from library.clews import (
            deterministic_clew_matches,
            rank_clew_matches,
        )

        matches = [
            _plain_match("pkg.BatchWriter.writeBatch"),
            _plain_match("pkg.LogWriter.writeLog"),
            _plain_match("pkg.FileWriter.writeFile"),
            _plain_match("pkg.Sink.addBatch"),
        ]

        ordered = rank_clew_matches(
            "how does the sink write each batch", matches)

        assert len(ordered) == len(matches)
        assert ordered[0].clew.route == ("pkg.Sink.addBatch",)
        shortlist = deterministic_clew_matches(
            "how does the sink write each batch", matches, limit=1)
        assert shortlist[0].clew.id == ordered[0].clew.id
