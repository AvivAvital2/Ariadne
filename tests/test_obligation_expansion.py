"""Obligation-seeded bounded expansion: reach driven by proof needs.

The clew-mode pool historically ran its caller walk with empty seeds, so
upstream entries and registrars were deterministically unreachable even
when fully present in the store. Expansion seeds from obligation targets,
clew-route endpoints, and question symbols; walks reverse call and
type_ref edges to a bounded depth; bridges ownership exactly one hop
(member -> exact owner, never siblings); and records every cap event
instead of silently discarding candidates. Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.catalog_writer import _element_doc_id
from library import Library
from library.chain_answer import evidence_for
from library.clews import Clew, ClewMatch
from library.scip import init_scip_schema
from library.structural_assembly import (
    ObligationExpansion,
    connect_obligation_targets,
    obligation_seeded_expansion,
)

SOURCE = "src1"
RULE = "scip-x x src1 0.1 `pkg`/Rule#"
RULE_APPLY = "scip-x x src1 0.1 `pkg`/Rule#apply()."
RULE_OTHER = "scip-x x src1 0.1 `pkg`/Rule#other()."
EXT_INSTALL = "scip-x x src1 0.1 `pkg`/Extension#install()."
ENTRY_START = "scip-x x src1 0.1 `pkg`/Entry#start()."
GRAND_MAIN = "scip-x x src1 0.1 `pkg`/Grand#main()."


def _symbol(conn, cid, *, file, qn, line_start, line_end, parent=""):
    conn.execute(
        "INSERT INTO scip_symbols (canonical_id, source_name, language, file, "
        "line_start, line_end, kind, display_name, qualified_name, "
        "parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (cid, SOURCE, "scala", file, line_start, line_end, "", "", qn, parent))


def _edge(conn, caller, callee, *, line, file, edge_type="call"):
    conn.execute(
        "INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, "
        "edge_type, file, line, confidence) VALUES (?,?,?,?,?,'exact')",
        (caller, callee, edge_type, file, line))


def _match(route, targets):
    return ClewMatch(
        clew=Clew(id="k", source_name=SOURCE, entry_symbol=route[0],
                  route=list(route), files=[], strategy="test"),
        similarity=0.9,
        obligations=tuple(sorted({number for number, _ in targets})),
        target_symbols=tuple(targets))


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    init_scip_schema(connection)
    _symbol(connection, RULE, file="pkg/rule.scala", qn="pkg.Rule",
            line_start=10, line_end=60)
    _symbol(connection, RULE_APPLY, file="pkg/rule.scala", qn="pkg.Rule.apply",
            line_start=20, line_end=40, parent="pkg.Rule")
    _symbol(connection, RULE_OTHER, file="pkg/rule.scala", qn="pkg.Rule.other",
            line_start=45, line_end=55, parent="pkg.Rule")
    _symbol(connection, EXT_INSTALL, file="pkg/extension.scala",
            qn="pkg.Extension.install", line_start=5, line_end=15,
            parent="pkg.Extension")
    _symbol(connection, ENTRY_START, file="pkg/entry.scala",
            qn="pkg.Entry.start", line_start=8, line_end=30,
            parent="pkg.Entry")
    _symbol(connection, GRAND_MAIN, file="pkg/grand.scala",
            qn="pkg.Grand.main", line_start=3, line_end=12,
            parent="pkg.Grand")
    _edge(connection, ENTRY_START, RULE_APPLY, line=12, file="pkg/entry.scala")
    _edge(connection, EXT_INSTALL, RULE, line=9, file="pkg/extension.scala",
          edge_type="type_ref")
    _edge(connection, GRAND_MAIN, ENTRY_START, line=6, file="pkg/grand.scala")
    connection.commit()
    yield connection
    connection.close()


class TestReverseEntryDiscovery:
    def test_reverse_call_recovers_the_upstream_entry(self, conn):
        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=1)

        names = {citation.qualified_name for citation in expansion.citations}
        assert "pkg.Entry.start" in names
        root = next(c for c in expansion.citations
                    if c.qualified_name == "pkg.Entry.start")
        assert root.relation == "called_by"
        child = next(c for c in expansion.citations
                     if c.parent_qualified_name == "pkg.Entry.start")
        assert child.qualified_name == "pkg.Rule.apply"
        assert child.relation == "calls"
        assert (child.call_site_file, child.call_site_line) == (
            "pkg/entry.scala", 12)

    def test_owner_bridge_recovers_a_registrar_referencing_the_class(
            self, conn):
        # The route ends at the member; the registrar only references the
        # owning type. One structural bridge hop makes it reachable.
        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=1)

        names = {citation.qualified_name for citation in expansion.citations}
        assert "pkg.Extension.install" in names
        registrar = next(c for c in expansion.citations
                         if c.qualified_name == "pkg.Extension.install")
        assert registrar.relation == "shared_reference"
        referenced = next(
            c for c in expansion.citations
            if c.parent_qualified_name == "pkg.Extension.install")
        assert referenced.qualified_name == "pkg.Rule"
        assert referenced.relation == "references"

    def test_owner_bridge_never_fans_out_to_siblings(self, conn):
        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=2)

        names = {citation.qualified_name for citation in expansion.citations}
        assert "pkg.Rule.other" not in names

    def test_depth_two_reaches_the_caller_of_the_entry(self, conn):
        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=2)

        grand = [c for c in expansion.citations
                 if c.qualified_name == "pkg.Grand.main"]
        assert grand and grand[0].relation == "called_by"
        linked = [c for c in expansion.citations
                  if c.parent_qualified_name == "pkg.Grand.main"]
        assert [c.qualified_name for c in linked] == ["pkg.Entry.start"]

    def test_nonproduction_callers_stay_out(self, conn):
        fixture_caller = "scip-x x src1 0.1 `pkg`/RuleSpec#check()."
        _symbol(conn, fixture_caller, file="src/tests/rule_spec.scala",
                qn="pkg.RuleSpec.check", line_start=4, line_end=9,
                parent="pkg.RuleSpec")
        _edge(conn, fixture_caller, RULE_APPLY, line=6,
              file="src/tests/rule_spec.scala")
        conn.commit()

        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=1)

        names = {citation.qualified_name for citation in expansion.citations}
        assert "pkg.RuleSpec.check" not in names


class TestBoundsAndDeterminism:
    def test_per_seed_shortlist_reports_overflow_never_silence(self, conn):
        for index in range(12):
            caller = f"scip-x x src1 0.1 `pkg`/Consumer{index}#use()."
            _symbol(conn, caller, file=f"pkg/consumer{index}.scala",
                    qn=f"pkg.Consumer{index}.use",
                    line_start=2, line_end=8, parent=f"pkg.Consumer{index}")
            _edge(conn, caller, RULE_APPLY, line=4,
                  file=f"pkg/consumer{index}.scala", edge_type="type_ref")
        conn.commit()

        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=1, per_seed_limit=2, reserve_limit=3)

        assert RULE_APPLY in expansion.truncated_seeds
        record = expansion.reasons[RULE_APPLY]
        assert record["available"] >= 12
        assert record["retained"] == 2
        assert record["reserve"] == 3
        assert record["discarded"] == record["available"] - 5

    def test_every_canonical_id_of_a_target_is_seeded(self, conn):
        overload_a = "scip-x x src1 0.1 `pkg`/Over#apply()."
        overload_b = "scip-x x src1 0.1 `pkg`/Over#apply(+1)."
        caller = "scip-x x src1 0.1 `pkg`/Caller#go()."
        _symbol(conn, overload_a, file="pkg/over.scala", qn="pkg.Over.apply",
                line_start=5, line_end=9, parent="pkg.Over")
        _symbol(conn, overload_b, file="pkg/over.scala", qn="pkg.Over.apply",
                line_start=11, line_end=19, parent="pkg.Over")
        _symbol(conn, caller, file="pkg/caller.scala", qn="pkg.Caller.go",
                line_start=2, line_end=8, parent="pkg.Caller")
        _edge(conn, caller, overload_b, line=4, file="pkg/caller.scala")
        conn.commit()

        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Over.apply"], [(1, "pkg.Over.apply")])],
            source=SOURCE, depth=1)

        names = {citation.qualified_name for citation in expansion.citations}
        assert "pkg.Caller.go" in names

    def test_cycles_terminate(self, conn):
        loop_a = "scip-x x src1 0.1 `pkg`/LoopA#run()."
        loop_b = "scip-x x src1 0.1 `pkg`/LoopB#run()."
        _symbol(conn, loop_a, file="pkg/loop_a.scala", qn="pkg.LoopA.run",
                line_start=1, line_end=5, parent="pkg.LoopA")
        _symbol(conn, loop_b, file="pkg/loop_b.scala", qn="pkg.LoopB.run",
                line_start=1, line_end=5, parent="pkg.LoopB")
        _edge(conn, loop_a, loop_b, line=2, file="pkg/loop_a.scala")
        _edge(conn, loop_b, loop_a, line=2, file="pkg/loop_b.scala")
        conn.commit()

        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.LoopB.run"], [(1, "pkg.LoopB.run")])],
            source=SOURCE, depth=2)

        assert isinstance(expansion, ObligationExpansion)

    def test_repeated_runs_are_identical(self, conn):
        matches = [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])]

        first = obligation_seeded_expansion(
            conn, matches, source=SOURCE, depth=2)
        second = obligation_seeded_expansion(
            conn, matches, source=SOURCE, depth=2)

        assert first == second


class TestObligationTargetConnection:
    def test_connected_obligation_targets_keep_all_overloads(self, conn):
        overload_a = "scip-x x src1 0.1 `pkg`/Fork#apply()."
        overload_b = "scip-x x src1 0.1 `pkg`/Fork#apply(+1)."
        sink = "scip-x x src1 0.1 `pkg`/Sink#write()."
        _symbol(conn, overload_a, file="pkg/fork.scala", qn="pkg.Fork.apply",
                line_start=5, line_end=9, parent="pkg.Fork")
        _symbol(conn, overload_b, file="pkg/fork.scala", qn="pkg.Fork.apply",
                line_start=11, line_end=19, parent="pkg.Fork")
        _symbol(conn, sink, file="pkg/sink.scala", qn="pkg.Sink.write",
                line_start=3, line_end=9, parent="pkg.Sink")
        _edge(conn, overload_b, sink, line=14, file="pkg/fork.scala")
        conn.commit()

        citations = connect_obligation_targets(
            conn, [_match(
                ["pkg.Fork.apply"],
                [(1, "pkg.Fork.apply"), (1, "pkg.Sink.write")])],
            source=SOURCE)

        names = {citation.qualified_name for citation in citations}
        assert {"pkg.Fork.apply", "pkg.Sink.write"} <= names


class TestClewEvidenceIntegration:
    def test_clew_evidence_reaches_upstream_registration(self, tmp_path):
        library = Library(tmp_path / "l.db")
        try:
            with library._conn_provider.acquire() as connection:
                init_scip_schema(connection)
                _symbol(connection, RULE, file="pkg/rule.scala", qn="pkg.Rule",
                        line_start=10, line_end=60)
                _symbol(connection, RULE_APPLY, file="pkg/rule.scala",
                        qn="pkg.Rule.apply", line_start=20, line_end=40,
                        parent="pkg.Rule")
                _symbol(connection, EXT_INSTALL, file="pkg/extension.scala",
                        qn="pkg.Extension.install", line_start=5, line_end=15,
                        parent="pkg.Extension")
                _edge(connection, EXT_INSTALL, RULE, line=9,
                      file="pkg/extension.scala", edge_type="type_ref")
                connection.commit()
            library.add_document(
                content_type="catalog", title="apply",
                content="Applies the rule.", source_files=["pkg/rule.scala"],
                doc_id=_element_doc_id(SOURCE, "pkg.Rule.apply"),
                source_name=SOURCE)

            evidence = evidence_for(
                library, [{"source_files": ["pkg/rule.scala"]}],
                source=SOURCE,
                clew_matches=[_match(
                    ["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])])

            names = {c.qualified_name for c in evidence.bundle_citations}
            assert "pkg.Extension.install" in names
            assert len(evidence.bundle_citations) < 20
            origins = {
                entry["symbol"]: entry["origins"]
                for entry in evidence.seed_provenance}
            assert "obligation_expansion" in origins.get(
                "pkg.Extension.install", ())
        finally:
            library.close()


class TestSharedReferenceDeterminism:
    def test_insertion_order_never_changes_shared_reference_callers(self):
        from library.structural_assembly import _shared_reference_callers

        def build(count, ordering_reversed):
            rows = [
                (f"scip-x x src1 0.1 `pkg`/Consumer{index}#use().",
                 f"pkg/consumer{index}.scala", index + 2)
                for index in range(count)]
            if ordering_reversed:
                rows = list(reversed(rows))
            connection = sqlite3.connect(":memory:")
            init_scip_schema(connection)
            for caller, file, line in rows:
                _symbol(connection, caller, file=file,
                        qn=caller.rsplit("`/", 1)[-1].rstrip("()."),
                        line_start=1, line_end=9, parent="")
                _edge(connection, caller, RULE, line=line, file=file,
                      edge_type="type_ref")
            _symbol(connection, RULE, file="pkg/rule.scala", qn="pkg.Rule",
                    line_start=10, line_end=60)
            connection.commit()
            result = _shared_reference_callers(connection, RULE, source=SOURCE)
            connection.close()
            return result

        small = [build(5, reversed_order) for reversed_order in (False, True)]
        assert small[0] == small[1]
        assert len(small[0]) == 5
        assert [row[0] for row in small[0]] == sorted(
            row[0] for row in small[0])

        large = [build(12, reversed_order) for reversed_order in (False, True)]
        assert large[0] == large[1] == ()


class TestSeedChannels:
    def test_catalog_only_seed_participates_in_expansion(self, conn):
        expansion = obligation_seeded_expansion(
            conn, [], source=SOURCE, depth=1,
            catalog_seed_ids=(RULE_APPLY,))

        names = {citation.qualified_name for citation in expansion.citations}
        assert "pkg.Entry.start" in names

    def test_facet_seeds_resolve_display_and_suffix_literally(self, conn):
        from library.structural_assembly import facet_symbol_seeds

        rewrite = "scip-x x src1 0.1 `pkg`/RewriteMerge#"
        snake = "scip-x x src1 0.1 `pkg`/preprocess_table()."
        decoy = "scip-x x src1 0.1 `pkg`/preprocessXtable()."
        conn.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rewrite, SOURCE, "x", "pkg/rw.scala", 4, 40, "",
             "RewriteMerge", "pkg.analysis.RewriteMerge", ""))
        conn.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (snake, SOURCE, "x", "pkg/pp.py", 2, 9, "",
             "preprocess_table", "pkg.preprocess_table", ""))
        conn.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (decoy, SOURCE, "x", "pkg/px.py", 2, 9, "",
             "preprocessXtable", "pkg.preprocessXtable", ""))
        conn.commit()

        by_display = facet_symbol_seeds(
            conn, ["RewriteMerge"], source=SOURCE)
        assert rewrite in by_display

        by_suffix = facet_symbol_seeds(
            conn, ["preprocess_table"], source=SOURCE)
        assert snake in by_suffix
        assert decoy not in by_suffix

    def test_question_facet_identifier_reaches_its_upstream_caller(
            self, tmp_path):
        library = Library(tmp_path / "l.db")
        try:
            with library._conn_provider.acquire() as connection:
                init_scip_schema(connection)
                rewrite = "scip-x x src1 0.1 `pkg`/RewriteMerge#apply()."
                caller = "scip-x x src1 0.1 `pkg`/Analyzer#execute()."
                _symbol(connection, rewrite, file="pkg/rw.scala",
                        qn="pkg.RewriteMerge.apply", line_start=4,
                        line_end=30, parent="pkg.RewriteMerge")
                connection.execute(
                    "UPDATE scip_symbols SET display_name = ? "
                    "WHERE canonical_id = ?", ("apply", rewrite))
                _symbol(connection, caller, file="pkg/an.scala",
                        qn="pkg.Analyzer.execute", line_start=2,
                        line_end=20, parent="pkg.Analyzer")
                _edge(connection, caller, rewrite, line=8,
                      file="pkg/an.scala")
                connection.commit()
            library.add_document(
                content_type="catalog", title="apply",
                content="Rewrites merges.", source_files=["pkg/rw.scala"],
                doc_id=_element_doc_id(SOURCE, "pkg.RewriteMerge.apply"),
                source_name=SOURCE)

            evidence = evidence_for(
                library, [{"source_files": ["pkg/rw.scala"]}],
                source=SOURCE,
                question="How does RewriteMerge.apply divert the plan?",
                clew_matches=[_match(["pkg.Unrelated.node"], ())])

            names = {c.qualified_name for c in evidence.bundle_citations}
            assert "pkg.Analyzer.execute" in names
        finally:
            library.close()


class TestForwardContinuation:
    def test_forward_call_chain_extends_from_a_seed(self, conn):
        sink = "scip-x x src1 0.1 `pkg`/Sink#write()."
        _symbol(conn, sink, file="pkg/sink.scala", qn="pkg.Sink.write",
                line_start=3, line_end=9, parent="pkg.Sink")
        _edge(conn, RULE_APPLY, sink, line=25, file="pkg/rule.scala")
        conn.commit()

        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=2, forward_depth=2)

        names = {citation.qualified_name for citation in expansion.citations}
        assert "pkg.Sink.write" in names
        child = next(c for c in expansion.citations
                     if c.qualified_name == "pkg.Sink.write")
        assert child.parent_qualified_name == "pkg.Rule.apply"
        assert child.relation == "calls"
        assert (child.call_site_file, child.call_site_line) == (
            "pkg/rule.scala", 25)
        assert expansion.reasons["forward:enabled"] == {"depth": 2}
    def test_forward_depth_two_reaches_the_callee_of_the_callee(self, conn):
        sink = "scip-x x src1 0.1 `pkg`/Sink#write()."
        flush = "scip-x x src1 0.1 `pkg`/Sink#flush()."
        _symbol(conn, sink, file="pkg/sink.scala", qn="pkg.Sink.write",
                line_start=3, line_end=9, parent="pkg.Sink")
        _symbol(conn, flush, file="pkg/sink.scala", qn="pkg.Sink.flush",
                line_start=11, line_end=19, parent="pkg.Sink")
        _edge(conn, RULE_APPLY, sink, line=25, file="pkg/rule.scala")
        _edge(conn, sink, flush, line=6, file="pkg/sink.scala")
        conn.commit()

        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=2, forward_depth=2)

        names = {citation.qualified_name for citation in expansion.citations}
        assert "pkg.Sink.flush" in names
    def test_forward_fanout_reports_overflow_never_silence(self, conn):
        for index in range(12):
            callee = f"scip-x x src1 0.1 `pkg`/Leaf{index}#go()."
            _symbol(conn, callee, file=f"pkg/leaf{index}.scala",
                    qn=f"pkg.Leaf{index}.go", line_start=2, line_end=6,
                    parent=f"pkg.Leaf{index}")
            _edge(conn, RULE_APPLY, callee, line=20 + index,
                  file="pkg/rule.scala")
        conn.commit()

        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=1, forward_depth=1,
            per_seed_limit=2, reserve_limit=3)

        forward = expansion.reasons.get(f"forward:{RULE_APPLY}")
        assert forward is not None
        assert forward["available"] >= 12
        assert forward["retained"] == 2


class TestForwardDefaultOff:
    def test_default_expansion_produces_no_forward_hops(self, conn):
        sink = "scip-x x src1 0.1 `pkg`/Sink#write()."
        _symbol(conn, sink, file="pkg/sink.scala", qn="pkg.Sink.write",
                line_start=3, line_end=9, parent="pkg.Sink")
        _edge(conn, RULE_APPLY, sink, line=25, file="pkg/rule.scala")
        conn.commit()

        expansion = obligation_seeded_expansion(
            conn, [_match(["pkg.Rule.apply"], [(1, "pkg.Rule.apply")])],
            source=SOURCE, depth=2)

        assert not [c for c in expansion.citations
                    if c.stop_reason == "obligation_continuation"]
        assert "pkg.Sink.write" not in {
            c.qualified_name for c in expansion.citations}
        assert "forward:enabled" not in expansion.reasons
        # Reverse discovery is untouched by the default.
        assert "pkg.Entry.start" in {
            c.qualified_name for c in expansion.citations}
