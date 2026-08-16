"""Exact store grading: identity is source, file, extent, type, and site.

A canonical id present in the wrong module must fail; an ownership edge
must never satisfy a call; a reversed call is a direction failure, not a
pass; and two identical exact rows are ambiguity, never a free choice.
Every failure carries exactly one primary reason.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp_store_truth",
    ROOT / "evaluation" / "chain-benchmark" / "exp_store_truth.py")
store_truth = importlib.util.module_from_spec(SPEC)
sys.modules["exp_store_truth"] = store_truth
SPEC.loader.exec_module(store_truth)

from library.scip import init_scip_schema

SOURCE = "src1"
BUILDER = "scip-x x src1 0.1 `pkg`/Builder#run()."
SINK = "scip-x x src1 0.1 `pkg`/Sink#write()."


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    init_scip_schema(connection)
    connection.execute(
        "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
        (BUILDER, SOURCE, "x", "core/pkg/builder.scala", 10, 40, "method",
         "run", "pkg.Builder.run", "pkg.Builder"))
    connection.execute(
        "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
        (SINK, SOURCE, "x", "core/pkg/sink.scala", 3, 9, "method",
         "write", "pkg.Sink.write", "pkg.Sink"))
    connection.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        (BUILDER, SINK, "call", "core/pkg/builder.scala", 20, "exact"))
    connection.commit()
    yield connection
    connection.close()


def expectation(**overrides):
    parameters = dict(
        source=SOURCE, canonical_id=BUILDER,
        qualified_name="pkg.Builder.run",
        file="core/pkg/builder.scala", line_start=10, line_end=40,
        kind="method")
    parameters.update(overrides)
    return store_truth.DefinitionExpectation(**parameters)


def edge(**overrides):
    parameters = dict(
        caller_canonical_id=BUILDER, callee_canonical_id=SINK,
        edge_type="call", file="core/pkg/builder.scala", line=20)
    parameters.update(overrides)
    return store_truth.EdgeExpectation(**parameters)


class TestDefinitionGrading:
    def test_exact_definition_passes(self, conn):
        assert store_truth.grade_definition(conn, expectation()) is None

    def test_absent_symbol(self, conn):
        gap = store_truth.grade_definition(
            conn, expectation(canonical_id="scip-x x src1 0.1 ghost."))
        assert gap.primary_reason == "absent_symbol"

    def test_same_canonical_in_wrong_module_is_wrong_module(self, conn):
        conn.execute("DELETE FROM scip_symbols WHERE canonical_id = ?",
                     (BUILDER,))
        conn.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (BUILDER, SOURCE, "x", "connect-client/pkg/builder.scala",
             10, 40, "method", "run", "pkg.Builder.run", "pkg.Builder"))
        conn.commit()

        gap = store_truth.grade_definition(conn, expectation())
        assert gap.primary_reason == "wrong_module"

    def test_same_module_different_file_is_wrong_file(self, conn):
        conn.execute("DELETE FROM scip_symbols WHERE canonical_id = ?",
                     (BUILDER,))
        conn.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (BUILDER, SOURCE, "x", "core/pkg/other.scala", 10, 40,
             "method", "run", "pkg.Builder.run", "pkg.Builder"))
        conn.commit()

        gap = store_truth.grade_definition(conn, expectation())
        assert gap.primary_reason == "wrong_file"

    def test_wrong_source_is_wrong_source_instance(self, conn):
        conn.execute("UPDATE scip_symbols SET source_name = 'other' "
                     "WHERE canonical_id = ?", (BUILDER,))
        conn.commit()

        gap = store_truth.grade_definition(conn, expectation())
        assert gap.primary_reason == "wrong_source_instance"

    def test_correct_canonical_wrong_extent_fails(self, conn):
        gap = store_truth.grade_definition(
            conn, expectation(line_start=10, line_end=99))
        assert gap.primary_reason == "wrong_extent"

    def test_two_identical_exact_rows_are_ambiguous(self, tmp_path):
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE scip_symbols (canonical_id TEXT, source_name "
            "TEXT, language TEXT, file TEXT, line_start INTEGER, "
            "line_end INTEGER, kind TEXT, display_name TEXT, "
            "qualified_name TEXT, parent_qualified_name TEXT)")
        for _ in range(2):
            connection.execute(
                "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
                (BUILDER, SOURCE, "x", "core/pkg/builder.scala", 10, 40,
                 "method", "run", "pkg.Builder.run", "pkg.Builder"))
        connection.commit()

        gap = store_truth.grade_definition(connection, expectation())
        assert gap.primary_reason == "ambiguous_identity"


class TestEdgeGrading:
    def test_exact_edge_passes(self, conn):
        assert store_truth.grade_edge(conn, edge()) is None

    def test_ownership_cannot_satisfy_a_call(self, conn):
        conn.execute("DELETE FROM scip_edges")
        conn.execute(
            "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
            (BUILDER, SINK, "contains", "core/pkg/builder.scala", 20,
             "exact"))
        conn.commit()

        gap = store_truth.grade_edge(conn, edge())
        assert gap.primary_reason == "wrong_edge_type"

    def test_reversed_call_is_wrong_direction(self, conn):
        conn.execute("DELETE FROM scip_edges")
        conn.execute(
            "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
            (SINK, BUILDER, "call", "core/pkg/sink.scala", 5, "exact"))
        conn.commit()

        gap = store_truth.grade_edge(conn, edge())
        assert gap.primary_reason == "wrong_edge_direction"

    def test_correct_type_at_wrong_line_is_wrong_site(self, conn):
        gap = store_truth.grade_edge(conn, edge(line=99))
        assert gap.primary_reason == "wrong_edge_site"

    def test_missing_companion_edge_is_absent_edge(self, conn):
        gap = store_truth.grade_edge(conn, edge(edge_type="companion"))
        # The endpoints share a call edge, so a companion expectation is
        # a type failure — a fully absent pair is absent_edge.
        assert gap.primary_reason == "wrong_edge_type"
        missing = store_truth.grade_edge(conn, store_truth.EdgeExpectation(
            "scip-x x src1 0.1 `pkg`/Ghost#", SINK, "companion", "", 0))
        assert missing.primary_reason == "absent_edge"
