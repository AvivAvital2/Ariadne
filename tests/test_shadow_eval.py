"""The blind shadow evaluator separates a gold-blind runner from grading.

Phase A must be blind by construction: its inputs carry no reviewed
fields, its artifacts carry no reviewed fields, and repeated runs are
identical. Phase B computes the store ceiling from the live store and
grades each claim's furthest consecutive stage.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from docgen.catalog_writer import _element_doc_id
from library import Library
from library.scip import init_scip_schema

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "shadow_eval", ROOT / "evaluation" / "chain-benchmark" / "shadow_eval.py")
shadow_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shadow_eval)

SOURCE = "src1"
RUN = "scip-x x src1 0.1 `m`/run()."
HELPER = "scip-x x src1 0.1 `m`/helper()."


@pytest.fixture
def service(tmp_path):
    library = Library(tmp_path / "l.db")
    with library._conn_provider.acquire() as conn:
        init_scip_schema(conn)
        conn.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (RUN, SOURCE, "x", "m.py", 5, 20, "", "", "m.run", ""))
        conn.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (HELPER, SOURCE, "x", "h.py", 3, 9, "", "", "m.helper", ""))
        conn.execute(
            "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
            (RUN, HELPER, "call", "m.py", 8, "exact"))
        conn.commit()
    library.add_document(
        content_type="catalog", title="run", content="Runs the operation.",
        source_files=["m.py"], doc_id=_element_doc_id(SOURCE, "m.run"),
        source_name=SOURCE)
    yield SimpleNamespace(
        library=library,
        config=SimpleNamespace(get_all_source_paths=lambda: {}))
    library.close()


class TestBlindRunner:
    def test_artifact_is_deterministic_and_carries_no_reviewed_fields(
            self, service):
        vector = np.zeros(3072, dtype=np.float32)

        first = shadow_eval.blind_artifact(
            service, 1, "how does run reach helper?", vector, source=SOURCE)
        second = shadow_eval.blind_artifact(
            service, 1, "how does run reach helper?", vector, source=SOURCE)

        first.pop("timings")
        second.pop("timings")
        assert first == second
        banned = {"claims", "anchors", "witnesses", "review",
                  "candidate_paths", "selected_path_ids"}
        assert not banned.intersection(first)
        assert first["stage_hashes"]["raw_pool"]

    def test_runner_source_never_touches_reviewed_gold(self):
        source = (ROOT / "evaluation" / "chain-benchmark" /
                  "shadow_eval.py").read_text()
        run_section = source[
            source.index("def blind_artifact"):source.index(
                "def required_items")]
        assert "gold" not in run_section.replace(
            "The word gold appears in this module only to be excluded", "")

    def test_questions_file_with_reviewed_fields_is_refused(
            self, tmp_path):
        questions = tmp_path / "questions.json"
        questions.write_text(json.dumps(
            [{"id": 1, "question": "q", "claims": []}]))
        cache = tmp_path / "cache.npz"
        np.savez(cache, q1=np.zeros(3072, dtype=np.float32))

        with pytest.raises(SystemExit, match="reviewed fields"):
            shadow_eval.run_blind([
                "--questions", str(questions),
                "--embedding-cache", str(cache),
                "--out", str(tmp_path / "out.json")])


class TestGrader:
    def make_claim(self):
        return {
            "id": "claim-1",
            "anchors": [
                {"target": {"symbol": "m.run"}},
                {"target": {"symbol": "m.helper"}}],
            "review": {
                "selected_path_ids": ["p1"],
                "selected_candidate_by_anchor": {"a": RUN, "b": HELPER}},
            "candidate_paths": [{
                "id": "p1",
                "nodes": [
                    {"qualified_name": "m.run", "canonical_id": RUN},
                    {"qualified_name": "m.helper", "canonical_id": HELPER}],
                "edges": [{
                    "caller_canonical_id": RUN,
                    "callee_canonical_id": HELPER,
                    "edge_type": "call"}]}],
            "witnesses": [{
                "file": "m.py", "line_start": 8, "line_end": 8,
                "contains": ["helper("]}],
        }
    def make_artifact(self, *, menu_has_helper: bool):
        routes = {"R1": ["m.run", "m.helper"] if menu_has_helper
                  else ["m.run"]}
        return {
            "id": 1,
            "seeds": {"pool_clews": ["m.run"], "selected_clews": []},
            "raw_pool": [["m.run", "m.py", 5, 20, "", "", 0, "calls", 1,
                          "descended"],
                         ["m.helper", "h.py", 3, 9, "m.run", "m.py", 8,
                          "calls", 2, "leaf"]],
            "menu": {"routes": routes, "scoped_routes": routes},
            "retained": {
                "symbols": ["m.run", "m.helper"],
                "occurrence_keys": [], "body_symbols": []},
            "materialized_excerpts": [
                ["m.py", 5, 20, "definition_body", "sha", "body"]],
            "ledger": "m.py:8 `    helper()`",
            "final_artifact": "m.py:8 `    helper()`",
        }

    def test_store_ceiling_is_computed_from_the_live_store(self, tmp_path):
        connection = sqlite3.connect(":memory:")
        init_scip_schema(connection)
        connection.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (RUN, SOURCE, "x", "m.py", 5, 20, "", "", "m.run", ""))
        connection.commit()

        items = shadow_eval.required_items(self.make_claim())
        assert shadow_eval.store_recoverable(connection, items) is False

        connection.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (HELPER, SOURCE, "x", "h.py", 3, 9, "", "", "m.helper", ""))
        connection.execute(
            "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
            (RUN, HELPER, "call", "m.py", 8, "exact"))
        connection.commit()
        assert shadow_eval.store_recoverable(connection, items) is True
    def test_claim_reaches_a_stage_only_with_every_required_item(self):
        items = shadow_eval.required_items(self.make_claim())

        complete = shadow_eval.claim_stage_flags(
            self.make_artifact(menu_has_helper=True), items)
        assert complete == {
            "raw": True, "menu": True, "retained": True,
            "materialized": True, "ledger": True, "final": True}

        menu_loss = shadow_eval.claim_stage_flags(
            self.make_artifact(menu_has_helper=False), items)
        assert menu_loss["raw"] is True
        assert menu_loss["menu"] is False


class TestFrontierSemantics:
    def test_retention_without_menu_visibility_stops_the_frontier_at_raw(
            self):
        grader = TestGrader()
        items = shadow_eval.required_items(grader.make_claim())
        flags = shadow_eval.claim_stage_flags(
            grader.make_artifact(menu_has_helper=False), items)

        # Retained by a net while never menu-visible: real information
        # in the flags, but the consecutive frontier stays at raw.
        assert flags["retained"] is True
        assert flags["menu"] is False

    def test_raw_surface_includes_the_clew_candidate_pool(self):
        grader = TestGrader()
        artifact = grader.make_artifact(menu_has_helper=True)
        artifact["raw_pool"] = [row for row in artifact["raw_pool"]
                                if row[0] != "m.run"]
        artifact["seeds"] = {"pool_clews": ["m.run"],
                             "selected_clews": []}
        items = shadow_eval.required_items(grader.make_claim())

        flags = shadow_eval.claim_stage_flags(artifact, items)

        assert flags["raw"] is True


class TestArtifactRecording:
    def test_artifact_records_seed_union_expansion_and_body_plan(
            self, service):
        vector = np.zeros(3072, dtype=np.float32)

        artifact = shadow_eval.blind_artifact(
            service, 1, "how does run reach helper?", vector,
            source=SOURCE)

        assert "seed_union" in artifact
        assert isinstance(artifact["expansion"], dict)
        assert "required" in artifact["body_plan"]
        assert isinstance(artifact["facets"], list)
