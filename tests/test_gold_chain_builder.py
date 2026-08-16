from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from library.scip import init_scip_schema


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "evaluation/chain-benchmark/build_gold_chains.py"
    spec = importlib.util.spec_from_file_location("build_gold_chains", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    init_scip_schema(connection)
    yield connection
    connection.close()


def _symbol(conn, canonical, qualified, file, *, source="src", kind="Method"):
    conn.execute(
        "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
        (canonical, source, "python", file, 1, 3, kind,
         qualified.rsplit(".", 1)[-1], qualified,
         qualified.rsplit(".", 1)[0]))


def _edge(conn, caller, callee, *, line, file="bridge.py"):
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 (caller, callee, "call", file, line, "exact"))


def test_anchor_candidates_prefer_exact_production_symbols(builder, conn):
    _symbol(conn, "exact", "pkg.Replay.recordBatch", "src/Replay.py")
    _symbol(conn, "substring", "pkg.Replay.recordBatchMetrics", "src/Metrics.py")
    _symbol(conn, "test", "pkg.Replay.recordBatch", "tests/ReplayTest.py")

    found = builder.anchor_candidates(
        conn, "src", "recordBatch", "record a replayed batch", ["pkg"], limit=3)

    assert [item["canonical_id"] for item in found] == ["exact", "substring"]
    assert found[0]["match"] == "exact"


def test_candidate_path_discloses_reverse_traversal_and_real_edge_direction(builder, conn):
    _symbol(conn, "left", "pkg.Left.finish", "left.py")
    _symbol(conn, "bridge", "pkg.Bridge.coordinate", "bridge.py")
    _symbol(conn, "right", "pkg.Right.begin", "right.py")
    _edge(conn, "bridge", "left", line=10)
    _edge(conn, "bridge", "right", line=11)
    left = [{"canonical_id": "left"}]
    right = [{"canonical_id": "right"}]

    path = builder.candidate_path(
        conn, "src", left, right, max_depth=3, max_frontier=20)

    assert path["orientation"] == "bidirectional-reference"
    assert [node["qualified_name"] for node in path["nodes"]] == [
        "pkg.Left.finish", "pkg.Bridge.coordinate", "pkg.Right.begin"]
    assert [edge["traversal"] for edge in path["edges"]] == [
        "callee_to_caller", "caller_to_callee"]
    assert [(edge["caller"], edge["callee"]) for edge in path["edges"]] == [
        ("pkg.Bridge.coordinate", "pkg.Left.finish"),
        ("pkg.Bridge.coordinate", "pkg.Right.begin")]
    assert path["all_edges_compiler_verified"]
def test_bidirectional_route_selection_prefers_shorter_reverse_proof(builder, conn):
    for canonical_id in ("left", "step1", "step2", "right"):
        _symbol(conn, canonical_id, f"pkg.{canonical_id}", f"src/{canonical_id}.scala")
    _edge(conn, "left", "step1", line=1)
    _edge(conn, "step1", "step2", line=2)
    _edge(conn, "step2", "right", line=3)
    _edge(conn, "right", "left", line=4)

    path = builder.bidirectional_candidate_path(
        conn, "src",
        [{"canonical_id": "left", "qualified_name": "pkg.left", "rank": 0}],
        [{"canonical_id": "right", "qualified_name": "pkg.right", "rank": 0}],
        max_depth=4, max_frontier=20, search_node_limit=20)

    assert path["orientation"] == "right-to-left"
    assert path["hop_count"] == 1
    assert [node["canonical_id"] for node in path["nodes"]] == ["right", "left"]
def test_coherent_sets_prefer_shorter_proof_before_direction(builder):
    def route(path_id, orientation_penalty, hop_count):
        return {
            "id": path_id,
            "connects": ["left", "right"],
            "endpoint_candidates": {
                "left": {"canonical_id": "left-id"},
                "right": {"canonical_id": "right-id"}},
            "selection_score": {
                "endpoint_rank_sum": 0,
                "orientation_penalty": orientation_penalty,
                "hop_count": hop_count},
            "hop_count": hop_count,
            "proof_errors": [],
        }

    sets = builder.coherent_path_sets(
        ["left", "right"],
        [route("long-forward", 0, 6), route("short-reverse", 1, 1)],
        limit=2)

    assert sets[0]["path_ids"] == ["short-reverse"]


def test_candidate_path_never_crosses_the_requested_source(builder, conn):
    _symbol(conn, "left", "pkg.Left.finish", "left.py")
    _symbol(conn, "foreign", "foreign.Bridge.coordinate", "foreign.py", source="other")
    _symbol(conn, "right", "pkg.Right.begin", "right.py")
    _edge(conn, "left", "foreign", line=10)
    _edge(conn, "foreign", "right", line=11)

    path = builder.candidate_path(
        conn, "src", [{"canonical_id": "left"}], [{"canonical_id": "right"}],
        max_depth=3, max_frontier=20)

    assert path is None


def test_materialization_hashes_every_path_node_and_edge_site(builder, conn, tmp_path):
    for name in ("left", "bridge", "right"):
        (tmp_path / f"{name}.py").write_text("first\nsecond\nthird\n")
    path = {
        "nodes": [
            {"qualified_name": "pkg.Left.finish", "file": "left.py",
             "line_start": 1, "line_end": 3},
            {"qualified_name": "pkg.Bridge.coordinate", "file": "bridge.py",
             "line_start": 1, "line_end": 3},
            {"qualified_name": "pkg.Right.begin", "file": "right.py",
             "line_start": 1, "line_end": 3}],
        "edges": [
            {"caller": "pkg.Bridge.coordinate", "callee": "pkg.Left.finish",
             "edge_type": "call", "file": "bridge.py", "line": 2},
            {"caller": "pkg.Bridge.coordinate", "callee": "pkg.Right.begin",
             "edge_type": "call", "file": "bridge.py", "line": 3}]}

    result = builder.materialize_path(path, source="src", source_root=str(tmp_path))

    definitions = {item["file"] for item in result["excerpts"]
                   if item["kind"] == "definition"}
    assert definitions == {"left.py", "bridge.py", "right.py"}
    assert all(len(item["sha256"]) == 64 for item in result["excerpts"])
    assert not result["gaps"]
def test_review_validation_refuses_unreviewed_or_incomplete_claims(builder, tmp_path):
    candidate = {
        "questions": [{"id": 1, "review": {
            "status": "accepted", "answer": "Reviewed answer"}, "claims": [{
            "id": "claim",
            "anchors": [{"anchor": "Owner", "candidates": [{
                "canonical_id": "owner-id", "qualified_name": "pkg.Owner"}]}],
            "candidate_paths": [{
                "id": "Owner->Terminal",
                "connects": ["Owner"],
                "endpoint_candidates": {
                    "left": {"canonical_id": "owner-id"}},
                "nodes": [{
                    "canonical_id": "owner-id", "qualified_name": "pkg.Owner",
                    "file": "owner.py", "line_start": 1, "line_end": 1}],
                "edges": [], "all_edges_compiler_verified": True,
                "materialization": {"gaps": [], "excerpts": [{
                    "file": "owner.py", "line_start": 1, "line_end": 1,
                    "kind": "definition", "content": "class Owner:",
                    "sha256": "a" * 64}]}}],
            "review": {"status": "pending", "selected_candidate_by_anchor": {},
                       "selected_path_ids": [], "claim_correct": None,
                       "complete": None}}]}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(candidate))
    assert builder.validate_review(path)

    review = candidate["questions"][0]["claims"][0]["review"]
    review.update({"status": "accepted", "selected_candidate_by_anchor": {
        "Owner": "owner-id"}, "selected_path_ids": ["Owner->Terminal"],
        "claim_correct": True, "complete": True})
    path.write_text(json.dumps(candidate))
    assert builder.validate_review(path) == []
def test_candidate_path_rejects_nonproduction_internal_nodes(builder, conn):
    _symbol(conn, "left", "pkg.Left.finish", "src/Left.py")
    _symbol(conn, "test_bridge", "pkg.Bridge.coordinate", "tests/BridgeTest.py")
    _symbol(conn, "right", "pkg.Right.begin", "src/Right.py")
    _edge(conn, "left", "test_bridge", line=10, file="src/Left.py")
    _edge(conn, "test_bridge", "right", line=11, file="tests/BridgeTest.py")

    path = builder.candidate_path(
        conn, "src", [{"canonical_id": "left", "rank": 0}],
        [{"canonical_id": "right", "rank": 0}],
        max_depth=3, max_frontier=20)

    assert path is None


def test_candidate_path_rejects_nonproduction_edge_sites(builder, conn):
    _symbol(conn, "left", "pkg.Left.finish", "src/Left.py")
    _symbol(conn, "bridge", "pkg.Bridge.coordinate", "src/Bridge.py")
    _symbol(conn, "right", "pkg.Right.begin", "src/Right.py")
    _edge(conn, "left", "bridge", line=10, file="tests/BridgeTest.py")
    _edge(conn, "bridge", "right", line=11, file="src/Bridge.py")

    path = builder.candidate_path(
        conn, "src", [{"canonical_id": "left", "rank": 0}],
        [{"canonical_id": "right", "rank": 0}],
        max_depth=3, max_frontier=20)

    assert path is None


def test_anchor_ranking_prefers_claim_context_over_generic_exact_name(builder, conn):
    _symbol(conn, "generic", "pyspark.pandas.Series.align", "python/pyspark/pandas/series.py")
    _symbol(
        conn, "owner", 
        "org.apache.spark.sql.catalyst.analysis.ResolveRowLevelCommandAssignments.alignActions",
        "sql/catalyst/analysis/ResolveRowLevelCommandAssignments.scala")

    found = builder.anchor_candidates(
        conn, "src", "align",
        "ResolveRowLevelCommandAssignments aligns row-level assignments",
        ["spark"], limit=2)

    assert [item["canonical_id"] for item in found] == ["owner", "generic"]
    assert found[0]["context_overlap"] > found[1]["context_overlap"]
    assert [item["rank"] for item in found] == [0, 1]


def test_candidate_path_records_the_actual_ranked_endpoints(builder, conn):
    _symbol(conn, "left_wrong", "pkg.Unrelated.begin", "src/Unrelated.py")
    _symbol(conn, "left_right", "pkg.Expected.begin", "src/Expected.py")
    _symbol(conn, "right", "pkg.Terminal.finish", "src/Terminal.py")
    _edge(conn, "left_right", "right", line=8, file="src/Expected.py")
    left = [
        {"canonical_id": "left_wrong", "qualified_name": "pkg.Unrelated.begin", "rank": 0},
        {"canonical_id": "left_right", "qualified_name": "pkg.Expected.begin", "rank": 1}]
    right = [
        {"canonical_id": "right", "qualified_name": "pkg.Terminal.finish", "rank": 0}]

    path = builder.candidate_path(
        conn, "src", left, right, max_depth=2, max_frontier=20)

    assert path["endpoint_candidates"] == {
        "left": {"canonical_id": "left_right", "qualified_name": "pkg.Expected.begin", "rank": 1},
        "right": {"canonical_id": "right", "qualified_name": "pkg.Terminal.finish", "rank": 0}}


def test_claim_path_coverage_requires_every_anchor_in_one_component(builder):
    anchors = ["entry", "middle", "terminal"]
    complete = builder.claim_path_coverage(anchors, [
        {"id": "entry->middle", "connects": ["entry", "middle"]},
        {"id": "middle->terminal", "connects": ["middle", "terminal"]}])
    incomplete = builder.claim_path_coverage(anchors, [
        {"id": "entry->middle", "connects": ["entry", "middle"]}])

    assert complete == {
        "covered_anchors": anchors, "missing_anchors": [],
        "components": [anchors], "complete": True}
    assert incomplete["covered_anchors"] == ["entry", "middle"]
    assert incomplete["missing_anchors"] == ["terminal"]
    assert incomplete["components"] == [["entry", "middle"], ["terminal"]]
    assert not incomplete["complete"]


def test_review_validation_requires_selected_paths_to_cover_the_claim(builder, tmp_path):
    candidate = {
        "questions": [{"id": 1, "review": {
            "status": "accepted", "answer": "Reviewed answer"}, "claims": [{
            "id": "claim",
            "anchors": [
                {"anchor": "entry", "candidates": [{"canonical_id": "entry-id"}]},
                {"anchor": "middle", "candidates": [{"canonical_id": "middle-id"}]},
                {"anchor": "terminal", "candidates": [{"canonical_id": "terminal-id"}]}],
            "candidate_paths": [
                {"id": "entry->middle", "connects": ["entry", "middle"],
                 "endpoint_candidates": {
                     "left": {"canonical_id": "entry-id"},
                     "right": {"canonical_id": "middle-id"}},
                 "all_edges_compiler_verified": True,
                 "materialization": {"gaps": []}}],
            "review": {
                "status": "accepted", "selected_candidate_by_anchor": {
                    "entry": "entry-id", "middle": "middle-id", "terminal": "terminal-id"},
                "selected_path_ids": ["entry->middle"],
                "claim_correct": True, "complete": True}}]}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(candidate))

    errors = builder.validate_review(path)

    assert any("selected paths do not connect every anchor" in error for error in errors)


def test_review_validation_requires_question_answer_and_review(builder, tmp_path):
    candidate = {"questions": [{"id": 1, "review": {
        "status": "pending", "answer": ""}, "claims": []}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(candidate))

    errors = builder.validate_review(path)

    assert "Q1: question review is not accepted" in errors
    assert "Q1: reviewed answer is empty" in errors
def test_distinct_anchors_cannot_collapse_to_the_same_symbol(builder, conn):
    _symbol(conn, "shared", "pkg.MergeRowsExec.applyInstructions", "src/MergeRowsExec.scala")
    candidate = {
        "canonical_id": "shared",
        "qualified_name": "pkg.MergeRowsExec.applyInstructions",
        "rank": 0}

    path = builder.candidate_path(
        conn, "src", [candidate], [candidate],
        max_depth=2, max_frontier=20)

    assert path is None


def test_type_anchor_prefers_the_type_over_a_contextual_member(builder, conn):
    _symbol(conn, "type", "pkg.MergeRows", "src/MergeRows.scala", kind="Class")
    _symbol(
        conn, "member", "pkg.MergeRowsExec.applyInstructions",
        "src/MergeRowsExec.scala")

    found = builder.anchor_candidates(
        conn, "src", "MergeRows",
        "MergeRowsExec evaluates MergeRows instructions", ["pkg"], limit=2)

    assert [item["canonical_id"] for item in found] == ["type", "member"]
    assert found[0]["anchor_role"] == "type"
    assert found[0]["identity_rank"] < found[1]["identity_rank"]


def test_candidate_paths_preserve_distinct_endpoint_alternatives(builder, conn):
    _symbol(conn, "left_a", "pkg.Preferred.begin", "src/Preferred.py")
    _symbol(conn, "left_b", "pkg.Alternative.begin", "src/Alternative.py")
    _symbol(conn, "bridge", "pkg.Bridge.forward", "src/Bridge.py")
    _symbol(conn, "right", "pkg.Terminal.finish", "src/Terminal.py")
    _edge(conn, "left_a", "bridge", line=4, file="src/Preferred.py")
    _edge(conn, "bridge", "right", line=5, file="src/Bridge.py")
    _edge(conn, "left_b", "right", line=6, file="src/Alternative.py")
    left = [
        {"canonical_id": "left_a", "qualified_name": "pkg.Preferred.begin", "rank": 0},
        {"canonical_id": "left_b", "qualified_name": "pkg.Alternative.begin", "rank": 1}]
    right = [
        {"canonical_id": "right", "qualified_name": "pkg.Terminal.finish", "rank": 0}]

    paths = builder.candidate_paths(
        conn, "src", left, right, max_depth=3, max_frontier=20,
        endpoint_limit=2, path_limit=4)

    assert {
        item["endpoint_candidates"]["left"]["canonical_id"] for item in paths
    } == {"left_a", "left_b"}
    assert [item["alternative_rank"] for item in paths] == list(range(len(paths)))
    assert all(item["selection_score"]["endpoint_rank_sum"] >= 0 for item in paths)


def test_path_proof_requires_hashed_definitions_and_edge_sites(builder):
    path = {
        "nodes": [
            {"canonical_id": "left", "qualified_name": "pkg.Left.begin",
             "file": "src/Left.py", "line_start": 2, "line_end": 4},
            {"canonical_id": "right", "qualified_name": "pkg.Right.finish",
             "file": "src/Right.py", "line_start": 7, "line_end": 9}],
        "edges": [{
            "caller_canonical_id": "left", "callee_canonical_id": "right",
            "caller": "pkg.Left.begin", "callee": "pkg.Right.finish",
            "edge_type": "call", "file": "src/Left.py", "line": 3,
            "compiler_verified": True}],
        "materialization": {"gaps": [], "excerpts": [{
            "file": "src/Left.py", "line_start": 2, "line_end": 2,
            "kind": "definition", "content": "def begin():",
            "sha256": "a" * 64}]}}

    errors = builder.path_proof_errors(path)

    assert "missing definition proof: pkg.Right.finish" in errors
    assert "missing call-site proof: src/Left.py:3" in errors


def test_review_validation_rejects_empty_materialization_proof(builder, tmp_path):
    candidate = {
        "questions": [{"id": 1, "review": {
            "status": "accepted", "answer": "Reviewed answer"}, "claims": [{
            "id": "claim",
            "anchors": [
                {"anchor": "entry", "candidates": [{"canonical_id": "entry-id"}]},
                {"anchor": "terminal", "candidates": [{"canonical_id": "terminal-id"}]}],
            "candidate_paths": [{
                "id": "entry->terminal#1",
                "connects": ["entry", "terminal"],
                "endpoint_candidates": {
                    "left": {"canonical_id": "entry-id"},
                    "right": {"canonical_id": "terminal-id"}},
                "nodes": [
                    {"canonical_id": "entry-id", "qualified_name": "pkg.Entry",
                     "file": "src/Entry.py", "line_start": 1, "line_end": 2},
                    {"canonical_id": "terminal-id", "qualified_name": "pkg.Terminal",
                     "file": "src/Terminal.py", "line_start": 1, "line_end": 2}],
                "edges": [], "all_edges_compiler_verified": True,
                "materialization": {"gaps": [], "excerpts": []}}],
            "review": {
                "status": "accepted",
                "selected_candidate_by_anchor": {
                    "entry": "entry-id", "terminal": "terminal-id"},
                "selected_path_ids": ["entry->terminal#1"],
                "claim_correct": True, "complete": True}}]}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(candidate))

    errors = builder.validate_review(path)

    assert any("missing definition proof" in error for error in errors)
def _review_path(path_id, left_anchor, right_anchor, left_id, right_id, rank=0):
    return {
        "id": path_id, "connects": [left_anchor, right_anchor],
        "endpoint_candidates": {
            "left": {"canonical_id": left_id, "qualified_name": f"pkg.{left_id}", "rank": rank},
            "right": {"canonical_id": right_id, "qualified_name": f"pkg.{right_id}", "rank": rank}},
        "selection_score": {
            "endpoint_rank_sum": rank, "orientation_penalty": 0, "hop_count": 1},
        "orientation": "left-to-right", "hop_count": 1,
        "all_edges_compiler_verified": True,
        "nodes": [
            {"canonical_id": left_id, "qualified_name": f"pkg.{left_id}",
             "file": f"src/{left_id}.py", "line_start": 1, "line_end": 2},
            {"canonical_id": right_id, "qualified_name": f"pkg.{right_id}",
             "file": f"src/{right_id}.py", "line_start": 1, "line_end": 2}],
        "edges": [], "materialization": {"gaps": [], "excerpts": []},
        "proof_errors": []}


def test_coherent_path_sets_reject_conflicting_shared_anchor_endpoints(builder):
    paths = [
        _review_path("a-b1", "a", "b", "a1", "b1"),
        _review_path("b2-c", "b", "c", "b2", "c1")]

    found = builder.coherent_path_sets(["a", "b", "c"], paths, limit=5)

    assert found == []


def test_coherent_path_sets_return_complete_consistent_assignments(builder):
    paths = [
        _review_path("a-b1", "a", "b", "a1", "b1"),
        _review_path("b1-c", "b", "c", "b1", "c1"),
        _review_path("a-b2", "a", "b", "a1", "b2", rank=2),
        _review_path("b2-c", "b", "c", "b2", "c1", rank=2)]

    found = builder.coherent_path_sets(["a", "b", "c"], paths, limit=5)

    assert len(found) == 2
    assert found[0]["selected_candidate_by_anchor"] == {
        "a": "a1", "b": "b1", "c": "c1"}
    assert found[0]["path_ids"] == ["a-b1", "b1-c"]
    assert found[0]["complete"]
    assert found[0]["score"] < found[1]["score"]


def test_candidate_path_rejects_example_only_bridge(builder, conn):
    _symbol(conn, "left", "pkg.Left.finish", "src/Left.py")
    _symbol(conn, "example", "pkg.Example.coordinate", "examples/Example.py")
    _symbol(conn, "right", "pkg.Right.begin", "src/Right.py")
    _edge(conn, "left", "example", line=10, file="src/Left.py")
    _edge(conn, "example", "right", line=11, file="examples/Example.py")

    path = builder.candidate_path(
        conn, "src", [{"canonical_id": "left", "rank": 0}],
        [{"canonical_id": "right", "rank": 0}],
        max_depth=3, max_frontier=20)

    assert path is None


def test_anchor_recall_prioritizes_repo_scoped_rows_before_global_limit(builder, conn):
    for index in range(5001):
        _symbol(
            conn, f"generic-{index}", f"unrelated.Validator.validateExtra{index}",
            f"src/unrelated/Validator{index}.py")
    _symbol(
        conn, "correct", "io.delta.GeneratedColumn.validate",
        "src/delta/GeneratedColumn.scala")

    found = builder.anchor_candidates(
        conn, "src", "validate",
        "Delta validates generated-column metadata", ["delta"], limit=1)

    assert found[0]["canonical_id"] == "correct"
    assert found[0]["repo_hints"] == ["delta"]


def test_render_review_queue_exposes_endpoints_routes_and_proof(builder):
    report = {"status": "candidate-oracle; not gold until reviewed", "questions": [{
        "id": 9, "question": "How does it flow?", "claims": [{
            "id": "flow", "assertion": "Entry reaches Terminal.",
            "candidate_coverage": {
                "complete": True, "missing_anchors": [], "components": [["Entry", "Terminal"]]},
            "coherent_path_sets": [{
                "path_ids": ["Entry->Terminal#1"],
                "selected_candidate_by_anchor": {"Entry": "entry-id", "Terminal": "terminal-id"},
                "score": 1, "complete": True}],
            "anchors": [
                {"anchor": "Entry", "candidates": [{
                    "rank": 0, "canonical_id": "entry-id",
                    "qualified_name": "pkg.Entry", "file": "src/Entry.py",
                    "line_start": 4, "context_overlap": 2, "match": "exact"}]},
                {"anchor": "Terminal", "candidates": [{
                    "rank": 0, "canonical_id": "terminal-id",
                    "qualified_name": "pkg.Terminal", "file": "src/Terminal.py",
                    "line_start": 8, "context_overlap": 2, "match": "exact"}]}],
            "candidate_paths": [{
                "id": "Entry->Terminal#1", "connects": ["Entry", "Terminal"],
                "orientation": "left-to-right", "hop_count": 1,
                "endpoint_candidates": {
                    "left": {"canonical_id": "entry-id", "qualified_name": "pkg.Entry"},
                    "right": {"canonical_id": "terminal-id", "qualified_name": "pkg.Terminal"}},
                "proof_errors": ["missing call-site proof: src/Entry.py:7"]}]}]}]}

    rendered = builder.render_review_queue(report)

    assert "# Gold-chain candidate review" in rendered
    assert "Q9: How does it flow?" in rendered
    assert "Entry reaches Terminal." in rendered
    assert "pkg.Entry" in rendered
    assert "Entry->Terminal#1" in rendered
    assert "missing call-site proof: src/Entry.py:7" in rendered
    assert "Coherent complete sets: 1" in rendered
def test_render_review_queue_exposes_claim_witness(builder):
    report = {"questions": [{
        "id": 1, "question": "Why skip replay?",
        "claims": [{
            "id": "guard", "assertion": "Committed batches return early.",
            "anchors": [], "candidate_paths": [],
            "candidate_coverage": {"complete": True, "missing_anchors": [], "components": []},
            "coherent_path_sets": [],
            "witnesses": [{
                "id": "already-committed", "file": "src/Sink.scala",
                "line_start": 10, "line_end": 12,
                "contains": ["return false"],
                "materialization": {"gaps": [], "excerpts": [{
                    "content": "if (done) {\n  return false\n}",
                    "sha256": "a" * 64}]},
                "proof_errors": [],
            }],
        }],
    }]}

    rendered = builder.render_review_queue(report)

    assert "Witness already-committed" in rendered
    assert "src/Sink.scala:10-12" in rendered
    assert "return false" in rendered
def test_review_validation_rejects_unproven_claim_witness(builder, tmp_path):
    candidate = {"questions": [{
        "id": 1,
        "review": {"status": "accepted", "answer": "Reviewed"},
        "claims": [{
            "id": "guard", "anchors": [], "candidate_paths": [],
            "witnesses": [{
                "id": "already-committed",
                "materialization": {"gaps": ["source unavailable"], "excerpts": []},
                "proof_errors": ["missing required fragment: return false"],
            }],
            "review": {
                "status": "accepted",
                "selected_candidate_by_anchor": {},
                "selected_path_ids": [],
                "claim_correct": True, "complete": True,
            },
        }],
    }]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(candidate))

    errors = builder.validate_review(path)

    assert any("witness already-committed: materialization gap" in error for error in errors)
    assert any("witness already-committed: missing required fragment" in error for error in errors)


def test_review_validation_names_incoherent_selected_endpoint_set(builder, tmp_path):
    paths = [
        _review_path("a-b1", "a", "b", "a1", "b1"),
        _review_path("b2-c", "b", "c", "b2", "c1")]
    for path_row in paths:
        excerpts = []
        for node in path_row["nodes"]:
            excerpts.append({
                "file": node["file"], "line_start": 1, "line_end": 1,
                "kind": "definition", "content": "symbol",
                "sha256": "a" * 64})
        path_row["materialization"]["excerpts"] = excerpts
    candidate = {
        "questions": [{"id": 1, "review": {
            "status": "accepted", "answer": "Reviewed answer"}, "claims": [{
            "id": "claim",
            "anchors": [
                {"anchor": "a", "candidates": [{"canonical_id": "a1"}]},
                {"anchor": "b", "candidates": [
                    {"canonical_id": "b1"}, {"canonical_id": "b2"}]},
                {"anchor": "c", "candidates": [{"canonical_id": "c1"}]}],
            "candidate_paths": paths,
            "review": {
                "status": "accepted",
                "selected_candidate_by_anchor": {"a": "a1", "b": "b1", "c": "c1"},
                "selected_path_ids": ["a-b1", "b2-c"],
                "claim_correct": True, "complete": True}}]}]}
    path = tmp_path / "review.json"
    path.write_text(json.dumps(candidate))

    errors = builder.validate_review(path)

    assert any("selected paths have no coherent endpoint assignment" in error
               for error in errors)
def test_exact_type_identity_outranks_multi_repo_suffix(builder, conn):
    _symbol(
        conn, "delta", "org.apache.spark.sql.delta.files.DeltaFileFormatWriter",
        "delta/DeltaFileFormatWriter.scala", kind="Class")
    _symbol(
        conn, "spark", "org.apache.spark.sql.execution.datasources.FileFormatWriter",
        "spark/FileFormatWriter.scala", kind="Class")

    found = builder.anchor_candidates(
        conn, "src", "FileFormatWriter",
        "DeltaFileFormatWriter forks Spark FileFormatWriter",
        ["delta", "spark"], limit=2)

    assert [item["canonical_id"] for item in found] == ["spark", "delta"]
    assert found[0]["identity_rank"] == 0
    assert found[1]["identity_rank"] > 0
def test_directed_path_can_leave_owner_through_owned_member(builder, conn):
    _symbol(conn, "owner", "pkg.Owner", "src/Owner.scala", kind="Class")
    _symbol(conn, "member", "pkg.Owner.run", "src/Owner.scala")
    _symbol(conn, "terminal", "pkg.Terminal", "src/Terminal.scala", kind="Class")
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("owner", "member", "contains", "src/Owner.scala", 4, "exact"))
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("member", "terminal", "type_ref", "src/Owner.scala", 7, "exact"))

    path = builder.candidate_path(
        conn, "src",
        [{"canonical_id": "owner", "qualified_name": "pkg.Owner", "rank": 0}],
        [{"canonical_id": "terminal", "qualified_name": "pkg.Terminal", "rank": 0}],
        max_depth=3, max_frontier=20)

    assert [node["canonical_id"] for node in path["nodes"]] == [
        "owner", "member", "terminal"]
    assert [edge["edge_type"] for edge in path["edges"]] == [
        "contains", "type_ref"]
    assert path["edges"][0]["compiler_source"] == "scip-parent"
    assert path["edges"][0]["caller"] == "pkg.Owner"
    assert path["edges"][0]["callee"] == "pkg.Owner.run"
def test_undirected_path_connects_sibling_members_through_owner(builder, conn):
    _symbol(conn, "owner", "pkg.Owner", "src/Owner.scala", kind="Class")
    _symbol(conn, "left", "pkg.Owner.left", "src/Owner.scala")
    _symbol(conn, "right", "pkg.Owner.right", "src/Owner.scala")
    conn.executemany(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        [
            ("owner", "left", "contains", "src/Owner.scala", 4, "exact"),
            ("owner", "right", "contains", "src/Owner.scala", 8, "exact"),
        ])

    path = builder.candidate_path(
        conn, "src",
        [{"canonical_id": "left", "qualified_name": "pkg.Owner.left", "rank": 0}],
        [{"canonical_id": "right", "qualified_name": "pkg.Owner.right", "rank": 0}],
        max_depth=2, max_frontier=20)

    assert path["orientation"] == "bidirectional-reference"
    assert [node["canonical_id"] for node in path["nodes"]] == [
        "left", "owner", "right"]
    assert [edge["edge_type"] for edge in path["edges"]] == [
        "contains", "contains"]
    assert [edge["traversal"] for edge in path["edges"]] == [
        "callee_to_caller", "caller_to_callee"]


def test_containment_edge_is_proven_by_child_definition(builder):
    path = {
        "nodes": [
            {"canonical_id": "owner", "qualified_name": "pkg.Owner",
             "file": "src/Owner.scala", "line_start": 1, "line_end": 10},
            {"canonical_id": "member", "qualified_name": "pkg.Owner.run",
             "file": "src/Owner.scala", "line_start": 4, "line_end": 6}],
        "edges": [{
            "caller_canonical_id": "owner", "callee_canonical_id": "member",
            "caller": "pkg.Owner", "callee": "pkg.Owner.run",
            "edge_type": "contains", "file": "src/Owner.scala", "line": 4,
            "compiler_verified": True, "compiler_source": "scip-parent",
            "traversal": "caller_to_callee"}],
        "materialization": {"gaps": [], "excerpts": [
            {"file": "src/Owner.scala", "line_start": 1, "line_end": 1,
             "kind": "definition", "content": "class Owner",
             "sha256": "a" * 64},
            {"file": "src/Owner.scala", "line_start": 4, "line_end": 4,
             "kind": "definition", "content": "def run",
             "sha256": "a" * 64}]}}

    assert builder.path_proof_errors(path) == []
def test_candidate_paths_cache_reuses_shared_adjacency(builder, conn):
    _symbol(conn, "left_a", "pkg.LeftA.begin", "src/LeftA.py")
    _symbol(conn, "left_b", "pkg.LeftB.begin", "src/LeftB.py")
    _symbol(conn, "shared", "pkg.Shared.forward", "src/Shared.py")
    _symbol(conn, "right", "pkg.Right.finish", "src/Right.py")
    _edge(conn, "left_a", "shared", line=2, file="src/LeftA.py")
    _edge(conn, "left_b", "shared", line=2, file="src/LeftB.py")
    _edge(conn, "shared", "right", line=3, file="src/Shared.py")
    statements = []
    conn.set_trace_callback(statements.append)

    paths = builder.candidate_paths(
        conn, "src",
        [
            {"canonical_id": "left_a", "qualified_name": "pkg.LeftA.begin", "rank": 0},
            {"canonical_id": "left_b", "qualified_name": "pkg.LeftB.begin", "rank": 1}],
        [{"canonical_id": "right", "qualified_name": "pkg.Right.finish", "rank": 0}],
        max_depth=3, max_frontier=20, endpoint_limit=2, path_limit=2)

    edge_reads = [
        statement for statement in statements
        if "FROM scip_edges" in statement]
    assert len(paths) == 2
    # One extra read proves that no shorter reverse edge beats the cached route.
    assert len(edge_reads) <= 4, ",".join(
        path["endpoint_candidates"]["left"]["canonical_id"]
        + ":" + str(path.get("search", {}).get("expanded_nodes"))
        + ":" + str(path.get("search", {}).get("exhausted"))
        for path in paths)
def test_cache_miss_does_not_multiply_edge_bound(builder, conn, monkeypatch):
    calls = []

    def fake_loader(conn_arg, source, frontier, *, incoming, bound):
        calls.append((set(frontier), bound))
        return []

    monkeypatch.setattr(builder, "_load_scoped_edges", fake_loader)

    builder._scoped_edges(
        conn, "src", {"left", "right"}, incoming=False, bound=10,
        edge_cache={})

    assert calls == [({"left"}, 10), ({"right"}, 10)]
def test_parse_claim_filter_targets_exact_claims(builder):
    selected = builder.parse_claim_filter(
        "10:spark-api,67:stable-query-id,67:transaction-record")

    assert selected == {
        (10, "spark-api"),
        (67, "stable-query-id"),
        (67, "transaction-record")}
    assert builder.claim_selected(10, "spark-api", selected)
    assert not builder.claim_selected(10, "delta-api-convergence", selected)
    assert builder.claim_selected(10, "anything", None)


def test_parse_claim_filter_rejects_malformed_selector(builder):
    with pytest.raises(ValueError, match="QID:claim-id"):
        builder.parse_claim_filter("spark-api")
def test_candidate_paths_stops_after_best_path_limit(builder, conn, monkeypatch):
    calls = []

    def fake_bidirectional(
            conn_arg, source, left, right, *, max_depth, max_frontier,
            edge_cache=None, search_node_limit=None, excluded_pairs=None):
        calls.append((left[0]["canonical_id"], right[0]["canonical_id"]))
        return {
            "orientation": "left-to-right",
            "nodes": [
                {"canonical_id": left[0]["canonical_id"]},
                {"canonical_id": right[0]["canonical_id"]}],
            "edges": [], "hop_count": 1,
            "endpoint_candidates": {
                "left": left[0], "right": right[0]},
            "search": {
                "algorithm": "bidirectional-dijkstra",
                "expanded_nodes": 2},
            "all_edges_compiler_verified": True}

    monkeypatch.setattr(
        builder, "bidirectional_candidate_path", fake_bidirectional)
    left = [
        {"canonical_id": "left-0", "rank": 0},
        {"canonical_id": "left-1", "rank": 1}]
    right = [
        {"canonical_id": "right-0", "rank": 0},
        {"canonical_id": "right-1", "rank": 1}]

    paths = builder.candidate_paths(
        conn, "src", left, right, max_depth=3, max_frontier=20,
        endpoint_limit=2, path_limit=1, search_node_limit=10)

    assert len(paths) == 1
    assert calls == [("left-0", "right-0")]


def test_search_node_limit_moves_to_next_ranked_endpoint(builder, conn):
    _symbol(conn, "wrong", "pkg.Wrong.begin", "src/Wrong.py")
    _symbol(conn, "good", "pkg.Good.begin", "src/Good.py")
    _symbol(conn, "right", "pkg.Right.finish", "src/Right.py")
    for index in range(10):
        canonical_id = f"dead-{index}"
        _symbol(conn, canonical_id, f"pkg.Dead{index}.step", f"src/Dead{index}.py")
        _edge(conn, "wrong", canonical_id, line=index + 1, file="src/Wrong.py")
    _edge(conn, "good", "right", line=2, file="src/Good.py")

    paths = builder.candidate_paths(
        conn, "src",
        [
            {"canonical_id": "wrong", "qualified_name": "pkg.Wrong.begin", "rank": 0},
            {"canonical_id": "good", "qualified_name": "pkg.Good.begin", "rank": 1}],
        [{"canonical_id": "right", "qualified_name": "pkg.Right.finish", "rank": 0}],
        max_depth=6, max_frontier=20, endpoint_limit=2, path_limit=1,
        search_node_limit=1)

    assert len(paths) == 1
    assert paths[0]["endpoint_candidates"]["left"]["canonical_id"] == "good"
def test_bidirectional_search_meets_before_expanding_owner_fanout(builder, conn):
    _symbol(conn, "owner", "pkg.Owner", "src/Owner.scala", kind="Class")
    _symbol(conn, "terminal", "pkg.Terminal", "src/Terminal.scala", kind="Class")
    for index in range(50):
        member = f"member-{index}"
        _symbol(
            conn, member, f"pkg.Owner.member{index}",
            "src/Owner.scala")
        conn.execute(
            "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
            ("owner", member, "contains", "src/Owner.scala", index + 2, "exact"))
    _symbol(conn, "bridge", "pkg.Owner.targetBridge", "src/Owner.scala")
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("owner", "bridge", "contains", "src/Owner.scala", 70, "exact"))
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("bridge", "terminal", "type_ref", "src/Owner.scala", 70, "exact"))

    path = builder.bidirectional_candidate_path(
        conn, "src",
        [{"canonical_id": "owner", "qualified_name": "pkg.Owner", "rank": 0}],
        [{"canonical_id": "terminal", "qualified_name": "pkg.Terminal", "rank": 0}],
        max_depth=3, max_frontier=80, edge_cache={},
        search_node_limit=6)

    assert [node["canonical_id"] for node in path["nodes"]] == [
        "owner", "bridge", "terminal"]
    assert [edge["edge_type"] for edge in path["edges"]] == [
        "contains", "type_ref"]
    assert path["search"]["algorithm"] == "bidirectional-dijkstra"
    assert path["search"]["expanded_nodes"] <= 6


def test_candidate_paths_requests_one_multisource_search_per_result(
        builder, conn, monkeypatch):
    calls = []

    def fake_bidirectional(
            conn_arg, source, left, right, *, max_depth, max_frontier,
            edge_cache=None, search_node_limit=None, excluded_pairs=None):
        calls.append(set(excluded_pairs or set()))
        if excluded_pairs:
            return None
        return {
            "orientation": "left-to-right",
            "nodes": [
                {"canonical_id": left[0]["canonical_id"]},
                {"canonical_id": right[0]["canonical_id"]}],
            "edges": [], "hop_count": 1,
            "endpoint_candidates": {
                "left": left[0], "right": right[0]},
            "search": {
                "algorithm": "bidirectional-dijkstra",
                "expanded_nodes": 2},
            "all_edges_compiler_verified": True}

    monkeypatch.setattr(
        builder, "bidirectional_candidate_path", fake_bidirectional)

    paths = builder.candidate_paths(
        conn, "src",
        [
            {"canonical_id": "left-0", "rank": 0},
            {"canonical_id": "left-1", "rank": 1},
            {"canonical_id": "left-2", "rank": 2}],
        [
            {"canonical_id": "right-0", "rank": 0},
            {"canonical_id": "right-1", "rank": 1},
            {"canonical_id": "right-2", "rank": 2}],
        max_depth=3, max_frontier=20, endpoint_limit=3,
        path_limit=1, search_node_limit=20)

    assert len(paths) == 1
    assert calls == [set()]
def test_same_named_companion_cannot_borrow_type_members(builder, conn):
    _symbol(conn, "owner-type", "pkg.Owner", "src/Owner.scala", kind="Class")
    _symbol(conn, "owner-term", "pkg.Owner", "src/Owner.scala", kind="Object")
    _symbol(conn, "member", "pkg.Owner.run", "src/Owner.scala")
    _symbol(conn, "terminal", "pkg.Terminal", "src/Terminal.scala", kind="Class")
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("owner-type", "member", "contains", "src/Owner.scala", 4, "exact"))
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("member", "terminal", "type_ref", "src/Owner.scala", 6, "exact"))

    type_path = builder.bidirectional_candidate_path(
        conn, "src",
        [{"canonical_id": "owner-type", "qualified_name": "pkg.Owner", "rank": 0}],
        [{"canonical_id": "terminal", "qualified_name": "pkg.Terminal", "rank": 0}],
        max_depth=3, max_frontier=20)
    companion_path = builder.bidirectional_candidate_path(
        conn, "src",
        [{"canonical_id": "owner-term", "qualified_name": "pkg.Owner", "rank": 0}],
        [{"canonical_id": "terminal", "qualified_name": "pkg.Terminal", "rank": 0}],
        max_depth=3, max_frontier=20)

    assert [node["canonical_id"] for node in type_path["nodes"]] == [
        "owner-type", "member", "terminal"]
    assert companion_path is None
def test_structured_anchor_targets_exact_qualified_symbol(builder, conn):
    _symbol(
        conn, "wrong", "pkg.state.StateStoreId.toString",
        "src/StateStore.scala")
    _symbol(
        conn, "correct", "pkg.streaming.StreamExecution.id",
        "src/StreamExecution.scala", kind="Field")

    found = builder.anchor_candidates(
        conn,
        "src",
        {
            "anchor": "stable-query-id",
            "symbol": "pkg.streaming.StreamExecution.id",
            "strict": True,
        },
        "StreamExecution publishes its stable id",
        ["streaming"],
        limit=6,
    )

    assert [item["canonical_id"] for item in found] == ["correct"]
    assert found[0]["target_match"] == "qualified-exact"


def test_structured_anchor_can_constrain_file_kind_and_source_line(builder, conn):
    _symbol(
        conn, "wrong-file", "pkg.Owner.run",
        "src/Other.scala", kind="Method")
    _symbol(
        conn, "correct", "pkg.Owner.run",
        "src/Owner.scala", kind="Method")
    conn.execute(
        "UPDATE scip_symbols SET line_start=10,line_end=20 "
        "WHERE canonical_id='correct'")

    found = builder.anchor_candidates(
        conn,
        "src",
        {
            "anchor": "owner-run",
            "symbol": "pkg.Owner.run",
            "strict": True,
            "file": "src/Owner.scala",
            "kind": "Method",
            "line": 15,
        },
        "Owner runs the operation",
        ["pkg"],
        limit=6,
    )

    assert [item["canonical_id"] for item in found] == ["correct"]
def test_structured_anchor_recovers_duplicate_definition_from_caller_site(
        builder, conn):
    _symbol(
        conn, "run", "pkg.Owner.run",
        "connect/src/Owner.scala")
    _symbol(
        conn, "terminal", "pkg.Terminal.finish",
        "src/Terminal.scala")
    _edge(
        conn, "run", "terminal", line=11,
        file="standalone/src/Owner.scala")

    found = builder.anchor_candidates(
        conn, "src",
        {
            "anchor": "standalone-run",
            "symbol": "pkg.Owner.run",
            "file": "standalone/src/Owner.scala",
            "line": 10,
        },
        "the standalone owner runs the terminal",
        ["standalone"],
        limit=1,
    )

    assert len(found) == 1
    assert found[0]["file"] == "standalone/src/Owner.scala"
    assert found[0]["line_start"] == 10
    assert found[0]["line_end"] == 11
    path = builder.bidirectional_candidate_path(
        conn, "src", found,
        [{
            "canonical_id": "terminal",
            "qualified_name": "pkg.Terminal.finish",
            "file": "src/Terminal.scala",
            "line_start": 1,
            "line_end": 3,
            "rank": 0,
        }],
        max_depth=2,
        max_frontier=20,
    )
    assert path["nodes"][0]["file"] == "standalone/src/Owner.scala"
    assert path["nodes"][0]["line_start"] == 10


def test_normalize_anchor_spec_separates_label_from_lookup_target(builder):
    legacy = builder.normalize_anchor_spec("run")
    structured = builder.normalize_anchor_spec({
        "anchor": "publish-query-id",
        "symbol": "pkg.StreamExecution.runStream",
        "file": "src/StreamExecution.scala",
    })

    assert legacy == {
        "anchor": "run",
        "symbol": "run",
        "strict": False,
    }
    assert structured["anchor"] == "publish-query-id"
    assert structured["symbol"] == "pkg.StreamExecution.runStream"
    assert structured["strict"] is True
    assert structured["file"] == "src/StreamExecution.scala"


def test_structured_endpoints_recover_same_site_scip_witness(builder, conn):
    _symbol(
        conn, "run", "pkg.StreamExecution.runStream",
        "src/StreamExecution.scala")
    _symbol(
        conn, "id", "pkg.StreamExecution.id",
        "src/StreamExecution.scala", kind="Field")
    _symbol(
        conn, "key", "pkg.StreamExecution.QUERY_ID_KEY",
        "src/StreamExecution.scala", kind="Field")
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("run", "id", "type_ref", "src/StreamExecution.scala", 12, "exact"))
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("run", "key", "type_ref", "src/StreamExecution.scala", 12, "exact"))

    left = builder.anchor_candidates(
        conn, "src",
        {"anchor": "stable-id", "symbol": "pkg.StreamExecution.id"},
        "publish the stable id", ["pkg"], limit=3)
    right = builder.anchor_candidates(
        conn, "src",
        {"anchor": "query-key", "symbol": "pkg.StreamExecution.QUERY_ID_KEY"},
        "publish the stable id", ["pkg"], limit=3)
    paths = builder.candidate_paths(
        conn, "src", left, right,
        max_depth=3, max_frontier=20, endpoint_limit=3, path_limit=1)

    assert len(paths) == 1
    assert [node["canonical_id"] for node in paths[0]["nodes"]] == [
        "id", "run", "key"]
    assert [edge["line"] for edge in paths[0]["edges"]] == [12, 12]
def test_q67_oracle_uses_compiler_addressable_anchors():
    path = Path(__file__).parents[1] / "evaluation/chain-benchmark/requirements.json"
    requirements = json.loads(path.read_text())
    question = next(item for item in requirements if item["id"] == 67)
    claims = {claim["id"]: claim for claim in question["claims"]}

    expected = {
        "stable-query-id": {
            "org.apache.spark.sql.execution.streaming.StreamExecution.streamMetadata",
            "org.apache.spark.sql.execution.streaming.StreamExecution.id",
            "org.apache.spark.sql.execution.streaming.StreamExecution.runStream",
            "org.apache.spark.sql.execution.streaming.StreamExecution.QUERY_ID_KEY",
            "org.apache.spark.sql.delta.sources.DeltaSink.queryId",
        },
        "replay-check": {
            "org.apache.spark.sql.delta.sources.DeltaSink.addBatch",
            "org.apache.spark.sql.delta.sources.DeltaSink.addBatchWithStatusImpl",
            "org.apache.spark.sql.delta.OptimisticTransactionImpl.txnVersion",
            "org.apache.spark.sql.delta.sources.DeltaSink.queryId",
        },
        "transaction-record": {
            "org.apache.spark.sql.delta.sources.DeltaSink.addBatchWithStatusImpl",
            "org.apache.spark.sql.delta.sources.DeltaSink.PendingTxn.commit",
            "org.apache.spark.sql.delta.actions.SetTransaction",
            "org.apache.spark.sql.delta.sources.DeltaSink.queryId",
            "org.apache.spark.sql.delta.sources.DeltaSink.PendingTxn.batchId",
            "org.apache.spark.sql.delta.OptimisticTransactionImpl.commit",
        },
    }

    assert set(claims) == set(expected)
    for claim_id, symbols in expected.items():
        anchors = claims[claim_id]["anchors"]
        assert all(isinstance(anchor, dict) for anchor in anchors)
        assert {anchor["symbol"] for anchor in anchors} == symbols
        assert all({"file", "line"} <= set(anchor) for anchor in anchors)
def test_q67_oracle_declares_control_flow_witnesses():
    path = Path(__file__).parents[1] / "evaluation/chain-benchmark/requirements.json"
    requirements = json.loads(path.read_text())
    question = next(item for item in requirements if item["id"] == 67)
    claims = {claim["id"]: claim for claim in question["claims"]}

    stable = {
        witness["id"]: witness
        for witness in claims["stable-query-id"]["witnesses"]}
    replay = {
        witness["id"]: witness
        for witness in claims["replay-check"]["witnesses"]}
    transaction = {
        witness["id"]: witness
        for witness in claims["transaction-record"]["witnesses"]}

    assert {"StreamMetadata.read", "StreamMetadata.write"} <= set(
        stable["checkpoint-stream-metadata"]["contains"])
    assert {
        "txn.txnVersion(queryId)", "currentVersion >= batchId", "return false"
    } <= set(replay["already-committed-guard"]["contains"])
    assert any(
        "SetTransaction(appId = queryId, version = batchId" in fragment
        for fragment in transaction[
            "pending-transaction-record-is-committed"]["contains"])
def test_merge_update_oracles_use_exact_symbols_and_semantic_witnesses():
    path = Path(__file__).parents[1] / "evaluation/chain-benchmark/requirements.json"
    requirements = {
        item["id"]: item for item in json.loads(path.read_text())
    }
    expected = {
        (4, "spark-join-classify"): {
            "org.apache.spark.sql.execution.datasources.v2.MergeRowsExec.doExecute",
            "org.apache.spark.sql.execution.datasources.v2.MergeRowsExec.processPartition",
        },
        (4, "delta-join-classify"): {
            "org.apache.spark.sql.delta.commands.merge.ClassicMergeExecutor.writeAllChanges",
            "org.apache.spark.sql.delta.commands.merge.MergeOutputGeneration.generateWriteAllChangesOutputCols",
        },
        (6, "spark-rewrite"): {
            "org.apache.spark.sql.catalyst.plans.logical.MergeIntoTable",
            "org.apache.spark.sql.catalyst.analysis.RewriteMergeIntoTable.apply",
        },
        (6, "delta-diversion"): {
            "org.apache.spark.sql.delta.DeltaAnalysis.apply",
            "org.apache.spark.sql.catalyst.plans.logical.DeltaMergeInto",
            "org.apache.spark.sql.delta.PreprocessTableMerge.apply",
        },
        (8, "delta-values"): {
            "org.apache.spark.sql.delta.PreprocessTableMerge.resolveImplicitColumns",
            "org.apache.spark.sql.delta.GeneratedColumn.getGenerationExpression",
            "org.apache.spark.sql.delta.IdentityColumn.createIdentityColumnGenerationExpr",
        },
        (8, "spark-values"): {
            "org.apache.spark.sql.catalyst.analysis.ResolveRowLevelCommandAssignments.apply",
            "org.apache.spark.sql.catalyst.analysis.AssignmentUtils.alignInsertAssignments",
        },
        (10, "delta-api-convergence"): {
            "io.delta.tables.DeltaTable.merge",
            "io.delta.tables.DeltaMergeBuilder.execute",
            "io.delta.tables.DeltaMergeBuilder.mergePlan",
            "org.apache.spark.sql.catalyst.plans.logical.DeltaMergeInto",
            "org.apache.spark.sql.delta.PreprocessTableMerge.apply","io.delta.tables.DeltaMergeBuilder"
        },
        (10, "spark-api"): {
            "org.apache.spark.sql.classic.MergeIntoWriter.merge",
            "org.apache.spark.sql.catalyst.plans.logical.MergeIntoTable",
        },
        (12, "delta-update"): {
            "org.apache.spark.sql.delta.DeltaAnalysis.apply",
            "org.apache.spark.sql.catalyst.plans.logical.DeltaUpdateTable",
            "org.apache.spark.sql.delta.PreprocessTableUpdate.toCommand",
            "org.apache.spark.sql.delta.UpdateExpressionsSupport.generateUpdateExpressions",
            "org.apache.spark.sql.delta.commands.UpdateCommand",
        },
        (12, "spark-update"): {
            "org.apache.spark.sql.catalyst.analysis.ResolveRowLevelCommandAssignments.apply",
            "org.apache.spark.sql.catalyst.analysis.AssignmentUtils.alignUpdateAssignments",
            "org.apache.spark.sql.catalyst.plans.logical.UpdateTable",
            "org.apache.spark.sql.catalyst.analysis.RewriteUpdateTable.apply",
            "org.apache.spark.sql.catalyst.plans.logical.ReplaceData",
            "org.apache.spark.sql.catalyst.plans.logical.WriteDelta",
        },
        (13, "spark-pruning"): {
            "org.apache.spark.sql.execution.dynamicpruning.RowLevelOperationRuntimeGroupFiltering.apply",
            "org.apache.spark.sql.connector.read.SupportsRuntimeV2Filtering",
        },
        (13, "delta-pruning"): {
            "org.apache.spark.sql.delta.commands.merge.ClassicMergeExecutor.findTouchedFiles",
            "org.apache.spark.sql.delta.OptimisticTransactionImpl.filterFiles",
        },
    }

    found = {}
    for question_id in (4, 6, 8, 10, 12, 13):
        for claim in requirements[question_id]["claims"]:
            anchors = claim["anchors"]
            assert all(isinstance(anchor, dict) for anchor in anchors)
            assert len({anchor["anchor"] for anchor in anchors}) == len(anchors)
            assert claim.get("witnesses"), (
                question_id, claim["id"], "missing semantic witness")
            found[(question_id, claim["id"])] = {
                anchor["symbol"] for anchor in anchors
            }

    assert found == expected
def test_structured_exact_anchor_avoids_wildcard_catalog_scan(builder, conn):
    _symbol(conn, "correct", "pkg.StreamExecution.id", "src/StreamExecution.scala")
    statements = []
    conn.set_trace_callback(statements.append)

    found = builder.anchor_candidates(
        conn, "src",
        {"anchor": "stable-id", "symbol": "pkg.StreamExecution.id"},
        "stable stream id", ["spark"], limit=3)

    assert [item["canonical_id"] for item in found] == ["correct"]
    catalog_reads = [statement.lower() for statement in statements
                     if "from scip_symbols" in statement.lower()]
    assert any("lower(qualified_name)=" in statement
               for statement in catalog_reads)
    assert all(" like " not in statement for statement in catalog_reads)
def test_claim_witnesses_require_exact_hashed_source_sections(builder, tmp_path):
    source_file = tmp_path / "src" / "Guard.scala"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        "def write(batchId: Long) = {\n"
        "  val currentVersion = txnVersion(queryId)\n"
        "  if (currentVersion >= batchId) {\n"
        "    return false\n"
        "  }\n"
        "}\n")

    witnesses = builder.materialize_claim_witnesses(
        [{
            "id": "already-committed-guard",
            "file": "src/Guard.scala",
            "line_start": 2,
            "line_end": 5,
            "contains": ["currentVersion >= batchId", "return false"],
        }],
        source="src", source_root=str(tmp_path))

    assert len(witnesses) == 1
    witness = witnesses[0]
    assert witness["id"] == "already-committed-guard"
    assert witness["proof_errors"] == []
    assert witness["materialization"]["gaps"] == []
    excerpt = witness["materialization"]["excerpts"][0]
    assert (excerpt["line_start"], excerpt["line_end"], excerpt["kind"]) == (
        2, 5, "claim_witness")
    assert len(excerpt["sha256"]) == 64
def test_bidirectional_path_rejects_shared_callee_shortcut(builder, conn):
    _symbol(conn, "left", "pkg.Left.route", "src/Left.scala")
    _symbol(conn, "shared", "pkg.Shared.schema", "src/Shared.scala")
    _symbol(conn, "right", "pkg.Right.route", "src/Right.scala")
    _edge(conn, "left", "shared", line=10, file="src/Left.scala")
    _edge(conn, "right", "shared", line=20, file="src/Right.scala")

    path = builder.bidirectional_candidate_path(
        conn, "src",
        [{"canonical_id": "left", "qualified_name": "pkg.Left.route", "rank": 0}],
        [{"canonical_id": "right", "qualified_name": "pkg.Right.route", "rank": 0}],
        max_depth=3, max_frontier=20, search_node_limit=20)

    assert path is None


def test_companion_object_bridges_construction_to_type_consumption(builder, conn):
    term = "semanticdb maven . . pkg/Plan."
    plan_type = "semanticdb maven . . pkg/Plan#"
    _symbol(conn, "analysis", "pkg.Analysis.apply", "src/Analysis.scala")
    _symbol(conn, term, "pkg.Plan", "src/Plan.scala", kind="Object")
    _symbol(conn, plan_type, "pkg.Plan", "src/Plan.scala", kind="Class")
    _symbol(conn, "consumer", "pkg.Consumer.apply", "src/Consumer.scala")
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("analysis", term, "type_ref", "src/Analysis.scala", 10, "exact"))
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("consumer", plan_type, "type_ref", "src/Consumer.scala", 20, "exact"))

    path = builder.bidirectional_candidate_path(
        conn, "src",
        [{"canonical_id": "analysis", "qualified_name": "pkg.Analysis.apply", "rank": 0}],
        [{"canonical_id": "consumer", "qualified_name": "pkg.Consumer.apply", "rank": 0}],
        max_depth=4, max_frontier=20, search_node_limit=30)

    assert path is not None
    assert [edge["edge_type"] for edge in path["edges"]] == [
        "type_ref", "companion", "type_ref"]
    assert [node["canonical_id"] for node in path["nodes"]] == [
        "analysis", term, plan_type, "consumer"]
    assert all(edge["compiler_verified"] for edge in path["edges"])
def test_merge_update_oracles_distinguish_scala_term_and_type_symbols():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    claims = {
        (question_id, claim["id"]): claim
        for question_id in (10, 12)
        for claim in requirements[question_id]["claims"]}

    spark_api = {
        anchor["anchor"]: anchor
        for anchor in claims[(10, "spark-api")]["anchors"]}
    assert spark_api["spark-merge-plan"]["kind"] == "Object"
    assert spark_api["spark-merge-plan"]["line"] == 808

    delta_update = {
        anchor["anchor"]: anchor
        for anchor in claims[(12, "delta-update")]["anchors"]}
    assert delta_update["delta-update-constructor"]["kind"] == "Object"
    assert delta_update["delta-update-constructor"]["line"] == 47
    assert delta_update["delta-update-plan"]["kind"] == "Class"
    assert delta_update["delta-update-plan"]["line"] == 32
def test_main_builds_a_reviewable_single_claim_report(builder, tmp_path, monkeypatch):
    database = tmp_path / "scip.db"
    connection = sqlite3.connect(database)
    init_scip_schema(connection)
    _symbol(
        connection, "entry", "pkg.Entry.run", "src/Entry.py",
        source="src")
    connection.commit()
    connection.close()

    source_root = tmp_path / "source"
    source_file = source_root / "src" / "Entry.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("def run():\n    return True\n")
    requirements = tmp_path / "requirements.json"
    requirements.write_text(json.dumps([{
        "id": 1,
        "claims": [{
            "id": "entry-claim",
            "assertion": "Entry.run returns successfully.",
            "anchors": [{
                "anchor": "entry",
                "symbol": "pkg.Entry.run",
                "file": "src/Entry.py"}],
            "witnesses": [{
                "id": "entry-source",
                "file": "src/Entry.py",
                "line_start": 1,
                "line_end": 2,
                "contains": ["def run", "return True"]}],
            "repos": ["pkg"]}]}]))
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([{"id": 1, "after": "Does entry run?"}]))
    output = tmp_path / "candidate.json"
    review = tmp_path / "review.md"
    monkeypatch.setattr("sys.argv", [
        "build_gold_chains.py",
        "--db", str(database),
        "--requirements", str(requirements),
        "--questions", str(questions),
        "--source", "src",
        "--source-root", str(source_root),
        "--candidate-limit", "1",
        "--review-out", str(review),
        "--out", str(output)])

    assert builder.main() == 0

    report = json.loads(output.read_text())
    claim = report["questions"][0]["claims"][0]
    assert claim["unresolved_anchors"] == []
    assert claim["witnesses"][0]["proof_errors"] == []
    assert review.read_text().startswith("# Gold-chain candidate review")
def test_coherent_path_sets_honor_required_claim_transitions(builder):
    paths = [
        _review_path("a-c", "a", "c", "a1", "c1"),
        _review_path("c-b", "c", "b", "c1", "b1"),
        _review_path("a-b", "a", "b", "a1", "b1", rank=2),
        _review_path("b-c", "b", "c", "b1", "c1", rank=2)]

    found = builder.coherent_path_sets(
        ["a", "b", "c"], paths, limit=3,
        required_transitions=[["a", "b"], ["b", "c"]])

    assert len(found) == 1
    assert found[0]["path_ids"] == ["a-b", "b-c"]
    assert found[0]["required_transitions_complete"]
    assert found[0]["required_transitions"] == [["a", "b"], ["b", "c"]]


def test_review_validation_enforces_required_claim_transitions(builder, tmp_path):
    paths = [
        _review_path("a-c", "a", "c", "a1", "c1"),
        _review_path("c-b", "c", "b", "c1", "b1")]
    for path in paths:
        path["materialization"] = {"gaps": [], "excerpts": [
            {"kind": "definition", "file": node["file"],
             "line_start": node["line_start"], "line_end": node["line_end"],
             "content": "proof", "sha256": "a" * 64}
            for node in path["nodes"]]}
    report = {"questions": [{
        "id": 1,
        "review": {"status": "accepted", "answer": "reviewed"},
        "claims": [{
            "id": "claim",
            "transitions": [["a", "b"], ["b", "c"]],
            "anchors": [
                {"anchor": "a", "candidates": [{"canonical_id": "a1"}]},
                {"anchor": "b", "candidates": [{"canonical_id": "b1"}]},
                {"anchor": "c", "candidates": [{"canonical_id": "c1"}]}],
            "candidate_paths": paths,
            "review": {
                "status": "accepted",
                "selected_candidate_by_anchor": {
                    "a": "a1", "b": "b1", "c": "c1"},
                "selected_path_ids": ["a-c", "c-b"],
                "claim_correct": True,
                "complete": True}}]}]}
    review_file = tmp_path / "review.json"
    review_file.write_text(json.dumps(report))

    errors = builder.validate_review(review_file)

    assert any("required transition proof" in error for error in errors)
def test_delta_update_oracle_declares_causal_topology_and_command_term():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    claim = next(
        item for item in requirements[12]["claims"]
        if item["id"] == "delta-update")
    anchors = {item["anchor"]: item for item in claim["anchors"]}

    assert anchors["delta-update-command"]["kind"] == "Object"
    assert anchors["delta-update-command"]["line"] == 501
    assert claim["transitions"] == [
        ["delta-analysis", "delta-update-constructor"],
        ["delta-update-constructor", "delta-update-plan"],
        ["delta-update-plan", "delta-update-preprocessor"],
        ["delta-update-preprocessor", "delta-update-expression-generator"],
        ["delta-update-preprocessor", "delta-update-command"]]
def test_coherent_path_sets_deduplicate_branch_traversal_order(builder):
    paths = [
        _review_path("a-b", "a", "b", "a1", "b1"),
        _review_path("b-c", "b", "c", "b1", "c1"),
        _review_path("b-d", "b", "d", "b1", "d1")]

    found = builder.coherent_path_sets(
        ["a", "b", "c", "d"], paths, limit=5,
        required_transitions=[
            ["a", "b"], ["b", "c"], ["b", "d"]])

    assert len(found) == 1
    assert found[0]["path_ids"] == ["a-b", "b-c", "b-d"]
def test_first_gold_batch_declares_complete_causal_trees():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    expected = {
        (4, "spark-join-classify"): [
            ["execute-joined-child", "classify-and-emit"]],
        (4, "delta-join-classify"): [
            ["delta-join-and-write", "delta-row-classification"]],
        (6, "spark-rewrite"): [
            ["spark-merge-plan", "spark-row-level-rewrite"]],
        (6, "delta-diversion"): [
            ["delta-analysis", "delta-merge-constructor"],
            ["delta-merge-constructor", "delta-merge-plan"],
            ["delta-merge-plan", "delta-merge-preprocessor"]],
        (8, "delta-values"): [
            ["resolve-delta-implicit-columns", "generated-column-expression"],
            ["resolve-delta-implicit-columns", "identity-column-generator"]],
        (8, "spark-values"): [
            ["spark-row-level-assignment-rule", "spark-align-insert-assignments"]],
        (10, "delta-api-convergence"): [["delta-table-merge-api", "delta-builder-constructor"], ["delta-builder-constructor", "delta-builder-type"], ["delta-builder-type", "build-delta-merge-plan"], ["execute-delta-builder", "build-delta-merge-plan"], ["build-delta-merge-plan", "delta-merge-constructor"], ["delta-merge-constructor", "delta-merge-plan"], ["delta-merge-plan", "delta-merge-preprocessor"]],
        (10, "spark-api"): [
            ["spark-merge-writer", "spark-merge-plan"]],
        (12, "delta-update"): [
            ["delta-analysis", "delta-update-constructor"],
            ["delta-update-constructor", "delta-update-plan"],
            ["delta-update-plan", "delta-update-preprocessor"],
            ["delta-update-preprocessor", "delta-update-expression-generator"],
            ["delta-update-preprocessor", "delta-update-command"]],
        (12, "spark-update"): [
            ["spark-row-level-assignment-rule", "spark-align-update-assignments"],
            ["spark-row-level-assignment-rule", "spark-update-plan"],
            ["spark-update-plan", "spark-update-rewrite"],
            ["spark-update-rewrite", "spark-replace-data-plan"],
            ["spark-update-rewrite", "spark-write-delta-plan"]],
        (13, "spark-pruning"): [
            ["spark-runtime-group-filter-rule", "spark-runtime-filter-capability"]],
        (13, "delta-pruning"): [
            ["delta-find-touched-files", "delta-transaction-file-filter"]]}

    found = {}
    for question_id in (4, 6, 8, 10, 12, 13):
        for claim in requirements[question_id]["claims"]:
            labels = [anchor["anchor"] for anchor in claim["anchors"]]
            transitions = claim.get("transitions", [])
            assert len(transitions) == len(labels) - 1
            assert {value for pair in transitions for value in pair} == set(labels)
            found[(question_id, claim["id"])] = transitions

    assert found == expected
    for question_id, claim_id in ((6, "delta-diversion"), (10, "delta-api-convergence")):
        claim = next(
            item for item in requirements[question_id]["claims"]
            if item["id"] == claim_id)
        anchors = {item["anchor"]: item for item in claim["anchors"]}
        assert anchors["delta-merge-constructor"]["kind"] == "Object"
        assert anchors["delta-merge-constructor"]["line"] == 336
        assert anchors["delta-merge-plan"]["kind"] == "Class"
        assert anchors["delta-merge-plan"]["line"] == 315
def test_bidirectional_search_prefers_edges_local_to_selected_endpoint_files(builder, conn):
    _symbol(conn, "left", "pkg.Left.run", "zzz-core/Left.scala")
    _symbol(conn, 'aaa_wrong', "pkg.ConnectBridge.run", "aaa-connect/Bridge.scala")
    _symbol(conn, 'zzz_correct', "pkg.CoreBridge.run", "zzz-core/Bridge.scala")
    _symbol(conn, "right", "pkg.Right.run", "zzz-core/Right.scala")
    _edge(conn, "left", 'aaa_wrong', line=1, file="aaa-connect/Left.scala")
    _edge(conn, 'aaa_wrong', "right", line=2, file="aaa-connect/Bridge.scala")
    _edge(conn, "left", 'zzz_correct', line=3, file="zzz-core/Left.scala")
    _edge(conn, 'zzz_correct', "right", line=4, file="zzz-core/Bridge.scala")

    path = builder.bidirectional_candidate_path(
        conn, "src",
        [{"canonical_id": "left", "qualified_name": "pkg.Left.run",
          "file": "zzz-core/Left.scala", "rank": 0}],
        [{"canonical_id": "right", "qualified_name": "pkg.Right.run",
          "file": "zzz-core/Right.scala", "rank": 0}],
        max_depth=3, max_frontier=20, search_node_limit=30)

    assert [node["canonical_id"] for node in path["nodes"]] == [
        "left", 'zzz_correct', "right"]
    assert {edge["file"] for edge in path["edges"]} == {
        "zzz-core/Left.scala", "zzz-core/Bridge.scala"}
def test_companion_edge_uses_selected_duplicate_definition_site(builder, conn):
    term = "semanticdb maven . . pkg/Plan."
    plan_type = "semanticdb maven . . pkg/Plan#"
    _symbol(conn, term, "pkg.Plan", "connect/Plan.scala", kind="Object")
    _symbol(conn, plan_type, "pkg.Plan", "connect/Plan.scala", kind="Class")

    path = builder.bidirectional_candidate_path(
        conn, "src",
        [{"canonical_id": term, "qualified_name": "pkg.Plan",
          "file": "core/Plan.scala", "line_start": 20,
          "line_end": 30, "rank": 0}],
        [{"canonical_id": plan_type, "qualified_name": "pkg.Plan",
          "file": "core/Plan.scala", "line_start": 5,
          "line_end": 19, "rank": 0}],
        max_depth=2, max_frontier=20, search_node_limit=20)

    assert path["edges"][0]["edge_type"] == "companion"
    assert path["edges"][0]["file"] == "core/Plan.scala"
    assert path["edges"][0]["line"] == 20
    assert {node["file"] for node in path["nodes"]} == {"core/Plan.scala"}
def test_delta_merge_api_oracle_tracks_builder_construction_identity():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    claim = next(
        item for item in requirements[10]["claims"]
        if item["id"] == "delta-api-convergence")
    anchors = {item["anchor"]: item for item in claim["anchors"]}

    assert anchors["delta-builder-constructor"] == {
        "anchor": "delta-builder-constructor",
        "symbol": "io.delta.tables.DeltaMergeBuilder",
        "file": "spark/src/main/scala/io/delta/tables/DeltaMergeBuilder.scala",
        "line": 386,
        "kind": "Object"}
    assert anchors["delta-builder-type"] == {
        "anchor": "delta-builder-type",
        "symbol": "io.delta.tables.DeltaMergeBuilder",
        "file": "spark/src/main/scala/io/delta/tables/DeltaMergeBuilder.scala",
        "line": 153,
        "kind": "Class"}
    assert claim["transitions"] == [
        ["delta-table-merge-api", "delta-builder-constructor"],
        ["delta-builder-constructor", "delta-builder-type"],
        ["delta-builder-type", "build-delta-merge-plan"],
        ["execute-delta-builder", "build-delta-merge-plan"],
        ["build-delta-merge-plan", "delta-merge-constructor"],
        ["delta-merge-constructor", "delta-merge-plan"],
        ["delta-merge-plan", "delta-merge-preprocessor"]]
def test_second_gold_batch_declares_source_proven_causal_trees():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    expected = {
        (15, "writer-fork"): [
            ["transactional-write-files", "delta-file-writer"]],
        (15, "partition-capability"): [
            ["transactional-write-files", "delta-file-writer"],
            ["delta-file-writer", "write-partition-columns-option"]],
        (16, "delta-commit"): [
            ["delta-write-and-commit", "delta-commit-job"],
            ["delta-commit-job", "delta-added-statuses"],
            ["delta-commit-job", "delta-add-file"]],
        (16, "spark-commit"): [
            ["spark-commit-job", "spark-staging-directory"]],
        (17, "callback-wiring"): [
        ["delta-file-writer", "delta-execute-write"],
        ["delta-execute-write", "delta-write-and-commit"],
        ["delta-write-and-commit", "file-commit-job"],
        ["delta-execute-write", "delta-execute-task"],
        ["delta-execute-task", "file-setup-task"]],
        (17, "temp-repurpose"): [
            ["delta-new-task-temp-file", "delta-added-files"],
            ["delta-commit-task", "delta-added-files"],
            ["delta-commit-task", "delta-build-file-action"],
            ["delta-build-file-action", "delta-add-file"]],
        (23, "nullable-output"): [
            ["transactional-write-files", "normalize-data"],
            ["normalize-data", "normalize-schema"],
            ["normalize-schema", "make-output-nullable"],
            ["transactional-write-files", "delta-file-writer"]],
        (23, "constraint-enforcement"): [
            ["transactional-write-files", "constraints-get-all"],
            ["constraints-get-all", "invariants-from-schema"],
            ["invariants-from-schema", "not-null-constraint"],
            ["transactional-write-files", "invariant-checker"],
            ["invariant-checker", "build-invariant-checks"],
            ["build-invariant-checks", "check-delta-invariant"],
            ["check-delta-invariant", "assert-invariant-rule"]],
        (27, "delegation"): [
            ["delta-create-table", "create-catalog-table"],
            ["create-catalog-table", "delegated-create-table"],
            ["delegated-create-table", "delegate-as-table-catalog"],
            ["delegate-as-table-catalog", "delegate-field"]],
        (27, "delta-intercept"): [
            ["delta-create-table", "delta-provider-check"],
            ["delta-create-table", "create-delta-table"],
            ["create-delta-table", "delta-load-table"],
            ["delta-load-table", "delta-table-constructor"],
            ["delta-table-constructor", "delta-table-type"]],
        (31, "catalog-chain"): [
            ["delta-load-table", "delegated-load-table"],
            ["delegated-load-table", "delegate-as-table-catalog"],
            ["delegate-as-table-catalog", "delegate-field"]],
        (31, "enrichment"): [
            ["delta-load-table", "delta-table-constructor"],
            ["delta-table-constructor", "delta-table-type"],
            ["delta-table-type", "initial-snapshot"],
            ["initial-snapshot", "time-travel-spec"]]}

    found = {}
    for question_id in (15, 16, 17, 23, 27, 31):
        for claim in requirements[question_id]["claims"]:
            anchors = claim["anchors"]
            assert anchors
            assert all(isinstance(anchor, dict) for anchor in anchors)
            assert all(
                {"anchor", "symbol", "file"} <= set(anchor)
                for anchor in anchors)
            labels = [anchor["anchor"] for anchor in anchors]
            assert len(labels) == len(set(labels))
            transitions = claim["transitions"]
            assert len(transitions) == len(labels) - 1
            assert {value for pair in transitions for value in pair} == set(labels)
            assert claim["witnesses"]
            assert all(witness["contains"] for witness in claim["witnesses"])
            found[(question_id, claim["id"])] = transitions

    assert found == expected

    critical = {
        (15, "partition-capability", "write-partition-columns-option"):
            ("org.apache.spark.sql.delta.DeltaOptions.WRITE_PARTITION_COLUMNS", 305),
        (16, "delta-commit", "delta-commit-job"):
            ("org.apache.spark.sql.delta.files.DelayedCommitProtocol.commitJob", 89),
        (23, "constraint-enforcement", "not-null-constraint"):
            ("org.apache.spark.sql.delta.constraints.Constraints.NotNull", 45),
        (27, "delegation", "delegate-field"):
            ("org.apache.spark.sql.connector.catalog.DelegatingCatalogExtension.delegate", 41),
        (31, "enrichment", "delta-table-constructor"):
            ("org.apache.spark.sql.delta.catalog.DeltaTableV2", 408),
        (31, "enrichment", "delta-table-type"):
            ("org.apache.spark.sql.delta.catalog.DeltaTableV2", 64)}
    for (question_id, claim_id, anchor_id), (symbol, line) in critical.items():
        claim = next(
            item for item in requirements[question_id]["claims"]
            if item["id"] == claim_id)
        anchor = next(
            item for item in claim["anchors"]
            if item["anchor"] == anchor_id)
        assert (anchor["symbol"], anchor["line"]) == (symbol, line)
def test_required_transition_matching_ignores_anchor_declaration_order(builder):
    reverse_declared_path = _review_path(
        "later-earlier", "later", "earlier", "later-id", "earlier-id")
    reverse_declared_path["orientation"] = "right-to-left"
    wrong_direction = _review_path(
        "wrong-direction", "later", "earlier", "later-id", "earlier-id")

    found = builder.coherent_path_sets(
        ["later", "earlier"], [reverse_declared_path], limit=1,
        required_transitions=[["earlier", "later"]])
    rejected = builder.coherent_path_sets(
        ["later", "earlier"], [wrong_direction], limit=1,
        required_transitions=[["earlier", "later"]])

    assert len(found) == 1
    assert found[0]["path_ids"] == ["later-earlier"]
    assert found[0]["required_transitions"] == [["earlier", "later"]]
    assert rejected == []


def test_q17_callback_oracle_names_each_real_writer_transition():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    callback = next(
        claim for claim in requirements[17]["claims"]
        if claim["id"] == "callback-wiring")
    anchors = {item["anchor"]: item for item in callback["anchors"]}

    assert callback["transitions"] == [
        ["delta-file-writer", "delta-execute-write"],
        ["delta-execute-write", "delta-write-and-commit"],
        ["delta-write-and-commit", "file-commit-job"],
        ["delta-execute-write", "delta-execute-task"],
        ["delta-execute-task", "file-setup-task"]]
    assert (anchors["delta-execute-write"]["symbol"],
            anchors["delta-execute-write"]["line"]) == (
        "org.apache.spark.sql.delta.files.DeltaFileFormatWriter.executeWrite", 224)
    assert (anchors["delta-write-and-commit"]["symbol"],
            anchors["delta-write-and-commit"]["line"]) == (
        "org.apache.spark.sql.delta.files.DeltaFileFormatWriter.writeAndCommit", 294)
    assert (anchors["delta-execute-task"]["symbol"],
            anchors["delta-execute-task"]["line"]) == (
        "org.apache.spark.sql.delta.files.DeltaFileFormatWriter.executeTask", 395)
def test_third_gold_batch_declares_source_proven_causal_trees():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    expected = {
        (62, "spark-request"): [
            ["microbatch-next", "microbatch-start-offset"],
            ["microbatch-next", "admission-latest-offset"],
            ["admission-latest-offset", "read-limit"]],
        (62, "delta-bounded-offset"): [
            ["delta-latest-offset", "delta-latest-offset-internal"],
            ["delta-latest-offset-internal", "next-offset-from-previous"],
            ["next-offset-from-previous", "rate-limited-file-changes"],
            ["rate-limited-file-changes", "delta-file-changes"],
            ["delta-file-changes", "index-files"],
            ["index-files", "indexed-file"],
            ["rate-limited-file-changes", "admission-control"],
            ["next-offset-from-previous", "last-indexed-file"],
            ["next-offset-from-previous", "build-source-offset"],
            ["build-source-offset", "delta-source-offset"]],
        (66, "indexed-position"): [
            ["index-files", "indexed-file"],
            ["build-source-offset", "indexed-file"],
            ["build-source-offset", "delta-source-offset"]],
        (66, "monotonic-batches"): [
            ["delta-latest-offset-internal", "next-offset-from-previous"],
            ["next-offset-from-previous", "rate-limited-file-changes"],
            ["rate-limited-file-changes", "delta-file-changes"],
            ["delta-file-changes", "index-files"],
            ["next-offset-from-previous", "last-indexed-file"],
            ["next-offset-from-previous", "build-source-offset"],
            ["delta-latest-offset-internal", "validate-offsets"],
            ["validate-offsets", "delta-source-offset"],
            ["delta-source-offset", "compare-offsets"]],
        (67, "stable-query-id"): [
            ["stream-id", "stream-metadata"],
            ["publish-query-id", "stream-id"],
            ["publish-query-id", "query-id-key"],
            ["sink-query-id", "query-id-key"]],
        (67, "replay-check"): [
            ["sink-add-batch", "sink-add-batch-with-status"],
            ["sink-add-batch-with-status", "transaction-version"],
            ["sink-add-batch-with-status", "sink-query-id"]],
        (67, "transaction-record"): [
            ["sink-add-batch-with-status", "pending-commit"],
            ["pending-commit", "set-transaction"],
            ["pending-commit", "sink-query-id"],
            ["pending-commit", "pending-batch-id"],
            ["pending-commit", "transaction-commit"]],
        (84, "metadata-validation"): [
            ["metadata-update", "metadata-assertion"],
            ["metadata-assertion", "has-generated-columns"],
            ["metadata-assertion", "validate-generated-columns"],
            ["validate-generated-columns", "generation-expression-field"],
            ["generation-expression-field", "generation-expression-metadata"],
            ["generation-expression-metadata", "generation-expression-key"]],
        (84, "protocol-gate"): [
            ["metadata-update", "protocol-for-new-table"],
            ["protocol-for-new-table", "protocol-components-from-metadata"],
            ["protocol-components-from-metadata", "extract-auto-features"],
            ["extract-auto-features", "feature-metadata-requirement"],
            ["feature-metadata-requirement", "generated-feature-requirement"],
            ["generated-columns-feature", "generated-feature-requirement"],
            ["generated-feature-requirement", "has-generated-columns"]],
        (87, "spark-capability"): [
            ["create-table-strategy", "validate-identity-column"],
            ["validate-identity-column", "catalog-capabilities"],
            ["validate-identity-column", "identity-create-capability"],
            ["catalog-capabilities", "delegating-capabilities"],
            ["delegating-catalog", "delegating-capabilities"],
            ["delta-catalog", "delegating-catalog"]],
        (87, "delta-protocol"): [
            ["metadata-update", "protocol-for-new-table"],
            ["metadata-update", "satisfies-identity-protocol"],
            ["satisfies-identity-protocol", "identity-columns-feature"],
            ["identity-columns-feature", "identity-feature-requirement"],
            ["identity-feature-requirement", "has-identity-column"]],
        (88, "parser-metadata"): [
            ["spark-create-schema", "spark-column-definition-list"],
            ["spark-column-definition-list", "spark-column-definition"],
            ["spark-column-definition", "spark-identity-parser"],
            ["spark-identity-parser", "spark-identity-spec"],
            ["spark-create-schema", "spark-v1-column"],
            ["spark-v1-column", "spark-encode-identity"],
            ["spark-encode-identity", "spark-allow-explicit-insert"]],
        (88, "delta-enforcement"): [
            ["delta-write", "delta-block-explicit-insert"],
            ["delta-block-explicit-insert", "delta-is-identity-column"],
            ["delta-block-explicit-insert", "delta-allow-explicit-insert"],
            ["delta-block-explicit-insert", "delta-block-identity-column"],
            ["delta-block-identity-column", "delta-explicit-insert-error"]]}

    found = {}
    for question_id in (62, 66, 67, 84, 87, 88):
        for claim in requirements[question_id]["claims"]:
            anchors = claim["anchors"]
            assert anchors
            assert all(isinstance(anchor, dict) for anchor in anchors)
            assert all(
                {"anchor", "symbol", "file", "line"} <= set(anchor)
                for anchor in anchors)
            labels = [anchor["anchor"] for anchor in anchors]
            assert len(labels) == len(set(labels))
            transitions = claim["transitions"]
            assert len(transitions) == len(labels) - 1
            assert {value for pair in transitions for value in pair} == set(labels)
            assert claim["witnesses"]
            assert all(witness["contains"] for witness in claim["witnesses"])
            found[(question_id, claim["id"])] = transitions

    assert found == expected

    spark_identity = next(
        claim for claim in requirements[87]["claims"]
        if claim["id"] == "spark-capability")
    assert "forwards" in spark_identity["assertion"]
    assert "advertises" not in spark_identity["assertion"]
    spark_identity_anchors = {
        anchor["anchor"]: anchor for anchor in spark_identity["anchors"]}
    assert spark_identity_anchors["delta-catalog"]["kind"] == "Class"
    assert spark_identity_anchors["delegating-catalog"]["kind"] == "Class"

    parser_claim = next(
        claim for claim in requirements[88]["claims"]
        if claim["id"] == "parser-metadata")
    enforcement_claim = next(
        claim for claim in requirements[88]["claims"]
        if claim["id"] == "delta-enforcement")
    assert parser_claim["repos"] == ["spark"]
    assert enforcement_claim["repos"] == ["delta"]

def test_final_gold_batch_declares_source_proven_causal_trees():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    expected = {
        (107, "delta-casts-first"): [
            ["delta-analysis", "delta-analysis-apply"],
            ["delta-analysis-apply", "resolve-insert-by-ordinal"],
            ["resolve-insert-by-ordinal", "add-cast-to-column"],
            ["add-cast-to-column", "get-cast-function"]],
        (107, "spark-preparation"): [
            ["v2-writes", "v2-writes-apply"],
            ["v2-writes-apply", "prepare-query"],
            ["prepare-query", "requires-distribution-and-ordering"],
            ["prepare-query", "type-coercion-executor"]],
        (147, "plan-filter"): [
            ["scan-with-deletion-vectors", "dv-enabled-scan"],
            ["dv-enabled-scan", "add-skip-row-scan"],
            ["add-skip-row-scan", "deleted-row-field"],
            ["dv-enabled-scan", "create-row-filter"],
            ["create-row-filter", "deleted-row-column-name"],
            ["create-row-filter", "keep-row-value"],
            ["create-row-filter", "logical-filter"]],
        (147, "reader-population"): [
            ["delta-reader-build", "deleted-row-column-name"],
            ["delta-reader-build", "metadata-iterator"],
            ["metadata-iterator", "drop-marked-builder"],
            ["metadata-iterator", "load-row-filter"],
            ["metadata-iterator", "row-filter-interface"],
            ['row-filter-interface', 'row-filter-implementation'],
            ["row-filter-implementation", "bitmap-membership"],
            ["bitmap-membership", "bitmap-contains"]],
        (156, "dv-production"): [
            ["find-touched-files", "build-row-index-sets"],
            ["build-row-index-sets", "parquet-row-index"],
            ["build-row-index-sets", "bitmap-row-index-column"],
            ["build-row-index-sets", "build-deletion-vectors"],
            ["build-deletion-vectors", "compute-bitmap-result"],
            ["compute-bitmap-result", "bitmap-aggregate-columns"],
            ["bitmap-aggregate-columns", "create-bitmap-aggregator"],
            ["create-bitmap-aggregator", "bitmap-aggregator"],
            ["bitmap-aggregator", "bitmap-aggregator-update"],
            ["bitmap-aggregator-update", "bitmap-add"]],
        (156, "row-index-alignment"): [
            ["delta-reader-build", "spark-parquet-reader-build"],
            ["delta-reader-build", "metadata-iterator"],
            ["spark-parquet-reader-build", "vectorized-parquet-reader"],
            ["vectorized-parquet-reader", "row-index-generator-field"],
            ["row-index-generator-field", "row-index-generator"],
            ["row-index-generator", "init-row-index-from-pages"],
            ["row-index-generator", "populate-row-index"],
            ["metadata-iterator", "row-filter-interface"],
            ['row-filter-interface', 'row-filter-implementation'],
            ["row-filter-implementation", "bitmap-membership"],
            ["bitmap-membership", "bitmap-contains"]],
        (187, "fork-capability"): [
            ["transactional-write-files", "delta-file-writer"],
            ["transactional-write-files", "write-partition-columns-option"],
            ["delta-writer-owner", "delta-file-writer"]],
        (187, "writefiles-path"): [
            ["delta-file-writer", "get-write-files"],
            ["delta-file-writer", "delta-planned-execute"],
            ["delta-planned-execute", "spark-execute-write"],
            ["spark-execute-write", "spark-do-execute-write"],
            ['spark-do-execute-write', 'write-files-do-execute'],
            ["write-files-exec", "write-files-do-execute"]]}

    found = {}
    for question_id in (107, 147, 156, 187):
        for claim in requirements[question_id]["claims"]:
            anchors = claim["anchors"]
            assert anchors
            assert all(isinstance(anchor, dict) for anchor in anchors)
            assert all(
                {"anchor", "symbol", "file", "line", "kind"} <= set(anchor)
                for anchor in anchors)
            labels = [anchor["anchor"] for anchor in anchors]
            assert len(labels) == len(set(labels))
            transitions = claim["transitions"]
            assert len(transitions) == len(labels) - 1
            assert {value for pair in transitions for value in pair} == set(labels)
            assert claim["witnesses"]
            assert all(witness["contains"] for witness in claim["witnesses"])
            found[(question_id, claim["id"])] = transitions

    assert found == expected

    claims = {
        (question_id, claim["id"]): claim
        for question_id in (107, 147, 156, 187)
        for claim in requirements[question_id]["claims"]}
    assert "target type" in claims[(107, "delta-casts-first")]["assertion"]
    assert "type widening" in claims[(107, "delta-casts-first")]["assertion"]
    assert "does not replace" in claims[(107, "spark-preparation")]["assertion"]
    assert "actually evaluates" in claims[(147, "plan-filter")]["assertion"]
    assert "file-relative" in claims[(156, "row-index-alignment")]["assertion"]
    assert "stock writer" in claims[(187, "fork-capability")]["assertion"]

    critical = {
        (107, "delta-casts-first", "get-cast-function"):
            ("org.apache.spark.sql.delta.DeltaAnalysis.getCastFunction", 1102),
        (147, "plan-filter", "logical-filter"):
            ("org.apache.spark.sql.catalyst.plans.logical.Filter", 333),
        (156, "row-index-alignment", "row-index-generator-field"):
            ("org.apache.spark.sql.execution.datasources.parquet."
             "VectorizedParquetRecordReader.rowIndexGenerator", 143),
        (187, "writefiles-path", "delta-planned-execute"):
            ("org.apache.spark.sql.delta.files."
             "DeltaFileFormatWriter.executeWrite", 327)}
    for (question_id, claim_id, anchor_id), (symbol, line) in critical.items():
        anchor = next(
            item for item in claims[(question_id, claim_id)]["anchors"]
            if item["anchor"] == anchor_id)
        assert (anchor["symbol"], anchor["line"]) == (symbol, line)
def test_coherent_path_sets_accept_explicit_reverse_transition_proof(builder):
    path = _review_path(
        "plan-rule", "plan", "rule", "plan-id", "rule-id")
    path["orientation"] = "right-to-left"

    ordinary = builder.coherent_path_sets(
        ["plan", "rule"], [path], limit=1,
        required_transitions=[["plan", "rule"]])
    reverse_proven = builder.coherent_path_sets(
        ["plan", "rule"], [path], limit=1,
        required_transitions=[["plan", "rule"]],
        reverse_transitions=[["plan", "rule"]])

    assert ordinary == []
    assert len(reverse_proven) == 1
    assert reverse_proven[0]["path_ids"] == ["plan-rule"]
    assert reverse_proven[0]["reverse_transitions"] == [["plan", "rule"]]


def test_review_validation_accepts_declared_reverse_transition_proof(
        builder, tmp_path):
    path = _review_path(
        "plan-rule", "plan", "rule", "plan-id", "rule-id")
    path["orientation"] = "right-to-left"
    report = {"questions": [{
        "id": 1,
        "review": {"status": "accepted", "answer": "reviewed"},
        "claims": [{
            "id": "claim",
            "transitions": [["plan", "rule"]],
            "reverse_transitions": [["plan", "rule"]],
            "anchors": [
                {"anchor": "plan", "candidates": [{"canonical_id": "plan-id"}]},
                {"anchor": "rule", "candidates": [{"canonical_id": "rule-id"}]}],
            "candidate_paths": [path],
            "review": {
                "status": "accepted",
                "selected_candidate_by_anchor": {
                    "plan": "plan-id", "rule": "rule-id"},
                "selected_path_ids": ["plan-rule"],
                "claim_correct": True, "complete": True}}]}]}
    review_file = tmp_path / "review.json"
    review_file.write_text(json.dumps(report))

    errors = builder.validate_review(review_file)

    assert not any(
        "required transition proof" in error for error in errors)
def test_analyzer_consumption_transitions_declare_reverse_compiler_proof():
    requirements_path = (
        Path(__file__).parents[1]
        / "evaluation/chain-benchmark/requirements.json")
    requirements = {
        int(item["id"]): item
        for item in json.loads(requirements_path.read_text())}
    expected = {
        (6, "spark-rewrite"): [["spark-merge-plan", "spark-row-level-rewrite"]],
        (6, "delta-diversion"): [["delta-merge-plan", "delta-merge-preprocessor"]],
        (10, "delta-api-convergence"): [["delta-merge-plan", "delta-merge-preprocessor"]],
        (12, "delta-update"): [["delta-update-plan", "delta-update-preprocessor"]],
        (12, "spark-update"): [["spark-update-plan", "spark-update-rewrite"]],
    }

    found = {}
    for question_id, requirement in requirements.items():
        for claim in requirement["claims"]:
            reverse = claim.get("reverse_transitions", [])
            if reverse:
                found[(question_id, claim["id"])] = reverse

    assert found == expected
def test_main_recoheres_existing_paths_without_opening_scip_database(
        builder, tmp_path, monkeypatch):
    route = _review_path(
        "plan-rule", "plan", "rule", "plan-id", "rule-id")
    route["orientation"] = "right-to-left"
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({
        "status": "candidate-oracle; not gold until reviewed",
        "source": "src",
        "parameters": {"coherent_set_limit": 3},
        "questions": [{
            "id": 1, "question": "How?",
            "review": {"status": "pending", "answer": "", "notes": ""},
            "claims": [{
                "id": "claim",
                "anchors": [
                    {"anchor": "plan", "candidates": [{"canonical_id": "plan-id"}]},
                    {"anchor": "rule", "candidates": [{"canonical_id": "rule-id"}]}],
                "candidate_paths": [route],
                "coherent_path_sets": [],
                "review": {"status": "pending"}}]}]}))
    requirements = tmp_path / "requirements.json"
    requirements.write_text(json.dumps([{
        "id": 1, "claims": [{
            "id": "claim",
            "anchors": [{"anchor": "plan"}, {"anchor": "rule"}],
            "transitions": [["plan", "rule"]],
            "reverse_transitions": [["plan", "rule"]]}]}]))
    output = tmp_path / "recohered.json"
    monkeypatch.setattr("sys.argv", [
        "build_gold_chains.py",
        "--requirements", str(requirements),
        "--recohere", str(candidates),
        "--coherent-set-limit", "1",
        "--out", str(output)])

    assert builder.main() == 0

    report = json.loads(output.read_text())
    claim = report["questions"][0]["claims"][0]
    assert claim["reverse_transitions"] == [["plan", "rule"]]
    assert len(claim["coherent_path_sets"]) == 1
    assert claim["coherent_path_sets"][0]["path_ids"] == ["plan-rule"]
    assert report["parameters"]["coherent_set_limit"] == 1

