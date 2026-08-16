from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def reviewer():
    path = Path(__file__).parents[1] / "evaluation/chain-benchmark/review_gold_chains.py"
    spec = importlib.util.spec_from_file_location("review_gold_chains", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder():
    path = Path(__file__).parents[1] / "evaluation/chain-benchmark/build_gold_chains.py"
    spec = importlib.util.spec_from_file_location("build_gold_chains", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path(path_id, left_anchor, right_anchor, left_id, right_id, *,
          left_name=None, right_name=None, exhausted=False):
    left_name = left_name or f"pkg.{left_id}"
    right_name = right_name or f"pkg.{right_id}"
    left_file = f"src/{left_id}.py"
    right_file = f"src/{right_id}.py"
    digest = "a" * 64
    return {
        "id": path_id,
        "connects": [left_anchor, right_anchor],
        "orientation": "left-to-right",
        "hop_count": 1,
        "selection_score": {
            "endpoint_rank_sum": 0,
            "orientation_penalty": 0,
            "hop_count": 1,
        },
        "search": {
            "algorithm": "bidirectional-dijkstra",
            "expanded_nodes": 12,
            "limit": 10,
            "exhausted": exhausted,
            "mode": "directed",
        },
        "endpoint_candidates": {
            "left": {
                "canonical_id": left_id,
                "qualified_name": left_name,
                "rank": 0,
            },
            "right": {
                "canonical_id": right_id,
                "qualified_name": right_name,
                "rank": 0,
            },
        },
        "nodes": [
            {
                "canonical_id": left_id,
                "qualified_name": left_name,
                "file": left_file,
                "line_start": 1,
                "line_end": 2,
            },
            {
                "canonical_id": right_id,
                "qualified_name": right_name,
                "file": right_file,
                "line_start": 1,
                "line_end": 2,
            },
        ],
        "edges": [{
            "caller_canonical_id": left_id,
            "callee_canonical_id": right_id,
            "caller": left_name,
            "callee": right_name,
            "edge_type": "call",
            "file": left_file,
            "line": 2,
            "compiler_verified": True,
            "compiler_source": "scip-edge",
            "traversal": "caller_to_callee",
        }],
        "all_edges_compiler_verified": True,
        "materialization": {
            "gaps": [],
            "excerpts": [
                {
                    "file": left_file,
                    "line_start": 1,
                    "line_end": 1,
                    "kind": "definition",
                    "content": f"def {left_id}():",
                    "sha256": digest,
                },
                {
                    "file": left_file,
                    "line_start": 2,
                    "line_end": 2,
                    "kind": "call_site",
                    "content": f"    return {right_id}()",
                    "sha256": digest,
                },
                {
                    "file": right_file,
                    "line_start": 1,
                    "line_end": 1,
                    "kind": "definition",
                    "content": f"def {right_id}():",
                    "sha256": digest,
                },
            ],
        },
        "proof_errors": [],
    }


def _anchor(name, *candidates):
    return {
        "anchor": name,
        "candidates": [
            {
                "canonical_id": canonical,
                "qualified_name": f"pkg.{canonical}",
                "file": f"src/{canonical}.py",
                "line_start": 1,
                "line_end": 2,
                "match": "exact",
                "rank": index,
            }
            for index, canonical in enumerate(candidates)
        ],
        "resolved": True,
    }


def _sample_report():
    p1 = _path("A->B#1", "A", "B", "a1", "b1", exhausted=True)
    p2 = _path("A->B#2", "A", "B", "a2", "b2")
    p3 = _path("C->D#1", "C", "D", "c1", "d1")
    return {
        "status": "candidate-oracle; not gold until reviewed",
        "parameters": {"coherent_set_limit": 2},
        "questions": [{
            "id": 1,
            "question": "How does the request travel?",
            "claims": [
                {
                    "id": "ambiguous",
                    "assertion": "A calls B.",
                    "anchors": [_anchor("A", "a1", "a2"), _anchor("B", "b1", "b2")],
                    "candidate_paths": [p1, p2],
                    "candidate_coverage": {
                        "complete": True,
                        "missing_anchors": [],
                        "components": [["A", "B"]],
                    },
                    "coherent_path_sets": [
                        {
                            "path_ids": ["A->B#1"],
                            "selected_candidate_by_anchor": {"A": "a1", "B": "b1"},
                            "score": 1,
                            "complete": True,
                        },
                        {
                            "path_ids": ["A->B#2"],
                            "selected_candidate_by_anchor": {"A": "a2", "B": "b2"},
                            "score": 2,
                            "complete": True,
                        },
                    ],
                    "review": {
                        "status": "pending",
                        "selected_candidate_by_anchor": {},
                        "selected_path_ids": [],
                        "claim_correct": None,
                        "complete": None,
                        "notes": "",
                    },
                },
                {
                    "id": "settled",
                    "assertion": "C calls D.",
                    "anchors": [_anchor("C", "c1"), _anchor("D", "d1")],
                    "candidate_paths": [p3],
                    "candidate_coverage": {
                        "complete": True,
                        "missing_anchors": [],
                        "components": [["C", "D"]],
                    },
                    "coherent_path_sets": [
                        {
                            "path_ids": ["C->D#1"],
                            "selected_candidate_by_anchor": {"C": "c1", "D": "d1"},
                            "score": 3,
                            "complete": True,
                        }
                    ],
                    "review": {
                        "status": "pending",
                        "selected_candidate_by_anchor": {},
                        "selected_path_ids": [],
                        "claim_correct": None,
                        "complete": None,
                        "notes": "",
                    },
                },
            ],
            "review": {"status": "pending", "answer": "", "notes": ""},
        }],
    }


def _write_sources(root):
    for name, callee in (("a1", "b1"), ("a2", "b2"), ("c1", "d1"),
                         ("b1", "done"), ("b2", "done"), ("d1", "done")):
        path = root / "src" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def {name}():\n    return {callee}()\n# tail\n")


def test_unique_selections_collapse_duplicate_path_order(reviewer):
    claim = _sample_report()["questions"][0]["claims"][0]
    first = {
        "path_ids": ["b", "a"],
        "selected_candidate_by_anchor": {"B": "b1", "A": "a1"},
        "score": 8,
        "complete": True,
    }
    duplicate = {
        "path_ids": ["a", "b"],
        "selected_candidate_by_anchor": {"A": "a1", "B": "b1"},
        "score": 6,
        "complete": True,
    }
    claim["coherent_path_sets"] = [first, duplicate]

    selections = reviewer.unique_selections(claim)

    assert len(selections) == 1
    assert selections[0]["score"] == 6
    assert selections[0]["path_ids"] == ["a", "b"]
    assert selections[0]["selection_id"] == reviewer.selection_id(first)


def test_review_items_prioritize_semantic_ambiguity_and_expose_limits(reviewer):
    report = _sample_report()

    items = reviewer.review_items(report)

    assert [item["claim"]["id"] for item in items] == ["ambiguous", "settled"]
    diagnostics = items[0]["diagnostics"]
    assert diagnostics == {
        "coherent_sets": 2,
        "unique_selections": 2,
        "endpoint_assignments": 2,
        "exhausted_paths": 1,
        "at_report_cap": True,
    }


def test_render_bundle_is_compact_and_includes_source_context(reviewer, tmp_path):
    report = _sample_report()
    _write_sources(tmp_path)

    rendered = reviewer.render_review_bundle(
        report,
        source_root=tmp_path,
        context_lines=1,
        selection_limit=1,
    )

    first = reviewer.unique_selections(
        report["questions"][0]["claims"][0])[0]
    assert "# Gold-chain adjudication" in rendered
    assert "Q1 / ambiguous" in rendered
    assert "showing 1 of 2 unique selection(s)" in rendered
    assert first["selection_id"] in rendered
    assert "pkg.a1 -> pkg.b1" in rendered
    assert "2 |     return b1()" in rendered
    assert "A->B#2" not in rendered


def test_decision_template_is_stable_and_never_preaccepts(reviewer):
    report = _sample_report()

    template = reviewer.decision_template(report)

    question = template["questions"][0]
    claim = question["claims"][0]
    expected_ids = [
        item["selection_id"]
        for item in reviewer.unique_selections(
            report["questions"][0]["claims"][0])
    ]
    assert question["status"] == "pending"
    assert question["answer"] == ""
    assert claim["status"] == "pending"
    assert claim["selection_id"] == ""
    assert claim["available_selection_ids"] == expected_ids
    assert claim["claim_correct"] is None
    assert claim["complete"] is None


def test_apply_decisions_keeps_question_pending_until_every_claim_is_reviewed(reviewer):
    report = _sample_report()
    selection = reviewer.unique_selections(
        report["questions"][0]["claims"][0])[0]
    decisions = {
        "questions": [{
            "id": 1,
            "status": "pending",
            "answer": "",
            "claims": [{
                "id": "ambiguous",
                "status": "accepted",
                "selection_id": selection["selection_id"],
                "claim_correct": True,
                "complete": True,
                "notes": "Inspected both definitions and the call site.",
            }],
        }],
    }

    reviewed = reviewer.apply_decisions(report, decisions)

    question = reviewed["questions"][0]
    claim = question["claims"][0]
    assert question["review"]["status"] == "pending"
    assert claim["review"] == {
        "status": "accepted",
        "selected_candidate_by_anchor": {"A": "a1", "B": "b1"},
        "selected_path_ids": ["A->B#1"],
        "claim_correct": True,
        "complete": True,
        "notes": "Inspected both definitions and the call site.",
    }
    assert question["claims"][1]["review"]["status"] == "pending"


def test_apply_complete_decisions_produces_existing_validator_clean_report(
        reviewer, builder, tmp_path):
    report = _sample_report()
    ambiguous = reviewer.unique_selections(report["questions"][0]["claims"][0])[0]
    settled = reviewer.unique_selections(report["questions"][0]["claims"][1])[0]
    decisions = {
        "questions": [{
            "id": 1,
            "status": "accepted",
            "answer": "A calls B, and C calls D.",
            "notes": "Reviewed from compiler edges and exact source.",
            "claims": [
                {
                    "id": "ambiguous",
                    "status": "accepted",
                    "selection_id": ambiguous["selection_id"],
                    "claim_correct": True,
                    "complete": True,
                    "notes": "Reviewed.",
                },
                {
                    "id": "settled",
                    "status": "accepted",
                    "selection_id": settled["selection_id"],
                    "claim_correct": True,
                    "complete": True,
                    "notes": "Reviewed.",
                },
            ],
        }],
    }

    reviewed = reviewer.apply_decisions(report, decisions)
    path = tmp_path / "reviewed.json"
    path.write_text(json.dumps(reviewed))

    assert reviewed["questions"][0]["review"] == {
        "status": "accepted",
        "answer": "A calls B, and C calls D.",
        "notes": "Reviewed from compiler edges and exact source.",
    }
    assert builder.validate_review(path) == []


@pytest.mark.parametrize("mutation,match", [
    ({"selection_id": "stale"}, "unknown selection"),
    ({"claim_correct": False}, "claim_correct must be true"),
    ({"complete": False}, "complete must be true"),
])
def test_apply_decisions_rejects_unreviewed_or_stale_choices(
        reviewer, mutation, match):
    report = _sample_report()
    selection = reviewer.unique_selections(
        report["questions"][0]["claims"][0])[0]
    claim_decision = {
        "id": "ambiguous",
        "status": "accepted",
        "selection_id": selection["selection_id"],
        "claim_correct": True,
        "complete": True,
    }
    claim_decision.update(mutation)
    decisions = {
        "questions": [{
            "id": 1,
            "status": "pending",
            "answer": "",
            "claims": [claim_decision],
        }],
    }

    with pytest.raises(ValueError, match=match):
        reviewer.apply_decisions(report, decisions)


def test_context_reader_refuses_paths_outside_source_root(reviewer, tmp_path):
    outside = tmp_path.parent / "outside.py"
    outside.write_text("secret\n")

    context = reviewer.source_context(
        tmp_path, "../outside.py", line=1, context_lines=2, cache={})

    assert context is None
def test_selection_problems_rejects_missing_or_unproven_routes(reviewer):
    report = _sample_report()
    claim = report["questions"][0]["claims"][0]
    selection = reviewer.unique_selections(claim)[0]

    assert reviewer.selection_problems(claim, selection) == []

    claim["candidate_paths"][0]["proof_errors"] = ["missing definition proof"]
    assert reviewer.selection_problems(claim, selection) == [
        "A->B#1: missing definition proof"
    ]

    claim["candidate_paths"] = []
    assert reviewer.selection_problems(claim, selection) == [
        "unknown path: A->B#1"
    ]


def test_apply_decisions_rejects_mechanically_invalid_selection(reviewer):
    report = _sample_report()
    selection = reviewer.unique_selections(
        report["questions"][0]["claims"][0])[0]
    report["questions"][0]["claims"][0]["candidate_paths"][0][
        "proof_errors"] = ["missing call-site proof"]
    decisions = {
        "questions": [{
            "id": 1,
            "status": "pending",
            "answer": "",
            "claims": [{
                "id": "ambiguous",
                "status": "accepted",
                "selection_id": selection["selection_id"],
                "claim_correct": True,
                "complete": True,
            }],
        }],
    }

    with pytest.raises(ValueError, match="selection is not mechanically valid"):
        reviewer.apply_decisions(report, decisions)
def test_cli_render_template_apply_and_status(reviewer, tmp_path, capsys):
    report = _sample_report()
    _write_sources(tmp_path)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps(report))
    bundle = tmp_path / "review.md"
    decisions_path = tmp_path / "decisions.json"
    reviewed_path = tmp_path / "reviewed.json"

    assert reviewer.main([
        "render",
        "--candidates", str(candidates),
        "--out", str(bundle),
        "--source-root", str(tmp_path),
        "--context-lines", "1",
        "--selection-limit", "1",
        "--only-claims", "1:ambiguous",
    ]) == 0
    rendered = bundle.read_text()
    assert "Q1 / ambiguous" in rendered
    assert "Q1 / settled" not in rendered

    assert reviewer.main([
        "template",
        "--candidates", str(candidates),
        "--out", str(decisions_path),
    ]) == 0
    decisions = json.loads(decisions_path.read_text())
    question = decisions["questions"][0]
    question["status"] = "accepted"
    question["answer"] = "A calls B, and C calls D."
    for claim in question["claims"]:
        claim["status"] = "accepted"
        claim["selection_id"] = claim["available_selection_ids"][0]
        claim["claim_correct"] = True
        claim["complete"] = True
    decisions_path.write_text(json.dumps(decisions))

    assert reviewer.main([
        "apply",
        "--candidates", str(candidates),
        "--decisions", str(decisions_path),
        "--out", str(reviewed_path),
    ]) == 0
    assert json.loads(reviewed_path.read_text())["status"] == "reviewed-gold"

    assert reviewer.main([
        "status", "--candidates", str(reviewed_path)
    ]) == 0
    output = capsys.readouterr().out
    assert "review bundle ->" in output
    assert "decision template ->" in output
    assert "2/2 claims accepted" in output
    assert '"accepted_questions": 1' in output


def test_fingerprint_refuses_decisions_for_a_different_candidate_snapshot(reviewer):
    report = _sample_report()
    decisions = reviewer.decision_template(report)
    report["questions"][0]["question"] = "Changed after template generation"

    with pytest.raises(ValueError, match="fingerprint"):
        reviewer.apply_decisions(report, decisions)


@pytest.mark.parametrize("selector", ["claim-only", "x:claim", "1:"])
def test_parse_claim_filter_rejects_malformed_values(reviewer, selector):
    with pytest.raises(ValueError, match="QID:claim-id"):
        reviewer.parse_claim_filter(selector)


def test_source_context_refuses_hash_mismatch(reviewer, tmp_path):
    _write_sources(tmp_path)

    context = reviewer.source_context(
        tmp_path,
        "src/a1.py",
        line=1,
        context_lines=1,
        cache={},
        expected_sha256="0" * 64,
    )

    assert context is None
def test_selection_problems_reject_unproven_claim_witness(reviewer):
    claim = {
        "candidate_paths": [],
        "witnesses": [{
            "id": "guard",
            "materialization": {"gaps": ["source unavailable"], "excerpts": []},
            "proof_errors": ["missing required fragment: return false"],
        }],
    }

    problems = reviewer.selection_problems(
        claim, {"path_ids": [], "selected_candidate_by_anchor": {}})

    assert problems == [
        "witness guard: materialization gap: source unavailable",
        "witness guard: missing required fragment: return false",
    ]
def test_review_bundle_renders_claim_witness_sections(reviewer):
    report = _sample_report()
    claim = report["questions"][0]["claims"][1]
    claim["witnesses"] = [{
        "id": "already-committed-guard",
        "file": "src/Guard.scala",
        "line_start": 8,
        "line_end": 10,
        "contains": ["return false"],
        "materialization": {
            "gaps": [],
            "excerpts": [{
                "file": "src/Guard.scala",
                "line_start": 8,
                "line_end": 10,
                "kind": "claim_witness",
                "content": "if (done) {\n  return false\n}",
                "sha256": "a" * 64,
            }],
        },
        "proof_errors": [],
    }]

    rendered = reviewer.render_review_bundle(
        report, selection_limit=1, only_claims={(1, "settled")})

    assert "Claim witness `already-committed-guard`" in rendered
    assert "src/Guard.scala:8-10" in rendered
    assert "9 |   return false" in rendered
def test_merge_candidate_reports_overlays_claims_without_losing_siblings(reviewer):
    base = {
        "source": "databricks",
        "status": "candidate-oracle; not gold until reviewed",
        "questions": [{
            "id": 1, "question": "How?", "claims": [
                {"id": "first", "marker": "original-first"},
                {"id": "repair", "marker": "original-repair"}]}]}
    overlay = {
        "source": "databricks",
        "questions": [
            {"id": 1, "question": "How?", "claims": [
                {"id": "repair", "marker": "fixed"}]},
            {"id": 2, "question": "Why?", "claims": [
                {"id": "new", "marker": "added"}]}]}

    merged = reviewer.merge_candidate_reports([base, overlay])

    assert [question["id"] for question in merged["questions"]] == [1, 2]
    assert [claim["id"] for claim in merged["questions"][0]["claims"]] == [
        "first", "repair"]
    assert merged["questions"][0]["claims"][1]["marker"] == "fixed"
    assert base["questions"][0]["claims"][1]["marker"] == "original-repair"

    incompatible = {"source": "other", "questions": []}
    with pytest.raises(ValueError, match="source"):
        reviewer.merge_candidate_reports([base, incompatible])
def test_merge_cli_writes_a_single_overlayed_candidate_report(reviewer, tmp_path, capsys):
    base = tmp_path / "base.json"
    repair = tmp_path / "repair.json"
    out = tmp_path / "merged.json"
    base.write_text(json.dumps({
        "source": "databricks", "questions": [{
            "id": 1, "question": "How?", "claims": [
                {"id": "a", "marker": "old"}]}]}))
    repair.write_text(json.dumps({
        "source": "databricks", "questions": [{
            "id": 1, "question": "How?", "claims": [
                {"id": "a", "marker": "new"}]}]}))

    result = reviewer.main([
        "merge", "--candidates", str(base), str(repair),
        "--out", str(out)])

    merged = json.loads(out.read_text())
    assert result == 0
    assert merged["questions"][0]["claims"][0]["marker"] == "new"
    assert merged["merge_provenance"]["report_count"] == 2
    assert "1 questions, 1 claims" in capsys.readouterr().out
def test_selection_problems_rejects_stale_reverse_transition(reviewer):
    claim = _sample_report()["questions"][0]["claims"][0]
    claim["transitions"] = [["A", "B"]]
    path = claim["candidate_paths"][0]
    path["orientation"] = "right-to-left"
    selection = {
        "path_ids": [path["id"]],
        "selected_candidate_by_anchor": {"A": "a1", "B": "b1"},
    }

    problems = reviewer.selection_problems(claim, selection)

    assert problems == ["missing required transition: A -> B"]
def test_selection_problems_accepts_declared_reverse_transition(reviewer):
    claim = _sample_report()["questions"][0]["claims"][0]
    claim["transitions"] = [["A", "B"]]
    claim["reverse_transitions"] = [["A", "B"]]
    path = claim["candidate_paths"][0]
    path["orientation"] = "right-to-left"
    selection = {
        "path_ids": [path["id"]],
        "selected_candidate_by_anchor": {"A": "a1", "B": "b1"},
    }

    problems = reviewer.selection_problems(claim, selection)

    assert problems == []
def test_compact_reviewed_report_keeps_only_selected_proof(
        reviewer, builder, tmp_path):
    report = _sample_report()
    selections = [
        reviewer.unique_selections(claim)[0]
        for claim in report["questions"][0]["claims"]
    ]
    decisions = {
        "questions": [{
            "id": 1,
            "status": "accepted",
            "answer": "A calls B, and C calls D.",
            "claims": [{
                "id": claim["id"],
                "status": "accepted",
                "selection_id": selection["selection_id"],
                "claim_correct": True,
                "complete": True,
            } for claim, selection in zip(
                report["questions"][0]["claims"], selections)],
        }],
    }
    reviewed = reviewer.apply_decisions(report, decisions)
    reviewed["source_root"] = "/machine/specific/spool-corpus"
    reviewed["provenance"] = {
        "database": "/machine/specific/ariadne.db",
        "database_sha256": "f" * 64,
    }

    compact = reviewer.compact_reviewed_report(reviewed)

    assert compact["status"] == "reviewed-gold"
    assert compact["format"] == "compact-reviewed-gold-v1"
    assert "source_root" not in compact
    assert compact["provenance"]["database"] == "ariadne.db"
    first = compact["questions"][0]["claims"][0]
    assert [
        candidate["canonical_id"]
        for anchor in first["anchors"]
        for candidate in anchor["candidates"]
    ] == ["a1", "b1"]
    assert [path["id"] for path in first["candidate_paths"]] == ["A->B#1"]
    assert "coherent_path_sets" not in first
    assert first["candidate_paths"][0]["materialization"]["excerpts"]

    output = tmp_path / "compact.json"
    output.write_text(json.dumps(compact))
    assert builder.validate_review(output) == []


def test_compact_cli_refuses_pending_input_and_writes_reviewed_gold(
        reviewer, tmp_path):
    pending = tmp_path / "pending.json"
    pending.write_text(json.dumps(_sample_report()))
    with pytest.raises(ValueError, match="reviewed-gold"):
        reviewer.main([
            "compact", "--reviewed", str(pending),
            "--out", str(tmp_path / "bad.json")])
