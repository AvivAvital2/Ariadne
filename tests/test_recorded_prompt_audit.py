from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = (Path(__file__).parents[1] / "evaluation/chain-benchmark" /
            "audit_recorded_prompt.py")
    spec = importlib.util.spec_from_file_location("audit_recorded_prompt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _occurrence(name="pkg.Entry", line=2):
    return [
        name, "entry.py", line, line + 4, "pkg.Parent", "caller.py",
        line + 10, "calls", 1, "depth"]


def test_citation_from_occurrence_preserves_the_recorded_compiler_coordinates():
    module = _module()

    citation = module.citation_from_occurrence(_occurrence(), source="src")

    assert citation.qualified_name == "pkg.Entry"
    assert citation.file == "entry.py"
    assert citation.line_start == 2
    assert citation.line_end == 6
    assert citation.parent_qualified_name == "pkg.Parent"
    assert citation.call_site_file == "caller.py"
    assert citation.call_site_line == 12
    assert citation.relation == "calls"
    assert citation.hop == 1
    assert citation.stop_reason == "depth"
    assert citation.source_name == "src"


def test_citation_from_occurrence_rejects_incomplete_artifacts():
    module = _module()

    with pytest.raises(ValueError, match="10 fields"):
        module.citation_from_occurrence(["pkg.Entry", "entry.py"], source="src")


def test_artifact_citations_use_selected_routes_and_deduplicate_occurrences():
    module = _module()
    entry = _occurrence()
    done = _occurrence("pkg.Done", 20)
    answer = {
        "selected_route_ids": ["R2", "R1", "R2"],
        "route_candidate_occurrences": {
            "R1": [entry], "R2": [done, done],
            "R3": [_occurrence("pkg.Noise", 50)],
        },
    }

    citations = module.artifact_citations(answer, source="src")

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Done", "pkg.Entry"]


def test_exact_replay_curates_persisted_occurrences_without_reassembling_graph(
        monkeypatch, tmp_path):
    from types import SimpleNamespace

    module = _module()
    occurrence = _occurrence()
    answer = {
        "id": 4, "selected_route_ids": ["R1"],
        "selected_symbols": ["pkg.Entry", "pkg.Absent"],
        "route_candidate_occurrences": {"R1": [occurrence]},
    }
    question = {"id": 4, "claims": [{
        "id": "claim", "witnesses": [{
            "id": "witness", "contains": ["join first", "emit rows"]}],
    }]}
    excerpt = SimpleNamespace(kind="definition", content="join first and emit rows")
    captured = {}

    def curate(library, citations, **kwargs):
        captured["citations"] = citations
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            hops=[SimpleNamespace(citation=citations[0], source_excerpts=(excerpt,),
                                  document_id="doc")],
            source_gaps=("one gap",))

    service = SimpleNamespace(
        library=object(),
        config=SimpleNamespace(get_all_source_paths=lambda: {"src": tmp_path}))
    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, "get", staticmethod(lambda: service))
    monkeypatch.setattr("library.chain_bundle.curate_bundle", curate)
    monkeypatch.setattr(
        "library.chain_menu.fetch_selected",
        lambda *args: SimpleNamespace(definitions={"pkg.Entry": "definition"}, sections=[]))
    story = SimpleNamespace(
        nodes=(SimpleNamespace(excerpts=(excerpt,)),), edges=(),
        chunks=())
    monkeypatch.setattr("library.chain_story.build_story_ir", lambda *args: story)
    monkeypatch.setattr(
        "library.chain_story.render_story_evidence",
        lambda value, **kwargs: "join first and emit rows")

    report, prompt = module.exact_replay(answer, question, source="src")

    assert prompt == "join first and emit rows"
    assert captured["citations"][0].line_end == 6
    assert captured["kwargs"] == {
        "source": "src", "source_root": tmp_path,
        "materialize_source": True,
        "materialize_definition_bodies": True,
        "definition_body_symbols": None,
        "definition_body_query": None}
    assert report["replay_mode"] == "artifact-exact"
    assert report["artifact_occurrences"] == 1
    assert report["curated_hops"] == 1
    assert report["missing_selected_symbols"] == ["pkg.Absent"]
    assert report["source_excerpt_kinds"] == {"definition": 1}
    assert report["source_gaps"] == ["one gap"]
    assert report["fragment_recall"]["claims_passed"] == 1


def test_exact_replay_requires_occurrences_and_a_configured_source(monkeypatch):
    from types import SimpleNamespace

    module = _module()
    with pytest.raises(ValueError, match="no selected artifact occurrences"):
        module.exact_replay({}, {}, source="src")

    answer = {
        "selected_route_ids": ["R1"],
        "route_candidate_occurrences": {"R1": [_occurrence()]},
    }
    service = SimpleNamespace(
        library=object(),
        config=SimpleNamespace(get_all_source_paths=lambda: {}))
    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, "get", staticmethod(lambda: service))
    with pytest.raises(ValueError, match="source root unavailable"):
        module.exact_replay(answer, {}, source="src")


def test_main_writes_exact_report_and_single_prompt(tmp_path, monkeypatch, capsys):
    module = _module()
    answers = tmp_path / "answers.json"
    gold = tmp_path / "gold.json"
    output = tmp_path / "report.json"
    prompt_output = tmp_path / "prompt.txt"
    answers.write_text(json.dumps([{"id": 4}, {"id": 6}]))
    gold.write_text(json.dumps({"questions": [{"id": 4}, {"id": 6}]}))

    def replay(answer, question, *, source, complete_transitions=False, semantic_slices=False, compact_ledger=False):
        return ({
            "id": answer["id"], "fragment_recall": {
                "claims_passed": 1, "claims": 2,
                "fragments_found": 3, "fragments": 4},
            "artifact_occurrences": 5, "source_excerpts": 7},
                f"prompt-{answer['id']}")

    monkeypatch.setattr(module, "exact_replay", replay)

    assert module.main([
        "--answers", str(answers), "--gold", str(gold), "--only", "4",
        "--source", "src", "--out", str(output),
        "--prompt-out", str(prompt_output),
    ]) == 0
    assert json.loads(output.read_text())[0]["id"] == 4
    assert prompt_output.read_text() == "prompt-4\n"
    text = capsys.readouterr().out
    assert "id 4 exact prompt audit starting" in text
    assert "gold claims 1/2; fragments 3/4" in text
    assert "5 artifact occurrences, 7 excerpts" in text

    all_output = tmp_path / "all.json"
    assert module.main([
        "--answers", str(answers), "--gold", str(gold),
        "--source", "src", "--out", str(all_output),
    ]) == 0
    assert [row["id"] for row in json.loads(all_output.read_text())] == [4, 6]


def test_main_requires_one_question_when_writing_prompt(tmp_path):
    module = _module()
    answers = tmp_path / "answers.json"
    gold = tmp_path / "gold.json"
    answers.write_text(json.dumps([{"id": 4}, {"id": 6}]))
    gold.write_text(json.dumps({"questions": [{"id": 4}, {"id": 6}]}))

    with pytest.raises(ValueError, match="exactly one question"):
        module.main([
            "--answers", str(answers), "--gold", str(gold),
            "--out", str(tmp_path / "out.json"),
            "--prompt-out", str(tmp_path / "prompt.txt"),
        ])
def test_exact_replay_passes_an_explicit_definition_body_scope(monkeypatch, tmp_path):
    from types import SimpleNamespace
    module = _module()
    answer = {
        "id": 4, "selected_route_ids": ["R1"],
        "selected_symbols": ["pkg.Entry"],
        "route_candidate_occurrences": {"R1": [_occurrence()]},
    }
    captured = {}

    def curate(library, citations, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(hops=[], source_gaps=())

    service = SimpleNamespace(
        library=object(),
        config=SimpleNamespace(get_all_source_paths=lambda: {"src": tmp_path}))
    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, "get", staticmethod(lambda: service))
    monkeypatch.setattr("library.chain_bundle.curate_bundle", curate)
    monkeypatch.setattr(
        "library.chain_menu.fetch_selected",
        lambda *args: SimpleNamespace(definitions={}, sections=[]))
    story = SimpleNamespace(nodes=(), edges=(), chunks=())
    monkeypatch.setattr("library.chain_story.build_story_ir", lambda *args: story)
    monkeypatch.setattr("library.chain_story.render_story_evidence", lambda value, **kwargs: "")

    report, _ = module.exact_replay(
        answer, {"id": 4, "claims": []}, source="src",
        body_symbols=("pkg.Entry",))

    assert captured["definition_body_symbols"] == ("pkg.Entry",)
    assert report["selected_definition_bodies"] == ["pkg.Entry"]
def test_exact_replay_uses_recorded_body_scope_without_experimental_transforms(
        monkeypatch, tmp_path):
    from types import SimpleNamespace
    module = _module()
    answer = {
        "id": 4,
        "question": "How is the result persisted?",
        "selected_route_ids": ["R1"],
        "selected_symbols": ["pkg.Entry"],
        "selected_body_symbols": ["pkg.Entry"],
        "route_candidate_occurrences": {"R1": [_occurrence()]},
        "graph_diagnostics": {
            "clew_selection": {"coverage_plan": "C1: persist the result"}},
    }
    captured = {}

    def curate(library, citations, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(hops=[], source_gaps=())

    service = SimpleNamespace(
        library=object(),
        config=SimpleNamespace(get_all_source_paths=lambda: {"src": tmp_path}))
    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, "get", staticmethod(lambda: service))
    monkeypatch.setattr("library.chain_bundle.curate_bundle", curate)
    monkeypatch.setattr(
        "library.chain_menu.fetch_selected",
        lambda *args: SimpleNamespace(definitions={}, sections=[]))
    story = SimpleNamespace(nodes=(), edges=(), chunks=())
    monkeypatch.setattr("library.chain_story.build_story_ir", lambda *args: story)
    monkeypatch.setattr("library.chain_story.render_story_evidence", lambda value, **kwargs: "")

    report, _ = module.exact_replay(
        answer, {"id": 4, "claims": []}, source="src")

    assert captured["definition_body_symbols"] == ("pkg.Entry",)
    assert captured["definition_body_query"] is None
    assert report["selected_definition_bodies"] == ["pkg.Entry"]
def test_exact_replay_completes_recorded_body_scope_with_route_transitions(
        monkeypatch, tmp_path):
    from types import SimpleNamespace
    module = _module()
    root = [
        "pkg.Flow.start", "flow.py", 1, 5, "", "flow.py", 1,
        "localized", 1, "leaf"]
    middle = [
        "pkg.Flow.transform", "flow.py", 10, 15, "pkg.Flow.start",
        "flow.py", 3, "calls", 2, "leaf"]
    terminal = [
        "pkg.Flow.finish", "flow.py", 20, 25, "pkg.Flow.transform",
        "flow.py", 12, "calls", 3, "leaf"]
    answer = {
        "id": 4, "question": "How does the flow finish?",
        "selected_route_ids": ["R1"],
        "selected_symbols": [
            "pkg.Flow.start", "pkg.Flow.transform", "pkg.Flow.finish"],
        "selected_body_symbols": ["pkg.Flow.start"],
        "route_candidate_occurrences": {"R1": [root, middle, terminal]},
    }
    captured = {}

    def curate(library, citations, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(hops=[], source_gaps=())

    service = SimpleNamespace(
        library=object(),
        config=SimpleNamespace(get_all_source_paths=lambda: {"src": tmp_path}))
    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, "get", staticmethod(lambda: service))
    monkeypatch.setattr("library.chain_bundle.curate_bundle", curate)
    monkeypatch.setattr(
        "library.chain_menu.fetch_selected",
        lambda *args: SimpleNamespace(definitions={}, sections=[]))
    story = SimpleNamespace(nodes=(), edges=(), chunks=())
    monkeypatch.setattr("library.chain_story.build_story_ir", lambda *args: story)
    monkeypatch.setattr("library.chain_story.render_story_evidence", lambda value, **kwargs: "")

    report, _ = module.exact_replay(
        answer, {"id": 4, "claims": []}, source="src", complete_transitions = True, semantic_slices = True)

    assert captured["definition_body_symbols"] == (
        "pkg.Flow.start", "pkg.Flow.transform")
    assert report["selected_definition_bodies"] == [
        "pkg.Flow.start", "pkg.Flow.transform"]
def test_main_exposes_experimental_replay_flags(tmp_path, monkeypatch):
    module = _module()
    answers = tmp_path / "answers.json"
    gold = tmp_path / "gold.json"
    output = tmp_path / "report.json"
    answers.write_text(json.dumps([{"id": 4}]))
    gold.write_text(json.dumps({"questions": [{"id": 4}]}))
    captured = {}

    def replay(answer, question, *, source, complete_transitions=False,
               semantic_slices=False, compact_ledger=False):
        captured.update(
            complete_transitions=complete_transitions,
            semantic_slices=semantic_slices,
            compact_ledger=compact_ledger)
        return ({
            "id": 4, "fragment_recall": {
                "claims_passed": 0, "claims": 0,
                "fragments_found": 0, "fragments": 0},
            "artifact_occurrences": 1, "source_excerpts": 1}, "prompt")

    monkeypatch.setattr(module, "exact_replay", replay)

    assert module.main([
        "--answers", str(answers), "--gold", str(gold),
        "--source", "src", "--out", str(output),
        "--complete-transitions", "--semantic-slices",
        "--compact-ledger",
    ]) == 0
    assert captured == {
        "complete_transitions": True, "semantic_slices": True,
        "compact_ledger": True}

def test_exact_replay_keeps_reference_query_separate_from_coverage_plan():
    import inspect

    source = inspect.getsource(_module().exact_replay).replace(" ", "")
    assert "reference_query=question_text" in source
    assert "definition_body_query=(" in source

def test_exact_replay_delegates_to_production_hydration_policy():
    import inspect

    source = inspect.getsource(_module().exact_replay)
    assert "hydrate_selected_hops(" in source
    assert "qualified_caller_fanout" not in source
    assert "reference_bridges" not in source
    assert "bridge_dependencies" not in source
