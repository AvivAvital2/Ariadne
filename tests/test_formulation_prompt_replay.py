from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = (Path(__file__).parents[1] / "evaluation/chain-benchmark" /
            "replay_formulation_prompt.py")
    spec = importlib.util.spec_from_file_location("replay_formulation_prompt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recorded_selection_preserves_only_selected_occurrences():
    module = _module()
    answer = {
        "selected_route_ids": ["R2", "R1", "R2"],
        "selected_section_ids": ["S3"],
        "selected_symbols": ["pkg.Entry", "pkg.Done"],
        "route_candidate_occurrences": {
            "R1": [["entry", "file.py", 1]],
            "R2": [["done", "file.py", 2], ["done", "file.py", 2]],
            "R3": [["noise", "noise.py", 3]],
        },
    }

    selection = module.recorded_selection(answer)

    assert selection.route_ids == ("R2", "R1")
    assert selection.section_ids == ("S3",)
    assert selection.symbols == ["pkg.Entry", "pkg.Done"]
    assert selection.occurrence_keys == (
        ("done", "file.py", 2), ("entry", "file.py", 1))


def test_fragment_recall_requires_every_literal_fragment_in_each_witness():
    module = _module()
    question = {"claims": [{
        "id": "complete", "witnesses": [{
            "id": "w1", "contains": ["join first", "emit rows"]}],
    }, {
        "id": "incomplete", "witnesses": [{
            "id": "w2", "contains": ["write files", "filter dropped"]}],
    }]}

    recall = module.fragment_recall(
        "join first, then emit rows and write files", question)

    assert recall["claims_passed"] == 1
    assert recall["fragments_found"] == 3
    assert recall["fragments"] == 4
    assert recall["details"][1]["witnesses"][0]["missing"] == [
        "filter dropped"]


def test_replay_prompt_hydrates_recorded_occurrences_and_audits_story(
        monkeypatch, tmp_path):
    from types import SimpleNamespace

    module = _module()
    occurrence = ("pkg.Entry", "entry.py", 1)
    citation = SimpleNamespace(qualified_name="pkg.Entry")
    excerpt = SimpleNamespace(content="join first and emit rows")
    hop = SimpleNamespace(citation=citation, source_excerpts=(excerpt,))
    evidence = SimpleNamespace(hops=(hop,))
    service = SimpleNamespace(
        library=object(),
        config=SimpleNamespace(get_all_source_paths=lambda: {"src": tmp_path}))
    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, "get", staticmethod(lambda: service))
    monkeypatch.setattr(
        module.profiler, "recorded_route_specs",
        lambda answer: [{"id": "R1", "route": ("pkg.Entry",)}])
    monkeypatch.setattr(module.profiler, "_matches", lambda *args: ["match"])
    monkeypatch.setattr("library.chain_answer.evidence_for", lambda *args, **kwargs: evidence)
    monkeypatch.setattr("library.chain_menu._occurrence_key", lambda value: occurrence)
    monkeypatch.setattr(
        "library.chain_menu.hydrate_selected_hops",
        lambda *args, **kwargs: ((hop,), ()))
    monkeypatch.setattr(
        "library.chain_menu.fetch_selected",
        lambda *args, **kwargs: SimpleNamespace(
            definitions={"pkg.Entry": "definition"}, sections=[]))
    story = SimpleNamespace(nodes=(SimpleNamespace(excerpts=(excerpt,)),), edges=())
    monkeypatch.setattr("library.chain_story.build_story_ir", lambda *args: story)
    monkeypatch.setattr(
        "library.chain_story.render_story_evidence",
        lambda value: "join first and emit rows")
    answer = {
        "id": 4, "question": "How?", "selected_route_ids": ["R1"],
        "selected_symbols": ["pkg.Entry"],
        "route_candidate_occurrences": {"R1": [list(occurrence)]},
    }
    question = {"id": 4, "claims": [{
        "id": "claim", "witnesses": [{
            "id": "w", "contains": ["join first", "emit rows"]}],
    }]}

    report, prompt = module.replay_prompt(
        answer, question, source="src")

    assert prompt == "join first and emit rows"
    assert report["matched_occurrences"] == 1
    assert report["story_nodes"] == 1
    assert report["source_excerpts"] == 1
    assert report["fragment_recall"]["claims_passed"] == 1


def test_replay_prompt_rejects_missing_routes_or_unmatched_occurrences(monkeypatch):
    from types import SimpleNamespace

    module = _module()
    monkeypatch.setattr(module.profiler, "recorded_route_specs", lambda answer: [])
    with pytest.raises(ValueError, match="no recorded routes"):
        module.replay_prompt({}, {}, source="src")

    monkeypatch.setattr(
        module.profiler, "recorded_route_specs",
        lambda answer: [{"id": "R1", "route": ("pkg.Entry",)}])
    monkeypatch.setattr(module.profiler, "_matches", lambda *args: ["match"])
    service = SimpleNamespace(library=object())
    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, "get", staticmethod(lambda: service))
    monkeypatch.setattr(
        "library.chain_answer.evidence_for",
        lambda *args, **kwargs: SimpleNamespace(hops=()))
    answer = {
        "selected_route_ids": ["R1"], "selected_symbols": ["pkg.Entry"],
        "route_candidate_occurrences": {"R1": [["unmatched"]]},
    }
    with pytest.raises(ValueError, match="none of the recorded occurrences"):
        module.replay_prompt(answer, {}, source="src")


def test_main_writes_report_and_single_prompt(tmp_path, monkeypatch, capsys):
    module = _module()
    answers = tmp_path / "answers.json"
    gold = tmp_path / "gold.json"
    output = tmp_path / "report.json"
    prompt_output = tmp_path / "prompt.txt"
    answers.write_text(json.dumps([{"id": 4}, {"id": 6}]))
    gold.write_text(json.dumps({"questions": [{"id": 4}, {"id": 6}]}))

    def fake_replay(answer, question, *, source, **kwargs):
        return ({
            "id": answer["id"], "fragment_recall": {
                "claims_passed": 1, "claims": 2,
                "fragments_found": 3, "fragments": 4},
            "story_nodes": 2, "source_excerpts": 3},
                f"prompt-{answer['id']}")

    monkeypatch.setattr(module, "replay_prompt", fake_replay)

    assert module.main([
        "--answers", str(answers), "--gold", str(gold), "--only", "4",
        "--source", "src", "--out", str(output),
        "--prompt-out", str(prompt_output),
    ]) == 0
    assert json.loads(output.read_text())[0]["id"] == 4
    assert prompt_output.read_text() == "prompt-4\n"
    text = capsys.readouterr().out
    assert "id 4 prompt replay starting" in text
    assert "gold claims 1/2; fragments 3/4" in text

    all_output = tmp_path / "all.json"
    assert module.main([
        "--answers", str(answers), "--gold", str(gold),
        "--source", "src", "--out", str(all_output),
    ]) == 0
    assert [row["id"] for row in json.loads(all_output.read_text())] == [4, 6]


def test_main_requires_one_question_when_writing_prompt(tmp_path, monkeypatch):
    module = _module()
    answers = tmp_path / "answers.json"
    gold = tmp_path / "gold.json"
    answers.write_text(json.dumps([{"id": 4}, {"id": 6}]))
    gold.write_text(json.dumps({"questions": [{"id": 4}, {"id": 6}]}))
    monkeypatch.setattr(
        module, "replay_prompt",
        lambda answer, question, source: ({"id": answer["id"]}, "prompt"))

    with pytest.raises(ValueError, match="exactly one question"):
        module.main([
            "--answers", str(answers), "--gold", str(gold),
            "--out", str(tmp_path / "out.json"),
            "--prompt-out", str(tmp_path / "prompt.txt"),
        ])
def test_scoped_all_selection_keeps_summary_relevant_routes():
    module = _module()
    from library.chain_menu import RouteMenu
    menu = RouteMenu(
        routes={
            "R1": ("pkg.Command.run", "pkg.Output.finish"),
            "R2": ("pkg.Rows.scan",),
        },
        route_summaries={
            "R1": "Emits resulting rows after classification.",
            "R2": "Inspects a schema.",
        },
    )

    selection, scoped = module.scoped_all_selection(
        menu, "Which path emits resulting rows?", max_routes=1)

    assert selection.route_ids == ("R1",)
    assert tuple(scoped.routes) == ("R1",)
