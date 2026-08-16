from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def measure():
    path = (Path(__file__).parents[1]
            / "evaluation/chain-benchmark/measure_ariadne.py")
    spec = importlib.util.spec_from_file_location("measure_ariadne", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gold_question(question_id=1):
    entry = "semanticdb source entry"
    terminal = "semanticdb source terminal"
    return {
        "id": question_id,
        "question": "How does entry reach terminal?",
        "review": {
            "status": "accepted",
            "answer": "Entry calls terminal."},
        "claims": [{
            "id": "entry-terminal",
            "assertion": "Entry calls terminal.",
            "anchors": [
                {
                    "anchor": "entry",
                    "candidates": [{
                        "canonical_id": entry,
                        "qualified_name": "pkg.EntryHandlerLongName",
                        "file": "src/Chain.py",
                        "line_start": 1, "line_end": 2}]},
                {
                    "anchor": "terminal",
                    "candidates": [{
                        "canonical_id": terminal,
                        "qualified_name": "pkg.TerminalBackendLongName",
                        "file": "src/Chain.py",
                        "line_start": 3, "line_end": 4}]}],
            "witnesses": [{
                "id": "terminal-call",
                "file": "src/Chain.py",
                "line_start": 1, "line_end": 2,
                "contains": ["TerminalBackendLongName"]}],
            "candidate_paths": [{
                "id": "entry-terminal#1",
                "connects": ["entry", "terminal"],
                "nodes": [
                    {
                        "canonical_id": entry,
                        "qualified_name": "pkg.EntryHandlerLongName",
                        "file": "src/Chain.py",
                        "line_start": 1, "line_end": 2},
                    {
                        "canonical_id": terminal,
                        "qualified_name": "pkg.TerminalBackendLongName",
                        "file": "src/Chain.py",
                        "line_start": 3, "line_end": 4}],
                "edges": [{
                    "caller_canonical_id": entry,
                    "callee_canonical_id": terminal,
                    "caller": "pkg.EntryHandlerLongName",
                    "callee": "pkg.TerminalBackendLongName",
                    "edge_type": "call",
                    "file": "src/Chain.py", "line": 2,
                    "compiler_verified": True}],
                "all_edges_compiler_verified": True,
                "proof_errors": []}],
            "review": {
                "status": "accepted",
                "selected_candidate_by_anchor": {
                    "entry": entry, "terminal": terminal},
                "selected_path_ids": ["entry-terminal#1"],
                "claim_correct": True, "complete": True}}]}


def _write_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    source = corpus / "repo/src/Chain.py"
    source.parent.mkdir(parents=True)
    content = (
        "def EntryHandlerLongName():\n"
        "    return TerminalBackendLongName()\n"
        "def TerminalBackendLongName():\n"
        "    return DatabaseRecordLongName\n")
    source.write_text(content)
    claimed = "/corpus/repo/src/Chain.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return corpus, content, claimed, digest


def _answer(tmp_path, question_id=1):
    corpus, content, claimed, digest = _write_corpus(tmp_path)
    return corpus, {
        "id": question_id,
        "answer": (
            f"{claimed}:1\n```\n{content}```"),
        "files_read": [claimed],
        "file_hashes": {claimed: digest},
        "selected_symbols": [
            "pkg.EntryHandlerLongName",
            "pkg.TerminalBackendLongName"],
        "hydrated_symbols": [
            "pkg.EntryHandlerLongName",
            "pkg.TerminalBackendLongName"],
        "citations": [{
            "qualified_name": "pkg.EntryHandlerLongName",
            "file": "src/Chain.py",
            "line": 1, "line_end": 4,
            "call_site": "src/Chain.py:2"}],
        "elapsed_s": 4.5,
        "total_cost_usd": 0.25,
        "confidence": "high",
        "chain_complete": True}


def test_score_answers_requires_the_complete_reviewed_gold_proof(
        measure, tmp_path):
    corpus, answer = _answer(tmp_path)
    gold = {"status": "reviewed-gold", "questions": [_gold_question()]}

    report = measure.score_answers([answer], gold, corpus)

    assert report["summary"]["passed_questions"] == 1
    assert report["summary"]["questions"] == 1
    assert report["summary"]["passed_claims"] == 1
    assert report["summary"]["total_cost_usd"] == pytest.approx(0.25)
    claim = report["questions"][0]["claims"][0]
    assert claim["passed"] is True
    assert claim["missing_symbols"] == []
    assert claim["missing_definitions"] == []
    assert claim["missing_edges"] == []
    assert claim["missing_witness_fragments"] == []


def test_score_answers_keeps_missing_questions_and_middle_nodes_in_denominator(
        measure, tmp_path):
    corpus, answer = _answer(tmp_path)
    answer["selected_symbols"] = ["pkg.EntryHandlerLongName"]
    answer["hydrated_symbols"] = ["pkg.EntryHandlerLongName"]
    gold = {
        "status": "reviewed-gold",
        "questions": [_gold_question(1), _gold_question(2)]}

    report = measure.score_answers([answer], gold, corpus)

    assert report["summary"]["questions"] == 2
    assert report["summary"]["passed_questions"] == 0
    first, missing = report["questions"]
    assert first["claims"][0]["missing_symbols"] == [
        "pkg.TerminalBackendLongName"]
    assert missing["answer_present"] is False
    assert missing["passed"] is False


def test_score_answers_refuses_an_unreviewed_or_duplicate_gold(measure, tmp_path):
    corpus, answer = _answer(tmp_path)
    pending = {"status": "candidate-oracle", "questions": [_gold_question()]}
    with pytest.raises(ValueError, match="reviewed-gold"):
        measure.score_answers([answer], pending, corpus)

    duplicate = {
        "status": "reviewed-gold",
        "questions": [_gold_question(), _gold_question()]}
    with pytest.raises(ValueError, match="duplicate gold question"):
        measure.score_answers([answer], duplicate, corpus)


def test_ariadne_command_can_only_launch_the_ariadne_arm(measure, tmp_path):
    command = measure.ariadne_command(
        python=Path("/venv/python"),
        questions=tmp_path / "questions.json",
        answers=tmp_path / "answers.json",
        source="databricks", corpus=tmp_path / "corpus",
        only="4,6", concurrency=2, timeout=300)

    joined = " ".join(str(value) for value in command)
    assert "ariadne_arm.py" in joined
    assert "run_forced.py" not in joined
    assert "--arm" not in command
    assert "raw" not in command
    assert "docker" not in command
    assert command[-2:] == ["--only", "4,6"]
def test_cost_is_labeled_as_completion_only(measure, tmp_path):
    corpus, answer = _answer(tmp_path)
    gold = {"status": "reviewed-gold", "questions": [_gold_question()]}

    report = measure.score_answers([answer], gold, corpus)

    assert report["summary"]["cost_scope"] == (
        "captured LLM completions only; embedding API charges excluded")
def test_main_dry_run_prints_only_the_ariadne_command(measure, capsys):
    result = measure.main(["--dry-run", "--only", "4,6"])

    output = capsys.readouterr().out
    assert result == 0
    assert "ARIADNE-ONLY command" in output
    assert "ariadne_arm.py" in output
    assert "run_forced.py" not in output
    assert "--arm raw" not in output


def test_main_score_only_writes_the_measurement_report(measure, tmp_path):
    import json

    corpus, answer = _answer(tmp_path)
    gold_path = tmp_path / "gold.json"
    answers_path = tmp_path / "answers.json"
    report_path = tmp_path / "report.json"
    gold_path.write_text(json.dumps({
        "status": "reviewed-gold", "questions": [_gold_question()]}))
    answers_path.write_text(json.dumps([answer]))

    result = measure.main([
        "--score-only", "--gold", str(gold_path),
        "--answers", str(answers_path), "--report", str(report_path),
        "--corpus", str(corpus)])

    report = json.loads(report_path.read_text())
    assert result == 0
    assert report["measurement"] == "ariadne-only-reviewed-gold-proof"
    assert report["summary"]["passed_questions"] == 1
    assert report["gold_sha256"]


def test_main_requires_both_api_keys_before_running(
        measure, monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = measure.main([
        "--answers", str(tmp_path / "answers.json"),
        "--report", str(tmp_path / "report.json")])

    assert result == 2
    error = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in error
    assert "OPENAI_API_KEY" in error


def test_main_paid_path_invokes_ariadne_arm_and_scores_its_output(
        measure, monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    corpus, answer = _answer(tmp_path)
    gold_path = tmp_path / "gold.json"
    answers_path = tmp_path / "answers.json"
    report_path = tmp_path / "report.json"
    gold_path.write_text(json.dumps({
        "status": "reviewed-gold", "questions": [_gold_question()]}))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        output = Path(command[command.index("--out") + 1])
        output.write_text(json.dumps([answer]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(measure.subprocess, "run", fake_run)

    result = measure.main([
        "--gold", str(gold_path), "--answers", str(answers_path),
        "--report", str(report_path), "--corpus", str(corpus)])

    joined = " ".join(captured["command"])
    assert result == 0
    assert "ariadne_arm.py" in joined
    assert "run_forced.py" not in joined
    assert "--arm" not in captured["command"]
    assert json.loads(report_path.read_text())["summary"]["passed_questions"] == 1


def test_main_require_perfect_fails_when_any_gold_question_is_missing(
        measure, tmp_path):
    import json

    corpus, answer = _answer(tmp_path)
    gold_path = tmp_path / "gold.json"
    answers_path = tmp_path / "answers.json"
    report_path = tmp_path / "report.json"
    gold_path.write_text(json.dumps({
        "status": "reviewed-gold",
        "questions": [_gold_question(1), _gold_question(2)]}))
    answers_path.write_text(json.dumps([answer]))

    result = measure.main([
        "--score-only", "--require-perfect",
        "--gold", str(gold_path), "--answers", str(answers_path),
        "--report", str(report_path), "--corpus", str(corpus)])

    assert result == 1
    assert json.loads(report_path.read_text())["summary"]["questions"] == 2
def test_main_dry_run_forwards_resume_to_the_ariadne_arm(measure, capsys):
    result = measure.main(["--dry-run", "--resume", "--only", "4,6"])

    output = capsys.readouterr().out
    assert result == 0
    assert "ariadne_arm.py" in output
    assert "--resume" in output
def test_default_gold_is_the_compact_reviewed_artifact(measure):
    assert measure.GOLD.name == "gold-chain-reviewed-compact.json"
    gold = __import__("json").loads(measure.GOLD.read_text())
    assert gold["status"] == "reviewed-gold"
    assert gold["format"] == "compact-reviewed-gold-v1"
    assert len(gold["questions"]) == 22
    assert sum(len(question["claims"]) for question in gold["questions"]) == 45
def test_ariadne_command_forwards_per_question_diagnostic_trace_directory(
        measure, tmp_path):
    trace_dir = tmp_path / "traces"

    command = measure.ariadne_command(
        python=Path("/venv/python"),
        questions=tmp_path / "questions.json",
        answers=tmp_path / "answers.json",
        source="databricks", corpus=tmp_path / "corpus",
        only=None, concurrency=2, timeout=300, trace_dir=trace_dir)

    assert command[command.index("--trace-dir") + 1] == str(trace_dir)


def test_main_dry_run_enables_diagnostic_capture_by_default(measure, capsys):
    result = measure.main([
        "--dry-run",
        "--answers", "evaluation/chain-benchmark/live-answers.json"])

    output = capsys.readouterr().out
    assert result == 0
    assert "--trace-dir" in output
    assert "live-answers-traces" in output
def test_reviewed_gold_accepts_hash_verified_short_source_lines(
        measure, tmp_path):
    corpus, answer = _answer(tmp_path)
    source = corpus / "repo/src/Chain.py"
    content = (
        "def EntryHandlerLongName():\n"
        "    return false\n"
        "def TerminalBackendLongName():\n"
        "    return DatabaseRecordLongName\n")
    source.write_text(content)
    claimed = "/corpus/repo/src/Chain.py"
    answer["answer"] = f"{claimed}:1\n```\n{content}```"
    answer["file_hashes"] = {
        claimed: hashlib.sha256(source.read_bytes()).hexdigest()[:16]}
    gold_question = _gold_question()
    gold_question["claims"][0]["witnesses"][0]["contains"] = [
        "return false"]

    report = measure.score_answers(
        [answer], {"status": "reviewed-gold",
                   "questions": [gold_question]}, corpus)

    claim = report["questions"][0]["claims"][0]
    assert claim["missing_witness_fragments"] == []
    assert claim["passed"] is True
