from __future__ import annotations
import pytest

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    'ariadne_arm', ROOT / 'evaluation' / 'spool-clean-room' / 'ariadne_arm.py')
arm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(arm)


def test_materializes_only_cited_coordinates(tmp_path):
    source = tmp_path / 'delta' / 'Flow.scala'
    source.parent.mkdir(parents=True)
    source.write_text('object Flow {\n  def entry = Target.run()\n  def other = 1\n}\n')
    text, files, hashes = arm._materialize_citations([
        {'file': 'delta/Flow.scala', 'line': 2},
        {'file': 'delta/Flow.scala', 'line': 2},
        {'file': 'delta/Missing.scala', 'line': 1},
    ], tmp_path)
    assert '/corpus/delta/Flow.scala:2' in text
    assert 'def entry = Target.run()' in text
    assert files == ['/corpus/delta/Flow.scala']
    assert hashes[files[0]] == hashlib.sha256(source.read_bytes()).hexdigest()[:16]


def test_resolves_scip_path_under_unique_repo_prefix(tmp_path):
    source = tmp_path / 'spark' / 'sql' / 'Flow.scala'
    source.parent.mkdir(parents=True)
    source.write_text('object Flow\n')
    resolved = arm._resolve_corpus_file(tmp_path, 'sql/Flow.scala')
    assert resolved == (source, 'spark/sql/Flow.scala')


def test_ambiguous_repo_relative_path_fails_closed(tmp_path):
    for repo in ('spark', 'delta'):
        source = tmp_path / repo / 'src' / 'Flow.scala'
        source.parent.mkdir(parents=True)
        source.write_text(f'object {repo}\n')
    assert arm._resolve_corpus_file(tmp_path, 'src/Flow.scala') is None


def test_materializes_definition_and_call_site(tmp_path):
    source = tmp_path / 'delta' / 'Flow.scala'
    source.parent.mkdir(parents=True)
    source.write_text(
        'object Flow {\n  def target = 1\n  def caller = target\n}\n')
    text, files, _ = arm._materialize_citations([{
        'file': 'delta/Flow.scala', 'line': 2,
        'call_site': 'delta/Flow.scala:3',
    }], tmp_path)
    assert '/corpus/delta/Flow.scala:2' in text
    assert '/corpus/delta/Flow.scala:3' in text
    assert files == ['/corpus/delta/Flow.scala']
def test_response_diagnostics_survive_into_the_benchmark_artifact():
    class Response:
        confidence_reasons = ["one unsupported claim"]
        evidence_gaps = ["unsupported location"]
        chain_summary = {"hops": 3}
        claims = [{"text": "proved"}]
        chain_complete = False
        completeness_reasons = ["chain truncated"]
        formulation_complete = True
        formulation_reasons = []
        scope_complete = False
        scope_reasons = ["unresolved positioning path: missing.py"]
        transition_claims = [{"text": "edge"}]
        graph_diagnostics = {"clew_selection": {"status": "selected"}}

    diagnostics = arm._response_diagnostics(Response())

    assert diagnostics["confidence_reasons"] == ["one unsupported claim"]
    assert diagnostics["evidence_gaps"] == ["unsupported location"]
    assert diagnostics["chain_summary"] == {"hops": 3}
    assert diagnostics["claims"] == [{"text": "proved"}]
    assert diagnostics["chain_complete"] is False
    assert diagnostics["formulation_complete"] is True
    assert diagnostics["scope_complete"] is False
    assert diagnostics["graph_diagnostics"] == {
        "clew_selection": {"status": "selected"}}
    assert diagnostics["chain_citations"] == []
    assert diagnostics["phase_timings"] == {}
    assert diagnostics["llm_calls"] == 0
def test_usage_summary_prices_normal_cached_and_output_tokens():
    rows = [{
        "model": "claude-opus-4-8", "input_tokens": 1000,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 3000, "output_tokens": 4000,
    }]

    summary = arm._usage_summary(rows)

    assert summary["num_turns"] == 1
    assert summary["token_usage"]["input_tokens"] == 1000
    assert summary["token_usage"]["cache_creation_input_tokens"] == 2000
    assert summary["token_usage"]["cache_read_input_tokens"] == 3000
    assert summary["token_usage"]["output_tokens"] == 4000
    assert summary["total_cost_usd"] == pytest.approx(0.119)
def test_direct_script_entrypoint_is_after_every_helper_definition():
    source = (ROOT / "evaluation" / "spool-clean-room" / "ariadne_arm.py").read_text()
    assert source.index("def _usage_summary") < source.index("if __name__ == '__main__'")
def test_route_ledger_survives_into_the_benchmark_artifact():
    class Response:
        route_candidates = {"R1": ["api.run", "db.save"]}
        selected_route_ids = ["R1"]
        selected_section_ids = ["S2"]
        selected_symbols = ["api.run", "db.save"]
        hydrated_symbols = ["db.save"]
        hydrated_sections = [{"title": "DB", "heading": "Writes"}]
        excluded_question_symbols = ["cache.load"]
        cited_route_ids = ["R1"]

    diagnostics = arm._response_diagnostics(Response())

    assert diagnostics["route_candidates"] == {"R1": ["api.run", "db.save"]}
    assert diagnostics["selected_route_ids"] == ["R1"]
    assert diagnostics["hydrated_symbols"] == ["db.save"]
    assert diagnostics["excluded_question_symbols"] == ["cache.load"]
    assert diagnostics["cited_route_ids"] == ["R1"]
def test_materialization_reuses_one_source_snapshot_across_answers(tmp_path):
    source = tmp_path / "delta" / "Flow.scala"
    source.parent.mkdir(parents=True)
    source.write_text("first\nsecond\n")
    cache = {}

    first = arm._materialize_citations(
        [{"file": "delta/Flow.scala", "line": 1}], tmp_path,
        file_cache=cache)
    source.write_text("changed\n")
    second = arm._materialize_citations(
        [{"file": "delta/Flow.scala", "line": 1}], tmp_path,
        file_cache=cache)

    assert first == second
    assert len(cache) == 1


def test_resume_reuses_only_matching_question_source_and_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    output = tmp_path / "answers.json"
    output.write_text(__import__("json").dumps([
        {"id": 1, "question": "one", "benchmark_source": "databricks",
         "benchmark_corpus": str(corpus.resolve())},
        {"id": 2, "question": "stale", "benchmark_source": "databricks",
         "benchmark_corpus": str(corpus.resolve())},
        {"id": 3, "question": "three", "benchmark_source": "other",
         "benchmark_corpus": str(corpus.resolve())},
        {"id": 4, "question": "four", "benchmark_source": "databricks",
         "benchmark_corpus": str((tmp_path / "other-corpus").resolve())},
    ]))

    resumed = arm._load_resumable_answers(
        output,
        [{"id": 1, "after": "one"}, {"id": 2, "after": "two"},
         {"id": 3, "after": "three"}, {"id": 4, "after": "four"}],
        source="databricks", corpus=corpus)

    assert list(resumed) == [1]


def test_checkpoint_is_atomic_and_sorted(tmp_path):
    output = tmp_path / "answers.json"

    arm._write_checkpoint(output, [{"id": 2}, {"id": 1}])

    assert [row["id"] for row in __import__("json").loads(output.read_text())] == [1, 2]
    assert not output.with_name(output.name + ".tmp").exists()
@pytest.mark.asyncio
async def test_complete_resume_makes_zero_ariadne_service_calls(
        tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    questions = tmp_path / "questions.json"
    requirements = tmp_path / "requirements.json"
    output = tmp_path / "answers.json"
    questions.write_text(json.dumps([{"id": 4, "after": "question"}]))
    requirements.write_text(json.dumps([{"id": 4, "flag": False}]))
    output.write_text(json.dumps([{
        "id": 4, "question": "question",
        "benchmark_source": "databricks",
        "benchmark_corpus": str(corpus.resolve()),
    }]))
    monkeypatch.setattr(arm, "REQS", requirements)
    monkeypatch.setattr(
        __import__("config"), "get_config",
        lambda: SimpleNamespace(model="claude-opus-4-8"))

    class ForbiddenService:
        @staticmethod
        def get():
            raise AssertionError("complete resume initialized Ariadne")

    monkeypatch.setitem(
        sys.modules, "ariadne_mcp.service",
        SimpleNamespace(AriadneService=ForbiddenService))
    monkeypatch.setattr(sys, "argv", [
        "ariadne_arm.py", "--resume", "--questions", str(questions),
        "--out", str(output), "--corpus", str(corpus)])
    # arm.main() writes this env var process-wide; registering the key with
    # monkeypatch makes teardown restore its pre-test absence, so the
    # repair-path tests that run later still exercise repair.
    monkeypatch.setenv("ARIADNE_BENCHMARK_NO_REPAIR", "0")

    assert await arm.main() == 0
@pytest.mark.asyncio
async def test_benchmark_arm_passes_question_id_to_phase_trace(tmp_path):
    from types import SimpleNamespace

    seen = {}

    class Service:
        async def ask(self, question, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                answer="", chain_files=[], citations=[], sources=[],
                confidence="low")

    await arm._one(Service(), 67, "How does it commit?", "databricks", tmp_path)

    assert seen["trace_id"] == 67
@pytest.mark.asyncio
async def test_benchmark_arm_supports_service_without_trace_keyword(tmp_path):
    from types import SimpleNamespace

    class Service:
        async def ask(self, question, source=None):
            return SimpleNamespace(
                answer="answer", chain_files=[], citations=[], sources=[],
                confidence="low")

    row = await arm._one(
        Service(), 4, "How does it merge?", "databricks", tmp_path)

    assert row["id"] == 4
    assert row["answer"] == "answer"
def test_benchmark_arm_does_not_hide_scope_failures_with_prompt_char_ceiling():
    source = (
        ROOT / "evaluation" / "spool-clean-room" / "ariadne_arm.py"
    ).read_text()

    assert "ARIADNE_MAX_PROMPT_CHARS" not in source
def test_response_diagnostics_include_replay_critical_final_state():
    class Response:
        chain_citations = [{"qualified_name": "pkg.Target", "file": "Target.py"}]
        unsupported_locations = ["Unknown.py:9"]
        phase_timings = {"search": 0.1, "total": 1.5}
        llm_calls = 4

    diagnostics = arm._response_diagnostics(Response())

    assert diagnostics["chain_citations"] == Response.chain_citations
    assert diagnostics["unsupported_locations"] == ["Unknown.py:9"]
    assert diagnostics["phase_timings"] == {"search": 0.1, "total": 1.5}
    assert diagnostics["llm_calls"] == 4


def test_diagnostic_trace_sidecar_is_atomic_compressed_and_hash_verified(
        tmp_path):
    import gzip
    import json

    payload = {
        "schema": "ariadne-live-diagnostic-v1",
        "id": 7,
        "llm_completions": [{
            "phase": "formulation",
            "messages": [{"role": "user", "content": "exact ledger"}],
            "response": "answer",
        }],
    }

    receipt = arm._write_diagnostic_trace(tmp_path, 7, payload)

    path = tmp_path / "q7.json.gz"
    assert receipt["schema"] == "ariadne-live-diagnostic-v1"
    assert receipt["file"] == "q7.json.gz"
    assert receipt["compressed_bytes"] == path.stat().st_size
    assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(gzip.decompress(path.read_bytes())) == payload
    assert not (tmp_path / "q7.json.gz.tmp").exists()


@pytest.mark.asyncio
async def test_one_writes_replay_complete_trace_without_embedding_it_in_answer(
        tmp_path):
    import gzip
    import json
    from types import SimpleNamespace

    class Service:
        async def ask(self, question, **kwargs):
            return SimpleNamespace(
                answer="service answer",
                chain_files=["src/Flow.py"],
                citations=[],
                chain_citations=[{
                    "qualified_name": "pkg.Flow.run",
                    "file": "src/Flow.py",
                    "line": 4,
                }],
                sources=["Flow"],
                confidence="low",
                phase_timings={"search": 0.2, "total": 1.0},
                llm_calls=3,
                unsupported_locations=["src/Other.py:2"])

    trace_dir = tmp_path / "traces"
    row = await arm._one(
        Service(), 7, "How does it flow?", "databricks", tmp_path,
        trace_dir=trace_dir)

    assert "diagnostic_trace_payload" not in row
    receipt = row["diagnostic_trace"]
    payload = json.loads(gzip.decompress(
        (trace_dir / receipt["file"]).read_bytes()))
    assert payload["question"] == "How does it flow?"
    assert payload["service_answer"] == "service answer"
    assert payload["response_diagnostics"]["chain_citations"][0][
        "qualified_name"] == "pkg.Flow.run"
    assert payload["response_diagnostics"]["phase_timings"]["total"] == 1.0
    assert payload["response_diagnostics"]["llm_calls"] == 3
    assert payload["llm_completions"] == []
    assert payload["materialized_evidence"]["files_read"] == []


def test_resume_requires_an_intact_requested_diagnostic_sidecar(tmp_path):
    import json

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    traces = tmp_path / "traces"
    output = tmp_path / "answers.json"
    payload = {"schema": "ariadne-live-diagnostic-v1", "id": 1}
    receipt = arm._write_diagnostic_trace(traces, 1, payload)
    row = {
        "id": 1,
        "question": "one",
        "benchmark_source": "databricks",
        "benchmark_corpus": str(corpus.resolve()),
        "diagnostic_trace": receipt,
    }
    output.write_text(json.dumps([row]))
    questions = [{"id": 1, "after": "one"}]

    resumed = arm._load_resumable_answers(
        output, questions, source="databricks", corpus=corpus,
        trace_dir=traces)
    assert list(resumed) == [1]

    (traces / receipt["file"]).write_bytes(b"corrupt")
    assert arm._load_resumable_answers(
        output, questions, source="databricks", corpus=corpus,
        trace_dir=traces) == {}
