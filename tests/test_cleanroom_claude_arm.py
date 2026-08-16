from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def raw_arm():
    path = (
        Path(__file__).parents[1]
        / "evaluation" / "cleanroom-claude" / "entrypoint.py"
    )
    spec = importlib.util.spec_from_file_location("cleanroom_claude", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prompt_requires_source_for_every_hop_without_endpoint_shortcut(raw_arm):
    assert "Quote the code that establishes EVERY hop" in raw_arm.SYSTEM
    assert "naming an intermediate mechanism without" in raw_arm.SYSTEM
    assert "ROUTE B" not in raw_arm.SYSTEM
    assert raw_arm.ALLOWED == "Read,Grep,Glob"
    assert "Bash" in raw_arm.DISALLOWED


def test_single_attempt_protocol_does_not_retry_inadequate_answer(
        raw_arm, monkeypatch):
    calls = []

    def session(question):
        calls.append(question)
        return {"answer": "memory only", "files_read": []}

    monkeypatch.setattr(raw_arm, "_one_session", session)

    result = raw_arm.ask("question")

    assert calls == ["question"]
    assert result["attempts"] == 1
    assert result["rejections"] == []
    assert result["still_inadequate"] == (
        "you did not open a single file in /corpus")


def test_session_records_limits_usage_and_only_hashable_files(
        raw_arm, monkeypatch):
    events = [
        {
            "message": {"content": [{
                "type": "tool_use", "name": "Read",
                "input": {"file_path": "/corpus/repo/Flow.scala"},
            }]},
        },
        {
            "message": {"content": [{
                "type": "tool_use", "name": "Glob",
                "input": {"path": "/corpus/repo"},
            }]},
        },
        {
            "type": "result",
            "result": "grounded answer",
            "num_turns": 7,
            "total_cost_usd": 0.42,
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
    ]
    completed = SimpleNamespace(
        stdout="\n".join(json.dumps(event) for event in events),
        stderr="", returncode=0)
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return completed

    monkeypatch.setattr(raw_arm.subprocess, "run", run)
    monkeypatch.setattr(
        raw_arm, "_sha",
        lambda path: "abc123" if path.endswith("Flow.scala") else None)
    monkeypatch.setenv("MAX_AGENT_TURNS", "9")
    monkeypatch.setenv("MAX_COST_USD", "1.25")

    result = raw_arm._one_session("question")

    command = captured["command"]
    assert command[command.index("--max-turns") + 1] == "9"
    assert command[command.index("--max-budget-usd") + 1] == "1.25"
    assert captured["kwargs"]["cwd"] == "/corpus"
    assert result["files_read"] == ["/corpus/repo/Flow.scala"]
    assert result["file_hashes"] == {"/corpus/repo/Flow.scala": "abc123"}
    assert [call["tool"] for call in result["tool_calls"]] == ["Read", "Glob"]
    assert result["num_turns"] == 7
    assert result["total_cost_usd"] == pytest.approx(0.42)
    assert result["budget"] == {"max_turns": 9, "max_cost_usd": 1.25}
def test_container_runner_forwards_identical_turn_and_cost_limits():
    script = (
        Path(__file__).parents[1]
        / "evaluation" / "cleanroom-claude" / "run.sh"
    ).read_text()

    assert '-e MAX_AGENT_TURNS="${MAX_AGENT_TURNS:-20}"' in script
    assert '-e MAX_COST_USD="${MAX_COST_USD:-2.0}"' in script
