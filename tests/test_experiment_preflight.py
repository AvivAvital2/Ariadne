"""Preflight capture: exact model-facing prompts, never silent fallback.

Without scripted replies the provider captures the first prompt and
STOPS — a run that cannot answer a phase must not quietly continue as
something other than what it claims to be. With replies, every phase's
exact card surface is recorded structurally (label + card text, not
prompt substrings), replies are analyzed into parsed/unknown ids, and a
recorded reply that hit its output cap is flagged truncated from its own
usage telemetry.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp_preflight",
    ROOT / "evaluation" / "chain-benchmark" / "exp_preflight.py")
exp_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp_preflight)


def run(coroutine):
    import asyncio
    return asyncio.get_event_loop_policy().new_event_loop(
    ).run_until_complete(coroutine)


class TestScriptedChat:
    def test_unscripted_call_captures_the_prompt_and_stops(self):
        scripted = exp_preflight.ScriptedChat()

        with pytest.raises(exp_preflight.PreflightStop, match="captured"):
            run(scripted(
                [{"role": "user", "content": "S1. pkg.A\nS2. pkg.B"}],
                phase="scip-symbol-select", max_tokens=256))

        assert len(scripted.records) == 1
        record = scripted.records[0]
        assert record["phase"] == "scip-symbol-select"
        assert record["card_counts"] == {"symbol": 2}
        assert record["cards"] == [["S1", "pkg.A"], ["S2", "pkg.B"]]
        assert "scripted" not in record

    def test_recorded_reply_carries_parsing_and_truncation_telemetry(self):
        scripted = exp_preflight.ScriptedChat({
            "scip-body-select": [{
                "response": "B1, B9",
                "usage": {"output_tokens": 128, "max_tokens": 128}}]})

        reply = run(scripted(
            [{"role": "user", "content": "B1. a\nB2. b"}],
            phase="scip-body-select", max_tokens=128))

        assert reply == "B1, B9"
        record = scripted.records[0]
        assert record["parsed_selected_ids"] == ["B1"]
        assert record["unknown_ids"] == ["B9"]
        assert record["truncated"] is True
        assert record["mapping_failed"] is False

    def test_reply_with_no_resolvable_ids_is_flagged_unmapped(self):
        scripted = exp_preflight.ScriptedChat({
            "scip-symbol-select": [{
                "response": "C1: S7 S9",
                "usage": {"output_tokens": 10, "max_tokens": 256}}]})

        run(scripted(
            [{"role": "user", "content": "S1. pkg.A"}],
            phase="scip-symbol-select", max_tokens=256))

        record = scripted.records[0]
        assert record["mapping_failed"] is True
        assert record["unknown_ids"] == ["S7", "S9"]

    def test_symbol_menu_at_cap_reports_unknowable_overflow_as_null(self):
        prompt = "\n".join(f"S{index}. pkg.M{index}"
                           for index in range(1, 501))
        scripted = exp_preflight.ScriptedChat(
            {"scip-symbol-select": ["C1: S1"]})

        run(scripted([{"role": "user", "content": prompt}],
                     phase="scip-symbol-select", max_tokens=256))

        record = scripted.records[0]
        assert record["card_counts"]["symbol"] == 500
        assert record["at_cap"] is True
        assert record["cards_total_before_cap"] is None
        assert record["overflow"] is None

    def test_plain_string_replies_remain_supported(self):
        scripted = exp_preflight.ScriptedChat({"completion": ["answer"]})
        sink = []

        reply = run(scripted([{"role": "user", "content": "x" * 400}],
                             phase="completion", max_tokens=4096,
                             usage_sink=sink))

        assert reply == "answer"
        assert sink[0]["input_tokens"] == 100
