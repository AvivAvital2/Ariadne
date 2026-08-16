"""Historical cost calibration and projection from saved trace telemetry.

Cost is a first-class correctness constraint: the saved live22 traces
concentrate 92.7% of input tokens in three phases (formulation,
exact-route selection, symbol selection), and every symbol-selection
prompt reached its 500-card cap. This module reproduces those facts from
the traces themselves — calibration, not folklore — and projects costs
through a versioned local price file, never a browsed or assumed price.
"""
from __future__ import annotations

import gzip
import json
import statistics
from pathlib import Path


def load_trace(path) -> dict:
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def trace_cost_profile(trace_dir) -> dict:
    """Per-phase and per-question call/token telemetry from saved traces."""
    per_phase: dict = {}
    per_question: dict = {}
    for path in sorted(Path(trace_dir).glob("q*.json.gz")):
        payload = load_trace(path)
        question_id = int(payload["id"])
        question_row = per_question.setdefault(question_id, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0})
        for call in payload.get("llm_completions", ()):
            usage = call.get("usage") or {}
            phase = str(call.get("phase") or "unknown")
            row = per_phase.setdefault(phase, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "prompt_chars": 0, "at_output_cap": 0,
                "input_token_samples": []})
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
            row["calls"] += 1
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["prompt_chars"] += sum(
                len(str(message.get("content") or ""))
                for message in call.get("messages", ()))
            if output_tokens >= int(call.get("max_tokens") or 1 << 30):
                row["at_output_cap"] += 1
            row["input_token_samples"].append(input_tokens)
            question_row["calls"] += 1
            question_row["input_tokens"] += input_tokens
            question_row["output_tokens"] += output_tokens
    for row in per_phase.values():
        samples = sorted(row.pop("input_token_samples"))
        row["input_tokens_p50"] = int(statistics.median(samples))
        row["input_tokens_p95"] = samples[
            min(len(samples) - 1, int(len(samples) * 0.95))]
        row["input_tokens_max"] = samples[-1]
    total = {
        "calls": sum(row["calls"] for row in per_phase.values()),
        "input_tokens": sum(
            row["input_tokens"] for row in per_phase.values()),
        "output_tokens": sum(
            row["output_tokens"] for row in per_phase.values()),
        "questions": len(per_question),
    }
    if per_question:
        total["input_tokens_per_question"] = (
            total["input_tokens"] // len(per_question))
        total["calls_per_question"] = round(
            total["calls"] / len(per_question), 1)
    return {"per_phase": dict(sorted(per_phase.items())),
            "per_question": dict(sorted(per_question.items())),
            "total": total}


def load_price_config(path) -> dict:
    config = json.loads(Path(path).read_text())
    if "version" not in config or "models" not in config:
        raise ValueError("price config must carry version and models")
    return config


def project_cost(profile: dict, price_config: dict, model: str) -> dict:
    """Project USD cost from token telemetry under a versioned price file."""
    prices = price_config["models"].get(model)
    if prices is None:
        raise ValueError(
            f"model {model!r} absent from price config "
            f"{price_config['version']!r}")
    total = profile["total"]
    input_usd = total["input_tokens"] * prices["input_per_mtok"] / 1e6
    output_usd = total["output_tokens"] * prices["output_per_mtok"] / 1e6
    questions = max(total.get("questions", 0), 1)
    return {
        "price_config_version": price_config["version"],
        "model": model,
        "input_usd": round(input_usd, 3),
        "output_usd": round(output_usd, 3),
        "total_usd": round(input_usd + output_usd, 3),
        "usd_per_question": round(
            (input_usd + output_usd) / questions, 3),
    }
