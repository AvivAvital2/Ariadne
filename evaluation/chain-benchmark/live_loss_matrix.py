#!/usr/bin/env python3
"""The final live-path loss matrix: where each claim died in the real run.

Every classification comes from the saved live22 traces — the exact
prompts Ariadne showed the model, the exact replies, the recorded
selection state, the materialized evidence, and the scored final answer.
No offline selector, no replay, no reconstruction. Loss categories, in
live pipeline order:

    never-reached-live-prompt
    removed-by-500-cap
    visible-not-selected
    selected-then-pruned
    body-omitted
    materialized-omitted-from-answer
    passed

The matrix must reconcile exactly to the corrected live baseline —
9 passed claims, 36 failed, Q67 and Q147 passing — or it is wrong.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from shadow_eval import required_items

CARD_LINE = re.compile(r"(?m)^\s*([FKSGRB]\d+)\.\s*(.+)$")
LOSS_ORDER = (
    "never-reached-live-prompt", "removed-by-500-cap",
    "visible-not-selected", "selected-then-pruned", "body-omitted",
    "materialized-omitted-from-answer")


def load_trace(path) -> dict:
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def live_surfaces(trace: dict) -> dict:
    """Everything the live run actually showed, chose, and produced."""
    prompts_by_phase: dict = {}
    replies_by_phase: dict = {}
    for call in trace.get("llm_completions", ()):
        phase = str(call["phase"])
        prompt = "\n\n".join(
            str(message.get("content") or "")
            for message in call.get("messages", ()))
        prompts_by_phase.setdefault(phase, []).append(prompt)
        replies_by_phase.setdefault(phase, []).append(
            str(call.get("response") or ""))
    card_text = "\n".join(
        prompt for phase, prompts in prompts_by_phase.items()
        if phase != "completion" for prompt in prompts)
    prompt_card_names = {
        match.group(2).strip()
        for match in CARD_LINE.finditer(card_text)}
    symbol_cards = {
        match.group(2).strip()
        for prompt in prompts_by_phase.get("scip-symbol-select", ())
        for match in CARD_LINE.finditer(prompt)
        if match.group(1).startswith("S")}
    symbol_at_cap = any(
        len(CARD_LINE.findall(prompt)) >= 500
        for prompt in prompts_by_phase.get("scip-symbol-select", ()))
    diagnostics = trace.get("response_diagnostics", {})
    internal_pool = {
        name for route in (
            diagnostics.get("route_candidates") or {}).values()
        for name in route}
    return {
        "prompt_text": card_text,
        "prompt_card_names": prompt_card_names,
        "symbol_cards": symbol_cards,
        "symbol_at_cap": symbol_at_cap,
        "reply_text": "\n".join(
            reply for phase, replies in replies_by_phase.items()
            if phase != "completion" for reply in replies),
        "internal_pool": internal_pool,
        "selected_symbols": set(
            diagnostics.get("selected_symbols") or ()),
        "hydrated_symbols": set(
            diagnostics.get("hydrated_symbols") or ()),
        "body_symbols": set(
            diagnostics.get("selected_body_symbols") or ()),
        "materialized_text": (
            trace.get("materialized_evidence") or {}).get("text") or "",
        "final_answer": str(trace.get("benchmark_answer") or ""),
    }


def symbol_loss(surfaces: dict, symbol: str) -> "str | None":
    """Earliest live loss for one required symbol, or None if it survived
    every live stage a symbol can be judged at."""
    visible = (symbol in surfaces["symbol_cards"]
               or symbol in surfaces["prompt_card_names"]
               or symbol in surfaces["prompt_text"])
    if not visible:
        if (symbol in surfaces["internal_pool"]
                and surfaces["symbol_at_cap"]):
            return "removed-by-500-cap"
        return "never-reached-live-prompt"
    selected = (symbol in surfaces["selected_symbols"]
                or symbol in surfaces["reply_text"])
    if not selected:
        return "visible-not-selected"
    if (surfaces["hydrated_symbols"]
            and symbol not in surfaces["hydrated_symbols"]):
        return "selected-then-pruned"
    return None


def classify_claim(surfaces: dict, items: dict, passed: bool) -> dict:
    if passed:
        return {"loss": "passed", "blocking": []}
    blocking = []
    for symbol in sorted(items["symbols"]):
        loss = symbol_loss(surfaces, symbol)
        if loss is not None:
            blocking.append({"item": symbol, "loss": loss})
    fragments = [
        fragment for witness in items["witnesses"]
        for fragment in witness["contains"]]
    for fragment in fragments:
        if fragment in surfaces["final_answer"]:
            continue
        if fragment in surfaces["materialized_text"]:
            blocking.append({
                "item": f"fragment:{fragment[:60]}",
                "loss": "materialized-omitted-from-answer"})
        else:
            blocking.append({
                "item": f"fragment:{fragment[:60]}",
                "loss": "body-omitted"})
    order = {loss: index for index, loss in enumerate(LOSS_ORDER)}
    blocking.sort(key=lambda row: (order[row["loss"]], row["item"]))
    return {
        "loss": (blocking[0]["loss"] if blocking
                 else "materialized-omitted-from-answer"),
        "blocking": blocking,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=str(
        HERE / "live22-diagnostic-answers-traces"))
    parser.add_argument("--answers", default=str(
        HERE / "live22-diagnostic-answers.json"))
    parser.add_argument("--gold", default=str(
        HERE / "gold-chain-reviewed-compact.json"))
    parser.add_argument("--corpus", default=str(ROOT / "spool-corpus"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import measure_ariadne

    gold = json.loads(Path(args.gold).read_text())
    answers = json.loads(Path(args.answers).read_text())
    report = measure_ariadne.score_answers(
        answers, gold, Path(args.corpus))
    observed = {
        (int(question["id"]), str(claim["id"])): bool(claim.get("passed"))
        for question in report.get("questions", ())
        for claim in question.get("claims", ())}
    passing_questions = sorted(
        int(question["id"]) for question in report.get("questions", ())
        if question.get("passed"))

    matrix = []
    histogram: Counter = Counter()
    for question in gold["questions"]:
        question_id = int(question["id"])
        trace_path = Path(args.trace_dir) / f"q{question_id}.json.gz"
        surfaces = live_surfaces(load_trace(trace_path))
        for claim in question.get("claims", ()):
            key = (question_id, str(claim.get("id")))
            result = classify_claim(
                surfaces, required_items(claim),
                observed.get(key, False))
            histogram[result["loss"]] += 1
            matrix.append({
                "question": question_id, "claim": key[1],
                "loss": result["loss"],
                "blocking": result["blocking"][:12]})

    passed = histogram.get("passed", 0)
    reconciled = (
        passed == 9
        and sum(histogram.values()) - passed == 36
        and passing_questions == [67, 147])
    payload = {
        "measurement": "live-path-loss-matrix",
        "reconciled": reconciled,
        "baseline": {"passed_claims": passed,
                     "failed_claims": sum(histogram.values()) - passed,
                     "passing_questions": passing_questions},
        "loss_histogram": dict(sorted(
            histogram.items(), key=lambda item: -item[1])),
        "claims": matrix,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"reconciled to live baseline: {reconciled}")
    for loss, count in histogram.most_common():
        print(f"  {loss}: {count}")
    if not reconciled:
        print("RECONCILIATION FAILED — matrix must not be used")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
