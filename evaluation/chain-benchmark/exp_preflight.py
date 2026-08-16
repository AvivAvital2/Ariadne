"""Production-preflight capture: the exact prompts live Ariadne would send.

An internal graph route that never reaches a prompt is not menu-visible,
so menu grading must run over the exact card surface sent to each model
call. This mode runs the real ``ask`` path with ``llm.chat_complete``
replaced by a scripted provider: with saved replies it replays every
recorded response through the current renderers and captures every phase
prompt; without them it captures the first prompt and stops — it never
silently substitutes a deterministic selector while calling itself
live-shaped. No network access is possible: the patch happens before any
provider exists.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

CARD_GRAMMARS = {
    "family": re.compile(r"(?m)^\s*F\d+\."),
    "skeleton": re.compile(r"(?m)^\s*K\d+\."),
    "symbol": re.compile(r"(?m)^\s*S\d+\."),
    "component": re.compile(r"(?m)^\s*G\d+\."),
    "route": re.compile(r"(?m)^\s*R\d+\."),
    "body": re.compile(r"(?m)^\s*B\d+\."),
}

#: The live symbol menu's card limit; reaching it means overflow occurred.
SYMBOL_MENU_CAP = 500
CARD_LINE = re.compile(r"(?m)^\s*([FKSGRB]\d+)\.\s*(.+)$")


class PreflightStop(Exception):
    """Raised at the first unscripted model call."""
def phase_record(*, phase, messages, max_tokens, response=None,
                 usage=None) -> dict:
    prompt = "\n\n".join(
        str(message.get("content", "")) for message in messages)
    counts = {
        name: len(pattern.findall(prompt))
        for name, pattern in CARD_GRAMMARS.items()}
    counts = {name: count for name, count in counts.items() if count}
    card_lines = [
        [match.group(1), match.group(2).strip()[:200]]
        for match in CARD_LINE.finditer(prompt)]
    record = {
        "phase": str(phase),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_chars": len(prompt),
        "estimated_tokens": len(prompt) // 4,
        "max_output_tokens": int(max_tokens),
        "card_counts": counts,
        "cards": card_lines,
        "cards_sent": len(card_lines),
        "required_item_visibility": "grader-only",
    }
    if counts.get("symbol"):
        record["card_limit"] = SYMBOL_MENU_CAP
        record["at_cap"] = counts["symbol"] >= SYMBOL_MENU_CAP
        # The pre-cap candidate total is internal state the prompt cannot
        # reveal; unknowable is recorded as null, never invented.
        record["cards_total_before_cap"] = None
        record["overflow"] = None
    if response is not None:
        record["scripted_reply"] = response
        record["response_chars"] = len(response)
        record["scripted"] = True
        valid = {label for label, _text in card_lines}
        tokens = re.findall(r"\b[FKSGRB]\d+\b", response)
        parsed = [token for token in dict.fromkeys(tokens)
                  if token in valid]
        unknown = [token for token in dict.fromkeys(tokens)
                   if token not in valid]
        record["parsed_selected_ids"] = parsed
        record["unknown_ids"] = unknown
        record["mapping_failed"] = bool(tokens) and not parsed
    if usage is not None:
        record["recorded_usage"] = dict(usage)
        output_tokens = int(usage.get("output_tokens") or 0)
        recorded_cap = int(usage.get("max_tokens") or max_tokens)
        record["provider_status"] = "ok"
        record["truncated"] = output_tokens >= recorded_cap
    return record


class ScriptedChat:
    """A chat_complete replacement fed by recorded replies per phase."""

    def __init__(self, replies_by_phase=None):
        self._replies = {
            phase: list(responses)
            for phase, responses in (replies_by_phase or {}).items()}
        self.records: list = []

    async def __call__(self, messages, *, model=None, max_tokens=2048,
                       timeout=60.0, phase="completion", usage_sink=None):
        available = self._replies.get(phase)
        if not available:
            self.records.append(phase_record(
                phase=phase, messages=messages, max_tokens=max_tokens))
            raise PreflightStop(
                f"first unscripted call at phase {phase!r}; "
                "prompt captured, run stopped")
        if isinstance(available[0], dict):
            entry = available.pop(0)
            response = entry["response"]
            self.records.append(phase_record(
                phase=phase, messages=messages, max_tokens=max_tokens,
                response=response, usage=entry.get("usage")))
            if usage_sink is not None and entry.get("usage"):
                usage_sink.append({
                    "phase": phase, "model": "recorded",
                    **entry["usage"]})
            return response
        response = available.pop(0)
        self.records.append(phase_record(
            phase=phase, messages=messages, max_tokens=max_tokens,
            response=response))
        if usage_sink is not None:
            usage_sink.append({
                "phase": phase, "model": "scripted",
                "max_tokens": max_tokens,
                "input_tokens": len(
                    "\n\n".join(m.get("content", "")
                                for m in messages)) // 4,
                "output_tokens": len(response) // 4})
        return response
def replies_from_trace(trace_path) -> dict:
    """Recorded responses with their usage, grouped by phase in order."""
    payload = json.loads(gzip.decompress(Path(trace_path).read_bytes()))
    replies: dict = {}
    for call in payload.get("llm_completions", ()):
        replies.setdefault(str(call["phase"]), []).append({
            "response": str(call.get("response") or ""),
            "usage": {**(call.get("usage") or {}),
                      "max_tokens": call.get("max_tokens")},
        })
    return replies
