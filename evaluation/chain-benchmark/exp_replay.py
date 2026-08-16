#!/usr/bin/env python3
"""Recorded semantic replay: saved live replies through CURRENT code.

Each saved trace's model replies are injected phase by phase into the
real ``ask`` path — current retrieval, current menus, current guarded
selection, current finalization — through the scripted provider. Old
labels resolve only against the exact cards the current menus render; a
reply that resolves to nothing is flagged ``mapping_failed`` and the
pipeline's own fail-open semantics take over, never an invented
replacement. No network provider can be reached: ``llm.chat_complete``
is replaced before any provider exists, and the question embedding comes
from the frozen cache.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from exp_preflight import ScriptedChat, replies_from_trace
from exp_seal import seal


class FrozenQuestionEmbedder:
    """Serves the one cached vector registered for the current question;
    an unexpected embed request must fail loudly, never bill quietly."""

    def __init__(self, holder):
        self._holder = holder

    async def embed(self, text):
        return self._holder["__current__"]


def replay_question(service, trace_path, *, source) -> dict:
    payload = json.loads(gzip.decompress(Path(trace_path).read_bytes()))
    question_id = int(payload["id"])
    question = str(payload["question"])
    scripted = ScriptedChat(replies_from_trace(trace_path))

    import llm
    original_complete = llm.chat_complete
    original_key_check = llm.has_provider_key
    try:
        llm.chat_complete = scripted
        llm.has_provider_key = lambda model=None: True
        response = asyncio.new_event_loop().run_until_complete(
            service.ask(question, source=source, role="developer"))
    finally:
        llm.chat_complete = original_complete
        llm.has_provider_key = original_key_check

    return {
        "id": question_id,
        "question": question,
        "source": source,
        "mode": "recorded-replay",
        "phases": scripted.records,
        "answer": response.answer,
        "selected_route_ids": list(response.selected_route_ids),
        "selected_symbols": list(response.selected_symbols),
        "selected_body_symbols": list(response.selected_body_symbols),
        "hydrated_symbols": list(response.hydrated_symbols),
        "route_candidates": {
            label: list(route)
            for label, route in
            (response.route_candidates or {}).items()},
        "route_candidate_occurrences": {
            label: [list(key) for key in occurrences]
            for label, occurrences in
            (response.route_candidate_occurrences or {}).items()},
        "formulation_complete": bool(response.formulation_complete),
        "confidence": str(response.confidence),
        "original_trace": {
            "file": str(trace_path),
            "sha256": __import__("hashlib").sha256(
                Path(trace_path).read_bytes()).hexdigest(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--questions", required=True,
                        help="questions-only file (for provenance hashing)")
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--only", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import numpy as np

    from ariadne_mcp.service import AriadneService
    from shadow_eval import build_provenance

    vectors = np.load(args.embedding_cache, allow_pickle=False)
    only = {int(value) for value in args.only.split(",") if value}
    service = AriadneService.get()
    original_embedder = service._embedding_service

    rows = []
    try:
        for trace_path in sorted(Path(args.trace_dir).glob("q*.json.gz")):
            question_id = int(trace_path.stem.split(".")[0][1:])
            if only and question_id not in only:
                continue
            key = f"q{question_id}"
            if key not in vectors:
                print(f"q{question_id}: no cached embedding; skipped")
                continue
            holder = {"__current__": vectors[key]}
            service._embedding_service = FrozenQuestionEmbedder(holder)
            started = time.perf_counter()
            rows.append(replay_question(
                service, trace_path, source=args.source))
            print(f"q{question_id}: replayed in "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
    finally:
        service._embedding_service = original_embedder

    payload = seal({
        "schema": "ariadne-recorded-replay-v1",
        "mode": "recorded-replay",
        "provenance": build_provenance(args),
        "questions": rows})
    Path(args.out).write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {args.out} ({len(rows)} questions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
