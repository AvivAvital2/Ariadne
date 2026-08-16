#!/usr/bin/env python3
"""Bare knowledge-only arm, model-selectable — the scaling ladder against Ariadne.

Ariadne's 12 best-case answers (answers-ariadne.jsonl, retrieval-misses excluded)
are the reference bar. This runs a bare LLM — NO tools, NO corpus, NO MCP — over
the same 12 questions, at whatever model tier you pass, so we can see how bare
LLMs scale toward that bar as capability increases. The version-pinned facts are
the interesting axis: a bigger model still can't know dbr17.3's exact component
map or the managed→coordinated rename, no matter how capable.

"No leak" is structural: a Messages request with no ``tools`` field cannot
retrieve, read files, or reach any MCP server. Each question is an independent
request (no shared conversation), so answers can't contaminate each other.

max_tokens matches Ariadne's synthesis (1024); only the MODEL varies across runs.
Run one tier at a time in a shell that has ANTHROPIC_API_KEY::

    python evaluation/spool-clean-room/bare_knowledge.py --model claude-haiku-4-5
    python evaluation/spool-clean-room/bare_knowledge.py --model claude-sonnet-5
    python evaluation/spool-clean-room/bare_knowledge.py --model claude-opus-4-8
    python evaluation/spool-clean-room/bare_knowledge.py --model claude-fable-5

Each writes answers-bare-<model>.jsonl. Resumable: answered ids are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent

MAX_TOKENS = 1024   # matched to AriadneService.ask() synthesis
API = "https://api.anthropic.com/v1/messages"

# Mirrors the Ariadne synthesis persona (senior engineer, name the real
# classes/methods, admit uncertainty) MINUS any retrieved docs — that context is
# the thing under test, so the bare prompt must not smuggle it back in.
SYSTEM = (
    "You are a senior data engineer answering questions about Apache Spark and "
    "Delta Lake. Answer from your own expertise. Name the actual classes, "
    "methods, configs, and mechanisms involved, and be specific about versions "
    "where it matters. If you are unsure of a specific, say so plainly rather "
    "than inventing it."
)


def already_done(out_path: Path) -> set[int]:
    if not out_path.exists():
        return set()
    done: set[int] = set()
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("answer") and not rec.get("error"):
            done.add(rec["id"])
    return done


async def ask_one(client: httpx.AsyncClient, key: str, model: str, q: dict,
                  attempts: int = 3) -> dict:
    """One knowledge-only Messages call; a failure is recorded, never raised."""
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            r = await client.post(
                API,
                headers={"x-api-key": key,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": model, "max_tokens": MAX_TOKENS, "system": SYSTEM,
                      "messages": [{"role": "user", "content": q["question"]}]},
                timeout=180.0)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            text = "".join(
                b.get("text", "") for b in data.get("content", [])
                if b.get("type") == "text").strip()
            if not text:
                raise RuntimeError(f"empty content: {str(data)[:200]}")
            return {"id": q["id"], "model": model,
                    "question": q["question"], "answer": text}
        except Exception as e:  # noqa: BLE001 - record + continue by design
            last_err = f"{type(e).__name__}: {e}"
            if attempt < attempts:
                await asyncio.sleep(2.0 * attempt)
    return {"id": q["id"], "model": model,
            "question": q["question"], "error": last_err}


async def main() -> None:
    ap = argparse.ArgumentParser(
        description="Knowledge-only bare arm at a chosen model tier.")
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="Claude model id, e.g. claude-haiku-4-5 / claude-sonnet-5 "
                         "/ claude-opus-4-8 / claude-fable-5")
    ap.add_argument("--questions", default=str(HERE / "questions_ask_12.json"),
                    help="stripped {id, question} file (the 12-question best-case set)")
    ap.add_argument("--out", default=None,
                    help="default: answers-bare-<model>.jsonl")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(
            "ANTHROPIC_API_KEY must be set — the bare arm calls the Messages API "
            "directly. The sandbox has no key; run this in your own shell.")

    out_path = Path(args.out) if args.out else (
        HERE / f"answers-bare-{args.model.replace('claude-', '')}.jsonl")
    questions = json.loads(Path(args.questions).read_text())
    done = already_done(out_path)
    todo = [q for q in questions if q["id"] not in done]
    if done:
        print(f"resuming: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("nothing to do — all answered.", flush=True)
        return

    write_lock = asyncio.Lock()

    async def record(rec: dict) -> None:
        async with write_lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        tag = ("ERR " + rec["error"]) if rec.get("error") \
            else f"{len(rec['answer'])} chars"
        print(f"  [{rec['id']:>2}] {tag}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    print(f"bare arm @ {args.model}: {len(todo)} questions, max_tokens={MAX_TOKENS}, "
          f"NO tools/corpus/MCP, {args.concurrency}-wide → {out_path.name}",
          flush=True)
    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        async def bounded(q: dict) -> None:
            async with sem:
                await record(await ask_one(client, key, args.model, q))
        await asyncio.gather(*(bounded(q) for q in todo))
    print(f"done in {time.monotonic() - t0:.1f}s → {out_path.name}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
