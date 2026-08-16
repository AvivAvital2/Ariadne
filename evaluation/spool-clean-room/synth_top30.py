#!/usr/bin/env python3
"""Synthesize ariadne_ask answers for the top-30 cross-repo questions.

Launches ONE Ariadne engine (the ``AriadneService`` singleton), **warms it
with a single serial ask**, then fans the rest out concurrently against that
same warm instance.

Why this shape (see the mmap discussion): a fresh Python process is what's
expensive, not the matrix RAM. A read-only ``mmap`` (``mmap_mode='r'``) is
already shared by the kernel page cache across every reader, so the 3.5 GB is
never duplicated. What does NOT auto-share is per-process Python state — the
``EmbeddingMatrix`` object, the httpx pools (embedding + chat), the SQLite/WAL
connection, the query cache. The one serial warmup ask pays every one of those
cold costs exactly once (config load, WAL open, matrix mmap + cached object,
embedding pool, chat client, first page-fault of the matrix rows it touches).
After that, synthesis latency (~30-60 s of chat-completion HTTP per ask) is the
only wall, and it's I/O-bound — so a single-process ``asyncio`` fan-out under a
semaphore overlaps those waits without rebuilding anything. Multi-process would
re-pay all the per-process state and buy nothing here.

Run from the Ariadne repo root, in a shell that has the generation API key
(the MCP server has it; the sandbox does not)::

    .venv/bin/python evaluation/spool-clean-room/synth_top30.py

Resumable: ids already present (non-error) in the output file are skipped, so a
re-run continues rather than re-synthesizing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# The three no-LLM fallbacks in AriadneService.ask() (ask_synthesis off, no
# OPENAI_API_KEY, or synthesis raised) all return the raw retrieved context
# headed by "Based on {n} docs...:" — never real prose. Detect it so we never
# record or grade a dump as if it were a synthesized answer.
_DUMP_RE = re.compile(r"Based on \d+ docs")


def looks_unsynthesized(answer: str) -> bool:
    """True iff ``answer`` is a raw-context fallback rather than LLM prose."""
    return bool(_DUMP_RE.search(answer[:200]))


def load_questions(questions_path: Path, ids_path: Path) -> list[dict]:
    """The top-N question dicts, in the id order given by ``ids_path``."""
    by_id = {q["id"]: q for q in json.loads(questions_path.read_text())}
    ids = json.loads(ids_path.read_text())
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"ids not in {questions_path.name}: {missing}")
    return [by_id[i] for i in ids]


def already_done(out_path: Path) -> set[int]:
    """Ids with a non-error record already in the output file (for resume)."""
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
        if "answer" in rec and not rec.get("error") \
                and not looks_unsynthesized(rec["answer"]):
            done.add(rec["id"])
    return done


async def ask_one(svc, q: dict, source: str, attempts: int = 2) -> dict:
    """One ``svc.ask`` with a light retry; a failure is recorded, never raised,
    so one bad ask can't kill the batch."""
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            resp = await svc.ask(q["question"], source=source)
            return {
                "id": q["id"],
                "family": q.get("family"),
                "question": q["question"],
                "answer": resp.answer,
                "sources": list(resp.sources),
                "confidence": resp.confidence,
                "event_id": resp.event_id,
            }
        except Exception as e:  # noqa: BLE001 - record + continue by design
            last_err = f"{type(e).__name__}: {e}"
            if attempt < attempts:
                await asyncio.sleep(2.0 * attempt)
    return {
        "id": q["id"],
        "family": q.get("family"),
        "question": q["question"],
        "error": last_err,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(
        description="Warm one Ariadne engine, then fan out the top-30 asks.")
    ap.add_argument(
        "--source", default="databricks",
        help="ask source arg — scopes retrieval; the spool auto-joins. Must be "
             "set explicitly when the configured default source is unavailable.")
    ap.add_argument("--concurrency", type=int, default=12,
                    help="max concurrent asks after warmup (default 12)")
    ap.add_argument("--ids", default=str(HERE / "top30_to_synth.json"))
    ap.add_argument("--questions", default=str(HERE / "questions_200_xrepo.json"))
    ap.add_argument("--out", default=None,
                    help="output JSONL (default synth_top30.jsonl; "
                         "synth_top30.fresh.jsonl under --fresh)")
    ap.add_argument("--fresh", action="store_true",
                    help="clear the retrieval query_cache first so every ask "
                         "recomputes retrieval from scratch — diagnoses whether "
                         "the refusals are stale-cache or genuine mis-ranking")
    args = ap.parse_args()

    # Import Ariadne from the repo root: it exposes top-level modules (config,
    # library, ariadne_mcp) and AriadneService assumes cwd == install dir, and
    # the 'databricks' source path is yaml-relative — so run from root.
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)

    # Load the project .env the same way the server (server.py) and CLI
    # (cli/main.py) do — otherwise a bare shell has no OPENAI_API_KEY and
    # ariadne_ask SILENTLY returns raw retrieved context instead of a
    # synthesized answer. Fail loud rather than produce un-gradeable dumps.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set (even after loading .env). ariadne_ask "
            "gates LLM synthesis on it, so every answer would come back as raw "
            "retrieved context — un-gradeable. Set it / put it in .env, re-run.")

    from ariadne_mcp.service import AriadneService

    questions = load_questions(Path(args.questions), Path(args.ids))
    out_path = Path(args.out) if args.out else (HERE / (
        "synth_top30.fresh.jsonl" if args.fresh else "synth_top30.jsonl"))
    done = already_done(out_path)
    todo = [q for q in questions if q["id"] not in done]
    if done:
        print(f"resuming: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        print("nothing to do — all top-30 already synthesized.", flush=True)
        return

    svc = AriadneService.get()
    if args.fresh:
        # DELETE FROM query_cache (all rows) via the lazy-init'd Library — so
        # search() misses the persistent cache and recomputes retrieval fresh.
        # Note: this clears the shared cache the live MCP server uses too; it's
        # benign (entries just recompute on next use).
        svc.library.cache_clear()
        svc._query_cache.clear()
        print("fresh mode: retrieval query_cache cleared — every ask recomputes "
              "retrieval (fresh embedding + ranking), not cache-served.",
              flush=True)
    write_lock = asyncio.Lock()

    async def record(rec: dict) -> None:
        async with write_lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        tag = ("ERR " + rec["error"]) if rec.get("error") else (
            f"{len(rec['sources'])} sources, ev{rec['event_id']}, "
            f"{rec['confidence']}")
        print(f"  [{rec['id']:>3}] {tag}", flush=True)

    # --- WARM ONCE: one serial ask pays every cold cost before any fan-out ---
    warm_q = todo[0]
    print(f"launching Ariadne + warming on q{warm_q['id']} "
          f"(config, WAL, 3.5GB matrix mmap, embed pool, chat client)…",
          flush=True)
    t0 = time.monotonic()
    warm_rec = await ask_one(svc, warm_q, args.source)
    if not warm_rec.get("error") and looks_unsynthesized(warm_rec["answer"]):
        raise SystemExit(
            f"Warmup answer for q{warm_q['id']} came back as raw retrieved "
            "context, not a synthesized answer — aborting before fan-out. "
            "Check OPENAI_API_KEY and the configured provider's key.")
    await record(warm_rec)
    print(f"warm in {time.monotonic() - t0:.1f}s — fanning out "
          f"{len(todo) - 1} at concurrency {args.concurrency}", flush=True)

    # --- FAN OUT: the rest, bounded, against the one warm instance ---
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(q: dict) -> None:
        async with sem:
            await record(await ask_one(svc, q, args.source))

    t1 = time.monotonic()
    await asyncio.gather(*(bounded(q) for q in todo[1:]))
    total = time.monotonic() - t0
    print(f"fan-out done in {time.monotonic() - t1:.1f}s "
          f"({len(todo)} asks in {total:.1f}s) → {out_path.name}", flush=True)

    await svc.close()


if __name__ == "__main__":
    asyncio.run(main())
