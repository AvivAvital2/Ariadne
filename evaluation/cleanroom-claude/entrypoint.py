#!/usr/bin/env python3
"""Fresh Claude Code instances answer the questions, read-only, over /corpus.

Each question is its OWN `claude -p` — an independent, cold session — on purpose:
a shared session would let one question's reasoning/answer contaminate another,
which would invalidate the head-to-head. There is no local engine to "warm"
(the bare LLM is just the remote API), so the only lever on wall-clock is
CONCURRENCY: the independent sessions are fanned out so their boots + work
overlap instead of running one-at-a-time.

Exposure is WHITELIST, not blacklist: tools are restricted to an explicit allow
set (Read/Grep/Glob) — nothing runs that isn't listed, so there is no shell,
network, write, or MCP path. The model receives ONLY the question text on the
CLI and whatever it reads from the clean /corpus checkout.

Answers must be GROUNDED. Spark and Delta are among the most widely read
open-source codebases in existence, so a strong model can name their internals
from memory — which would make this a memory test wearing a retrieval test's
clothes. The system prompt therefore requires every specific to come from a file
opened in that session and cited by path, and the run records each Read/Grep/Glob
so a citation can be checked against what was actually opened. Both halves are
needed: the requirement alone just teaches the model to invent plausible paths.
"""
import concurrent.futures as cf
import json
import os
import hashlib
from pathlib import Path
import subprocess
import sys
SYSTEM = (
    "You are a senior data engineer answering questions about Apache Spark and "
    "Delta Lake. A read-only checkout (spark, delta, databricks-sdk-py) is at "
    "/corpus.\n\n"

    "HOW YOU WILL BE SCORED -- read this before answering.\n\n"

    "Two things are measured, separately.\n"
    "(1) CORRECTNESS: does your answer reach the actual mechanism.\n"
    "(2) COMPLETENESS: whether your answer is admitted for scoring at all. This "
    "is decided MECHANICALLY, not by judgement. Every quote is checked against "
    "the real file: the path must exist, the line number must be exact, and the "
    "code must match the file verbatim.\n\n"

    "An answer that is CORRECT but INCOMPLETE scores nothing. That is "
    "deliberate. A correct answer with no traced chain cannot be told apart from "
    "one recalled from memory and confirmed with a single lookup.\n\n"

    "COMPLETENESS -- you must show HOW the first mechanism reaches the last. The "
    "chain has a fixed number of hops, set in advance from the answer key, NOT "
    "from how you choose to structure your reply; shortening your explanation "
    "does not shorten the chain you owe evidence for. Either route is "
    "accepted:\n"
    "  ROUTE A (strongest) -- quote the code at EVERY hop.\n"
    "  ROUTE B -- quote the FIRST and the LAST mechanism, and name every "
    "mechanism in between, in order, so a chain A->D->C->B is stated rather "
    "than jumped.\n"
    "Both routes need at least TWO verified quotes: one quote plus recollection "
    "is never complete. Where the chain crosses repositories, your first and "
    "last quotes must come from DIFFERENT repositories.\n\n"

    "EVERY QUOTE carries the full /corpus path, the EXACT starting line number, "
    "and the code verbatim: no reformatting, no elision, no paraphrase, no "
    "invented line numbers. A quote that does not sit at the line you cite does "
    "not count, however accurate its content. The line number is the part that "
    "cannot be recalled -- it is what separates reading from remembering, and "
    "this corpus is pinned to one specific revision, so numbers carried over "
    "from another version will not match. Any of these forms is accepted, so "
    "formatting costs you nothing:\n"
    "    /corpus/<path>:<line> on the line above the code block,\n"
    "    or as the first line inside the block,\n"
    "    or every line inside the block prefixed with its own number.\n\n"

    "- A hop is the code that ESTABLISHES a step -- the call site where A "
    "invokes B, the assignment where B sets what C reads, the branch that "
    "selects the path. Naming a class is not a hop. Quoting only the conclusion "
    "is not a chain.\n"
    "- Where the chain crosses repositories, name the artifact that joins them: "
    "a shared table property, config key or protocol constant that provably "
    "appears in both files. There is no call path between the SDK and the other "
    "two, so a shared literal is the link -- show it on both sides rather than "
    "hunting for a call edge. Not every question crosses; do not invent a "
    "crossing that the corpus does not support.\n\n"

    "DO NOT answer from memory or prior familiarity with these projects, however "
    "confident you are. Where the corpus does not show a link, say so plainly and "
    "quote what you did check: a chain with a DECLARED GAP scores ABOVE one "
    "silently completed from prior knowledge."
)
# WHITELIST: the ONLY tools permitted. In headless -p, anything not on this list
# is denied (no prompt, no run) — no allow-all escape hatch. --disallowedTools is
# kept as a redundant belt; the allow set is the model.
ALLOWED = "Read,Grep,Glob"
DISALLOWED = "Bash,Write,Edit,MultiEdit,NotebookEdit,WebFetch,WebSearch,Task"
def _sha(path: str) -> str | None:
    """Hash a file the model opened, from inside the container.

    Computed by the harness, never asked of the model: it has no Bash, so any
    hash it produced would be copied or invented. This gives the verifier proof
    it is comparing byte-identical content rather than a local checkout matched
    only by version string.
    """
    p = path if path.startswith("/") else f"/corpus/{path}"
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _one_session(question: str) -> dict:
    """One cold session. Returns the answer, the tool calls, and file hashes."""
    p = subprocess.run(
        ["claude", "-p", question,
         "--append-system-prompt", SYSTEM,
         "--model", "claude-opus-4-8",
         "--allowedTools", ALLOWED,
         "--disallowedTools", DISALLOWED,
         "--output-format", "stream-json", "--verbose"],
        cwd="/corpus", capture_output=True, text=True, timeout=900)

    answer, calls = "", []
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result" and ev.get("result"):
            answer = ev["result"]
        content = (ev.get("message") or {}).get("content")
        for blk in content if isinstance(content, list) else []:
            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                ti = blk.get("input") or {}
                calls.append({
                    "tool": blk.get("name"),
                    "path": ti.get("file_path") or ti.get("path") or "",
                    "pattern": ti.get("pattern") or "",
                })
    files = sorted({c["path"] for c in calls if c["path"]})
    hashes = {f: _sha(f) for f in files}
    return {
        "answer": (answer or "").strip() or (
            f"(no output; rc={p.returncode}; err={p.stderr.strip()[:300]})"),
        "files_read": files,
        "file_hashes": {k: v for k, v in hashes.items() if v},
        "tool_calls": calls,
    }



_MAX_ATTEMPTS = 2


def _inadequate(result: dict) -> str | None:
    """Why this answer cannot be scored at all, or None if it can.

    Both checks are mechanical and free -- no verification, no judgement. They
    catch a session that never engaged with the corpus, which is a protocol
    failure rather than a wrong answer, and is worth one retry instead of a
    zero. Measured on the previous run: no session read zero files, but five
    read files and quoted nothing, so the second check is the one that bites.
    """
    if not result.get("files_read"):
        return "you did not open a single file in /corpus"
    if "```" not in (result.get("answer") or ""):
        return "you quoted no code at all"
    return None


def ask(question: str) -> dict:
    """Answer one question, retrying once if the session never used the corpus.

    Each attempt is its own cold `claude -p`, so the rejection has to travel in
    the prompt. The retry count is recorded: an arm that needs retries is
    telling us something, and hiding that would flatter it.
    """
    prompt, attempts, rejections = question, 0, []
    while True:
        attempts += 1
        result = _one_session(prompt)
        why = _inadequate(result)
        if why is None or attempts >= _MAX_ATTEMPTS:
            result["attempts"] = attempts
            result["rejections"] = rejections
            if why is not None:
                result["still_inadequate"] = why
            return result
        rejections.append(why)
        prompt = (
            f"{question}\n\n"
            f"[Your previous answer was REJECTED before scoring: {why}. "
            f"An answer that does not trace the chain through code in /corpus "
            f"cannot be scored at all. Read the relevant files and quote them "
            f"with exact line numbers.]"
        )


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY must be set")
    items = json.load(open("/work/questions.json"))
    conc = max(1, int(os.environ.get("CONCURRENCY", "6")))
    order = [it["id"] for it in items]
    done: dict = {}
    print(f"answering {len(items)} questions, {conc}-wide (independent sessions)",
          file=sys.stderr, flush=True)
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(ask, it["question"]): it for it in items}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            it = futs[fut]
            done[it["id"]] = {"id": it["id"], "question": it["question"], **fut.result()}
            print(f"[{n}/{len(items)}] Q{it['id']} done "
                  f"({len(done[it['id']]['files_read'])} files, "
                  f"{len(done[it['id']]['tool_calls'])} tool calls"
                  f"{', RETRIED' if done[it['id']].get('attempts', 1) > 1 else ''})",
                  file=sys.stderr, flush=True)
            json.dump([done[i] for i in order if i in done],
                      open("/work/answers.json", "w"), indent=2)
    print(f"wrote {len(done)} answers -> /work/answers.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
