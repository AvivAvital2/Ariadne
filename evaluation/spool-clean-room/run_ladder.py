#!/usr/bin/env python3
"""One-shot LLM-scaling ladder vs Ariadne's best case — run + grade + stats.

For each model rung it (1) runs the knowledge-only bare arm over the 12
best-case questions, (2) grades every answer against the HIDDEN rubric with an
LLM judge, and (3) prints statistics live. At the end it prints the scaling
curve — chain-completeness and version-correctness per rung — against Ariadne's
already-collected best-case answers as the reference bar.

Everything in one process. You run it (needs ANTHROPIC_API_KEY; the sandbox has
none). Fully resumable: per-model answers (answers-bare-<model>.jsonl) and
per-answer grades (grades-<tag>.jsonl) are cached, so a re-run continues and
never re-spends on completed work.

Isolation held: the bare rungs receive ONLY questions_ask_12.json ({id,question}).
The rubric (questions_blended.json) is seen only by the judge, never by a rung.
Grading is blind — the judge prompt never names the model or "Ariadne".
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
API = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
BARE_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS = 400

LADDER = ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8",
          "claude-opus-5-0", "claude-fable-5"]
PRICE = {  # $/1M (input, output) — for the live cost stat only
    "claude-haiku-4-5": (1, 5), "claude-sonnet-5": (2, 10),
    "claude-opus-4-8": (5, 25), "claude-opus-5-0": (5, 25),
    "claude-fable-5": (10, 50),
}
DEFAULT_JUDGE = "claude-opus-4-8"

BARE_SYSTEM = (
    "You are a senior data engineer answering questions about Apache Spark and "
    "Delta Lake. Answer from your own expertise. Name the actual classes, "
    "methods, configs, and mechanisms involved, and be specific about versions "
    "where it matters. If you are unsure of a specific, say so plainly rather "
    "than inventing it."
)
JUDGE_SYSTEM = (
    "You grade technical answers about Apache Spark and Delta Lake ONLY against "
    "the rubric you are given. You do not know which system or model produced "
    "the answer. Reward covering the rubric's specific named mechanisms; do not "
    "reward fluent prose that misses them. Respond with ONLY a JSON object."
)


def short(model: str) -> str:
    return model.replace("claude-", "")


def load_jsonl(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r["id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
    return out


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


async def messages_call(client, key, model, system, user, max_tokens):
    """One Messages call. Returns (text, in_tok, out_tok). Raises on non-200/empty."""
    r = await client.post(
        API,
        headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                 "content-type": "application/json"},
        json={"model": model, "max_tokens": max_tokens, "system": system,
              "messages": [{"role": "user", "content": user}]},
        timeout=180.0)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    d = r.json()
    text = "".join(b.get("text", "") for b in d.get("content", [])
                   if b.get("type") == "text").strip()
    if not text:
        raise RuntimeError(f"empty content: {str(d)[:160]}")
    u = d.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


async def retry(factory, attempts=3):
    last = None
    for i in range(1, attempts + 1):
        try:
            return await factory()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts:
                await asyncio.sleep(2.0 * i)
    raise last


def cost(model, in_tok, out_tok) -> float:
    pi, po = PRICE.get(model, (5, 25))
    return in_tok / 1e6 * pi + out_tok / 1e6 * po


# ---- grading ---------------------------------------------------------------

def judge_user(q: dict, answer: str) -> str:
    facts = q["rubric"]
    lines = [f"QUESTION:\n{q['question']}\n",
             "RUBRIC — count how many of these specific mechanisms the answer "
             "genuinely covers (naming the real classes/configs/behavior, not "
             "vaguely gesturing):"]
    lines += [f"  {i}. {f}" for i, f in enumerate(facts, 1)]
    if q.get("wrong_if"):   # sharp 17.3-vs-newer discriminator
        lines.append(f"\nVERSION CHECK (runtime dbr17.3-lts / spark 4.0.0 / delta 4.0.0). "
                     f"The 17.3-correct fact:\n  {q['version_hinge']}")
        lines.append(f"A WRONG (newer/blurred) answer would say:\n  {q['wrong_if']}")
        lines.append("version_correct = 'yes' if the answer matches the 17.3 fact, "
                     "'no' if it gives the wrong/newer answer or contradicts it, "
                     "'na' if it simply doesn't address it.")
    else:
        lines.append("\nNo version check for this question; version_correct = 'na'.")
    lines.append(f"\nANSWER TO GRADE:\n{answer}\n")
    lines.append(f'Respond with ONLY: {{"facts_hit": <int 0..{len(facts)}>, '
                 '"version_correct": "yes"|"no"|"na", "note": "<=12 words"}')
    return "\n".join(lines)


def parse_judge(text: str, n_facts: int):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    vc = str(o.get("version_correct", "na")).lower()
    return {
        "facts_hit": max(0, min(int(o.get("facts_hit", 0)), n_facts)),
        "facts_total": n_facts,
        "version_correct": vc if vc in ("yes", "no", "na") else "na",
        "note": str(o.get("note", ""))[:120],
    }


# ---- per-arm passes --------------------------------------------------------

async def run_bare(client, key, model, questions, conc):
    """Answer the 12 questions knowledge-only; cache; return {id: rec} + stats."""
    path = HERE / f"answers-bare-{short(model)}.jsonl"
    have = load_jsonl(path)
    todo = [q for q in questions if q["id"] not in have
            or have[q["id"]].get("error")]
    stats = {"in": 0, "out": 0, "errors": 0}
    sem = asyncio.Semaphore(conc)
    t0 = time.monotonic()

    async def one(q):
        async with sem:
            try:
                text, it, ot = await retry(lambda: messages_call(
                    client, key, model, BARE_SYSTEM, q["question"], BARE_MAX_TOKENS))
                rec = {"id": q["id"], "model": model, "answer": text}
                stats["in"] += it; stats["out"] += ot
            except Exception as e:  # noqa: BLE001
                rec = {"id": q["id"], "model": model, "error": f"{type(e).__name__}: {e}"}
                stats["errors"] += 1
            append_jsonl(path, rec)
            have[q["id"]] = rec
            m = "ERR" if rec.get("error") else f"{len(rec['answer'])}c"
            print(f"    answer q{q['id']:>2}: {m}", flush=True)

    if todo:
        await asyncio.gather(*(one(q) for q in todo))
    stats["secs"] = time.monotonic() - t0
    stats["cost"] = cost(model, stats["in"], stats["out"])
    return have, stats


async def grade_arm(client, key, judge_model, tag, answers, rubric, conc):
    """Grade each non-error answer against the rubric; cache; return {id: grade}."""
    path = HERE / f"grades-{tag}.jsonl"
    have = load_jsonl(path)
    todo = [i for i, r in answers.items()
            if i in rubric and r.get("answer") and i not in have]
    jstats = {"in": 0, "out": 0}
    sem = asyncio.Semaphore(conc)

    async def one(i):
        async with sem:
            q = rubric[i]
            try:
                text, it, ot = await retry(lambda: messages_call(
                    client, key, judge_model, JUDGE_SYSTEM,
                    judge_user(q, answers[i]["answer"]), JUDGE_MAX_TOKENS))
                jstats["in"] += it; jstats["out"] += ot
                g = parse_judge(text, len(q["rubric"])) or {
                    "facts_hit": 0, "facts_total": len(q["rubric"]),
                    "version_correct": "na", "note": "unparseable judge reply"}
            except Exception as e:  # noqa: BLE001
                g = {"facts_hit": 0, "facts_total": len(q["rubric"]),
                     "version_correct": "na", "note": f"judge error: {e}"}
            g["id"] = i
            append_jsonl(path, g)
            have[i] = g

    if todo:
        await asyncio.gather(*(one(i) for i in todo))
    jstats["cost"] = cost(judge_model, jstats["in"], jstats["out"])
    return have, jstats


def summarize(grades: dict, rubric: dict) -> dict:
    graded = [g for g in grades.values() if g["id"] in rubric]
    if not graded:
        return {"n": 0}
    chain = sum(g["facts_hit"] for g in graded) / max(1, sum(g["facts_total"] for g in graded))
    vq = [g for g in graded if rubric[g["id"]].get("wrong_if")]
    vyes = sum(1 for g in vq if g["version_correct"] == "yes")
    return {"n": len(graded), "chain_pct": 100 * chain,
            "v_yes": vyes, "v_total": len(vq)}


# ---- orchestration ---------------------------------------------------------

async def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-scaling ladder vs Ariadne's best case.")
    ap.add_argument("--models", nargs="*", default=LADDER)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY must be set — this calls the Messages "
                         "API for both the rungs and the judge. Run it in your shell.")

    questions = json.loads((HERE / "questions_ask_12.json").read_text())
    rubric = {q["id"]: q for q in json.loads((HERE / "questions_blended.json").read_text())
              if q["id"] in {x["id"] for x in questions}}
    ariadne = load_jsonl(HERE / "answers-ariadne.jsonl")

    print(f"ladder: {', '.join(short(m) for m in args.models)}")
    print(f"judge: {short(args.judge_model)}  |  {len(questions)} questions  |  "
          f"version-check questions: {sum(1 for q in rubric.values() if q.get('version_hinge'))}")
    print(f"NOTE: judge == a contestant ({short(args.judge_model)}); grading is blind + "
          "rubric-anchored, but keep the self-grade caveat in mind.\n")

    rows, spend = {}, 0.0
    async with httpx.AsyncClient() as client:
        # reference bar: grade Ariadne's already-collected best-case answers
        print("=== Ariadne (best-case bar) ===", flush=True)
        ag, ajs = await grade_arm(client, key, args.judge_model, "ariadne",
                                  ariadne, rubric, args.concurrency)
        rows["Ariadne (spool)"] = summarize(ag, rubric); spend += ajs["cost"]
        s = rows["Ariadne (spool)"]
        print(f"  chain {s['chain_pct']:.0f}% | version {s['v_yes']}/{s['v_total']} "
              f"| judge ${ajs['cost']:.3f}\n", flush=True)

        for model in args.models:
            print(f"=== {short(model)} ===", flush=True)
            answers, bs = await run_bare(client, key, model, questions, args.concurrency)
            live = [a for a in answers.values() if a.get("answer")]
            print(f"  answers: {len(live)}/{len(questions)} ok, {bs['errors']} err "
                  f"| {bs['secs']:.0f}s | rung ${bs['cost']:.3f}", flush=True)
            spend += bs["cost"]
            if not live:
                rows[short(model)] = {"n": 0, "unavailable": True}
                print("  (no answers — model unavailable? skipping grade)\n", flush=True)
                continue
            grades, js = await grade_arm(client, key, args.judge_model, short(model),
                                         answers, rubric, args.concurrency)
            spend += js["cost"]
            rows[short(model)] = summarize(grades, rubric)
            s = rows[short(model)]
            print(f"  chain {s['chain_pct']:.0f}% | version {s['v_yes']}/{s['v_total']} "
                  f"| judge ${js['cost']:.3f}\n", flush=True)

    # ---- scaling curve ----
    print("=" * 58)
    print(f"{'arm':<22}{'chain%':>8}{'version-correct':>18}")
    print("-" * 58)
    for name, s in rows.items():
        if s.get("unavailable"):
            print(f"{name:<22}{'—':>8}{'unavailable':>18}")
        elif s.get("n"):
            vc = f"{s['v_yes']}/{s['v_total']}" if s["v_total"] else "n/a"
            print(f"{name:<22}{s['chain_pct']:>7.0f}%{vc:>18}")
    print("=" * 58)
    print(f"total spend this run: ${spend:.2f}")
    out = HERE / "ladder_results.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"results → {out.name}")


if __name__ == "__main__":
    asyncio.run(main())
