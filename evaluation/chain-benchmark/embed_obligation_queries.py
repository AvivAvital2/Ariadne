"""Prepare obligation-query vectors in ONE batched embedding request.

The obligation-query retrieval experiment searches existing catalog and
clew vectors independently per obligation. That needs one embedding per
(question, obligation-text) pair — texts the cache does not hold. This
command collects every recorded obligation from the canary and replay
traces and embeds them in a single ``embed_batch`` call.

THE USER RUNS THIS (it spends): requires OPENAI_API_KEY. Roughly 30
short texts on text-embedding-3-large — well under a cent. Refuses to
overwrite an existing output so a cache is never respent by accident.

    .venv/bin/python evaluation/chain-benchmark/embed_obligation_queries.py \
        --out evaluation/chain-benchmark/obligation-query-vectors.npz
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

QUESTIONS = (15, 23, 62, 107, 67, 147)


def collect_texts(trace_dir: Path) -> dict:
    """{key: text} — the question itself plus one composite per
    obligation (question + requirement), keyed q<id>:question and
    q<id>:O<n>."""
    texts: dict = {}
    for question_id in QUESTIONS:
        trace = json.loads(gzip.decompress(
            (trace_dir / f"q{question_id}.json.gz").read_bytes()))
        question = str(trace["question"])
        plan = next(
            completion["response"]
            for completion in trace["llm_completions"]
            if completion["phase"] == "scip-obligation-plan")
        texts[f"q{question_id}:question"] = question
        for number, text in re.findall(
                r"(?m)^\s*C(\d{1,2})\s*:\s*(.+)$", plan)[:5]:
            texts[f"q{question_id}:O{int(number)}"] = (
                f"{question}\n{text.strip()}")
    return texts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=str(
        HERE / "live22-diagnostic-answers-traces"))
    parser.add_argument("--out", default=str(
        HERE / "obligation-query-vectors.npz"))
    parser.add_argument("--dry-run", action="store_true", help=(
        "list the texts and the single batch size; no spend"))
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(
            f"{out} already exists — refusing to respend; delete it "
            "deliberately to re-embed")
    texts = collect_texts(Path(args.trace_dir))
    print(f"one batch of {len(texts)} texts "
          f"({sum(len(t) for t in texts.values())} chars)")
    if args.dry_run:
        for key in texts:
            print(" ", key)
        return 0

    import numpy as np

    from embedding import EmbeddingConfig, EmbeddingService

    async def embed_all():
        service = EmbeddingService(EmbeddingConfig())
        try:
            keys = list(texts)
            vectors = await service.embed_batch(
                [texts[key] for key in keys])
            return dict(zip(keys, vectors))
        finally:
            await service.close()

    vectors = asyncio.run(embed_all())
    np.savez(out, **{key: np.asarray(vector, dtype=np.float32)
                     for key, vector in vectors.items()})
    print(f"wrote {len(vectors)} vectors -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
