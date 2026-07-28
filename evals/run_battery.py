"""Retrieval battery: does the environment speak exactly at the seam,
and is the answer's ground-truth doc actually in the context?

Verdicts come from the PRODUCTION participation criterion
(`library.lens_retrieval.breadth_speaks`) over the PRODUCTION candidate
pool (`doc_grade_spool_candidates`) — this file measures the pipeline, it
does not reimplement it. Three numbers per run:

- participation: speak/silent verdict vs the row's `want` (the seam
  contract — including the rows where the environment MUST stay quiet);
- ground-truth-in-context: for speak rows, does a doc matching the row's
  `ground_truth` title criteria rank inside the admitted window? When it
  doesn't, the LLM answers from training data — which cannot know your
  code, your pinned versions, or the seam between them;
- junk: environment docs admitted on a must-stay-silent row.

Run (after `build_eval_store.sh`):

    uv run python evals/run_battery.py

First run embeds the battery questions (needs an OpenAI-compatible
embedding endpoint — OPENAI_API_KEY / OPENAI_BASE_URL) and caches the
vectors in evals/query_vectors.npz; every later run is offline and free.
Exit code 0 iff every participation verdict matches its `want`.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml
from attrs import frozen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library.lens_retrieval import breadth_speaks, lens_share  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
BREADTH_WINDOW = 50
# Docs the assembly would actually take from the window (production share).
CONTEXT_K = lens_share(BREADTH_WINDOW // 2)


@frozen
class Matrix:
    """Minimal embedding-matrix view: doc id -> row, rows @ query = cosine
    (vectors are L2-normalized, as in the production store)."""
    id_to_row: dict
    M: object


@frozen
class Store:
    conn: object
    matrix: Matrix
    spool_doc_ids: list


@frozen
class RowResult:
    label: str
    archetype: str
    want: str
    verdict: str
    correct: bool
    top_spool: float
    top_repo: float
    breadth: int
    gt_in_context: bool | None   # None when not a speak-with-GT row
    gt_rank: int | None
    junk: int


@frozen
class Report:
    results: list
    correct: int
    total: int
    gt_hits: int
    gt_total: int
    junk_total: int

    @property
    def exit_code(self) -> int:
        return 0 if self.correct == self.total else 1


def _sims(matrix: Matrix, doc_ids: list, query: np.ndarray) -> np.ndarray:
    rows = [matrix.id_to_row[i] for i in doc_ids if i in matrix.id_to_row]
    if not rows:
        return np.array([], dtype=np.float32)
    return np.asarray(matrix.M, dtype=np.float32)[rows] @ query


def _consumer_doc_ids(store: Store, source: str) -> list:
    return [r[0] for r in store.conn.execute(
        'SELECT id FROM documents WHERE source_name = ?', (source,))]


def _title(store: Store, doc_id: str) -> str:
    row = store.conn.execute(
        'SELECT title FROM documents WHERE id = ?', (doc_id,)).fetchone()
    return (row[0] or '') if row else ''


def score_row(row: dict, store: Store, query: np.ndarray) -> RowResult:
    """One battery row through the production participation criterion."""
    spool_ids = [i for i in store.spool_doc_ids if i in store.matrix.id_to_row]
    spool_sims = _sims(store.matrix, spool_ids, query)
    order = np.argsort(-spool_sims)
    ranked_ids = [spool_ids[i] for i in order]
    ranked_sims = spool_sims[order]

    repo_sims = _sims(
        store.matrix, _consumer_doc_ids(store, row['consumer']), query)
    top_repo = float(repo_sims.max()) if len(repo_sims) else 0.0

    speaks = breadth_speaks(
        ranked_sims.tolist(), top_repo, window=BREADTH_WINDOW)
    verdict = 'speak' if speaks else 'silent'
    breadth = int((ranked_sims[:BREADTH_WINDOW] > top_repo).sum())

    gt_in_context: bool | None = None
    gt_rank: int | None = None
    needles = [n.lower() for n in
               (row.get('ground_truth') or {}).get('title_contains', [])]
    if row['want'] == 'speak' and needles:
        gt_in_context = False
        for rank, doc_id in enumerate(ranked_ids[:CONTEXT_K], start=1):
            title = _title(store, doc_id).lower()
            if any(n in title for n in needles):
                gt_in_context, gt_rank = True, rank
                break

    junk = breadth if (row['want'] == 'silent' and speaks) else 0
    return RowResult(
        label=row['label'], archetype=row['archetype'], want=row['want'],
        verdict=verdict, correct=verdict == row['want'],
        top_spool=float(ranked_sims[0]) if len(ranked_sims) else 0.0,
        top_repo=top_repo, breadth=breadth,
        gt_in_context=gt_in_context, gt_rank=gt_rank, junk=junk,
    )


def run(rows: list, store: Store, vector_for) -> Report:
    """Score every row; `vector_for(label)` supplies the query vector."""
    results = [
        score_row(row, store, np.asarray(vector_for(row['label']),
                                         dtype=np.float32))
        for row in rows
    ]
    with_gt = [r for r in results if r.gt_in_context is not None]
    return Report(
        results=results,
        correct=sum(r.correct for r in results),
        total=len(results),
        gt_hits=sum(bool(r.gt_in_context) for r in with_gt),
        gt_total=len(with_gt),
        junk_total=sum(r.junk for r in results),
    )


def print_report(report: Report) -> None:
    print(f'{"label":26s} {"archetype":11s} {"want":7s} {"topS":6s} '
          f'{"topR":6s} {"breadth":8s} {"GT":5s} verdict')
    for r in report.results:
        gt = ('-' if r.gt_in_context is None
              else f'#{r.gt_rank}' if r.gt_in_context else 'MISS')
        mark = 'OK' if r.correct else 'MISS'
        print(f'{r.label:26s} {r.archetype:11s} {r.want:7s} '
              f'{r.top_spool:.3f}  {r.top_repo:.3f}  '
              f'{r.breadth:3d}/{BREADTH_WINDOW}   {gt:5s} '
              f'{r.verdict} {mark}')
    print(f'\nparticipation: {report.correct}/{report.total} correct   '
          f'ground-truth-in-context: {report.gt_hits}/{report.gt_total}   '
          f'junk admissions: {report.junk_total}')


# --- wiring for the real eval store (everything above is dependency-free) --

def _load_real_store(store_dir: Path, env_source: str) -> Store:
    from library import Library
    from library.embedding_matrix import ensure_matrix
    from library.lens_retrieval import doc_grade_spool_candidates

    db = store_dir / 'ariadne.db'
    if not db.exists():
        sys.exit(f'no eval store at {db} — run evals/build_eval_store.sh '
                 f'first (builds it once; see evals/README.md)')
    with Library(db) as lib:
        # Build-or-reuse: absent OR stale (e.g. sources onboarded after the
        # last build) rebuilds from the DB's stored embeddings — offline,
        # no API calls.
        matrix = ensure_matrix(lib, store_dir / '.ariadne')
        pool = doc_grade_spool_candidates(
            lib, [f'spool:{env_source}', env_source])
    if matrix is None:
        sys.exit(f'could not build an embedding matrix from {db} — the '
                 f'store has no document embeddings (did onboarding run?)')
    conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    return Store(
        conn=conn,
        matrix=Matrix(id_to_row=matrix.id_to_row, M=matrix.M),
        spool_doc_ids=list(pool),
    )


def _vector_loader(rows: list, cache_path: Path):
    """Cached-first vectors; embeds (and re-caches) only what's missing."""
    cached = dict(np.load(cache_path)) if cache_path.exists() else {}
    missing = [r for r in rows if r['label'] not in cached]
    if missing:
        import asyncio

        from embedding import EmbeddingService

        async def _embed_all():
            service = EmbeddingService()
            for row in missing:
                cached[row['label']] = np.asarray(
                    await service.embed(row['question']), dtype=np.float32)

        print(f'embedding {len(missing)} uncached question(s)…')
        asyncio.run(_embed_all())
        np.savez(cache_path, **cached)
        print(f'query vectors cached: {cache_path}')
    return lambda label: cached[label]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--battery', default=EVALS_DIR / 'battery.yaml')
    parser.add_argument('--store', default=EVALS_DIR / 'store')
    parser.add_argument('--env-source', default='databricks')
    parser.add_argument('--vectors', default=EVALS_DIR / 'query_vectors.npz')
    args = parser.parse_args()

    rows = yaml.safe_load(Path(args.battery).read_text())['battery']
    store = _load_real_store(Path(args.store), args.env_source)
    report = run(rows, store, _vector_loader(rows, Path(args.vectors)))
    print_report(report)
    return report.exit_code


if __name__ == '__main__':
    sys.exit(main())
