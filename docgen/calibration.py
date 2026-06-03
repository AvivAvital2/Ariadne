"""Persisted LLM-usage telemetry for self-calibrating cost estimates.

Every LLM call reports exact token usage (input/output, and cache
read/create). We persist those, keyed by ``(phase, doc_type, language,
model)``, so the dry-run estimator can use the *empirical* tokens-per-
call for this codebase + model instead of fixed heuristic constants.
The heuristics remain the cold-start fallback (no data yet).

Storage is a tiny SQLite table; one row per recorded call. Aggregation
(:meth:`mean_tokens`) is computed on read so the store stays append-only
and cheap to write during a run.
"""
from __future__ import annotations

import contextvars
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from attrs import frozen


@frozen
class Calibration:
    """Aggregated usage for a bucket. ``mean_input`` / ``mean_output`` are
    average tokens per call; ``n`` is the sample size behind them."""
    n: int
    mean_input: float
    mean_output: float


class CalibrationStore:
    """Append-only store of per-call LLM token usage."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = str(db_path)
        # One long-lived connection, reused for every record/read. The
        # describe path emits one usage row per element (tens of
        # thousands in a large run), so reconnecting per call would
        # dominate the write cost. ``check_same_thread=False`` keeps it
        # safe if a writer offloads to a worker thread; access is
        # otherwise serialized on the event loop.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage (
                phase TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                language TEXT,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL
            )
            """,
        )
        self._conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_llm_usage_bucket '
            'ON llm_usage (model, phase, doc_type, language)',
        )
        self._conn.commit()

    def record(
        self,
        *,
        phase: str,
        doc_type: str,
        language: str | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Persist one call's usage. Cheap, append-only."""
        self._conn.execute(
            'INSERT INTO llm_usage '
            '(phase, doc_type, language, model, input_tokens, '
            'output_tokens) VALUES (?, ?, ?, ?, ?, ?)',
            (phase, doc_type, language, model,
             int(input_tokens), int(output_tokens)),
        )
        self._conn.commit()

    def mean_tokens(
        self,
        *,
        phase: str,
        model: str,
        doc_type: str | None = None,
        language: str | None = None,
    ) -> Calibration | None:
        """Average input/output tokens per call for the matching bucket.

        ``doc_type`` / ``language`` narrow the match; omit them to
        aggregate phase-wide (a coarser fallback). Returns ``None`` when
        no rows match, so callers fall back to the heuristic.
        """
        clauses = ['model = ?', 'phase = ?']
        params: list = [model, phase]
        if doc_type is not None:
            clauses.append('doc_type = ?')
            params.append(doc_type)
        if language is not None:
            clauses.append('language = ?')
            params.append(language)
        where = ' AND '.join(clauses)
        row = self._conn.execute(
            f'SELECT COUNT(*), AVG(input_tokens), AVG(output_tokens) '
            f'FROM llm_usage WHERE {where}',
            params,
        ).fetchone()
        n = row[0] if row else 0
        if not n:
            return None
        return Calibration(
            n=int(n), mean_input=float(row[1]), mean_output=float(row[2]),
        )


# ---------------------------------------------------------------------------
# Usage capture — a thin, optional seam so the low-level LLM provider can
# emit per-call token usage without knowing the calling context. The
# caller sets the (phase, doc_type, language) context and an observer
# (e.g. ``CalibrationStore.record``); the provider just calls
# :func:`emit_usage`. Both default to no-ops, so emit_usage is always
# safe to call.
# ---------------------------------------------------------------------------

_usage_observer: contextvars.ContextVar = contextvars.ContextVar(
    'ariadne_usage_observer', default=None,
)
_usage_context: contextvars.ContextVar = contextvars.ContextVar(
    'ariadne_usage_context', default=None,
)


@contextmanager
def set_usage_observer(observer):
    """Route :func:`emit_usage` calls to ``observer(phase=, doc_type=,
    language=, model=, input_tokens=, output_tokens=)`` for the duration
    of the block (e.g. ``CalibrationStore.record``)."""
    token = _usage_observer.set(observer)
    try:
        yield
    finally:
        _usage_observer.reset(token)


@contextmanager
def usage_context(*, phase: str, doc_type: str, language: str | None):
    """Tag usage emitted within the block with this bucket."""
    token = _usage_context.set((phase, doc_type, language))
    try:
        yield
    finally:
        _usage_context.reset(token)


def emit_usage(*, model: str, input_tokens: int, output_tokens: int) -> None:
    """Report one call's token usage. No-op unless both an observer and a
    context are active — so the provider can call this unconditionally."""
    observer = _usage_observer.get()
    ctx = _usage_context.get()
    if observer is None or ctx is None:
        return
    phase, doc_type, language = ctx
    observer(
        phase=phase, doc_type=doc_type, language=language, model=model,
        input_tokens=int(input_tokens), output_tokens=int(output_tokens),
    )


__all__ = [
    'Calibration', 'CalibrationStore',
    'emit_usage', 'set_usage_observer', 'usage_context',
]
