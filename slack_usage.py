"""Bridge-side per-user usage log — counts of questions/hits/misses by user.

The Slack bridge appends one line per answered turn to a JSONL file under the
swap-proof ``.ariadne/local/`` store (the same regen-proof location the
testimonials best-of uses, so it survives the ``ariadne.db`` rebuild that wipes
``usage_events``). Each record holds COUNTS + metadata only — the Slack user id,
the resolved display name, the hit/miss outcome, the self-score, and a
timestamp — and never the question text. ``ariadne usage`` calls :func:`aggregate`
to roll the log up per user.

Root-agnostic — callers pass the directory — so it is fully unit-testable.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from attrs import frozen

USAGE_FILE = 'slack_usage.jsonl'


@frozen
class UserUsage:
    """One user's rolled-up usage over the aggregation window."""

    actor: str
    name: str
    questions: int
    hits: int
    misses: int


def record(
    root: str | Path,
    *,
    asked_at: str,
    actor: str,
    name: str,
    outcome: str,
    score: int | None = None,
    source_ts: str | None = None,
) -> None:
    """Append one answered-turn record to the log.

    ``actor`` is the Slack user id, ``name`` its resolved display name (falls
    back to the id when resolution fails), ``outcome`` one of ``hit`` / ``miss``
    / ``answered`` / ``error``, and ``asked_at`` an ISO-8601 timestamp. No
    question text is stored.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {'ts': asked_at, 'actor': actor, 'name': name,
         'outcome': outcome, 'score': score, 'source_ts': source_ts}
    )
    with (root / USAGE_FILE).open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def aggregate(root: str | Path, *, days: int | None = None) -> list[UserUsage]:
    """Roll the log up per user, newest-resolved-name wins, busiest first.

    ``days`` limits to records within the last N days (``None`` = all time).
    Tolerates a missing file and skips malformed / out-of-window lines.
    """
    path = Path(root) / USAGE_FILE
    if not path.is_file():
        return []

    cutoff = datetime.now(UTC) - timedelta(days=days) if days is not None else None

    agg: dict[str, dict] = {}
    order: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        if cutoff is not None:
            try:
                if datetime.fromisoformat(str(rec.get('ts'))) < cutoff:
                    continue
            except (TypeError, ValueError):
                continue

        actor = rec.get('actor') or ''
        if actor not in agg:
            agg[actor] = {'name': actor, 'questions': 0, 'hits': 0, 'misses': 0}
            order.append(actor)
        bucket = agg[actor]
        bucket['questions'] += 1
        outcome = rec.get('outcome')
        if outcome == 'hit':
            bucket['hits'] += 1
        elif outcome == 'miss':
            bucket['misses'] += 1
        name = rec.get('name')
        if name and name != actor:        # a real resolved name beats the id fallback
            bucket['name'] = name

    users = [
        UserUsage(actor=actor, name=b['name'], questions=b['questions'],
                  hits=b['hits'], misses=b['misses'])
        for actor, b in ((a, agg[a]) for a in order)
    ]
    users.sort(key=lambda u: (-u.questions, u.actor))
    return users


def recorded_source_ts(root: str | Path) -> set[str]:
    """The originating Slack message ids already recorded (backfill dedup).

    A backfill skips any turn whose id is already here, so a re-scan only
    records new history. Live turns carry no ``source_ts`` and never appear.
    """
    path = Path(root) / USAGE_FILE
    if not path.is_file():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and rec.get("source_ts"):
            out.add(rec["source_ts"])
    return out


__all__ = ['USAGE_FILE', 'UserUsage', 'aggregate', 'record']
