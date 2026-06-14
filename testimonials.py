"""Local 'best-of' testimonials store — a bounded top-N of Q&A pairs on disk.

Each kept interaction is a self-contained folder under a `local/` directory
(in production: ``.ariadne/local/`` — gitignored and excluded from cataloguing,
so it is **regen-proof**: it survives the ``ariadne.db`` rebuild/swap that wipes
``usage_events``). The Slack bridge calls :func:`record` after a scored answer;
``ariadne testimonials`` calls :func:`top` to pull the showcase.

The store keeps at most :data:`MAX_KEEP` entries — the highest-scored of all
time. A new interaction is stored only if the store isn't full or it beats the
current lowest; the displaced one is deleted. Entry layout::

    local/
      09_20260614T143005Z/
        qna.md            # frontmatter (score, duration, asked_at, permalink) + Q + A
        diagram-1.png     # any response images, alongside

Root-agnostic — callers pass the directory, so it is fully unit-testable.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from attrs import frozen

MAX_KEEP = 20

# Production location, relative to the Ariadne working dir. `.ariadne/` is
# already gitignored + in the exclude policy, so the store is never committed
# or catalogued — and it is NOT inside ariadne.db, so a DB swap can't erase it.
LOCAL_SUBDIR = Path('.ariadne') / 'local'


def local_dir(ariadne_dir: str | Path) -> Path:
    """The production store path under an Ariadne working directory."""
    return Path(ariadne_dir) / LOCAL_SUBDIR


@frozen
class Testimonial:
    """One stored Q&A pair, read back from disk."""

    score: int
    duration_seconds: float
    asked_at: str
    question: str
    answer: str
    permalink: str | None
    source_ts: str | None
    images: tuple[Path, ...]
    path: Path


def _entry_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in root.iterdir() if p.is_dir() and (p / 'qna.md').is_file()]


def _score_of(entry: Path) -> int:
    """Score is the entry folder's name prefix (``NN_…``)."""
    try:
        return int(entry.name.split('_', 1)[0])
    except ValueError:
        return -1


def record(
    root: str | Path,
    *,
    question: str,
    answer: str,
    score: int,
    duration_seconds: float,
    asked_at: str,
    permalink: str | None = None,
    source_ts: str | None = None,
    images: list[bytes] = (),
) -> bool:
    """Store this interaction if it ranks in the all-time top :data:`MAX_KEEP`.

    Returns True if it was stored (evicting the lowest when the store was full),
    False if it didn't beat the current floor. ``images`` are raw PNG bytes.

    ``source_ts`` is the originating Slack message id (set by the channel
    backfill): if an entry with this id is already stored, the record is a
    no-op (returns False), so re-scanning a channel never duplicates an entry.
    Live captures pass no ``source_ts`` and skip the dedup scan entirely.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    entries = _entry_dirs(root)
    if source_ts is not None and any(_load(e).source_ts == source_ts for e in entries):
        return False
    if len(entries) >= MAX_KEEP:
        lowest = min(entries, key=_score_of)
        if score <= _score_of(lowest):
            return False
        shutil.rmtree(lowest)

    safe_ts = asked_at.replace(':', '').replace('-', '')
    entry = root / f'{int(score):02d}_{safe_ts}'
    suffix = 0
    while entry.exists():
        suffix += 1
        entry = root / f'{int(score):02d}_{safe_ts}_{suffix}'
    entry.mkdir(parents=True)

    for i, data in enumerate(images, 1):
        (entry / f'diagram-{i}.png').write_bytes(data)

    frontmatter = (
        '---\n'
        f'score: {int(score)}\n'
        f'duration_seconds: {duration_seconds}\n'
        f'asked_at: {asked_at}\n'
        f'permalink: {permalink or ""}\n'
        f'source_ts: {source_ts or ""}\n'
        '---\n'
    )
    body = f'\n## Question\n{question}\n\n## Answer\n{answer}\n'
    (entry / 'qna.md').write_text(frontmatter + body, encoding='utf-8')
    return True


def top(root: str | Path, limit: int = MAX_KEEP) -> list[Testimonial]:
    """The highest-scored stored interactions, best first (at most ``limit``)."""
    entries = sorted(_entry_dirs(Path(root)), key=_score_of, reverse=True)
    return [_load(e) for e in entries[:limit]]


def _load(entry: Path) -> Testimonial:
    text = (entry / 'qna.md').read_text(encoding='utf-8')
    meta: dict[str, str] = {}
    body = text
    if text.startswith('---\n'):
        fm, _, body = text[4:].partition('\n---\n')
        for line in fm.splitlines():
            key, sep, value = line.partition(':')
            if sep:
                meta[key.strip()] = value.strip()

    question = answer = ''
    after_q = body.split('## Question\n', 1)[-1] if '## Question\n' in body else ''
    if '\n## Answer\n' in after_q:
        q_part, _, a_part = after_q.partition('\n## Answer\n')
        question, answer = q_part.strip(), a_part.strip()
    else:
        question = after_q.strip()

    return Testimonial(
        score=int(meta.get('score') or _score_of(entry)),
        duration_seconds=float(meta.get('duration_seconds') or 0.0),
        asked_at=meta.get('asked_at', ''),
        question=question,
        answer=answer,
        permalink=(meta.get('permalink') or None),
        source_ts=(meta.get('source_ts') or None),
        images=tuple(sorted(entry.glob('diagram-*.png'))),
        path=entry,
    )


__all__ = ['MAX_KEEP', 'Testimonial', 'local_dir', 'record', 'top']
