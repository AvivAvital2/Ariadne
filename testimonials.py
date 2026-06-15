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

import re
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


# --- richness ranking -------------------------------------------------------
# The best-of is ranked by the quality SCORE plus concrete "feature-rich"
# signals — a diagram, citations of specific source files, and thoroughness —
# so a terse high score can't outrank a detailed, diagram-backed, file-grounded
# answer. Weights are modest (score stays dominant) and tunable.
_DIAGRAM_BONUS = 1.0
_FILE_REF_BONUS = 0.3          # per distinct source file cited
_FILE_REF_CAP = 4             # → up to +1.2
_THOROUGHNESS_CAP = 0.8
_FILE_REF_RE = re.compile(
    r'\b[\w./-]+\.'
    r'(?:py|pyi|ts|tsx|js|jsx|go|rs|java|kt|rb|c|cc|cpp|h|hpp|cs|php|swift|scala|'
    r'sql|sh|bash|zsh|yaml|yml|json|toml|ini|cfg|conf|xml|html|css|scss|vue|md|'
    r'rst|proto|gradle)\b',
    re.IGNORECASE,
)


def _count_file_refs(answer: str) -> int:
    """Distinct source-file paths cited in the answer (exact-file mapping)."""
    return len({m.lower() for m in _FILE_REF_RE.findall(answer)})


def _thoroughness(answer: str) -> float:
    """Detail signal (0…cap) from length + code blocks + bullet/numbered lists."""
    length = min(len(answer) / 1500.0, 1.0) * 0.4
    code = 0.25 if '```' in answer else 0.0
    bullets = 0.15 if re.search(r'\n\s*(?:[-*]|\d+\.)\s', answer) else 0.0
    return round(min(length + code + bullets, _THOROUGHNESS_CAP), 4)


def _richness_of(score: int, answer: str, n_images: int) -> float:
    """Composite rank: quality score + feature-rich signals.

    A diagram, citations of specific source files, and thoroughness each lift the
    rank, so the top-N favours the most detailed, feature-rich answers — not just
    the highest bare score.
    """
    diagram = _DIAGRAM_BONUS if n_images > 0 else 0.0
    files = min(_count_file_refs(answer), _FILE_REF_CAP) * _FILE_REF_BONUS
    return float(score) + diagram + files + _thoroughness(answer)


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

    Ranked by **richness** — the score plus feature-rich signals (a diagram,
    source-file citations, thoroughness) — not score alone. Returns True if
    stored (evicting the lowest-ranked when full), False if it didn't beat the
    floor. ``images`` are raw PNG bytes.

    ``source_ts`` is the originating Slack message id (set by the channel
    backfill): if an entry with this id is already stored, the record is a
    no-op (returns False), so re-scanning a channel never duplicates an entry.
    Live captures pass no ``source_ts`` and skip the dedup scan entirely.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    entries = _entry_dirs(root)
    new_richness = _richness_of(score, answer, len(images))
    if source_ts is not None or len(entries) >= MAX_KEEP:
        loaded = [_load(e) for e in entries]   # for dedup + richness-based eviction
        if source_ts is not None and any(t.source_ts == source_ts for t in loaded):
            return False
        if len(entries) >= MAX_KEEP:
            lowest = min(
                loaded, key=lambda t: _richness_of(t.score, t.answer, len(t.images)))
            if new_richness <= _richness_of(lowest.score, lowest.answer, len(lowest.images)):
                return False
            shutil.rmtree(lowest.path)

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
    """The richest stored interactions, best first (at most ``limit``).

    Ranked by :func:`_richness_of` — the quality score plus feature-rich signals
    (a diagram, source-file citations, thoroughness) — not by score alone.
    """
    loaded = [_load(e) for e in _entry_dirs(Path(root))]
    loaded.sort(key=lambda t: _richness_of(t.score, t.answer, len(t.images)), reverse=True)
    return loaded[:limit]


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
