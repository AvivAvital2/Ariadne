"""Evolving contract for the local 'best-of' testimonials store.

One test grows through the store's demands — record → serve top-N highest-first
→ bound + evict + reject below the floor → limit — asserting real on-disk
effects at each step (not banners). A second grows the robustness contract:
the store is human-inspectable markdown, so reads must survive a hand-edited
directory. The store is root-agnostic, so tests drive a tmp_path and never
touch a real .ariadne/.
"""
from __future__ import annotations

from pathlib import Path

from testimonials import MAX_KEEP, local_dir, record, top


def _rec(root: Path, score: int, *, q: str = 'q', **kw) -> bool:
    return record(
        root,
        question=f'{q}{score}',
        answer=f'answer-{score}',
        score=score,
        duration_seconds=1.0,
        asked_at=f'2026-06-14T00:00:{score % 60:02d}Z',
        **kw,
    )


def test_store_keeps_the_top_n_and_serves_them_faithfully(tmp_path: Path) -> None:
    # Demand 1 — a recorded pair round-trips: content, score, duration,
    # permalink, and images all survive to top() (answer may contain '##').
    assert record(
        tmp_path,
        question='How does X work?',
        answer='It does Y.\n\n## Note: even ## headers survive.',
        score=9, duration_seconds=12.5, asked_at='2026-06-14T14:30:05Z',
        permalink='https://acme.slack.com/p1', images=[b'\x89PNGfake'],
    ) is True
    one = top(tmp_path)
    assert len(one) == 1
    t = one[0]
    assert (t.question, t.score, t.duration_seconds, t.permalink) == (
        'How does X work?', 9, 12.5, 'https://acme.slack.com/p1')
    assert t.answer == 'It does Y.\n\n## Note: even ## headers survive.'
    assert len(t.images) == 1 and t.images[0].read_bytes() == b'\x89PNGfake'

    # Demand 2 — more pairs come back highest-first.
    for s in (3, 7, 5):
        _rec(tmp_path, s)
    assert [x.score for x in top(tmp_path)] == [9, 7, 5, 3]

    # Demand 3 — bounded: fill past the cap, only the highest MAX_KEEP survive.
    for s in range(10, 10 + MAX_KEEP + 5):
        _rec(tmp_path, s)
    kept = top(tmp_path)
    assert len(kept) == MAX_KEEP
    assert [x.score for x in kept] == sorted((x.score for x in kept), reverse=True)
    floor = min(x.score for x in kept)

    # Demand 4 — when full: below the floor is rejected, above it evicts the lowest.
    assert _rec(tmp_path, floor - 1, q='loser') is False
    assert all('loser' not in x.question for x in top(tmp_path))
    assert _rec(tmp_path, 999, q='winner') is True
    kept2 = top(tmp_path)
    assert len(kept2) == MAX_KEEP
    assert any(x.score == 999 for x in kept2)
    assert floor not in {x.score for x in kept2}

    # Demand 5 — limit caps the pull.
    assert [x.score for x in top(tmp_path, limit=3)] == \
        sorted((x.score for x in kept2), reverse=True)[:3]

    # Demand 6 — a source_ts (the originating Slack message) makes a re-record
    # idempotent, so a channel re-scan replaying the same messages never dups a
    # kept entry; a different source_ts is still kept on its own merits.
    assert _rec(tmp_path, 1000, q='scanned', source_ts='msg-1') is True
    n = len(top(tmp_path))
    assert _rec(tmp_path, 1000, q='scanned', source_ts='msg-1') is False   # same ts → skipped
    assert len(top(tmp_path)) == n
    assert _rec(tmp_path, 1000, q='other', source_ts='msg-2') is True       # different ts → kept
    assert sum('scanned' in x.question for x in top(tmp_path)) == 1


def test_store_tolerates_a_hand_edited_directory(tmp_path: Path) -> None:
    # A missing or empty store yields nothing — never crashes.
    assert top(tmp_path / 'missing') == []
    root = tmp_path / 'store'
    root.mkdir()
    assert top(root) == []
    _rec(root, 6)

    # It's human-editable markdown, so reads must survive hand-made entries:
    # (1) a non-numeric folder name + a body with no frontmatter and no Answer,
    weird = root / 'draft_note'
    weird.mkdir()
    (weird / 'qna.md').write_text('just jotting, no structure\n')
    # (2) frontmatter carrying a line without a colon,
    fm = root / '08_x'
    fm.mkdir()
    (fm / 'qna.md').write_text(
        '---\nscore: 8\nno colon here\n---\n\n## Question\nq\n\n## Answer\na\n')
    # (3) two qualifying answers with an identical score+timestamp (no clobber).
    record(root, question='first', answer='a', score=5,
           duration_seconds=1.0, asked_at='2026-06-14T00:00:05Z')
    record(root, question='second', answer='b', score=5,
           duration_seconds=1.0, asked_at='2026-06-14T00:00:05Z')

    got = top(root)
    draft = next(x for x in got if x.path.name == 'draft_note')
    assert draft.score == -1        # non-numeric prefix sorts last, no crash
    assert draft.answer == ''       # no Answer section parsed
    assert next(x for x in got if x.score == 8).answer == 'a'  # colon-less line ignored
    assert {x.question for x in got if x.score == 5} == {'first', 'second'}


def test_richer_answers_outrank_a_higher_bare_score(tmp_path: Path) -> None:
    # Ranking is by richness (score + diagram + source-file citations + detail),
    # not score alone — so a feature-rich 8 beats a terse 9.
    record(tmp_path, question='plain', answer='Short.', score=9,
           duration_seconds=1.0, asked_at='2026-06-14T00:00:09Z')
    record(tmp_path, question='rich', score=8,
           duration_seconds=1.0, asked_at='2026-06-14T00:00:08Z',
           answer='Detailed: it lives in `slack_bridge/scan.py` and `config.py`.\n'
                  '```py\nx = 1\n```\n- handles A\n- handles B',
           images=[b'\x89PNGdiagram'])
    assert [t.question for t in top(tmp_path)] == ['rich', 'plain']

    # ...but score still dominates between answers with the same richness signals:
    # a bare 4 stays below the bare 9.
    record(tmp_path, question='plain-low', answer='Short.', score=4,
           duration_seconds=1.0, asked_at='2026-06-14T00:00:04Z')
    assert [t.question for t in top(tmp_path)][-1] == 'plain-low'

    # Eviction is richness-gated too: fill the store with terser, higher-scored
    # entries (which evict the bare 4 and 9), then a fresh bare 9 still can't
    # displace the feature-rich 8.
    for s in range(20, 20 + MAX_KEEP - 1):       # 19 terse entries, all above rich-8
        record(tmp_path, question=f'terse{s}', answer='ok', score=s,
               duration_seconds=1.0, asked_at=f'2026-06-14T01:00:{s:02d}Z')
    assert record(tmp_path, question='plain9b', answer='ok', score=9,
                  duration_seconds=1.0, asked_at='2026-06-14T02:00:00Z') is False
    kept_qs = {t.question for t in top(tmp_path)}
    assert 'rich' in kept_qs                      # survived on richness, not bare score
    assert 'plain-low' not in kept_qs and 'plain9b' not in kept_qs


def test_local_dir_sits_under_dot_ariadne() -> None:
    assert local_dir('/srv/ariadne') == Path('/srv/ariadne/.ariadne/local')
