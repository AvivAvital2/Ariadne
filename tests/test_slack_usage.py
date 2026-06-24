"""Bridge-side per-user usage log: counts of questions/hits/misses by user.

The Slack bridge records one line per answered turn to a swap-proof JSONL file
under ``.ariadne/local/`` (alongside the testimonials store), so per-user usage
survives an ``ariadne.db`` rebuild. It stores COUNTS + metadata only — the
Slack user id, the resolved display name, the hit/miss outcome, the self-score,
and a timestamp — never the question text. ``aggregate`` rolls the log up per
user for ``ariadne usage``.

Root-agnostic — callers pass the directory — so it is fully unit-testable.
"""
from __future__ import annotations

import slack_usage
from schema import _now_iso


def _rec(root, **kw):
    base = dict(asked_at=_now_iso(), actor='U1', name='alice',
                outcome='answered', score=None)
    base.update(kw)
    slack_usage.record(root, **base)


def test_aggregate_groups_counts_per_user(tmp_path):
    _rec(tmp_path, actor='U_alice', name='alice', outcome='hit', score=8)
    _rec(tmp_path, actor='U_alice', name='alice', outcome='miss', score=2)
    _rec(tmp_path, actor='U_alice', name='alice', outcome='answered')
    _rec(tmp_path, actor='U_bob', name='bob', outcome='hit', score=9)

    users = slack_usage.aggregate(tmp_path)

    # questions = every recorded turn; sorted by questions desc → alice (3), bob (1)
    assert [(u.name, u.questions, u.hits, u.misses) for u in users] == [
        ('alice', 3, 1, 1),
        ('bob', 1, 1, 0),
    ]


def test_no_question_text_is_persisted(tmp_path):
    """Privacy pin: the on-disk shape carries counts/metadata only."""
    _rec(tmp_path, actor='U_alice', name='alice', outcome='hit', score=8)
    raw = (tmp_path / slack_usage.USAGE_FILE).read_text(encoding='utf-8')
    assert 'question' not in raw.lower()
    assert 'U_alice' in raw and 'alice' in raw and 'hit' in raw


def test_days_window_filters_old_records(tmp_path):
    _rec(tmp_path, actor='U_alice', name='alice', outcome='hit',
         asked_at='2000-01-01T00:00:00+00:00')
    _rec(tmp_path, actor='U_alice', name='alice', outcome='hit', asked_at=_now_iso())

    assert [(u.name, u.questions) for u in slack_usage.aggregate(tmp_path, days=7)] == [
        ('alice', 1),
    ]
    assert [(u.name, u.questions) for u in slack_usage.aggregate(tmp_path)] == [
        ('alice', 2),
    ]


def test_missing_or_malformed_store_is_tolerated(tmp_path):
    assert slack_usage.aggregate(tmp_path / 'nope') == []

    root = tmp_path / 'local'
    root.mkdir()
    # junk lines of every shape are skipped; the valid record still aggregates
    (root / slack_usage.USAGE_FILE).write_text(
        'not json\n'   # unparseable JSON
        '\n'           # blank line
        '42\n',        # valid JSON, but not an object
        encoding='utf-8',
    )
    _rec(root, actor='U_alice', name='alice', outcome='hit')

    users = slack_usage.aggregate(root)
    assert [(u.actor, u.questions, u.hits) for u in users] == [('U_alice', 1, 1)]
    # the backfill-dedup reader tolerates the same junk (and live records carry
    # no source_ts, so the recorded set is empty)
    assert slack_usage.recorded_source_ts(root) == set()

    # under a days window, a record with an unparseable timestamp is dropped,
    # not crashed — the well-timestamped record survives
    _rec(root, actor='U_bob', name='bob', outcome='hit', asked_at='not-a-timestamp')
    assert [u.actor for u in slack_usage.aggregate(root, days=7)] == ['U_alice']


def test_latest_resolved_name_wins_over_id_fallback(tmp_path):
    # first turn only had the id (name resolution failed → name == actor)
    _rec(tmp_path, actor='U_alice', name='U_alice', outcome='answered')
    _rec(tmp_path, actor='U_alice', name='alice', outcome='hit')

    users = slack_usage.aggregate(tmp_path)
    assert users[0].name == 'alice'


def test_source_ts_supports_idempotent_backfill(tmp_path):
    """Backfill records carry the originating Slack message ts so a re-scan can
    skip what's already stored; live records (no source_ts) never pollute it."""
    _rec(tmp_path, actor='U_a', name='alice', outcome='hit', source_ts='1718000000.000050')
    _rec(tmp_path, actor='U_b', name='bob', outcome='miss', source_ts='1718000100.000050')
    _rec(tmp_path, actor='U_a', name='alice', outcome='answered')   # live: no source_ts

    assert slack_usage.recorded_source_ts(tmp_path) == {
        '1718000000.000050', '1718000100.000050'}
    # every turn still counts toward the per-user totals
    assert {u.actor: u.questions for u in slack_usage.aggregate(tmp_path)} == {
        'U_a': 2, 'U_b': 1}


def test_recorded_source_ts_missing_store_is_empty(tmp_path):
    assert slack_usage.recorded_source_ts(tmp_path / 'nope') == set()
