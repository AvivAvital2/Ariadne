"""Contract tests for pending_batches persistence (#45.3).

The orchestrator's batch path needs durable state so a crash mid-poll
doesn't forfeit a batch the user's already paid Anthropic for. The
``pending_batches`` table stores: ``batch_id`` (Anthropic-assigned),
the prompts and file-to-indices map, and a ``config_hash`` so resume
refuses to adopt a batch from a different doc_types/provider/model
combination.

Tests are written so that under the stubbed methods (record/clear/
find/list as no-ops returning sentinels), each test independently
fails behaviorally. Where the contract is naturally a paired
positive+negative (e.g. "find returns record when hash matches,
None otherwise"), the pair is folded into a single test so the stub
can't pass either half by accident.

The schema migration test is the one exception: the migration is
real code (idempotent CREATE TABLE), so this test passes regardless
of stub state. Kept separate as a clear "the table exists" pin.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from docgen.staleness import PendingBatch, StalenessTracker


@pytest.fixture
def tracker(tmp_path: Path) -> StalenessTracker:
    """Fresh StalenessTracker on a tmp DB path. Schema migration runs
    in __attrs_post_init__ so the pending_batches table exists by the
    time tests touch it."""
    t = StalenessTracker(tmp_path / 'staleness.db')
    yield t
    t.close()


# ---------------------------------------------------------------------------
# Schema migration (real — always passes)
# ---------------------------------------------------------------------------


class TestPendingBatchSchema:
    def test_table_exists_after_init(self, tracker: StalenessTracker) -> None:
        """Schema migration must create pending_batches with the
        expected columns. Bites a fix that forgets the migration or
        names a column differently — downstream methods would error
        at SQL parse time."""
        assert tracker._conn is not None
        cursor = tracker._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='pending_batches'",
        )
        assert cursor.fetchone() is not None

        cols = {
            row['name']
            for row in tracker._conn.execute(
                'PRAGMA table_info(pending_batches)',
            )
        }
        assert cols == {
            'batch_id', 'submitted_at', 'prompts_json',
            'file_to_idxs_json', 'config_hash',
        }


# ---------------------------------------------------------------------------
# record_pending_batch
# ---------------------------------------------------------------------------


class TestRecordPendingBatch:
    def test_record_returns_pending_batch_with_input_values(
        self, tracker: StalenessTracker,
    ) -> None:
        """The PendingBatch returned must carry the inputs the caller
        passed — bites a fix that returns a stub or sentinel object."""
        result = tracker.record_pending_batch(
            batch_id='msgbatch_abc123',
            prompts_json='[{"foo": "bar"}]',
            file_to_idxs_json='{"a.py": [0]}',
            config_hash='hash-xyz',
        )
        assert isinstance(result, PendingBatch)
        assert result.batch_id == 'msgbatch_abc123'
        assert result.prompts_json == '[{"foo": "bar"}]'
        assert result.file_to_idxs_json == '{"a.py": [0]}'
        assert result.config_hash == 'hash-xyz'
        # submitted_at set internally; non-empty (real impl uses
        # ISO-8601 UTC).
        assert result.submitted_at

    def test_record_persists_so_find_recovers(
        self, tracker: StalenessTracker,
    ) -> None:
        """Recording must actually write to the DB — pins that
        record_pending_batch isn't a no-op. The roundtrip via
        find_pending_batch is the user-observable contract: resume
        finds what submit recorded."""
        tracker.record_pending_batch(
            batch_id='msgbatch_xyz',
            prompts_json='[]',
            file_to_idxs_json='{}',
            config_hash='hash-A',
        )
        found = tracker.find_pending_batch('hash-A')
        assert found is not None
        assert found.batch_id == 'msgbatch_xyz'
        # Round-trip must preserve the JSON payloads byte-for-byte.
        # Drift here corrupts prompts on resume.
        assert found.prompts_json == '[]'
        assert found.file_to_idxs_json == '{}'


# ---------------------------------------------------------------------------
# find_pending_batch — combined positive+negative
# ---------------------------------------------------------------------------


class TestFindPendingBatch:
    def test_finds_by_matching_hash_returns_none_otherwise(
        self, tracker: StalenessTracker,
    ) -> None:
        """Combined contract: empty DB → None, matching hash → record,
        non-matching hash → None. Folded into one test so a stub that
        always returns None passes the empty/non-matching branches by
        accident; the matching-hash assertion still fails under the
        stub."""
        # Empty DB: None
        assert tracker.find_pending_batch('any-hash') is None

        # After record under hash-A:
        tracker.record_pending_batch(
            'b1', '[]', '{}', 'hash-A',
        )

        # Matching hash returns the record
        found = tracker.find_pending_batch('hash-A')
        assert found is not None
        assert found.batch_id == 'b1'

        # Non-matching hash returns None — critical: a pending batch
        # from a different config must NOT be adopted by a new run,
        # otherwise the user gets docs they didn't ask for.
        assert tracker.find_pending_batch('hash-B') is None

    def test_returns_most_recent_when_multiple_match(
        self, tracker: StalenessTracker,
    ) -> None:
        """Two pending batches with the same config_hash — the more
        recent one is returned. Resume should pick the freshest
        in-flight batch (older orphans get cleaned via
        ``ariadne batch clear``)."""
        tracker.record_pending_batch(
            'b_old', '[]', '{}', 'shared-hash',
        )
        tracker.record_pending_batch(
            'b_new', '[]', '{}', 'shared-hash',
        )
        found = tracker.find_pending_batch('shared-hash')
        assert found is not None
        assert found.batch_id == 'b_new'


# ---------------------------------------------------------------------------
# clear_pending_batch — combined existing+missing+roundtrip
# ---------------------------------------------------------------------------


class TestClearPendingBatch:
    def test_clear_existing_true_missing_false_and_idempotent(
        self, tracker: StalenessTracker,
    ) -> None:
        """Combined contract:
        - Clearing a non-existent batch returns False (no exception —
          ``ariadne batch clear`` mustn't crash on a typo).
        - Clearing an existing batch returns True ("I removed
          something" so the CLI can print it).
        - Clearing the same batch twice is idempotent — the second
          call returns False because the row is gone.
        Fails under the stub on the True branch."""
        # Missing batch → False
        assert tracker.clear_pending_batch('nonexistent') is False

        tracker.record_pending_batch(
            'b1', '[]', '{}', 'h',
        )
        # Existing batch → True
        assert tracker.clear_pending_batch('b1') is True
        # Idempotent: second clear → False
        assert tracker.clear_pending_batch('b1') is False

    def test_clear_removes_record_so_find_returns_none(
        self, tracker: StalenessTracker,
    ) -> None:
        """End-to-end: record → confirm find recovers it → clear →
        confirm find no longer recovers it. Pins both that record
        actually persists AND that clear actually deletes — a stub
        that no-ops both passes the final ``find is None`` but fails
        the intermediate ``find is not None`` assertion."""
        tracker.record_pending_batch(
            'b1', '[]', '{}', 'h',
        )
        # Intermediate check: record persisted before clear runs.
        # Without this, a stub that no-ops both record and clear
        # would pass the test trivially.
        assert tracker.find_pending_batch('h') is not None

        tracker.clear_pending_batch('b1')
        assert tracker.find_pending_batch('h') is None


# ---------------------------------------------------------------------------
# list_pending_batches — combined empty + populated
# ---------------------------------------------------------------------------


class TestListPendingBatches:
    def test_list_empty_initially_grows_with_records(
        self, tracker: StalenessTracker,
    ) -> None:
        """Combined: starts empty, grows as batches are recorded.
        Folded so a stub that always returns [] passes the empty
        branch but fails the populated branch."""
        # Empty initially
        assert tracker.list_pending_batches() == []

        tracker.record_pending_batch(
            'b1', '[]', '{}', 'h1',
        )
        tracker.record_pending_batch(
            'b2', '[]', '{}', 'h2',
        )

        ids = {b.batch_id for b in tracker.list_pending_batches()}
        assert ids == {'b1', 'b2'}
