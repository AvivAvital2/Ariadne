"""Contract tests for ``ariadne batch list`` / ``clear`` (#45.10).

The CLI command surface for managing pending Anthropic batches.
A pending batch lives in the staleness DB — these commands let the
user inspect or delete those rows without going through
``ariadne generate``'s auto-resume path.

Tests pin:
- ``list`` with no batches → exit 0 + empty-state message.
- ``list`` with recorded batches → exit 0 + each batch_id shown.
- ``clear`` of an existing batch → exit 0 + confirmation message;
  the row actually goes away.
- ``clear`` of a non-existent batch → exit 1 + warning message
  (so a typo or already-cleared id doesn't pretend success).

Stubs return -1 for both commands so the exit-code assertions all
fail behaviorally.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest
from rich.console import Console

import cli.batch as cli_batch
from docgen.staleness import StalenessTracker


@pytest.fixture
def captured_console(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Replace ``cli_batch.console`` with one that writes to a
    StringIO buffer for capture. Rich's default Console doesn't
    play nicely with pytest's capsys for ANSI output."""
    buf = io.StringIO()
    monkeypatch.setattr(
        cli_batch, 'console',
        Console(file=buf, force_terminal=False, no_color=True),
    )
    return buf


@pytest.fixture
def patched_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Make ``cli_batch`` resolve config to a tmp staleness DB.

    The cmd_batch_* implementations call ``get_config()`` to find
    ``staleness_db_path``; this fixture redirects that to a tmp file
    so tests don't pollute the real DB.
    """
    db_path = tmp_path / 'staleness.db'

    class FakeCfg:
        staleness_db_path = str(db_path)

    monkeypatch.setattr(
        cli_batch, 'get_config', lambda: FakeCfg(), raising=False,
    )
    return db_path


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestBatchList:
    def test_empty_lists_no_batches(
        self, patched_config: Path, captured_console: io.StringIO,
    ) -> None:
        """No pending rows → exit 0 + empty-state message."""
        args = argparse.Namespace(batch_action='list')
        rc = cli_batch.cmd_batch_list(args)

        assert rc == 0
        out = captured_console.getvalue()
        assert 'no pending' in out.lower()

    def test_lists_recorded_batches(
        self, patched_config: Path, captured_console: io.StringIO,
    ) -> None:
        """Pre-record two batches → both appear in the output."""
        st = StalenessTracker(patched_config)
        st.record_pending_batch(
            'msgbatch_alpha', '[]', '{}', 'hash-A',
        )
        st.record_pending_batch(
            'msgbatch_beta', '[]', '{}', 'hash-B',
        )
        st.close()

        args = argparse.Namespace(batch_action='list')
        rc = cli_batch.cmd_batch_list(args)

        assert rc == 0
        out = captured_console.getvalue()
        assert 'msgbatch_alpha' in out
        assert 'msgbatch_beta' in out


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestBatchClear:
    def test_clear_existing_returns_zero_and_removes_row(
        self, patched_config: Path, captured_console: io.StringIO,
    ) -> None:
        """Pre-record a batch, clear it, verify exit 0 +
        confirmation + the row is actually gone (find returns None)."""
        st = StalenessTracker(patched_config)
        st.record_pending_batch(
            'msgbatch_to_clear', '[]', '{}', 'hash-X',
        )
        st.close()

        args = argparse.Namespace(
            batch_action='clear', batch_id='msgbatch_to_clear',
        )
        rc = cli_batch.cmd_batch_clear(args)

        assert rc == 0
        out = captured_console.getvalue()
        assert 'msgbatch_to_clear' in out

        # Row actually deleted.
        st2 = StalenessTracker(patched_config)
        try:
            assert st2.find_pending_batch('hash-X') is None
        finally:
            st2.close()

    def test_clear_missing_returns_one_with_warning(
        self, patched_config: Path, captured_console: io.StringIO,
    ) -> None:
        """Clearing a non-existent id → exit 1 + warning. Critical
        for safety: typos shouldn't pretend success."""
        args = argparse.Namespace(
            batch_action='clear', batch_id='nonexistent',
        )
        rc = cli_batch.cmd_batch_clear(args)

        assert rc == 1
        out = captured_console.getvalue()
        # Either 'No such' or the literal id appears in the warning.
        assert 'nonexistent' in out or 'no such' in out.lower()
