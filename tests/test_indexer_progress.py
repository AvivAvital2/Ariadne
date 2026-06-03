"""Contract tests for ``IndexerProgress`` parsing + streaming.

The CLI's ``cmd_index`` drives a Rich ``Progress`` widget from
events emitted by ``PythonIndexerAdapter`` while scip-python runs.
The adapter parses scip-python's free-form log lines into structured
``IndexerProgress`` events; this file pins the parser + stream
dispatcher contracts.

Tests are deliberately at the parser/stream layer, not the full
subprocess: spawning a real scip-python process is slow and
environment-dependent. The adapter's actual subprocess invocation
is covered by ``test_python_indexer_adapter.py``.
"""
from __future__ import annotations

import pytest

from docgen.scip_indexers import (
    IndexerProgress,
    _parse_scip_python_line,
    _stream_progress,
)


# ---------------------------------------------------------------------------
# _parse_scip_python_line
# ---------------------------------------------------------------------------


class TestParseScipPythonLine:
    def test_total_line_emits_total_event(self) -> None:
        """``(timestamp) Total Project Files N`` → ``total`` event with N.
        scip-python emits this once at startup so the CLI can size the
        progress bar before the first tick lands."""
        event = _parse_scip_python_line(
            '(01:53:07) Total Project Files 3192',
        )
        assert event is not None
        assert event.kind == 'total'
        assert event.total == 3192
        assert event.current == 0

    def test_tick_line_emits_tick_event(self) -> None:
        """``(timestamp)   N / M`` → ``tick`` event with current=N,
        total=M. Pins the canonical progress format scip-python emits
        every few seconds while parsing dependencies."""
        event = _parse_scip_python_line(
            '(01:47:46)   556 / 61657',
        )
        assert event is not None
        assert event.kind == 'tick'
        assert event.current == 556
        assert event.total == 61657

    def test_warning_prefix_emits_warning_event(self) -> None:
        """``Warning: ...`` lines are surfaced as warnings so the CLI
        can print them above the progress bar instead of hiding them."""
        event = _parse_scip_python_line(
            'Warning: Could not find package information for: anthropic',
        )
        assert event is not None
        assert event.kind == 'warning'
        assert 'anthropic' in event.text

    def test_unsupported_python_emits_warning_event(self) -> None:
        """The ``Python version X from interpreter is unsupported``
        message has no ``Warning:`` prefix but is genuinely a warning
        about reduced quality. Detected by ``unsupported`` substring.
        Bites a fix that drops it as a generic message."""
        event = _parse_scip_python_line(
            'Python version 3.14 from interpreter is unsupported',
        )
        assert event is not None
        assert event.kind == 'warning'

    def test_unrecognized_line_falls_through_to_message(self) -> None:
        """Any other non-empty line maps to ``message`` so the caller
        can choose to ignore it (default) or print at high verbosity."""
        event = _parse_scip_python_line(
            '(01:53:07) Indexing /path/to/ariadne with version 0.1',
        )
        assert event is not None
        assert event.kind == 'message'
        assert 'Indexing' in event.text

    def test_blank_line_returns_none(self) -> None:
        """Blank lines are noise; suppress at the parser layer."""
        assert _parse_scip_python_line('') is None
        assert _parse_scip_python_line('   ') is None

    def test_long_missing_package_warning_collapsed_to_summary(self) -> None:
        """Pyright's 'Could not find package information for: <list>' line
        with more than a handful of packages collapses to a one-line
        summary surfacing the count + a sample. The full 200-package
        list is never useful in a progress display — the count is the
        actionable signal."""
        many = ', '.join(f'pkg{i}' for i in range(50))
        line = f'Warning: Could not find package information for: {many}'
        event = _parse_scip_python_line(line)
        assert event is not None
        assert event.kind == 'warning'
        assert '50 packages' in event.text
        assert 'pkg0' in event.text  # sample
        # Full list NOT in summary — that's the whole point.
        assert 'pkg49' not in event.text

    def test_short_missing_package_warning_kept_verbatim(self) -> None:
        """A few-package version of the warning is genuinely
        informative — keep it as-is, only summarize the floods. Bites
        a fix that always summarizes regardless of count."""
        line = (
            'Warning: Could not find package information for: '
            'foo, bar, baz'
        )
        event = _parse_scip_python_line(line)
        assert event is not None
        assert event.kind == 'warning'
        # Original list survives.
        assert 'foo' in event.text and 'bar' in event.text and 'baz' in event.text


# ---------------------------------------------------------------------------
# _stream_progress
# ---------------------------------------------------------------------------


class TestStreamProgress:
    def test_dispatches_each_parsed_event_to_callback(self) -> None:
        """Walk a canned line iterator, callback fires once per parsed
        event (blank lines are skipped). Pins the dispatch contract
        the Popen wrapper uses."""
        lines = [
            '(01:53:07) Total Project Files 100',
            '',
            '(01:53:08) Indexing /tmp/foo with version 0.1',
            '(01:53:09)   42 / 100',
            'Warning: missing pkg',
            '(01:53:10)   100 / 100',
        ]
        events: list[IndexerProgress] = []
        buffered = _stream_progress(lines, events.append)

        # Five non-blank lines → five callback invocations.
        assert len(events) == 5
        assert events[0].kind == 'total'
        assert events[0].total == 100
        assert events[1].kind == 'message'  # 'Indexing' line
        assert events[2].kind == 'tick'
        assert events[2].current == 42
        assert events[3].kind == 'warning'
        assert events[4].kind == 'tick'
        assert events[4].current == 100

        # Buffered raw lines (used for error messages on failure)
        # carry every non-blank line as-is.
        assert len(buffered) == 5
        assert 'Total Project Files 100' in buffered[0]

    def test_blank_lines_are_skipped_from_buffer_too(self) -> None:
        """Empty lines aren't buffered — the failure-mode text we
        surface should be useful, not padded with blanks."""
        events: list[IndexerProgress] = []
        buffered = _stream_progress(['', '   ', 'Warning: x'], events.append)
        assert buffered == ['Warning: x']
        assert len(events) == 1
