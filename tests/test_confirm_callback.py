"""Contract tests for confirm_callback infrastructure (#45.7).

The orchestrator's batch fork needs to prompt the user before
submitting a batch, since Anthropic's batch SLA is up to 24 hours
and the run is otherwise opaque. ``confirm_callback`` is the hook
the CLI wires; ``--yes`` short-circuits it via ``yes_confirm``.

Both helpers are async-shaped because the orchestrator's ``run()``
is async, and the default ``cli_confirm`` uses ``asyncio.to_thread``
to avoid blocking the event loop on ``input()``.

Tests pin:
- ``DocGenOrchestrator.confirm_callback`` field exists, defaults
  None, and accepts an async callable assigned post-construction
  (the CLI wires it that way).
- ``yes_confirm`` always returns True without consulting stdin.
- ``cli_confirm`` reads stdin and returns True for 'y' / 'yes'
  (case-insensitive), False for everything else (including empty
  enter — the default is N).

The helper stubs return False unconditionally so the True-case
tests fail and the False-case tests pass. Paired tests catch the
"always False" stub via the True branch's failure.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cli.generate import cli_confirm, yes_confirm
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider='openai',
        model='gpt-5.2',
        doc_types=('explanation',),
        validate=False,
        dry_run=True,
    )


# ---------------------------------------------------------------------------
# Orchestrator field
# ---------------------------------------------------------------------------


class TestConfirmCallbackField:
    def test_default_is_none(self, tmp_path: Path) -> None:
        """Without explicit assignment the field is None — that's the
        "no confirmation needed" default for tests/CI. Pins the
        attribute name and default."""
        orch = DocGenOrchestrator(_make_config(tmp_path))
        assert orch.confirm_callback is None

    def test_can_be_assigned_post_construction(
        self, tmp_path: Path,
    ) -> None:
        """The CLI wires this AFTER constructing the orchestrator
        (parses --yes, picks cli_confirm or yes_confirm). The field
        must accept reassignment.

        Bites a fix that uses ``@frozen`` instead of ``@define`` and
        forgets the field needs to be settable at runtime."""
        orch = DocGenOrchestrator(_make_config(tmp_path))

        async def my_callback(msg: str) -> bool:
            return True

        orch.confirm_callback = my_callback
        assert orch.confirm_callback is my_callback


# ---------------------------------------------------------------------------
# yes_confirm — --yes short-circuit
# ---------------------------------------------------------------------------


class TestYesConfirm:
    @pytest.mark.asyncio
    async def test_returns_true_without_consulting_stdin(self) -> None:
        """The whole point of --yes is to skip the prompt. Bites
        a fix that calls ``input()`` regardless (would hang in CI)."""
        with patch('builtins.input') as mock_input:
            assert await yes_confirm('Continue?') is True
            assert mock_input.call_count == 0


# ---------------------------------------------------------------------------
# cli_confirm — stdin-driven prompt
# ---------------------------------------------------------------------------


class TestCliConfirm:
    @pytest.mark.asyncio
    async def test_y_returns_true(self) -> None:
        """'y' is the canonical accept response."""
        with patch('builtins.input', return_value='y'):
            assert await cli_confirm('Continue?') is True

    @pytest.mark.asyncio
    async def test_yes_returns_true_case_insensitive(self) -> None:
        """'yes' / 'YES' / 'Yes' all accept. Pins case-insensitive
        matching — easy to forget; bites a fix using exact-string
        compare."""
        with patch('builtins.input', return_value='YES'):
            assert await cli_confirm('Continue?') is True

    @pytest.mark.asyncio
    async def test_n_returns_false(self) -> None:
        """'n' explicitly declines. Paired with the 'y' test."""
        with patch('builtins.input', return_value='n'):
            assert await cli_confirm('Continue?') is False

    @pytest.mark.asyncio
    async def test_empty_returns_false(self) -> None:
        """Default-N: pressing enter without typing declines, since
        the prompt suffix is ``[y/N]`` (capital N = default).

        Critical for safety: a user who pressed enter by reflex
        shouldn't end up paying for a 24h batch run."""
        with patch('builtins.input', return_value=''):
            assert await cli_confirm('Continue?') is False
