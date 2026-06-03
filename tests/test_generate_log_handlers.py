"""``cmd_generate`` must not leak or stack root-logger handlers.

It attaches a per-run ``ariadne_runs/generate-*.log`` file handler to the
ROOT logger. Previously that handler was never removed, so:
  - repeated calls in one process stacked handlers → duplicate log lines;
  - in the test suite, a leaked handler captured *other* tests' logs and
    wrote them (with now-deleted tmp paths) into the repo's ariadne_runs/.

Contract: after ``cmd_generate`` returns, the root logger has exactly the
handlers it had before — and running it twice doesn't accumulate any.
"""
from __future__ import annotations

import argparse
import asyncio
import logging

import pytest


@pytest.fixture(autouse=True)
def _test_config(monkeypatch, tmp_path):
    from tests._scoped_config_fixture import install_test_config
    # No default_source → cmd_generate returns early, but still passes
    # through the run-log handler setup/teardown.
    install_test_config(monkeypatch, tmp_path, 'unused_src')


def _args(tmp_path):
    return argparse.Namespace(
        source=None,  # → no source resolved → early return inside the body
        db=str(tmp_path / 'library.db'),
        verbose=False,
    )


def test_cmd_generate_does_not_leak_or_stack_handlers(monkeypatch, tmp_path):
    from cli.generate import cmd_generate

    root = logging.getLogger()
    before = list(root.handlers)

    rc1 = asyncio.run(cmd_generate(_args(tmp_path)))
    after_one = list(root.handlers)
    assert after_one == before, (
        f'cmd_generate leaked handlers: {set(map(id, after_one)) - set(map(id, before))}'
    )

    # Run again — must not stack a second run-log handler.
    asyncio.run(cmd_generate(_args(tmp_path)))
    after_two = list(root.handlers)
    assert after_two == before, (
        'second cmd_generate stacked/leaked handlers'
    )
    # Sanity: the early-return path was exercised.
    assert rc1 == 1
