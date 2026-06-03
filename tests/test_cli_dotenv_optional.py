"""The `ariadne` CLI entry point must not hard-crash when python-dotenv
is absent from the environment running it.

`.env` loading is a convenience for picking up API keys locally. When
the bare `ariadne` entry point (e.g. a `uv tool install` shim) runs in
an environment without python-dotenv, the CLI should degrade silently
rather than raising ModuleNotFoundError before any command runs.
"""
from __future__ import annotations

import builtins

import pytest


@pytest.fixture
def _dotenv_missing(monkeypatch):
    """Simulate python-dotenv not being installed."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'dotenv' or name.startswith('dotenv.'):
            raise ImportError('No module named dotenv')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)


def test_env_loading_survives_missing_dotenv(_dotenv_missing):
    import cli.main as cli

    # Best-effort .env loading must swallow a missing dependency.
    cli._load_env()  # should not raise
