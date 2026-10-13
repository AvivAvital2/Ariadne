"""Pytest configuration: cross-test isolation.

Right now this only resets the ``AriadneService`` singleton between
tests. Without the reset, any test that calls ``AriadneService.get()``
(or invokes an @mcp.tool handler) caches the singleton on the class,
and subsequent tests inherit whatever ``_config`` / ``_library`` /
``_embedding_service`` the previous one happened to leave behind.
That's the difference between "tests pass in isolation" and "tests
pass in the full suite": invisible cross-test state.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_cli_config_aliases(monkeypatch):
    """An import inside a patched get_config must not leak that mock to later tests."""
    import sys

    import config

    for name, module in list(sys.modules.items()):
        if name.startswith('cli.') and hasattr(module, 'get_config'):
            monkeypatch.setattr(module, 'get_config', config.get_config)


@pytest.fixture(autouse=True)
def _reset_ariadne_service_singleton():
    """Clear the AriadneService class-level singleton before AND after
    each test. Setup-side reset clears whatever a prior test leaked;
    teardown-side reset keeps the next test's start state predictable
    even if pytest aborts mid-fixture.
    """
    from ariadne_mcp.service import AriadneService

    AriadneService._instance = None
    yield
    AriadneService._instance = None


@pytest.fixture
def local_deployment(tmp_path, monkeypatch):
    """Two real loopback endpoints and a fresh bound database pair."""
    from inference import InferenceContext
    from model_config import resolve_models
    from tests.local_model_server import deployments, local_config, trap_inference

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        ctx = InferenceContext(
            resolve_models(local_config(chat.base_url, embed.base_url)),
            tmp_path / 'local.db',
            tmp_path / 'local_staleness.db',
        )
        with ctx.write_session():
            ctx.open_library(operation='write').close()
        yield ctx, chat, embed
