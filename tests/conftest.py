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
