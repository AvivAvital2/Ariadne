"""Pins the AriadneService singleton-reset contract between tests.

``AriadneService._instance`` is a process-wide ClassVar. Without a
between-test reset, any test that calls ``AriadneService.get()`` (or
indirectly via an @mcp.tool handler) leaks the cached instance to
subsequent tests — they see whatever ``_config`` / ``_library`` the
previous test happened to leave behind. The contract: every test
starts with ``_instance is None``.
"""
from __future__ import annotations


class TestSingletonIsolation:
    # Pollute the singleton on purpose. By itself this should be
    # harmless — but only if the next test starts with a clean
    # singleton.
    def test_a_pollutes_the_singleton(self) -> None:
        from ariadne_mcp.service import AriadneService

        svc = AriadneService.get()
        assert AriadneService._instance is svc

    # If the autouse reset fixture is in place, this test starts with
    # _instance=None regardless of test order. Without the fixture, it
    # depends on which test ran first.
    def test_b_starts_with_clean_singleton(self) -> None:
        from ariadne_mcp.service import AriadneService

        assert AriadneService._instance is None, (
            'AriadneService._instance leaked across tests; expected '
            'a between-test reset (conftest.py autouse fixture)'
        )
