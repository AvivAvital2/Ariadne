"""Abandoned — alternative design for Phase 2s (resolution traversal).

The original tests pinned contracts of an alternative design for
Phase 2s (resolution traversal) that was abandoned mid-flight. The
replacement design lives in ``docgen/scip_resolution.py`` (tests in
``tests/test_scip_resolution.py``).

This stub keeps the file path intact so any tooling that knows about
it doesn't break, while marking the module as skipped at collection
time.
"""
import pytest

pytest.skip(
    'abandoned: see docgen/scip_resolution.py for the live Phase 2s design',
    allow_module_level=True,
)
