"""Abandoned — alternative design for Phase 2t (process invocations).

The original tests pinned the language-agnostic
``synthesize_process_invocations`` contract — an alternative design
for Phase 2t that was abandoned in favor of the per-language
``ingest_*_process_invocations`` extractors in
``docgen/scip_process_extractor.py`` (tests in
``tests/test_scip_process_extractor.py``).

This stub keeps the file path intact while marking the module as
skipped at collection time.
"""
import pytest

pytest.skip(
    'abandoned: see docgen/scip_process_extractor.py for the live Phase 2t design',
    allow_module_level=True,
)
