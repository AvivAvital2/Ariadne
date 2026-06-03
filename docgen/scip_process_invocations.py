"""Abandoned — Phase 2t persistence layer for the resolution-traversal design.

This module was the Phase 2t persistence layer for the abandoned
resolution-traversal design. The live implementation is in
``docgen.scip_process_extractor`` — per-language
``ingest_*_process_invocations`` functions. Verified via
``ariadne_impact_radius`` to have zero direct/transitive/test/doc
dependents before removal.

Importing this stub raises ``ImportError`` on purpose — it points
at the live replacement, rather than silently exposing a broken API
to a future caller.
"""

raise ImportError(
    'docgen.scip_process_invocations was removed (abandoned design). '
    'Use docgen.scip_process_extractor.ingest_*_process_invocations '
    'for the live Phase 2t API.',
)
