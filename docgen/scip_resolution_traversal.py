"""Abandoned — alternative design for Phase 2s (resolution traversal).

This module is an abandoned alternative design for Phase 2s
(resolution traversal). The live implementation is
``docgen.scip_resolution.resolve_arg_value``. Verified via
``ariadne_impact_radius`` to have zero direct/transitive/test/doc
dependents before removal.

Importing this stub raises ``ImportError`` on purpose — it points
at the live replacement, rather than silently exposing a broken API
to a future caller.
"""

raise ImportError(
    'docgen.scip_resolution_traversal was removed (abandoned design). '
    'Use docgen.scip_resolution.resolve_arg_value for the live '
    'Phase 2s API.',
)
