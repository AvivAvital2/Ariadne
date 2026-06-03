"""Vendored SCIP protobuf bindings.

Run ``python tests/setup_scip_pb2.py`` once to fetch ``scip.proto`` from
``github.com/sourcegraph/scip`` (pinned to v0.5.2) and generate
``scip_pb2.py`` via ``protoc``. Both files end up in this package.

The rest of the SCIP extractor depends on ``docgen.scip.scip_pb2`` being
importable; until you run the setup, scala/java extraction will fail
loud at import time with a clear instruction to run the setup.
"""
# Silence SyntaxWarnings from the protobuf library's runtime descriptor
# code generation. AddSerializedFile() compiles dynamic helpers via
# ``compile(..., '<unknown>', 'exec')`` and some helpers contain invalid
# escape sequences (\/, \[, \], \c) that newer Python versions warn
# about. The warnings carry filename "<unknown>", so module-scoped
# filters can't catch them after the fact — we eagerly import scip_pb2
# inside a catch_warnings context so the descriptor-build runs silently
# at package-import time. Subsequent re-imports of docgen.scip.scip_pb2
# are no-ops (already loaded), so the warnings never fire again.
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter('ignore', SyntaxWarning)
    from . import scip_pb2 as _scip_pb2  # noqa: F401  (eager-load only)

del _warnings
