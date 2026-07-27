"""A SCIP-routed language whose index was never built used to kill the generate
path: ``extract_elements`` raised a terse ``ScipUnavailableError`` deep in the
per-file enrich loop, and — uncaught on the generate path — it took the whole
``ariadne_onboard`` tool down with an opaque message.

The extraction seam (``catalog_enrich.enrich_file``) now translates that into a
single actionable, fail-loud ``ScipIndexNotReadyError`` naming the source, the
language, the artifact, and the remedy. Crucially it fires only when a real file
of that language is actually extracted through SCIP — a source that merely
declares SCIP, or a file of a different (ast-grep) language, is unaffected.
"""
from __future__ import annotations

import pytest

from docgen.catalog_enrich import enrich_file
from docgen.scip_config import (
    ScipError,
    ScipIndexNotReadyError,
    SourceScipConfig,
)


def _scip_cfg(tmp_path):
    """A source declaring javascript -> scip whose index was never built."""
    return SourceScipConfig(
        repo='webapp',
        artifact_path=tmp_path / 'never-built' / 'index.scip',
        index_kinds={'javascript': 'scip'},
    )


def test_enrich_file_translates_missing_index_to_actionable_error(tmp_path):
    # A JS/TS file routed through SCIP whose index was never built → one clean,
    # actionable, fail-loud error instead of a raw ScipUnavailableError.
    f = tmp_path / 'app.ts'
    f.write_text('export const x = 1\n', encoding='utf-8')

    with pytest.raises(ScipIndexNotReadyError) as exc:
        enrich_file(f, source_root=tmp_path, source_config=_scip_cfg(tmp_path))

    msg = str(exc.value)
    assert 'webapp' in msg               # names the source
    assert 'javascript' in msg           # names the SCIP-routed language
    assert 'index.scip' in msg           # names the missing artifact
    assert 'ariadne index' in msg        # actionable remedy (build the index)
    assert 'ast-grep' in msg             # the alternative: drop the declaration
    assert exc.value.language == 'javascript'
    assert exc.value.reason == 'index_missing'
    # Still a ScipError subclass, so it propagates exactly like the raw error it
    # replaces (the generate path never catches ScipError → onboard fails loud).
    assert isinstance(exc.value, ScipError)


def test_enrich_file_python_unaffected_by_missing_js_index(tmp_path):
    # PRECISION: Python is extracted via ast-grep, not SCIP. A source declaring
    # javascript:scip with a missing JS index must NOT fail when enriching a
    # .py file — the missing index is irrelevant to it. (This is the false
    # positive an up-front "declared SCIP" check would have produced.)
    f = tmp_path / 'mod.py'
    f.write_text('def greet():\n    return 1\n', encoding='utf-8')

    bundle = enrich_file(f, source_root=tmp_path, source_config=_scip_cfg(tmp_path))
    assert bundle is not None  # enriched normally, no raise


def test_enrich_file_no_scip_config_uses_ast_grep(tmp_path):
    # Without a SCIP config, a .ts file falls through to ast-grep — no index is
    # read, so nothing raises (the historical un-indexed path still works).
    f = tmp_path / 'app.ts'
    f.write_text('export function greet() {}\n', encoding='utf-8')

    bundle = enrich_file(f, source_root=tmp_path, source_config=None)
    assert bundle is not None
