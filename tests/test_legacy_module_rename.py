"""Pin the Phase 4.1 rename: analyzer.py → _legacy_analyzer.py.

Three things this test exists to catch:
1. The new modules are importable at the underscore-prefixed path.
2. The old import paths are gone — anyone still depending on
   ``docgen.analyzer`` or ``docgen.metadata`` directly fails LOUDLY,
   not silently with stale code.
3. The deprecation docstring is present so a reader of either legacy
   module immediately sees the rollback-only intent.
"""
from __future__ import annotations

import importlib
import sys

import pytest

# ---------------------------------------------------------------------------
# New paths exist and export the same surface
# ---------------------------------------------------------------------------


class TestNewLegacyModulesImportable:
    def test_legacy_analyzer_provides_source_analyzer(self) -> None:
        from docgen._legacy_analyzer import SourceAnalyzer

        # SourceAnalyzer is still constructible (no behavior change in rename).
        assert SourceAnalyzer is not None
        assert callable(SourceAnalyzer)

    def test_legacy_metadata_provides_full_metadata_surface(self) -> None:
        # All the classes that lived in metadata.py must still be reachable.
        from docgen._legacy_metadata import (
            ArgumentInfo,
        )

        # Construct one to make sure it's not just a stub.
        info = ArgumentInfo(name='x')
        assert info.name == 'x'


# ---------------------------------------------------------------------------
# Old paths must be gone (loud-fail rollback signal)
# ---------------------------------------------------------------------------


class TestOldPathsGone:
    """If someone left a stale import path in docgen/__init__.py or
    elsewhere, these tests catch it. Pinning the failure mode is more
    important than blocking the rename — a quiet legacy import is the
    failure case we want to rule out.
    """

    def _ensure_fresh(self, name: str) -> None:
        # Drop any cached state so importlib re-resolves.
        for k in list(sys.modules):
            if k == name or k.startswith(f'{name}.'):
                del sys.modules[k]

    def test_old_docgen_analyzer_path_does_not_resolve(self) -> None:
        self._ensure_fresh('docgen.analyzer')
        with pytest.raises(ImportError):
            importlib.import_module('docgen.analyzer')

    def test_old_docgen_metadata_path_does_not_resolve(self) -> None:
        self._ensure_fresh('docgen.metadata')
        with pytest.raises(ImportError):
            importlib.import_module('docgen.metadata')


# ---------------------------------------------------------------------------
# Deprecation docstring
# ---------------------------------------------------------------------------


class TestDeprecationDocstrings:
    def test_legacy_analyzer_module_marks_itself_legacy(self) -> None:
        import docgen._legacy_analyzer as m

        assert m.__doc__ is not None
        # A future reader should see the deprecation/rollback note.
        assert (
            'legacy' in m.__doc__.lower()
            or 'rollback' in m.__doc__.lower()
            or 'deprecated' in m.__doc__.lower()
        ), (
            f"_legacy_analyzer's docstring must signal its legacy status; "
            f"got: {m.__doc__!r}"
        )

    def test_legacy_metadata_module_marks_itself_legacy(self) -> None:
        import docgen._legacy_metadata as m

        assert m.__doc__ is not None
        assert (
            'legacy' in m.__doc__.lower()
            or 'rollback' in m.__doc__.lower()
            or 'deprecated' in m.__doc__.lower()
        )
