"""Tests for SCIP dispatch in ``extract_elements`` (SCIP plan, Phase B).

Pins three contracts:

1. **No-regression:** Python/HTML/JS files still go through ast-grep when
   no ``source_config`` is supplied (the path most callers exercise).
2. **Routing:** Scala/Java files go through ``scip_extractor`` when the
   source declares ``index_kinds.scala = "scip"`` (or ``.java``).
3. **Fail-loud (LOAD-BEARING):** When a source declares SCIP and the
   index is missing, ``extract_elements`` raises ``ScipUnavailableError``
   AND ``SgRoot`` is never invoked. This is the regression test that
   catches any future PR adding silent ast-grep fallback.

Tests must FAIL until ``extract_elements`` accepts a ``source_config``
kwarg and routes scala/java accordingly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from docgen.scip_config import (
    ScipUnavailableError,
    SourceScipConfig,
)

# ---------------------------------------------------------------------------
# No-regression: Python still uses ast-grep
# ---------------------------------------------------------------------------


class TestPythonUnchanged:
    def test_python_file_still_extracts_via_ast_grep(self, tmp_path: Path) -> None:
        from docgen.catalog_extractor import extract_elements

        f = tmp_path / 'm.py'
        f.write_text('def foo(): pass\n', encoding='utf-8')

        # No source_config → behaves exactly as before.
        elements = extract_elements(f, source_root=tmp_path)
        assert any(e.qualified_name.endswith('.foo') for e in elements)

    def test_python_with_unrelated_source_config_still_works(
        self, tmp_path: Path,
    ) -> None:
        """A source_config that declares SCIP for scala/java must not
        affect Python extraction — that's a different language.
        """
        from docgen.catalog_extractor import extract_elements

        f = tmp_path / 'm.py'
        f.write_text('def foo(): pass\n', encoding='utf-8')

        cfg = SourceScipConfig(
            repo='scalaproject',
            artifact_path=tmp_path / 'missing.scip',
            index_kinds={'scala': 'scip', 'java': 'scip'},
        )
        elements = extract_elements(f, source_root=tmp_path, source_config=cfg)
        assert any(e.qualified_name.endswith('.foo') for e in elements)


# ---------------------------------------------------------------------------
# Scala without SCIP declared → no extraction (return [])
# ---------------------------------------------------------------------------


class TestScalaNoConfig:
    def test_scala_file_with_no_source_config_returns_empty(
        self, tmp_path: Path,
    ) -> None:
        """Stray .scala file in a source that doesn't declare SCIP gets
        no ElementInfo records — we don't try ast-grep silently.
        """
        from docgen.catalog_extractor import extract_elements

        f = tmp_path / 'X.scala'
        f.write_text('class X\n', encoding='utf-8')

        elements = extract_elements(f, source_root=tmp_path)
        assert elements == []


# ---------------------------------------------------------------------------
# Scala with SCIP declared → routes to scip_extractor
# ---------------------------------------------------------------------------


class TestScalaWithScip:
    def test_routes_to_scip_extractor_when_declared(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When ``index_kinds.scala = "scip"`` and the index resolves,
        extract_elements pulls from scip_extractor — NOT from ast-grep.
        """
        from docgen import catalog_extractor as ce
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
        )

        f = tmp_path / 'src' / 'main' / 'scala' / 'Foo.scala'
        f.parent.mkdir(parents=True)
        f.write_text('class Foo\n', encoding='utf-8')

        sym = 'scip-java maven g a 1 com/example/Foo#'
        synthetic = ScipIndex(
            documents=(_ScipDoc(
                relative_path=str(f.relative_to(tmp_path).as_posix()),
                occurrences=(_ScipOccurrence(
                    symbol=sym, range=(0, 0, 0, 5), is_definition=True,
                ),),
                symbols=(_ScipSymbol(symbol=sym, kind='Class'),),
            ),),
            source_root=tmp_path,
        )

        # resolve_index returns our synthetic index — bypasses load() which
        # would need a real .scip file.
        def fake_resolve(cfg, lang):
            assert lang == 'scala'
            return synthetic

        monkeypatch.setattr('docgen.scip_config.resolve_index', fake_resolve)

        cfg = SourceScipConfig(
            repo='scalaproject',
            artifact_path=tmp_path / 'idx.scip',
            index_kinds={'scala': 'scip'},
        )
        elements = extract_elements_call(ce, f, tmp_path, cfg)

        assert len(elements) == 1
        assert elements[0].qualified_name == 'com.example.Foo'
        assert elements[0].subtype == 'scala_class'


def extract_elements_call(ce, file: Path, root: Path, cfg: SourceScipConfig):
    """Tiny wrapper that calls extract_elements with kwargs — kept here so
    the test class is decoupled from the exact signature.
    """
    return ce.extract_elements(file, source_root=root, source_config=cfg)


# ---------------------------------------------------------------------------
# THE TRIPWIRE — declared SCIP + missing index must NEVER call SgRoot
# ---------------------------------------------------------------------------


class TestFailLoudContract:
    def test_declared_scip_missing_index_raises_and_does_not_call_sgroot(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The single most important regression test for this whole
        integration. If a future PR adds a silent ast-grep fallback when
        SCIP fails, this test catches it.

        Plan §F: "test_declared_scip_never_silently_falls_back — it
        monkeypatches SgRoot and asserts it is never called when
        index_kinds.scala = 'scip' and the index is missing."
        """
        from docgen import catalog_extractor as ce

        f = tmp_path / 'src' / 'main' / 'scala' / 'X.scala'
        f.parent.mkdir(parents=True)
        f.write_text('class X\n', encoding='utf-8')

        sg_called = {'hit': False}

        def boom(*args, **kwargs):
            sg_called['hit'] = True
            raise AssertionError(
                'ast-grep MUST NOT be invoked for declared-SCIP files'
            )

        # Monkeypatch SgRoot so any ast-grep parse attempt explodes.
        monkeypatch.setattr('docgen.catalog_extractor.SgRoot', boom)

        cfg = SourceScipConfig(
            repo='scalaproject',
            artifact_path=tmp_path / 'missing.scip',  # does NOT exist
            index_kinds={'scala': 'scip'},
            allow_degraded=False,
        )

        with pytest.raises(ScipUnavailableError) as exc:
            ce.extract_elements(f, source_root=tmp_path, source_config=cfg)

        assert exc.value.repo == 'scalaproject'
        assert exc.value.reason == 'index_missing'
        assert sg_called['hit'] is False, (
            'SgRoot was called — silent ast-grep fallback is a regression'
        )

    def test_declared_scip_too_stale_raises_too_stale(
        self, tmp_path: Path,
    ) -> None:
        """Stale-index path also fails loud, with the specific subclass."""
        import os
        import time

        from docgen import catalog_extractor as ce
        from docgen.scip_config import ScipTooStaleError

        f = tmp_path / 'src' / 'main' / 'scala' / 'X.scala'
        f.parent.mkdir(parents=True)
        f.write_text('class X\n', encoding='utf-8')

        artifact = tmp_path / 'old.scip'
        artifact.write_bytes(b'')
        old = time.time() - 60 * 86400
        os.utime(artifact, (old, old))

        cfg = SourceScipConfig(
            repo='scalaproject',
            artifact_path=artifact,
            index_kinds={'scala': 'scip'},
            max_staleness_days=7,
        )

        with pytest.raises(ScipTooStaleError):
            ce.extract_elements(f, source_root=tmp_path, source_config=cfg)


# ---------------------------------------------------------------------------
# JavaScript / TypeScript dispatch — opts in via index_kinds.javascript
# ---------------------------------------------------------------------------


class TestJavascriptWithoutScip:
    def test_js_file_with_no_source_config_still_uses_ast_grep(
        self, tmp_path: Path,
    ) -> None:
        """No source_config → ast-grep extraction (the historical
        path). Backwards-compat guard for projects that haven't opted
        into scip-typescript routing."""
        from docgen.catalog_extractor import extract_elements

        f = tmp_path / 'app.ts'
        f.write_text('export function greet() {}\n', encoding='utf-8')

        elements = extract_elements(f, source_root=tmp_path)
        assert any(e.qualified_name.endswith('.greet') for e in elements)

    def test_js_file_with_scala_only_scip_falls_through_to_ast_grep(
        self, tmp_path: Path,
    ) -> None:
        """A source declaring SCIP for scala only — JS files in that
        same source still use ast-grep because index_kinds.javascript
        wasn't set."""
        from docgen.catalog_extractor import extract_elements

        f = tmp_path / 'app.ts'
        f.write_text('export function greet() {}\n', encoding='utf-8')

        cfg = SourceScipConfig(
            repo='polyglot',
            artifact_path=tmp_path / 'missing.scip',
            index_kinds={'scala': 'scip'},  # no 'javascript' key
        )
        elements = extract_elements(
            f, source_root=tmp_path, source_config=cfg,
        )
        assert any(e.qualified_name.endswith('.greet') for e in elements)


class TestJavascriptWithScip:
    def test_routes_through_scip_extractor_when_declared(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When ``index_kinds.javascript = "scip"`` and the index
        resolves, JS extraction pulls qualified_names from
        scip_extractor — NOT ast-grep. The match between catalog
        qualified_names and library_scip rows is exactly what
        Phase 2 Change 2's architecture-doc cross-source rendering
        depends on."""
        from docgen import catalog_extractor as ce
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
        )

        f = tmp_path / 'src' / 'utils' / 'helper.ts'
        f.parent.mkdir(parents=True)
        f.write_text('export class Helper {}\n', encoding='utf-8')

        sym = (
            'scip-typescript npm webapp 0.1 '
            'src/utils/`helper.ts`/Helper#'
        )
        synthetic = ScipIndex(
            documents=(_ScipDoc(
                relative_path=str(f.relative_to(tmp_path).as_posix()),
                occurrences=(_ScipOccurrence(
                    symbol=sym, range=(0, 13, 0, 19), is_definition=True,
                ),),
                symbols=(_ScipSymbol(
                    symbol=sym, kind='Class', display_name='Helper',
                ),),
            ),),
            source_root=tmp_path,
        )

        def fake_resolve(cfg, lang):
            assert lang == 'javascript'
            return synthetic

        monkeypatch.setattr(
            'docgen.scip_config.resolve_index', fake_resolve,
        )

        cfg = SourceScipConfig(
            repo='webapp',
            artifact_path=tmp_path / 'idx.scip',
            index_kinds={'javascript': 'scip'},
        )
        elements = ce.extract_elements(
            f, source_root=tmp_path, source_config=cfg,
        )

        assert len(elements) == 1
        # qualified_name from scip-typescript descriptors, not the
        # ast-grep ``src.utils.helper.Helper`` dotted-path form.
        assert 'Helper' in elements[0].qualified_name
        assert elements[0].subtype.startswith('js_') or elements[0].subtype == 'class'


class TestJavascriptFailLoud:
    def test_declared_scip_missing_index_raises_no_ast_grep_fallback(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The mirror of the Scala tripwire: declared SCIP + missing
        index must raise ``ScipUnavailableError`` AND ``SgRoot`` must
        never be invoked. If a future change adds silent ast-grep
        fallback when scip-typescript output is missing, this test
        catches it."""
        from docgen import catalog_extractor as ce

        f = tmp_path / 'src' / 'app.ts'
        f.parent.mkdir(parents=True)
        f.write_text('export const x = 1\n', encoding='utf-8')

        sg_called = {'hit': False}

        def boom(*args, **kwargs):
            sg_called['hit'] = True
            raise AssertionError(
                'ast-grep MUST NOT be invoked for declared-SCIP JS files'
            )

        monkeypatch.setattr('docgen.catalog_extractor.SgRoot', boom)

        cfg = SourceScipConfig(
            repo='webapp',
            artifact_path=tmp_path / 'missing.scip',
            index_kinds={'javascript': 'scip'},
            allow_degraded=False,
        )

        with pytest.raises(ScipUnavailableError) as exc:
            ce.extract_elements(
                f, source_root=tmp_path, source_config=cfg,
            )

        assert exc.value.repo == 'webapp'
        assert exc.value.reason == 'index_missing'
        assert sg_called['hit'] is False, (
            'SgRoot was called — silent ast-grep fallback is a regression'
        )
