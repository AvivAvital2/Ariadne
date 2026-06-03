"""Tests for docgen.catalog_enrich (Catalog transition Phase 2.1 + 2.2).

These cover:
- The dataclasses (PythonEnrichment, StructuredImport, EnrichedElementInfo,
  EnrichedFileBundle) — sane defaults and frozen invariants.
- enrich_file: the public entry point that combines catalog extraction
  with optional Python-specific enrichment via ast.parse.
- enrich_python_elements: the lookup-style helper that maps an
  ElementInfo's (line_start, line_end) range to its PythonEnrichment.

The contract: every field SourceAnalyzer extracted that *actually reaches
a generator prompt* must be reachable from EnrichedFileBundle/PythonEnrichment.
The plan §Spec table is the source of truth.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

# ---------------------------------------------------------------------------
# Dataclass shape
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_python_enrichment_defaults(self) -> None:
        from docgen.catalog_enrich import PythonEnrichment

        e = PythonEnrichment()
        assert e.decorators == ()
        assert e.arg_names == ()
        assert e.arg_kinds == ()
        assert e.arg_annotations == ()
        assert e.arg_defaults == ()
        assert e.return_annotation is None
        assert e.bases == ()
        assert e.docstring is None
        assert e.is_dataclass is False
        assert e.is_attrs is False
        assert e.is_abstract is False

    def test_python_enrichment_is_frozen(self) -> None:
        from docgen.catalog_enrich import PythonEnrichment

        e = PythonEnrichment()
        with pytest.raises(Exception):  # noqa: B017 attrs.exceptions.FrozenInstanceError
            e.decorators = ('foo',)  # type: ignore[misc]

    def test_structured_import_round_trip(self) -> None:
        from docgen.catalog_enrich import StructuredImport

        imp = StructuredImport(
            module='os.path',
            names=('join', 'exists'),
            is_from_import=True,
            alias=None,
            lineno=5,
        )
        assert imp.module == 'os.path'
        assert imp.is_from_import is True

    def test_enriched_file_bundle_defaults(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import EnrichedFileBundle

        bundle = EnrichedFileBundle(
            path=tmp_path / 'foo.py',
            language='python',
            module_name='foo',
        )
        assert bundle.module_docstring is None
        assert bundle.imports == ()
        assert bundle.elements == ()
        assert bundle.line_count == 0


# ---------------------------------------------------------------------------
# enrich_file — Python
# ---------------------------------------------------------------------------


def _write(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


class TestEnrichFilePython:
    def test_module_docstring_extracted(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            """Module top-level docstring."""

            def hello():
                pass
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        assert bundle.module_docstring == 'Module top-level docstring.'
        assert bundle.module_name == 'm'
        assert bundle.language == 'python'

    def test_line_count_present(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            x = 1
            y = 2
            z = 3
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        assert bundle.line_count >= 3

    def test_imports_structured(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            import os
            import json as j
            from pathlib import Path
            from typing import Literal, Iterable
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        modules = {imp.module for imp in bundle.imports}
        assert {'os', 'json', 'pathlib', 'typing'}.issubset(modules)

        # 'from typing import Literal, Iterable' — captured as is_from_import
        from_typing = next(i for i in bundle.imports if i.module == 'typing')
        assert from_typing.is_from_import is True
        assert 'Literal' in from_typing.names
        assert 'Iterable' in from_typing.names

        # 'import json as j' — captures alias
        json_imp = next(i for i in bundle.imports if i.module == 'json')
        assert json_imp.alias == 'j'
        assert json_imp.is_from_import is False

    def test_function_enrichment_decorators_and_args(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            from functools import lru_cache

            @lru_cache
            def add(a: int, b: int = 0) -> int:
                """Sum two numbers."""
                return a + b
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        # Find the add function element.
        add_el = next(
            e for e in bundle.elements
            if e.element.qualified_name.endswith('.add') or e.element.qualified_name == 'm.add'
        )
        assert add_el.python is not None
        py = add_el.python
        assert 'lru_cache' in py.decorators
        assert py.arg_names == ('a', 'b')
        assert py.arg_kinds == ('positional', 'positional')
        assert py.return_annotation == 'int'
        assert py.docstring == 'Sum two numbers.'

    def test_class_enrichment_bases_decorators_docstring(
        self, tmp_path: Path,
    ) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            from attrs import frozen
            from abc import ABC

            @frozen
            class Foo(Bar, Baz):
                """Foo class doc."""
                pass

            class Service(ABC):
                pass
        ''')

        bundle = enrich_file(f, source_root=tmp_path)

        foo = next(
            e for e in bundle.elements
            if e.element.qualified_name.endswith('.Foo')
        )
        assert foo.python is not None
        assert foo.python.bases == ('Bar', 'Baz')
        assert 'frozen' in foo.python.decorators
        assert foo.python.is_attrs is True
        assert foo.python.is_dataclass is False
        assert foo.python.is_abstract is False
        assert foo.python.docstring == 'Foo class doc.'

        svc = next(
            e for e in bundle.elements
            if e.element.qualified_name.endswith('.Service')
        )
        assert svc.python is not None
        assert 'ABC' in svc.python.bases
        assert svc.python.is_abstract is True

    def test_dataclass_detected(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            from dataclasses import dataclass

            @dataclass
            class Point:
                x: int
                y: int
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        pt = next(
            e for e in bundle.elements
            if e.element.qualified_name.endswith('.Point')
        )
        assert pt.python is not None
        assert pt.python.is_dataclass is True
        assert pt.python.is_attrs is False

    def test_async_function_enriched(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            async def fetch(url: str) -> bytes:
                """Fetch URL contents."""
                return b""
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        fetch = next(
            e for e in bundle.elements
            if e.element.qualified_name.endswith('.fetch')
        )
        assert fetch.python is not None
        assert fetch.python.docstring == 'Fetch URL contents.'
        assert fetch.python.return_annotation == 'bytes'

    def test_method_enriched_inside_class(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            class Calc:
                """Calculator."""

                @classmethod
                def add(cls, a, b):
                    """Add via classmethod."""
                    return a + b
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        add = next(
            e for e in bundle.elements
            if e.element.qualified_name.endswith('.Calc.add')
        )
        assert add.python is not None
        assert 'classmethod' in add.python.decorators
        assert add.python.docstring == 'Add via classmethod.'
        assert add.python.arg_names == ('cls', 'a', 'b')

    def test_multi_line_signature_captured(self, tmp_path: Path) -> None:
        """ElementInfo.signature historically truncated to 1 line — the
        generator's `def {sig}: ...` chunk needs the whole header up to `:`.
        Enrichment can reconstruct it from the parsed args.
        """
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        _write(f, '''
            def crunch(
                a: int,
                b: str = "x",
                *args: int,
                **kwargs: int,
            ) -> bool:
                """Multi-line signature."""
                return True
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        crunch = next(
            e for e in bundle.elements
            if e.element.qualified_name.endswith('.crunch')
        )
        assert crunch.python is not None
        assert crunch.python.arg_names == ('a', 'b', 'args', 'kwargs')
        # var_positional/var_keyword kinds preserve the meaning of */**.
        assert 'var_positional' in crunch.python.arg_kinds
        assert 'var_keyword' in crunch.python.arg_kinds


# ---------------------------------------------------------------------------
# enrich_file — non-Python (no Python enrichment, but bundle still works)
# ---------------------------------------------------------------------------


class TestEnrichFileNonPython:
    def test_javascript_no_python_enrichment(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'app.js'
        _write(f, '''
            function hello() { return 1; }
            class Greeter {}
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        assert bundle.language == 'javascript'
        assert bundle.module_docstring is None
        assert bundle.imports == ()
        # Catalog elements still extracted.
        assert any(e.element.subtype == 'js_function' for e in bundle.elements)
        # Every element on a non-Python path has no Python enrichment.
        for e in bundle.elements:
            assert e.python is None

    def test_json_bundle(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'config.json'
        f.write_text('{"name": "ariadne", "version": 1}', encoding='utf-8')

        bundle = enrich_file(f, source_root=tmp_path)
        assert bundle.language == 'json'
        assert bundle.module_docstring is None

    def test_unsupported_extension_returns_none_bundle(
        self, tmp_path: Path,
    ) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'binary.bin'
        f.write_bytes(b'\x00\x01\x02')

        bundle = enrich_file(f, source_root=tmp_path)
        # Either returns None for unsupported, or a bundle with empty elements.
        assert bundle is None or len(bundle.elements) == 0


# ---------------------------------------------------------------------------
# enrich_python_elements — direct (qualified-name lookup)
# ---------------------------------------------------------------------------


class TestEnrichPythonElementsDirect:
    """The lower-level helper that maps qualified_name → PythonEnrichment.

    Useful when callers already have ElementInfo from extract_elements and
    only want to add Python-specific data without re-extracting.
    """

    def test_returns_lookup_keyed_by_qualified_name(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_python_elements

        f = tmp_path / 'm.py'
        _write(f, '''
            def alpha(x): return x

            def beta(y): return y
        ''')
        src = f.read_text(encoding='utf-8')

        lookup = enrich_python_elements(f, src, module_name='m')

        # Every top-level function has an entry keyed by qualified_name.
        assert 'm.alpha' in lookup
        assert 'm.beta' in lookup
        assert lookup['m.alpha'].arg_names == ('x',)
        assert lookup['m.beta'].arg_names == ('y',)

    def test_methods_keyed_by_full_qualified_name(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_python_elements

        f = tmp_path / 'm.py'
        _write(f, '''
            class Outer:
                class Inner:
                    def deep(self): pass
        ''')
        src = f.read_text(encoding='utf-8')

        lookup = enrich_python_elements(f, src, module_name='m')
        # Nested classes/methods preserve their full path.
        assert 'm.Outer' in lookup
        assert 'm.Outer.Inner' in lookup
        assert 'm.Outer.Inner.deep' in lookup

    def test_no_crash_on_assignments_only(self, tmp_path: Path) -> None:
        """Variable-only files don't need Python enrichment; the helper
        returning a dict (possibly empty for variables) is fine.
        """
        from docgen.catalog_enrich import enrich_python_elements

        f = tmp_path / 'm.py'
        _write(f, '''
            X = 1
            Y = 2
        ''')
        src = f.read_text(encoding='utf-8')

        # Should not raise on a file with only assignments.
        lookup = enrich_python_elements(f, src, module_name='m')
        assert isinstance(lookup, dict)


# ---------------------------------------------------------------------------
# Integration: bundle elements line up with the catalog extractor
# ---------------------------------------------------------------------------


class TestBundleIntegration:
    def test_bundle_element_count_matches_extractor(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file
        from docgen.catalog_extractor import extract_elements

        f = tmp_path / 'm.py'
        _write(f, '''
            class A:
                def m1(self): pass
                def m2(self): pass

            def f1(): pass
            def f2(): pass
        ''')

        bundle = enrich_file(f, source_root=tmp_path)
        elements_directly = extract_elements(f, source_root=tmp_path)

        assert len(bundle.elements) == len(elements_directly)
        # Same qualified_names appear in both.
        bundle_qns = {e.element.qualified_name for e in bundle.elements}
        direct_qns = {e.qualified_name for e in elements_directly}
        assert bundle_qns == direct_qns
