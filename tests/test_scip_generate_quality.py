"""Quality tests for `generate` on Scala/Java elements (final SCIP audit).

Three fixes pinned here:
- ``_public_classes_from_bundle`` / ``_public_functions_from_bundle`` /
  ``_methods_for_class`` recognize Scala/Java subtypes (not just Python).
- ``EnrichedFileBundle.module_name`` for SCIP-extracted bundles uses the
  package (e.g. ``com.example``) not the path-derived stem.
- ``EnrichedFileBundle.imports`` is populated from ``import`` lines in
  Scala/Java source so the architecture prompt's Dependencies section
  isn't empty.

Tests must FAIL until each fix lands.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from docgen.catalog_enrich import (
    EnrichedElementInfo,
    EnrichedFileBundle,
)
from docgen.catalog_extractor import ElementInfo
from docgen.generator import DocGenerator


def _scala_element(
    qn: str, *, subtype: str, parent: str | None,
    line_start: int = 1, line_end: int = 1,
) -> EnrichedElementInfo:
    return EnrichedElementInfo(
        element=ElementInfo(
            language='scala', subtype=subtype,  # type: ignore[arg-type]
            file='X.scala',
            qualified_name=qn,
            signature=f"{subtype} {qn.rsplit('.', 1)[-1]}",
            line_start=line_start, line_end=line_end,
            col_start=0, col_end=10,
            parent_qualified_name=parent,
            body_sha='sha',
            documentation=None,
        ),
    )


# ---------------------------------------------------------------------------
# Subtype filters recognize Scala/Java
# ---------------------------------------------------------------------------


class TestPublicClassesFromBundle:
    def test_scala_class_object_trait_appear(self) -> None:
        bundle = EnrichedFileBundle(
            path=Path('X.scala'), language='scala',
            module_name='com.example',
            elements=(
                _scala_element('com.example.Foo', subtype='scala_class', parent='com.example'),
                _scala_element('com.example.Bar', subtype='scala_object', parent='com.example'),
                _scala_element('com.example.Baz', subtype='scala_trait', parent='com.example'),
            ),
        )
        names = DocGenerator._public_classes_from_bundle(bundle)
        assert set(names) == {'Foo', 'Bar', 'Baz'}

    def test_java_class_interface_enum_appear(self) -> None:
        bundle = EnrichedFileBundle(
            path=Path('X.java'), language='java',
            module_name='com.example',
            elements=(
                EnrichedElementInfo(element=ElementInfo(
                    language='java', subtype='java_class',
                    file='X.java', qualified_name='com.example.A',
                    signature='class A',
                    line_start=1, line_end=1, col_start=0, col_end=10,
                    parent_qualified_name='com.example',
                )),
                EnrichedElementInfo(element=ElementInfo(
                    language='java', subtype='java_interface',
                    file='X.java', qualified_name='com.example.B',
                    signature='interface B',
                    line_start=1, line_end=1, col_start=0, col_end=10,
                    parent_qualified_name='com.example',
                )),
                EnrichedElementInfo(element=ElementInfo(
                    language='java', subtype='java_enum',
                    file='X.java', qualified_name='com.example.C',
                    signature='enum C',
                    line_start=1, line_end=1, col_start=0, col_end=10,
                    parent_qualified_name='com.example',
                )),
            ),
        )
        names = DocGenerator._public_classes_from_bundle(bundle)
        assert set(names) == {'A', 'B', 'C'}


class TestPublicFunctionsFromBundle:
    def test_scala_top_level_def_appears(self) -> None:
        bundle = EnrichedFileBundle(
            path=Path('X.scala'), language='scala',
            module_name='com.example',
            elements=(
                _scala_element('com.example.helper', subtype='scala_def', parent='com.example'),
                _scala_element('com.example.implicitly', subtype='scala_implicit', parent='com.example'),
            ),
        )
        names = DocGenerator._public_functions_from_bundle(bundle)
        assert set(names) == {'helper', 'implicitly'}

    def test_method_under_class_excluded_from_public_functions(self) -> None:
        """A def whose parent is a class is a method, not a top-level function."""
        bundle = EnrichedFileBundle(
            path=Path('X.scala'), language='scala',
            module_name='com.example',
            elements=(
                _scala_element('com.example.Foo', subtype='scala_class', parent='com.example'),
                _scala_element('com.example.Foo.method', subtype='scala_def', parent='com.example.Foo'),
            ),
        )
        names = DocGenerator._public_functions_from_bundle(bundle)
        assert 'method' not in names


class TestMethodsForClass:
    def test_scala_methods_attributed_to_their_class(self) -> None:
        class_qn = 'com.example.Foo'
        bundle = EnrichedFileBundle(
            path=Path('X.scala'), language='scala',
            module_name='com.example',
            elements=(
                _scala_element(class_qn, subtype='scala_class', parent='com.example'),
                _scala_element(f'{class_qn}.greet', subtype='scala_def', parent=class_qn),
                _scala_element(f'{class_qn}.size', subtype='scala_def', parent=class_qn),
                _scala_element(f'{class_qn}.implicitOp', subtype='scala_implicit', parent=class_qn),
            ),
        )
        methods = DocGenerator._methods_for_class(bundle, class_qn)
        assert set(methods) == {'greet', 'size', 'implicitOp'}


# ---------------------------------------------------------------------------
# module_name derivation for SCIP-extracted bundles
# ---------------------------------------------------------------------------


class TestModuleNameDerivation:
    def test_scala_module_name_combines_package_and_file_stem(
        self, tmp_path: Path,
    ) -> None:
        """A Scala file under ``src/main/scala/com/example/Foo.scala`` must
        get ``module_name="com.example.Foo"`` (package + file stem) so
        each file in a package gets a unique identity. Without the file
        stem, multiple files in one package would all share a title and
        the library would accumulate duplicate generated docs.
        """
        from docgen.catalog_enrich import enrich_file
        from docgen.scip_config import SourceScipConfig
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
        )

        deep = tmp_path / 'src' / 'main' / 'scala' / 'com' / 'example'
        deep.mkdir(parents=True)
        f = deep / 'Foo.scala'
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
        )

        from unittest import mock
        with mock.patch(
            'docgen.scip_config.resolve_index', lambda cfg, lang: synthetic,
        ):
            cfg = SourceScipConfig(
                repo='scalaproject',
                artifact_path=tmp_path / 'idx.scip',
                index_kinds={'scala': 'scip'},
            )
            bundle = enrich_file(f, source_root=tmp_path, source_config=cfg)

        assert bundle is not None
        assert bundle.module_name == 'com.example.Foo', (
            f'expected package + file stem; got {bundle.module_name!r}'
        )


# ---------------------------------------------------------------------------
# JVM imports populated for Scala/Java
# ---------------------------------------------------------------------------


class TestJvmImports:
    def test_scala_simple_import_extracted(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'X.scala'
        f.write_text(dedent('''\
            package com.example

            import scala.collection.mutable
            import com.foo.Bar

            class X
        '''), encoding='utf-8')

        bundle = enrich_file(f, source_root=tmp_path)
        assert bundle is not None
        modules = {imp.module for imp in bundle.imports}
        assert 'scala.collection.mutable' in modules
        assert 'com.foo.Bar' in modules

    def test_scala_brace_form_extracts_package(self, tmp_path: Path) -> None:
        """`import com.foo.{Bar, Baz}` should populate the package; the
        named items are a refinement (not pinned here).
        """
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'X.scala'
        f.write_text(dedent('''\
            import com.foo.{Bar, Baz}

            class X
        '''), encoding='utf-8')

        bundle = enrich_file(f, source_root=tmp_path)
        assert bundle is not None
        modules = {imp.module for imp in bundle.imports}
        assert 'com.foo' in modules

    def test_scala_wildcard_import_extracted(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'X.scala'
        f.write_text(dedent('''\
            import com.foo._

            class X
        '''), encoding='utf-8')

        bundle = enrich_file(f, source_root=tmp_path)
        assert bundle is not None
        modules = {imp.module for imp in bundle.imports}
        assert 'com.foo' in modules

    def test_java_import_extracted(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'X.java'
        f.write_text(dedent('''\
            package com.example;

            import java.util.List;
            import com.foo.Bar;
            import static com.foo.Util.helper;

            class X {}
        '''), encoding='utf-8')

        bundle = enrich_file(f, source_root=tmp_path)
        assert bundle is not None
        modules = {imp.module for imp in bundle.imports}
        assert 'java.util.List' in modules
        assert 'com.foo.Bar' in modules
        assert 'com.foo.Util.helper' in modules
