"""Tests for the SCIP extractor + index loader (SCIP plan, Phase A.5).

The extractor turns SCIP-shaped data (synthetic in these tests; real
protobuf in production) into ``ElementInfo`` lists. Tests target the
load contract (missing/stale → specific ScipError subclass), the
subtype dispatch table, the qualified-name composition, and
documentation enrichment via the doc parser.

Tests construct synthetic SCIP data directly via the extractor's small
duck-typed dataclasses (``_ScipDoc``, ``_ScipSymbol``, ``_ScipOccurrence``)
so the protobuf bindings are NOT required to run this suite.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from docgen.scip_config import (
    ScipCorruptError,
    ScipTooStaleError,
    ScipUnavailableError,
)

# ---------------------------------------------------------------------------
# ScipIndex.load — error contract (no protobuf needed for these paths)
# ---------------------------------------------------------------------------


class TestScipIndexLoadErrors:
    def test_missing_file_raises_unavailable(self, tmp_path: Path) -> None:
        from docgen.scip_extractor import ScipIndex

        with pytest.raises(ScipUnavailableError) as exc:
            ScipIndex.load(
                tmp_path / 'nope.scip',
                repo='scalaproject',
                max_staleness_days=7,
            )
        assert exc.value.repo == 'scalaproject'
        assert exc.value.reason == 'index_missing'

    def test_stale_file_raises_too_stale(self, tmp_path: Path) -> None:
        from docgen.scip_extractor import ScipIndex

        # Write any bytes — the staleness check must fire before parse.
        artifact = tmp_path / 'old.scip'
        artifact.write_bytes(b'')
        # Set mtime well past the staleness window.
        old_ts = time.time() - 60 * 86400  # 60 days ago
        os.utime(artifact, (old_ts, old_ts))

        with pytest.raises(ScipTooStaleError) as exc:
            ScipIndex.load(artifact, repo='scalaproject', max_staleness_days=7)
        assert exc.value.reason == 'index_too_stale'
        assert exc.value.last_good_age_days is not None
        assert exc.value.last_good_age_days >= 7

    def test_fresh_file_passes_staleness_check(self, tmp_path: Path) -> None:
        """A fresh-but-corrupt file should NOT raise ScipTooStaleError —
        the staleness check has to come before parse.
        """
        from docgen.scip_extractor import ScipIndex

        artifact = tmp_path / 'fresh-but-bad.scip'
        artifact.write_bytes(b'\x00\x01\x02\x03')  # invalid protobuf

        # Either ScipCorruptError OR a successful empty-ish load is fine —
        # what we reject is misclassifying a fresh file as stale.
        try:
            ScipIndex.load(artifact, repo='scalaproject', max_staleness_days=7)
        except ScipTooStaleError:
            pytest.fail('fresh file must not raise ScipTooStaleError')
        except ScipCorruptError:
            pass  # acceptable — we wanted a non-stale failure mode
        except ScipUnavailableError:
            pytest.fail('fresh existing file must not raise Unavailable')


# ---------------------------------------------------------------------------
# document_for — relative-path lookup
# ---------------------------------------------------------------------------


class TestDocumentForCallerSourceRoot:
    """The artifact path (``scalaproject/target/index.scip``) is NOT the
    same as the project root (``scalaproject/``). scip-java records
    relative paths from the project root. ``document_for`` must look
    up using the caller's source_root, not the artifact's directory.
    """

    def test_artifact_in_subdirectory_resolves_via_caller_source_root(
        self, tmp_path,
    ) -> None:
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        # Scalaproject-style layout: project root has subdirs and a target/
        # subdirectory holding the artifact.
        project = tmp_path / 'project'
        scala_src = project / 'engine' / 'src' / 'main' / 'scala'
        scala_src.mkdir(parents=True)
        f = scala_src / 'Foo.scala'
        f.write_text('class Foo\n', encoding='utf-8')

        sym = 'scip-java maven g a 1 com/example/Foo#'
        # SCIP records relative_path from project root.
        scip_doc = _ScipDoc(
            relative_path='engine/src/main/scala/Foo.scala',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 0, 5), is_definition=True,
            ),),
            symbols=(_ScipSymbol(symbol=sym, kind='Class'),),
        )

        # Construct ScipIndex without a source_root; the lookup must use
        # the caller-supplied source_root (the project root).
        index = ScipIndex(documents=(scip_doc,))
        elements = extract(f, source_root=project, index=index)

        assert len(elements) == 1, (
            'extract returned no elements — document_for is using the '
            'wrong source_root for relative-path lookups'
        )
        assert elements[0].qualified_name == 'com.example.Foo'


class TestDocumentFor:
    def test_lookup_by_relative_path(self) -> None:
        """A ScipIndex constructed directly (bypassing load) routes
        document_for() lookups by the document's relative_path.
        """
        from docgen.scip_extractor import ScipIndex, _ScipDoc

        doc = _ScipDoc(
            relative_path='src/main/scala/Foo.scala',
            occurrences=(),
            symbols=(),
        )
        index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

        result = index.document_for(Path('/repo/src/main/scala/Foo.scala'))
        assert result is doc

    def test_unknown_file_returns_none(self) -> None:
        from docgen.scip_extractor import ScipIndex

        index = ScipIndex(documents=(), source_root=Path('/repo'))
        assert index.document_for(Path('/repo/nope.scala')) is None


# ---------------------------------------------------------------------------
# extract — happy path (Scala class + method)
# ---------------------------------------------------------------------------


class TestExtractScala:
    def _build_index(self) -> object:
        """Synthetic SCIP data for: ``com.example.Foo`` (class) with
        method ``com.example.Foo.bar()``.
        """
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
        )

        cls_sym = 'scip-java maven g a 1 com/example/Foo#'
        m_sym = 'scip-java maven g a 1 com/example/Foo#bar().'
        doc = _ScipDoc(
            relative_path='src/main/scala/com/example/Foo.scala',
            occurrences=(
                _ScipOccurrence(
                    symbol=cls_sym, range=(0, 0, 0, 12), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=m_sym, range=(2, 2, 4, 0), is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(symbol=cls_sym, kind='Class'),
                _ScipSymbol(symbol=m_sym, kind='Method'),
            ),
        )
        return ScipIndex(documents=(doc,), source_root=Path('/repo'))

    def test_class_emitted_with_scala_class_subtype(self) -> None:
        from docgen.scip_extractor import extract

        index = self._build_index()
        elements = extract(
            Path('/repo/src/main/scala/com/example/Foo.scala'),
            source_root=Path('/repo'),
            index=index,
        )
        cls = next(e for e in elements if e.qualified_name == 'com.example.Foo')
        assert cls.subtype == 'scala_class'
        assert cls.language == 'scala'

    def test_method_qualified_name_for_scala(self) -> None:
        """Scala does NOT decode the method's JVM signature — qn ends in
        ``bar``, not ``bar()``.
        """
        from docgen.scip_extractor import extract

        index = self._build_index()
        elements = extract(
            Path('/repo/src/main/scala/com/example/Foo.scala'),
            source_root=Path('/repo'),
            index=index,
        )
        m = next(e for e in elements if e.qualified_name.endswith('.bar'))
        assert m.subtype == 'scala_def'
        assert m.parent_qualified_name == 'com.example.Foo'

    def test_lines_are_one_indexed(self) -> None:
        """SCIP ranges are 0-indexed; ElementInfo lines are 1-indexed.
        A symbol at proto line 0 must surface as line_start=1.
        """
        from docgen.scip_extractor import extract

        index = self._build_index()
        elements = extract(
            Path('/repo/src/main/scala/com/example/Foo.scala'),
            source_root=Path('/repo'),
            index=index,
        )
        cls = next(e for e in elements if e.qualified_name == 'com.example.Foo')
        assert cls.line_start == 1
        assert cls.line_end == 1


# ---------------------------------------------------------------------------
# extract — Java overload disambiguation
# ---------------------------------------------------------------------------


class TestExtractJava:
    def test_two_overloads_distinct_qns(self) -> None:
        """The whole point of decoding the JVM disambiguator: two
        ``bar`` methods with different signatures must round-trip to
        distinct qualified_names.
        """
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym_a = 'scip-java maven g a 1 com/example/Foo#bar(I).'
        sym_b = 'scip-java maven g a 1 com/example/Foo#bar(Ljava/lang/String;).'
        doc = _ScipDoc(
            relative_path='src/main/java/com/example/Foo.java',
            occurrences=(
                _ScipOccurrence(symbol=sym_a, range=(0, 0, 0, 1), is_definition=True),
                _ScipOccurrence(symbol=sym_b, range=(1, 0, 1, 1), is_definition=True),
            ),
            symbols=(
                _ScipSymbol(symbol=sym_a, kind='Method'),
                _ScipSymbol(symbol=sym_b, kind='Method'),
            ),
        )
        index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

        elements = extract(
            Path('/repo/src/main/java/com/example/Foo.java'),
            source_root=Path('/repo'),
            index=index,
        )
        qns = {e.qualified_name for e in elements}
        assert 'com.example.Foo.bar(int)' in qns
        assert 'com.example.Foo.bar(java.lang.String)' in qns

    def test_constructor_subtype(self) -> None:
        """Java constructor (display_name='<init>') gets java_constructor subtype."""
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym = 'scip-java maven g a 1 com/example/Foo#`<init>`().'
        doc = _ScipDoc(
            relative_path='src/main/java/com/example/Foo.java',
            occurrences=(
                _ScipOccurrence(symbol=sym, range=(0, 0, 0, 1), is_definition=True),
            ),
            symbols=(
                _ScipSymbol(symbol=sym, kind='Method', display_name='<init>'),
            ),
        )
        index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

        elements = extract(
            Path('/repo/src/main/java/com/example/Foo.java'),
            source_root=Path('/repo'),
            index=index,
        )
        ctor = next(iter(elements))
        assert ctor.subtype == 'java_constructor'


# ---------------------------------------------------------------------------
# Subtype dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'kind,is_implicit,is_var,language,expected_subtype',
    [
        ('Class', False, False, 'scala', 'scala_class'),
        ('Object', False, False, 'scala', 'scala_object'),
        ('Trait', False, False, 'scala', 'scala_trait'),
        ('Method', True, False, 'scala', 'scala_implicit'),
        ('Method', False, False, 'scala', 'scala_def'),
        ('Field', False, True, 'scala', 'scala_var'),
        ('Field', False, False, 'scala', 'scala_val'),
        ('TypeAlias', False, False, 'scala', 'scala_type'),
        ('PackageObject', False, False, 'scala', 'scala_package_object'),
        ('Class', False, False, 'java', 'java_class'),
        ('Interface', False, False, 'java', 'java_interface'),
        ('Enum', False, False, 'java', 'java_enum'),
        ('Method', False, False, 'java', 'java_method'),
        ('Field', False, False, 'java', 'java_field'),
    ],
)
def test_subtype_dispatch(
    kind: str, is_implicit: bool, is_var: bool, language: str, expected_subtype: str,
) -> None:
    """Every row in SCIP plan §4.2's table must map correctly."""
    from docgen.scip_extractor import (
        ScipIndex,
        _ScipDoc,
        _ScipOccurrence,
        _ScipSymbol,
        extract,
    )

    sym = 'scip-java maven g a 1 com/example/Foo#'
    if kind == 'Method':
        sym = 'scip-java maven g a 1 com/example/Foo#bar().'
    elif kind == 'Field':
        sym = 'scip-java maven g a 1 com/example/Foo#x.'
    elif kind == 'TypeAlias':
        sym = 'scip-java maven g a 1 com/example/MyAlias:'

    rel = 'src/main/scala/X.scala' if language == 'scala' else 'src/main/java/X.java'
    doc = _ScipDoc(
        relative_path=rel,
        occurrences=(
            _ScipOccurrence(symbol=sym, range=(0, 0, 0, 1), is_definition=True),
        ),
        symbols=(
            _ScipSymbol(
                symbol=sym, kind=kind,
                is_implicit=is_implicit, is_var=is_var,
            ),
        ),
    )
    index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

    elements = extract(Path('/repo') / rel, source_root=Path('/repo'), index=index)
    assert len(elements) == 1
    assert elements[0].subtype == expected_subtype


# ---------------------------------------------------------------------------
# Symbols WITHOUT a Definition occurrence are skipped
# ---------------------------------------------------------------------------


class TestNonDefinitionFilter:
    def test_reference_only_symbol_not_emitted(self) -> None:
        """A symbol that only has reference (non-definition) occurrences
        must not surface as an ElementInfo — those are someone else's
        definitions in this file.
        """
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym = 'scip-java maven g a 1 com/example/External#'
        doc = _ScipDoc(
            relative_path='src/main/scala/Foo.scala',
            occurrences=(
                _ScipOccurrence(symbol=sym, range=(0, 0, 0, 1), is_definition=False),
            ),
            symbols=(_ScipSymbol(symbol=sym, kind='Class'),),
        )
        index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

        elements = extract(
            Path('/repo/src/main/scala/Foo.scala'),
            source_root=Path('/repo'),
            index=index,
        )
        assert elements == []


# ---------------------------------------------------------------------------
# Documentation enrichment
# ---------------------------------------------------------------------------


class TestSignaturePopulation:
    """``ElementInfo.signature`` should be populated from
    ``SymbolInformation.signature_documentation.text`` when present, else
    from the first non-blank line of the file slice for the symbol's
    definition range. Empty signature is a quality regression — pin it.
    """

    def test_signature_text_first_line_used_when_present(
        self, tmp_path,
    ) -> None:
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        # File doesn't need to exist on disk for this branch — signature
        # comes from the SCIP signature_text directly.
        sym = 'scip-java maven g a 1 com/example/Foo#bar().'
        doc = _ScipDoc(
            relative_path='x.scala',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(2, 0, 4, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method',
                signature_text='def bar(x: Int): String\n  body...',
            ),),
        )
        f = tmp_path / 'x.scala'
        f.write_text('class Foo {\n  // padding\n  def bar(x: Int): String = ???\n}\n', encoding='utf-8')
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        assert elements[0].signature == 'def bar(x: Int): String'

    def test_falls_back_to_file_slice_when_signature_text_missing(
        self, tmp_path,
    ) -> None:
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        f = tmp_path / 'y.scala'
        # The class definition starts at 1-indexed line 1 (range start = 0
        # 0-indexed → +1).
        f.write_text('class Greeter(name: String) extends Speaker {\n  // body\n}\n', encoding='utf-8')

        sym = 'scip-java maven g a 1 com/example/Greeter#'
        doc = _ScipDoc(
            relative_path='y.scala',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 2, 1), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Class', signature_text='',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        sig = elements[0].signature
        # First line of the definition slice — minus trailing brace.
        assert 'class Greeter' in sig
        assert 'Speaker' in sig

    def test_signature_is_empty_string_only_when_both_sources_empty(
        self, tmp_path,
    ) -> None:
        """Defensive: an unreadable file + no signature_text yields empty
        string (not None, not crash).
        """
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        # No file on disk — read will fail.
        f = tmp_path / 'missing.scala'

        sym = 'scip-java maven g a 1 com/example/Z#'
        doc = _ScipDoc(
            relative_path='missing.scala',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 0, 1), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Class', signature_text='',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        assert elements[0].signature == ''


class TestBodyShaPopulation:
    """``ElementInfo.body_sha`` is what catalog_writer's incremental
    update uses to detect whether an element's body changed between syncs.
    With an empty string for every SCIP-extracted element, edits to Scala
    files would be classified as "unchanged" (old "" matches new "").
    Body SHA must reflect the file slice for the element's range.
    """

    def test_two_different_bodies_produce_different_shas(
        self, tmp_path,
    ) -> None:
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym = 'scip-java maven g a 1 com/example/Foo#bar().'
        doc = _ScipDoc(
            relative_path='m.scala',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 1, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(symbol=sym, kind='Method'),),
        )

        f1 = tmp_path / 'v1.scala'
        f1.write_text('def bar(): Int = 42\n', encoding='utf-8')
        idx1 = ScipIndex(documents=(doc,), source_root=tmp_path)
        # Patch document_for to use this filename
        doc1 = _ScipDoc(
            relative_path=f1.name,
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 1, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(symbol=sym, kind='Method'),),
        )
        idx1 = ScipIndex(documents=(doc1,), source_root=tmp_path)
        sha_v1 = extract(f1, source_root=tmp_path, index=idx1)[0].body_sha

        f2 = tmp_path / 'v2.scala'
        f2.write_text('def bar(): Int = 999\n', encoding='utf-8')
        doc2 = _ScipDoc(
            relative_path=f2.name,
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 1, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(symbol=sym, kind='Method'),),
        )
        idx2 = ScipIndex(documents=(doc2,), source_root=tmp_path)
        sha_v2 = extract(f2, source_root=tmp_path, index=idx2)[0].body_sha

        assert sha_v1 != '', (
            "body_sha is empty — edits to Scala source would silently "
            "appear as 'unchanged' in incremental catalog sync"
        )
        assert sha_v1 != sha_v2, (
            f'different bodies produced the same sha: {sha_v1!r}'
        )

    def test_same_body_produces_same_sha(self, tmp_path) -> None:
        """Stability: identical file slice → identical sha across two runs."""
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym = 'scip-java maven g a 1 com/example/X#'
        f = tmp_path / 'stable.scala'
        f.write_text('class X { def hi() = 1 }\n', encoding='utf-8')

        def make_index():
            return ScipIndex(documents=(_ScipDoc(
                relative_path='stable.scala',
                occurrences=(_ScipOccurrence(
                    symbol=sym, range=(0, 0, 0, 24), is_definition=True,
                ),),
                symbols=(_ScipSymbol(symbol=sym, kind='Class'),),
            ),), source_root=tmp_path)

        sha_a = extract(f, source_root=tmp_path, index=make_index())[0].body_sha
        sha_b = extract(f, source_root=tmp_path, index=make_index())[0].body_sha
        assert sha_a == sha_b


class TestDocumentationEnrichment:
    def test_no_doc_means_documentation_is_none(self) -> None:
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym = 'scip-java maven g a 1 com/example/Foo#'
        doc = _ScipDoc(
            relative_path='src/main/scala/Foo.scala',
            occurrences=(
                _ScipOccurrence(symbol=sym, range=(0, 0, 0, 1), is_definition=True),
            ),
            symbols=(_ScipSymbol(symbol=sym, kind='Class', documentation=''),),
        )
        index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

        elements = extract(
            Path('/repo/src/main/scala/Foo.scala'),
            source_root=Path('/repo'),
            index=index,
        )
        assert elements[0].documentation is None

    def test_scaladoc_populates_structured_documentation(self) -> None:
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym = 'scip-java maven g a 1 com/example/Foo#bar().'
        doc_str = 'Computes the result.\n\n@param x The input.\n@return The output.'
        doc = _ScipDoc(
            relative_path='src/main/scala/Foo.scala',
            occurrences=(
                _ScipOccurrence(symbol=sym, range=(0, 0, 0, 1), is_definition=True),
            ),
            symbols=(
                _ScipSymbol(symbol=sym, kind='Method', documentation=doc_str),
            ),
        )
        index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

        elements = extract(
            Path('/repo/src/main/scala/Foo.scala'),
            source_root=Path('/repo'),
            index=index,
        )
        info = elements[0].documentation
        assert info is not None
        assert info['summary'] == 'Computes the result.'
        assert info['params'] == {'x': 'The input.'}
        assert info['returns'] == 'The output.'

    def test_javadoc_uses_javadoc_grammar(self) -> None:
        """For a .java file, the doc must be parsed as Javadoc — so
        ``@exception`` aliases ``@throws``.
        """
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym = 'scip-java maven g a 1 com/example/Foo#bar(I).'
        doc_str = 'Compute.\n\n@exception IOException If reading fails.'
        doc = _ScipDoc(
            relative_path='src/main/java/Foo.java',
            occurrences=(
                _ScipOccurrence(symbol=sym, range=(0, 0, 0, 1), is_definition=True),
            ),
            symbols=(
                _ScipSymbol(symbol=sym, kind='Method', documentation=doc_str),
            ),
        )
        index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

        elements = extract(
            Path('/repo/src/main/java/Foo.java'),
            source_root=Path('/repo'),
            index=index,
        )
        info = elements[0].documentation
        assert info is not None
        # @exception was parsed as @throws (Javadoc alias).
        assert info['throws'] == {'IOException': 'If reading fails.'}

    def test_jsdoc_parsed_for_typescript_symbol(self) -> None:
        """A .ts symbol's JSDoc documentation must be parsed into the
        structured form — with JSDoc's `@returns` alias and `{type}`
        annotations handled. Pins the JS/TS/Vue doc wiring."""
        from docgen.scip_extractor import (
            ScipIndex,
            _ScipDoc,
            _ScipOccurrence,
            _ScipSymbol,
            extract,
        )

        sym = 'scip-typescript npm web 0.1 src/`util.ts`/greet().'
        doc_str = (
            'Greets a user.\n\n'
            '@param {string} name The person to greet.\n'
            '@returns {string} The greeting.'
        )
        doc = _ScipDoc(
            relative_path='src/util.ts',
            occurrences=(
                _ScipOccurrence(symbol=sym, range=(0, 0, 0, 1), is_definition=True),
            ),
            symbols=(
                _ScipSymbol(symbol=sym, kind='Method', documentation=doc_str),
            ),
        )
        index = ScipIndex(documents=(doc,), source_root=Path('/repo'))

        elements = extract(
            Path('/repo/src/util.ts'), source_root=Path('/repo'), index=index,
        )
        info = elements[0].documentation
        assert info is not None
        assert info['summary'] == 'Greets a user.'
        # {type} stripped from the param; name→desc preserved.
        assert info['params'] == {'name': 'The person to greet.'}
        # @returns alias honored, {type} stripped.
        assert info['returns'] == 'The greeting.'


# ---------------------------------------------------------------------------
# extract — Python (Phase 2c)
# ---------------------------------------------------------------------------


class TestExtractPython:
    """scip-python emits SCIP symbols with the standard descriptor format.
    The extractor must dispatch ``.py`` files to a ``language='python'``
    extraction path with subtype mapping that matches catalog_extractor's
    Python subtypes (so SCIP-derived ElementInfo is a drop-in replacement
    for ast-grep at the catalog level).
    """

    def test_class_emitted_with_class_subtype(self, tmp_path) -> None:
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = 'scip-python python scalaproject 0.1 licensing/LicenseService#'
        doc = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(0, 6, 0, 20), is_definition=True,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Class', display_name='LicenseService',
            ),),
        )
        f = tmp_path / 'licensing.py'
        f.write_text('class LicenseService:\n    pass\n', encoding='utf-8')
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        assert len(elements) == 1
        assert elements[0].subtype == 'class'
        assert elements[0].language == 'python'
        assert elements[0].qualified_name == 'licensing.LicenseService'

    def test_class_method_subtype_is_method(self, tmp_path) -> None:
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#validate_token().'
        )
        doc = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(2, 4, 4, 0), is_definition=True,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='validate_token',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(
            tmp_path / 'licensing.py', source_root=tmp_path, index=index,
        )
        assert len(elements) == 1
        assert elements[0].subtype == 'method'
        assert elements[0].language == 'python'
        assert elements[0].parent_qualified_name == 'licensing.LicenseService'

    def test_top_level_function_subtype_is_function(self, tmp_path) -> None:
        """A Method symbol whose parent is a package (not a type/class)
        must dispatch to ``function``, not ``method``. scip-python emits
        Pyright's notion of 'method' for any callable; the descriptor
        chain is what disambiguates.
        """
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = 'scip-python python scalaproject 0.1 utils/compute_score().'
        doc = _ScipDoc(
            relative_path='utils.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(0, 4, 2, 0), is_definition=True,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='compute_score',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(
            tmp_path / 'utils.py', source_root=tmp_path, index=index,
        )
        assert len(elements) == 1
        assert elements[0].subtype == 'function'
        assert elements[0].parent_qualified_name == 'utils'

    def test_variable_kind_maps_to_variable(self, tmp_path) -> None:
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = 'scip-python python scalaproject 0.1 config/SECRET_KEY.'
        doc = _ScipDoc(
            relative_path='config.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(0, 0, 0, 10), is_definition=True,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Variable', display_name='SECRET_KEY',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(
            tmp_path / 'config.py', source_root=tmp_path, index=index,
        )
        assert len(elements) == 1
        assert elements[0].subtype == 'variable'


# ---------------------------------------------------------------------------
# extract — TypeScript / JavaScript (Phase 2c)
# ---------------------------------------------------------------------------


class TestExtractTypescript:
    """scip-typescript handles ``.ts``/``.tsx``/``.js``/``.jsx``/``.mjs``.
    The Subtype literal in catalog_extractor.py is shared across JS and
    TS (both surface as ``language='javascript'``)."""

    def test_class_emitted_with_js_class_subtype(self, tmp_path) -> None:
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = (
            'scip-typescript npm frontend 0.1 src/auth/AuthService#'
        )
        doc = _ScipDoc(
            relative_path='src/auth/AuthService.ts',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(0, 6, 0, 20), is_definition=True,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Class', display_name='AuthService',
            ),),
        )
        f = tmp_path / 'src' / 'auth' / 'AuthService.ts'
        f.parent.mkdir(parents=True)
        f.write_text('export class AuthService {}\n', encoding='utf-8')
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        assert len(elements) == 1
        assert elements[0].subtype == 'js_class'
        assert elements[0].language == 'javascript'

    def test_class_method_subtype_is_method(self, tmp_path) -> None:
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = (
            'scip-typescript npm frontend 0.1 '
            'src/auth/AuthService#login().'
        )
        doc = _ScipDoc(
            relative_path='src/auth/AuthService.ts',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(2, 2, 4, 0), is_definition=True,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='login',
            ),),
        )
        f = tmp_path / 'src' / 'auth' / 'AuthService.ts'
        f.parent.mkdir(parents=True)
        f.write_text('class X {\n  login() {}\n}\n', encoding='utf-8')
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        assert len(elements) == 1
        assert elements[0].subtype == 'method'

    def test_top_level_function_subtype_is_js_function(
        self, tmp_path,
    ) -> None:
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = (
            'scip-typescript npm frontend 0.1 src/utils/parseUrl().'
        )
        doc = _ScipDoc(
            relative_path='src/utils.ts',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(0, 0, 2, 0), is_definition=True,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='parseUrl',
            ),),
        )
        f = tmp_path / 'src' / 'utils.ts'
        f.parent.mkdir(parents=True)
        f.write_text('export function parseUrl() {}\n', encoding='utf-8')
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        assert len(elements) == 1
        assert elements[0].subtype == 'js_function'

    def test_javascript_extension_routes_to_typescript_handler(
        self, tmp_path,
    ) -> None:
        """scip-typescript indexes .js too; the same path handles
        non-.ts extensions."""
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = (
            'scip-typescript npm frontend 0.1 src/legacy/Foo#'
        )
        doc = _ScipDoc(
            relative_path='src/legacy/Foo.js',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(0, 6, 0, 9), is_definition=True,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Class', display_name='Foo',
            ),),
        )
        f = tmp_path / 'src' / 'legacy' / 'Foo.js'
        f.parent.mkdir(parents=True)
        f.write_text('class Foo {}\n', encoding='utf-8')
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        assert len(elements) == 1
        assert elements[0].language == 'javascript'
        assert elements[0].subtype == 'js_class'


# ---------------------------------------------------------------------------
# Phase 2c retro-audit — gaps surfaced by the adversarial review
# ---------------------------------------------------------------------------


class TestExtractAdversarial:
    """Tests added during the Phase 2c retro-audit. Each test probes a
    specific gap that the original Phase 2c tests didn't cover."""

    def test_typescript_function_kind_emits_js_function(
        self, tmp_path,
    ) -> None:
        """scip-typescript may emit ``kind='Function'`` for top-level
        functions instead of ``'Method'``. The dispatch table must
        handle both — otherwise top-level TS/JS functions silently
        disappear from the catalog."""
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = 'scip-typescript npm frontend 0.1 src/utils/parseUrl().'
        doc = _ScipDoc(
            relative_path='src/utils.ts',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 2, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Function',
                display_name='parseUrl',
            ),),
        )
        f = tmp_path / 'src' / 'utils.ts'
        f.parent.mkdir(parents=True)
        f.write_text('export function parseUrl() {}\n', encoding='utf-8')
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(f, source_root=tmp_path, index=index)
        assert len(elements) == 1, (
            'kind=Function should not silently drop — there is no other '
            'subtype left for top-level JS/TS functions'
        )
        assert elements[0].subtype == 'js_function'

    def test_python_function_kind_emits_function(self, tmp_path) -> None:
        """Same risk for Python: a SCIP indexer (current or future) may
        emit ``kind='Function'`` for top-level defs."""
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = 'scip-python python scalaproject 0.1 utils/helper().'
        doc = _ScipDoc(
            relative_path='utils.py',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 4, 2, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Function', display_name='helper',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(
            tmp_path / 'utils.py', source_root=tmp_path, index=index,
        )
        assert len(elements) == 1
        assert elements[0].subtype == 'function'

    def test_python_field_kind_emits_variable(self, tmp_path) -> None:
        """SCIP may emit ``kind='Field'`` for class attributes; only
        ``'Variable'`` was tested. Both must map to subtype 'variable'."""
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#token_expiry_seconds.'
        )
        doc = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(2, 4, 2, 30), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Field',
                display_name='token_expiry_seconds',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(
            tmp_path / 'licensing.py', source_root=tmp_path, index=index,
        )
        assert len(elements) == 1
        assert elements[0].subtype == 'variable'

    def test_python_documentation_does_not_crash_when_present(
        self, tmp_path,
    ) -> None:
        """Phase 2c punts on Python doc parsing — ``parse_doc is None``
        for Python. If a SCIP symbol has documentation, the extractor
        must drop it gracefully (not crash trying to parse with None).

        Regression guard: changing the parse_doc check to ``if
        sym.documentation:`` (without the ``parse_doc is not None``
        guard) would silently break this test."""
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        sym = 'scip-python python scalaproject 0.1 utils/compute_score().'
        doc = _ScipDoc(
            relative_path='utils.py',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 2, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method',
                display_name='compute_score',
                documentation='Computes a score.\n\n@param x: input',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(
            tmp_path / 'utils.py', source_root=tmp_path, index=index,
        )
        # No crash, element produced, documentation field is None
        # (Python doc parsing not yet wired)
        assert len(elements) == 1
        assert elements[0].documentation is None

    def test_nested_class_method_still_dispatches_to_method(
        self, tmp_path,
    ) -> None:
        """A class nested inside another class still has ``parent
        descriptor_kind == 'type'`` for its methods. Nested-class methods
        must dispatch to ``method``, not accidentally to ``function``."""
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol, extract,
        )

        # Outer.Inner.method — nested class method
        sym = (
            'scip-python python scalaproject 0.1 '
            'mod/Outer#Inner#method().'
        )
        doc = _ScipDoc(
            relative_path='mod.py',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(4, 8, 6, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='method',
            ),),
        )
        index = ScipIndex(documents=(doc,), source_root=tmp_path)

        elements = extract(
            tmp_path / 'mod.py', source_root=tmp_path, index=index,
        )
        assert len(elements) == 1
        assert elements[0].subtype == 'method', (
            'nested class methods must still resolve to "method"'
        )
