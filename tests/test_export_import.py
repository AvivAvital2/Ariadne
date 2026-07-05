"""Markdown export → import round-trip: delta import and H1 handling.

Slices (evolving TDD):
  1. _strip_title_h1 — normalize a title-matching leading H1 off a body.
  2. Import stops prepending the exported H1 to stored content (root fix).
  3. _is_unchanged — field-level equality between an incoming doc and a stored one.
  4. import_from_markdown skips already-identical docs (delta import).

Helpers under test are accessed as module attributes (export._name) so the
red phase fails at call time instead of erroring at collection.
"""
from __future__ import annotations

import pytest


class TestStripTitleH1:
    def test_strips_matching_leading_h1(self):
        import export
        assert export._strip_title_h1('# My Doc\n\nbody text', 'My Doc') == 'body text'

    def test_leaves_text_without_h1(self):
        import export
        assert export._strip_title_h1('body only', 'My Doc') == 'body only'

    def test_leaves_mismatched_h1(self):
        import export
        assert export._strip_title_h1('# Other\n\nbody', 'My Doc') == '# Other\n\nbody'

    def test_title_that_is_prefix_of_h1_is_not_stripped(self):
        import export
        assert export._strip_title_h1('# My Doc\n\nbody', 'My') == '# My Doc\n\nbody'

    def test_h1_only_document_becomes_empty(self):
        import export
        assert export._strip_title_h1('# My Doc', 'My Doc') == ''


class TestImportDoesNotMutateContent:
    def test_roundtrip_preserves_content_verbatim(self, tmp_path):
        import numpy as np
        from export import LibraryExporter, import_from_markdown
        from library import Library

        src = Library(tmp_path / 'src.db')
        doc = src.add_document(
            content_type='explanation', title='My Doc',
            content='Some explanation.\n\nMore text.',
            source_files=['a.py'], embedding=np.zeros(8, dtype=np.float32),
            metadata={'topic': 'x'},
        )
        out = tmp_path / 'docs'
        out.mkdir()
        LibraryExporter(src).export_document(doc, out)

        dst = Library(tmp_path / 'dst.db')
        assert import_from_markdown(dst, out) == 1
        imported = dst.get_document(doc.id)
        assert imported is not None
        assert imported.content == 'Some explanation.\n\nMore text.'


class TestIsUnchanged:
    @pytest.fixture
    def stored(self, tmp_path):
        import numpy as np
        from library import Library

        lib = Library(tmp_path / 'lib.db')
        return lib.add_document(
            content_type='explanation', title='My Doc',
            content='Body line.\n\nSecond.',
            source_files=['a.py'], embedding=np.zeros(8, dtype=np.float32),
            metadata={'topic': 'x'},
        )

    def test_identical_is_unchanged(self, stored):
        import export
        assert export._is_unchanged(
            stored, 'My Doc', 'Body line.\n\nSecond.', ['a.py'], {'topic': 'x'})

    def test_whitespace_only_difference_is_unchanged(self, stored):
        import export
        assert export._is_unchanged(
            stored, 'My Doc', '\nBody line.\n\nSecond.\n', ['a.py'], {'topic': 'x'})

    def test_legacy_h1_prefixed_stored_content_is_unchanged(self, tmp_path):
        import numpy as np
        import export
        from library import Library

        lib = Library(tmp_path / 'legacy.db')
        stored = lib.add_document(
            content_type='explanation', title='My Doc',
            content='# My Doc\n\nBody line.', source_files=[],
            embedding=np.zeros(8, dtype=np.float32), metadata={},
        )
        assert export._is_unchanged(stored, 'My Doc', 'Body line.', [], {})

    def test_different_title_is_changed(self, stored):
        import export
        assert not export._is_unchanged(
            stored, 'Other Doc', 'Body line.\n\nSecond.', ['a.py'], {'topic': 'x'})

    def test_different_content_is_changed(self, stored):
        import export
        assert not export._is_unchanged(
            stored, 'My Doc', 'Rewritten body.', ['a.py'], {'topic': 'x'})

    def test_different_source_files_is_changed(self, stored):
        import export
        assert not export._is_unchanged(
            stored, 'My Doc', 'Body line.\n\nSecond.', ['a.py', 'b.py'], {'topic': 'x'})

    def test_different_metadata_is_changed(self, stored):
        import export
        assert not export._is_unchanged(
            stored, 'My Doc', 'Body line.\n\nSecond.', ['a.py'], {'topic': 'y'})


class TestDeltaImport:
    def _roundtrip_library(self, tmp_path):
        """A library with one embedded doc, plus its export directory."""
        import numpy as np
        from export import LibraryExporter
        from library import Library

        lib = Library(tmp_path / 'lib.db')
        doc = lib.add_document(
            content_type='explanation', title='My Doc',
            content='Body line.\n\nSecond.',
            source_files=['a.py'], embedding=np.zeros(8, dtype=np.float32),
            metadata={'topic': 'x'},
        )
        out = tmp_path / 'docs'
        out.mkdir()
        path = LibraryExporter(lib).export_document(doc, out)
        return lib, doc, out, path

    def test_reimport_unchanged_writes_nothing(self, tmp_path):
        from export import import_from_markdown

        lib, doc, out, _ = self._roundtrip_library(tmp_path)
        before = lib.get_document(doc.id)
        assert import_from_markdown(lib, out) == 0
        after = lib.get_document(doc.id)
        assert after.updated_at == before.updated_at
        assert after.embedding is not None

    def test_changed_doc_is_written(self, tmp_path):
        from export import import_from_markdown

        lib, doc, out, path = self._roundtrip_library(tmp_path)
        path.write_text(path.read_text().replace('Second.', 'Rewritten.'))
        assert import_from_markdown(lib, out) == 1
        assert 'Rewritten.' in lib.get_document(doc.id).content

    def test_new_doc_is_created_and_counted_alone(self, tmp_path):
        from export import import_from_markdown

        lib, doc, out, _ = self._roundtrip_library(tmp_path)
        (out / 'explanations' / 'brand-new.md').write_text('# Brand New\n\nFresh content.')
        assert import_from_markdown(lib, out) == 1
        assert lib.get_document(doc.id) is not None

    def test_legacy_h1_database_skips_identical_export(self, tmp_path):
        import numpy as np
        from export import import_from_markdown
        from library import Library

        lib, doc, out, _ = self._roundtrip_library(tmp_path)
        legacy = Library(tmp_path / 'legacy.db')
        legacy.add_document(
            content_type='explanation', title='My Doc',
            content='# My Doc\n\nBody line.\n\nSecond.',
            source_files=['a.py'], embedding=np.zeros(8, dtype=np.float32),
            metadata={'topic': 'x'}, doc_id=doc.id,
        )
        assert import_from_markdown(legacy, out) == 0


class TestImportEdgeCases:
    def test_md_without_frontmatter_or_h1_uses_stem_title(self, tmp_path):
        from export import import_from_markdown
        from library import Library
        from schema import generate_deterministic_id

        out = tmp_path / 'docs'
        out.mkdir()
        (out / 'plain-note.md').write_text('Just prose, no heading at all.')
        lib = Library(tmp_path / 'lib.db')
        assert import_from_markdown(lib, out) == 1
        doc = lib.get_document(generate_deterministic_id('explanation', 'plain-note'))
        assert doc is not None
        assert doc.content == 'Just prose, no heading at all.'

    def test_invalid_frontmatter_type_infers_from_directory(self, tmp_path):
        from export import import_from_markdown
        from library import Library
        from schema import generate_deterministic_id

        sub = tmp_path / 'docs' / 'findings'
        sub.mkdir(parents=True)
        (sub / 'insight.md').write_text(
            '---\ntype: bogus\ntitle: "Insight"\n---\n\nA finding body.')
        lib = Library(tmp_path / 'lib.db')
        assert import_from_markdown(lib, tmp_path / 'docs') == 1
        assert lib.get_document(generate_deterministic_id('finding', 'Insight')) is not None

    def test_non_dict_metadata_is_dropped(self, tmp_path):
        from export import import_from_markdown
        from library import Library
        from schema import generate_deterministic_id

        out = tmp_path / 'docs'
        out.mkdir()
        (out / 'odd.md').write_text('---\ntitle: "Odd"\nmetadata: scalar\n---\n\nBody.')
        lib = Library(tmp_path / 'lib.db')
        assert import_from_markdown(lib, out) == 1
        assert lib.get_document(generate_deterministic_id('explanation', 'Odd')).metadata == {}

    def test_file_index_docs_are_skipped(self, tmp_path):
        from export import import_from_markdown
        from library import Library
        from schema import CATALOG_KIND_FILE_INDEX

        out = tmp_path / 'docs'
        out.mkdir()
        (out / 'idx.md').write_text(
            f'---\ntitle: "Idx"\nmetadata:\n  kind: {CATALOG_KIND_FILE_INDEX}\n---\n\nDerived.')
        lib = Library(tmp_path / 'lib.db')
        assert import_from_markdown(lib, out) == 0

    def test_readme_and_manifest_are_ignored(self, tmp_path):
        from export import import_from_markdown
        from library import Library

        out = tmp_path / 'docs'
        out.mkdir()
        (out / 'README.md').write_text('# Not a doc')
        lib = Library(tmp_path / 'lib.db')
        assert import_from_markdown(lib, out) == 0


class TestExportAll:
    @pytest.fixture
    def populated(self, tmp_path):
        import numpy as np
        from library import Library
        from schema import CATALOG_KIND_FILE_INDEX

        lib = Library(tmp_path / 'lib.db')
        lib.add_document(
            content_type='explanation', title='How Parsing Works',
            content='Parsing body.', source_files=['p.py'],
            embedding=np.zeros(4, dtype=np.float32), metadata={'n': 1})
        lib.add_document(
            content_type='architecture', title='Storage Design',
            content='Design body.', source_files=[],
            embedding=np.zeros(4, dtype=np.float32), metadata={})
        lib.add_document(
            content_type='finding', title='Perf Insight',
            content='Finding body.', source_files=['f.py'],
            embedding=np.zeros(4, dtype=np.float32), metadata={'kind': 'x'})
        lib.add_document(
            content_type='catalog', title='files idx', content='idx',
            source_files=[], embedding=np.zeros(4, dtype=np.float32),
            metadata={'kind': CATALOG_KIND_FILE_INDEX})
        return lib

    def test_export_all_generates_meta_files_and_skips_file_index(self, populated, tmp_path):
        from export import LibraryExporter

        out = tmp_path / 'docs'
        src_dir = tmp_path / 'proj'
        src_dir.mkdir()
        (src_dir / 'CLAUDE.md').write_text(
            'Project intro.\n\n---\n\n# Ariadne-Generated Sections\nold stuff\n')
        paths = LibraryExporter(populated).export_all(
            out, source_name='proj', source_path=src_dir, dependencies=['dep1'])
        assert (out / 'README.md').is_file()
        assert (out / 'manifest.yaml').is_file()
        assert (out / 'INDEX.md').is_file()
        assert (out / 'CLAUDE.md').is_file()
        assert len(paths) == 4  # 3 real docs + CLAUDE.md; file_index doc excluded
        claude = (out / 'CLAUDE.md').read_text()
        assert 'Project intro.' in claude
        assert 'old stuff' not in claude

    def test_export_all_roundtrip_then_reimport_is_full_delta(self, populated, tmp_path):
        from export import LibraryExporter, import_from_markdown
        from library import Library

        out = tmp_path / 'docs'
        LibraryExporter(populated).export_all(out)
        dst = Library(tmp_path / 'dst.db')
        # 3 real docs — INDEX.md is an exporter meta file, skipped on import
        assert import_from_markdown(dst, out) == 3
        # second pass over the same export: pure delta, nothing written
        assert import_from_markdown(dst, out) == 0

    def test_export_document_bare_config(self, populated, tmp_path):
        from export import ExportConfig, LibraryExporter

        doc = next(d for d in populated.list_documents() if d.title == 'How Parsing Works')
        out = tmp_path / 'bare'
        out.mkdir()
        exporter = LibraryExporter(populated, config=ExportConfig(
            include_frontmatter=False, include_source_files=False, organize_by_type=False))
        path = exporter.export_document(doc, out)
        assert path.parent == out
        text = path.read_text()
        assert not text.startswith('---')
        assert '**Related files:**' not in text
