"""Markdown export → import round-trip: delta import and H1 handling.

Slices (evolving TDD):
  1. _strip_title_h1 — normalize a title-matching leading H1 off a body.
  2. Import stops prepending the exported H1 to stored content (root fix).
  3. _is_unchanged — field-level equality between an incoming doc and a stored one.
  4. import_from_markdown skips already-identical docs (delta import).
  5. Scoped export — export_all(source_name=...) selects only that source's
     docs (explicit source_name wins both ways; unattributed docs fall back
     to absolute source_files under source_path); manifest/README/INDEX
     describe exactly the scoped set.
  6. Collision-proof filenames — same-titled docs never overwrite each
     other: deterministic id-sorted order, first keeps the clean slug, the
     rest get an id suffix; re-export is filename-stable.
  7. Attribution round-trip — the wire format carries the EFFECTIVE
     attribution (export scope > source_name column > legacy metadata,
     never invented); import restores the column, so a scoped re-export
     from the destination still selects the docs.

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

    def test_column_attribution_equals_its_serialized_form(self, tmp_path):
        """A column-attributed doc must compare unchanged against its own
        export, whose wire metadata carries the stamped source_name —
        otherwise every re-import into the source db churns."""
        import numpy as np
        import export
        from library import Library

        lib = Library(tmp_path / 'col.db')
        stored = lib.add_document(
            content_type='explanation', title='My Doc',
            content='Body line.', source_files=['a.py'],
            embedding=np.zeros(8, dtype=np.float32),
            metadata={'topic': 'x'}, source_name='src1',
        )
        assert export._is_unchanged(
            stored, 'My Doc', 'Body line.', ['a.py'],
            {'topic': 'x', 'source_name': 'src1'})
        # A genuinely different attribution is still a change.
        assert not export._is_unchanged(
            stored, 'My Doc', 'Body line.', ['a.py'],
            {'topic': 'x', 'source_name': 'src2'})


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
            embedding=np.zeros(4, dtype=np.float32),
            metadata={'n': 1, 'source_name': 'proj'})
        lib.add_document(
            content_type='architecture', title='Storage Design',
            content='Design body.', source_files=[],
            embedding=np.zeros(4, dtype=np.float32),
            metadata={'source_name': 'proj'})
        lib.add_document(
            content_type='finding', title='Perf Insight',
            content='Finding body.', source_files=['f.py'],
            embedding=np.zeros(4, dtype=np.float32),
            metadata={'kind': 'x', 'source_name': 'proj'})
        lib.add_document(
            content_type='catalog', title='files idx', content='idx',
            source_files=[], embedding=np.zeros(4, dtype=np.float32),
            metadata={'kind': CATALOG_KIND_FILE_INDEX, 'source_name': 'proj'})
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

    def test_scoped_export_selects_only_the_named_sources_docs(self, tmp_path):
        import numpy as np
        from export import LibraryExporter
        from library import Library

        src1_dir = tmp_path / 'src1'
        src1_dir.mkdir()
        src2_dir = tmp_path / 'src2'
        src2_dir.mkdir()

        def _zeros():
            return np.zeros(4, dtype=np.float32)

        lib = Library(tmp_path / 'scoped.db')
        # Explicit attribution is authoritative in BOTH directions and lives
        # in TWO channels — the source_name column (modern) and the metadata
        # key (legacy): a doc named src1 is in even when its files live
        # under src2, and a doc named src2 is out even when its files live
        # under src1.
        lib.add_document(
            content_type='explanation', title='Named Src1', content='b',
            source_files=[str(src2_dir / 'x.py')], embedding=_zeros(),
            source_name='src1')
        lib.add_document(
            content_type='explanation', title='Named Src2', content='b',
            source_files=[str(src1_dir / 'y.py')], embedding=_zeros(),
            source_name='src2')
        lib.add_document(
            content_type='explanation', title='Meta Src1', content='b',
            source_files=[], embedding=_zeros(),
            metadata={'source_name': 'src1'})
        lib.add_document(
            content_type='explanation', title='Meta Src2', content='b',
            source_files=[str(src1_dir / 'z.py')], embedding=_zeros(),
            metadata={'source_name': 'src2'})
        # Unattributed docs fall back to absolute source_files under the
        # source path; relative or missing files are unattributable.
        lib.add_document(
            content_type='explanation', title='Legacy Under Src1', content='b',
            source_files=[str(src1_dir / 'c.py')], embedding=_zeros(),
            metadata={})
        lib.add_document(
            content_type='explanation', title='Legacy Elsewhere', content='b',
            source_files=[str(tmp_path / 'other' / 'd.py')],
            embedding=_zeros(), metadata={})
        lib.add_document(
            content_type='explanation', title='Legacy Relative', content='b',
            source_files=['rel.py'], embedding=_zeros(), metadata={})
        lib.add_document(
            content_type='explanation', title='Legacy No Files', content='b',
            source_files=[], embedding=_zeros(), metadata={})

        out = tmp_path / 'docs'
        LibraryExporter(lib).export_all(
            out, source_name='src1', source_path=src1_dir)

        exported = {p.stem for p in out.rglob('*.md')
                    if p.name not in ('README.md', 'INDEX.md', 'CLAUDE.md')}
        assert exported == {'named-src1', 'meta-src1', 'legacy-under-src1'}
        # Every meta surface must describe exactly the scoped set: manifest
        # and INDEX list titles; README carries the total count.
        for meta_name in ('manifest.yaml', 'INDEX.md'):
            text = (out / meta_name).read_text()
            assert 'Named Src1' in text and 'Legacy Under Src1' in text
            assert 'Named Src2' not in text and 'Meta Src2' not in text
            assert 'Legacy Elsewhere' not in text
        assert 'Total documents: 3' in (out / 'README.md').read_text()

        # Without a source_path there is no fallback: named docs only.
        out2 = tmp_path / 'docs-no-path'
        LibraryExporter(lib).export_all(out2, source_name='src1')
        exported2 = {p.stem for p in out2.rglob('*.md')
                     if p.name not in ('README.md', 'INDEX.md', 'CLAUDE.md')}
        assert exported2 == {'named-src1', 'meta-src1'}

    def test_colliding_titles_all_export_without_loss(self, tmp_path):
        import numpy as np
        from export import LibraryExporter, import_from_markdown
        from library import Library

        lib = Library(tmp_path / 'collide.db')
        ids = [
            lib.add_document(
                content_type='explanation', title='Shared Title',
                content=f'body variant {i}', source_files=[],
                embedding=np.zeros(4, dtype=np.float32),
                source_name='src1').id
            for i in range(3)
        ]
        out = tmp_path / 'docs'
        LibraryExporter(lib).export_all(out, source_name='src1')
        md = sorted(p for p in out.rglob('*.md')
                    if p.name not in ('README.md', 'INDEX.md', 'CLAUDE.md'))
        assert len(md) == 3, 'colliding titles must never overwrite silently'
        # Deterministic naming: the id-sorted winner keeps the clean slug,
        # the rest carry an id suffix.
        assert {p.name for p in md} == {'shared-title.md'} | {
            f'shared-title-{i[:8]}.md' for i in sorted(ids)[1:]}
        # Filenames are stable across re-export of the same doc set.
        out2 = tmp_path / 'docs2'
        LibraryExporter(lib).export_all(out2, source_name='src1')
        assert ({p.name for p in out2.rglob('*.md')}
                == {p.name for p in out.rglob('*.md')})
        # Nothing is lost end-to-end: all three docs import distinctly.
        dst = Library(tmp_path / 'dst.db')
        assert import_from_markdown(dst, out) == 3
        assert {d.id for d in dst.list_documents()} == set(ids)

    def test_attribution_survives_the_round_trip(self, tmp_path):
        import numpy as np
        from export import LibraryExporter, import_from_markdown
        from library import Library

        src1_dir = tmp_path / 'src1'
        src1_dir.mkdir()

        def _zeros():
            return np.zeros(4, dtype=np.float32)

        meta_files = ('README.md', 'INDEX.md', 'CLAUDE.md')
        lib = Library(tmp_path / 'attr.db')
        col = lib.add_document(
            content_type='explanation', title='Column Attributed',
            content='b', source_files=[], embedding=_zeros(),
            source_name='src1')
        lib.add_document(
            content_type='explanation', title='Divergent Channels',
            content='b', source_files=[], embedding=_zeros(),
            source_name='src1', metadata={'source_name': 'src2'})
        legacy = lib.add_document(
            content_type='explanation', title='Legacy Under Src1',
            content='b', source_files=[str(src1_dir / 'c.py')],
            embedding=_zeros())
        lib.add_document(
            content_type='explanation', title='Plain Unattributed',
            content='b', source_files=[], embedding=_zeros())

        # A scoped export serializes the EFFECTIVE attribution: the column
        # beats a divergent legacy metadata value, and a path-matched
        # legacy doc is stamped with the scope it was selected for (its
        # absolute paths mean nothing on the destination machine).
        out = tmp_path / 'docs'
        LibraryExporter(lib).export_all(
            out, source_name='src1', source_path=src1_dir)
        texts = {p.stem: p.read_text() for p in out.rglob('*.md')
                 if p.name not in meta_files}
        assert set(texts) == {
            'column-attributed', 'divergent-channels', 'legacy-under-src1'}
        for stem in texts:
            assert 'source_name: "src1"' in texts[stem]

        # Import into a fresh destination restores the attribution column,
        # and a scoped re-export from the destination (which has no source
        # paths to fall back on) still selects every doc.
        dst = Library(tmp_path / 'dst.db')
        assert import_from_markdown(dst, out) == 3
        assert dst.get_document(col.id).source_name == 'src1'
        assert dst.get_document(legacy.id).source_name == 'src1'
        out2 = tmp_path / 'docs2'
        LibraryExporter(dst).export_all(out2, source_name='src1')
        stems2 = {p.stem for p in out2.rglob('*.md')
                  if p.name not in meta_files}
        assert stems2 == {
            'column-attributed', 'divergent-channels', 'legacy-under-src1'}
        # Re-import of the same tree is a clean delta no-op.
        assert import_from_markdown(dst, out) == 0

        # An unscoped export serializes existing attribution but never
        # invents one for unattributed docs.
        out3 = tmp_path / 'docs3'
        LibraryExporter(lib).export_all(out3)
        texts3 = {p.stem: p.read_text() for p in out3.rglob('*.md')
                  if p.name not in meta_files}
        assert 'source_name: "src1"' in texts3['column-attributed']
        assert 'source_name' not in texts3['plain-unattributed']
        assert 'source_name' not in texts3['legacy-under-src1']

        # Re-importing the scoped tree into the SOURCE db converges: only
        # the path-fallback doc gets its stamped attribution written back
        # (once), column-attributed docs never churn, and the second pass
        # is a clean no-op.
        assert import_from_markdown(lib, out) == 1
        assert import_from_markdown(lib, out) == 0

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
