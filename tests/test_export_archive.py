"""Single-artifact export/import: archive mode (designs/export-single-artifact.md).

Slices (evolving TDD):
  1. LibraryExporter.export_archive — exactly one zip, marker comment,
     member parity with a tree export.
  2. (in test_cli_core.py) archive is the CLI default; --no-archive
     restores the tree export.
  3. Marker-gated overwrite guard — an unmarked zip is a user file,
     never replaced.
  4. import_from_archive — safe extract (zip-slip abort, macOS junk
     skip), export-root discovery, delegation to import_from_markdown.
  5-6. Archive delta parity — re-import is a no-op; exporter meta files
     (README/INDEX/CLAUDE) never import as documents.
  7. Scoped archive round-trip (design intent: export-scoping fix) — a
     source-scoped archive plants exactly that source's docs into a fresh
     destination; foreign and unattributable docs never travel.

New surface under test is accessed as an attribute (exporter.export_archive,
export.import_from_archive) so each red phase fails at call time instead of
erroring at collection.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pytest
import export
from export import LibraryExporter, import_from_markdown
from library import Library

ARCHIVE_MARKER = b'ariadne-export'


def _make_library(db_path):
    lib = Library(db_path)
    lib.add_document(
        content_type='explanation', title='First Topic',
        content='How the first thing works.',
        source_files=['pkg/first.py'], embedding=np.zeros(8, dtype=np.float32),
        metadata={'topic': 'first'}, source_name='src1',
    )
    lib.add_document(
        content_type='architecture', title='Second Topic',
        content='Why the second thing is shaped this way.',
        source_files=['pkg/second.py'], embedding=np.zeros(8, dtype=np.float32),
        metadata={'topic': 'second'}, source_name='src1',
    )
    return lib


class TestExportArchive:
    def test_single_zip_with_tree_parity_and_marker(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')

        tree = tmp_path / 'tree'
        LibraryExporter(lib).export_all(tree)

        out_dir = tmp_path / 'out'
        out_dir.mkdir()
        archive_path = out_dir / 'src1.zip'
        result = LibraryExporter(lib).export_archive(archive_path)

        assert result == archive_path
        assert [p.name for p in out_dir.iterdir()] == ['src1.zip']

        tree_files = {
            p.relative_to(tree).as_posix(): p
            for p in tree.rglob('*') if p.is_file()
        }
        with zipfile.ZipFile(archive_path) as zf:
            assert zf.comment == ARCHIVE_MARKER
            members = {i.filename for i in zf.infolist() if not i.is_dir()}
            assert members == {f'src1/{rel}' for rel in tree_files}
            # manifest.yaml and INDEX.md embed a generation timestamp
            # (export.py:235,379); everything else must match byte-for-byte.
            for rel, tree_file in tree_files.items():
                if rel in ('manifest.yaml', 'INDEX.md'):
                    assert zf.read(f'src1/{rel}')
                else:
                    assert zf.read(f'src1/{rel}') == tree_file.read_bytes()


class TestOverwriteGuard:
    def test_replaces_archive_carrying_the_marker(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')
        target = tmp_path / 'src1.zip'
        LibraryExporter(lib).export_archive(target)

        result = LibraryExporter(lib).export_archive(target)

        assert result == target
        with zipfile.ZipFile(target) as zf:
            assert zf.comment == ARCHIVE_MARKER
            assert any(n.endswith('.md') for n in zf.namelist())

    def test_refuses_zip_without_marker(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')
        target = tmp_path / 'src1.zip'
        with zipfile.ZipFile(target, 'w') as zf:
            zf.writestr('keep/precious.md', 'user data, not ours')
        before = target.read_bytes()

        with pytest.raises(FileExistsError):
            LibraryExporter(lib).export_archive(target)

        assert target.read_bytes() == before  # untouched

    def test_refuses_non_zip_file_at_target(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')
        target = tmp_path / 'src1.zip'
        target.write_text('plain text, merely named .zip')
        before = target.read_bytes()

        with pytest.raises(FileExistsError):
            LibraryExporter(lib).export_archive(target)

        assert target.read_bytes() == before


class TestImportFromArchive:
    def test_archive_import_matches_tree_import(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')
        tree = tmp_path / 'tree'
        LibraryExporter(lib).export_all(tree)
        archive = tmp_path / 'src1.zip'
        LibraryExporter(lib).export_archive(archive)

        from_tree = Library(tmp_path / 'from_tree.db')
        from_zip = Library(tmp_path / 'from_zip.db')
        tree_count = import_from_markdown(from_tree, tree)
        zip_count = export.import_from_archive(from_zip, archive)

        assert zip_count == tree_count

        def snapshot(lib_):
            return {
                d.id: (d.title, d.content, list(d.source_files),
                       dict(d.metadata), d.content_type)
                for d in lib_.list_documents()
            }

        assert snapshot(from_zip) == snapshot(from_tree)

    def test_hand_zipped_flat_tree_with_junk_imports_clean(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')
        tree = tmp_path / 'tree'
        LibraryExporter(lib).export_all(tree)

        # user-made zip: flat root, no marker comment, macOS junk inside
        target = tmp_path / 'handmade.zip'
        with zipfile.ZipFile(target, 'w') as zf:
            for p in sorted(tree.rglob('*')):
                if p.is_file():
                    zf.writestr(p.relative_to(tree).as_posix(), p.read_bytes())
            zf.writestr('__MACOSX/explanations/._first-topic.md', 'resource fork junk')
            zf.writestr('explanations/._second-topic.md', 'appledouble junk')

        dst = Library(tmp_path / 'dst.db')
        count = export.import_from_archive(dst, target)

        docs = dst.list_documents()
        assert count == len(docs)
        titles = {d.title for d in docs}
        assert {'First Topic', 'Second Topic'} <= titles
        assert all('junk' not in d.content for d in docs)

    def test_archive_without_manifest_fails_loud(self, tmp_path):
        target = tmp_path / 'noroot.zip'
        with zipfile.ZipFile(target, 'w') as zf:
            zf.writestr('loose.md', 'no manifest anywhere')
        dst = Library(tmp_path / 'dst.db')
        with pytest.raises(ValueError):
            export.import_from_archive(dst, target)
        assert dst.list_documents() == []

    def test_non_zip_input_fails_loud(self, tmp_path):
        bogus = tmp_path / 'bogus.zip'
        bogus.write_text('not an archive')
        dst = Library(tmp_path / 'dst.db')
        with pytest.raises(ValueError):
            export.import_from_archive(dst, bogus)

    def test_zip_slip_member_aborts_before_any_write(self, tmp_path):
        target = tmp_path / 'evil.zip'
        with zipfile.ZipFile(target, 'w') as zf:
            zf.writestr('src1/manifest.yaml', 'docs: []')
            zf.writestr('../escape.md', 'break out')
        dst = Library(tmp_path / 'dst.db')
        with pytest.raises(ValueError):
            export.import_from_archive(dst, target)
        assert dst.list_documents() == []
        assert not (tmp_path / 'escape.md').exists()


class TestArchiveDeltaParity:
    def test_reimport_into_source_is_noop(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')
        before = {
            d.id: (d.updated_at, d.embedding is not None)
            for d in lib.list_documents()
        }
        assert all(has_emb for _, has_emb in before.values())

        archive = tmp_path / 'src1.zip'
        LibraryExporter(lib).export_archive(archive)

        assert export.import_from_archive(lib, archive) == 0

        after = {
            d.id: (d.updated_at, d.embedding is not None)
            for d in lib.list_documents()
        }
        assert after == before

    def test_meta_files_never_become_docs(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')
        archive = tmp_path / 'src1.zip'
        # source_name makes the export generate CLAUDE.md alongside
        # README.md and INDEX.md — none of them are knowledge docs
        LibraryExporter(lib).export_archive(archive, source_name='src1')

        dst = Library(tmp_path / 'dst.db')
        export.import_from_archive(dst, archive)

        assert {d.title for d in dst.list_documents()} == {
            'First Topic', 'Second Topic',
        }

    def test_only_changed_docs_are_written_through_archive(self, tmp_path):
        lib = _make_library(tmp_path / 'src.db')
        first_archive = tmp_path / 'first.zip'
        LibraryExporter(lib).export_archive(first_archive)

        dst = Library(tmp_path / 'dst.db')
        assert export.import_from_archive(dst, first_archive) == 2

        second_before = next(
            d for d in dst.list_documents() if d.title == 'Second Topic')

        # revise one doc at the source, ship a fresh archive
        changed = next(
            d for d in lib.list_documents() if d.title == 'First Topic')
        lib.add_document(
            content_type='explanation', title='First Topic',
            content='How the first thing works — revised.',
            source_files=['pkg/first.py'],
            embedding=np.zeros(8, dtype=np.float32),
            metadata={'topic': 'first'}, doc_id=changed.id,
        )
        second_archive = tmp_path / 'second.zip'
        LibraryExporter(lib).export_archive(second_archive)

        assert export.import_from_archive(dst, second_archive) == 1
        assert 'revised' in dst.get_document(changed.id).content
        second_after = next(
            d for d in dst.list_documents() if d.title == 'Second Topic')
        assert second_after.updated_at == second_before.updated_at


class TestScopedArchiveRoundTrip:
    def test_scoped_archive_plants_only_the_sources_docs(self, tmp_path):
        """Integration of the export-scoping fix (design intent): shipping
        a src1 archive to a fresh destination must plant exactly src1's
        corpus — never foreign sources, never unattributable docs — and a
        re-import of the same archive is a clean delta no-op."""
        src1_dir = tmp_path / 'src1'
        src1_dir.mkdir()
        lib = Library(tmp_path / 'src.db')
        keep_named = lib.add_document(
            content_type='explanation', title='Src1 Doc',
            content='src1 body.', source_files=[],
            embedding=np.zeros(8, dtype=np.float32), source_name='src1')
        keep_legacy = lib.add_document(
            content_type='explanation', title='Legacy Under Src1',
            content='legacy body.', source_files=[str(src1_dir / 'm.py')],
            embedding=np.zeros(8, dtype=np.float32))
        lib.add_document(
            content_type='explanation', title='Foreign Doc',
            content='other source.', source_files=[],
            embedding=np.zeros(8, dtype=np.float32), source_name='src2')
        lib.add_document(
            content_type='explanation', title='Unattributable Doc',
            content='no home.', source_files=[],
            embedding=np.zeros(8, dtype=np.float32))

        archive = tmp_path / 'src1.zip'
        LibraryExporter(lib).export_archive(
            archive, source_name='src1', source_path=src1_dir)

        dst = Library(tmp_path / 'dst.db')
        assert export.import_from_archive(dst, archive) == 2
        planted = {d.id for d in dst.list_documents()}
        assert planted == {keep_named.id, keep_legacy.id}
        # Both docs arrive attributed, ready for scoped use on the
        # destination (whose filesystem knows nothing of src1_dir).
        assert dst.get_document(keep_named.id).source_name == 'src1'
        assert dst.get_document(keep_legacy.id).source_name == 'src1'
        assert export.import_from_archive(dst, archive) == 0
