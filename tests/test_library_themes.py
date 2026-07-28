"""Tests for ThemesMixin and theme dataclasses (Themes plan, Phase 1).

Encodes the contract the implementation must satisfy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from library import Library
from library.themes import ClusterMapping, Theme


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'themes-test.db')
    yield lib
    lib.close()


def _add_doc(
    library: Library,
    doc_id: str,
    *,
    content_type: str = 'theme',
    source_name: str | None = None,
) -> None:
    """Helper to insert a document for FK targets (themes.doc_id, theme_members.element_id)."""
    library.add_document(
        content_type=content_type,  # type: ignore[arg-type]
        title=f'Doc {doc_id}',
        content='content',
        source_files=[],
        metadata={},
        doc_id=doc_id,
    )
    # add_document doesn't expose source_name as a kwarg today; set the column directly.
    if source_name is not None:
        with library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE documents SET source_name = ? WHERE id = ?',
                (source_name, doc_id),
            )


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


class TestThemesSchema:
    def test_library_init_creates_themes_tables(self, library: Library) -> None:
        with library._conn_provider.acquire() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert 'themes' in tables
        assert 'theme_members' in tables
        assert 'cluster_history' in tables

    def test_theme_is_valid_content_type(self) -> None:
        from schema import CONTENT_TYPES
        assert 'theme' in CONTENT_TYPES


# ---------------------------------------------------------------------------
# Theme CRUD
# ---------------------------------------------------------------------------


class TestThemeCRUD:
    def test_add_theme_inserts_row(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        library.add_theme(
            cluster_id='c1',
            doc_id='doc-c1',
            member_count=10,
            resolution=1.0,
            summary_hash='abc123',
            coherent=True,
        )
        theme = library.get_theme('c1')
        assert theme is not None
        assert theme.cluster_id == 'c1'
        assert theme.doc_id == 'doc-c1'
        assert theme.member_count == 10
        assert theme.resolution == pytest.approx(1.0)
        assert theme.summary_hash == 'abc123'
        assert theme.coherent is True
        # Newly added theme is dirty (needs summarization) by default.
        assert theme.dirty is True
        # Timestamps are auto-populated.
        assert theme.last_built_at
        assert theme.last_summarized_at

    def test_add_theme_with_explicit_dirty_false(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        library.add_theme(
            cluster_id='c1',
            doc_id='doc-c1',
            member_count=5,
            resolution=1.0,
            summary_hash='h',
            dirty=False,
        )
        theme = library.get_theme('c1')
        assert theme is not None
        assert theme.dirty is False

    def test_get_theme_returns_none_when_missing(self, library: Library) -> None:
        assert library.get_theme('nonexistent') is None

    def test_list_themes_returns_all_when_coherent_only_false(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'doc-c2', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=5, resolution=1.0, summary_hash='h1',
            coherent=True,
        )
        library.add_theme(
            cluster_id='c2', doc_id='doc-c2',
            member_count=3, resolution=1.0, summary_hash='h2',
            coherent=False,
        )
        themes = library.list_themes(coherent_only=False)
        assert {t.cluster_id for t in themes} == {'c1', 'c2'}

    def test_list_themes_coherent_only_filters_incoherent(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'doc-c2', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=5, resolution=1.0, summary_hash='h1',
            coherent=True,
        )
        library.add_theme(
            cluster_id='c2', doc_id='doc-c2',
            member_count=3, resolution=1.0, summary_hash='h2',
            coherent=False,
        )
        themes = library.list_themes(coherent_only=True)
        assert [t.cluster_id for t in themes] == ['c1']

    def test_list_themes_source_filter(self, library: Library) -> None:
        """``source=`` filters by MEMBER source, not the summary doc.

        Production reality: a theme's summary doc is cross-source — its
        ``source_name`` is NULL, because ``docgen/cluster.py`` never tags
        it. A theme belongs to a source through its *members* (each a
        document that does carry a ``source_name``). So a cross-cutting
        theme with members in several sources matches every one of them,
        and a theme with several members in the same source appears once
        (DISTINCT).
        """
        # Summary docs: cross-source (NULL), as cluster.py writes them.
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'doc-c2', content_type='theme')
        _add_doc(library, 'doc-c3', content_type='theme')
        # Members carry the real source.
        _add_doc(library, 'el-a', content_type='catalog', source_name='src1')
        _add_doc(library, 'el-b', content_type='catalog', source_name='src1')
        _add_doc(library, 'el-c', content_type='catalog', source_name='src2')
        for cid, doc in (('c1', 'doc-c1'), ('c2', 'doc-c2'), ('c3', 'doc-c3')):
            library.add_theme(
                cluster_id=cid, doc_id=doc,
                member_count=1, resolution=1.0, summary_hash=f'h-{cid}',
            )
        library.set_theme_members('c1', [('el-a', 1.0), ('el-b', 0.5)])  # two src1 members
        library.set_theme_members('c2', [('el-c', 1.0)])                 # src2 only
        library.set_theme_members('c3', [('el-a', 1.0), ('el-c', 0.5)])  # cross-cutting

        # src1: c1 (once, despite two members) + the cross-cutting c3.
        assert [t.cluster_id for t in library.list_themes(source='src1')] == ['c1', 'c3']
        # src2: c2 + the cross-cutting c3.
        assert [t.cluster_id for t in library.list_themes(source='src2')] == ['c2', 'c3']

    def test_delete_theme_cascades_members(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'el1', content_type='catalog')
        _add_doc(library, 'el2', content_type='catalog')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=2, resolution=1.0, summary_hash='h',
        )
        library.set_theme_members('c1', [('el1', 1.0), ('el2', 0.5)])

        library.delete_theme('c1')

        assert library.get_theme('c1') is None
        assert library.get_theme_members('c1') == []


class TestThemeCoherenceCounts:
    """``theme_coherence_counts`` — the self-serve coherence-rate readout.

    Reuses ``list_themes`` so the numbers can never diverge from what the
    same ``source=`` filter would list.
    """

    def test_counts_split_coherent_and_incoherent(self, library: Library) -> None:
        for cid, ok in (('c1', True), ('c2', True), ('c3', False)):
            _add_doc(library, f'd-{cid}', content_type='theme')
            library.add_theme(
                cluster_id=cid, doc_id=f'd-{cid}',
                member_count=1, resolution=1.0, summary_hash='h', coherent=ok,
            )
        assert library.theme_coherence_counts() == {
            'coherent': 2, 'incoherent': 1, 'total': 3,
        }

    def test_counts_scoped_by_member_source(self, library: Library) -> None:
        _add_doc(library, 'd1', content_type='theme')
        _add_doc(library, 'd2', content_type='theme')
        _add_doc(library, 'el-a', content_type='catalog', source_name='src1')
        _add_doc(library, 'el-b', content_type='catalog', source_name='src2')
        library.add_theme(cluster_id='c1', doc_id='d1', member_count=1,
                          resolution=1.0, summary_hash='h', coherent=True)
        library.add_theme(cluster_id='c2', doc_id='d2', member_count=1,
                          resolution=1.0, summary_hash='h', coherent=False)
        library.set_theme_members('c1', [('el-a', 1.0)])
        library.set_theme_members('c2', [('el-b', 1.0)])
        assert library.theme_coherence_counts(source='src1') == {
            'coherent': 1, 'incoherent': 0, 'total': 1,
        }
        assert library.theme_coherence_counts(source='src2') == {
            'coherent': 0, 'incoherent': 1, 'total': 1,
        }

    def test_counts_empty_library(self, library: Library) -> None:
        assert library.theme_coherence_counts() == {
            'coherent': 0, 'incoherent': 0, 'total': 0,
        }


# ---------------------------------------------------------------------------
# Theme membership
# ---------------------------------------------------------------------------


class TestThemeMembers:
    def test_set_theme_members_replaces_existing(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        for el in ['el1', 'el2', 'el3']:
            _add_doc(library, el, content_type='catalog')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=2, resolution=1.0, summary_hash='h',
        )

        library.set_theme_members('c1', [('el1', 1.0), ('el2', 0.5)])
        library.set_theme_members('c1', [('el2', 0.8), ('el3', 0.3)])

        member_ids = {m[0] for m in library.get_theme_members('c1')}
        assert member_ids == {'el2', 'el3'}

    def test_get_theme_members_returns_id_weight_pairs(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'el1', content_type='catalog')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h',
        )
        library.set_theme_members('c1', [('el1', 0.7)])
        members = library.get_theme_members('c1')
        assert members == [('el1', 0.7)]

    def test_get_themes_for_element_returns_cluster_ids(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'doc-c2', content_type='theme')
        _add_doc(library, 'el1', content_type='catalog')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h1',
        )
        library.add_theme(
            cluster_id='c2', doc_id='doc-c2',
            member_count=1, resolution=1.0, summary_hash='h2',
        )
        library.set_theme_members('c1', [('el1', 1.0)])
        library.set_theme_members('c2', [('el1', 0.5)])

        cluster_ids = library.get_themes_for_element('el1')

        assert sorted(cluster_ids) == ['c1', 'c2']

    def test_get_themes_for_element_empty_when_no_membership(self, library: Library) -> None:
        assert library.get_themes_for_element('orphan') == []


# ---------------------------------------------------------------------------
# Dirty / clean tracking
# ---------------------------------------------------------------------------


class TestThemeDirtyTracking:
    def test_mark_theme_dirty_flips_flag(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h',
            dirty=False,
        )
        library.mark_theme_dirty('c1')
        theme = library.get_theme('c1')
        assert theme is not None
        assert theme.dirty is True

    def test_mark_themes_clean_clears_flags(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'doc-c2', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h',
            dirty=True,
        )
        library.add_theme(
            cluster_id='c2', doc_id='doc-c2',
            member_count=1, resolution=1.0, summary_hash='h',
            dirty=True,
        )
        library.mark_themes_clean(['c1', 'c2'])

        c1 = library.get_theme('c1')
        c2 = library.get_theme('c2')
        assert c1 is not None and c1.dirty is False
        assert c2 is not None and c2.dirty is False

    def test_get_dirty_themes_returns_dirty_only(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'doc-c2', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h',
            dirty=True,
        )
        library.add_theme(
            cluster_id='c2', doc_id='doc-c2',
            member_count=1, resolution=1.0, summary_hash='h',
            dirty=False,
        )

        assert sorted(library.get_dirty_themes()) == ['c1']

    def test_update_summary_hash_persists(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='old',
        )
        library.update_summary_hash('c1', 'new')
        theme = library.get_theme('c1')
        assert theme is not None
        assert theme.summary_hash == 'new'


# ---------------------------------------------------------------------------
# Cluster history
# ---------------------------------------------------------------------------


class TestClusterHistory:
    def test_record_cluster_history_persists_mappings(self, library: Library) -> None:
        mappings = [
            ClusterMapping(cluster_id='c1', prev_cluster_id='c0', overlap_ratio=0.8),
            ClusterMapping(cluster_id='c2', prev_cluster_id=None, overlap_ratio=None),
        ]
        library.record_cluster_history(run_id=1, mappings=mappings)

        with library._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT cluster_id, prev_cluster_id, overlap_ratio '
                'FROM cluster_history WHERE run_id = ? ORDER BY cluster_id',
                (1,),
            ).fetchall()
        assert len(rows) == 2
        result = {r[0]: (r[1], r[2]) for r in rows}
        assert result['c1'] == ('c0', 0.8)
        assert result['c2'] == (None, None)

    def test_latest_cluster_run_returns_none_initially(self, library: Library) -> None:
        assert library.latest_cluster_run() is None

    def test_latest_cluster_run_returns_max_run_id(self, library: Library) -> None:
        library.record_cluster_history(
            run_id=1,
            mappings=[ClusterMapping(cluster_id='c1', prev_cluster_id=None, overlap_ratio=None)],
        )
        library.record_cluster_history(
            run_id=3,
            mappings=[ClusterMapping(cluster_id='c2', prev_cluster_id=None, overlap_ratio=None)],
        )
        library.record_cluster_history(
            run_id=2,
            mappings=[ClusterMapping(cluster_id='c3', prev_cluster_id=None, overlap_ratio=None)],
        )

        assert library.latest_cluster_run() == 3


# ---------------------------------------------------------------------------
# Theme dataclass
# ---------------------------------------------------------------------------


class TestForeignKeyEnforcement:
    """Schema declares FK ON DELETE CASCADE; verify it's actually enforced."""

    def test_add_theme_with_nonexistent_doc_id_raises(self, library: Library) -> None:
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            library.add_theme(
                cluster_id='c1',
                doc_id='nonexistent-doc',
                member_count=1,
                resolution=1.0,
                summary_hash='h',
            )

    def test_set_theme_member_with_nonexistent_element_raises(self, library: Library) -> None:
        import sqlite3
        _add_doc(library, 'doc-c1', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h',
        )
        with pytest.raises(sqlite3.IntegrityError):
            library.set_theme_members('c1', [('missing-el', 1.0)])

    def test_delete_document_cascades_to_theme(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h',
        )
        library.delete_document('doc-c1')
        assert library.get_theme('c1') is None

    def test_delete_document_cascades_to_theme_members(self, library: Library) -> None:
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'el1', content_type='catalog')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h',
        )
        library.set_theme_members('c1', [('el1', 1.0)])
        # Cascade chain: delete doc → theme deleted → members deleted.
        library.delete_document('doc-c1')
        assert library.get_themes_for_element('el1') == []

    def test_duplicate_cluster_id_raises(self, library: Library) -> None:
        import sqlite3
        _add_doc(library, 'doc-c1', content_type='theme')
        _add_doc(library, 'doc-c2', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h1',
        )
        with pytest.raises(sqlite3.IntegrityError):
            library.add_theme(
                cluster_id='c1', doc_id='doc-c2',
                member_count=1, resolution=1.0, summary_hash='h2',
            )

    def test_unique_doc_id_constraint(self, library: Library) -> None:
        """themes.doc_id is UNIQUE — two themes can't share a doc."""
        import sqlite3
        _add_doc(library, 'doc-c1', content_type='theme')
        library.add_theme(
            cluster_id='c1', doc_id='doc-c1',
            member_count=1, resolution=1.0, summary_hash='h1',
        )
        with pytest.raises(sqlite3.IntegrityError):
            library.add_theme(
                cluster_id='c2', doc_id='doc-c1',  # same doc_id
                member_count=1, resolution=1.0, summary_hash='h2',
            )


class TestSchemaMigrationOnExistingDb:
    """A user upgrading from a pre-Themes Ariadne should pick up the new
    themes / theme_members / cluster_history tables on re-open without
    losing existing data. CREATE TABLE IF NOT EXISTS is the load-bearing
    primitive; this test proves it actually carries the load.
    """

    def test_reopening_db_after_dropping_theme_tables_restores_them(
        self, tmp_path: Path,
    ) -> None:
        import sqlite3

        db_path = tmp_path / 'preexisting.db'

        # 1. Initial Library creates all tables and writes a document.
        lib1 = Library(db_path)
        try:
            lib1.add_document(
                content_type='catalog',
                title='legacy',
                content='legacy content',
                source_files=[],
                metadata={'kind': 'element'},
                doc_id='legacy1',
            )
        finally:
            lib1.close()

        # 2. Simulate a pre-Themes DB by dropping the new tables.
        conn = sqlite3.connect(db_path)
        try:
            conn.execute('DROP TABLE IF EXISTS theme_members')
            conn.execute('DROP TABLE IF EXISTS themes')
            conn.execute('DROP TABLE IF EXISTS cluster_history')
            conn.commit()
        finally:
            conn.close()

        # 3. Re-open Library — __attrs_post_init__ should re-create tables.
        lib2 = Library(db_path)
        try:
            # Existing document survived.
            doc = lib2.get_document('legacy1')
            assert doc is not None
            assert doc.title == 'legacy'

            # New themes operations work cleanly.
            lib2.add_document(
                content_type='theme', title='t1', content='x',
                source_files=[], metadata={}, doc_id='theme-d1',
            )
            lib2.add_theme(
                cluster_id='c1', doc_id='theme-d1',
                member_count=0, resolution=1.0, summary_hash='h',
            )
            assert lib2.get_theme('c1') is not None
        finally:
            lib2.close()

    def test_init_is_idempotent_on_repeated_opens(self, tmp_path: Path) -> None:
        """Opening the same Library multiple times should not error or
        duplicate schema state.
        """
        db_path = tmp_path / 'repeat.db'

        lib1 = Library(db_path)
        lib1.close()

        lib2 = Library(db_path)
        try:
            lib2.add_document(
                content_type='theme', title='t', content='x',
                source_files=[], metadata={}, doc_id='theme-x',
            )
            lib2.add_theme(
                cluster_id='x', doc_id='theme-x',
                member_count=0, resolution=1.0, summary_hash='h',
            )
        finally:
            lib2.close()

        lib3 = Library(db_path)
        try:
            assert lib3.get_theme('x') is not None
        finally:
            lib3.close()


class TestThemeDataclass:
    def test_theme_is_frozen(self) -> None:
        import attrs
        # @frozen attrs class — assignment raises FrozenInstanceError.
        t = Theme(
            cluster_id='c1',
            doc_id='d1',
            member_count=1,
            resolution=1.0,
            last_built_at='2026-04-27T00:00:00',
            last_summarized_at='2026-04-27T00:00:00',
            summary_hash='h',
        )
        with pytest.raises(attrs.exceptions.FrozenInstanceError):
            t.cluster_id = 'c2'  # type: ignore[misc]

    def test_theme_defaults(self) -> None:
        t = Theme(
            cluster_id='c1',
            doc_id='d1',
            member_count=1,
            resolution=1.0,
            last_built_at='2026-04-27T00:00:00',
            last_summarized_at='2026-04-27T00:00:00',
            summary_hash='h',
        )
        # coherent and dirty have defaults
        assert t.coherent is True
        assert t.dirty is False
