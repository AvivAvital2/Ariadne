"""Tests for ``Library.migrate_doc_ids`` — the UUID4 → deterministic-UUID5 migration.

The orchestrator now writes deterministic IDs (``doc_id_for(source, type, primary_key)``)
so re-running ``ariadne generate`` updates the same row instead of inserting a duplicate.
This migration backfills the same scheme for legacy docs that were written under the
old UUID4 scheme.

Behavior the migration must preserve:
  - Per-file docs: keyed on path-relative-to-source-root (``primary_file``).
  - Group docs: keyed on ``"group:<package_name>"``.
  - Topic docs: keyed on ``"topic:<topic_title>"``.
  - When two legacy docs map to the same deterministic ID (the bug we hit:
    "Llm Architecture" generated twice for two different ``llm`` modules),
    the newest doc wins; older losers are deleted with their chunks/sections.
  - All FK / reference columns get rewritten: ``chunks.document_id``,
    ``sections.document_id``, ``themes.doc_id``, ``theme_members.element_id``,
    ``doc_graph.{source_id,target_id}``.
  - ``catalog`` / ``theme`` / ``finding`` content types use a different ID scheme
    (``generate_deterministic_id``) and must not be touched.
  - The migration is idempotent: a second run is a no-op.
"""
from __future__ import annotations

import sqlite3
import uuid

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_doc(
    library,
    *,
    doc_id: str | None = None,
    content_type: str = 'explanation',
    title: str = 'Doc',
    source_files: list[str] | None = None,
    source_name: str | None = 'mylib',
    metadata: dict | None = None,
):
    """Insert a doc with a known (legacy UUID4 by default) ID."""
    if doc_id is None:
        doc_id = str(uuid.uuid4())
    if source_files is None:
        source_files = ['x.py']
    return library.add_document(
        content_type=content_type,
        title=title,
        content=f'content for {title}',
        source_files=source_files,
        metadata=metadata or {},
        doc_id=doc_id,
        source_name=source_name,
    )


def _add_chunk(library, doc_id: str, idx: int = 0, text: str = 'chunk'):
    from schema import Chunk
    library.add_chunk(Chunk(
        document_id=doc_id, chunk_index=idx, content=text,
    ))


def _count(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    return conn.execute(sql, params).fetchone()[0]


@pytest.fixture
def lib(tmp_path):
    from library import Library
    library = Library(tmp_path / 'test.db')
    try:
        yield library
    finally:
        library.close()


@pytest.fixture
def source_map(tmp_path):
    """Pretend ariadne.yaml has one source 'mylib' rooted at tmp_path/mylib."""
    src = tmp_path / 'mylib'
    src.mkdir()
    return {'mylib': src}


# ---------------------------------------------------------------------------
# Core remap
# ---------------------------------------------------------------------------


def test_migrate_remaps_legacy_uuid4_to_deterministic(lib, source_map):
    """A doc with a UUID4 id is rewritten to ``doc_id_for(source, type, rel_path)``."""
    from schema import doc_id_for

    src_path = source_map['mylib']
    target_file = src_path / 'a.py'
    target_file.write_text('# a')

    legacy_id = '11111111-1111-1111-1111-111111111111'
    _seed_doc(
        lib, doc_id=legacy_id,
        content_type='explanation', title='A',
        source_files=[str(target_file)],
    )

    expected_new = doc_id_for('mylib', 'explanation', 'a.py')
    assert legacy_id != expected_new

    result = lib.migrate_doc_ids(source_name_to_path=source_map)

    assert result.remapped == 1
    assert result.duplicates_collapsed == 0
    assert lib.get_document(legacy_id) is None
    assert lib.get_document(expected_new) is not None


def test_migrate_cascades_chunks_and_sections(lib, source_map):
    """Chunks and sections rows must follow the new ID after rewrite."""
    from schema import Section

    src_path = source_map['mylib']
    target_file = src_path / 'b.py'
    target_file.write_text('# b')

    legacy_id = '22222222-2222-2222-2222-222222222222'
    _seed_doc(lib, doc_id=legacy_id, source_files=[str(target_file)], title='B')
    _add_chunk(lib, legacy_id, 0, 'c0')
    _add_chunk(lib, legacy_id, 1, 'c1')
    lib.store_sections(legacy_id, [
        Section(document_id=legacy_id, index=0, heading='H', description='d', content='## H'),
    ])

    lib.migrate_doc_ids(source_name_to_path=source_map)

    with lib._conn_provider.acquire() as conn:
        # No row left under the legacy id.
        assert _count(conn, 'SELECT COUNT(*) FROM chunks WHERE document_id=?', (legacy_id,)) == 0
        assert _count(conn, 'SELECT COUNT(*) FROM sections WHERE document_id=?', (legacy_id,)) == 0

        # And the new id has them.
        new_id = conn.execute('SELECT id FROM documents').fetchone()[0]
        assert _count(conn, 'SELECT COUNT(*) FROM chunks WHERE document_id=?', (new_id,)) == 2
        assert _count(conn, 'SELECT COUNT(*) FROM sections WHERE document_id=?', (new_id,)) == 1


def test_migrate_cascades_doc_graph_edges(lib, source_map):
    """``doc_graph`` has no FK; the migration must rewrite both endpoints by hand."""
    src_path = source_map['mylib']
    f1 = src_path / 'left.py'; f1.write_text('# l')
    f2 = src_path / 'right.py'; f2.write_text('# r')

    left = '33333333-3333-3333-3333-333333333333'
    right = '44444444-4444-4444-4444-444444444444'
    _seed_doc(lib, doc_id=left, source_files=[str(f1)], title='Left')
    _seed_doc(lib, doc_id=right, source_files=[str(f2)], title='Right')

    with lib._conn_provider.acquire() as conn:
        conn.execute(
            'INSERT INTO doc_graph (source_id, target_id, edge_type, weight) VALUES (?,?,?,?)',
            (left, right, 'imports', 1.0),
        )

    lib.migrate_doc_ids(source_name_to_path=source_map)

    from schema import doc_id_for
    new_left = doc_id_for('mylib', 'explanation', 'left.py')
    new_right = doc_id_for('mylib', 'explanation', 'right.py')

    with lib._conn_provider.acquire() as conn:
        row = conn.execute(
            'SELECT source_id, target_id FROM doc_graph WHERE edge_type=?',
            ('imports',),
        ).fetchone()
    assert row == (new_left, new_right)


def test_migrate_collapses_duplicates_keeping_newest(lib, source_map):
    """Two legacy docs that resolve to the same deterministic ID collapse to one
    (the newest by ``updated_at``); the older's row + chunks/sections vanish.
    """
    from schema import doc_id_for

    src_path = source_map['mylib']
    target_file = src_path / 'shared.py'
    target_file.write_text('# s')

    older = '55555555-5555-5555-5555-555555555555'
    newer = '66666666-6666-6666-6666-666666666666'

    # Insert older first, then newer — newer wins by updated_at.
    _seed_doc(lib, doc_id=older, source_files=[str(target_file)], title='Older')
    _add_chunk(lib, older, 0, 'older-chunk')

    _seed_doc(lib, doc_id=newer, source_files=[str(target_file)], title='Newer')
    _add_chunk(lib, newer, 0, 'newer-chunk')

    expected_new = doc_id_for('mylib', 'explanation', 'shared.py')

    result = lib.migrate_doc_ids(source_name_to_path=source_map)
    assert result.duplicates_collapsed == 1
    assert result.remapped == 1  # only the survivor was renamed

    surviving = lib.get_document(expected_new)
    assert surviving is not None
    assert surviving.title == 'Newer'

    with lib._conn_provider.acquire() as conn:
        # Total docs: 1 (winner) under the new deterministic id.
        assert _count(conn, 'SELECT COUNT(*) FROM documents') == 1
        # Older's chunk should be gone; newer's chunk should follow the new id.
        assert _count(conn, 'SELECT COUNT(*) FROM chunks WHERE document_id=?', (older,)) == 0
        assert _count(conn, 'SELECT COUNT(*) FROM chunks WHERE document_id=?', (expected_new,)) == 1


def test_migrate_is_idempotent(lib, source_map):
    """A second migrate run after a clean one is a no-op (everything is already
    deterministic).
    """
    from schema import doc_id_for

    src_path = source_map['mylib']
    target_file = src_path / 'idem.py'
    target_file.write_text('# i')

    legacy_id = '77777777-7777-7777-7777-777777777777'
    _seed_doc(lib, doc_id=legacy_id, source_files=[str(target_file)], title='I')

    lib.migrate_doc_ids(source_name_to_path=source_map)

    second = lib.migrate_doc_ids(source_name_to_path=source_map)
    assert second.remapped == 0
    assert second.duplicates_collapsed == 0
    assert lib.get_document(doc_id_for('mylib', 'explanation', 'idem.py')) is not None


def test_migrate_dry_run_makes_no_changes(lib, source_map):
    """With dry_run=True the result reports counts but no rows are touched."""
    src_path = source_map['mylib']
    target_file = src_path / 'd.py'
    target_file.write_text('# d')

    legacy_id = '88888888-8888-8888-8888-888888888888'
    _seed_doc(lib, doc_id=legacy_id, source_files=[str(target_file)], title='D')

    result = lib.migrate_doc_ids(source_name_to_path=source_map, dry_run=True)

    assert result.remapped == 1  # plan reports it
    assert lib.get_document(legacy_id) is not None  # ...but row still there
    with lib._conn_provider.acquire() as conn:
        assert _count(conn, 'SELECT COUNT(*) FROM documents WHERE id=?', (legacy_id,)) == 1


# ---------------------------------------------------------------------------
# Skips
# ---------------------------------------------------------------------------


def test_migrate_skips_unsupported_content_types(lib, source_map):
    """``catalog`` / ``theme`` / ``finding`` use ``generate_deterministic_id`` —
    leave them alone.
    """
    src_path = source_map['mylib']
    target_file = src_path / 'cat.py'
    target_file.write_text('# c')

    cat_id = '99999999-9999-9999-9999-999999999999'
    _seed_doc(
        lib, doc_id=cat_id, content_type='catalog',
        source_files=[str(target_file)], title='Cat',
    )
    fnd_id = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
    _seed_doc(
        lib, doc_id=fnd_id, content_type='finding',
        source_files=[str(target_file)], title='Finding',
    )

    lib.migrate_doc_ids(source_name_to_path=source_map)

    assert lib.get_document(cat_id) is not None
    assert lib.get_document(fnd_id) is not None


def test_migrate_skips_docs_without_source_name(lib, source_map):
    """A doc with no source_name (or one not in the map) is reported as skipped."""
    src_path = source_map['mylib']
    target_file = src_path / 'orphan.py'
    target_file.write_text('# o')

    legacy_id = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
    _seed_doc(
        lib, doc_id=legacy_id, source_files=[str(target_file)],
        title='Orphan', source_name=None,
    )

    result = lib.migrate_doc_ids(source_name_to_path=source_map)
    assert result.skipped_no_source == 1
    assert result.remapped == 0
    assert lib.get_document(legacy_id) is not None


# ---------------------------------------------------------------------------
# Group / topic primary keys
# ---------------------------------------------------------------------------


def test_migrate_uses_group_primary_key_when_metadata_says_group(lib, source_map):
    """A package-level architecture doc is keyed on ``group:<package_name>``."""
    from schema import doc_id_for

    src_path = source_map['mylib']
    legacy_id = 'cccccccc-cccc-cccc-cccc-cccccccccccc'
    _seed_doc(
        lib, doc_id=legacy_id, content_type='architecture',
        title='docgen Architecture',
        source_files=[str(src_path / 'docgen' / '__init__.py')],
        metadata={'group': True, 'package_name': 'docgen'},
    )

    lib.migrate_doc_ids(source_name_to_path=source_map)

    expected = doc_id_for('mylib', 'architecture', 'group:docgen')
    assert lib.get_document(expected) is not None
    assert lib.get_document(legacy_id) is None


def test_migrate_uses_topic_primary_key_when_metadata_says_topic(lib, source_map):
    """A cross-cutting topic doc is keyed on ``topic:<topic_title>``."""
    from schema import doc_id_for

    legacy_id = 'dddddddd-dddd-dddd-dddd-dddddddddddd'
    _seed_doc(
        lib, doc_id=legacy_id, content_type='explanation',
        title='Themes Pipeline',
        source_files=[],  # topics don't necessarily map to one file
        metadata={'topic': True, 'topic_title': 'Themes Pipeline'},
    )

    lib.migrate_doc_ids(source_name_to_path=source_map)

    expected = doc_id_for('mylib', 'explanation', 'topic:Themes Pipeline')
    assert lib.get_document(expected) is not None
    assert lib.get_document(legacy_id) is None


# ---------------------------------------------------------------------------
# Theme membership cascade
# ---------------------------------------------------------------------------


def test_migrate_rewrites_theme_members_element_id(lib, source_map):
    """If a doc is a theme member, ``theme_members.element_id`` must point to the
    new deterministic id after migration.
    """
    from schema import doc_id_for, generate_deterministic_id

    src_path = source_map['mylib']
    target_file = src_path / 'tm.py'
    target_file.write_text('# tm')

    legacy_id = 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee'
    _seed_doc(lib, doc_id=legacy_id, source_files=[str(target_file)], title='TM')

    # Theme + placeholder doc (theme docs use generate_deterministic_id).
    cluster = 'C1'
    theme_doc = generate_deterministic_id('theme', cluster)
    lib.add_document(
        content_type='theme', title=f'Theme {cluster}',
        content='(pending)', source_files=[],
        metadata={'cluster_id': cluster},
        doc_id=theme_doc,
    )
    lib.add_theme(
        cluster_id=cluster, doc_id=theme_doc, member_count=1,
        resolution=1.0, summary_hash='', coherent=True, dirty=True,
    )
    lib.set_theme_members(cluster, [(legacy_id, 1.0)])

    lib.migrate_doc_ids(source_name_to_path=source_map)

    new_id = doc_id_for('mylib', 'explanation', 'tm.py')
    members = lib.get_theme_members(cluster)
    assert members == [(new_id, 1.0)]
