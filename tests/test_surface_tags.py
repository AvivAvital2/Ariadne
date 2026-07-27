"""Slice 3 — surface tags (designs/spool-lens-router.md §4, north star §3).

The recipe declares the target kind's interaction surfaces as keyword
vocabularies; a deterministic tag pass writes (doc, surface) rows for the
spool's DOC-GRADE docs, the pack ships them, and the question side matches
the same vocabularies — the surface tier's bridge when no entity is crisp.
Deterministic stems only (word-start bounded), no embeddings, no LLM.
Synthetic fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import library.surface_tags as surface_tags
from library import Library

SURFACES = {
    'serialization': ['serializ', 'pickle', 'arrow'],
    'memory': ['memory', 'cache', 'spill'],
}


class TestMatchSurfaces:
    def test_stem_matching_word_start_bounded(self):
        got = surface_tags.match_surfaces(
            ['Data Serialization Tuning Guide'], SURFACES)
        assert got == {'serialization'}
        # 'serializ' is a stem: matches 'serializer'/'serialization', but
        # never mid-word ('deserializer' has its own word start... a word
        # STARTING with the stem is required).
        assert surface_tags.match_surfaces(
            ['the pickled payload'], SURFACES) == {'serialization'}
        assert surface_tags.match_surfaces(
            ['ram and caches'], SURFACES) == {'memory'}
        assert surface_tags.match_surfaces(
            ['unrelated topic'], SURFACES) == set()

    def test_multiple_parts_and_surfaces(self):
        got = surface_tags.match_surfaces(
            ['Arrow batches', 'spill to disk'], SURFACES)
        assert got == {'serialization', 'memory'}

    def test_question_side_matching(self):
        got = surface_tags.surfaces_for_question(
            'How should I cache intermediate results?', SURFACES)
        assert got == ['memory']
        assert surface_tags.surfaces_for_question(
            'Anything about governance?', SURFACES) == []


class TestTagPassAndQuery:
    def test_tags_doc_grade_docs_and_queries_back(self, tmp_path: Path):
        with Library(tmp_path / 'st.db') as lib:
            prose = lib.add_document(
                'explanation', 'Serialization Tuning', 'body',
                source_name='spool:env1', _allow_reserved_source=True,
            )
            section = lib.add_document(
                'catalog', 'docs.guide.cache_management',
                'md_section docs.guide.cache_management :: ## Cache management',
                metadata={'kind': 'element', 'subtype': 'md_section',
                          'source_name': 'env1',
                          'qualified_name': 'docs.guide.cache_management'},
                source_name='spool:env1', _allow_reserved_source=True,
            )
            lib.add_document(   # generated element stub: NOT tagged
                'catalog', 'pkg.mod.PickleHelper',
                'class pkg.mod.PickleHelper :: class PickleHelper',
                metadata={'kind': 'element', 'subtype': 'class',
                          'source_name': 'env1',
                          'qualified_name': 'pkg.mod.PickleHelper'},
                source_name='spool:env1', _allow_reserved_source=True,
            )
            lib.add_document(   # other source: never tagged
                'explanation', 'Serialization Elsewhere', 'body',
                source_name='otherenv',
            )
            n = surface_tags.tag_surfaces_for_source(
                lib, outer_sources=['spool:env1'], corpus_source='env1',
                surfaces=SURFACES)
            assert n == 2
            with lib._conn_provider.acquire() as conn:
                ser_docs = surface_tags.docs_for_surfaces(
                    conn, ['env1'], ['serialization'])
                assert ser_docs == {prose.id}
                mem_docs = surface_tags.docs_for_surfaces(
                    conn, ['env1'], ['memory'])
                assert mem_docs == {section.id}
                assert surface_tags.docs_for_surfaces(
                    conn, ['env1'], ['serialization', 'memory'],
                ) == {prose.id, section.id}
                # Idempotent re-tag.
                surface_tags.tag_surfaces_for_source(
                    lib, outer_sources=['spool:env1'], corpus_source='env1',
                    surfaces=SURFACES)
                total = conn.execute(
                    'SELECT COUNT(*) FROM surface_tags').fetchone()[0]
                assert total == 2
