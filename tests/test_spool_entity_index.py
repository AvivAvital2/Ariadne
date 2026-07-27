"""Slice 1 entity index — three layers over Library, per design §4.

Symbols (kind-filtered: never variables), doc titles, section headings;
the spool side unions the reserved spool:<name> source with its corpus
source id; recipe aliases join the title layer. Synthetic fixtures only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import library.spool_entity_index as entity_index
from library import Library
from schema import Section


@pytest.fixture
def lib(tmp_path: Path):
    library = Library(tmp_path / 'entity-index-test.db')
    yield library
    library.close()


def _add_element(lib, source, qn, subtype='function'):
    lib.add_document(
        content_type='catalog', title=qn, content=f'element {qn}',
        source_files=[f'/src/{source}/mod.py'],
        metadata={'kind': 'element', 'source_name': source,
                  'qualified_name': qn, 'subtype': subtype},
        doc_id=f'{source}:{qn}',
    )


class TestBuildAndResolve:
    def test_three_layers_and_classes(self, lib) -> None:
        _add_element(lib, 'src1', 'pkg.mod.FrobnicateSelector')
        doc = lib.add_document(
            content_type='explanation', title='Quantum Mesh Overview',
            content='body', source_name='src1',
        )
        lib.store_sections(doc.id, [
            Section(document_id=doc.id, index=0, heading='Shared Ledger Limits',
                    description='d', content='c'),
        ])

        idx = entity_index.build_entity_index(lib, ['src1'])

        sym_hits = idx.resolve('FrobnicateSelector')
        assert [(h.layer, h.entity_class) for h in sym_hits] == [
            ('symbol', 'api')]
        title_hits = idx.resolve('Quantum Mesh')
        assert ('title', 'product') in [
            (h.layer, h.entity_class) for h in title_hits]
        heading_hits = idx.resolve('shared ledger')
        assert ('heading', 'phrase') in [
            (h.layer, h.entity_class) for h in heading_hits]

    def test_structural_only_and_distinctive_only(self, lib) -> None:
        _add_element(lib, 'src1', 'pkg.mod.FrobnicateSelector')
        lib.add_document(
            content_type='explanation', title='The pipeline handbook',
            content='body', source_name='src1',
        )
        idx = entity_index.build_entity_index(lib, ['src1'])
        assert idx.resolve('FrobnicateSelectr') == ()   # typo: no route
        assert idx.resolve('pipeline') == ()            # common word: never

    def test_variables_never_resolve(self, lib) -> None:
        _add_element(lib, 'src1', 'pkg.mod.LocalTemp', subtype='variable')
        idx = entity_index.build_entity_index(lib, ['src1'])
        assert idx.resolve('LocalTemp') == ()

    def test_source_union_and_isolation(self, lib) -> None:
        # The spool side unions the corpus source with the reserved
        # spool:<name> id; other sources stay invisible.
        _add_element(lib, 'env1', 'lake.core.QuantumMesh')
        lib.add_document(
            content_type='gotcha', title='Mesh Compaction Gotchas',
            content='body', source_name='spool:env1',
            _allow_reserved_source=True,
        )
        _add_element(lib, 'other', 'other.pkg.ForeignThing')

        idx = entity_index.build_entity_index(lib, ['env1', 'spool:env1'])
        assert idx.resolve('QuantumMesh')
        assert idx.resolve('Mesh Compaction')
        assert idx.resolve('ForeignThing') == ()

    def test_recipe_aliases_join_the_title_layer(self, lib) -> None:
        idx = entity_index.build_entity_index(
            lib, ['env1'], aliases=['Quantum Mesh'])
        hits = idx.resolve('Quantum Mesh')
        assert [(h.layer, h.entity_class) for h in hits] == [
            ('alias', 'product')]

    def test_md_sections_are_headings_not_symbols(self, lib) -> None:
        # Doc-derived md_section elements carry normalized HEADINGS as their
        # qualified names ('docs.guide.enable_flux_capacitor' = the heading
        # "Enable flux capacitor") — they are documentation vocabulary, not
        # API symbols, and must resolve via the heading layer (live
        # regression: 'use liquid clustering' got a SYMBOL hit through an
        # md_section and polluted the routing evidence).
        _add_element(lib, 'env1', 'docs.guide.enable_flux_capacitor',
                     subtype='md_section')
        idx = entity_index.build_entity_index(lib, ['env1'])
        hits = idx.resolve('flux capacitor')
        assert [(h.layer, h.entity_class) for h in hits] == [
            ('heading', 'phrase')]
