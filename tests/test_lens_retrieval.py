"""Slice 1 scoped retrieval — designs/spool-lens-router.md §5.

P13 throughout: admission is categorical (entity matches) or an explicit
precision gate over DOC-GRADE candidates; similarity ranks, and
similarity-to-the-repo never gates. Synthetic fixtures only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import library.lens_retrieval as lens_retrieval
from library import Library
from library.embedding_matrix import EmbeddingMatrix
from library.lens_router import EntityHit
from schema import Section


@pytest.fixture
def lib(tmp_path: Path):
    library = Library(tmp_path / 'lens-retrieval-test.db')
    yield library
    library.close()


def _spool_doc(lib, doc_id, title, *, content_type='explanation',
               metadata=None):
    return lib.add_document(
        content_type=content_type, title=title, content=f'body of {title}',
        source_files=[], metadata=metadata or {}, doc_id=doc_id,
        source_name='spool:env1', _allow_reserved_source=True,
    )


def _matrix(rows):
    ids = list(rows)
    return EmbeddingMatrix(
        matrix=np.stack([np.asarray(v, dtype=np.float32)
                         for v in rows.values()]),
        ids=ids, dim=len(next(iter(rows.values()))),
        build_stamp='synthetic',
    )


class TestDocGradeCandidates:
    def test_prose_and_md_sections_in_stubs_out(self, lib) -> None:
        _spool_doc(lib, 'd-prose', 'Mesh Tuning Guide')
        _spool_doc(lib, 'd-section', 'docs.guide.enable_flux',
                   content_type='catalog',
                   metadata={'kind': 'element', 'subtype': 'md_section',
                             'source_name': 'env1',
                             'qualified_name': 'docs.guide.enable_flux'})
        _spool_doc(lib, 'd-stub', 'pkg.mod.helper_fn',
                   content_type='catalog',
                   metadata={'kind': 'element', 'subtype': 'function',
                             'source_name': 'env1',
                             'qualified_name': 'pkg.mod.helper_fn'})
        ids = lens_retrieval.doc_grade_spool_candidates(
            lib, ['spool:env1'])
        assert set(ids) == {'d-prose', 'd-section'}


class TestDocsForEntityHits:
    def test_title_heading_and_symbol_hits_map_to_docs(self, lib) -> None:
        title_doc = _spool_doc(lib, 'd-title', 'Quantum Mesh Overview')
        head_doc = _spool_doc(lib, 'd-head', 'Operations Handbook')
        lib.store_sections(head_doc.id, [
            Section(document_id=head_doc.id, index=0,
                    heading='Shared Ledger Limits', description='d',
                    content='c'),
        ])
        _spool_doc(lib, 'd-elem', 'lake.core.FrobnicateSelector',
                   content_type='catalog',
                   metadata={'kind': 'element', 'subtype': 'class',
                             'source_name': 'env1',
                             'qualified_name': 'lake.core.FrobnicateSelector'})

        got = lens_retrieval.docs_for_entity_hits(
            lib, ['spool:env1'],
            [EntityHit('Quantum Mesh', 'title', 'product'),
             EntityHit('shared ledger', 'heading', 'phrase'),
             EntityHit('FrobnicateSelector', 'symbol', 'api')],
        )
        assert got['d-title'] == 'Quantum Mesh'
        assert got['d-head'] == 'shared ledger'
        # A symbol hit admits its ELEMENT doc even though stubs are not
        # doc-grade — entity admission is categorical (P13), the doc-grade
        # predicate filters only the SEMANTIC candidate pool.
        assert got['d-elem'] == 'FrobnicateSelector'

    def test_md_section_heading_hits_admit_the_section_element(self, lib) -> None:
        _spool_doc(lib, 'd-mdsec', 'docs.guide.enable_flux_capacitor',
                   content_type='catalog',
                   metadata={'kind': 'element', 'subtype': 'md_section',
                             'source_name': 'env1',
                             'qualified_name':
                                 'docs.guide.enable_flux_capacitor'})
        got = lens_retrieval.docs_for_entity_hits(
            lib, ['spool:env1'],
            [EntityHit('flux capacitor', 'heading', 'phrase')],
        )
        assert got['d-mdsec'] == 'flux capacitor'


class TestSelectSpoolDocs:
    def test_entity_docs_first_then_gated_cosine_fill(self, lib) -> None:
        _spool_doc(lib, 'd-entity', 'Quantum Mesh Overview')
        _spool_doc(lib, 'd-near', 'Mesh Compaction Deep Dive')
        _spool_doc(lib, 'd-far', 'Unrelated Billing Notes')
        matrix = _matrix({
            'd-entity': (1.0, 0.0),
            'd-near': (0.9, np.sqrt(1 - 0.81)),
            'd-far': (0.0, 1.0),
        })
        query = np.array([1.0, 0.0], dtype=np.float32)
        picked = lens_retrieval.select_spool_docs(
            lib, matrix, ['spool:env1'],
            [EntityHit('Quantum Mesh', 'title', 'product')],
            query_embedding=query, limit=4,
        )
        by_id = {c.doc_id: c for c in picked}
        assert list(by_id)[0] == 'd-entity'
        assert by_id['d-entity'].connection == 'entity'
        assert by_id['d-entity'].detail == 'Quantum Mesh'
        assert by_id['d-near'].connection == 'semantic'   # 0.9 ≥ gate
        assert 'd-far' not in by_id                       # 0.0 < gate

    def test_entity_admissions_capped_and_cosine_ordered(self, lib) -> None:
        # ONLINE-ACCEPTANCE REGRESSION: shallow name-matches (readme/code-of-
        # conduct style docs matching 'Quantum Mesh' by title) saturated the
        # whole window, so the semantic fill never ran and the high-cosine
        # ground-truth doc never surfaced. Entity admission stays categorical
        # but is CAPPED (limit//2) and cosine-ordered when an embedding is
        # available; the fill always gets its reserved share.
        _spool_doc(lib, 'e-weak', 'Quantum Mesh Code of Conduct')
        _spool_doc(lib, 'e-mid', 'Quantum Mesh Readme Notes')
        _spool_doc(lib, 'e-best', 'Quantum Mesh Overview')
        _spool_doc(lib, 's-truth', 'Ledger Compaction Deep Dive')
        matrix = _matrix({
            'e-weak': (0.2, np.sqrt(1 - 0.04)),
            'e-mid': (0.4, np.sqrt(1 - 0.16)),
            'e-best': (0.6, 0.8),
            's-truth': (0.9, np.sqrt(1 - 0.81)),
        })
        query = np.array([1.0, 0.0], dtype=np.float32)
        picked = lens_retrieval.select_spool_docs(
            lib, matrix, ['spool:env1'],
            [EntityHit('Quantum Mesh', 'title', 'product')],
            query_embedding=query, limit=3,
        )
        ids = [c.doc_id for c in picked]
        assert 's-truth' in ids                      # the fill ran
        assert picked[0].doc_id == 'e-best'          # entity slot = best cosine
        by_id = {c.doc_id: c for c in picked}
        assert by_id['s-truth'].connection == 'semantic'
        n_entity = sum(1 for c in picked if c.connection == 'entity')
        assert n_entity <= 2                         # cap held (limit 3 -> 1+backfill)

    def test_without_query_embedding_entity_only(self, lib) -> None:
        _spool_doc(lib, 'd-entity', 'Quantum Mesh Overview')
        _spool_doc(lib, 'd-near', 'Mesh Compaction Deep Dive')
        matrix = _matrix({'d-entity': (1.0, 0.0), 'd-near': (0.9, 0.436)})
        picked = lens_retrieval.select_spool_docs(
            lib, matrix, ['spool:env1'],
            [EntityHit('Quantum Mesh', 'title', 'product')],
            query_embedding=None, limit=4,
        )
        assert [c.doc_id for c in picked] == ['d-entity']


class TestFallbackSpoolDocs:
    def test_two_hop_gate_admits_only_above_threshold(self, lib) -> None:
        seed = lib.add_document(
            content_type='explanation', title='repo io doc', content='b',
            source_name='src1', doc_id='r-seed',
        )
        _spool_doc(lib, 'd-close', 'Mesh IO Guide')
        _spool_doc(lib, 'd-borderline', 'Mesh Sidecar Notes')
        _spool_doc(lib, 'd-far', 'Unrelated Billing Notes')
        matrix = _matrix({
            'r-seed': (1.0, 0.0),
            'd-close': (0.8, 0.6),        # cos 0.8
            'd-borderline': (0.5, np.sqrt(0.75)),   # cos 0.50 < gate
            'd-far': (0.0, 1.0),
        })
        picked = lens_retrieval.fallback_spool_docs(
            lib, matrix, ['spool:env1'], [seed.id], limit=3,
        )
        assert [c.doc_id for c in picked] == ['d-close']
        assert picked[0].connection == 'semantic'
        assert float(picked[0].detail) >= lens_retrieval.SPOOL_FALLBACK_GATE

    def test_restrict_to_scopes_the_fallback_candidates(self, lib) -> None:
        # The surface tier's consumption: when the question resolves to
        # surfaces, the fallback pool is RESTRICTED to surface-tagged docs —
        # categorical scoping (P13), similarity still ranks inside it.
        seed = lib.add_document(
            content_type='explanation', title='repo io doc', content='b',
            source_name='src1', doc_id='r-seed',
        )
        _spool_doc(lib, 'd-tagged', 'Mesh IO Guide')
        _spool_doc(lib, 'd-untagged', 'Mesh Sidecar Notes')
        matrix = _matrix({
            'r-seed': (1.0, 0.0),
            'd-tagged': (0.8, 0.6),
            'd-untagged': (0.9, np.sqrt(1 - 0.81)),   # closer, but off-surface
        })
        picked = lens_retrieval.fallback_spool_docs(
            lib, matrix, ['spool:env1'], [seed.id], limit=3,
            restrict_to={'d-tagged'},
        )
        assert [c.doc_id for c in picked] == ['d-tagged']

    def test_no_seeds_no_fallback(self, lib) -> None:
        _spool_doc(lib, 'd-close', 'Mesh IO Guide')
        matrix = _matrix({'d-close': (1.0, 0.0)})
        assert lens_retrieval.fallback_spool_docs(
            lib, matrix, ['spool:env1'], [], limit=3) == []
