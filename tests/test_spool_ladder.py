"""Slice (b1) of the Spool plugin: the three-tier trust ladder + scope union.

Evolutionary tests — one per code piece (ranking weights; scope union),
grown demand by demand per IMPLEMENT.md. Synthetic fixtures only.
Design: designs/spool-environment-plugin.md §18.3 (ladder) ·
§18.6.2 (provenance axes) · §18.6.4 (scope semantics).
"""
import textwrap

import numpy as np

from config import Config
from library import Library
from library.embedding_matrix import EmbeddingMatrix
from library.search import provenance_weight
from scope_resolution import make_scoped_library


class TestTrustLadder:
    def test_provenance_ladder(self):
        # Demand 1 — the ladder's weights: code (default) 1.0, official 0.9,
        # human-doc 0.8 — strictly ordered; the stale flag composes.
        code_weight = provenance_weight({})
        official_weight = provenance_weight({'provenance': 'official'})
        human_weight = provenance_weight({'provenance': 'human-doc'})
        assert code_weight == 1.0
        assert official_weight == 0.9
        assert code_weight > official_weight > human_weight
        assert provenance_weight(
            {'provenance': 'official', 'stale_autodoc': True},
        ) == official_weight * 0.6

        # Demand 2 — the ladder through the real ranking path: three docs at
        # EQUAL raw similarity rank code > official > human-doc, and the
        # returned scores are exactly the weighted similarities.
        vector = np.array([1.0, 0.0], dtype=np.float32)
        matrix = EmbeddingMatrix(
            matrix=np.stack([vector, vector, vector]),
            ids=['doc-code', 'doc-official', 'doc-human'],
            dim=2,
            build_stamp='synthetic',
        )
        weights = {
            'doc-code': provenance_weight({}),
            'doc-official': provenance_weight({'provenance': 'official'}),
            'doc-human': provenance_weight({'provenance': 'human-doc'}),
        }
        # rank() normalizes the query and expects pre-normalized rows, so all
        # three docs sit at raw cosine similarity 1.0 — only the ladder
        # separates them.
        query = np.array([0.8, 0.0], dtype=np.float32)
        ranked = matrix.rank(
            query, candidate_ids=matrix.ids, limit=3, weights=weights,
        )
        assert [doc_id for doc_id, _ in ranked] == [
            'doc-code', 'doc-official', 'doc-human',
        ]
        scores = dict(ranked)
        assert scores['doc-code'] == np.float32(1.0)
        assert scores['doc-official'] == np.float32(0.9)
        assert scores['doc-human'] == np.float32(0.8)


class TestSpoolScope:
    def test_scope_unions_registered_spools(self, tmp_path):
        # Demand 3 — a doc in the spool source is visible through the scoped
        # view iff the spool resolves (enabled + cached pack); an enabled
        # spool with no cached pack degrades to exclusion, never a crash.
        src_dir = tmp_path / 'src1'
        src_dir.mkdir()
        config_path = tmp_path / 'ariadne.yaml'
        config_path.write_text(textwrap.dedent(f'''
            sources:
              src1:
                path: {src_dir}
            spools:
              fakebricks:
                runtime: fake-17.3
        '''))
        cfg = Config(config_path=config_path)

        with Library(tmp_path / 'library.db') as library:
            own_doc = library.add_document(
                'explanation', 'own doc', 'body', source_name='src1',
            )
            spool_doc = library.add_document(
                'explanation', 'spool doc', 'body', source_name='spool:fakebricks',
                _allow_reserved_source=True,  # simulate an installed spool doc
            )

            # Enabled but no cached pack -> spool source stays OUT of scope.
            scoped = make_scoped_library(cfg, library, 'src1', use_cwd=False)
            visible = {meta.id for meta in scoped.list_documents_lite()}
            assert visible == {own_doc.id}

            # Cache a matching pack -> the spool source joins the scope.
            manifest_dir = tmp_path / '.ariadne' / 'spools' / 'fakebricks'
            manifest_dir.mkdir(parents=True)
            (manifest_dir / 'manifest.yaml').write_text(textwrap.dedent('''
                environment: fakebricks
                version: '1.0.0'
                target_runtime: fake-17.3
                checksum: abc123
            '''))
            scoped = make_scoped_library(cfg, library, 'src1', use_cwd=False)
            visible = {meta.id for meta in scoped.list_documents_lite()}
            assert visible == {own_doc.id, spool_doc.id}

            # CRIT-3 — a corrupt cached manifest must NOT brick queries:
            # the spool degrades to excluded; the scoped view still works.
            (manifest_dir / 'manifest.yaml').write_text('{{{{ not yaml')
            scoped = make_scoped_library(cfg, library, 'src1', use_cwd=False)
            visible = {meta.id for meta in scoped.list_documents_lite()}
            assert visible == {own_doc.id}
