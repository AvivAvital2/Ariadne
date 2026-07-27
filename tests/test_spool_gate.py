"""Slice (c1) of the Spool plugin: the scarcity gate.

Tier-2 (official) docs stay OUT of the default candidate set; the gate
opens only when the tier-1 (code) layer runs thin, and then a re-rank
includes tier-2 — fill, don't dilute. Constants are provisional
parameters (calibration = slice c2, needs real score distributions).
Design: designs/spool-environment-plugin.md §18.6.3.
"""
import asyncio
from types import SimpleNamespace

import numpy as np

from ariadne_mcp.service import AriadneService
from config import CODE_PROVENANCE, OFFICIAL_DOC_PROVENANCE
from library import Library
from spools import (
    partition_tier2,
    rank_with_scarcity_gate,
    scarcity_gate_open,
    spool_gap_hint,
)


def _doc(doc_id, provenance=None, source_name='spool-src'):
    metadata = {} if provenance is None else {'provenance': provenance}
    return SimpleNamespace(id=doc_id, metadata=metadata,
                           source_name=source_name)


class TestScarcityGate:
    def test_gate_lifecycle(self):
        # Demand 1 (HIGH-1) — partition gates on BOTH axes: only
        # SPOOL-ORIGIN official docs leave the candidate set. A user's own
        # doc tagged 'official' (source not a spool) stays tier-1, and so
        # do code/human-doc.
        docs = [
            _doc('code-1'),
            _doc('official-1', 'official'),
            _doc('human-1', 'human-doc'),
            _doc('official-2', 'official'),
            _doc('user-official', 'official', source_name='my-code'),
        ]
        tier1, tier2 = partition_tier2(docs, spool_sources={'spool-src'})
        assert [d.id for d in tier1] == [
            'code-1', 'human-1', 'user-official',
        ]
        assert [d.id for d in tier2] == ['official-1', 'official-2']
        # No spools active -> nothing is tier-2, whatever the provenance.
        t1, t2 = partition_tier2(docs, spool_sources=frozenset())
        assert t2 == []

        # Demand 2 — the predicate: open on empty / zero-top; open when
        # fewer than N hits reach τ·top; closed at N strong hits.
        assert scarcity_gate_open([]) is True
        assert scarcity_gate_open([('a', 0.0)]) is True
        dense = [('a', 1.0), ('b', 0.95), ('c', 0.9), ('d', 0.2)]
        assert scarcity_gate_open(
            dense, min_strong_hits=3, relative_threshold=0.85,
        ) is False
        sparse = [('a', 1.0), ('b', 0.5), ('c', 0.4)]
        assert scarcity_gate_open(
            sparse, min_strong_hits=3, relative_threshold=0.85,
        ) is True

        # Demand 3 — the rank wrapper: dense tier-1 never ranks tier-2;
        # scarce tier-1 re-ranks with tier-2 included; without tier-2 the
        # gate never triggers a second call.
        calls = []

        def make_rank(result_by_call):
            async def rank(ids):
                calls.append(list(ids))
                return result_by_call[len(calls) - 1]
            return rank

        calls.clear()
        rank = make_rank([dense])
        result = asyncio.run(rank_with_scarcity_gate(
            rank, ['a', 'b', 'c', 'd'], ['off-1'],
            min_strong_hits=3, relative_threshold=0.85,
        ))
        assert result.ranked == dense
        assert result.gate_opened is False
        assert result.tier2_ranked is False
        assert calls == [['a', 'b', 'c', 'd']]          # tier-2 never ranked

        calls.clear()
        merged = [('off-1', 0.9), ('a', 1.0), ('b', 0.5), ('c', 0.4)]
        rank = make_rank([sparse, merged])
        result = asyncio.run(rank_with_scarcity_gate(
            rank, ['a', 'b', 'c'], ['off-1'],
            min_strong_hits=3, relative_threshold=0.85,
        ))
        assert result.ranked == merged
        assert result.gate_opened is True
        assert result.tier2_ranked is True
        assert calls == [['a', 'b', 'c'], ['a', 'b', 'c', 'off-1']]

        calls.clear()
        rank = make_rank([sparse])
        result = asyncio.run(rank_with_scarcity_gate(
            rank, ['a', 'b', 'c'], [],
            min_strong_hits=3, relative_threshold=0.85,
        ))
        assert result.ranked == sparse
        assert result.gate_opened is True
        assert result.tier2_ranked is False
        assert calls == [['a', 'b', 'c']]               # no tier-2 -> one call

        # Demand (c2a) — the honest-gap hint: emitted ONLY when the gate
        # opened, no tier-2 existed to fill, AND a spool is registered
        # (non-spool projects never see the noise).
        hint = spool_gap_hint(
            gate_opened=True, tier2_present=False, spools_registered=True,
        )
        assert hint is not None and 'official' in hint
        assert spool_gap_hint(
            gate_opened=False, tier2_present=False, spools_registered=True,
        ) is None
        assert spool_gap_hint(
            gate_opened=True, tier2_present=True, spools_registered=True,
        ) is None
        assert spool_gap_hint(
            gate_opened=True, tier2_present=False, spools_registered=False,
        ) is None


class TestScarcityGateSearchIntegration:
    async def test_limit_one_does_not_spuriously_open_gate(
        self, tmp_path, monkeypatch,
    ):
        # HIGH-3 — the gate's scarcity determination must be INDEPENDENT of
        # the caller's requested result count. The candidate window was
        # limit*2; at limit=1 that is 2 < min_strong_hits (3), so the gate
        # ALWAYS saw fewer than 3 hits and opened even with abundant strong
        # code — letting a spool 'official' doc take the single slot. With
        # the window floored at min_strong_hits, abundant code keeps it shut.
        e_q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        lib = Library(tmp_path / 'gate.db')
        try:
            # Four strong code docs (raw sims 0.85–0.88, all within τ of the
            # top) and one spool 'official' doc that is the CLOSEST raw match
            # (1.0). If the gate wrongly opens, the official doc (weighted
            # 1.0 × 0.9 = 0.9) outranks every code doc (≤ 0.88) and wins the
            # single limit=1 slot; if it stays shut, a code doc wins.
            for i, sim in enumerate((0.88, 0.87, 0.86, 0.85)):
                lib.add_document(
                    'explanation', f'Code {i}', f'body {i}',
                    embedding=np.array([sim, 0.0, 0.0], dtype=np.float32),
                    metadata={'provenance': CODE_PROVENANCE},
                )
            lib.add_document(
                'explanation', 'Official', 'official body',
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                metadata={'provenance': OFFICIAL_DOC_PROVENANCE},
                source_name='spool:databricks',
                _allow_reserved_source=True,  # simulate an installed spool doc
            )

            class _FakeEmbed:
                async def embed(self, *_a, **_k):
                    return e_q

            class _FakeResolution:
                def scope_sources(self):
                    return frozenset({'spool:databricks'})

                def fingerprint(self):
                    return 'testfp'

            svc = AriadneService()
            svc._library = lib
            svc._embedding_service = _FakeEmbed()
            monkeypatch.setattr(svc, '_resolve_scope', lambda *a, **k: lib)
            monkeypatch.setattr(
                'spools.resolve_spools', lambda *a, **k: _FakeResolution(),
            )

            resp = await svc._search_uncached(query='deploy', limit=1)
            titles = [d.title for d in resp.documents]
            # Abundant strong code → gate stays closed → the official spool
            # doc is NOT surfaced; the top (only) result is a code doc.
            assert titles, 'expected at least one result'
            assert 'Official' not in titles
            assert titles[0].startswith('Code')
        finally:
            lib.close()
