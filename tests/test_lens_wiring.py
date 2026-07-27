"""Slice 1 wiring — the lens routes live search (design §6 supersession).

On routed questions the regime REPLACES the anchored floor, the scarcity
gate, and the tier-2 holdback: repo-only keeps the spool silent,
expert-only drops the repo take, fuse carries both. Synthetic fixtures;
service driven like the other integration suites.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ariadne_mcp.service import AriadneService
from library import Library
from library.embedding_matrix import EmbeddingMatrix
from library.surface_tags import (
    SurfaceTagRow,
    init_surface_tags_schema,
    upsert_surface_tags,
)

pytestmark = pytest.mark.asyncio


def _vec(*xs):
    a = np.zeros(3, dtype=np.float32)
    for i, x in enumerate(xs):
        a[i] = x
    return a


class _FakeEmbed:
    async def embed(self, *_a, **_k):
        return _vec(1.0, 0.0, 0.0)


class _FakeResolution:
    def __init__(self, surfaces=None):
        self.registered = {}
        if surfaces:
            self.registered['fakebricks'] = SimpleNamespace(
                manifest=SimpleNamespace(surfaces=surfaces))

    def scope_sources(self):
        return frozenset({'spool:fakebricks'})

    def fingerprint(self):
        return 'fp-lens'


def _service(lib, monkeypatch, surfaces=None):
    svc = AriadneService()
    svc._library = lib
    svc._embedding_service = _FakeEmbed()
    monkeypatch.setattr(svc, '_resolve_scope', lambda *a, **k: lib)
    monkeypatch.setattr(
        'spools.resolve_spools',
        lambda *a, **k: _FakeResolution(surfaces=surfaces))
    return svc


def _seed_store(lib):
    lib.add_document(
        'explanation', 'Combinatorial Pruning Guide', 'repo pruning body',
        embedding=_vec(1, 0, 0), source_name='src1',
    )
    lib.add_document(
        'explanation', 'General Notes', 'repo general body',
        embedding=_vec(1, 0, 0), source_name='src1',
    )
    lib.add_document(
        'explanation', 'Quantum Mesh Overview', 'spool overview body',
        embedding=_vec(1, 0, 0),
        source_name='spool:fakebricks', _allow_reserved_source=True,
    )


class TestLensWiredSearch:
    async def test_repo_only_keeps_spool_silent(self, tmp_path, monkeypatch):
        # Crisp repo signal, no spool signal -> the spool contributes
        # NOTHING (the old anchored path admitted spool ground on plain
        # query similarity).
        lib = Library(tmp_path / 'lens-wiring.db')
        try:
            _seed_store(lib)
            svc = _service(lib, monkeypatch)
            resp = await svc._search_uncached(
                query='How does src1 handle combinatorial pruning?', limit=4)
            titles = [d.title for d in resp.documents]
            assert 'Combinatorial Pruning Guide' in titles, titles
            assert 'Quantum Mesh Overview' not in titles, titles
        finally:
            lib.close()

    async def test_expert_only_drops_the_repo_take(self, tmp_path, monkeypatch):
        # Pure-target question (no repo subject, spool product entity) ->
        # spool docs ARE the results; repo docs dropped (the old anchor
        # floor guaranteed repo docs here).
        lib = Library(tmp_path / 'lens-wiring.db')
        try:
            _seed_store(lib)
            svc = _service(lib, monkeypatch)
            resp = await svc._search_uncached(
                query='How does Quantum Mesh handle two jobs writing to the same ledger?',
                limit=4)
            titles = [d.title for d in resp.documents]
            assert 'Quantum Mesh Overview' in titles, titles
            assert not any(t.startswith(('Combinatorial', 'General'))
                           for t in titles), titles
        finally:
            lib.close()

    async def test_fuse_carries_both_paths(self, tmp_path, monkeypatch):
        # Seam question (names the repo subject + spool product entity) ->
        # repo docs AND the entity-admitted spool doc together.
        lib = Library(tmp_path / 'lens-wiring.db')
        try:
            _seed_store(lib)
            svc = _service(lib, monkeypatch)
            resp = await svc._search_uncached(
                query='What should I watch for running src1 on Quantum Mesh?',
                limit=4)
            titles = [d.title for d in resp.documents]
            assert any(t.startswith(('Combinatorial', 'General'))
                       for t in titles), titles
            assert 'Quantum Mesh Overview' in titles, titles
            # Connection labels travel with the response so the ask-side
            # assembly can attribute every environment doc.
            spool_doc = next(
                d for d in resp.documents if d.title == 'Quantum Mesh Overview')
            assert resp.spool_connections is not None
            assert resp.spool_connections[spool_doc.id] == 'entity(Quantum Mesh)'
        finally:
            lib.close()

    async def test_surface_scopes_the_gated_fallback(self, tmp_path, monkeypatch):
        # No crisp entity anywhere -> repo-only with gated fallback; when
        # the question resolves to a declared surface, the fallback pool is
        # RESTRICTED to surface-tagged docs — a CLOSER untagged spool doc
        # stays out (without the restriction the gate admits both).
        lib = Library(tmp_path / 'lens-wiring.db')
        try:
            repo_doc = lib.add_document(
                'explanation', 'Stage Boundary Notes', 'repo body',
                embedding=_vec(1, 0, 0), source_name='src1',
            )
            tagged = lib.add_document(
                'explanation', 'Wire Format Deep Dive', 'spool body',
                embedding=_vec(0.8, 0.6, 0),
                source_name='spool:fakebricks', _allow_reserved_source=True,
            )
            off = lib.add_document(
                'explanation', 'Off Surface Handbook', 'spool body',
                embedding=_vec(1, 0, 0),
                source_name='spool:fakebricks', _allow_reserved_source=True,
            )
            with lib._conn_provider.acquire() as conn:
                init_surface_tags_schema(conn)
                upsert_surface_tags(conn, [SurfaceTagRow(
                    'fakebricks', tagged.id, 'serialization')])
            svc = _service(lib, monkeypatch,
                           surfaces={'serialization': ['serializ', 'pickle']})
            # The gated fallback ranks against the embedding matrix; a fresh
            # store has none built, so hand the service a synthetic one over
            # the fixture docs (the fallback is a no-op without it).
            embeds = {repo_doc.id: _vec(1, 0, 0),
                      tagged.id: _vec(0.8, 0.6, 0),
                      off.id: _vec(1, 0, 0)}
            matrix = EmbeddingMatrix(
                matrix=np.stack(list(embeds.values())),
                ids=list(embeds), dim=3, build_stamp='synthetic')
            monkeypatch.setattr(
                svc, '_get_embedding_matrix', lambda: matrix)
            resp = await svc._search_uncached(
                query='How should src1 pickle intermediate results?', limit=4)
            titles = [d.title for d in resp.documents]
            assert 'Wire Format Deep Dive' in titles, titles
            assert 'Off Surface Handbook' not in titles, titles
        finally:
            lib.close()
