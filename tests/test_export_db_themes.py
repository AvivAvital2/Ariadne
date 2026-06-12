"""Batch 2: themes + theme_members + doc_graph parity for the DB slice.

A single-source slice must carry the cross-cutting structure that natural-language
Q&A leans on, scoped correctly:

- a theme travels iff it has >=1 member in the source; its members are restricted
  to in-source elements (matching ``ScopedLibrary.get_theme_members``);
- doc_graph edges travel iff both endpoints are in-source docs.

Parity baseline is the full library viewed through ``ScopedLibrary({source})`` —
what the MCP/Slack path serves today.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from export_db import export_source_db
from library import Library, ScopedLibrary
from schema import Section

EMBED_DIM = 16


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture(autouse=True)
def _no_token_count(monkeypatch):
    monkeypatch.setattr('library.core.CoreMixin._count_tokens', lambda self, content: None)


@pytest.fixture
def rich_db(tmp_path):
    """Two sources + a theme that spans them + a noise-only theme + doc_graph edges."""
    path = tmp_path / 'full.db'
    lib = Library(path)

    def add(sn, ct, title, seed):
        return lib.add_document(
            content_type=ct, title=title, content=f'{title} body',
            source_files=[f'{sn}/x.py'], embedding=_vec(seed), source_name=sn,
        ).id

    exp_sdk = add('sdk', 'explanation', 'SDK Overview', 1)
    cat_sdk1 = add('sdk', 'catalog', 'sdk.auth', 2)
    cat_sdk2 = add('sdk', 'catalog', 'sdk.client', 3)
    cat_noise = add('noise', 'catalog', 'noise.thing', 4)

    # theme summary docs are written source_name=NULL (cross-source by design)
    theme_touch = lib.add_document(content_type='theme', title='Authentication Theme',
                                   content='Spans auth code', source_files=[],
                                   embedding=_vec(10), source_name=None).id
    theme_noise = lib.add_document(content_type='theme', title='Noise Theme',
                                   content='Only noise', source_files=[],
                                   embedding=_vec(11), source_name=None).id

    lib.add_theme(cluster_id='c_touch', doc_id=theme_touch, member_count=3,
                  resolution=1.0, summary_hash='h1', coherent=True, dirty=False)
    lib.set_theme_members('c_touch', [(cat_sdk1, 1.0), (cat_sdk2, 1.0), (cat_noise, 1.0)])

    lib.add_theme(cluster_id='c_noise', doc_id=theme_noise, member_count=1,
                  resolution=1.0, summary_hash='h2', coherent=True, dirty=False)
    lib.set_theme_members('c_noise', [(cat_noise, 1.0)])
    lib.close()

    # doc_graph edges: one fully in-source, one reaching out to noise
    conn = sqlite3.connect(path)
    conn.executemany(
        'INSERT OR REPLACE INTO doc_graph (source_id, target_id, edge_type, weight) VALUES (?,?,?,?)',
        [(exp_sdk, cat_sdk1, 'related', 1.0), (exp_sdk, cat_noise, 'related', 1.0)],
    )
    conn.commit()
    conn.close()

    return {
        'path': str(path), 'exp_sdk': exp_sdk, 'cat_sdk1': cat_sdk1,
        'cat_sdk2': cat_sdk2, 'cat_noise': cat_noise,
        'theme_touch': theme_touch, 'theme_noise': theme_noise,
    }


def _export(rich_db, tmp_path):
    out = tmp_path / 'slice.db'
    export_source_db(rich_db['path'], 'sdk', out)
    return out


def test_theme_touching_source_is_carried_with_scoped_members(rich_db, tmp_path):
    out = _export(rich_db, tmp_path)
    full = Library(Path(rich_db['path']))
    sl = Library(out)
    try:
        clusters = {t.cluster_id for t in sl.list_themes(coherent_only=True)}
        assert 'c_touch' in clusters
        assert sl.get_document(rich_db['theme_touch']) is not None, 'theme summary doc missing'

        sliced = sorted(eid for eid, _ in sl.get_theme_members('c_touch'))
        scoped = sorted(
            eid for eid, _ in ScopedLibrary(full, frozenset({'sdk'})).get_theme_members('c_touch')
        )
    finally:
        full.close(); sl.close()
    assert sliced == sorted([rich_db['cat_sdk1'], rich_db['cat_sdk2']]), 'members not restricted to source'
    assert sliced == scoped, 'theme membership diverged from scoped-full'


def test_theme_without_source_members_is_omitted(rich_db, tmp_path):
    out = _export(rich_db, tmp_path)
    sl = Library(out)
    try:
        clusters = {t.cluster_id for t in sl.list_themes(coherent_only=True)}
    finally:
        sl.close()
    assert 'c_touch' in clusters          # positive: the relevant theme is here
    assert 'c_noise' not in clusters      # negative: a noise-only theme is not


def test_no_dangling_theme_members(rich_db, tmp_path):
    out = _export(rich_db, tmp_path)
    conn = sqlite3.connect(out)
    try:
        members = [r[0] for r in conn.execute('SELECT element_id FROM theme_members')]
        doc_ids = {r[0] for r in conn.execute('SELECT id FROM documents')}
    finally:
        conn.close()
    assert members, 'expected at least one theme member in the slice'
    assert all(m in doc_ids for m in members), 'theme_members reference docs absent from the slice'


def test_doc_graph_carries_internal_edges_and_drops_external(rich_db, tmp_path):
    out = _export(rich_db, tmp_path)
    conn = sqlite3.connect(out)
    try:
        edges = conn.execute('SELECT source_id, target_id FROM doc_graph').fetchall()
        doc_ids = {r[0] for r in conn.execute('SELECT id FROM documents')}
    finally:
        conn.close()
    assert (rich_db['exp_sdk'], rich_db['cat_sdk1']) in edges, 'in-source edge missing'
    assert rich_db['cat_noise'] not in {n for e in edges for n in e}, 'external edge leaked'
    assert all(s in doc_ids and t in doc_ids for s, t in edges), 'dangling doc_graph endpoint'


def test_get_related_parity_one_hop(rich_db, tmp_path):
    out = _export(rich_db, tmp_path)
    full = Library(Path(rich_db['path']))
    sl = Library(out)
    try:
        sliced = sorted(r['id'] for r in sl.get_related(rich_db['exp_sdk']))
        scoped = sorted(
            r['id'] for r in ScopedLibrary(full, frozenset({'sdk'})).get_related(rich_db['exp_sdk'])
        )
    finally:
        full.close(); sl.close()
    assert sliced == [rich_db['cat_sdk1']]
    assert sliced == scoped, 'get_related diverged from scoped-full'
