"""The stale-matrix hole (found live): `spools install` added 1,902 embedded
theme docs, nothing rebuilt the matrix, and the serve path silently degraded
— `_get_embedding_matrix` freshness-checks, returns None, and NEVER retries
the disk load, so the lens's semantic fill stayed dark even after an offline
rebuild. Store-mutating spool commands now refresh the matrix, and the
accessor recovers a fresh on-disk matrix without a server restart.
"""
from __future__ import annotations

from types import SimpleNamespace

from ariadne_mcp.service import AriadneService
from cli.spools_cmd import _install
from library import Library
from library.embedding_matrix import EmbeddingMatrix


class TestServiceMatrixRecovery:
    def test_stale_cached_matrix_reloads_from_disk(self, tmp_path, monkeypatch):
        # In-process cache holds a STALE matrix; a fresh one has since been
        # rebuilt on disk — the accessor must swap it in (no restart), and
        # only fall back to None when the disk copy is stale too.
        lib = Library(tmp_path / 'recover.db')
        try:
            svc = AriadneService()
            svc._library = lib
            stale = SimpleNamespace(is_fresh=lambda conn: False)
            fresh = SimpleNamespace(is_fresh=lambda conn: True)
            svc._embedding_matrix_cache = stale
            monkeypatch.setattr(
                EmbeddingMatrix, 'load', classmethod(lambda cls, d: fresh))
            assert svc._get_embedding_matrix() is fresh
            assert svc._embedding_matrix_cache is fresh   # re-cached

            svc._embedding_matrix_cache = stale
            monkeypatch.setattr(
                EmbeddingMatrix, 'load', classmethod(lambda cls, d: stale))
            assert svc._get_embedding_matrix() is None
        finally:
            lib.close()


class TestSpoolCommandsRefreshMatrix:
    def test_install_refreshes_the_matrix(self, tmp_path, monkeypatch):
        # `spools install` mutates the store (docs + embeddings) — the matrix
        # must be ensured afterwards so the serve path never silently loses
        # semantic ranking.
        cfg = SimpleNamespace(db_path=str(tmp_path / 'c.db'))
        monkeypatch.setattr('cli.spools_cmd.get_config', lambda: cfg)
        manifest = SimpleNamespace(environment='envx', target_runtime='r-1',
                                   version='1.0.0')
        monkeypatch.setattr('spool_pack.install_pack',
                            lambda library, pack, cache_dir: manifest)
        ensured = []
        monkeypatch.setattr('library.embedding_matrix.ensure_matrix',
                            lambda library, out_dir=None: ensured.append(library))
        args = SimpleNamespace(pack=str(tmp_path / 'p.zip'),
                               cache_dir=str(tmp_path / 'cache'))
        assert _install(args) == 0
        assert len(ensured) == 1
