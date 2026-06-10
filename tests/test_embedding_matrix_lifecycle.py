"""Tier 3 (OPERATE) — matrix lifecycle: ensure_matrix (absent/fresh/stale), the
rebuild trigger, the mmap-load guarantee, and cache invalidation.

See designs/embedding-matrix-tier3-operate.md. Built via the evolving-TDD loop,
then split into focused tests for failure localization. (The multi-process RSS
sharing check is an out-of-band ops step, documented in the design, not a unit
assertion.)

Fixtures are synthetic only: source ``src1``, docs ``d1, d2, …``, tiny
``dim = 4`` hand-chosen vectors.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from library import Library
from library.embedding_matrix import (
    ARTIFACT_NAME,
    EmbeddingMatrix,
    build_doc_embedding_matrix,
    ensure_matrix,
    matrix_dir_for,
)


@pytest.fixture
def lib(tmp_path: Path) -> Library:
    library = Library(tmp_path / 'lib.db')
    yield library
    library.close()


def _add(library: Library, doc_id: str, vec: list[float]) -> None:
    library.add_document(
        content_type='explanation',
        title=f'title-{doc_id}',
        content=f'content-{doc_id}',
        embedding=np.array(vec, dtype=np.float32),
        doc_id=doc_id,
        source_name='src1',
    )


def _bump_updated_at(library: Library, doc_id: str) -> None:
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE documents SET updated_at = ? WHERE id = ?',
            ('2099-01-01T00:00:00', doc_id),
        )


class _FakeWriter:
    """Stands in for writer.LibraryWriter so _rebuild_embeddings runs without
    calling the embedding API."""

    def __init__(self, library: Library) -> None:
        self._library = library

    async def __aenter__(self) -> '_FakeWriter':
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def rebuild_all_embeddings(self) -> int:
        return 1


def _spy_on_build(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record calls to build_doc_embedding_matrix (still running the real one)."""
    import library.embedding_matrix as em

    builds: list = []
    real = em.build_doc_embedding_matrix
    monkeypatch.setattr(
        em, 'build_doc_embedding_matrix',
        lambda library, out: builds.append(out) or real(library, out),
    )
    return builds


# --- ensure_matrix ---------------------------------------------------------

def test_ensure_matrix_builds_when_absent(lib: Library, tmp_path: Path) -> None:
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    matrix = ensure_matrix(lib, tmp_path)
    assert matrix is not None
    assert (tmp_path / ARTIFACT_NAME).exists()
    with lib._conn_provider.acquire() as conn:
        assert matrix.is_fresh(conn)


def test_ensure_matrix_skips_when_fresh(lib: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    ensure_matrix(lib, tmp_path)  # initial build
    builds = _spy_on_build(monkeypatch)
    ensure_matrix(lib, tmp_path)
    assert builds == []


def test_ensure_matrix_rebuilds_when_stale(lib: Library, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    ensure_matrix(lib, tmp_path)  # initial build
    builds = _spy_on_build(monkeypatch)
    _bump_updated_at(lib, 'd1')
    ensure_matrix(lib, tmp_path)
    assert len(builds) == 1


def test_load_returns_memmap(lib: Library, tmp_path: Path) -> None:
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    build_doc_embedding_matrix(lib, tmp_path)
    loaded = EmbeddingMatrix.load(tmp_path)
    assert isinstance(loaded.M, np.memmap)


# --- triggers / invalidation ----------------------------------------------

def test_rebuild_triggers_matrix_refresh(lib: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    from cli import core as cli_core

    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr('writer.LibraryWriter', _FakeWriter)
    asyncio.run(cli_core._rebuild_embeddings(lib))
    built = EmbeddingMatrix.load(matrix_dir_for(lib))
    assert built is not None
    with lib._conn_provider.acquire() as conn:
        assert built.is_fresh(conn)


def test_clear_cache_drops_matrix_handle(lib: Library) -> None:
    from ariadne_mcp.service import AriadneService

    svc = AriadneService()
    svc._library = lib
    svc._embedding_matrix_cache = object()  # pretend a loaded handle
    svc.clear_cache()
    assert not hasattr(svc, '_embedding_matrix_cache')
    svc.clear_cache()  # idempotent when no handle is cached
    assert not hasattr(svc, '_embedding_matrix_cache')


def test_build_on_startup_builds_matrix(lib: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    from ariadne_mcp.service import AriadneService
    from cli.integration import _build_embedding_matrix_on_startup

    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    svc = AriadneService()
    svc._library = lib
    monkeypatch.setattr(AriadneService, '_instance', svc)

    _build_embedding_matrix_on_startup()

    built = EmbeddingMatrix.load(matrix_dir_for(lib))
    assert built is not None
    with lib._conn_provider.acquire() as conn:
        assert built.is_fresh(conn)


def test_build_on_startup_degrades_on_error(lib: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    from ariadne_mcp.service import AriadneService
    from cli.integration import _build_embedding_matrix_on_startup

    svc = AriadneService()
    svc._library = lib
    monkeypatch.setattr(AriadneService, '_instance', svc)

    def _boom(*a, **k):
        raise RuntimeError('build failed')

    monkeypatch.setattr('library.embedding_matrix.ensure_matrix', _boom)
    # Must not raise — serving continues on the SQLite fallback.
    _build_embedding_matrix_on_startup()


def test_cmd_mcp_triggers_startup_build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import argparse

    import ariadne_mcp.server as server
    from cli import integration as integ

    calls = {'startup': 0, 'run': 0}
    monkeypatch.setattr(integ, '_build_embedding_matrix_on_startup',
                        lambda: calls.__setitem__('startup', calls['startup'] + 1))
    monkeypatch.setattr(server.mcp, 'run', lambda **k: calls.__setitem__('run', calls['run'] + 1))
    monkeypatch.chdir(tmp_path)

    rc = integ.cmd_mcp(argparse.Namespace(directory=str(tmp_path)))

    assert rc == 0
    assert calls == {'startup': 1, 'run': 1}  # built before serving
