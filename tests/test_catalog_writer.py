"""Tests for docgen.catalog_writer."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from docgen.catalog_writer import (
    iter_catalog_files,
    sync_file_catalog,
    sync_source_catalog,
)
from library import Library
from writer import LibraryWriter


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)                                                                                                                                                            
    return p
                                                                                                                                                                                     
                
@pytest.fixture(autouse=True)
def _test_config(tmp_path: Path, monkeypatch):
    """Provide a Config that knows about the 'test' source name."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'test')


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'catalog-test.db')
    yield lib
    lib.close()                                                                                                                                                                      
                

@pytest.fixture
def mocked_embedding(monkeypatch):
    """Patch EmbeddingService.embed and embed_batch so tests make no API calls."""                                                                                                   
    async def fake_embed(self, text):                                                                                                                                                
        return np.zeros(1536, dtype=np.float32)                                                                                                                                      
                                                                                                                                                                                     
    async def fake_embed_batch(self, texts):
        return [np.zeros(1536, dtype=np.float32) for _ in texts]                                                                                                                     
                
    async def fake_get_client(self):
        return None                                                                                                                                                                  
 
    async def fake_close(self):                                                                                                                                                      
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr('embedding.EmbeddingService.embed_batch', fake_embed_batch)                                                                                                  
    monkeypatch.setattr('embedding.EmbeddingService._get_client', fake_get_client)
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)                                                                                                              
                                                                                                                                                                                     
                                                                                                                                                                                     
def _run_sync_file(library, source_root, file):                                                                                                                                      
    async def go():                                                                                                                                                                  
        async with LibraryWriter(library) as writer:
            return await sync_file_catalog(library, writer, 'test', source_root, file)                                                                                               
    return asyncio.run(go())                                                                                                                                                         
 
                                                                                                                                                                                     
def _run_sync_source(library, source_root):
    async def go():                                                                                                                                                                  
        async with LibraryWriter(library) as writer:
            return await sync_source_catalog(library, writer, 'test', source_root)                                                                                                   
    return asyncio.run(go())


def test_iter_catalog_files_skips_cost_noise(tmp_path: Path):
    # Minified/vendored bundles, lockfiles, JSON under a test/fixtures
    # directory, and a framework's generated CSS output (Tailwind) pass the
    # extension filter but are never worth the embed + LLM-describe cost.
    # Generated *source* (*_pb2.py), non-test JSON, and a Tailwind *input*
    # stylesheet are kept.
    from docgen.catalog_writer import iter_catalog_files

    noise = (
        'app.min.js', 'styles.min.css', 'vendor/jquery.min.js',
        'package-lock.json', 'web/something-lock.json',
        'cypress/fixtures/data.json', 'src/test/resources/fixture.json',
        'webclient/tests/data/big.json',
        'store/modules/hypothesisSpace/fixtures/features.json',
        'assets/tempoutput/tailwind-output.css',
    )
    kept = (
        'app.js', 'config.json', 'pkg/foo.py', 'proto/foo_pb2.py',
        'src/styles/tailwind.css',
    )
    for rel in (*noise, *kept):
        _write(tmp_path, rel, '{}' if rel.endswith('.json') else 'x = 1\n')

    # exclude_dir_names=() so dir-pruning doesn't mask the file-level logic.
    found = {
        p.relative_to(tmp_path).as_posix()
        for p in iter_catalog_files(tmp_path, (), ())
    }

    for rel in noise:
        assert rel not in found, f'{rel} should be excluded as cost-noise'
    for rel in kept:
        assert rel in found, f'{rel} should be kept (real/generated source or non-test JSON)'


def test_default_policy_excludes_ci_and_runtime_dirs_at_any_depth(tmp_path: Path):
    # CI / VCS-platform + runtime dirs are pruned wherever they appear, not
    # only at the root. Exact-name match: '.git' != '.github', and a file
    # named github_helper.py is not the '.github' directory.
    from docgen.catalog_writer import iter_catalog_files

    excluded = (
        '.github/workflows/ci.yml', 'pkg/.gitlab/x.yml',
        '.circleci/config.yml', 'web/.husky/pre.js',
        '.changeset/c.md', 'a/b/tmp/scratch.py', 'logs/app.md',
    )
    kept = ('app.py', 'src/github_helper.py')
    for rel in (*excluded, *kept):
        _write(tmp_path, rel, 'x = 1\n')

    found = {
        p.relative_to(tmp_path).as_posix()
        for p in iter_catalog_files(tmp_path)
    }
    for rel in excluded:
        assert rel not in found, f'{rel} should be pruned at any depth'
    for rel in kept:
        assert rel in found, f'{rel} should be kept'


class TestIterCatalogFiles:
    def test_finds_multiple_languages(self, tmp_path: Path) -> None:                                                                                                                 
        _write(tmp_path, 'a.py', 'x = 1')
        _write(tmp_path, 'b.html', '<section>Hi</section>')                                                                                                                          
        _write(tmp_path, 'c.js', 'const x = 1;')
        files = iter_catalog_files(tmp_path)                                                                                                                                         
        assert len(files) == 3                                                                                                                                                       
 
    def test_skips_dep_dirs(self, tmp_path: Path) -> None:                                                                                                                           
        _write(tmp_path, 'app.py', 'x = 1')
        _write(tmp_path, 'node_modules/pkg/lib.js', 'y = 2')                                                                                                                         
        _write(tmp_path, '.venv/lib.py', 'z = 3')
        _write(tmp_path, '__pycache__/cached.py', 'w = 4')                                                                                                                           
        files = iter_catalog_files(tmp_path)
        assert len(files) == 1                                                                                                                                                       
                
    def test_ignores_unsupported_extensions(self, tmp_path: Path) -> None:
        _write(tmp_path, 'a.py', 'x = 1')
        _write(tmp_path, 'notes.txt', 'plain text')
        _write(tmp_path, 'data.csv', 'a,b,c')
        _write(tmp_path, 'trace.log', 'INFO: hello')
        files = iter_catalog_files(tmp_path)
        # Only a.py is in CATALOG_EXTS; .txt / .csv / .log are not.
        assert len(files) == 1
 
                                                                                                                                                                                     
class TestSyncFileCatalog:
    def test_bootstrap_creates_elements_and_index(
        self, tmp_path: Path, library, mocked_embedding,                                                                                                                             
    ) -> None:
        f = _write(tmp_path, 'mod.py', 'def foo(): pass\nclass Bar: pass\n')                                                                                                       
        summary = _run_sync_file(library, tmp_path, f)                                                                                                                               
        assert summary.added == 2
        assert summary.modified == 0                                                                                                                                                 
        assert summary.removed == 0                                                                                                                                                  
        # 2 element docs + 1 file_index doc = 3 catalog docs
        assert library.count_documents(content_type='catalog') == 3                                                                                                                  
                
    def test_idempotent_on_unchanged_file(                                                                                                                                           
        self, tmp_path: Path, library, mocked_embedding,                                                                                                                             
    ) -> None:
        f = _write(tmp_path, 'mod.py', 'def foo(): pass\n')                                                                                                                         
        _run_sync_file(library, tmp_path, f)                                                                                                                                         
        second = _run_sync_file(library, tmp_path, f)
        assert second.added == 0                                                                                                                                                     
        assert second.modified == 0                                                                                                                                                  
        assert second.removed == 0
        assert second.unchanged == 1                                                                                                                                                 
                
    def test_detects_added_element(
        self, tmp_path: Path, library, mocked_embedding,                                                                                                                             
    ) -> None:
        f = _write(tmp_path, 'mod.py', 'def foo(): pass\n')                                                                                                                         
        _run_sync_file(library, tmp_path, f)                                                                                                                                         
        f.write_text('def foo(): pass\ndef bar(): pass\n')
        second = _run_sync_file(library, tmp_path, f)                                                                                                                                
        assert second.added == 1
                                                                                                                                                                                     
    def test_detects_removed_element(
        self, tmp_path: Path, library, mocked_embedding,
    ) -> None:                                                                                                                                                                       
        f = _write(tmp_path, 'mod.py', 'def foo(): pass\ndef bar(): pass\n')
        _run_sync_file(library, tmp_path, f)                                                                                                                                         
        f.write_text('def foo(): pass\n')                                                                                                                                           
        second = _run_sync_file(library, tmp_path, f)
        assert second.removed == 1                                                                                                                                                   
                                                                                                                                                                                     
    def test_detects_modified_element(
        self, tmp_path: Path, library, mocked_embedding,                                                                                                                             
    ) -> None:  
        f = _write(tmp_path, 'mod.py', 'def foo(): return 1\n')
        _run_sync_file(library, tmp_path, f)                                                                                                                                         
        f.write_text('def foo(): return 2\n')
        second = _run_sync_file(library, tmp_path, f)                                                                                                                                
        assert second.modified == 1                                                                                                                                                  
 
                                                                                                                                                                                     
class TestSyncSourceCatalog:
    def test_processes_all_files(
        self, tmp_path: Path, library, mocked_embedding,                                                                                                                             
    ) -> None:
        _write(tmp_path, 'a.py', 'def f(): pass\n')                                                                                                                                 
        _write(tmp_path, 'b.py', 'def g(): pass\n')                                                                                                                                 
        summaries = _run_sync_source(library, tmp_path)
        assert len(summaries) == 2                                                                                                                                                   
        assert sum(s.added for s in summaries) == 2
