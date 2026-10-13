import shutil
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pytest

from docgen.llm.local import EmbeddingContractError
from inference import InferenceContext
from library.model_binding import ModelBindingMismatch
from model_config import resolve_models
from tests.local_model_server import DOCUMENT, deployments, trap_inference
from tests.test_local_maintenance import invoke
from tests.test_local_workflows import fixture_config
from writer import LibraryWriter


def test_doctor_probes_both_endpoints_without_binding(tmp_path, monkeypatch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        assert invoke(monkeypatch, 'doctor', '--models') == 0
        assert not Path(cfg.db_path).exists()
        assert not Path(cfg.staleness_db_path).exists()
        for endpoint in (chat, embed):
            assert len(endpoint.requests) == 2
            assert endpoint.requests[0][0].endswith('/models')
            assert endpoint.requests[1][0].endswith('/tokenize')


@pytest.mark.parametrize('table', ['documents', 'chunks', 'sections'])
@pytest.mark.asyncio
async def test_corrupt_stored_vectors_fail_before_ranking(local_deployment, table):
    ctx, _, _ = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        doc = await LibraryWriter(library).add_document('explanation', 'Arithmetic', DOCUMENT)
        query = await ctx.embeddings.embed_query('Arithmetic')
        with library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE ' + table + ' SET embedding=?',
                (np.zeros(3072, dtype=np.float32).tobytes(),),
            )
        with pytest.raises(EmbeddingContractError, match='unit norm'):
            if table == 'documents':
                library.search(query)
            elif table == 'chunks':
                library.search_chunks(query)
            else:
                library.get_section_embeddings_for_doc(doc.id)
    await ctx.aclose()


def test_new_hosted_library_cannot_claim_populated_legacy_sidecar(tmp_path):
    sidecar = tmp_path / 'side.db'
    with sqlite3.connect(sidecar) as conn:
        conn.execute('CREATE TABLE pending_batches(id TEXT)')
        conn.execute("INSERT INTO pending_batches VALUES ('old-cloud-run')")
    before = sidecar.read_bytes()
    main = tmp_path / 'new.db'
    ctx = InferenceContext(resolve_models({}), main, sidecar)
    with pytest.raises(ModelBindingMismatch, match='Nonempty unbound'):
        with ctx.write_session():
            pass
    assert not main.exists()
    assert sidecar.read_bytes() == before


@pytest.mark.parametrize('fail', [False, True])
def test_incremental_sync_only_advances_after_required_work(tmp_path, monkeypatch, fail):
    import config

    git_path = shutil.which('git')
    assert git_path

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        root = cfg.get_source_path('alpha')
        root.joinpath('core.py').rename(root / 'settings.json')
        root.joinpath('settings.json').write_text('{"timeout": 30}\n')
        monkeypatch.setattr(config, '_global_config', cfg)

        def git(*args):
            return subprocess.run(  # noqa: S603 - fixed git operations on an isolated test repository.
                [git_path, '-C', str(root), *args], check=True, capture_output=True, text=True
            ).stdout.strip()

        git('init')
        git('add', 'settings.json')
        git(
            '-c',
            'user.name=Ariadne Test',
            '-c',
            'user.email=test@example.invalid',
            'commit',
            '-m',
            'Initial',
        )
        initial = git('rev-parse', 'HEAD')
        assert invoke(monkeypatch, 'sync', '--force', '--skip-generate', '--no-export') == 0
        root.joinpath('settings.json').write_text('{"timeout": 60}\n')
        git('add', 'settings.json')
        git(
            '-c',
            'user.name=Ariadne Test',
            '-c',
            'user.email=test@example.invalid',
            'commit',
            '-m',
            'Update',
        )
        updated = git('rev-parse', 'HEAD')
        if fail:
            chat.failures = [400]
        assert invoke(monkeypatch, 'sync', '--no-export') == (1 if fail else 0)
        with sqlite3.connect(cfg.db_path) as conn:
            stored = conn.execute(
                'SELECT git_hash FROM sync_state WHERE source_name=?', ('alpha',)
            ).fetchone()[0]
            assert stored == (initial if fail else updated)
            if not fail:
                assert conn.execute('SELECT crossrefs_fingerprint FROM library_state').fetchone()[0]


def test_exhausted_phase_budget_is_visible_as_failed(local_deployment):
    from docgen.llm.local import LocalInferenceError
    from docgen.spool import SpoolStore

    ctx, _, _ = local_deployment
    with ctx.write_session():
        store = SpoolStore(ctx, ctx.open_library(operation='write'))
        run_id = store.create([], operation='onboard', scope={}, phase='structural')
        for _ in range(ctx.settings.execution.max_attempts):
            assert store.begin_phase(run_id, 'structural', 'fixed-input')
        with pytest.raises(LocalInferenceError, match='budget exhausted'):
            store.begin_phase(run_id, 'structural', 'fixed-input')
        assert store.get_run(run_id)['state'] == 'failed'
        assert store.finish(run_id)['state'] == 'failed'


@pytest.mark.parametrize('fail', [False, True])
def test_merge_regenerates_locally_before_consuming_branch_documents(tmp_path, monkeypatch, fail):
    import asyncio

    import config
    import docgen.merge
    from library import Library

    git_path = shutil.which('git')
    assert git_path
    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        root = cfg.get_source_path('alpha')

        def git(*args):
            return subprocess.run(  # noqa: S603 - isolated fixture repository.
                [git_path, '-C', str(root), *args], check=True, capture_output=True, text=True
            ).stdout.strip()

        git('init', '-b', 'main')
        git('add', 'core.py')
        git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'Add')
        git('branch', 'feature/math')
        # GitHub merge detection is external to inference; generation and storage stay real.
        monkeypatch.setattr(docgen.merge, 'find_merged_branches', lambda *_: ['feature/math'])
        context = InferenceContext(
            cfg.model_settings(execution=True), Path(cfg.db_path), Path(cfg.staleness_db_path)
        )

        async def seed():
            with context.write_session():
                library = context.open_library(operation='write')
                await LibraryWriter(library).add_document(
                    'explanation',
                    'Branch arithmetic',
                    DOCUMENT,
                    ['core.py'],
                    metadata={
                        'generated': True,
                        'status': 'experimental',
                        'branches': ['feature/math'],
                        'generation_scope': 'branch:feature/math',
                    },
                    source_name='alpha',
                    doc_id='branch-document',
                )
            await context.aclose()

        asyncio.run(seed())
        if fail:
            chat.failures = [400, 400]
        assert invoke(monkeypatch, 'merge', '--source', 'alpha', '--no-export') == (
            1 if fail else 0
        )
        with Library(Path(cfg.db_path)) as library:
            assert library.get_document('branch-document').metadata['status'] == (
                'experimental' if fail else 'deprecated'
            )
            stable = [
                doc
                for doc in library.list_documents()
                if doc.metadata.get('generation_scope') == 'stable'
            ]
            if not fail:
                assert {doc.content_type for doc in stable} >= {'explanation', 'architecture'}
                assert all(
                    doc.embedding is not None and library.get_sections(doc.id) for doc in stable
                )
        with sqlite3.connect(cfg.db_path) as conn:
            sync = conn.execute(
                'SELECT git_hash FROM sync_state WHERE source_name=?', ('alpha',)
            ).fetchone()
            assert sync is None if fail else sync[0] == git('rev-parse', 'HEAD')
        assert any(route.endswith('/chat/completions') for route, _, _ in chat.requests) or fail
