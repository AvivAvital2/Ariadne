import json
import sys
from pathlib import Path

import pytest
import yaml

from ariadne_mcp.service import AriadneService
from cli.main import main
from config import Config
from library import Library
from library.model_binding import inspect_binding
from model_config import QUERY_INSTRUCTION
from tests.local_model_server import deployments, local_config, trap_inference


def fixture_config(tmp_path, chat, embed):
    source = tmp_path / 'alpha'
    source.mkdir()
    source.joinpath('core.py').write_text(
        'def add(a: int, b: int) -> int:\n    """Add two integers."""\n    return a + b\n'
    )
    data = local_config(chat.base_url, embed.base_url)
    data.update(default_source='alpha', sources={'alpha': str(source)}, themes_enabled=False)
    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump(data))
    return Config(path)


@pytest.mark.parametrize('cloud_keys', [False, True])
def test_real_cli_generation_search_and_mcp_ask(tmp_path, monkeypatch, cloud_keys):
    import asyncio

    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
            if cloud_keys:
                monkeypatch.setenv(key, 'hosted-credentials-must-not-affect-routing')
            else:
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'ariadne',
                'generate',
                '--source',
                'alpha',
                '--path',
                'core.py',
                '--types',
                'explanation',
                '--no-batch',
                '--no-crossrefs',
                '--yes',
            ],
        )
        assert main() == 0
        binding = inspect_binding(Path(cfg.db_path))
        assert binding is not None
        with Library(Path(cfg.db_path)) as library:
            docs = library.list_documents('explanation')
            assert len(docs) == 1
            doc = docs[0]
            assert doc.source_name == 'alpha'
            assert doc.embedding.nbytes == 12288
            assert library.get_sections(doc.id)
            assert library.get_chunks(doc.id)
        for chunks in (False, True):
            argv = ['ariadne', 'search', 'How does arithmetic work?', '--source', 'alpha']
            if chunks:
                argv.append('--chunks')
            monkeypatch.setattr(sys, 'argv', argv)
            assert main() == 0

        async def mcp():
            service = AriadneService()
            service._config = cfg
            try:
                result = await service.search(
                    query='How does arithmetic work?', sections_only=True, source='alpha'
                )
                assert result.documents
                assert result.documents[0].id == doc.id
                answer = await service.ask('How does arithmetic work?', source='alpha')
                assert 'Core arithmetic' in answer.answer
            finally:
                await service.close()

        asyncio.run(mcp())
        inputs = [
            text
            for route, payload, _ in embed.requests
            if route.endswith('/embeddings')
            for text in payload['input']
        ]
        assert any(text.startswith(f'Instruct: {QUERY_INSTRUCTION}\nQuery:') for text in inputs)
        assert all(
            'Authorization' not in headers for _, _, headers in chat.requests + embed.requests
        )


def test_local_preview_does_not_create_state_or_call_models(tmp_path, monkeypatch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        for argv in (
            ['ariadne', 'dry-run', '--source', 'alpha'],
            ['ariadne', 'generate', '--source', 'alpha', '--dry-run'],
        ):
            monkeypatch.setattr(sys, 'argv', argv)
            assert main() == 0
        assert not Path(cfg.db_path).exists()
        assert not Path(cfg.staleness_db_path).exists()
        assert chat.requests == [] and embed.requests == []


def test_colliding_sources_language_filtering_and_repeat(tmp_path, monkeypatch):
    import config
    from docgen.staleness import StalenessTracker

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        beta = tmp_path / 'beta'
        beta.mkdir()
        beta.joinpath('core.py').write_text('def subtract(a, b):\n    return a - b\n')
        beta.joinpath('settings.json').write_text('{"timeout": 30}\n')
        data = yaml.safe_load(cfg.config_path.read_text())
        data['sources']['beta'] = str(beta)
        cfg.config_path.write_text(yaml.safe_dump(data))
        cfg = Config(cfg.config_path)
        monkeypatch.setattr(config, '_global_config', cfg)
        for source, path in [('alpha', 'core.py'), ('beta', 'core.py'), ('beta', 'settings.json')]:
            argv = [
                'ariadne',
                'generate',
                '--source',
                source,
                '--path',
                path,
                '--no-batch',
                '--no-crossrefs',
                '--yes',
            ]
            if path.endswith('.py'):
                argv += ['--types', 'explanation']
            monkeypatch.setattr(sys, 'argv', argv)
            assert main() == 0
            requests = len(chat.requests)
            assert main() == 0
            assert len(chat.requests) == requests
        with Library(Path(cfg.db_path)) as library:
            docs = library.list_documents('explanation')
            assert len(docs) == 3
            assert {d.source_name for d in docs} == {'alpha', 'beta'}
            assert len({d.id for d in docs}) == 3
        with StalenessTracker(Path(cfg.staleness_db_path), source_name='alpha') as tracker:
            assert len(list(tracker.get_all_records())) == 1
        with StalenessTracker(Path(cfg.staleness_db_path), source_name='beta') as tracker:
            assert len(list(tracker.get_all_records())) == 2


@pytest.mark.asyncio
async def test_warm_service_cache_observes_another_writer(local_deployment, tmp_path):
    from inference import InferenceContext
    from writer import LibraryWriter

    ctx, chat, embed = local_deployment
    cfg = fixture_config(tmp_path, chat, embed)
    # This fixture's library is already bound; point the service at that pair.
    data = yaml.safe_load(cfg.config_path.read_text())
    data['defaults'] = {
        'db_path': str(ctx.db_path),
        'staleness_db_path': str(ctx.staleness_db_path),
    }
    cfg.config_path.write_text(yaml.safe_dump(data))
    service = AriadneService()
    service._config = Config(cfg.config_path)
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        await LibraryWriter(library).add_document(
            'finding', 'First', 'Original content', source_name='alpha', doc_id='first'
        )
    initial = await service.search(query='arithmetic', source='alpha')
    assert len(initial.documents) == 1
    assert await service.search(query='arithmetic', source='alpha') is initial
    other = InferenceContext(ctx.settings, ctx.db_path, ctx.staleness_db_path)
    with other.write_session():
        library = other.open_library(operation='write')
        await LibraryWriter(library).add_document(
            'finding', 'Second', 'Additional content', source_name='alpha', doc_id='second'
        )
    revised = await service.search(query='arithmetic', source='alpha')
    assert len(revised.documents) == 2
    assert revised is not initial
    await other.aclose()
    await service.close()
    await ctx.aclose()


def test_local_cli_batch_lifecycle(tmp_path, monkeypatch, capsys):
    import sqlite3

    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'ariadne',
                'generate',
                '--source',
                'alpha',
                '--types',
                'explanation',
                '--batch',
                '--no-crossrefs',
                '--yes',
            ],
        )
        assert main() == 0
        with sqlite3.connect(cfg.db_path) as conn:
            run_id, state, phases = conn.execute(
                'SELECT run_id,state,checkpoints_json FROM spool_runs'
            ).fetchone()
            assert state == 'completed'
            assert set(json.loads(phases)) == {
                'base',
                'reverse',
                'graph',
                'themes',
                'crossrefs',
                'reconcile',
            }
            assert conn.execute(
                'SELECT state,request_attempts,apply_attempts FROM spool_jobs'
            ).fetchone() == ('applied', 1, 1)
            timestamp = conn.execute('SELECT updated_at FROM documents').fetchone()
        chat_requests = len(chat.requests)
        embedding_requests = len(embed.requests)
        for action in ('list', 'status', 'resume'):
            monkeypatch.setattr(
                sys, 'argv', ['ariadne', 'batch', action] + ([] if action == 'list' else [run_id])
            )
            assert main() == 0
        assert len(chat.requests) == chat_requests
        assert len(embed.requests) == embedding_requests
        with sqlite3.connect(cfg.db_path) as conn:
            assert conn.execute('SELECT updated_at FROM documents').fetchone() == timestamp
        monkeypatch.setattr(sys, 'argv', ['ariadne', 'batch', 'clear', run_id])
        assert main() == 0
        with sqlite3.connect(cfg.db_path) as conn:
            assert conn.execute('SELECT count(*) FROM documents').fetchone()[0] == 1
            assert conn.execute('SELECT count(*) FROM spool_runs').fetchone()[0] == 0
        assert 'responses saved' in capsys.readouterr().out
