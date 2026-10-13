import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

from ariadne_mcp.service import AriadneService
from cli.main import main
from tests.local_model_server import DOCUMENT, deployments, trap_inference
from tests.test_local_workflows import fixture_config


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['ariadne', *args])
    return main()


def test_topic_and_mcp_authored_contribution_share_local_pair(tmp_path, monkeypatch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        monkeypatch.setenv('OPENAI_API_KEY', 'must-not-route-topic-to-cloud')
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        assert (
            invoke(monkeypatch, 'topic', 'Arithmetic', '--source', 'alpha', '--files', 'core.py')
            == 0
        )

        async def contribute():
            service = AriadneService()
            service._config = cfg
            try:
                response = await service.contribute(
                    'Authored finding', 'Preserve this exact authored observation.', source='alpha'
                )
                assert response.success
                doc = service.library.get_document(response.document_id)
                assert doc.content == 'Preserve this exact authored observation.'
                assert doc.embedding is not None
                assert doc.metadata['model_provenance']['kind'] == 'authored'
            finally:
                await service.close()

        asyncio.run(contribute())
        with sqlite3.connect(cfg.db_path) as conn:
            assert conn.execute('SELECT count(*) FROM documents').fetchone()[0] == 2
            assert all(
                row[0] is None for row in conn.execute('SELECT content_token_count FROM documents')
            )


@pytest.mark.parametrize('cloud_keys', [False, True])
def test_issue_and_design_review_use_real_local_helpers(tmp_path, monkeypatch, cloud_keys):
    import subprocess

    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'):
            if cloud_keys:
                monkeypatch.setenv(key, 'must-not-affect-local-routing')
            else:
                monkeypatch.delenv(key, raising=False)
        run = subprocess.run

        def fetch_issue(command, **kwargs):
            if command[:3] == ['gh', 'issue', 'view']:
                return subprocess.CompletedProcess(
                    command, 0, json.dumps({'title': 'Document arithmetic', 'body': 'Explain add.'})
                )
            return run(command, **kwargs)

        monkeypatch.setattr(subprocess, 'run', fetch_issue)

        async def exercise():
            from ariadne_mcp.server_debug import ariadne_analyze_issue
            from ariadne_mcp.server_knowledge import ariadne_review

            service = AriadneService()
            service._config = cfg
            monkeypatch.setattr(AriadneService, 'get', classmethod(lambda _cls: service))
            try:
                assert (await service.contribute('Arithmetic', DOCUMENT, source='alpha')).success
                issue = await ariadne_analyze_issue('fixture/repo', 7, source='alpha')
                assert issue.proposal == DOCUMENT and issue.relevant_docs
                chat.chat_response = {
                    'choices': [
                        {
                            'message': {'content': '{"verdict":"aligned","reasons":[]}'},
                            'finish_reason': 'stop',
                        }
                    ]
                }
                review = await ariadne_review(
                    'Document arithmetic',
                    ['core.py'],
                    'Explain add with an example.',
                    source='alpha',
                )
                assert review.design_alignment_verdict == 'aligned'
                assert review.relevant_docs
                with pytest.raises(LookupError, match='Unknown source'):
                    await ariadne_review(
                        'Document arithmetic',
                        ['core.py'],
                        'Explain add with an example.',
                        source='missing',
                    )
            finally:
                await service.close()

        asyncio.run(exercise())
        assert [
            payload['max_tokens']
            for route, payload, _ in chat.requests
            if route.endswith('/chat/completions')
        ] == [2048, 1024]


def test_catalog_describe_theme_and_changed_catalog(tmp_path, monkeypatch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        assert invoke(monkeypatch, 'catalog-sync', '--source', 'alpha') == 0
        chat.chat_response = {
            'choices': [
                {'message': {'content': 'Adds two integer arguments.'}, 'finish_reason': 'stop'}
            ]
        }
        assert invoke(monkeypatch, 'catalog-describe', '--source', 'alpha', '--batch') == 0
        with sqlite3.connect(cfg.db_path) as conn:
            content, raw, _timestamp = conn.execute(
                "SELECT content,metadata,updated_at FROM documents WHERE json_extract(metadata,'$.kind')='element'"
            ).fetchone()
            assert content.count('Description: ') == 1
            assert json.loads(raw)['model_provenance']['kind'] == 'generated'
        count = len(embed.requests)
        assert invoke(monkeypatch, 'catalog-sync', '--source', 'alpha') == 0
        assert invoke(monkeypatch, 'catalog-describe', '--source', 'alpha', '--batch') == 0
        assert len(embed.requests) == count
        assert invoke(monkeypatch, 'themes', 'build') == 0
        cfg.get_source_path('alpha').joinpath('core.py').write_text(
            'def multiply(a, b):\n    return a * b\n'
        )
        assert invoke(monkeypatch, 'catalog-sync', '--source', 'alpha') == 0
        with sqlite3.connect(cfg.db_path) as conn:
            rows = conn.execute(
                "SELECT title,content FROM documents WHERE json_extract(metadata,'$.kind')='element'"
            ).fetchall()
            assert len(rows) == 1 and 'multiply' in rows[0][0]
            assert 'Description:' not in rows[0][1]
            assert (
                conn.execute("SELECT count(*) FROM spool_jobs WHERE state!='applied'").fetchone()[0]
                == 0
            )


@pytest.mark.parametrize('batch', [False, True])
def test_all_document_types_and_staleness(tmp_path, monkeypatch, batch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        text = (
            DOCUMENT
            + '\n## Design\n\nThe add function returns the sum of its arguments.\n\n## Question\n\nWhat does add return?\n\n## Answer\n\nThe sum.\n\n```dot\ndigraph G { A [label="Arguments"]; B [label="Addition"]; C [label="Result"]; A -> B; B -> C; }\n```\n'
        )
        chat.chat_response = {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        assert (
            invoke(
                monkeypatch,
                'generate',
                '--source',
                'alpha',
                '--batch' if batch else '--no-batch',
                '--no-crossrefs',
                '--yes',
            )
            == 0
        )
        with sqlite3.connect(cfg.db_path) as conn:
            assert {r[0] for r in conn.execute('SELECT content_type FROM documents')} == {
                'explanation',
                'architecture',
                'qa',
                'gotcha',
                'diagram',
            }
        count = len(chat.requests)
        assert (
            invoke(
                monkeypatch,
                'generate',
                '--source',
                'alpha',
                '--batch' if batch else '--no-batch',
                '--no-crossrefs',
                '--yes',
            )
            == 0
        )
        assert len(chat.requests) == count
        assert all(
            p['max_tokens'] == 8192 for r, p, _ in chat.requests if r.endswith('/chat/completions')
        )


def test_role_cache_has_full_embeddings_and_scope(tmp_path, monkeypatch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        assert (
            invoke(
                monkeypatch,
                'generate',
                '--source',
                'alpha',
                '--types',
                'explanation',
                '--no-batch',
                '--no-crossrefs',
                '--yes',
            )
            == 0
        )

        async def run():
            service = AriadneService()
            service._config = cfg
            try:
                first = await service.ask(
                    'What does arithmetic do?', source='alpha', role='product_manager'
                )
                count = len(chat.requests)
                assert (
                    await service.ask(
                        'What does arithmetic do?', source='alpha', role='product_manager'
                    )
                ).answer == first.answer
                assert len(chat.requests) == count
                docs = service.library.list_documents('audience_response')
                assert len(docs) == 1 and docs[0].embedding is not None
                assert service.library.get_sections(docs[0].id)
                assert docs[0].metadata['model_provenance']['kind'] == 'generated'
            finally:
                await service.close()

        asyncio.run(run())


def test_local_onboard_preview_is_pure(tmp_path, monkeypatch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        assert invoke(monkeypatch, 'onboard', '--source', 'alpha') == 0
        assert not Path(cfg.db_path).exists()
        assert not cfg.get_source_path('alpha').joinpath('.ariadne').exists()
        assert chat.requests == [] and embed.requests == []


def test_local_onboard_completes_one_resumable_run(tmp_path, monkeypatch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        root = cfg.get_source_path('alpha')
        root.joinpath('core.py').rename(root / 'settings.json')
        root.joinpath('settings.json').write_text('{"timeout": 30, "enabled": true}\n')
        monkeypatch.setattr(config, '_global_config', cfg)
        assert (
            invoke(
                monkeypatch, 'onboard', '--source', 'alpha', '--approve', '--types', 'explanation'
            )
            == 0
        )
        with sqlite3.connect(cfg.db_path) as conn:
            run_id, state, checkpoints = conn.execute(
                'SELECT run_id,state,checkpoints_json FROM spool_runs'
            ).fetchone()
            assert state == 'completed'
            assert set(json.loads(checkpoints)) == {
                'structural',
                'catalog-embed',
                'catalog-describe',
                'base',
                'reverse',
                'graph',
                'themes',
                'crossrefs',
                'reconcile',
            }
            assert (
                conn.execute(
                    "SELECT count(*) FROM documents WHERE content_type='explanation'"
                ).fetchone()[0]
                == 1
            )
        count = len(chat.requests) + len(embed.requests)
        assert invoke(monkeypatch, 'batch', 'resume', run_id) == 0
        assert len(chat.requests) + len(embed.requests) == count


@pytest.mark.asyncio
async def test_bound_export_import_nested_provenance(local_deployment, tmp_path):
    from export import LibraryExporter
    from inference import InferenceContext
    from library.imports import import_bound_markdown
    from library.model_binding import ModelBindingMismatch
    from writer import LibraryWriter

    context, _, embed = local_deployment
    with context.write_session():
        library = context.open_library(operation='write')
        original = await LibraryWriter(library).add_document(
            'explanation',
            'Quoted "title"',
            DOCUMENT,
            ['core.py'],
            metadata={
                'generated': True,
                'nested': {'items': ['one', 'two: value'], 'multiline': 'one\ntwo'},
            },
            source_name='alpha',
            doc_id='round-trip',
        )
    exports = tmp_path / 'export'
    path = LibraryExporter(library).export_document(original, exports)
    second = InferenceContext(
        context.settings, tmp_path / 'other.db', tmp_path / 'other_staleness.db'
    )
    with second.write_session():
        destination = second.open_library(operation='write')
        assert await import_bound_markdown(destination, exports) == 1
        doc = destination.get_document(original.id)
        assert doc.content == original.content and doc.metadata == original.metadata
        assert doc.embedding is not None and destination.get_sections(doc.id)
        count = len(embed.requests)
        wrong = yaml.safe_load(path.read_text().split('---\n', 2)[1])
        wrong['id'] = 'mismatched'
        wrong['ariadne_export']['models']['profile_hash'] = 'incorrect'
        exports.joinpath('z-mismatch.md').write_text(
            '---\n' + yaml.safe_dump(wrong) + '---\n' + DOCUMENT
        )
        with pytest.raises(ModelBindingMismatch):
            await import_bound_markdown(destination, exports)
        assert len(embed.requests) == count
        assert destination.count_documents() == 1
    await second.aclose()
    await context.aclose()


@pytest.mark.asyncio
async def test_auxiliary_helpers_use_local_pair(local_deployment, tmp_path, monkeypatch):
    import config
    from docgen.trace_flow_llm_bridge import build_llm_bridge
    from gap_analysis import analyze_gaps
    from tests.test_trace_flow_llm_bridge import _insert_symbol
    from writer import LibraryWriter

    context, chat, embed = local_deployment
    cfg = fixture_config(tmp_path, chat, embed)
    monkeypatch.setattr(config, '_global_config', cfg)
    with context.write_session():
        library = context.open_library(operation='write')
        await LibraryWriter(library).add_document(
            'finding', 'Core helper', 'Core calls the arithmetic target.', source_name='alpha'
        )
        with library._conn_provider.acquire() as conn:
            _insert_symbol(conn, canonical_id='myapp.cursor')
            _insert_symbol(conn, canonical_id='myapp.target')
        chat.chat_response = {
            'choices': [
                {
                    'message': {
                        'content': '{"symbol_query":"myapp.target","reasoning":"The target implements arithmetic."}'
                    },
                    'finish_reason': 'stop',
                }
            ]
        }
        bridge = build_llm_bridge()
        with library._conn_provider.acquire() as conn:
            assert bridge('myapp.cursor', conn) == (
                'myapp.target',
                'The target implements arithmetic.',
            )
        chat.chat_response = {
            'choices': [
                {
                    'message': {'content': '{"summary":"Add examples","recommendations":[]}'},
                    'finish_reason': 'stop',
                }
            ]
        }
        report = await analyze_gaps([{'query': 'How does add work?', 'feedback': 'Missing usage'}])
        assert report.summary == 'Add examples'
        caps = [
            payload['max_tokens']
            for route, payload, _ in chat.requests
            if route.endswith('/chat/completions')
        ]
        assert caps == [1024, 4096]
    await context.aclose()


@pytest.mark.parametrize('coherent', [True, False])
@pytest.mark.asyncio
async def test_theme_job_application_is_atomic(local_deployment, coherent):
    from docgen.local_pipeline import plan_themes
    from docgen.spool import LocalExecutor, SpoolStore
    from writer import LibraryWriter

    context, chat, embed = local_deployment
    with context.write_session():
        library = context.open_library(operation='write')
        member = await LibraryWriter(library).add_document(
            'catalog',
            'add',
            'Adds two integers.',
            metadata={'kind': 'element', 'description': 'Adds integers.'},
            source_name='alpha',
        )
        theme = library.add_document(
            'theme', 'Pending theme', '(pending summarization)', metadata={'pending': True}
        )
        library.add_theme(
            cluster_id='arithmetic',
            doc_id=theme.id,
            member_count=1,
            resolution=1.0,
            summary_hash='',
        )
        library.set_theme_members('arithmetic', [(member.id, 1.0)])
        chat.chat_response = {
            'choices': [
                {
                    'message': {
                        'content': '# Arithmetic\n\nAdds integers.' if coherent else 'INCOHERENT'
                    },
                    'finish_reason': 'stop',
                }
            ]
        }
        store = SpoolStore(context, library)
        run_id = store.create(plan_themes(library), operation='theme-summary', scope={})
        before = len(embed.requests)
        result = await LocalExecutor(store).drive(run_id)
        assert result['state'] == 'completed'
        summary = library.get_theme('arithmetic')
        assert summary.coherent is coherent and not summary.dirty and summary.summary_hash
        stored = library.get_document(theme.id)
        if coherent:
            assert stored.embedding is not None and stored.title == 'Arithmetic'
        else:
            assert len(embed.requests) == before and stored.updated_at == theme.updated_at
    await context.aclose()
