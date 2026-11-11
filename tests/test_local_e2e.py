"""Ingest -> reopen -> retrieve -> decompose -> select evidence -> summarize.

The same scenario runs against HTTP fixtures and real, already-running servers.
For live runs, set ARIADNE_LIVE_MODEL_CONFIGS to a JSON array of config paths;
optional ARIADNE_E2E_REPORT_DIR retains requests, responses and stage results.
Compiler graph input is a controlled fixture; the production indexer is not run.
"""
import json
import os
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from ariadne_mcp.service import AriadneService
from config import Config
from inference import InferenceContext
from library.clew_embeddings import embed_clews
from llm import capture_completion_trace, chat_complete
from local_models import managed_model_profile
from local_runtime import managed_profile_id
from model_config import query_input
from tests.clew_workflow import QUESTION, ROUTE, SOURCE, build_routes, seed_graph
from tests.local_model_server import local_config, trap_inference
from writer import LibraryWriter


@contextmanager
def record_stage(report, name):
    started = time.perf_counter()
    try:
        yield
    finally:
        report.setdefault('stage_seconds', {})[name] = round(time.perf_counter() - started, 3)


async def complete_local_flow(profile, root):
    """Use real application services throughout; assert stages, not answer wording."""
    root.mkdir(parents=True, exist_ok=True)
    source = root / 'source'
    data = deepcopy(profile)
    data.update(db_path=str(root / 'library.db'), staleness_db_path=str(root / 'staleness.db'),
                default_source='alpha', ask_pipeline='compact', ask_selector_mode='llm',
                ask_synthesis=True, sources={'alpha': {'path': str(source)}})
    path = root / 'ariadne.yaml'
    path.write_text(yaml.safe_dump(data))
    cfg = Config(path)
    context = InferenceContext.from_config(cfg)
    report = {'profile_hash': context.settings.profile_hash,
              'generation_model': context.settings.chat.model,
              'embedding_model': context.settings.embeddings.model, 'completed': False}
    service = None
    with capture_completion_trace() as trace:
        try:
            with context.activate(), context.write_session(), context.open_library(operation='write') as library:
                seed_graph(library, source, 'alpha')
                build_routes(library, 'alpha')
                with record_stage(report, 'embed_routes'):
                    assert await embed_clews(library, source_name='alpha') == 2
                with record_stage(report, 'generate_document'):
                    content = await chat_complete(
                        system_prompt='Document only the supplied payment functions. Use a short Markdown explanation.',
                        user_prompt=SOURCE, max_tokens=context.settings.chat.max_output_tokens,
                        phase='e2e-document', context=context)
                with record_stage(report, 'embed_and_store_document'):
                    document = await LibraryWriter(library).add_document(
                        'explanation', 'Payment receipt workflow', content,
                        source_files=['payments.py'], source_name='alpha', doc_id='payment-guide')
                for row in [document, *library.get_chunks(document.id), *library.get_sections(document.id)]:
                    assert row.embedding.shape == (3072,)
                    assert np.isfinite(row.embedding).all()
                    assert np.linalg.norm(row.embedding.astype(np.float64)) == pytest.approx(1)
                original_vector = document.embedding.tobytes()
                report['ingested'] = {'document_id': document.id, 'dimensions': 3072}
            await context.aclose()

            with record_stage(report, 'reopen_library'):
                context = InferenceContext.from_config(Config(path))
                with context.open_library(operation='inspect') as library:
                    saved = library.get_document(document.id)
                    assert saved.content == content and saved.embedding.tobytes() == original_vector
            report['reopened'] = True
            service = AriadneService()
            service._config, service._inference_context = Config(path), context
            query_requests = []

            async def record_query_request(request):
                if request.url.path.endswith('/embeddings'):
                    query_requests.append(json.loads(request.content))

            context.embeddings._local_transport.client().event_hooks['request'].append(record_query_request)
            with record_stage(report, 'embed_query_and_search'):
                found = await service.search(query=QUESTION, source='alpha')
            assert document.id in {row.id for row in found.documents}
            assert any(request['input'] == [query_input(QUESTION)]
                       and request['model'] == context.settings.embeddings.request_model
                       for request in query_requests)
            report['query_embedding_requests'] = query_requests
            report['retrieved'] = [row.id for row in found.documents]
            with record_stage(report, 'ask'):
                answer = await service.ask(QUESTION, source='alpha')
            report['answer'] = answer.model_dump(mode='json')
            compact = answer.graph_diagnostics.get('compact', {})
            assert compact.get('status') == 'complete', compact
            assert compact['obligations'] and compact['selected_by_obligation']
            assert not compact['unresolved_obligations'] and not compact['unknown_ids']
            assert compact['expanded_cards'] and compact['chunks'] > 0
            assert not compact['expansion_gaps']
            assert set(ROUTE).issubset(answer.selected_symbols)
            assert answer.answer.strip() and answer.llm_calls == 3
            assert [call['phase'] for call in trace] == [
                'e2e-document', 'scip-obligation-plan', 'scip-route-family-select', 'completion']
            assert all(call['status'] == 'ok' and call['model'] == context.settings.chat.model
                       and call['response'].strip() for call in trace)
            # The summary must receive materialized source, not merely retrieval labels.
            assert "return {'receipt': amount}" in trace[-1]['messages'][1]['content']
            assert 'payments.py' in answer.answer
            report['completed'] = True
            return report
        finally:
            report['calls'] = trace
            (root / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
            if service is not None:
                await service.close()
            await context.aclose()


def script_model(chat, monkeypatch):
    handler_class = chat.server.RequestHandlerClass
    reply = handler_class.reply

    def scripted_reply(handler, status, body):
        if handler.path.endswith('/chat/completions') and status == 200:
            messages = chat.requests[-1][1]['messages']
            system, user = messages[0]['content'], messages[1]['content']
            if 'plan proof obligations' in system:
                content = 'C1: Establish how payments.accept_payment authorizes a payment and records its receipt.'
            elif 'select route families' in system:
                families = [match.group(1) for line in user.splitlines()
                            if (match := re.match(r'\s+(F\d+)\.', line))
                            and all(symbol in line for symbol in ROUTE)]
                content = 'O1: ' + ' '.join(families)
            else:
                content = '# Payment receipt workflow\n\nThe entry point authorizes payment and records a receipt (payments.py:2).'
            body = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 32, 'completion_tokens': 8}}
        reply(handler, status, body)

    monkeypatch.setattr(handler_class, 'reply', scripted_reply)


@pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
@pytest.mark.parametrize('synthesis_cap', [4096, 8192])
async def test_native_complete_local_flow(native_pair, tmp_path, monkeypatch, synthesis_cap):
    context, chat, embed = native_pair
    profile = managed_model_profile(managed_profile_id(context.settings))
    profile['llm']['base_url'], profile['embeddings']['base_url'] = chat.base_url, embed.base_url
    profile['ask_synthesis_max_tokens'] = synthesis_cap
    script_model(chat, monkeypatch)
    try:
        report = await complete_local_flow(profile, tmp_path / 'flow')
        assert report['calls'][-1]['max_tokens'] == synthesis_cap
    finally:
        await context.aclose()


async def test_bf16_complete_local_flow(local_deployment, tmp_path, monkeypatch):
    context, chat, embed = local_deployment
    script_model(chat, monkeypatch)
    try:
        await complete_local_flow(local_config(chat.base_url, embed.base_url), tmp_path / 'flow')
    finally:
        await context.aclose()


@pytest.mark.parametrize('native_pair', ['accuracy'], indirect=True)
async def test_27b_complete_flow_with_16k_budgets(native_pair, tmp_path, monkeypatch):
    context, chat, embed = native_pair
    profile = managed_model_profile(managed_profile_id(context.settings))
    profile['llm'].update(base_url=chat.base_url, max_output_tokens=16384,
                          selector_max_tokens=16384)
    profile['embeddings']['base_url'] = embed.base_url
    profile['ask_synthesis_max_tokens'] = 16384
    script_model(chat, monkeypatch)
    try:
        report = await complete_local_flow(profile, tmp_path / 'flow')
        assert [call['max_tokens'] for call in report['calls']] == [16384] * 4
    finally:
        await context.aclose()


@pytest.mark.parametrize('config_path', json.loads(os.environ.get('ARIADNE_LIVE_MODEL_CONFIGS', '[]')))
async def test_live_complete_local_flow(config_path, tmp_path, monkeypatch):
    cfg = Config(Path(config_path))
    settings = cfg.model_settings(execution=True)
    assert settings.local
    trap_inference(monkeypatch, [settings.chat.base_url, settings.embeddings.base_url])
    root = tmp_path / 'live'
    if directory := os.environ.get('ARIADNE_E2E_REPORT_DIR'):
        root = Path(directory) / settings.profile_hash
        assert not (root / 'library.db').exists(), 'Use a fresh report directory for each run'
    await complete_local_flow(cfg._explicit, root)
