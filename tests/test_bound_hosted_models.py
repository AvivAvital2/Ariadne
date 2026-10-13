import json

import httpx
import pytest

from docgen.llm.batch import BatchRequest
from docgen.llm.factory import batch_strategy_for
from inference import InferenceContext
from model_config import resolve_models
from tests.local_model_server import DOCUMENT
from writer import LibraryWriter


@pytest.mark.parametrize(
    ('provider', 'model'), [('openai', 'gpt-5.5'), ('anthropic', 'claude-opus-4-8')]
)
@pytest.mark.asyncio
async def test_new_bound_hosted_unary_batch_and_embeddings(tmp_path, monkeypatch, provider, model):
    monkeypatch.setenv('OPENAI_API_KEY', 'hosted-openai-test')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'hosted-anthropic-test')
    calls = []
    message = {
        'content': [{'type': 'text', 'text': DOCUMENT}],
        'usage': {'input_tokens': 30, 'output_tokens': 200},
    }
    completion = {'choices': [{'message': {'content': DOCUMENT}}]}

    def respond(request):
        assert request.url.host == 'hosted-fixture.invalid'
        path = request.url.path
        calls.append(path)
        if path.endswith('/embeddings'):
            payload = json.loads(request.content)
            assert payload['model'] == 'text-embedding-3-large'
            assert payload['dimensions'] == 3072
            return httpx.Response(
                200,
                json={
                    'data': [
                        {'index': i, 'embedding': [1.0] * 3072}
                        for i in range(len(payload['input']))
                    ]
                },
            )
        if path.endswith(('/messages', '/chat/completions')):
            payload = json.loads(request.content)
            assert payload['model'] == model
            cap = 'max_completion_tokens' if provider == 'openai' else 'max_tokens'
            assert payload[cap] == 8192
            return httpx.Response(200, json=completion if provider == 'openai' else message)
        if path.endswith('/files'):
            assert b'"max_completion_tokens": 8192' in request.content
            return httpx.Response(200, json={'id': 'input-file'})
        if path.endswith('/batches'):
            if provider == 'anthropic':
                assert json.loads(request.content)['requests'][0]['params']['max_tokens'] == 8192
            return httpx.Response(200, json={'id': 'batch-test'})
        if path.endswith('/batch-test'):
            return httpx.Response(
                200,
                json={
                    'id': 'batch-test',
                    'status': 'completed',
                    'processing_status': 'ended',
                    'request_counts': {'total': 1, 'completed': 1, 'succeeded': 1},
                    'output_file_id': 'output-file',
                },
            )
        if path.endswith('/results'):
            return httpx.Response(
                200,
                text=json.dumps(
                    {'custom_id': 'doc', 'result': {'type': 'succeeded', 'message': message}}
                ),
            )
        if path.endswith('/content'):
            return httpx.Response(
                200,
                text=json.dumps(
                    {'custom_id': 'doc', 'response': {'status_code': 200, 'body': completion}}
                ),
            )
        raise AssertionError(f'Unexpected hosted fixture request: {path}')

    client_type = httpx.AsyncClient

    def client(**kwargs):
        return client_type(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(httpx, 'AsyncClient', client)
    settings = resolve_models(
        {
            'llm': {
                'provider': provider,
                'model': model,
                'base_url': 'https://hosted-fixture.invalid/v1',
            },
            'embeddings': {
                'provider': 'openai',
                'model': 'text-embedding-3-large',
                'base_url': 'https://hosted-fixture.invalid/v1',
            },
        }
    )
    ctx = InferenceContext(settings, tmp_path / 'hosted.db', tmp_path / 'hosted_staleness.db')
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        assert ctx.binding is not None
        assert await ctx.chat.call('System', 'User', max_tokens=8192) == DOCUMENT
        batch = batch_strategy_for(ctx.chat)
        submitted = await batch.submit_batch(
            [BatchRequest('doc', 'System', 'User', max_tokens=8192)]
        )
        assert (await batch.poll_batch(submitted.batch_id)).succeeded == 1
        assert await batch.fetch_batch_results(submitted.batch_id) == {'doc': DOCUMENT}
        doc = await LibraryWriter(library).add_document(
            'explanation', 'Arithmetic', DOCUMENT, metadata={'generated': True}
        )
        assert doc.embedding.shape == (3072,)
        assert library.get_sections(doc.id)
        assert doc.metadata['model_provenance']['profile_hash'] == settings.profile_hash
        assert any(path.endswith('/embeddings') for path in calls)
    await ctx.aclose()
