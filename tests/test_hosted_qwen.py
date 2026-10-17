"""Production hosted Qwen contracts, independent of live credentials."""
import json

import httpx
import numpy as np
import pytest

from docgen.llm.factory import batch_strategy_for, make_llm_provider
from docgen.llm.local import EmbeddingContractError, LocalInferenceError
from embedding import EmbeddingConfig, EmbeddingService
from model_config import CHAT_MODEL, EMBEDDING_MODEL, HOSTED_RECIPE, query_input, resolve_models


def profile():
    return {'llm': {'provider': 'deepinfra', 'model': CHAT_MODEL},
            'embeddings': {'provider': 'deepinfra', 'model': EMBEDDING_MODEL}}


def test_hosted_identity_credentials_and_batch_contract(monkeypatch):
    monkeypatch.setenv('DEEPINFRA_API_KEY', 'test-deepinfra')
    settings = resolve_models(profile())
    assert not settings.local
    assert settings.embeddings.recipe == HOSTED_RECIPE
    assert settings.chat.credential() == settings.embeddings.credential() == 'test-deepinfra'
    bad = profile()
    bad['embeddings'] = {'provider': 'openai'}
    with pytest.raises(ValueError, match='pair'):
        resolve_models(bad)
    bad = profile()
    bad['execution'] = {'batch_mode': 'always'}
    with pytest.raises(ValueError, match='live'):
        resolve_models(bad)
    provider = make_llm_provider('deepinfra', model=CHAT_MODEL, api_key='test')
    with pytest.raises(ValueError, match='live'):
        batch_strategy_for(provider)


def test_hosted_qwen_cannot_adopt_unknown_legacy_vectors(tmp_path):
    from inference import InferenceContext
    from library import Library
    from library.model_binding import ModelBindingMismatch

    path = tmp_path / 'library.db'
    with Library(path) as library:
        library.add_document(content_type='explanation', title='Old', content='Unknown model pair')
    with pytest.raises(ModelBindingMismatch, match='unbound'):
        InferenceContext(resolve_models(profile()), path, tmp_path / 'staleness.db')


async def test_hosted_chat_disables_reasoning_and_rejects_truncation():
    provider = make_llm_provider('deepinfra', model=CHAT_MODEL, api_key='test')
    stop = 'stop'

    def respond(request):
        payload = json.loads(request.content)
        assert payload['reasoning_effort'] == 'none'
        assert payload['stream'] is False
        assert payload['max_tokens'] == 100
        return httpx.Response(200, json={
            'choices': [{'finish_reason': stop, 'message': {'content': 'A complete answer.'}}],
        })

    provider._client = httpx.AsyncClient(base_url=provider.base_url, transport=httpx.MockTransport(respond))
    try:
        assert await provider.call('system', 'question', max_tokens=100) == 'A complete answer.'
        stop = 'length'
        with pytest.raises(LocalInferenceError, match='Incomplete'):
            await provider.call('system', 'question', max_tokens=100)
    finally:
        await provider.aclose()


async def test_hosted_embeddings_use_query_recipe_native_width_and_normalization():
    calls = []
    width = 4096

    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert 'dimensions' not in payload
        assert payload['encoding_format'] == 'float'
        rows = [{'index': i, 'embedding': [float(i + 1)] * width}
                for i, _ in enumerate(payload['input'])]
        return httpx.Response(200, json={'data': rows[::-1]})

    service = EmbeddingService(EmbeddingConfig(provider='deepinfra', model=EMBEDDING_MODEL, api_key='test'))
    service._client = httpx.AsyncClient(base_url=service.config.get_base_url(), transport=httpx.MockTransport(respond))
    try:
        docs = await service.embed_documents(['Document A', 'Document B'])
        query = await service.embed_query('Where is A?')
        assert calls[0]['input'] == ['Document A', 'Document B']
        assert calls[1]['input'] == [query_input('Where is A?')]
        for vector in [*docs, query]:
            assert vector.shape == (3072,) and vector.dtype == np.float32
            assert float(np.linalg.norm(vector)) == pytest.approx(1.0)
        await service.embed_query('Where is A?')
        assert len(calls) == 2
        width = 3072
        with pytest.raises(EmbeddingContractError):
            await service.embed_documents(['Wrong native dimensions'])
    finally:
        await service.close()
