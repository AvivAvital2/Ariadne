"""User-owned servers must pass the production adapter without touching a library."""

import json

import pytest

from local_models import mac_accuracy_profile
from tests.llamacpp_server import LlamaServer


@pytest.fixture
def endpoints():
    profile = mac_accuracy_profile()
    chat = LlamaServer(profile['llm']['served_model'], thinking=True)
    embed = LlamaServer(profile['embeddings']['served_model'], embeddings=True)
    try:
        yield chat, embed
    finally:
        chat.close()
        embed.close()


def request_for(endpoints):
    chat, embed = endpoints
    return {'chat_url': chat.base_url, 'embedding_url': embed.base_url}


async def test_connection_smoke_uses_native_chat_document_and_query_contract(endpoints):
    from web.external_models import build_profile, probe_profile

    profile = build_profile(request_for(endpoints))
    result = await probe_profile(profile)
    chat, embed = endpoints
    assert result['chat']['completed'] is True
    assert result['embeddings']['dimensions'] == 3072
    assert result['embeddings']['normalized'] is True
    assert profile['llm']['model_revision'] == mac_accuracy_profile()['llm']['model_revision']
    assert profile['embeddings']['recipe'] == mac_accuracy_profile()['embeddings']['recipe']
    assert any(p.endswith('/apply-template') for p, _ in chat.requests)
    completions = [body for p, body in chat.requests if p.endswith('/chat/completions')]
    assert len(completions) == 1 and completions[0]['max_tokens'] == 2048
    assert completions[0]['chat_template_kwargs'] == {'enable_thinking': True}
    vectors = [body for p, body in embed.requests if p.endswith('/embeddings')]
    assert len(vectors) == 2 and all('dimensions' not in body for body in vectors)
    assert not vectors[0]['input'][0].startswith('Instruct:')
    assert vectors[1]['input'][0].startswith('Instruct:')


@pytest.mark.parametrize('extra', [{'api_key': 'do-not-save'}, {'model': 'replacement'}])
def test_config_rejects_unknown_fields_and_raw_secrets(endpoints, extra):
    from web.external_models import build_profile

    with pytest.raises(ValueError, match='server URLs'):
        build_profile({**request_for(endpoints), **extra})
    assert all(not server.requests for server in endpoints)


@pytest.mark.parametrize('url', ['file:///tmp/model', 'http://user:secret@host/v1',
                                'https://host/v1?token=secret', 'http://host/api'])
def test_endpoint_configuration_requires_explicit_supported_url(url):
    from web.external_models import build_profile

    with pytest.raises(ValueError, match='base_url'):
        build_profile({'chat_url': url, 'embedding_url': 'https://embedding.example/v1'})


async def test_missing_credential_fails_before_any_request(endpoints, monkeypatch):
    from web.external_models import build_profile, probe_profile

    monkeypatch.delenv('ARIADNE_TEST_MISSING_CREDENTIAL', raising=False)
    profile = build_profile({**request_for(endpoints),
                             'chat_api_key_env': 'ARIADNE_TEST_MISSING_CREDENTIAL'})
    with pytest.raises(ValueError, match='ARIADNE_TEST_MISSING_CREDENTIAL'):
        await probe_profile(profile)
    assert all(not server.requests for server in endpoints)


@pytest.mark.parametrize(('message', 'finish'), [({'content': ''}, 'stop'),
                                          ({'content': 'partial'}, 'length'),
                                          ({'reasoning_content': 'private'}, 'stop')])
async def test_incomplete_chat_never_passes_connection_check(endpoints, message, finish):
    from web.external_models import build_profile, probe_profile

    endpoints[0].overrides['/native/v1/chat/completions'] = {
        'choices': [{'message': message, 'finish_reason': finish}]}
    with pytest.raises(ValueError, match='Chat'):
        await probe_profile(build_profile(request_for(endpoints)))
    assert not any(p.endswith('/embeddings') for p, _ in endpoints[1].requests)


@pytest.mark.parametrize('vector', [[1.0] * 3072, [0.0] * 4096, [float('nan')] * 4096])
async def test_native_embedding_contract_failures_do_not_pass(endpoints, vector):
    from web.external_models import build_profile, probe_profile

    endpoints[1].overrides['/native/v1/embeddings'] = {'data': [{'index': 0, 'embedding': vector}]}
    with pytest.raises(ValueError, match='Embedding'):
        await probe_profile(build_profile(request_for(endpoints)))


def test_endpoint_fingerprint_changes_without_changing_model_artifact_identity():
    from model_config import resolve_models
    from web.external_models import build_profile, profile_id

    first = build_profile({'chat_url': 'https://first.example/v1/',
                           'embedding_url': 'https://embed.example/v1'})
    same = build_profile({'chat_url': 'https://first.example/v1',
                          'embedding_url': 'https://embed.example/v1'})
    other = build_profile({'chat_url': 'https://second.example/v1',
                           'embedding_url': 'https://embed.example/v1'})
    assert profile_id(first) == profile_id(same)
    assert profile_id(first) != profile_id(other)
    assert resolve_models(first).identity == resolve_models(other).identity
    assert 'managed_local_profile' not in json.dumps(first)


@pytest.mark.parametrize('credential', [False, 0, [], 'contains spaces'])
def test_credential_reference_must_be_string_even_when_false_valued(credential):
    from web.external_models import build_profile

    with pytest.raises(ValueError, match='environment-variable names'):
        build_profile({'chat_url': 'https://chat.example/v1',
                       'embedding_url': 'https://embed.example/v1',
                       'chat_api_key_env': credential})


def test_omitted_request_timeout_preserves_the_existing_profile_and_identity():
    from model_config import identity_hash, resolve_models
    from web.external_models import build_profile, profile_id

    body = {'chat_url': 'https://chat.example/v1',
            'embedding_url': 'https://embed.example/v1'}
    previous = mac_accuracy_profile()
    previous['llm'].update(base_url=body['chat_url'], api_key_env=None)
    previous['embeddings'].update(base_url=body['embedding_url'], api_key_env=None)

    profile = build_profile(body)

    assert profile == previous
    assert profile_id(profile) == 'external-' + identity_hash(previous)[:24]
    settings = resolve_models(profile)
    assert settings.chat.timeout_seconds == 600
    assert settings.embeddings.timeout_seconds == 120


@pytest.mark.parametrize('seconds', [600, 7200, 14400])
def test_explicit_request_timeout_applies_to_both_clients_without_rebinding_models(seconds):
    from model_config import resolve_models
    from web.external_models import build_profile, profile_id

    body = {'chat_url': 'https://chat.example/v1',
            'embedding_url': 'https://embed.example/v1'}
    previous = build_profile(body)
    profile = build_profile({**body, 'request_timeout_seconds': seconds})

    assert profile['llm']['timeout_seconds'] == seconds
    assert profile['embeddings']['timeout_seconds'] == seconds
    settings = resolve_models(profile)
    assert settings.chat.timeout_seconds == seconds
    assert settings.embeddings.timeout_seconds == seconds
    assert settings.identity == resolve_models(previous).identity
    assert profile_id(profile) != profile_id(previous)


@pytest.mark.parametrize('seconds', [None, True, False, 0, 599, 14401, 600.5, '7200'])
def test_request_timeout_rejects_non_integer_or_out_of_range_values(seconds):
    from web.external_models import build_profile

    with pytest.raises(ValueError, match=r'600.*14400'):
        build_profile({'chat_url': 'https://chat.example/v1',
                       'embedding_url': 'https://embed.example/v1',
                       'request_timeout_seconds': seconds})


@pytest.mark.parametrize('seconds', [None, 7200])
async def test_probe_uses_selected_timeout_for_overall_check_and_real_requests(
        endpoints, monkeypatch, seconds):
    import asyncio

    import httpx

    from web.external_models import build_profile, probe_profile

    original_timeout = asyncio.timeout
    original_send = httpx.AsyncClient.send
    deadlines, requests = [], []

    def recorded_timeout(delay):
        deadlines.append(delay)
        return original_timeout(delay)

    async def recorded_send(client, request, **kwargs):
        requests.append((request.url, dict(request.extensions['timeout'])))
        return await original_send(client, request, **kwargs)

    monkeypatch.setattr('web.external_models.asyncio.timeout', recorded_timeout)
    monkeypatch.setattr(httpx.AsyncClient, 'send', recorded_send)
    body = request_for(endpoints)
    if seconds is not None:
        body['request_timeout_seconds'] = seconds
    result = await probe_profile(build_profile(body))

    expected = seconds if seconds is not None else 120
    assert deadlines[0] == expected
    assert result['chat']['completed']
    completions = [timeout for url, timeout in requests if url.path.endswith('/chat/completions')]
    embeddings = [timeout for url, timeout in requests if url.path.endswith('/embeddings')]
    assert len(completions) == 1 and len(embeddings) == 2
    for timeout in completions + embeddings:
        assert timeout == {'connect': expected, 'read': expected, 'write': expected, 'pool': expected}
    if seconds is not None:
        assert all(set(timeout.values()) == {seconds} for _url, timeout in requests)
