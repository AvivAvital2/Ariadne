"""Gemma generation shares Qwen's embedding space without rebinding old libraries."""

from copy import deepcopy

import pytest

from config import Config
from docgen.llm.local import LocalInferenceError, parse_chat_response
from inference import InferenceContext
from library.model_binding import ModelBindingMismatch
from local_models import (
    GEMMA_ARTIFACTS, GEMMA_PROFILE_ID, MAC_ACCURACY_PROFILE_ID, MAC_ARTIFACTS,
    gemma_model_profile, mac_accuracy_profile, mac_model_profile, managed_model_profile,
)
from local_runtime import MacRuntime, managed_profile_id, server_command
from model_config import ModelConfigurationError, resolve_models
from web.external_models import build_profile, load_profiles, save_profile
from web.projects import ProjectStore


def test_gemma_preserves_the_complete_qwen_embedding_identity():
    settings = resolve_models(gemma_model_profile(), execution=True)
    qwen = resolve_models(mac_accuracy_profile(), execution=True)
    assert settings.embeddings == qwen.embeddings
    assert settings.embeddings.embedding_space_id == '9d40de70f1deb65107f8bd5fe83f24228f1044d6c8cb905b119b10e971c97db2'
    assert settings.profile_hash != qwen.profile_hash
    assert settings.chat.identity['generation_contract'] == 'gemma4-nonthinking-v1'
    assert settings.chat.request_model == 'google/gemma-4-12b-it'
    assert settings.chat.context_window == 16384
    assert settings.chat.max_output_tokens == 8192
    assert GEMMA_ARTIFACTS['embeddings'] == MAC_ARTIFACTS['embeddings']
    assert managed_model_profile(GEMMA_PROFILE_ID) == gemma_model_profile()
    assert resolve_models(mac_model_profile()).profile_hash == '5e55a0a453fdf8a6a00ed4d44ca0d6457ceaf4ad8acd99cd90e8785ec0ab8a8e'


@pytest.mark.parametrize(('key', 'value'), [
    ('chat_template_kwargs', {'enable_thinking': True}),
    ('reasoning_format', 'none'), ('reasoning_effort', 'xhigh'),
    ('min_p', float('nan')), ('repeat_penalty', 0),
])
def test_gemma_rejects_requests_that_change_its_generation_contract(key, value):
    profile = gemma_model_profile()
    profile['llm']['request_options'][key] = value
    with pytest.raises(ValueError):
        resolve_models(profile, execution=True)


def test_gemma_does_not_accept_embeddinggemma_or_change_qwen_dimensions():
    for mutation in ({'model': 'google/embeddinggemma-2'}, {'dimensions': 768}):
        profile = gemma_model_profile()
        profile['embeddings'].update(mutation)
        with pytest.raises(ModelConfigurationError):
            resolve_models(profile, execution=True)


def test_gemma_commands_select_correct_artifacts_template_and_context(tmp_path):
    settings = resolve_models(gemma_model_profile())
    assert managed_profile_id(settings) == GEMMA_PROFILE_ID
    chat = server_command(settings, 'llm', tmp_path / 'llama-server', tmp_path)
    embed = server_command(settings, 'embeddings', tmp_path / 'llama-server', tmp_path)
    assert chat[chat.index('--model') + 1] == str(tmp_path / GEMMA_ARTIFACTS['llm'].filename)
    assert chat[chat.index('--ctx-size') + 1] == '16384'
    assert chat[chat.index('--chat-template-kwargs') + 1] == '{"enable_thinking":false}'
    assert chat[chat.index('--reasoning-format') + 1] == 'auto'
    assert embed == server_command(resolve_models(mac_accuracy_profile()), 'embeddings',
                                   tmp_path / 'llama-server', tmp_path)


@pytest.mark.parametrize('content', [
    '<|channel>thought\nunfinished', '<channel|>answer', '<|think|>reasoning',
])
def test_unparsed_gemma_reasoning_cannot_be_persisted_as_an_answer(content):
    with pytest.raises(LocalInferenceError, match='reasoning'):
        parse_chat_response({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]})


def test_identical_embedding_space_does_not_allow_rebinding_generation(tmp_path):
    paths = (tmp_path / 'library.db', tmp_path / 'staleness.db')
    qwen = InferenceContext(resolve_models(mac_accuracy_profile()), *paths)
    with qwen.write_session():
        pass
    before = [path.read_bytes() for path in paths]
    with pytest.raises(ModelBindingMismatch):
        InferenceContext(resolve_models(gemma_model_profile()), *paths)
    assert [path.read_bytes() for path in paths] == before


def test_gemma_external_profile_round_trips_beside_existing_qwen(tmp_path):
    body = {'chat_url': 'https://chat.example/v1', 'embedding_url': 'https://embed.example/v1'}
    path = tmp_path / 'profiles.json'
    qwen = build_profile(body)
    qwen_id = save_profile(path, qwen)
    gemma = build_profile({**body, 'model_profile': GEMMA_PROFILE_ID})
    gemma_id = save_profile(path, gemma)
    assert gemma_id != qwen_id
    assert load_profiles(path) == {qwen_id: qwen, gemma_id: gemma}
    assert build_profile({**body, 'model_profile': MAC_ACCURACY_PROFILE_ID}) == qwen
    tampered = deepcopy(gemma)
    tampered['llm']['request_options']['chat_template_kwargs']['enable_thinking'] = True
    before = path.read_bytes()
    with pytest.raises(ValueError):
        save_profile(path, tampered)
    assert path.read_bytes() == before


@pytest.mark.parametrize('selected', ['unknown', None, False, {}, []])
def test_external_profile_choice_is_a_supported_pair(selected):
    with pytest.raises(ValueError, match='supported'):
        build_profile({'model_profile': selected})


def test_gemma_managed_project_selection_keeps_qwen_project_intact(tmp_path, monkeypatch):
    path = tmp_path / 'ariadne.yaml'
    path.write_text('sources: {}\n')
    monkeypatch.setattr('local_runtime.platform_error', lambda: None)
    store = ProjectStore(Config(path))
    store.runtime._state.update(phase='ready')
    qwen = store.prepare(MAC_ACCURACY_PROFILE_ID, 'demo', str(tmp_path))
    before = store.path(qwen['id']).read_bytes()
    choice = next(choice for choice in store.choices() if choice['id'] == GEMMA_PROFILE_ID)
    assert not choice['available'] and choice['preview_available']
    with pytest.raises(ValueError, match='saved for this project'):
        store.prepare(GEMMA_PROFILE_ID, 'demo', str(tmp_path))
    store.runtime = MacRuntime(store.runtime.root, settings=resolve_models(gemma_model_profile()))
    store.runtime._state.update(phase='ready')
    gemma = store.prepare(GEMMA_PROFILE_ID, 'demo', str(tmp_path))
    assert gemma['id'] != qwen['id'] and 'Gemma' in gemma['label']
    assert store.path(qwen['id']).read_bytes() == before
    reopened = ProjectStore(Config(path))
    assert {p['id'] for p in reopened.list_projects()} == {qwen['id'], gemma['id']}
    assert Config(reopened.path(gemma['id'])).model_settings().identity == resolve_models(gemma_model_profile()).identity
