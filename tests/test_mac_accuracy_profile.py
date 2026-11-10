"""Accuracy candidate contracts must not reinterpret an existing Mac library."""

from dataclasses import replace

import pytest

from model_config import ModelConfigurationError, resolve_models


def test_accuracy_profile_pins_selected_pair_and_explicit_thinking_recipe():
    from local_models import MAC_ACCURACY_ARTIFACTS, mac_accuracy_profile

    settings = resolve_models(mac_accuracy_profile(), execution=True)
    assert settings.chat.model == 'bartowski/Qwen3.8-27B-GGUF'
    assert settings.chat.request_model == 'Qwen/Qwen3.8-27B'
    assert settings.chat.model_revision == '87410e7b9a5c21e6ca70c0c9cc5237de909fe621'
    artifact = MAC_ACCURACY_ARTIFACTS['llm']
    assert (artifact.filename, artifact.size, artifact.sha256) == (
        'Qwen3.8-27B-Q4_K_M.gguf', 17442399968,
        '880a52af8317974f7b4b064a6bc29ff49e188b91aa218e08eea3078b6b1cc673',
    )
    assert artifact.repository == settings.chat.model
    assert artifact.revision == settings.chat.model_revision
    assert settings.chat.request_options == {
        'chat_template_kwargs': {'enable_thinking': True},
        'reasoning_effort': 'xhigh', 'reasoning_format': 'deepseek',
        'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
        'min_p': 0.0, 'presence_penalty': 0.0, 'repeat_penalty': 1.0,
    }
    assert settings.chat.context_window == 32768
    assert settings.chat.max_output_tokens == 8192  # Total reasoning plus final.
    assert settings.embeddings.dimensions == 3072
    assert settings.chat.identity['generation_contract'] == 'qwen3.8-thinking-xhigh-v1'


def test_new_generator_preserves_legacy_identity_and_exact_embedding_space():
    from local_models import mac_accuracy_profile, mac_model_profile

    legacy = resolve_models(mac_model_profile(), execution=True)
    new = resolve_models(mac_accuracy_profile(), execution=True)
    assert legacy.profile_hash == '5e55a0a453fdf8a6a00ed4d44ca0d6457ceaf4ad8acd99cd90e8785ec0ab8a8e'
    assert new.profile_hash != legacy.profile_hash
    assert new.embeddings.identity == legacy.embeddings.identity
    assert new.embeddings.embedding_space_id == '9d40de70f1deb65107f8bd5fe83f24228f1044d6c8cb905b119b10e971c97db2'
    assert legacy.chat.request_options['chat_template_kwargs'] == {'enable_thinking': False}


@pytest.mark.parametrize(('key', 'value'), [
    ('chat_template_kwargs', {'enable_thinking': False}),
    ('chat_template_kwargs', {'enable_thinking': True, 'unexpected': True}),
    ('reasoning_effort', 'none'), ('reasoning_format', 'none'),
    ('min_p', float('inf')), ('min_p', -0.1), ('min_p', True),
    ('repeat_penalty', 0), ('repeat_penalty', True),
    ('repetition_penalty', 1.0), ('unexpected_option', 1),
])
def test_accuracy_profile_rejects_ambiguous_or_invalid_request_options(key, value):
    from local_models import mac_accuracy_profile

    data = mac_accuracy_profile()
    data['llm']['request_options'][key] = value
    with pytest.raises((ModelConfigurationError, ValueError)):
        resolve_models(data, execution=True)


def test_profile_registry_returns_fresh_data_and_rejects_unknown_ids():
    from local_models import (
        GEMMA_PROFILE_ID,
        MAC_ACCURACY_PROFILE_ID,
        MAC_PROFILE_ID,
        MAC_PROFILE_IDS,
        mac_accuracy_profile,
        mac_model_profile,
        managed_model_profile,
    )

    assert set(MAC_PROFILE_IDS) == {MAC_PROFILE_ID, MAC_ACCURACY_PROFILE_ID, GEMMA_PROFILE_ID}
    assert managed_model_profile(MAC_PROFILE_ID) == mac_model_profile()
    assert managed_model_profile(MAC_ACCURACY_PROFILE_ID) == mac_accuracy_profile()
    data = managed_model_profile(MAC_ACCURACY_PROFILE_ID)
    data['llm']['request_options']['chat_template_kwargs']['enable_thinking'] = False
    assert managed_model_profile(MAC_ACCURACY_PROFILE_ID) == mac_accuracy_profile()
    with pytest.raises(ValueError, match='Unknown'):
        managed_model_profile('not-a-profile')


def test_legacy_library_cannot_be_rebound_to_accuracy_model(tmp_path):
    from inference import InferenceContext
    from library.model_binding import ModelBindingMismatch
    from local_models import mac_accuracy_profile, mac_model_profile

    main, sidecar = tmp_path / 'library.db', tmp_path / 'staleness.db'
    legacy = resolve_models(mac_model_profile(), execution=True)
    context = InferenceContext(legacy, main, sidecar)
    with context.write_session(), context.open_library(operation='write'):
        pass
    before = main.read_bytes(), sidecar.read_bytes()
    with pytest.raises(ModelBindingMismatch):
        InferenceContext(resolve_models(mac_accuracy_profile()), main, sidecar)
    assert (main.read_bytes(), sidecar.read_bytes()) == before
    with InferenceContext(legacy, main, sidecar).open_library(operation='inspect'):
        pass


def test_runtime_defaults_to_accuracy_and_launches_its_exact_artifact(tmp_path):
    import json

    from local_models import MAC_ACCURACY_ARTIFACTS, MAC_ACCURACY_PROFILE_ID
    from local_runtime import MacRuntime, server_command

    runtime = MacRuntime(tmp_path)
    assert runtime.status()['profile_id'] == MAC_ACCURACY_PROFILE_ID
    assert runtime.status()['profile_hash'] == runtime.settings.profile_hash
    assert runtime.status()['qualification'] == 'unqualified'
    command = server_command(runtime.settings, 'llm', tmp_path / 'llama-server', tmp_path)
    assert command[command.index('--model') + 1] == str(tmp_path / MAC_ACCURACY_ARTIFACTS['llm'].filename)
    assert json.loads(command[command.index('--chat-template-kwargs') + 1]) == {'enable_thinking': True}
    assert command[command.index('--reasoning-format') + 1] == 'deepseek'


@pytest.mark.parametrize('selected', ['legacy', 'accuracy', 'gemma'])
async def test_profile_switch_during_running_servers_fails_without_killing_or_rebinding(tmp_path, selected):
    from local_models import MAC_PROFILE_IDS, managed_model_profile
    from local_runtime import MacRuntime, RuntimeSetupError

    profile_id = dict(zip(('legacy', 'accuracy', 'gemma'), MAC_PROFILE_IDS, strict=True))[selected]
    runtime = MacRuntime(tmp_path, settings=resolve_models(managed_model_profile(profile_id)))
    sentinel = object()
    runtime.processes['llm'] = sentinel
    before = runtime.settings
    for other in MAC_PROFILE_IDS:
        if other != profile_id:
            for operation in (runtime.setup, runtime.start):
                with pytest.raises(RuntimeSetupError, match='Stop'):
                    await operation(profile_id=other)
                assert runtime.settings is before
                assert runtime.processes == {'llm': sentinel}


async def test_unknown_profile_rejected_before_download_or_process_side_effects(tmp_path):
    from local_runtime import MacRuntime, RuntimeSetupError

    runtime = MacRuntime(tmp_path)
    before = runtime.settings
    for operation in (runtime.setup, runtime.start):
        with pytest.raises(RuntimeSetupError, match='Unknown'):
            await operation(profile_id='not-a-profile')
        assert runtime.settings is before
        assert runtime.processes == {}
        assert not list(tmp_path.iterdir())


def test_managed_runtime_rejects_forged_accuracy_artifact(tmp_path):
    from local_models import mac_accuracy_profile
    from local_runtime import RuntimeSetupError, server_command

    settings = resolve_models(mac_accuracy_profile())
    forged = replace(settings, chat=replace(settings.chat, model_revision='f' * 40))
    with pytest.raises(RuntimeSetupError, match='exact pinned'):
        server_command(forged, 'llm', tmp_path / 'llama-server', tmp_path)
