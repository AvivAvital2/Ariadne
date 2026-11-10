"""The Mac candidate is a distinct, reproducible local model pair."""

import pytest

from config import Config
from model_config import ModelConfigurationError, resolve_models


def test_mac_profile_loads_through_real_configuration(tmp_path):
    import yaml

    from local_models import mac_model_profile

    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump(mac_model_profile()))
    settings = Config(path).model_settings(execution=True)

    assert settings.local
    assert settings.chat.model == 'bartowski/Qwen_Qwen3.5-9B-GGUF'
    assert settings.chat.request_model == 'Qwen/Qwen3.5-9B'
    assert settings.embeddings.model == 'Qwen/Qwen3-Embedding-8B-GGUF'
    assert settings.embeddings.request_model == 'Qwen/Qwen3-Embedding-8B'
    assert settings.chat.weight_format == settings.embeddings.weight_format == 'gguf-q4_k_m'
    assert settings.chat.model_revision == '182be2fd6c7bc44887d88a91cb03ff009cc9f549'
    assert settings.embeddings.model_revision == '69d0e58a13e463cd99a9b83e3f5fee7c10265fab'
    assert settings.embeddings.dimensions == 3072
    assert settings.embeddings.recipe == 'qwen3-embedding-8b-gguf-q4-k-m-3072-v1'
    assert settings.chat.max_output_tokens == 8192
    assert settings.chat.request_options['chat_template_kwargs'] == {'enable_thinking': False}
    assert settings.profile_hash == resolve_models(mac_model_profile()).profile_hash


def test_download_manifest_identifies_exact_weights_and_embedded_tokenizers():
    from local_models import MAC_ARTIFACTS, mac_model_profile

    expected = {
        'llm': ('Qwen_Qwen3.5-9B-Q4_K_M.gguf', 6169341984,
                'd784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43'),
        'embeddings': ('Qwen3-Embedding-8B-Q4_K_M.gguf', 4676804928,
                       '3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85'),
    }
    settings = resolve_models(mac_model_profile(), execution=True)
    for role, artifact in MAC_ARTIFACTS.items():
        configured = settings.chat if role == 'llm' else settings.embeddings
        assert (artifact.filename, artifact.size, artifact.sha256) == expected[role]
        assert artifact.repository == configured.model
        assert artifact.revision == configured.model_revision
        assert artifact.url == (
            f'https://huggingface.co/{configured.model}/resolve/'
            f'{configured.model_revision}/{artifact.filename}'
        )
        identity = configured.identity
        tokenizer = (identity['tokenizer_revision'] if role == 'llm'
                     else identity['recipe']['tokenizer_revision'])
        assert tokenizer == artifact.revision


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('llm', 'model', 'Qwen/Qwen3.8-27B'),
    ('llm', 'weight_format', 'bf16'),
    ('embeddings', 'weight_format', 'gguf-q8_0'),
    ('embeddings', 'recipe', 'qwen3-embedding-8b-3072-v1'),
    ('embeddings', 'dimensions', 4096),
    ('embeddings', 'model_revision', 'main'),
    ('llm', 'model_revision', None),
    ('llm', 'request_options', {'chat_template_kwargs': {'enable_thinking': True}}),
])
def test_mac_profile_rejects_unmatched_contracts(section, key, value):
    from local_models import mac_model_profile

    data = mac_model_profile()
    data[section][key] = value
    with pytest.raises(ModelConfigurationError):
        resolve_models(data, execution=True)


def test_mac_profile_keeps_operational_settings_out_of_identity():
    from local_models import mac_model_profile

    original = resolve_models(mac_model_profile())
    changed = mac_model_profile()
    changed['llm'].update(base_url='http://localhost:9000/v1', served_model='alias')
    changed['embeddings']['timeout_seconds'] = 900
    assert resolve_models(changed).identity == original.identity
    changed['embeddings']['model_revision'] = 'c' * 40
    assert resolve_models(changed).embeddings.embedding_space_id != original.embeddings.embedding_space_id
    assert resolve_models(mac_model_profile()).identity == original.identity
