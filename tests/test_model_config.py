from copy import deepcopy

import pytest
import yaml

from config import Config
from model_config import ModelConfigurationError, resolve_models, resolve_paths, tokenize_url
from tests.local_model_server import local_config


def test_local_yaml_paths_identity_and_operational_changes(tmp_path):
    data = local_config()
    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump(data))
    cfg = Config(path)
    settings = cfg.model_settings(execution=True)
    assert settings.local and settings.chat.max_concurrency == 1
    assert resolve_paths(cfg, local=True) == (
        tmp_path / 'state/local.db',
        tmp_path / 'state/local_staleness.db',
    )
    assert (
        resolve_paths(cfg, db=tmp_path / 'new.db', local=True)[1] == tmp_path / 'new_staleness.db'
    )
    changed = deepcopy(data)
    changed['llm'].update(base_url='http://localhost:9000/prefix/v1/', api_key_env='PRIVATE_CHAT')
    changed['embeddings']['api_key_env'] = 'PRIVATE_EMBED'
    assert resolve_models(changed).profile_hash == settings.profile_hash
    changed['llm']['model_revision'] = 'c' * 40
    assert resolve_models(changed).profile_hash != settings.profile_hash


def test_alias_precedence_and_explicit_conflicts():
    assert (
        resolve_models({'defaults': {'model': 'gpt-old'}, 'model': 'gpt-new'}).chat.model
        == 'gpt-new'
    )
    assert (
        resolve_models({'llm': {'provider': 'anthropic', 'model': 'claude-model'}}).chat.provider
        == 'anthropic'
    )
    with pytest.raises(ModelConfigurationError, match='conflicts'):
        resolve_models({'llm': {'model': 'gpt-new'}, 'defaults': {'model': 'gpt-old'}})
    assert resolve_models(local_config()).chat.model.startswith('Qwen/')


@pytest.mark.parametrize(
    ('section', 'key', 'value'),
    [
        ('llm', 'base_url', None),
        ('llm', 'max_concurrency', 3),
        ('embeddings', 'dimensions', 4096),
        ('embeddings', 'recipe', 'unknown'),
        ('embeddings', 'provider', 'openai'),
        ('llm', 'request_options', {'model': 'evil'}),
        ('llm', 'model_revision', 'REPLACE_WITH_PINNED_REVISION'),
    ],
)
def test_reject_invalid_local_settings(section, key, value):
    data = local_config()
    data[section][key] = value
    with pytest.raises(ModelConfigurationError):
        resolve_models(data, execution=True)


def test_url_prefix_and_credentials(monkeypatch):
    assert (
        tokenize_url('http://localhost:8000/prefix/v1/') == 'http://localhost:8000/prefix/tokenize'
    )
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-used')
    settings = resolve_models(local_config())
    assert settings.chat.credential() is None
    data = local_config()
    data['embeddings']['api_key_env'] = 'MISSING_LOCAL_KEY'
    monkeypatch.delenv('MISSING_LOCAL_KEY', raising=False)
    with pytest.raises(ModelConfigurationError, match='MISSING_LOCAL_KEY'):
        resolve_models(data).check_credentials()
