from __future__ import annotations

from pathlib import Path

from slack_bridge.config import BridgeConfig
from tests._slack_bridge_helpers import bridge_config


def test_allowlist_grants_by_user_or_channel_and_fails_closed():
    cfg = bridge_config(users=frozenset({'UALICE'}), channels=frozenset({'CENG'}))
    assert cfg.is_allowed(user='UALICE', channel='CRANDOM')    # allow-listed user
    assert cfg.is_allowed(user='UBOB', channel='CENG')         # allow-listed channel
    assert not cfg.is_allowed(user='UBOB', channel='CRANDOM')  # neither

    # Empty allowlists deny everyone (fail-closed).
    assert not bridge_config().is_allowed(user='UANY', channel='CANY')


def test_ariadne_dir_defaults_to_repo_root_not_a_hardcoded_name():
    cfg = bridge_config()
    # Derived from the package location at runtime — survives a directory rename.
    assert cfg.ariadne_dir == Path(__file__).resolve().parents[1]
    assert (cfg.ariadne_dir / 'slack_bridge' / '__init__.py').is_file()


def test_from_env_loads_tokens_from_env_and_rest_from_yaml(tmp_path, monkeypatch):
    cfg_file = tmp_path / 'slack_bridge.yaml'
    cfg_file.write_text(
        'allowed_users: [U1, U2]\n'
        'allowed_channels: [C1]\n'
        'pool: {max_size: 7, idle_ttl_seconds: 60, turn_timeout_seconds: 30, soft_timeout_seconds: 15}\n'
        'enable_feedback: true\n'
        'source_descriptions:\n  projecta: First project\n'
        'source_aliases:\n  projecta: [proja]\n'
    )
    bot_val = 'xoxb-test'
    app_val = 'xapp-test'
    monkeypatch.setenv('SLACK_BOT_TOKEN', bot_val)
    monkeypatch.setenv('SLACK_APP_TOKEN', app_val)
    monkeypatch.setenv('ARIADNE_SLACK_CONFIG', str(cfg_file))
    monkeypatch.delenv('ARIADNE_DIR', raising=False)

    cfg = BridgeConfig.from_env()

    # Secrets/tokens come from the environment.
    assert cfg.slack_bot_token == bot_val
    assert cfg.slack_app_token == app_val
    # Operational config comes from the yaml file.
    assert cfg.allowed_users == frozenset({'U1', 'U2'})
    assert cfg.allowed_channels == frozenset({'C1'})
    assert (cfg.max_size, cfg.idle_ttl_seconds, cfg.turn_timeout_seconds, cfg.soft_timeout_seconds) == (7, 60.0, 30.0, 15.0)
    assert cfg.enable_feedback is True
    assert cfg.source_descriptions['projecta'] == 'First project'
    assert cfg.source_aliases['projecta'] == ['proja']
    # ariadne_dir still defaults to the rename-proof repo root.
    assert (cfg.ariadne_dir / 'slack_bridge' / '__init__.py').is_file()
def test_allow_all_opens_the_bot_to_everyone():
    """`allow_all` is the org-wide switch: with it on, any user in any channel
    (and any DM) is allowed regardless of the (possibly empty) allowlists. The
    default stays fail-closed."""
    cfg = bridge_config(allow_all=True)
    assert cfg.is_allowed(user='UANY', channel='CANY')
    assert cfg.is_allowed(user='', channel='')
    # default (allow_all=False) is unchanged — fail-closed
    assert not bridge_config().is_allowed(user='UANY', channel='CANY')


def test_from_env_parses_allow_all(tmp_path, monkeypatch):
    """`allow_all: true` in the operational yaml flips the org-wide switch;
    absent, it defaults False (fail-closed)."""
    cfg_file = tmp_path / 'slack_bridge.yaml'
    cfg_file.write_text('allow_all: true\n')
    monkeypatch.setenv('ARIADNE_SLACK_CONFIG', str(cfg_file))
    monkeypatch.delenv('ARIADNE_DIR', raising=False)
    cfg = BridgeConfig.from_env()
    assert cfg.allow_all is True
    assert cfg.is_allowed(user='UANY', channel='CANY')

    empty = tmp_path / 'empty.yaml'
    empty.write_text('allowed_users: []\n')
    monkeypatch.setenv('ARIADNE_SLACK_CONFIG', str(empty))
    assert BridgeConfig.from_env().allow_all is False
