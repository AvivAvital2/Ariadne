from __future__ import annotations

import pytest

from slack_bridge import app
from slack_bridge.agent_factory import AgentCredentialError
from slack_bridge.pool import SessionPool
from tests._slack_bridge_helpers import bridge_config


def test_build_pool_wires_a_bounded_session_pool():
    cfg = bridge_config(
        max_size=11,
        source_descriptions={'projecta': 'First project'},
        source_aliases={'projecta': ['proja']},
    )
    pool = app.build_pool(cfg)

    assert isinstance(pool, SessionPool)
    assert 'nothing-yet' not in pool   # fresh + usable


def test_main_refuses_to_start_when_anthropic_key_is_in_the_bridge_env(monkeypatch):
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', 'oauth-x')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-x')   # would flip the agent to metered

    with pytest.raises(AgentCredentialError):
        app.main()


def test_make_app_and_register_listeners_exercise_the_real_bolt_api():
    pytest.importorskip('slack_bolt.async_app')
    from slack_bolt.async_app import AsyncApp

    cfg = bridge_config(source_descriptions={'projecta': 'First project'})
    bolt_app = app.make_app(cfg)
    assert isinstance(bolt_app, AsyncApp)

    # Registering must not raise — exercises app.event(...) / app.command(...).
    app.register_listeners(bolt_app, cfg, pool=object(), bot_user_id='UBOT')
