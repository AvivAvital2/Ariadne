from __future__ import annotations

import pytest

from slack_bridge.agent_factory import (
    AgentCredentialError,
    assert_agent_credentials,
    mcp_servers,
)
from tests._slack_bridge_helpers import bridge_config


def test_assert_agent_credentials_requires_oauth_and_forbids_anthropic_key():
    oauth_val = 'oauth-x'
    scoped_val = 'anthropic-x'

    # OK: subscription token present, no ANTHROPIC_API_KEY in the bridge env.
    assert_agent_credentials({'CLAUDE_CODE_OAUTH_TOKEN': oauth_val})
    # OK: the Ariadne-scoped key uses a different name the agent ignores.
    assert_agent_credentials(
        {'CLAUDE_CODE_OAUTH_TOKEN': oauth_val, 'ARIADNE_ANTHROPIC_API_KEY': scoped_val}
    )

    # Missing subscription token → refuse to start.
    with pytest.raises(AgentCredentialError):
        assert_agent_credentials({})
    # ANTHROPIC_API_KEY in the bridge env would flip the agent to metered billing.
    with pytest.raises(AgentCredentialError):
        assert_agent_credentials({'CLAUDE_CODE_OAUTH_TOKEN': oauth_val, 'ANTHROPIC_API_KEY': scoped_val})


def test_mcp_servers_scopes_secrets_to_the_ariadne_subprocess():
    cfg = bridge_config()
    openai_val = 'openai-x'
    anthropic_val = 'anthropic-x'
    env = {'OPENAI_API_KEY': openai_val, 'ARIADNE_ANTHROPIC_API_KEY': anthropic_val}

    ariadne = mcp_servers(cfg, env)['ariadne']

    # Launches the existing CLI MCP server, pinned to the (rename-proof) repo dir.
    assert ariadne['command'] == 'uv'
    assert ariadne['args'][:3] == ['run', '--directory', str(cfg.ariadne_dir)]
    assert ariadne['args'][-1] == 'mcp'

    # Secrets are scoped to the subprocess env, with the Ariadne-scoped Anthropic
    # key mapped to the name the API expects.
    assert ariadne['env']['OPENAI_API_KEY'] == openai_val
    assert ariadne['env']['ANTHROPIC_API_KEY'] == anthropic_val
    # The agent's subscription token never leaks into Ariadne's subprocess.
    assert 'CLAUDE_CODE_OAUTH_TOKEN' not in ariadne['env']
