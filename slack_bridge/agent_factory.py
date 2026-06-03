from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from slack_bridge.agent_runner import AgentRunner, build_agent_options
from slack_bridge.allowed_tools import allowed_tools
from slack_bridge.config import BridgeConfig


class AgentCredentialError(RuntimeError):
    """Raised when the bridge process's credentials would break the cost model."""


def assert_agent_credentials(env: Mapping[str, str]) -> None:
    """Fail fast unless the bridge process is set up for $0 subscription billing.

    The agent (Claude Code via the SDK) reads ``ANTHROPIC_API_KEY`` from this
    process's environment; if present it switches off the Claude subscription
    onto metered API billing. So the bridge process must NOT carry it — the
    Ariadne-scoped Anthropic key lives under ``ARIADNE_ANTHROPIC_API_KEY`` (a
    name the agent ignores) and is mapped into Ariadne's subprocess by
    :func:`mcp_servers`.
    """
    if 'ANTHROPIC_API_KEY' in env:
        raise AgentCredentialError(
            'ANTHROPIC_API_KEY must not be set in the bridge process — it would '
            'switch the agent off the Claude subscription onto metered billing. '
            'Put the Ariadne-scoped key in ARIADNE_ANTHROPIC_API_KEY instead.'
        )
    if not env.get('CLAUDE_CODE_OAUTH_TOKEN'):
        raise AgentCredentialError(
            'CLAUDE_CODE_OAUTH_TOKEN is required (run `claude setup-token`) so the '
            'agent authenticates with the Claude subscription.'
        )


def mcp_servers(cfg: BridgeConfig, env: Mapping[str, str]) -> dict[str, Any]:
    """Build the Ariadne stdio MCP server config, secrets scoped to the subprocess.

    Launches the existing CLI server (``uv run --directory <ariadne_dir> ariadne
    mcp``) against the rename-proof repo dir. Secrets are read from ``env`` at
    runtime (never literals in a committed file) and passed only to this
    subprocess: ``OPENAI_API_KEY`` for embeddings, and ``ARIADNE_ANTHROPIC_API_KEY``
    re-exported as ``ANTHROPIC_API_KEY`` (the name Ariadne's ``ariadne_ask`` needs)
    — keeping ``ANTHROPIC_API_KEY`` out of the agent's own environment.
    """
    sub_env: dict[str, str] = {}
    if env.get('OPENAI_API_KEY'):
        sub_env['OPENAI_API_KEY'] = env['OPENAI_API_KEY']
    if env.get('ARIADNE_ANTHROPIC_API_KEY'):
        sub_env['ANTHROPIC_API_KEY'] = env['ARIADNE_ANTHROPIC_API_KEY']
    return {
        'ariadne': {
            'command': 'uv',
            'args': ['run', '--directory', str(cfg.ariadne_dir), 'ariadne', 'mcp'],
            'env': sub_env,
        }
    }


def make_runner_factory(
    cfg: BridgeConfig,
    system_prompt: str,
    env: Mapping[str, str] | None = None,
) -> Callable[[str, Any], AgentRunner]:
    """Build the pool's ``factory(thread_ts, seed)`` → :class:`AgentRunner`.

    Each call constructs a fresh ``ClaudeSDKClient`` (one conversation, its own
    Ariadne MCP subprocess) wrapped in an :class:`AgentRunner`. Options are fixed
    per process (system prompt, allowlist, MCP config, model).
    """
    from claude_agent_sdk import ClaudeSDKClient  # local import: heavy, only needed at runtime

    resolved_env = os.environ if env is None else env
    servers = mcp_servers(cfg, resolved_env)
    tools = allowed_tools(cfg.enable_feedback)

    def factory(thread_ts: str, seed: Any = None) -> AgentRunner:  # noqa: ARG001 — pool factory contract
        options = build_agent_options(
            system_prompt=system_prompt,
            allowed_tools=tools,
            mcp_servers=servers,
            model=cfg.model,
        )
        return AgentRunner(ClaudeSDKClient(options=options))

    return factory
