from __future__ import annotations

# Read-only Ariadne MCP tools the Slack bot may call. Names verified against the
# registered ``@mcp.tool`` functions in ``mcp_server*.py``. All are retrieval or
# synthesis only. ``ariadne_ask`` performs LLM synthesis and therefore needs an
# ANTHROPIC_API_KEY scoped to Ariadne's subprocess (see designs/slack-bridge.md,
# "Secrets handling"); the rest need only OPENAI_API_KEY (embeddings) or no key.
READ_ONLY_TOOLS: tuple[str, ...] = (
    'mcp__ariadne__ariadne_search',
    'mcp__ariadne__ariadne_read',
    'mcp__ariadne__ariadne_body',
    'mcp__ariadne__ariadne_symbol',
    'mcp__ariadne__ariadne_list_all',
    'mcp__ariadne__ariadne_themes',
    'mcp__ariadne__ariadne_expand',
    'mcp__ariadne__ariadne_explain',
    'mcp__ariadne__ariadne_summarize',
    'mcp__ariadne__ariadne_impact_radius',
    'mcp__ariadne__ariadne_coverage',
    'mcp__ariadne__ariadne_source_path',
    'mcp__ariadne__ariadne_config_usage',
    'mcp__ariadne__ariadne_ask',
)

# Opt-in: these WRITE to the ``usage_events`` table (hit/miss feedback). Off by
# default so the bot is purely read-only unless the operator enables it.
FEEDBACK_TOOLS: tuple[str, ...] = (
    'mcp__ariadne__ariadne_log_hit',
    'mcp__ariadne__ariadne_log_miss',
)


def allowed_tools(enable_feedback: bool = False) -> list[str]:
    """The agent's tool allowlist for ``permission_mode="dontAsk"``.

    Anything not returned here is denied silently by the SDK. Write/admin tools
    and the other LLM-backed tools (``ariadne_review``/``ariadne_task_context``/
    …) are intentionally excluded: the agent synthesises from retrieval, and
    ``ariadne_ask`` covers direct-question synthesis.
    """
    tools = list(READ_ONLY_TOOLS)
    if enable_feedback:
        tools.extend(FEEDBACK_TOOLS)
    return tools
