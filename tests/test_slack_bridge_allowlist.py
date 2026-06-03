from __future__ import annotations

from slack_bridge.allowed_tools import allowed_tools


def test_allowed_tools_read_only_and_feedback_toggle():
    base = allowed_tools(enable_feedback=False)
    with_feedback = allowed_tools(enable_feedback=True)

    # Core read-only retrieval tools (incl. ariadne_ask synthesis) are present.
    for name in (
        'mcp__ariadne__ariadne_search',
        'mcp__ariadne__ariadne_read',
        'mcp__ariadne__ariadne_ask',
        'mcp__ariadne__ariadne_explain',
        'mcp__ariadne__ariadne_impact_radius',
    ):
        assert name in base

    # Feedback tools write usage_events — opt-in only.
    assert 'mcp__ariadne__ariadne_log_hit' not in base
    assert 'mcp__ariadne__ariadne_log_miss' not in base
    assert 'mcp__ariadne__ariadne_log_hit' in with_feedback
    assert 'mcp__ariadne__ariadne_log_miss' in with_feedback

    # Write/admin and the OFF LLM-backed tools are never exposed, in either mode.
    for name in (
        'mcp__ariadne__ariadne_generate',
        'mcp__ariadne__ariadne_contribute',
        'mcp__ariadne__ariadne_notify_changed',
        'mcp__ariadne__ariadne_review',
        'mcp__ariadne__ariadne_task_context',
    ):
        assert name not in base
        assert name not in with_feedback

    # Every entry is a namespaced ariadne MCP tool.
    assert all(t.startswith('mcp__ariadne__') for t in with_feedback)


def test_allowlist_names_exist_in_mcp_server_sources():
    """Drift guard: each allow-listed tool must be a real Ariadne MCP tool.

    Under permission_mode='dontAsk', a name that doesn't match a registered tool
    is silently uncallable. We parse the ariadne_mcp/server*.py sources for tool defs
    (importing them would trigger FastMCP's .env settings, which the sandbox
    blocks) and assert every allow-listed bare name is defined.
    """
    import re
    from pathlib import Path

    from slack_bridge.allowed_tools import FEEDBACK_TOOLS, READ_ONLY_TOOLS

    repo = Path(__file__).resolve().parents[1]
    defined: set[str] = set()
    for src in repo.glob('ariadne_mcp/server*.py'):
        for match in re.finditer(r'^(?:async )?def (ariadne_\w+)', src.read_text(encoding='utf-8'), re.MULTILINE):
            defined.add(match.group(1))

    for tool in (*READ_ONLY_TOOLS, *FEEDBACK_TOOLS):
        bare = tool.removeprefix('mcp__ariadne__')
        assert bare in defined, f'{bare} is allow-listed but not defined in ariadne_mcp/server*.py'
