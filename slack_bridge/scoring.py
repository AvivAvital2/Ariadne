"""LLM backfill scorer — rate a past answer 1-10 via Claude (subscription).

The bridge can't read a historical answer's quality from the DB if that turn was
never scored, so the channel backfill (``ariadne-slack scan --generate-scores``)
calls this to judge each unscored Q&A. It runs on the subscription through the
Claude Agent SDK (no ``ANTHROPIC_API_KEY``), like every other agent call here.
"""
from __future__ import annotations

import re

from claude_agent_sdk import ClaudeAgentOptions, query

_SYSTEM = (
    'You score how well a chatbot answer addressed its question, for a "best of" '
    'showcase. Weigh accuracy, completeness, and helpfulness. Reply with ONLY a '
    'single integer from 1 (unhelpful or wrong) to 10 (excellent) — no other text.'
)


async def llm_score(question: str, answer: str, *, model: str | None = None) -> int | None:
    """Rate ``answer`` to ``question`` from 1-10 via Claude. None if none parses."""
    kwargs: dict = {
        'system_prompt': _SYSTEM,
        'allowed_tools': [],
        'permission_mode': 'dontAsk',
        'max_turns': 1,
    }
    if model:
        kwargs['model'] = model
    options = ClaudeAgentOptions(**kwargs)
    prompt = f'Question:\n{question}\n\nAnswer:\n{answer}\n\nScore (1-10):'

    parts: list[str] = []
    async for msg in query(prompt=prompt, options=options):
        content = getattr(msg, 'content', None)
        if isinstance(content, list):
            for block in content:
                text = getattr(block, 'text', None)
                if isinstance(text, str):
                    parts.append(text)

    m = re.search(r'\d+', ''.join(parts))
    if not m:
        return None
    return max(1, min(10, int(m.group())))


__all__ = ['llm_score']
