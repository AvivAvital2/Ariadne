"""LLM-powered gap analysis for Ariadne usage data.

Analyzes miss feedback to identify documentation gaps and recommend
improvements. Uses the OpenAI API following the same pattern as
docgen/generator.py.
"""
from __future__ import annotations

import json
import logging

from attrs import frozen

_logger = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 4096


@frozen
class GapRecommendation:
    """A single documentation gap recommendation."""

    theme: str
    miss_count: int
    description: str
    recommendation: str
    example_queries: tuple[str, ...]


@frozen
class GapReport:
    """LLM-generated gap analysis report."""

    total_misses: int
    analysis_period_days: int
    recommendations: tuple[GapRecommendation, ...]
    summary: str


_SYSTEM_PROMPT = '''\
You are Ariadne's gap analyst. You analyze feedback from documentation misses \
(cases where a user searched Ariadne's knowledge base but didn't find what \
they needed) and produce actionable recommendations.

You will receive a list of miss events, each with a query, tool name, and \
feedback describing what was missing. Your job is to:

1. Cluster similar feedback into themes
2. Prioritize by frequency and impact
3. Recommend specific documentation to create or improve
4. Suggest new capabilities if patterns indicate structural gaps

Respond with valid JSON matching this schema:
{
  "summary": "Executive summary of the gap analysis (2-3 sentences)",
  "recommendations": [
    {
      "theme": "Short theme name",
      "miss_count": <number of misses in this theme>,
      "description": "What documentation is missing or inadequate",
      "recommendation": "Specific action to take (create doc, update doc, add feature)",
      "example_queries": ["query1", "query2"]
    }
  ]
}

Order recommendations by miss_count descending. Limit to top 10.\
'''


async def analyze_gaps(
    misses: list[dict[str, str]],
    model: str | None = None,
) -> GapReport:
    """Analyze miss feedback using an LLM.

    Args:
        misses: List of miss event dicts with keys:
            timestamp, tool_name, query, feedback.
        model: LLM model to use (default: from config or gpt-4.1).

    Returns:
        Structured GapReport with recommendations.

    Raises:
        RuntimeError: If the LLM call fails.
    """
    from config import get_config

    if not misses:
        return GapReport(
            total_misses=0,
            analysis_period_days=0,
            recommendations=(),
            summary='No misses to analyze.',
        )

    from llm import chat_complete

    cfg = get_config()
    if model is None:
        model = cfg.model

    user_prompt = (
        f'Analyze these {len(misses)} documentation miss events:\n\n'
        + json.dumps(misses, indent=2)
    )

    content = await chat_complete(
        [
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=120.0,
    )

    # Parse JSON response
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code block
        import re
        match = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
        else:
            raise RuntimeError(f'Failed to parse LLM response as JSON: {content[:200]}')

    recommendations = tuple(
        GapRecommendation(
            theme=r.get('theme', 'Unknown'),
            miss_count=r.get('miss_count', 0),
            description=r.get('description', ''),
            recommendation=r.get('recommendation', ''),
            example_queries=tuple(r.get('example_queries', ())),
        )
        for r in parsed.get('recommendations', [])
    )

    return GapReport(
        total_misses=len(misses),
        analysis_period_days=30,
        recommendations=recommendations,
        summary=parsed.get('summary', 'No summary available.'),
    )
