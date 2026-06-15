"""Tests for the LLM backfill scorer — rates a past answer 1-10 via Claude.

The SDK `query` is stubbed, so these never call a model; they pin the contract:
parse the integer, clamp to 1-10, honor the configured model, and tolerate
mixed message/block shapes.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip('claude_agent_sdk')
from slack_bridge import scoring


async def test_llm_score_parses_the_rating(monkeypatch):
    async def fake_query(*, prompt, options):
        # a list-content message (a text block + a non-text block), then a
        # terminal message whose content isn't a list — both must be tolerated.
        yield types.SimpleNamespace(content=[types.SimpleNamespace(text='8'),
                                             types.SimpleNamespace(tool='x')])
        yield types.SimpleNamespace(content=None)

    monkeypatch.setattr('slack_bridge.scoring.query', fake_query)
    assert await scoring.llm_score('How does caching work?', 'It uses an LRU.') == 8


async def test_llm_score_clamps_and_honors_model(monkeypatch):
    seen = {}

    async def fake_query(*, prompt, options):
        seen['model'] = options.model
        yield types.SimpleNamespace(content=[types.SimpleNamespace(text='99')])

    monkeypatch.setattr('slack_bridge.scoring.query', fake_query)
    assert await scoring.llm_score('q', 'a', model='claude-x') == 10   # clamped into range
    assert seen['model'] == 'claude-x'


async def test_llm_score_returns_none_without_a_number(monkeypatch):
    async def fake_query(*, prompt, options):
        yield types.SimpleNamespace(content=[types.SimpleNamespace(text='no idea')])

    monkeypatch.setattr('slack_bridge.scoring.query', fake_query)
    assert await scoring.llm_score('q', 'a') is None
