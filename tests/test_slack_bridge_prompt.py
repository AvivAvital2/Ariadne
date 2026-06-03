from __future__ import annotations

from slack_bridge.prompt import render_system_prompt
from slack_bridge.roster import SourceEntry


def test_render_system_prompt_includes_roster_posture_and_rules():
    roster = [
        SourceEntry('projecta', 'First example project', ('proja', 'project alpha')),
        SourceEntry('projectb', None, ('projb',)),
    ]
    prompt = render_system_prompt(roster)

    # Roster: names, descriptions, and aliases are all surfaced so the agent
    # can route (including a source with no description).
    assert 'projecta' in prompt
    assert 'First example project' in prompt
    assert 'project alpha' in prompt
    assert 'projectb' in prompt
    assert 'projb' in prompt

    low = prompt.lower()
    # Read-only posture.
    assert 'read-only' in low or 'read only' in low
    # Source routing + clarify-back: must require a source and forbid guessing.
    assert 'source' in low
    assert 'ask' in low
    # Tool guidance: ariadne_ask for direct questions.
    assert 'ariadne_ask' in low
    # Audience/altitude adaptation cue.
    assert '10k' in low or 'altitude' in low or 'audience' in low
