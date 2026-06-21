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


def test_scoring_directive_is_added_only_when_feedback_is_enabled():
    roster = [SourceEntry('projecta', 'First example project', ())]

    off = render_system_prompt(roster)                       # default: feedback off
    on = render_system_prompt(roster, enable_feedback=True)

    # Off: no instruction to log feedback or emit a score.
    assert 'score:' not in off.lower() and 'log_hit' not in off.lower()
    # On: the agent is told to log a hit/miss AND begin it with a 1-10 score:N,
    # which is what populates quality_score (and thus testimonials).
    low = on.lower()
    assert 'ariadne_log_hit' in low
    assert 'score:' in low
    assert '1' in on and '10' in on
    # The score belongs in the tool feedback only — the agent must be told NOT
    # to echo it into the user-facing reply (the deterministic strip is the
    # backstop, but the directive keeps the model from emitting it at all).
    assert 'not include the score in your reply' in low
