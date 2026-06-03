from __future__ import annotations

from slack_bridge.roster import build_roster, resolve_alias


def test_build_roster_lists_all_sources_with_descriptions_and_aliases():
    names = ['ariadne', 'projecta', 'projectb']
    descriptions = {'ariadne': 'docs engine', 'projecta': 'first project'}
    aliases = {'projecta': ['proja', 'project alpha'], 'projectb': ['projb']}

    roster = build_roster(names, descriptions, aliases)
    by_name = {e.name: e for e in roster}

    # Every configured source is listed, even one with no description.
    assert set(by_name) == {'ariadne', 'projecta', 'projectb'}
    assert by_name['ariadne'].description == 'docs engine'
    assert by_name['projectb'].description is None
    assert 'project alpha' in by_name['projecta'].aliases
    assert by_name['ariadne'].aliases == ()

    # Alias resolution: a known alias or the canonical name (case-insensitive)
    # resolves; an unknown term does not.
    assert resolve_alias('project alpha', roster) == 'projecta'
    assert resolve_alias('PROJECTA', roster) == 'projecta'
    assert resolve_alias('projb', roster) == 'projectb'
    assert resolve_alias('nonsense', roster) is None
