"""Consistency guard: the Slack app manifest must declare every scope (and event
subscription) the bridge actually needs.

This is what keeps the manual cross-check in the deploy guide from rotting — a
new Slack API call, or a dropped scope/event, fails here instead of silently
breaking the deployed bot (or, for the org gate, fail-closing every DM).
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / 'slack-app-manifest.yaml'

# Slack Web API method (as called on the client) → the bot scope(s) it requires.
# Add a new Slack call here AND to the manifest, or `test_every_call_is_mapped` fails.
_METHOD_SCOPES: dict[str, set[str]] = {
    'auth_test': set(),                         # no scope
    'chat_getPermalink': set(),                 # no scope
    'chat_postMessage': {'chat:write'},
    'chat_update': {'chat:write'},
    'conversations_history': {'channels:history', 'groups:history', 'im:history', 'mpim:history'},
    'conversations_replies': {'channels:history', 'groups:history', 'im:history', 'mpim:history'},
    'conversations_list': {'channels:read'},
    'conversations_info': {'channels:read', 'groups:read', 'im:read', 'mpim:read'},
    'files_upload_v2': {'files:write'},
    'users_info': {'users:read'},

}

# Required by things that aren't a Web API client call:
#   files:read                  — url_private image/diagram downloads (aiohttp, bot token)
#   app_mentions:read, commands — receiving @mentions / the /ariadne slash command
_NON_CALL_SCOPES = {'files:read', 'app_mentions:read', 'commands'}

# Events the bridge registers handlers for (app.py register_listeners) — the
# manifest must subscribe to these or the bot never receives them.
_HANDLED_EVENTS = {'app_mention', 'message.im', 'message.channels', 'message.groups', 'message.mpim'}


def _called_methods() -> set[str]:
    src = '\n'.join(p.read_text(encoding='utf-8') for p in (_ROOT / 'slack_bridge').glob('*.py'))
    # calls on the Slack client (slack / client / app.client); _client is the
    # Claude SDK client and is excluded by the word boundary.
    return set(re.findall(r'\b(?:slack|client|app\.client)\.([a-z][a-zA-Z0-9_]*)\(', src))


def _manifest() -> dict:
    return yaml.safe_load(_MANIFEST.read_text(encoding='utf-8'))


def test_every_slack_call_is_mapped_to_a_scope():
    unmapped = _called_methods() - set(_METHOD_SCOPES)
    assert not unmapped, (
        f'Slack methods called with no scope mapping: {sorted(unmapped)} — '
        'add them to _METHOD_SCOPES and grant the scope in the manifest.')


def test_manifest_grants_every_scope_the_code_requires():
    required = set(_NON_CALL_SCOPES)
    for method in _called_methods():
        required |= _METHOD_SCOPES.get(method, set())
    missing = required - set(_manifest()['oauth_config']['scopes']['bot'])
    assert not missing, f'manifest is missing scopes the code needs: {sorted(missing)}'


def test_manifest_subscribes_to_every_handled_event():
    events = set(_manifest()['settings']['event_subscriptions']['bot_events'])
    missing = _HANDLED_EVENTS - events
    assert not missing, f'manifest is missing event subscriptions: {sorted(missing)}'
