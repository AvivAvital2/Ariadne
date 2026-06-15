from __future__ import annotations

import logging

import pytest

import testimonials
from slack_bridge import app
from slack_bridge.agent_factory import AgentCredentialError
from slack_bridge.pool import SessionPool
from tests._slack_bridge_helpers import bridge_config


def test_build_pool_wires_a_bounded_session_pool():
    cfg = bridge_config(
        max_size=11,
        source_descriptions={'projecta': 'First project'},
        source_aliases={'projecta': ['proja']},
    )
    pool = app.build_pool(cfg)

    assert isinstance(pool, SessionPool)
    assert 'nothing-yet' not in pool   # fresh + usable


def test_main_refuses_to_start_when_anthropic_key_is_in_the_bridge_env(monkeypatch):
    monkeypatch.setenv('CLAUDE_CODE_OAUTH_TOKEN', 'oauth-x')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'anthropic-x')   # would flip the agent to metered

    with pytest.raises(AgentCredentialError):
        app.main([])   # bare serve path → hits the cost gate


def test_scan_subcommand_runs_the_backfill_past_the_agent_cost_gate(monkeypatch):
    # `scan` reads scores from the DB and never runs the agent, so it must NOT
    # invoke the serve-time credential gate; it must route to the backfill with
    # the parsed --limit.
    creds_checked: list[bool] = []
    monkeypatch.setattr(app, 'assert_agent_credentials',
                        lambda env: creds_checked.append(True))
    monkeypatch.setattr(app.BridgeConfig, 'from_env',
                        classmethod(lambda cls: bridge_config()))
    ran: dict = {}

    async def fake_run_scan(cfg, *, max_pairs=None, channels=None,
                            generate_scores=False, rescore=False):
        ran.update(max_pairs=max_pairs, channels=channels,
                   generate_scores=generate_scores, rescore=rescore)

    monkeypatch.setattr(app, '_run_scan', fake_run_scan)

    assert app._parse_args([]).command is None            # bare → serve
    assert app._parse_args(['scan']).rescore is False     # default off
    args = app._parse_args(['scan', '--limit', '5', '--channel', 'C1',
                            '--generate-scores', '--rescore'])
    assert (args.command, args.limit, args.channel, args.generate_scores, args.rescore) \
        == ('scan', 5, ['C1'], True, True)

    app.main(['scan', '--generate-scores', '--rescore'])
    assert ran == {'max_pairs': None, 'channels': None,
                   'generate_scores': True, 'rescore': True}
    assert creds_checked == []                            # main() doesn't gate scan (handled in _run_scan)


async def test_run_scan_opens_the_db_and_backfills(monkeypatch, tmp_path):
    captured: dict = {}

    async def fake_scan(slack, conn, *, store_dir, bot_user_id, max_pairs=None,
                        channels=None, score_fn=None, rescore=False):
        captured.update(store_dir=store_dir, bot_user_id=bot_user_id, max_pairs=max_pairs,
                        channels=channels, score_fn=score_fn, rescore=rescore, slack=slack)
        return 3

    monkeypatch.setattr('slack_bridge.scan.scan', fake_scan)

    class _FakeClient:
        async def auth_test(self):
            return {'user_id': 'UBOT'}

    monkeypatch.setattr(app, '_make_web_client', lambda token: _FakeClient())
    (tmp_path / 'ariadne.db').touch()

    n = await app._run_scan(bridge_config(ariadne_dir=tmp_path), max_pairs=7, channels=['C1'])
    assert n == 3
    assert captured['bot_user_id'] == 'UBOT'
    assert captured['max_pairs'] == 7
    assert captured['channels'] == ['C1']
    assert captured['score_fn'] is None                  # DB-only by default (no LLM)
    assert captured['rescore'] is False                  # delta by default (no re-judge)
    assert captured['store_dir'] == testimonials.local_dir(tmp_path)
    assert isinstance(captured['slack'], _FakeClient)


async def test_run_scan_wires_an_llm_scorer_when_generating(monkeypatch, tmp_path):
    pytest.importorskip('claude_agent_sdk')
    captured: dict = {}

    async def fake_scan(slack, conn, *, store_dir, bot_user_id, max_pairs=None,
                        channels=None, score_fn=None, rescore=False):
        captured['score_fn'] = score_fn
        return 0

    monkeypatch.setattr('slack_bridge.scan.scan', fake_scan)

    class _FakeClient:
        async def auth_test(self):
            return {'user_id': 'UBOT'}

    monkeypatch.setattr(app, '_make_web_client', lambda token: _FakeClient())
    gate: list[bool] = []
    monkeypatch.setattr(app, 'assert_agent_credentials', lambda env: gate.append(True))
    scored: list = []

    async def fake_llm_score(question, answer, *, model=None):
        scored.append((question, answer, model))
        return 5

    monkeypatch.setattr('slack_bridge.scoring.llm_score', fake_llm_score)
    (tmp_path / 'ariadne.db').touch()

    await app._run_scan(bridge_config(ariadne_dir=tmp_path, model='claude-x'),
                        generate_scores=True)
    assert gate == [True]                                 # LLM scoring → cost gate enforced
    assert captured['score_fn'] is not None
    assert await captured['score_fn']('q', 'a') == 5      # delegates to llm_score…
    assert scored == [('q', 'a', 'claude-x')]             # …with the configured model


def test_warns_when_allow_all_without_org_gate(caplog):
    """#9: allow_all with no allowed_orgs is open to Slack Connect — warn loudly."""
    with caplog.at_level(logging.WARNING, logger='slack_bridge.app'):
        app._warn_if_wide_open(bridge_config(allow_all=True))
    assert any('allowed_orgs' in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='slack_bridge.app'):
        app._warn_if_wide_open(bridge_config(allow_all=True, allowed_orgs=frozenset({'T0'})))  # gated
        app._warn_if_wide_open(bridge_config())                                                # not open
    assert caplog.records == []


def test_make_web_client_builds_a_real_async_client():
    pytest.importorskip('slack_sdk.web.async_client')
    from slack_sdk.web.async_client import AsyncWebClient

    assert isinstance(app._make_web_client('xoxb-test'), AsyncWebClient)


def test_make_app_and_register_listeners_exercise_the_real_bolt_api():
    pytest.importorskip('slack_bolt.async_app')
    from slack_bolt.async_app import AsyncApp

    cfg = bridge_config(source_descriptions={'projecta': 'First project'})
    bolt_app = app.make_app(cfg)
    assert isinstance(bolt_app, AsyncApp)

    # Registering must not raise — exercises app.event(...) / app.command(...).
    app.register_listeners(bolt_app, cfg, pool=object(), bot_user_id='UBOT')
