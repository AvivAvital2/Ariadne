from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import testimonials
from slack_bridge.agent_factory import assert_agent_credentials, make_runner_factory
from slack_bridge.config import BridgeConfig
from slack_bridge.handlers import make_listeners
from slack_bridge.pool import SessionPool
from slack_bridge.prompt import render_system_prompt
from slack_bridge.roster import build_roster

_logger = logging.getLogger(__name__)
_SLASH_COMMAND = '/ariadne'


def build_pool(cfg: BridgeConfig) -> SessionPool:
    """Assemble the warm-session pool: roster → system prompt → runner factory."""
    source_names = sorted(set(cfg.source_descriptions) | set(cfg.source_aliases))
    roster = build_roster(source_names, cfg.source_descriptions, cfg.source_aliases)
    system_prompt = render_system_prompt(roster)
    return SessionPool(
        runner_factory=make_runner_factory(cfg, system_prompt),
        max_size=cfg.max_size,
        idle_ttl=cfg.idle_ttl_seconds,
        clock=time.monotonic,
    )


def make_app(cfg: BridgeConfig) -> Any:
    """Construct the Bolt ``AsyncApp`` (slack_bolt/aiohttp imported lazily here)."""
    from slack_bolt.async_app import AsyncApp

    return AsyncApp(token=cfg.slack_bot_token)


def register_listeners(app: Any, cfg: BridgeConfig, pool: Any, bot_user_id: str) -> None:
    """Wire the app_mention / message(DM) / slash-command listeners onto the app."""
    listeners = make_listeners(cfg, pool, bot_user_id)
    app.event('app_mention')(listeners['app_mention'])
    app.event('message')(listeners['message'])
    app.command(_SLASH_COMMAND)(listeners['command'])


async def _run(cfg: BridgeConfig) -> None:
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

    app = make_app(cfg)
    bot_user_id = (await app.client.auth_test())['user_id']
    pool = build_pool(cfg)
    register_listeners(app, cfg, pool, bot_user_id)

    async def _evict_idle_loop() -> None:
        interval = max(cfg.idle_ttl_seconds / 2, 30.0)
        while True:
            await asyncio.sleep(interval)
            await pool.evict_idle()

    evictor = asyncio.create_task(_evict_idle_loop())
    _logger.info('Ariadne Slack bridge starting (Socket Mode)…')
    try:
        await AsyncSocketModeHandler(app, cfg.slack_app_token).start_async()
    finally:
        evictor.cancel()


def _make_web_client(token: str) -> Any:
    """The Slack Web API client used by the scan (slack_sdk imported lazily)."""
    from slack_sdk.web.async_client import AsyncWebClient

    return AsyncWebClient(token=token)


async def _run_scan(cfg: BridgeConfig, *, max_pairs: int | None = None) -> int:
    """Backfill the local best-of store from the bot's public channel history.

    Reads the scores Ariadne already logged in ``ariadne.db`` and snapshots the
    top Q&A into the swap-proof store — no agent turn, so no LLM cost.
    """
    from slack_bridge.scan import scan

    client = _make_web_client(cfg.slack_bot_token)
    bot_user_id = (await client.auth_test())['user_id']
    store_dir = testimonials.local_dir(cfg.ariadne_dir)
    conn = sqlite3.connect(Path(cfg.ariadne_dir) / 'ariadne.db')
    try:
        recorded = await scan(
            client, conn, store_dir=store_dir,
            bot_user_id=bot_user_id, max_pairs=max_pairs)
    finally:
        conn.close()
    _logger.info('Backfilled %d testimonial(s) from public channels into %s',
                 recorded, store_dir)
    return recorded


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """``ariadne-slack`` (serve) or ``ariadne-slack scan [--limit N]`` (backfill)."""
    parser = argparse.ArgumentParser(
        prog='ariadne-slack',
        description='Run the Ariadne Slack bridge, or backfill its testimonials.')
    sub = parser.add_subparsers(dest='command')
    scan_p = sub.add_parser(
        'scan',
        help='Backfill the best-of store from public channel history (no agent run)')
    scan_p.add_argument(
        '--limit', type=int, default=None,
        help='Max past Q&A pairs to process, newest first (default: all)')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Console entry point (``ariadne-slack`` / ``python -m slack_bridge``)."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    if args.command == 'scan':
        # The backfill only reads scores from the DB — it never runs the agent,
        # so the serve-time cost gate (which forbids ANTHROPIC_API_KEY) doesn't
        # apply here.
        asyncio.run(_run_scan(BridgeConfig.from_env(), max_pairs=args.limit))
        return
    # Fail fast on the cost invariant before doing any work: the bridge process
    # must carry CLAUDE_CODE_OAUTH_TOKEN and must NOT carry ANTHROPIC_API_KEY.
    assert_agent_credentials(os.environ)
    asyncio.run(_run(BridgeConfig.from_env()))


if __name__ == '__main__':
    main()
