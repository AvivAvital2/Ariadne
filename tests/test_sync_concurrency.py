"""`-c/--concurrency` on `ariadne sync` makes theme-generation parallelism
user-tunable: the flag must drive the post-regen theme refresh (and doc
regeneration), instead of theme summarization being pinned to
``generate_themes``' hardcoded default.
"""
from __future__ import annotations

import argparse

from cli import sync as sync_mod
from cli.main import create_parser


def test_sync_parser_exposes_concurrency_flag() -> None:
    parser = create_parser()
    assert parser.parse_args(['sync']).concurrency == 3
    assert parser.parse_args(['sync', '-c', '9']).concurrency == 9
    assert parser.parse_args(['sync', '--concurrency', '6']).concurrency == 6


async def test_refresh_themes_after_regen_forwards_concurrency(monkeypatch) -> None:
    """The shared post-regen theme refresh forwards --concurrency into
    refresh_themes' summarize_kwargs (and preserves the themes_enabled gate),
    so themes summarize at the requested width."""
    captured: dict = {}

    async def fake_refresh(library, writer, **kwargs):
        captured.update(kwargs)
        return {
            'path': 'noop', 'summarized': 0, 'incoherent': 0,
            'failed': 0, 'total_dirty': 0, 'changed': 0,
        }

    class _FakeWriter:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr('docgen.themes.refresh_themes', fake_refresh)
    monkeypatch.setattr('writer.LibraryWriter', _FakeWriter)

    cfg = argparse.Namespace(themes_enabled=True)
    await sync_mod._refresh_themes_after_regen(object(), cfg, concurrency=9)

    assert captured['summarize_kwargs'] == {'concurrency': 9}
    assert captured['enabled'] is True
