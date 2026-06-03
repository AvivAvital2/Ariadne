"""Regression: ``ariadne search --include-all`` must not crash.

With ``--include-all`` the scope block that binds ``source_name`` is
skipped, but the conflict-resolution guard later (`if results and
source_name`) references it unconditionally. Before the fix that raised
``UnboundLocalError`` on any query that returned at least one result.
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np

import cli.core as cli_core
from cli.core import cmd_search


class _FakeLib:
    """Minimal library whose search returns a non-empty result list, so
    the conflict-resolution guard at issue is actually reached."""

    def search(self, *args, **kwargs):
        return [object()]

    def close(self):
        pass


def test_search_include_all_does_not_crash(monkeypatch):
    monkeypatch.setattr(cli_core, 'get_library', lambda db=None: _FakeLib())
    monkeypatch.setattr('cli.core._print_search_results', lambda results: None)

    async def _embed(self, text):
        return np.zeros(8, dtype=np.float32)

    async def _aenter(self):
        return self

    async def _aexit(self, *exc):
        return False

    monkeypatch.setattr('embedding.EmbeddingService.embed', _embed)
    monkeypatch.setattr('embedding.EmbeddingService.__aenter__', _aenter)
    monkeypatch.setattr('embedding.EmbeddingService.__aexit__', _aexit)

    args = argparse.Namespace(
        query='anything', include_all=True, source=None,
        type=None, chunks=False, k=5, db=None,
    )
    assert asyncio.run(cmd_search(args)) == 0
