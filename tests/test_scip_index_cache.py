"""Tests for ScipIndex caching in resolve_index.

Without caching, every Scala/Java file triggers a fresh
``ScipIndex.load`` — re-reading and re-parsing the full ``.scip``
artifact (often tens of MB). For a repo with thousands of files this
turns into hundreds of redundant index parses.

Cache invalidates on mtime change so a re-indexed `.scip` file isn't
silently served stale.
"""
from __future__ import annotations

import os
from pathlib import Path

from docgen.scip_config import SourceScipConfig


class TestResolveIndexCache:
    def test_repeated_calls_load_index_once(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import scip_config
        from docgen.scip_extractor import ScipIndex

        artifact = tmp_path / 'idx.scip'
        artifact.write_bytes(b'fake-content')

        load_calls = {'n': 0}

        def fake_load(path, *, repo, max_staleness_days):
            load_calls['n'] += 1
            return ScipIndex(documents=())

        monkeypatch.setattr(
            'docgen.scip_extractor.ScipIndex.load',
            staticmethod(fake_load),
        )
        # Drop any cache leftover from prior tests.
        scip_config._index_cache.clear()

        cfg = SourceScipConfig(
            repo='r', artifact_path=artifact,
            index_kinds={'scala': 'scip'},
        )
        for _ in range(5):
            scip_config.resolve_index(cfg, 'scala')

        assert load_calls['n'] == 1, (
            f"resolve_index re-parsed the index {load_calls['n']} times; "
            f"expected 1 (cached)"
        )

    def test_cache_invalidates_on_mtime_change(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import scip_config
        from docgen.scip_extractor import ScipIndex

        artifact = tmp_path / 'idx.scip'
        artifact.write_bytes(b'v1')

        load_calls = {'n': 0}

        def fake_load(path, *, repo, max_staleness_days):
            load_calls['n'] += 1
            return ScipIndex(documents=())

        monkeypatch.setattr(
            'docgen.scip_extractor.ScipIndex.load',
            staticmethod(fake_load),
        )
        scip_config._index_cache.clear()

        cfg = SourceScipConfig(
            repo='r', artifact_path=artifact,
            index_kinds={'scala': 'scip'},
        )
        scip_config.resolve_index(cfg, 'scala')

        # Bump mtime by 2s and rewrite — cache must invalidate.
        new_mtime = artifact.stat().st_mtime + 2
        os.utime(artifact, (new_mtime, new_mtime))
        artifact.write_bytes(b'v2')
        # write_bytes also updates mtime — ensure we're past the previous one.
        os.utime(artifact, (new_mtime, new_mtime))

        scip_config.resolve_index(cfg, 'scala')

        assert load_calls['n'] == 2, (
            f"cache must invalidate on mtime change; "
            f"load called {load_calls['n']} times"
        )

    def test_different_artifacts_dont_share_cache(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import scip_config
        from docgen.scip_extractor import ScipIndex

        a = tmp_path / 'a.scip'
        b = tmp_path / 'b.scip'
        a.write_bytes(b'a')
        b.write_bytes(b'b')

        load_calls = {'n': 0}

        def fake_load(path, *, repo, max_staleness_days):
            load_calls['n'] += 1
            return ScipIndex(documents=())

        monkeypatch.setattr(
            'docgen.scip_extractor.ScipIndex.load',
            staticmethod(fake_load),
        )
        scip_config._index_cache.clear()

        cfg_a = SourceScipConfig(
            repo='r', artifact_path=a, index_kinds={'scala': 'scip'},
        )
        cfg_b = SourceScipConfig(
            repo='r', artifact_path=b, index_kinds={'scala': 'scip'},
        )
        scip_config.resolve_index(cfg_a, 'scala')
        scip_config.resolve_index(cfg_b, 'scala')
        scip_config.resolve_index(cfg_a, 'scala')  # should hit cache
        scip_config.resolve_index(cfg_b, 'scala')  # should hit cache

        assert load_calls['n'] == 2, (
            f"two different artifacts should each load once; "
            f"got {load_calls['n']} loads"
        )
