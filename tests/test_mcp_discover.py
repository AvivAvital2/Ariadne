"""Contract test for the ``ariadne_discover`` MCP tool — the onboarding
"Discover" step (language histogram + SCIP index plan + manifest write).

A single evolving test:

  D1 — discovering a source returns a positive file count and a language
       histogram that includes the languages actually present.
  D2 — the SCIP index plan detects the python package, and the
       ``.ariadne/manifest.json`` is written to disk.
  D3 — a TypeScript package is detected too, and the catalog SCIP routing
       (``index_kinds``) picks up ``javascript``.
  D4 — each detected package carries its ACTUAL name, read from the marker:
       the npm ``name`` from package.json, the sbt ``name :=`` from a root
       build.sbt (NOT the Ariadne source name), and the package dir for
       python. The root JVM project is never labelled by the source name.

Drives the registered tool through the in-process service path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ariadne_mcp.server_admin import ariadne_discover, ariadne_source_add


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'ariadne.yaml'
    cfg_path.write_text('sources: {}\n')
    monkeypatch.setenv('ARIADNE_CONFIG', str(cfg_path))
    monkeypatch.chdir(tmp_path)

    import config as config_module
    monkeypatch.setattr(config_module, '_global_config', None, raising=False)

    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, '_instance', None, raising=False)
    return cfg_path


def _make_source(root: Path) -> None:
    """A python package + a typescript package + a root sbt project + a doc.

    The npm name and the sbt name are deliberately DIFFERENT from their
    directory and from the Ariadne source name, so the test proves the name
    is *read from the marker* rather than guessed from the path/source.
    """
    (root / 'pymod').mkdir(parents=True)
    (root / 'pymod' / '__init__.py').write_text('')
    (root / 'pymod' / 'core.py').write_text('def go():\n    return 1\n')
    (root / 'webapp').mkdir()
    (root / 'webapp' / 'package.json').write_text('{"name": "@acme/webclient"}\n')
    (root / 'webapp' / 'app.ts').write_text('export const f = (): number => 3;\n')
    # Root JVM project — its real name lives in build.sbt, not the folder name.
    (root / 'build.sbt').write_text('name := "coolproj"\nscalaVersion := "2.13.12"\n')
    (root / 'Main.scala').write_text('package com.acme.coolproj\nobject Main\n')
    (root / 'README.md').write_text('# Project\n')


async def test_discover_tool_evolves_through_contract(monkeypatch, tmp_path):
    src = tmp_path / 'proj'
    src.mkdir()
    _make_source(src)
    await ariadne_source_add('proj', path=str(src))

    res = await ariadne_discover('proj')

    # ---- D1: file count + language histogram -------------------------
    assert res.source == 'proj'
    assert res.file_count >= 3
    assert res.dir_count >= 1
    langs = {lc.language for lc in res.languages}
    assert 'python' in langs
    assert abs(sum(lc.percent for lc in res.languages) - 100.0) < 1.0

    # ---- D2: python indexer detected + manifest written to disk ------
    kinds = {ix.kind for ix in res.indexers}
    assert 'python' in kinds
    assert res.manifest_written is True
    manifest_path = src / '.ariadne' / 'manifest.json'
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source_name'] == 'proj'
    assert any(ix['kind'] == 'python' for ix in manifest['indexers'])

    # ---- D3: typescript detected → index_kinds routes javascript -----
    assert 'typescript' in kinds
    assert 'javascript' in res.index_kinds

    # ---- D4: each package carries its ACTUAL name, read from the marker ----
    names_by_kind: dict[str, set[str]] = {}
    for ix in res.indexers:
        names_by_kind.setdefault(ix.kind, set()).update(ix.names)
    # npm name from package.json (not the 'webapp' folder)
    assert '@acme/webclient' in names_by_kind.get('typescript', set())
    # sbt name from a ROOT build.sbt — never the source name ('proj')
    jvm = names_by_kind.get('java', set())
    assert 'coolproj' in jvm
    assert 'proj' not in jvm
    # python package keeps its package path/dir
    assert any('pymod' in n for n in names_by_kind.get('python', set()))
    # names are aligned 1:1 with markers
    for ix in res.indexers:
        assert len(ix.names) == len(ix.markers)
    # persisted to the manifest too
    assert any('coolproj' in (ix.get('names') or [])
               for ix in manifest['indexers'] if ix['kind'] == 'java')
