"""Increment 6: `ariadne spools enable/disable` — per-project cross-check.

`enable` adds a project to ``spools.<name>.projects`` and reconciles (clusters
the cross-source pass); `disable` removes it and deletes that partition.
Synthetic fixtures only (fake spool/project names, tmp store).
"""
import argparse
import os

import numpy as np

import config as config_module
from cli.spools_cmd import _disable, _enable, _reconcile
from config import Config
from docgen.cluster import _association_key
from library import Library
from spools import enabled_spools, spool_source_id


def _add(lib, doc_id, vec, source):
    v = np.array(vec, dtype=np.float32)
    v /= np.linalg.norm(v)
    lib.add_document(
        content_type='catalog', title=doc_id, content='x', source_files=[],
        embedding=v, metadata={'kind': 'element', 'source_name': source},
        doc_id=doc_id,
    )
    with lib._conn_provider.acquire() as c:
        c.execute('UPDATE documents SET source_name=? WHERE id=?', (source, doc_id))


def _cache_manifest(cfg_dir, name, runtime='rt'):
    """Simulate an installed pack: a cached manifest at the default cache path
    (<cfg dir>/.ariadne/spools/<name>) so resolve_spools registers the spool. A
    real enable/reconcile follows `spools install`, which caches this."""
    d = cfg_dir / '.ariadne' / 'spools' / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'manifest.yaml').write_text(
        f'environment: {name}\nversion: "1.0.0"\n'
        f'target_runtime: {runtime}\nchecksum: abc123\n'
    )


def test_set_spool_projects_persists(tmp_path):
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text('spools:\n  databricks:\n    runtime: rt\n')
    cfg = Config(cfg_file)
    assert cfg.set_spool_projects('databricks', ['proj1', 'proj2'])
    assert enabled_spools(cfg)['databricks'].projects == ('proj1', 'proj2')
    # A fresh load sees the persisted list (and enablement is preserved).
    assert enabled_spools(Config(cfg_file))['databricks'].projects == (
        'proj1', 'proj2',
    )

    # A bare `true` entry is promoted to a settings mapping, staying enabled.
    cfg_file.write_text('spools:\n  bricks: true\n')
    cfg2 = Config(cfg_file)
    assert cfg2.set_spool_projects('bricks', ['p'])
    reloaded = enabled_spools(Config(cfg_file))
    assert 'bricks' in reloaded
    assert reloaded['bricks'].projects == ('p',)


def test_enable_then_disable_reconciles(tmp_path):
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text(
        'spools:\n  databricks:\n    runtime: rt\n    projects: []\n',
    )
    db = tmp_path / 'lib.db'
    sp = spool_source_id('databricks')
    vec = [1.0, 0.05, 0, 0, 0, 0, 0, 0]
    with Library(db) as lib:
        for i in range(3):
            _add(lib, f'p1_{i}', vec, 'proj1')      # project + spool docs
            _add(lib, f'sp_{i}', vec, sp)           # cluster cross-source
    _cache_manifest(tmp_path, 'databricks')          # installed (registered)

    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    os.environ['ARIADNE_CONFIG'] = str(cfg_file)
    config_module._global_config = Config(cfg_file)
    try:
        assoc = _association_key(frozenset({'proj1', sp}))
        ns = argparse.Namespace(spool='databricks', project='proj1', db=str(db))

        assert _enable(ns) == 0
        # Config now lists proj1, and a cross-source theme exists.
        assert 'proj1' in enabled_spools(
            config_module._global_config,
        )['databricks'].projects
        with Library(db) as lib:
            assert lib.list_themes(coherent_only=False, association=assoc)

        assert _disable(ns) == 0
        # proj1 dropped from config and its theme partition deleted.
        assert 'proj1' not in enabled_spools(
            config_module._global_config,
        )['databricks'].projects
        with Library(db) as lib:
            assert not lib.list_themes(coherent_only=False, association=assoc)
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env


def test_reconcile_command_wiring(tmp_path):
    # `spools reconcile` refreshes enabled spools; --spool targets one; an
    # unknown spool errors; no spools enabled is a clean no-op.
    db = tmp_path / 'lib.db'
    sp = spool_source_id('databricks')
    vec = [1.0, 0.05, 0, 0, 0, 0, 0, 0]
    with Library(db) as lib:
        for i in range(3):
            _add(lib, f'p1_{i}', vec, 'proj1')
            _add(lib, f'sp_{i}', vec, sp)
    _cache_manifest(tmp_path, 'databricks')          # installed (registered)
    _cache_manifest(tmp_path, 'lonely')

    empty_cfg = tmp_path / 'empty.yaml'
    empty_cfg.write_text('sources: {}\n')
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text(
        'spools:\n'
        '  databricks:\n    runtime: rt\n    projects: [proj1]\n'
        '  lonely:\n    runtime: rt\n    projects: []\n',
    )
    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    try:
        # No spools enabled -> clean no-op.
        os.environ['ARIADNE_CONFIG'] = str(empty_cfg)
        config_module._global_config = Config(empty_cfg)
        assert _reconcile(argparse.Namespace(spool=None, db=str(db))) == 0

        os.environ['ARIADNE_CONFIG'] = str(cfg_file)
        config_module._global_config = Config(cfg_file)
        assoc = _association_key(frozenset({'proj1', sp}))
        # Reconcile one spool by name -> creates the cross-source theme.
        assert _reconcile(argparse.Namespace(spool='databricks', db=str(db))) == 0
        with Library(db) as lib:
            assert lib.list_themes(coherent_only=False, association=assoc)
        # A spool with no projects reconciles to zero pending -> clean, no hint.
        assert _reconcile(argparse.Namespace(spool='lonely', db=str(db))) == 0
        # Unknown spool -> loud error.
        assert _reconcile(argparse.Namespace(spool='ghost', db=str(db))) == 1
        # Reconcile-all (default target) is a fine idempotent refresh.
        assert _reconcile(argparse.Namespace(spool=None, db=str(db))) == 0
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env


def test_enable_on_not_installed_spool_surfaces_gap(tmp_path, capsys):
    # MEDIUM-2: enabling a project against a pinned-but-NOT-installed spool must
    # surface the missing-pack gap (fail-loud), not silently reconcile to
    # nothing and report "No cross-source themes surfaced".
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text('spools:\n  databricks:\n    runtime: rt\n')
    db = tmp_path / 'lib.db'
    with Library(db):
        pass  # empty store; no pack installed / cached

    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    os.environ['ARIADNE_CONFIG'] = str(cfg_file)
    config_module._global_config = Config(cfg_file)
    try:
        rc = _enable(argparse.Namespace(
            spool='databricks', project='proj1', db=str(db), cache_dir=None))
        out = capsys.readouterr().out
        assert rc == 0
        assert 'fetch' in out.lower()                       # gap surfaced
        assert 'No cross-source themes surfaced' not in out  # not the silent path
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env


def test_reconcile_skips_not_installed_spool(tmp_path, capsys):
    # MEDIUM-2 (reconcile side): a not-installed enabled spool is skipped
    # loudly with the reason, not silently reconciled to nothing.
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text(
        'spools:\n  databricks:\n    runtime: rt\n    projects: [proj1]\n',
    )
    db = tmp_path / 'lib.db'
    with Library(db):
        pass  # no pack cached / installed

    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    os.environ['ARIADNE_CONFIG'] = str(cfg_file)
    config_module._global_config = Config(cfg_file)
    try:
        rc = _reconcile(argparse.Namespace(
            spool='databricks', db=str(db), cache_dir=None))
        out = capsys.readouterr().out
        assert rc == 0
        assert 'skipped' in out.lower()
        assert 'fetch' in out.lower()   # the missing-pack gap message
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env


def test_enable_disable_edge_paths(tmp_path):
    # Unknown spool errors loudly; enabling a project with no cross-source
    # overlap surfaces no themes; disabling one that produced no partition is
    # a clean no-op.
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text(
        'spools:\n  databricks:\n    runtime: rt\n    projects: []\n',
    )
    db = tmp_path / 'lib.db'
    sp = spool_source_id('databricks')
    with Library(db) as lib:
        for i in range(3):
            _add(lib, f'solo_{i}', [1.0, 0.05, 0, 0, 0, 0, 0, 0], 'solo')
            _add(lib, f'sp_{i}', [0, 0, 1.0, 0.05, 0, 0, 0, 0], sp)  # orthogonal

    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    os.environ['ARIADNE_CONFIG'] = str(cfg_file)
    config_module._global_config = Config(cfg_file)
    try:
        # Unknown spool -> loud error (non-zero), no crash.
        assert _enable(argparse.Namespace(
            spool='ghost', project='p', db=str(db))) == 1
        assert _disable(argparse.Namespace(
            spool='ghost', project='p', db=str(db))) == 1

        # Enable a project with no cross-source overlap -> no themes surfaced.
        assert _enable(argparse.Namespace(
            spool='databricks', project='solo', db=str(db))) == 0
        assoc = _association_key(frozenset({'solo', sp}))
        with Library(db) as lib:
            assert not lib.list_themes(coherent_only=False, association=assoc)

        # Disable a project that produced no partition -> nothing to remove.
        assert _disable(argparse.Namespace(
            spool='databricks', project='solo', db=str(db))) == 0
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env
