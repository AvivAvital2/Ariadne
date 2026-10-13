"""CLI wiring tests for ``ariadne export-db`` / ``import-db``.

Drives the command handlers in-process (an ``argparse.Namespace`` + a patched
``get_config``) so the tests are hermetic — no subprocess, no real config file,
no network. ``cli.core.get_config`` is patched to a controllable fake.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

import cli.core as core
from library import Library
from schema import Chunk, Section

EMBED_DIM = 16


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


class _FakeCfg:
    def __init__(self, default_source=None, db_path='ariadne.db'):
        self.default_source = default_source
        self.db_path = db_path


@pytest.fixture(autouse=True)
def _no_token_count(monkeypatch):
    monkeypatch.setattr('library.core.CoreMixin._count_tokens', lambda self, content: None)


@pytest.fixture
def cfg(monkeypatch):
    """Patch cli.core.get_config; tests mutate the returned fake as needed."""
    fake = _FakeCfg()
    monkeypatch.setattr(core, 'get_config', lambda: fake)
    return fake


@pytest.fixture
def full_db(tmp_path):
    path = tmp_path / 'full.db'
    lib = Library(path)
    sdk_ids = []
    for i, (ct, title) in enumerate([
        ('explanation', 'SDK Auth'), ('qa', 'How to auth?'),
        ('architecture', 'SDK Client'), ('qa', 'How to retry?'),
    ]):
        sdk_ids.append(lib.add_document(
            content_type=ct, title=title, content=f'{title} body',
            source_files=['sdk/x.py'], embedding=_vec(i + 1), source_name='sdk',
        ).id)
    lib.add_chunks_batch([Chunk(document_id=sdk_ids[0], chunk_index=0, content='c', embedding=_vec(99))])
    lib.store_sections(sdk_ids[0], [Section(document_id=sdk_ids[0], index=0, heading='H',
                                            description='d', content='## H', embedding=_vec(98))])
    lib.add_document(content_type='qa', title='Noise', content='noise body',
                     source_files=['z/z.py'], embedding=_vec(500), source_name='noise')
    lib.close()
    return {'path': str(path), 'sdk_ids': sdk_ids, 'count': len(sdk_ids)}


def _ns(**kw):
    return argparse.Namespace(**kw)


# --------------------------------------------------------------------------
# export-db handler
# --------------------------------------------------------------------------

def test_export_db_creates_bundle(full_db, tmp_path, cfg):
    out = tmp_path / 'slice.db'
    rc = core.cmd_export_db(_ns(db=Path(full_db['path']), source='sdk', out=str(out), no_embeddings=False))
    assert rc == 0
    assert out.exists()
    sl = Library(out)
    try:
        ids = {d.id for d in sl.list_documents()}
    finally:
        sl.close()
    assert ids == set(full_db['sdk_ids'])


def test_export_db_falls_back_to_default_source(full_db, tmp_path, cfg):
    cfg.default_source = 'sdk'  # exercises the `args.source or cfg.default_source` else-arm
    out = tmp_path / 'slice.db'
    rc = core.cmd_export_db(_ns(db=Path(full_db['path']), source=None, out=str(out), no_embeddings=False))
    assert rc == 0
    sl = Library(out)
    try:
        assert len(sl.list_documents()) == full_db['count']
    finally:
        sl.close()


def test_export_db_errors_when_no_source_and_no_default(full_db, tmp_path, cfg):
    out = tmp_path / 'slice.db'
    rc = core.cmd_export_db(_ns(db=Path(full_db['path']), source=None, out=str(out), no_embeddings=False))
    assert rc == 1
    assert not out.exists()


def test_export_db_uses_config_db_when_db_arg_missing(full_db, tmp_path, cfg):
    cfg.db_path = full_db['path']  # exercises the `args.db or cfg.db_path` else-arm
    out = tmp_path / 'slice.db'
    rc = core.cmd_export_db(_ns(db=None, source='sdk', out=str(out), no_embeddings=False))
    assert rc == 0
    assert out.exists()


# --------------------------------------------------------------------------
# import-db handler
# --------------------------------------------------------------------------

def test_import_db_merges_into_target(full_db, tmp_path, cfg):
    bundle = tmp_path / 'bundle.db'
    core.cmd_export_db(_ns(db=Path(full_db['path']), source='sdk', out=str(bundle), no_embeddings=False))
    target = tmp_path / 'target.db'
    Library(target).close()

    rc = core.cmd_import_db(_ns(db=target, bundle=str(bundle),
                               on_conflict='replace', embedding_model='text-embedding-3-large'))
    assert rc == 0
    tgt = Library(target)
    try:
        n = len([d for d in tgt.list_documents() if d.source_name == 'sdk'])
    finally:
        tgt.close()
    assert n == full_db['count']


def test_import_db_uses_config_db_when_db_arg_missing(full_db, tmp_path, cfg):
    bundle = tmp_path / 'bundle.db'
    core.cmd_export_db(_ns(db=Path(full_db['path']), source='sdk', out=str(bundle), no_embeddings=False))
    target = tmp_path / 'target.db'
    Library(target).close()
    cfg.db_path = str(target)  # exercises the `args.db or cfg.db_path` else-arm

    rc = core.cmd_import_db(_ns(db=None, bundle=str(bundle),
                               on_conflict='replace', embedding_model='text-embedding-3-large'))
    assert rc == 0
    tgt = Library(target)
    try:
        n = len([d for d in tgt.list_documents() if d.source_name == 'sdk'])
    finally:
        tgt.close()
    assert n == full_db['count']


# --------------------------------------------------------------------------
# parser + handler registration
# --------------------------------------------------------------------------

def test_parser_and_handlers_register_slice_commands():
    from cli.main import create_parser
    parser = create_parser()
    a = parser.parse_args(['--db', 'x.db', 'export-db', '--source', 's', '--out', 'o.db'])
    assert a.command == 'export-db'
    b = parser.parse_args(['import-db', 'bundle.db'])
    assert b.command == 'import-db'
    assert 'export-db' in core.HANDLERS
    assert 'import-db' in core.HANDLERS


def test_slice_commands_work_through_main_without_inference(full_db, tmp_path, monkeypatch):
    import sys
    import config
    from cli.main import main
    from inference import InferenceContext

    path = tmp_path / 'ariadne.yaml'
    path.write_text('sources: {}\n')
    monkeypatch.setattr(config, '_global_config', config.Config(path))

    def no_inference(*args, **kwargs):
        raise AssertionError('SQLite slice transfer must not invoke inference')

    monkeypatch.setattr(InferenceContext, 'chat', property(no_inference))
    monkeypatch.setattr(InferenceContext, 'embeddings', property(no_inference))
    bundle = tmp_path / 'slice.db'
    target = tmp_path / 'target.db'
    monkeypatch.setattr(sys, 'argv', [
        'ariadne', '--db', str(full_db['path']), 'export-db',
        '--source', 'sdk', '--out', str(bundle),
    ])
    assert main() == 0
    monkeypatch.setattr(sys, 'argv', ['ariadne', '--db', str(target), 'import-db', str(bundle)])
    assert main() == 0
    with Library(target) as library:
        assert len(library.list_documents()) == full_db['count']


def test_binary_slices_cannot_strip_or_bypass_model_binding(full_db, tmp_path):
    from export_db import export_source_db, import_source_db
    from library.model_binding import ModelBindingMismatch
    from tests.test_model_binding import context

    ctx = context(tmp_path, 'bound')
    with ctx.write_session():
        ctx.open_library(operation='write').close()
    before = ctx.db_path.read_bytes()
    out = tmp_path / 'out.db'
    with pytest.raises(ModelBindingMismatch, match='provenance'):
        export_source_db(ctx.db_path, 'sdk', out)
    assert not out.exists()
    export_source_db(full_db['path'], 'sdk', out)
    with pytest.raises(ModelBindingMismatch, match='provenance'):
        import_source_db(ctx.db_path, out)
    new_target = tmp_path / 'new.db'
    with pytest.raises(ModelBindingMismatch, match='provenance'):
        import_source_db(new_target, ctx.db_path)
    assert not new_target.exists()
    assert ctx.db_path.read_bytes() == before
