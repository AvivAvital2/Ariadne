"""Guardrail/contract tests for the document-CRUD commands in cli/core.py
(``search`` / ``list`` / ``get`` / ``add`` / ``finding`` / ``delete`` /
``export`` / ``import`` / ``rebuild`` / ``build-matrix`` / ``tag``).

Black-box against a real Library at a tmp ``--db`` (argparse passes ``--db``
as a ``Path``, so we do too) with synthetic, neutral data. Mutating commands
are checked by their *persisted effect* — we reopen the library / inspect the
filesystem — not by their stdout banner, so a logic bug that still prints
"success" is caught. Config is isolated via ``set_config`` (restores the
global singleton); the only mocks are stable external boundaries (the OpenAI
embedding client, ``sys.stdin``, the interactive ``console.input`` prompt).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import textwrap
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from rich.text import Text

import config as config_module
import cli.main as cli_main
from cli.core import (
    HANDLERS,
    _print_document_list,
    cmd_add,
    cmd_build_matrix,
    cmd_delete,
    cmd_export,
    cmd_finding,
    cmd_get,
    cmd_import_,
    cmd_list,
    cmd_rebuild,
    cmd_search,
    cmd_tag,
    get_library,
)
from cli.core import console as core_console
from cli.main import create_parser
from config import Config
import cli.main as cli_main
from export import LibraryExporter
from library import Library
from library.embedding_matrix import ARTIFACT_NAME, matrix_dir_for
from schema import Chunk

# 16-dim synthetic embedding space (dim is arbitrary but must be consistent).
_DIM = 16
_QVEC = np.eye(_DIM, dtype=np.float32)[0]  # one-hot; doc 'Alpha' matches it


def _vec(i):
    return np.eye(_DIM, dtype=np.float32)[i % _DIM]


def _ns(**kw):
    base = {
        'db': None, 'query': 'q', 'k': 5, 'type': None, 'chunks': False,
        'source': None, 'include_all': False, 'status': None, 'branch': None,
        'limit': None, 'id': None, 'title': None, 'file': None,
        'source_files': None, 'finding': None, 'topic': None, 'no_embed': False,
        'force': False, 'output': None, 'input': None, 'skip_embeddings': False,
        'feature': None, 'alias': None, 'remove_branch': None, 'clear': False,
        'recreate': False, 'yes': False,'archive': True, 'batch': False, 'live': False
    }
    return argparse.Namespace(**{**base, **kw})


@pytest.fixture(autouse=True)
def wide_console():
    old = os.environ.get('COLUMNS')
    os.environ['COLUMNS'] = '240'
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('COLUMNS', None)
        else:
            os.environ['COLUMNS'] = old


@pytest.fixture
def set_config(tmp_path):
    """Swap the cached Config singleton to one built from given yaml text.
    Monkeypatch-free; restores the singleton + $ARIADNE_CONFIG on teardown.
    """
    saved_singleton = config_module._global_config
    saved_env = os.environ.get('ARIADNE_CONFIG')
    made = []

    def _set(yaml_text):
        f = tmp_path / f'cfg{len(made)}.yaml'
        f.write_text(yaml_text)
        made.append(f)
        os.environ['ARIADNE_CONFIG'] = str(f)
        config_module._global_config = Config(f)
        return config_module._global_config

    try:
        yield _set
    finally:
        config_module._global_config = saved_singleton
        if saved_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = saved_env


@pytest.fixture
def srcdir(tmp_path):
    d = tmp_path / 'srctree'
    d.mkdir()
    return d


def _src1_yaml(srcdir, tmp_path, *, default=True, extra=''):
    docs = tmp_path / 'docs_out'
    return textwrap.dedent(f'''\
        {'default_source: src1' if default else ''}
        docs_base: {docs}
        sources:
          src1:
            path: {srcdir}
        {extra}''')


@pytest.fixture
def fake_embed():
    """Patch the OpenAI embedding boundary: ``embed`` returns the query
    vector (so 'Alpha' ranks first); ``embed_batch`` returns zeros.
    """
    async def embed(self, text):
        return _QVEC.copy()

    async def embed_batch(self, texts):
        return [np.zeros(_DIM, dtype=np.float32) for _ in texts]

    async def noop(self, *a, **k):
        return None

    with mock.patch('embedding.EmbeddingService.embed', embed), \
         mock.patch('embedding.EmbeddingService.embed_batch', embed_batch), \
         mock.patch('embedding.EmbeddingService._get_client', noop), \
         mock.patch('embedding.EmbeddingService.close', noop):
        yield


def _lib(tmp_path, name='lib.db'):
    return Library(tmp_path / name)


def _reopen(db):
    """Reopen the on-disk library so tests assert the persisted effect of a
    command rather than its stdout banner.
    """
    return Library(db)


def _add(lib, *, title, source='src1', ctype='explanation', content='body text',
         embedding=None, source_files=None, metadata=None):
    return lib.add_document(
        content_type=ctype, title=title, content=content,
        source_name=source, source_files=source_files,
        embedding=embedding, metadata=metadata,
    )


def _out(capsys):
    return ' '.join(capsys.readouterr().out.split())


# --- search -----------------------------------------------------------------

def test_search_no_results(tmp_path, fake_embed, capsys):
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, include_all=True)))
    assert rc == 0
    assert 'No results found' in _out(capsys)


def test_search_include_all_ranks_best_match_first(tmp_path, fake_embed, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='Alpha', embedding=_vec(0))  # == query vector
    _add(lib, title='Beta', embedding=_vec(1))   # orthogonal
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, include_all=True, k=2)))
    out = _out(capsys)
    assert rc == 0
    # both returned, and the closer match is ranked first (order is the contract)
    assert 'Alpha' in out and 'Beta' in out
    assert out.index('Alpha') < out.index('Beta')


def test_search_scoped_ranks_best_match_first(set_config, srcdir, tmp_path, fake_embed, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='Alpha', source='src1', embedding=_vec(0))
    _add(lib, title='Beta', source='src1', embedding=_vec(1))
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, source='src1', k=2)))
    out = _out(capsys)
    assert rc == 0
    assert out.index('Alpha') < out.index('Beta')


def test_search_scope_excludes_out_of_scope_doc(set_config, srcdir, tmp_path, fake_embed, capsys):
    """Scoping must EXCLUDE a doc whose source files lie outside the scope —
    proving the filter does work, not merely that the in-scope doc survives.
    Both docs share the query embedding, so only scope can separate them.
    """
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='InScopeDoc', source='src1',
         source_files=[str(srcdir / 'a.py')], embedding=_vec(0))
    _add(lib, title='OutScopeDoc', source='src1',
         source_files=['/elsewhere/z.py'], embedding=_vec(0))
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, source='src1', k=5)))
    out = _out(capsys)
    assert rc == 0
    assert 'InScopeDoc' in out
    assert 'OutScopeDoc' not in out


def test_search_scoped_with_dependencies(set_config, srcdir, tmp_path, fake_embed, capsys):
    # src1 depends on src2 (has path) and src3 (serve-only, no path) — exercises
    # both the dep-path-present and dep-path-absent branches of scope building.
    dep = tmp_path / 'dep2'
    dep.mkdir()
    yaml_text = textwrap.dedent(f'''\
        default_source: src1
        docs_base: {tmp_path / 'docs_out'}
        sources:
          src1:
            path: {srcdir}
            depends_on: [src2, src3]
          src2:
            path: {dep}
          src3: {{}}
        ''')
    set_config(yaml_text)
    lib = _lib(tmp_path)
    _add(lib, title='Alpha', source='src1', embedding=_vec(0))
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, source='src1')))
    assert rc == 0
    assert 'Alpha' in _out(capsys)


def test_search_serve_only_source_has_no_scope(set_config, tmp_path, fake_embed, capsys):
    # src1 configured but serve-only (no path) -> no scope paths -> plain search.
    yaml_text = textwrap.dedent(f'''\
        default_source: src1
        docs_base: {tmp_path / 'docs_out'}
        sources:
          src1: {{}}
        ''')
    set_config(yaml_text)
    lib = _lib(tmp_path)
    _add(lib, title='Alpha', source='src1', embedding=_vec(0))
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, source='src1')))
    assert rc == 0
    assert 'Alpha' in _out(capsys)


def test_search_no_source_no_default(set_config, tmp_path, fake_embed, capsys):
    set_config('sources: {}\n')
    lib = _lib(tmp_path)
    _add(lib, title='Alpha', source='src1', embedding=_vec(0))
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, source=None)))
    assert rc == 0
    assert 'Alpha' in _out(capsys)


def test_search_chunks_scoped(set_config, srcdir, tmp_path, fake_embed, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='Alpha', source='src1', embedding=_vec(0))
    lib.add_chunk(Chunk(document_id=doc.id, chunk_index=0,
                        content='chunk body about alpha', embedding=_vec(0)))
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, source='src1', chunks=True)))
    out = _out(capsys)
    assert rc == 0
    # chunk search surfaces the matching chunk's text as the preview
    assert 'chunk body about alpha' in out


def test_search_chunks_include_all(tmp_path, fake_embed, capsys):
    lib = _lib(tmp_path)
    doc = _add(lib, title='Alpha', embedding=_vec(0))
    lib.add_chunk(Chunk(document_id=doc.id, chunk_index=0,
                        content='ia chunk', embedding=_vec(0)))
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_search(_ns(db=db, include_all=True, chunks=True)))
    assert rc == 0
    assert 'ia chunk' in _out(capsys)


# --- list -------------------------------------------------------------------

def test_list_empty(tmp_path, capsys):
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    assert cmd_list(_ns(db=db)) == 0
    assert 'No documents found' in _out(capsys)


def test_list_renders_with_status_column(tmp_path, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='Stable Doc', metadata={'status': 'stable'})
    _add(lib, title='No Status Doc')  # -> defaults to 'stable' in the column
    db = lib.path
    lib.close()
    rc = cmd_list(_ns(db=db, status=None))
    out = _out(capsys)
    assert rc == 0
    assert 'Status' in out  # status column header present
    assert 'Stable Doc' in out and 'No Status Doc' in out


def test_list_filters_by_status(tmp_path, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='Keep', metadata={'status': 'experimental'})
    _add(lib, title='Drop', metadata={'status': 'stable'})
    db = lib.path
    lib.close()
    rc = cmd_list(_ns(db=db, status='experimental'))
    out = _out(capsys)
    assert rc == 0
    assert 'Keep' in out and 'Drop' not in out


def test_list_filters_by_branch_and_limit(tmp_path, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='Feat1', metadata={'branches': ['feature/*']})
    _add(lib, title='Feat2', metadata={'branches': ['feature/*']})
    _add(lib, title='Feat3', metadata={'branches': ['feature/*']})
    db = lib.path
    lib.close()
    # branch filter runs (all three match feature/x), then limit truncates to 2
    rc = cmd_list(_ns(db=db, branch='feature/x', limit=2))
    out = _out(capsys)
    assert rc == 0
    assert 'Documents (2)' in out


# --- get --------------------------------------------------------------------

def test_get_found_with_source_files(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='Gettable', source='src1', content='the body text',
               source_files=['a/b.py'])
    db = lib.path
    lib.close()
    rc = cmd_get(_ns(db=db, id=doc.id, source='src1'))
    out = _out(capsys)
    assert rc == 0
    # the actual stored fields are rendered: title, id, source files, content
    assert 'Gettable' in out
    assert doc.id in out
    assert 'Source files' in out and 'a/b.py' in out
    assert 'the body text' in out


def test_get_not_found(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='X', source='src1')
    db = lib.path
    lib.close()
    rc = cmd_get(_ns(db=db, id='nope', source='src1'))
    assert rc == 1
    assert 'Document not found' in _out(capsys)


def test_get_unresolvable_source(set_config, tmp_path, capsys):
    set_config('sources: {}\n')
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = cmd_get(_ns(db=db, id='x', source=None))
    assert rc == 1
    assert 'Cannot resolve source' in _out(capsys)


# --- add --------------------------------------------------------------------

def test_add_from_file_persists_document(set_config, srcdir, tmp_path, fake_embed, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    f = tmp_path / 'content.md'
    f.write_text('hello world content')
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_add(
        _ns(db=db, source='src1', type='explanation', title='New Doc',
            file=str(f), source_files='x.py,y.py')))
    assert rc == 0
    assert 'Document created' in _out(capsys)
    # the document must actually be in the library, with the right fields
    lib2 = _reopen(db)
    docs = lib2.list_documents()
    lib2.close()
    new = next(d for d in docs if d.title == 'New Doc')
    assert new.content == 'hello world content'
    assert new.content_type == 'explanation'
    assert new.source_name == 'src1'
    assert new.source_files == ['x.py', 'y.py']


def test_add_from_stdin_persists_document(set_config, srcdir, tmp_path, fake_embed):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    with mock.patch('sys.stdin', io.StringIO('typed content')):
        rc = asyncio.run(cmd_add(
            _ns(db=db, source='src1', type='explanation', title='Typed',
                file=None)))
    assert rc == 0
    lib2 = _reopen(db)
    docs = lib2.list_documents()
    lib2.close()
    new = next(d for d in docs if d.title == 'Typed')
    assert new.content == 'typed content'


def test_add_unresolvable_source_persists_nothing(set_config, tmp_path, fake_embed, capsys):
    set_config('sources: {}\n')
    f = tmp_path / 'c.md'
    f.write_text('x')
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_add(
        _ns(db=db, source=None, title='T', file=str(f))))
    assert rc == 1
    assert 'Cannot resolve a source' in _out(capsys)
    lib2 = _reopen(db)
    count = lib2.count_documents()
    lib2.close()
    assert count == 0  # nothing was written


# --- finding ----------------------------------------------------------------

def test_finding_no_embed_persists_and_exports(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_finding(
        _ns(db=db, source='src1', finding='A discovery.', topic='My Topic',
            no_embed=True)))
    out = _out(capsys)
    assert rc == 0
    assert 'Finding saved' in out
    assert 'rebuild' in out  # the no-embed note
    assert 'Exported to' in out
    # the finding is persisted as a 'finding' doc with the topic as title ...
    lib2 = _reopen(db)
    docs = lib2.list_documents(content_type='finding')
    lib2.close()
    saved = next(d for d in docs if d.title == 'My Topic')
    assert saved.content == 'A discovery.'
    assert saved.embedding is None  # no_embed -> not embedded
    # ... and the auto-export actually wrote markdown to the docs path
    docs_dir = tmp_path / 'docs_out' / 'src1'
    assert docs_dir.exists()
    assert list(docs_dir.rglob('*.md'))


def test_finding_with_embed_autotitle(set_config, srcdir, tmp_path, fake_embed):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_finding(
        _ns(db=db, source='src1', finding='x' * 100, topic=None,
            no_embed=False, source_files='a.py')))
    assert rc == 0
    lib2 = _reopen(db)
    docs = lib2.list_documents(content_type='finding')
    lib2.close()
    saved = docs[0]
    # no topic -> title is the first line truncated to 80 chars
    assert saved.title == 'x' * 80
    assert saved.embedding is not None  # embedded via writer


def test_finding_unresolvable_source(set_config, tmp_path, capsys):
    set_config('sources: {}\n')
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_finding(
        _ns(db=db, source=None, finding='x', no_embed=True)))
    assert rc == 1
    assert 'Cannot resolve a source' in _out(capsys)


def test_finding_no_default_skips_export(set_config, srcdir, tmp_path, capsys):
    # explicit source resolves the save, but no default_source -> export skipped
    set_config(_src1_yaml(srcdir, tmp_path, default=False))
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_finding(
        _ns(db=db, source='src1', finding='note', topic='T', no_embed=True)))
    out = _out(capsys)
    assert rc == 0
    assert 'Finding saved' in out
    assert 'Exported to' not in out
    # the finding is still saved, but no docs dir was written
    lib2 = _reopen(db)
    assert lib2.count_documents(content_type='finding') == 1
    lib2.close()
    assert not (tmp_path / 'docs_out').exists()


# --- delete -----------------------------------------------------------------

def test_delete_force_actually_removes(tmp_path, capsys):
    lib = _lib(tmp_path)
    doc = _add(lib, title='Doomed')
    db = lib.path
    lib.close()
    rc = cmd_delete(_ns(db=db, id=doc.id, force=True))
    assert rc == 0
    assert 'Document deleted' in _out(capsys)
    lib2 = _reopen(db)
    gone = lib2.get_document(doc.id)
    lib2.close()
    assert gone is None  # the doc is really gone


def test_delete_force_missing(tmp_path, capsys):
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = cmd_delete(_ns(db=db, id='ghost', force=True))
    assert rc == 1
    assert 'Document not found' in _out(capsys)


def test_delete_confirm_yes_removes(set_config, srcdir, tmp_path):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='Confirmed', source='src1')
    db = lib.path
    lib.close()
    with mock.patch.object(core_console, 'input', return_value='y') as prompt:
        rc = cmd_delete(_ns(db=db, id=doc.id, source='src1', force=False))
    assert rc == 0
    lib2 = _reopen(db)
    gone = lib2.get_document(doc.id)
    lib2.close()
    assert gone is None
    # The y/N hint must survive Rich markup rendering — an unescaped
    # '[y/N]' parses as a markup tag and vanishes from the terminal.
    assert '[y/N]' in Text.from_markup(prompt.call_args[0][0]).plain


def test_delete_confirm_no_keeps_document(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='Spared', source='src1')
    db = lib.path
    lib.close()
    with mock.patch.object(core_console, 'input', return_value='n'):
        rc = cmd_delete(_ns(db=db, id=doc.id, source='src1', force=False))
    assert rc == 0
    assert 'Cancelled' in _out(capsys)
    lib2 = _reopen(db)
    survivor = lib2.get_document(doc.id)
    lib2.close()
    assert survivor is not None  # declining the prompt must NOT delete


def test_delete_unresolvable_source(set_config, tmp_path, capsys):
    set_config('sources: {}\n')
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = cmd_delete(_ns(db=db, id='x', source=None, force=False))
    assert rc == 1
    assert 'Cannot resolve source' in _out(capsys)


def test_delete_out_of_scope_refuses(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='Foreign', source='other')  # not in src1 closure
    db = lib.path
    lib.close()
    rc = cmd_delete(_ns(db=db, id=doc.id, source='src1', force=False))
    assert rc == 1
    assert 'not found in scope' in _out(capsys)
    # the out-of-scope doc must survive a scoped delete attempt
    lib2 = _reopen(db)
    survivor = lib2.get_document(doc.id)
    lib2.close()
    assert survivor is not None


# --- export -----------------------------------------------------------------

def test_export_default_emits_single_zip(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='ExportMe', source='src1', content='exported content')
    db = lib.path
    lib.close()
    rc = cmd_export(_ns(db=db, source='src1', output=None))
    assert rc == 0
    assert 'Exported' in _out(capsys)
    # one artifact: docs_base/src1.zip — and no docs tree left behind
    docs_base = tmp_path / 'docs_out'
    assert (docs_base / 'src1.zip').exists()
    assert not (docs_base / 'src1').exists()
    with zipfile.ZipFile(docs_base / 'src1.zip') as zf:
        assert zf.comment == b'ariadne-export'
        assert any(n.endswith('.md') for n in zf.namelist())


def test_export_explicit_output_archive_writes_zip(set_config, srcdir, tmp_path):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='ExportMe', source='src1')
    db = lib.path
    lib.close()
    target = tmp_path / 'explicit_out' / 'bundle.zip'
    rc = cmd_export(_ns(db=db, source='src1', output=str(target)))
    assert rc == 0
    # the positional output names the zip itself in archive mode
    assert target.is_file() and zipfile.is_zipfile(target)
    assert list(target.parent.iterdir()) == [target]


def test_export_no_archive_to_explicit_output_writes_files(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='ExportMe', source='src1', content='exported content')
    db = lib.path
    lib.close()
    outdir = tmp_path / 'explicit_out'
    rc = cmd_export(_ns(db=db, source='src1', output=str(outdir), archive=False))
    assert rc == 0
    assert 'Exported' in _out(capsys)
    # files actually landed in the chosen output dir
    assert (outdir / 'manifest.yaml').exists()
    assert list(outdir.rglob('*.md'))


def test_export_no_archive_default_source_path_writes_files(set_config, srcdir, tmp_path):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='ExportMe', source='src1')
    db = lib.path
    lib.close()
    rc = cmd_export(_ns(db=db, source='src1', output=None, archive=False))
    assert rc == 0
    # default output resolves to docs_base/src1
    assert (tmp_path / 'docs_out' / 'src1' / 'manifest.yaml').exists()


def test_export_no_source_falls_back_to_default_path(set_config, tmp_path):
    set_config('sources: {}\n')
    lib = _lib(tmp_path)
    _add(lib, title='ExportMe', source=None)
    db = lib.path
    lib.close()
    cwd = Path.cwd()
    os.chdir(tmp_path)  # DEFAULT_EXPORT_PATH is ./docs — keep it under tmp
    try:
        rc = cmd_export(_ns(db=db, source=None, output=None))
        assert rc == 0
        # archive default: the fallback docs dir becomes ./docs.zip
        assert (tmp_path / 'docs.zip').exists()
        assert not (tmp_path / 'docs').exists()
    finally:
        os.chdir(cwd)


def test_export_no_archive_no_source_falls_back_to_default_path(set_config, tmp_path):
    set_config('sources: {}\n')
    lib = _lib(tmp_path)
    _add(lib, title='ExportMe', source=None)
    db = lib.path
    lib.close()
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        rc = cmd_export(_ns(db=db, source=None, output=None, archive=False))
        assert rc == 0
        assert (tmp_path / 'docs' / 'manifest.yaml').exists()
    finally:
        os.chdir(cwd)


def test_export_refuses_overwriting_foreign_zip(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='ExportMe', source='src1')
    db = lib.path
    lib.close()
    docs_base = tmp_path / 'docs_out'
    docs_base.mkdir()
    foreign = docs_base / 'src1.zip'
    with zipfile.ZipFile(foreign, 'w') as zf:
        zf.writestr('keep/precious.md', 'user data, not ours')
    before = foreign.read_bytes()

    rc = cmd_export(_ns(db=db, source='src1', output=None))

    assert rc == 1
    assert 'not written by Ariadne' in _out(capsys)
    assert foreign.read_bytes() == before  # the user file survives


# --- import -----------------------------------------------------------------

def test_import_missing_dir(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = cmd_import_(_ns(db=db, source='src1', input=str(tmp_path / 'nope'),
                         skip_embeddings=True))
    assert rc == 1
    assert 'Input not found' in _out(capsys)


def test_import_prefers_default_zip(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    _add(lib, title='Zipped Doc', source='src1', content='travels by zip')
    db = lib.path
    lib.close()
    rc = cmd_export(_ns(db=db, source='src1', output=None))
    assert rc == 0  # produces docs_base/src1.zip

    target = tmp_path / 'target.db'
    Library(target).close()
    rc = cmd_import_(_ns(db=target, source='src1', input=None, skip_embeddings=True))
    assert rc == 0
    assert 'Imported' in _out(capsys)
    lib2 = _reopen(target)
    titles = {d.title for d in lib2.list_documents()}
    lib2.close()
    assert 'Zipped Doc' in titles


def test_import_bogus_zip_input_fails(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    bogus = tmp_path / 'fake.zip'
    bogus.write_text('not an archive')
    rc = cmd_import_(_ns(db=db, source='src1', input=str(bogus),
                         skip_embeddings=True))
    assert rc == 1
    assert 'not a zip archive' in _out(capsys)


def test_import_populates_fresh_db(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    # export from a source library, then import into a SEPARATE empty db
    src_lib = _lib(tmp_path, name='src.db')
    _add(src_lib, title='Exported Doc', source='src1', content='some content here')
    exp_dir = tmp_path / 'roundtrip'
    LibraryExporter(src_lib).export_all(output_dir=exp_dir, source_name='src1',
                                        source_path=srcdir)
    src_lib.close()

    target = tmp_path / 'target.db'
    Library(target).close()  # create empty target
    rc = cmd_import_(_ns(db=target, source='src1', input=str(exp_dir),
                         skip_embeddings=True))
    assert rc == 0
    assert 'Imported' in _out(capsys)
    # the doc must now exist in the previously-empty target db
    lib2 = _reopen(target)
    titles = {d.title for d in lib2.list_documents()}
    lib2.close()
    assert 'Exported Doc' in titles


def test_import_with_embeddings_regenerates(set_config, srcdir, tmp_path, fake_embed, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    src_lib = _lib(tmp_path, name='src.db')
    _add(src_lib, title='Exported Doc', source='src1', content='content body')
    exp_dir = tmp_path / 'rt2'
    LibraryExporter(src_lib).export_all(output_dir=exp_dir, source_name='src1',
                                        source_path=srcdir)
    src_lib.close()

    target = tmp_path / 'target2.db'
    Library(target).close()
    rc = cmd_import_(_ns(db=target, source='src1', input=str(exp_dir),
                         skip_embeddings=False))
    out = _out(capsys)
    assert rc == 0
    assert 'Imported' in out and 'Embeddings regenerated' in out
    # the imported doc was embedded by the regeneration pass
    lib2 = _reopen(target)
    docs = lib2.list_documents()
    lib2.close()
    assert docs and all(d.embedding is not None for d in docs)


# --- rebuild ----------------------------------------------------------------

def test_rebuild_embeds_unembedded_docs(tmp_path, fake_embed, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='R', content='to be embedded')  # no embedding yet
    db = lib.path
    lib.close()
    rc = asyncio.run(cmd_rebuild(_ns(db=db)))
    assert rc == 0
    assert 'All embeddings rebuilt' in _out(capsys)
    lib2 = _reopen(db)
    docs = lib2.list_documents()
    lib2.close()
    assert docs[0].embedding is not None  # now embedded


# --- build-matrix -----------------------------------------------------------

def test_build_matrix_writes_artifact(tmp_path, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='M', embedding=_vec(0))
    db = lib.path
    lib.close()
    rc = cmd_build_matrix(_ns(db=db))
    out = _out(capsys)
    assert rc == 0
    assert 'Embedding matrix ready' in out and '1 docs' in out
    # the artifact file is actually on disk
    lib2 = _reopen(db)
    artifact = matrix_dir_for(lib2) / ARTIFACT_NAME
    lib2.close()
    assert artifact.exists()


def test_build_matrix_no_embeddings(tmp_path, capsys):
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = cmd_build_matrix(_ns(db=db))
    assert rc == 0
    assert 'nothing to build' in _out(capsys)


def test_build_matrix_recreate_confirm_no_keeps_artifact(tmp_path, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='M', embedding=_vec(0))
    db = lib.path
    lib.close()
    assert cmd_build_matrix(_ns(db=db)) == 0  # build it first
    capsys.readouterr()
    lib2 = _reopen(db)
    artifact = matrix_dir_for(lib2) / ARTIFACT_NAME
    lib2.close()
    with mock.patch.object(core_console, 'input', return_value='n') as prompt:
        rc = cmd_build_matrix(_ns(db=db, recreate=True))
    assert rc == 0
    assert 'Cancelled' in _out(capsys)
    assert artifact.exists()  # declining recreate leaves the artifact in place
    # The y/N hint must survive Rich markup rendering (same swallowed-tag
    # bug as the embedding proceed prompt).
    assert '[y/N]' in Text.from_markup(prompt.call_args[0][0]).plain


def test_build_matrix_recreate_yes(tmp_path, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='M', embedding=_vec(0))
    db = lib.path
    lib.close()
    assert cmd_build_matrix(_ns(db=db)) == 0
    capsys.readouterr()
    rc = cmd_build_matrix(_ns(db=db, recreate=True, yes=True))
    assert rc == 0
    assert 'Embedding matrix ready' in _out(capsys)


def test_build_matrix_recreate_no_existing(tmp_path, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='M', embedding=_vec(0))
    db = lib.path
    lib.close()
    # recreate when no artifact exists yet -> skips the unlink, just builds
    rc = cmd_build_matrix(_ns(db=db, recreate=True, yes=True))
    assert rc == 0
    assert 'Embedding matrix ready' in _out(capsys)


# --- tag --------------------------------------------------------------------

def test_tag_sets_all_metadata(set_config, srcdir, tmp_path):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='Taggable', source='src1')
    db = lib.path
    lib.close()
    rc = cmd_tag(_ns(db=db, id=doc.id, source='src1', status='experimental',
                     branch='feature/x', feature='auth', alias='login'))
    assert rc == 0
    # the metadata must be persisted exactly as set
    lib2 = _reopen(db)
    meta = lib2.get_document(doc.id).metadata
    lib2.close()
    assert meta['status'] == 'experimental'
    assert meta['branches'] == ['feature/x']
    assert meta['feature'] == 'auth'
    assert meta['aliases'] == ['login']


def test_tag_idempotent_does_not_duplicate(set_config, srcdir, tmp_path):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='T4', source='src1',
               metadata={'branches': ['b1'], 'aliases': ['a1']})
    db = lib.path
    lib.close()
    # branch/alias already present -> must NOT be appended again; absent
    # remove-branch is a no-op
    rc = cmd_tag(_ns(db=db, id=doc.id, source='src1', branch='b1', alias='a1',
                     remove_branch='zzz'))
    assert rc == 0
    lib2 = _reopen(db)
    meta = lib2.get_document(doc.id).metadata
    lib2.close()
    assert meta['branches'] == ['b1']   # no duplicate
    assert meta['aliases'] == ['a1']    # no duplicate


def test_tag_remove_branch_and_coerces_non_list(set_config, srcdir, tmp_path):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    # pre-seed branches/aliases as non-list to exercise the coercion branches
    doc = _add(lib, title='T2', source='src1',
               metadata={'branches': 'notalist', 'aliases': 'notalist'})
    db = lib.path
    lib.close()
    rc = cmd_tag(_ns(db=db, id=doc.id, source='src1', branch='b1',
                     alias='a1', remove_branch='b1'))
    assert rc == 0
    lib2 = _reopen(db)
    meta = lib2.get_document(doc.id).metadata
    lib2.close()
    assert meta['branches'] == []        # b1 added then removed
    assert meta['aliases'] == ['a1']     # coerced from 'notalist', a1 appended


def test_tag_clear_removes_metadata(set_config, srcdir, tmp_path):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='T3', source='src1',
               metadata={'status': 'stable', 'branches': ['x'], 'feature': 'f',
                         'aliases': ['y']})
    db = lib.path
    lib.close()
    rc = cmd_tag(_ns(db=db, id=doc.id, source='src1', clear=True))
    assert rc == 0
    lib2 = _reopen(db)
    meta = lib2.get_document(doc.id).metadata
    lib2.close()
    for key in ('status', 'branches', 'feature', 'aliases'):
        assert key not in meta


def test_tag_not_found(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = cmd_tag(_ns(db=db, id='ghost', source='src1'))
    assert rc == 1
    assert 'Document not found' in _out(capsys)


def test_tag_unresolvable_source(set_config, tmp_path, capsys):
    set_config('sources: {}\n')
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    rc = cmd_tag(_ns(db=db, id='x', source=None))
    assert rc == 1
    assert 'Cannot resolve source' in _out(capsys)


# --- helpers / edge branches ------------------------------------------------

def test_get_library_uses_config_db(set_config, tmp_path):
    dbp = tmp_path / 'cfgdb.db'
    set_config(f'db_path: {dbp}\nsources: {{}}\n')
    lib = get_library(None)  # db_path None -> resolved from config
    try:
        assert lib.path.name == 'cfgdb.db'
    finally:
        lib.close()


def test_print_document_list_without_status_column(tmp_path, capsys):
    lib = _lib(tmp_path)
    _add(lib, title='Plain')
    docs = lib.list_documents()
    lib.close()
    _print_document_list(docs, show_status=False)
    out = _out(capsys)
    assert 'Plain' in out
    assert 'Status' not in out


def test_get_found_without_source_files(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    doc = _add(lib, title='NoFiles', source='src1')  # no source_files
    db = lib.path
    lib.close()
    rc = cmd_get(_ns(db=db, id=doc.id, source='src1'))
    out = _out(capsys)
    assert rc == 0
    assert 'NoFiles' in out
    assert 'Source files' not in out


def test_import_input_none_uses_docs_path(set_config, srcdir, tmp_path, capsys):
    set_config(_src1_yaml(srcdir, tmp_path))
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    # input=None + source -> resolve_docs_path(src1), which doesn't exist
    rc = cmd_import_(_ns(db=db, source='src1', input=None, skip_embeddings=True))
    assert rc == 1
    assert 'Input not found' in _out(capsys)


def test_import_input_none_default_path(set_config, tmp_path, capsys):
    set_config('sources: {}\n')
    lib = _lib(tmp_path)
    db = lib.path
    lib.close()
    cwd = Path.cwd()
    os.chdir(tmp_path)  # so the default ./docs path doesn't exist
    try:
        rc = cmd_import_(_ns(db=db, source=None, input=None, skip_embeddings=True))
    finally:
        os.chdir(cwd)
    assert rc == 1
    assert 'Input not found' in _out(capsys)


# --- wiring -----------------------------------------------------------------

def test_commands_registered_and_dispatchable():
    parser = create_parser()
    assert parser.parse_args(['search', 'q']).command == 'search'
    assert parser.parse_args(['list']).command == 'list'
    assert parser.parse_args(['get', 'idv']).command == 'get'
    assert parser.parse_args(['add', '--title', 't']).command == 'add'
    assert parser.parse_args(['delete', 'idv']).command == 'delete'
    expected = {'search', 'list', 'get', 'add', 'finding', 'delete',
                'export', 'import', 'rebuild', 'build-matrix', 'tag'}
    assert expected <= set(HANDLERS)


def test_global_debug_flag_parses_and_configures_logging():
    parser = create_parser()
    assert parser.parse_args(['--debug', 'config']).debug is True
    assert parser.parse_args(['config']).debug is False

    root = logging.getLogger()
    saved = root.level
    httpx_saved = logging.getLogger('httpx').level
    httpcore_saved = logging.getLogger('httpcore').level
    try:
        cli_main._configure_logging(True)
        assert root.level == logging.DEBUG
        # --debug reveals Ariadne's own retry/backoff chatter, but httpx/httpcore
        # log every request at INFO ("HTTP Request: POST .../embeddings 200 OK"),
        # which floods the console and tears up the Rich progress bar. They stay
        # at WARNING regardless of --debug.
        assert logging.getLogger('httpx').level == logging.WARNING
        assert logging.getLogger('httpcore').level == logging.WARNING
        cli_main._configure_logging(False)
        assert root.level == logging.WARNING
    finally:
        root.setLevel(saved)
        logging.getLogger('httpx').setLevel(httpx_saved)
        logging.getLogger('httpcore').setLevel(httpcore_saved)


def test_main_configures_logging_from_debug_flag(monkeypatch, capsys):
    root = logging.getLogger()
    saved = root.level
    monkeypatch.setattr('sys.argv', ['ariadne', '--debug'])
    try:
        assert cli_main.main() == 0  # no command -> help, after logging setup
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(saved)
