"""Guardrail/contract tests for the read-only catalog inspection commands
(``ariadne symbol`` / ``body`` / ``list-file-scopes``) in cli/lookup.py.

These encode the *expected* user-facing behavior (what each command promises
in its help text), driven black-box: a real Library at a tmp ``--db`` with a
known catalog element, an explicit ``--source``, and assertions on the JSON
the command prints. No monkeypatch — config is isolated via a small fixture
that restores the global singleton. A failure here means a command stopped
honoring its contract.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pytest

import config as config_module
from cli.lookup import HANDLERS, cmd_body, cmd_list_file_scopes, cmd_symbol
from cli.main import create_parser
from config import Config
from docgen.catalog_writer import _element_doc_id
from library import Library


def _ns(**kw):
    base = {'source': 'src1', 'db': None, 'name': None, 'file': None}
    return argparse.Namespace(**{**base, **kw})


@pytest.fixture
def empty_config(tmp_path):
    """Isolate get_config() to a sources-less, default-less ariadne.yaml so
    `default_source` is None (for the no-source path). Monkeypatch-free:
    swap the cached singleton + $ARIADNE_CONFIG and restore on teardown.
    """
    cfg_dir = tmp_path / 'cfg'
    cfg_dir.mkdir()
    cfg_file = cfg_dir / 'ariadne.yaml'
    cfg_file.write_text('sources: {}\n')
    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    os.environ['ARIADNE_CONFIG'] = str(cfg_file)
    config_module._global_config = Config(cfg_file)
    try:
        yield
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env


def _add_element(lib, *, source, qn, file, line_start, line_end,
                 language='python', subtype='function', signature=None):
    lib.add_document(
        content_type='catalog',
        title=qn,
        content=signature or f'def {qn.split(".")[-1]}(): ...',
        source_files=[str(file)],
        embedding=np.ones(8, dtype=np.float32) / np.sqrt(8.0),
        metadata={
            'kind': 'element',
            'source_name': source,
            'qualified_name': qn,
            'language': language,
            'subtype': subtype,
            'signature': signature or f'def {qn.split(".")[-1]}()',
            'location': {'line_start': line_start, 'line_end': line_end,
                         'col_start': 0, 'col_end': 0},
            'parent_qualified_name': None,
            'description': f'{qn} description',
        },
        doc_id=_element_doc_id(source, qn),
    )


@pytest.fixture
def db_with_element(tmp_path):
    """A library db holding one known element (src1 / pkg.mod.alpha) plus a
    sibling, written against a real on-disk source file (for `body`).
    """
    src = tmp_path / 'pkg' / 'mod.py'
    src.parent.mkdir(parents=True)
    src.write_text('"""m."""\n\n\ndef alpha():\n    return 1\n\n\ndef beta():\n    return 2\n')

    db = tmp_path / 'lib.db'
    lib = Library(db)
    _add_element(lib, source='src1', qn='pkg.mod.alpha', file=src,
                 line_start=4, line_end=5)
    _add_element(lib, source='src1', qn='pkg.mod.beta', file=src,
                 line_start=8, line_end=9)
    lib.close()
    return db, src


def _run(fn, args, capsys):
    rc = fn(args)
    return rc, json.loads(capsys.readouterr().out)


# --- symbol -----------------------------------------------------------------

def test_symbol_found_returns_element_fields(db_with_element, capsys):
    db, _ = db_with_element
    rc, out = _run(cmd_symbol, _ns(db=str(db), name='pkg.mod.alpha'), capsys)
    assert rc == 0
    assert out['found'] is True
    assert out['qualified_name'] == 'pkg.mod.alpha'
    assert out['language'] == 'python'
    assert out['location']['line_start'] == 4


def test_symbol_miss_suggests_close_names(db_with_element, capsys):
    db, _ = db_with_element
    rc, out = _run(cmd_symbol, _ns(db=str(db), name='pkg.mod.alpa'), capsys)
    assert rc == 0
    assert out['found'] is False
    assert out['error'] == 'not_in_catalog'
    # the typo'd lookup should suggest the real sibling
    assert 'pkg.mod.alpha' in out['suggestions_in_source']


def test_symbol_no_source_errors(empty_config, capsys):
    rc, out = _run(cmd_symbol, _ns(source=None, name='x'), capsys)
    assert rc == 1
    assert out == {'error': 'no_source'}


# --- body -------------------------------------------------------------------

def test_body_returns_source_lines(db_with_element, capsys):
    db, _ = db_with_element
    rc, out = _run(cmd_body, _ns(db=str(db), name='pkg.mod.alpha'), capsys)
    assert rc == 0
    assert out['found'] is True
    assert 'def alpha():' in out['body']
    assert out['body_line_count'] == 2


def test_body_no_source_errors(empty_config, capsys):
    rc, out = _run(cmd_body, _ns(source=None, name='x'), capsys)
    assert rc == 1
    assert out == {'error': 'no_source'}


# --- list-file-scopes -------------------------------------------------------

def test_list_file_scopes_lists_qns_in_file(db_with_element, capsys):
    db, _ = db_with_element
    rc, out = _run(cmd_list_file_scopes,
                   _ns(db=str(db), file='mod.py'), capsys)
    assert rc == 0
    assert sorted(out) == ['pkg.mod.alpha', 'pkg.mod.beta']


def test_list_file_scopes_no_source_errors(empty_config, capsys):
    rc, out = _run(cmd_list_file_scopes, _ns(source=None, file='x.py'), capsys)
    assert rc == 1
    assert out == {'error': 'no_source'}


# --- wiring -----------------------------------------------------------------

def test_commands_registered_and_dispatchable():
    parser = create_parser()
    for argv in (['symbol', '--name', 'x'],
                 ['list-file-scopes', '-s', 'src1', '-f', 'x.py'],
                 ['body', '--name', 'x']):
        assert parser.parse_args(argv).command == argv[0]
    assert {'symbol', 'list-file-scopes', 'body'} <= set(HANDLERS)
