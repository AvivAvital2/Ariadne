"""Guardrail: a full re-embed announces its cost and asks before a big bill.

* ``_rebuild_embeddings`` prints an estimate (doc count + ~$) before embedding.
* A run at/above the confirm threshold prompts; declining skips embedding.
* ``--yes`` / ``assume_yes`` skips the prompt; below threshold never prompts;
  zero docs is a no-op.
* ``ariadne rebuild`` exposes ``--yes`` and threads it; ``ariadne import``
  never prompts (it is the incremental, opted-in path).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli.core as core
from cli.main import create_parser
from library import Library


@pytest.fixture
def printed(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(
        core.console, 'print',
        lambda *a, **k: lines.append(' '.join(str(x) for x in a)),
    )
    return lines


@pytest.fixture
def embed_recorder(monkeypatch):
    rec = {'called': False}

    async def fake_rebuild_all(self, only_missing=False, on_progress=None):
        rec['called'] = True
        rec['only_missing'] = only_missing
        return 0

    monkeypatch.setattr('writer.LibraryWriter.rebuild_all_embeddings', fake_rebuild_all)
    monkeypatch.setattr('library.embedding_matrix.ensure_matrix', lambda library: None)
    return rec


def _seed(tmp_path, n: int) -> Library:
    lib = Library(tmp_path / 't.db')
    for i in range(n):
        lib.add_document(content_type='explanation', title=f'D{i}', content=f'body {i}')
    return lib


async def test_estimate_is_printed(printed, embed_recorder, tmp_path):
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=True)
    finally:
        lib.close()
    blob = '\n'.join(printed)
    assert '3' in blob and '$' in blob, 'estimate with doc count and cost must be printed'


async def test_declined_large_run_skips_embedding(printed, embed_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)
    monkeypatch.setattr(core.console, 'input', lambda *a, **k: 'n')
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=False)
    finally:
        lib.close()
    assert embed_recorder['called'] is False, 'declining must skip embedding'


async def test_zero_docs_skips_embedding(printed, embed_recorder, tmp_path):
    lib = _seed(tmp_path, 0)
    try:
        await core._rebuild_embeddings(lib, only_missing=True, assume_yes=False)
    finally:
        lib.close()
    assert embed_recorder['called'] is False, 'no missing docs → nothing to embed'


async def test_confirmed_large_run_embeds(printed, embed_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)
    monkeypatch.setattr(core.console, 'input', lambda *a, **k: 'y')
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=False)
    finally:
        lib.close()
    assert embed_recorder['called'] is True


async def test_assume_yes_skips_prompt(printed, embed_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)

    def _boom(*a, **k):
        raise AssertionError('must not prompt when assume_yes is set')

    monkeypatch.setattr(core.console, 'input', _boom)
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=True)
    finally:
        lib.close()
    assert embed_recorder['called'] is True


async def test_below_threshold_no_prompt(printed, embed_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 100)

    def _boom(*a, **k):
        raise AssertionError('must not prompt below threshold')

    monkeypatch.setattr(core.console, 'input', _boom)
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=False)
    finally:
        lib.close()
    assert embed_recorder['called'] is True


def test_rebuild_parser_has_yes_flag():
    parser = create_parser()
    assert parser.parse_args(['rebuild', '--yes']).yes is True
    assert parser.parse_args(['rebuild']).yes is False


async def test_cmd_rebuild_threads_yes(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    async def fake(library, only_missing=False, assume_yes=False):
        seen['assume_yes'] = assume_yes

    monkeypatch.setattr(core, '_rebuild_embeddings', fake)
    await core.cmd_rebuild(SimpleNamespace(db=str(tmp_path / 't.db'), only_missing=False, yes=True))
    assert seen['assume_yes'] is True


def test_cmd_import_never_prompts(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    async def fake(library, only_missing=False, assume_yes=False):
        seen['assume_yes'] = assume_yes

    monkeypatch.setattr(core, '_rebuild_embeddings', fake)
    monkeypatch.setattr(core, 'get_config', lambda: SimpleNamespace(default_source=None))
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'n.md').write_text('---\ntype: explanation\ntitle: "N"\n---\n# N\n\nbody\n')
    core.cmd_import_(SimpleNamespace(
        input=str(docs), source=None, db=str(tmp_path / 't.db'), skip_embeddings=False,
    ))
    assert seen['assume_yes'] is True
