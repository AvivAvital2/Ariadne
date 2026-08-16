"""Guardrail: the locked-in CLI contract for incremental embedding.

* ``ariadne rebuild`` defaults to a FULL re-embed; ``--only-missing`` opts
  into the cheap top-up.
* ``ariadne import`` always embeds incrementally (only the missing/changed
  docs) — the high-frequency command must not re-bill the whole corpus.

We assert the contract at the handler boundary: what ``only_missing`` value
reaches ``_rebuild_embeddings``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli.core as core
from cli.main import create_parser


@pytest.fixture
def recorder(monkeypatch):
    """Capture the only_missing value handed to the embedding rebuild."""
    seen: dict[str, object] = {}

    async def fake_rebuild(library, only_missing=False, assume_yes=False, use_batch=False, source=None):
        seen['only_missing'] = only_missing

    monkeypatch.setattr(core, '_rebuild_embeddings', fake_rebuild)
    return seen


def test_rebuild_parser_exposes_only_missing_flag():
    parser = create_parser()
    assert parser.parse_args(['rebuild', '--only-missing']).only_missing is True
    assert parser.parse_args(['rebuild']).only_missing is False


async def test_cmd_rebuild_threads_only_missing(recorder, tmp_path):
    args = SimpleNamespace(db=str(tmp_path / 't.db'), only_missing=True)
    await core.cmd_rebuild(args)
    assert recorder['only_missing'] is True


async def test_cmd_rebuild_defaults_to_full(recorder, tmp_path):
    args = SimpleNamespace(db=str(tmp_path / 't.db'), only_missing=False)
    await core.cmd_rebuild(args)
    assert recorder['only_missing'] is False


def test_cmd_import_defaults_to_only_missing(recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'get_config', lambda: SimpleNamespace(default_source=None))
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'note.md').write_text('---\ntype: explanation\ntitle: "N"\n---\n# N\n\nbody\n')
    args = SimpleNamespace(
        input=str(docs), source=None, db=str(tmp_path / 't.db'), skip_embeddings=False,
    yes=False, batch=False, live=False)

    core.cmd_import_(args)

    assert recorder['only_missing'] is True


async def test_rebuild_embeddings_threads_only_missing_to_writer(monkeypatch, tmp_path):
    """The real _rebuild_embeddings must pass only_missing down to the writer."""
    seen: dict[str, object] = {}

    async def fake_rebuild_all(self, only_missing=False, on_progress=None, source=None):
        seen['only_missing'] = only_missing
        return 0

    monkeypatch.setattr('writer.LibraryWriter.rebuild_all_embeddings', fake_rebuild_all)
    monkeypatch.setattr('library.embedding_matrix.ensure_matrix', lambda library: None)

    from library import Library
    lib = Library(tmp_path / 't.db')
    lib.add_document(content_type='explanation', title='X', content='x')  # 1 missing → not a no-op
    try:
        await core._rebuild_embeddings(lib, only_missing=True)
    finally:
        lib.close()

    assert seen['only_missing'] is True
