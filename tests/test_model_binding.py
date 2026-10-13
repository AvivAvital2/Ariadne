import asyncio
import sqlite3
from dataclasses import replace

import pytest

from inference import InferenceContext
from library import Library
from library.locking import LibraryBusy
from library.model_binding import ModelBindingMismatch, inspect_binding
from model_config import resolve_models
from tests.local_model_server import local_config


def context(tmp_path, name='library'):
    return InferenceContext(
        resolve_models(local_config()),
        tmp_path / (name + '.db'),
        tmp_path / (name + '_staleness.db'),
    )


def test_binding_mismatch_before_constructor_and_sidecar_ownership(tmp_path):
    ctx = context(tmp_path)
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        library.add_document('finding', 'Authored', 'Text')
    before = ctx.db_path.read_bytes()
    changed = replace(ctx.settings, chat=replace(ctx.settings.chat, model_revision='c' * 40))
    with pytest.raises(ModelBindingMismatch, match='generation'):
        InferenceContext(changed, ctx.db_path, ctx.staleness_db_path)
    assert ctx.db_path.read_bytes() == before
    with pytest.raises(ModelBindingMismatch, match='ownership'):
        InferenceContext(ctx.settings, tmp_path / 'second.db', ctx.staleness_db_path)
    library.close()


def test_populated_unbound_structural_database_rejected_unchanged(tmp_path):
    main = tmp_path / 'legacy.db'
    with sqlite3.connect(main) as conn:
        conn.execute('CREATE TABLE scip_only(symbol TEXT)')
        conn.execute('INSERT INTO scip_only VALUES (?)', ('symbol',))
    before = main.read_bytes()
    with pytest.raises(ModelBindingMismatch, match='Populated unbound'):
        InferenceContext(resolve_models(local_config()), main, tmp_path / 'side.db')
    assert main.read_bytes() == before
    assert not (tmp_path / 'side.db').exists()


def test_offline_inspection_and_bare_handle_no_mutations(tmp_path):
    ctx = context(tmp_path)
    assert inspect_binding(ctx.db_path) is None
    assert not ctx.db_path.exists()
    with ctx.write_session():
        ctx.open_library(operation='write').close()
    with Library(ctx.db_path) as library:
        assert library.list_documents() == []
        with pytest.raises(sqlite3.OperationalError):
            library.add_document('finding', 'Unowned', 'Must fail')
    with ctx.open_library(operation='inspect') as library:
        assert library.list_documents() == []


@pytest.mark.asyncio
async def test_independent_tasks_do_not_share_writer_lock(tmp_path):
    ctx = context(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()

    async def first():
        with ctx.write_session() as session:
            with ctx.write_session(reuse=session):
                entered.set()
                await release.wait()

    task = asyncio.create_task(first())
    await entered.wait()
    try:
        with pytest.raises(LibraryBusy):
            with ctx.write_session():
                pass
    finally:
        release.set()
        await task
    with ctx.write_session():
        assert ctx.binding is not None
