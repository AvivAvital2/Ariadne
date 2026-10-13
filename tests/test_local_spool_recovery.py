import asyncio
import json
import threading
from pathlib import Path

import pytest

from docgen.spool import LocalExecutor, SpoolStore
from tests.test_local_spool import document_spec


class ProcessCrash(BaseException):
    pass


@pytest.mark.asyncio
async def test_crash_after_response_resumes_without_new_chat(local_deployment, monkeypatch):
    ctx, chat, _ = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create([document_spec()], operation='generate', scope={})
        update = store.update_job

        def crash(job_id, **values):
            update(job_id, **values)
            if values.get('state') == 'response_ready':
                raise ProcessCrash

        monkeypatch.setattr(store, 'update_job', crash)
        with pytest.raises(ProcessCrash):
            await LocalExecutor(store).drive(run_id)
        assert store.jobs(run_id)[0]['response_text']
    with ctx.write_session():
        resumed = SpoolStore(ctx, library)
        assert (await LocalExecutor(resumed).drive(run_id))['applied'] == 1
        assert len([r for r, _, _ in chat.requests if r.endswith('/chat/completions')]) == 1
    await ctx.aclose()


@pytest.mark.asyncio
async def test_cancellation_while_model_request_is_in_flight(local_deployment):
    ctx, chat, _ = local_deployment
    started, release = threading.Event(), threading.Event()

    def barrier():
        started.set()
        release.wait(10)

    chat.barrier = barrier

    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create([document_spec()], operation='generate', scope={})
        task = asyncio.create_task(LocalExecutor(store).drive(run_id))
        try:
            for _ in range(500):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()
            # A separate control-state connection deliberately does not acquire
            # the corpus writer lock owned by the worker.
            SpoolStore(ctx).cancel(run_id)
        finally:
            release.set()
        result = await task
        assert result['state'] == 'cancelled' and result['applied'] == 0
        assert library.get_document('doc') is None
    await ctx.aclose()


@pytest.mark.parametrize(
    'transcript',
    json.loads((Path(__file__).parent / 'fixtures/model_support/contract-v1.json').read_text())[
        'crash_boundaries'
    ],
    ids=lambda row: row['boundary'],
)
def test_sigkill_releases_lock_and_resumes_saved_stage(local_deployment, tmp_path, transcript):
    import subprocess
    import sys
    import time
    from dataclasses import asdict

    boundary = transcript['boundary']
    ctx, chat, embed = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create([document_spec()], operation='generate', scope={})
    settings = tmp_path / 'settings.json'
    settings.write_text(json.dumps(asdict(ctx.settings)))
    ready = tmp_path / 'ready'
    release = threading.Event()
    if boundary in {'during_request', 'during_embedding'}:

        def pause_server():
            ready.touch()
            release.wait(20)

        (chat if boundary == 'during_request' else embed).barrier = pause_server
    code = """
import asyncio,json,signal,sys
from pathlib import Path
from model_config import ModelSettings,ChatSettings,EmbeddingSettings,ExecutionSettings
from inference import InferenceContext
from docgen.spool import SpoolStore,LocalExecutor
from library import Library
raw=json.loads(Path(sys.argv[1]).read_text())
ctx=InferenceContext(ModelSettings(ChatSettings(**raw['chat']),EmbeddingSettings(**raw['embeddings']),ExecutionSettings(**raw['execution'])),Path(sys.argv[2]),Path(sys.argv[3]))
boundary=sys.argv[5]
def stop():
    Path(sys.argv[6]).touch()
    signal.pause()
async def run():
    with ctx.write_session():
        lib=ctx.open_library(operation='write')
        store=SpoolStore(ctx,lib)
        original=store.update_job
        def update(job_id,**values):
            original(job_id,**values)
            if (boundary=='before_request' and values.get('state')=='running') or (boundary=='after_response' and values.get('state')=='response_ready'):
                stop()
        store.update_job=update
        apply=Library.apply_prepared_document
        def applied(self,*args,**kwargs):
            result=apply(self,*args,**kwargs)
            if boundary=='after_commit':stop()
            return result
        Library.apply_prepared_document=applied
        await LocalExecutor(store).drive(sys.argv[4])
    await ctx.aclose()
asyncio.run(run())
"""  # noqa: Q001 - repository quote rule conflicts with the formatter.
    process = subprocess.Popen(  # noqa: S603 - fixed test program and temporary fixture paths.
        [
            sys.executable,
            '-c',
            code,
            str(settings),
            str(ctx.db_path),
            str(ctx.staleness_db_path),
            run_id,
            boundary,
            str(ready),
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(500):
            if ready.exists() or process.poll() is not None:
                break
            time.sleep(0.02)
        assert ready.exists(), process.communicate(timeout=2)
        from library.locking import LibraryBusy

        with pytest.raises(LibraryBusy), ctx.write_session():
            pass
        process.kill()
        process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        release.set()
        chat.barrier = embed.barrier = None
    before = store.jobs(run_id)[0]
    saved_chats = len(
        [route for route, _, _ in chat.requests if route.endswith('/chat/completions')]
    )
    assert before['state'] == transcript['persisted_state']
    assert bool(before['response_text']) == transcript['saved_response']
    assert saved_chats == transcript['chat_calls_before_resume']

    async def resume():
        with ctx.write_session():
            result = await LocalExecutor(store).drive(run_id)
            assert result['state'] == 'completed' and result['applied'] == 1
        await ctx.aclose()

    asyncio.run(resume())
    assert (
        len([route for route, _, _ in chat.requests if route.endswith('/chat/completions')])
        == saved_chats + transcript['additional_chat_calls_on_resume']
    )
    if boundary in {'after_response', 'during_embedding', 'after_commit'}:
        assert before['response_text']
        assert (
            len([route for route, _, _ in chat.requests if route.endswith('/chat/completions')])
            == saved_chats
        )
