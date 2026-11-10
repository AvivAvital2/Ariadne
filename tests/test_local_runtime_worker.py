"""Actual child processes must not outlive their controlling application pipe."""

import asyncio
import os
import sys

import pytest


@pytest.mark.parametrize('shutdown', ['pipe_closed', 'terminate'])
@pytest.mark.asyncio
async def test_runtime_worker_reaps_server_when_parent_disappears(shutdown):
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'local_runtime_worker', '--', sys.executable, '-c',
        'import os,time; print(os.getpid(), flush=True); time.sleep(60)',
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        line = await asyncio.wait_for(process.stdout.readline(), timeout=3)
        assert line.strip().isdigit(), (await process.stderr.read()).decode()
        child_pid = int(line)
        if shutdown == 'pipe_closed':
            process.stdin.close()
        else:
            process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
