"""Actual child processes must not outlive their controlling application pipe."""

import asyncio
import os
import sys

import pytest


@pytest.mark.parametrize(('shutdown', 'stubborn'), [
    ('pipe_closed', False), ('terminate', False),
    ('pipe_closed', True), ('terminate', True), ('child_exit', False),
])
@pytest.mark.asyncio
async def test_runtime_worker_reaps_server_when_parent_disappears(shutdown, stubborn):
    command = 'import os,signal,sys,time; '
    if stubborn:
        command += 'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
    command += 'print(os.getpid(), flush=True); '
    command += 'sys.exit(7)' if shutdown == 'child_exit' else 'time.sleep(60)'
    process = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'local_runtime_worker', '--', sys.executable, '-c',
        command,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        line = await asyncio.wait_for(process.stdout.readline(), timeout=3)
        assert line.strip().isdigit(), (await process.stderr.read()).decode()
        child_pid = int(line)
        if shutdown == 'pipe_closed':
            process.stdin.write(b'still-owned')
            await process.stdin.drain()
            process.stdin.close()
        elif shutdown == 'terminate':
            process.terminate()
        await asyncio.wait_for(process.wait(), timeout=5)
        assert process.returncode == (7 if shutdown == 'child_exit' else 0)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
