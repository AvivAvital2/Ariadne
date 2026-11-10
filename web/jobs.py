"""Owned browser executions and replayable progress; durable work lives in SpoolStore."""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

from aiohttp import web

from web.mcp_client import MCPBridge

MAX_COMPLETED_JOBS = 64


class ProjectBusyError(ValueError):
    """An owned project execution or its admission is already in progress."""


@contextmanager
def project_admission(app: web.Application, project_id: str | None, *,
                      allow_active: bool = False) -> Iterator[None]:
    """Reserve a project across awaited setup, before any session is replaced."""
    if project_id is None:
        yield
        return
    admissions = app['project_admissions']
    if project_id in admissions or (not allow_active and active_jobs(app, project_id)):
        raise ProjectBusyError('A build is already running for this project')
    admissions.add(project_id)
    try:
        yield
    finally:
        admissions.remove(project_id)


def check_managed_admission(app: web.Application, *, managed: bool) -> None:
    if managed and app.get('runtime_actions', {}).get('stopping'):
        raise ValueError('Local models are stopping; start them before continuing')


def check_project_ready(app: web.Application, project_id: str | None) -> None:
    """Keep managed inference closed during worker and model shutdown."""
    store = app.get('projects')
    if project_id is None or store is None:
        return
    check_managed_admission(app, managed=bool(store.describe(project_id).get('managed_local')))
    store.check_runtime_ready(project_id)


class JobEvents:
    """Keep recent progress and the terminal result for every reconnecting reader."""

    def __init__(self) -> None:
        self.events = deque(maxlen=128)
        self.sequence = 0
        self.terminal = None
        self.changed = asyncio.Condition()

    async def put(self, event: dict) -> None:
        async with self.changed:
            if self.terminal is not None:
                return
            self.sequence += 1
            self.events.append((self.sequence, event))
            if event['type'] in ('done', 'error', 'cancelled', 'interrupted'):
                self.terminal = event
            self.changed.notify_all()

    async def read_after(self, sequence: int) -> list[tuple[int, dict]]:
        async with self.changed:
            await self.changed.wait_for(lambda: self.sequence > sequence or self.terminal is not None)
            return [(number, event) for number, event in self.events if number > sequence] or [self.events[-1]]


async def run_tool(bridge: MCPBridge, tool: str, args: dict, events: JobEvents | asyncio.Queue) -> None:
    async def progress(current: float, total: float | None, message: str | None) -> None:
        await events.put({'type': 'progress', 'current': current, 'total': total, 'message': message})

    try:
        result = await bridge.call(tool, args, progress_callback=progress)
        await events.put({'type': 'done', 'result': result})
    except asyncio.CancelledError:
        await events.put({'type': 'interrupted', 'error': 'Execution interrupted; inspect saved work before resuming.'})
        raise
    except Exception as exc:  # noqa: BLE001 — terminal errors must reach every reconnecting browser
        await events.put({'type': 'error', 'error': str(exc)})


def launch_job(app: web.Application, bridge: MCPBridge, tool: str, args: dict, *, project: dict | None = None,
               job_id: str | None = None, profile_id: str | None = None) -> dict:
    job_id = job_id or uuid.uuid4().hex[:12]
    job = {'id': job_id, 'queue': JobEvents(), 'project_id': (project or {}).get('id'),
           'managed_local': (project or {}).get('managed_local', False),
           'source': args.get('source'), 'run_id': args.get('run_id'), 'profile_id': profile_id}
    app['jobs'][job_id] = job
    job['task'] = asyncio.create_task(run_tool(bridge, tool, args, job['queue']))
    job['task'].add_done_callback(lambda _task: _retain_completed(app, job))
    return job


def _retain_completed(app: web.Application, job: dict) -> None:
    """Bound SSE history; persisted runs remain discoverable through the run API."""
    job['finished_at'] = time.monotonic_ns()
    completed = sorted((entry for entry in app['jobs'].values() if 'finished_at' in entry),
                       key=lambda entry: entry['finished_at'])
    for old in completed[:-MAX_COMPLETED_JOBS]:
        app['jobs'].pop(old['id'], None)


def active_jobs(app: web.Application, project_id: str) -> list[dict]:
    return [job for job in app['jobs'].values()
            if job.get('project_id') == project_id and not job['task'].done()]


async def finish_callers(jobs: list[dict], kind: str) -> None:
    """Call only after the owned MCP execution has been interrupted and reaped."""
    for job in jobs:
        await job['queue'].put({'type': kind, 'error': 'Build cancelled.' if kind == 'cancelled'
                                else 'Build interrupted. Saved work can be resumed.'})
        job['task'].cancel()
    await asyncio.gather(*(job['task'] for job in jobs), return_exceptions=True)
