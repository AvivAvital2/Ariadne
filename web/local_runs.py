"""Project-bound browser controls for the existing durable local job engine."""
from __future__ import annotations

import asyncio

from aiohttp import web

from web.jobs import (
    ProjectBusyError,
    active_jobs,
    check_project_ready,
    finish_callers,
    launch_job,
    project_admission,
)
from web.mcp_client import MCPCallError
from web.projects import ProjectStore

TERMINAL = {'completed', 'cancelled', 'failed', 'obsolete'}


def project_store(app: web.Application, project_id: str | None) -> tuple[ProjectStore, dict]:
    store = app.get('projects')
    if store is None:
        raise ValueError('Project onboarding is not configured')
    project = store.describe(project_id)
    if project['provider'] != 'local':
        raise ValueError('Durable local run controls require a local-model project')
    return store, project


def _control_body(body: object) -> dict:
    if (not isinstance(body, dict) or set(body) != {'project_id', 'run_id', 'action'}
            or body['action'] not in ('cancel', 'resume') or not isinstance(body['run_id'], str)):
        raise ValueError('Expected project_id, run_id and action cancel or resume')
    return body


def _check_resumable(runs: list[dict], run_id: str) -> None:
    if len(runs) != 1 or runs[0]['run_id'] != run_id:
        raise ValueError('Unknown local run')
    if runs[0]['state'] in TERMINAL:
        raise ValueError('This run is terminal; start a new build to do more work')


async def _cancel_owned_run(app: web.Application, store: ProjectStore, project: dict, args: dict) -> dict:
    bridge = await store.bridge(project['id'])
    # Persist cancellation before reaping a positively matched owned worker.
    result = await bridge.call('ariadne_local_run_cancel', args)
    matching = [job for job in active_jobs(app, project['id'])
                if job.get('run_id') == args['run_id'] or job['id'] == result['run'].get('client_run_id')]
    if result['run']['state'] == 'cancelled' and matching:
        with project_admission(app, project['id'], allow_active=True):
            for job in matching:
                await job['queue'].put({'type': 'cancelled', 'error': 'Build cancelled.'})
            await store.interrupt(project['id'])
            await finish_callers(matching, 'cancelled')
    return result


async def list_runs(request: web.Request) -> web.Response:
    try:
        store, project = project_store(request.app, request.query.get('project_id'))
        bridge = await store.bridge(project['id'])
        result = await bridge.call('ariadne_local_runs', {'source': project['source']})
        for run in result['runs']:
            matching = [job for job in request.app['jobs'].values()
                        if job.get('project_id') == project['id'] and (
                            job.get('run_id') == run['run_id'] or job['id'] == run.get('client_run_id'))]
            for job in matching:
                job['run_id'] = run['run_id']
            run['active_here'] = any(not job['task'].done() for job in matching)
            if matching and (terminal := matching[-1]['queue'].terminal) and terminal['type'] == 'error':
                run['execution_error'] = terminal['error']
        return web.json_response(result)
    except (ValueError, MCPCallError) as exc:
        return web.json_response({'error': str(exc)}, status=400)


async def control_run(request: web.Request) -> web.Response:
    origin = request.headers.get('Origin')
    if origin and origin != f'{request.scheme}://{request.host}':
        return web.json_response({'error': 'Cross-origin run control rejected'}, status=403)
    try:
        body = _control_body(await request.json())
        store, project = project_store(request.app, body['project_id'])
        args = {'source': project['source'], 'run_id': body['run_id']}
        if body['action'] == 'cancel':
            cleanup = asyncio.create_task(_cancel_owned_run(request.app, store, project, args))
            try:
                result = await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # Disconnecting the control request must not strand its browser job.
                await asyncio.shield(cleanup)
                raise
            return web.json_response(result)
        with project_admission(request.app, project['id']):
            check_project_ready(request.app, project['id'])
            bridge = await store.bridge(project['id'])
            check_project_ready(request.app, project['id'])
            runs = (await bridge.call('ariadne_local_runs', args))['runs']
            _check_resumable(runs, body['run_id'])
            check_project_ready(request.app, project['id'])
            job = launch_job(request.app, bridge, 'ariadne_local_run_resume', args, project=project)
        return web.json_response({'job_id': job['id'], 'run_id': body['run_id'],
                                  'project_id': project['id']}, status=202)
    except ProjectBusyError as exc:
        return web.json_response({'error': str(exc)}, status=409)
    except (ValueError, MCPCallError) as exc:
        return web.json_response({'error': str(exc)}, status=400)


def register_routes(app: web.Application) -> None:
    app.router.add_get('/api/local-runs', list_runs)
    app.router.add_post('/api/local-runs', control_run)
