"""aiohttp backend for the Ariadne onboarding wizard.

Serves the single-page wizard and bridges its requests to the Ariadne MCP
server. No business logic lives here — each ``/api`` endpoint forwards 1:1 to
an MCP tool and returns the structured result as JSON, so the browser drives
Ariadne entirely "over MCP".
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from aiohttp import web

from local_models import MAC_PROFILE_IDS
from web.jobs import (
    ProjectBusyError,
    check_managed_admission,
    check_project_ready,
    finish_callers,
    launch_job,
    project_admission,
    run_tool,
)
from web.local_runs import register_routes as register_run_routes
from web.mcp_client import (
    MCPCallError,
    connect_http,
    connect_stdio,
    stdio_server_params,
)

STATIC_DIR = Path(__file__).resolve().parent / 'static'

# Browser endpoint → MCP tool. Onboarding "Generate" (ariadne_onboard) is
# added once that tool + its progress stream land.
TOOL_ROUTES = {
    '/api/source_add': 'ariadne_source_add',
    '/api/sources': 'ariadne_list_sources',
    '/api/discover': 'ariadne_discover',
    '/api/estimate': 'ariadne_estimate',
    '/api/ask': 'ariadne_ask',
    '/api/search': 'ariadne_search',
}


async def dispatch_tool(bridge, tool: str, args: dict | None) -> tuple[int, dict]:
    """Forward ``(tool, args)`` to the MCP bridge → ``(status, json_body)``.

    Pure of aiohttp so the bridge wiring is testable without binding a socket.
    A tool error maps to HTTP 400 with an ``error`` field.
    """
    try:
        data = await bridge.call(tool, args or {})
    except MCPCallError as exc:
        return 400, {'error': str(exc)}
    return 200, data


async def _read_json(request: web.Request) -> dict:
    """Best-effort parse of a JSON request body → dict ({} when absent/invalid)."""
    try:
        return await request.json() if request.can_read_body else {}
    except Exception:
        return {}


def _make_tool_handler(tool: str):
    async def handler(request: web.Request) -> web.Response:
        args = await _read_json(request)
        project_id = args.pop('project_id', None)
        choice = args.pop('onboarding_model', None)
        try:
            if choice is not None:
                if tool != 'ariadne_estimate' or project_id is not None or args.get('model'):
                    raise ValueError('Compare model pairs in Preview before starting a build')
                store = request.app.get('projects')
                if store is None:
                    raise ValueError('Project onboarding is not configured on this server')
                async with store.preview(choice, args.get('source')) as preview:
                    status, body = await dispatch_tool(preview, tool, args)
                return web.json_response(body, status=status)
            if project_id and tool in ('ariadne_ask', 'ariadne_search'):
                check_project_ready(request.app, project_id)
            bridge = await _project_bridge(request.app, project_id, args)
            if project_id and tool in ('ariadne_ask', 'ariadne_search'):
                check_project_ready(request.app, project_id)
        except (ValueError, KeyError) as exc:
            return web.json_response({'error': str(exc)}, status=400)
        status, body = await dispatch_tool(bridge, tool, args)
        if project_id and status == 200:
            body = {**body, 'project_id': project_id}
        return web.json_response(body, status=status)

    return handler


async def _project_bridge(app, project_id, args):
    if project_id is None:
        return app['bridge']
    store = app.get('projects')
    if store is None:
        raise ValueError('Project onboarding is not configured on this server')
    project = store.describe(project_id)
    if args.get('model') and args['model'] != project['model']:
        raise ValueError('This project uses the model pair saved during onboarding')
    for field in ('source', 'name'):
        if args.get(field) and args[field] != project['source']:
            raise ValueError('Source does not belong to this project')
    if args.get('path'):
        from config import Config

        cfg = Config(store.path(project_id))
        if Path(args['path']).expanduser().resolve() != cfg.resolve_source(project['source']).resolve():
            raise ValueError('Source path differs from the project saved during onboarding')
    return await store.bridge(project_id)


async def _onboarding_models(request):
    store = request.app.get('projects')
    return web.json_response({'models': store.choices() if store else [], 'default': 'configured'})


async def _projects(request):
    store = request.app.get('projects')
    return web.json_response({'projects': store.list_projects() if store else []})


def _runtime_status(app: web.Application) -> dict:
    control = app['runtime_actions']
    status = app['projects'].runtime.status()
    task = control['task']
    status['busy'] = task is not None and not task.done()
    if control.get('stopping'):
        status['phase'] = 'stopping'
    if control['error']:
        status.update(phase='error', error=control['error'])
    return status


async def _local_models(request: web.Request) -> web.Response:
    if request.app.get('projects') is None:
        return web.json_response({'error': 'Project onboarding is not configured'}, status=404)
    return web.json_response(_runtime_status(request.app))


async def _local_model_action(request: web.Request) -> web.Response:
    """Start a fixed, owned operation; model paths and commands never come from HTTP."""
    origin = request.headers.get('Origin')
    if origin and origin != f'{request.scheme}://{request.host}':
        return web.json_response({'error': 'Cross-origin runtime action rejected'}, status=403)
    if request.app.get('projects') is None:
        return web.json_response({'error': 'Project onboarding is not configured'}, status=404)
    body = await _read_json(request)
    if (not isinstance(body, dict) or set(body) not in ({'action'}, {'action', 'profile_id'})
            or body['action'] not in ('download', 'start', 'stop')):
        return web.json_response({'error': 'Expected one action: download, start or stop'}, status=400)
    action = body['action']
    if 'profile_id' in body and (action == 'stop' or not isinstance(body['profile_id'], str)
                                 or body['profile_id'] not in MAC_PROFILE_IDS):
        return web.json_response({'error': 'Select a registered local model profile for download or start'}, status=400)
    status = _runtime_status(request.app)
    if status.get('platform_error') and action != 'stop':
        return web.json_response({'error': status['platform_error']}, status=400)
    if status['busy'] and action != 'stop':
        return web.json_response({'error': 'A local model operation is already running'}, status=409)
    control = request.app['runtime_actions']
    previous = control['task']
    runtime = request.app['projects'].runtime
    control['error'] = None
    control['stopping'] = action == 'stop'
    if action == 'stop' and previous is not None:
        previous.cancel()

    async def run() -> None:
        try:
            if action == 'stop' and previous is not None:
                await asyncio.gather(previous, return_exceptions=True)
            if action == 'stop':
                jobs = [job for job in request.app['jobs'].values()
                        if job.get('managed_local') and not job['task'].done()]
                for job in jobs:
                    await job['queue'].put({'type': 'interrupted', 'error': 'Models stopped; saved work can be resumed.'})
                await request.app['projects'].interrupt_managed()
                await finish_callers(jobs, 'interrupted')
            options = {'profile_id': body['profile_id']} if 'profile_id' in body else {}
            await {'download': runtime.setup, 'start': runtime.start, 'stop': runtime.stop}[action](**options)
        except Exception as exc:  # noqa: BLE001 — expose background operation failure to the UI
            control['error'] = str(exc)
        finally:
            if control['task'] is asyncio.current_task():
                control['stopping'] = False

    control['task'] = asyncio.create_task(run())
    return web.json_response(_runtime_status(request.app), status=202)


async def _index(request: web.Request) -> web.Response:
    page = STATIC_DIR / 'onboarding.html'
    if not page.exists():
        return web.Response(text='onboarding.html not found', status=404)
    return web.FileResponse(page)


def list_dirs(path: str | None) -> dict:
    """List immediate subdirectories of ``path`` for the "Browse…" picker.

    The web server runs locally, so it can read the user's filesystem — this
    is UI plumbing for choosing a source directory, not an Ariadne knowledge
    operation, hence a plain backend endpoint rather than an MCP tool.
    Defaults to (and falls back to) the home directory; dot-prefixed dirs are
    omitted. Returns ``{path, parent, dirs:[name, ...]}``.
    """
    base = Path(path).expanduser() if path else Path.home()
    if not base.is_dir():
        base = Path.home()
    base = base.resolve()
    try:
        dirs = sorted(
            (d.name for d in base.iterdir()
             if d.is_dir() and not d.name.startswith('.')),
            key=str.lower,
        )
    except (OSError, PermissionError):
        dirs = []
    parent = str(base.parent) if base.parent != base else None
    return {'path': str(base), 'parent': parent, 'dirs': dirs}


async def _browse(request: web.Request) -> web.Response:
    return web.json_response(list_dirs(request.query.get('path')))


def native_picker_command(start: str | None) -> list[str] | None:
    """The OS-native folder-picker command for this platform, or None if
    we don't know one. ``start`` (when given) is the dialog's initial dir.

    Local-only: ``ariadne serve`` runs on the user's machine, so the dialog
    appears on their desktop and returns a real absolute path — which the
    browser's own pickers deliberately withhold.
    """
    if sys.platform == 'darwin':
        loc = f' default location (POSIX file "{start}")' if start else ''
        script = (
            'POSIX path of (choose folder with prompt '
            f'"Select the source directory"{loc})'
        )
        return ['osascript', '-e', script]
    if sys.platform.startswith('linux'):
        cmd = ['zenity', '--file-selection', '--directory',
               '--title=Select the source directory']
        if start:
            cmd.append(f'--filename={start.rstrip("/")}/')
        return cmd
    if sys.platform == 'win32':
        ps = (
            'Add-Type -AssemblyName System.Windows.Forms;'
            '$d=New-Object System.Windows.Forms.FolderBrowserDialog;'
            'if($d.ShowDialog() -eq "OK"){$d.SelectedPath}'
        )
        return ['powershell', '-NoProfile', '-Command', ps]
    return None


async def pick_folder(start: str | None) -> dict:
    """Open the native OS folder dialog and return the chosen path.

    Returns ``{path}`` on selection, ``{cancelled: True}`` if the user
    dismissed the dialog, or ``{unavailable: True}`` when no native picker
    is available (headless / unknown platform / tool not installed) — the
    frontend then falls back to the in-browser directory browser.
    """
    safe_start = start if (start and Path(start).expanduser().is_dir()) else None
    cmd = native_picker_command(safe_start)
    if cmd is None:
        return {'unavailable': True}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _err = await proc.communicate()
    except (FileNotFoundError, OSError):
        return {'unavailable': True}  # picker tool not installed
    path = (out or b'').decode().strip().rstrip('/')
    if proc.returncode != 0 or not path:
        return {'cancelled': True}  # user dismissed the dialog
    return {'path': path}


async def _pick_folder(request: web.Request) -> web.Response:
    return web.json_response(await pick_folder(request.query.get('start')))


def _onboard_tool_args(body: dict) -> dict:
    """Map the browser build request → ``ariadne_onboard`` tool arguments.

    Only the fields the user chose are forwarded; the tool applies its own
    defaults for anything omitted (model, doc types, live/batch, concurrency).
    """
    args: dict = {'source': str(body['source'])}
    if body.get('model'):
        args['model'] = str(body['model'])
    if body.get('batch'):
        args['batch'] = True
    types = body.get('types')
    if types:
        args['doc_types'] = (
            list(types) if isinstance(types, (list, tuple)) else [str(types)])
    if body.get('concurrency') is not None:
        args['concurrency'] = int(body['concurrency'])
    return args


async def _run_onboard(bridge, args: dict, queue: asyncio.Queue) -> None:
    """Run ``ariadne_onboard`` over MCP, relaying its progress notifications
    and the terminal result into ``queue`` as SSE-shaped events.

    Emits ``{'type': 'progress', current, total, message}`` per phase, then a
    single ``{'type': 'done', 'result': {...}}`` or ``{'type': 'error', ...}``.
    The terminal event always fires, so the SSE stream can never hang.
    """
    await run_tool(bridge, 'ariadne_onboard', args, queue)


async def _onboard_start(request: web.Request) -> web.Response:
    """Start an ``ariadne_onboard`` build as a background MCP tool call and
    return its job id; the browser subscribes to progress over
    ``GET /api/onboard/events`` (SSE)."""
    body = await _read_json(request)
    if not body.get('source'):
        return web.json_response({'error': 'source is required'}, status=400)
    try:
        project_id = body.get('project_id')
        if body.get('onboarding_model'):
            if project_id is not None or body.get('model'):
                raise ValueError('Start with the model pair selected in Preview')
            store = request.app.get('projects')
            if store is None:
                raise ValueError('Project onboarding is not configured on this server')
            check_managed_admission(request.app, managed=body['onboarding_model'] in MAC_PROFILE_IDS)
            project_id = store.project_id_for(body['onboarding_model'], body['source'])
        with project_admission(request.app, project_id):
            if body.get('onboarding_model'):
                project = await store.start(body['onboarding_model'], body['source'],
                                            themes_enabled=body.get('themes'))
                project_id = project['id']
            check_project_ready(request.app, project_id)
            bridge = await _project_bridge(request.app, project_id, body)
            check_project_ready(request.app, project_id)
            job_id = uuid.uuid4().hex[:12]
            project = request.app['projects'].describe(project_id) if project_id else None
            args = _onboard_tool_args(body)
            if project and project['provider'] == 'local':
                args['client_run_id'] = job_id
            launch_job(request.app, bridge, 'ariadne_onboard', args, project=project,
                       job_id=job_id, profile_id=body.get('onboarding_model'))
    except ProjectBusyError as exc:
        return web.json_response({'error': str(exc)}, status=409)
    except ValueError as exc:
        return web.json_response({'error': str(exc)}, status=400)
    return web.json_response({'job_id': job_id, 'status': 'running',
                              **({'project_id': project_id} if project_id else {})})


async def _onboard_events(request: web.Request) -> web.StreamResponse:
    """Server-sent-events stream of a build job's progress + terminal result."""
    job_id = request.query.get('job_id')
    job = request.app['jobs'].get(job_id) if job_id else None
    if job is None:
        return web.json_response({'error': 'unknown job_id'}, status=404)
    try:
        sequence = int(request.headers.get('Last-Event-ID', '0'))
    except ValueError:
        return web.json_response({'error': 'Invalid event id'}, status=400)
    resp = web.StreamResponse(headers={
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    })
    await resp.prepare(request)
    try:
        while True:
            for sequence, event in await job['queue'].read_after(sequence):
                await resp.write(f'id: {sequence}\ndata: {json.dumps(event)}\n\n'.encode())
                if event['type'] in ('done', 'error', 'cancelled', 'interrupted'):
                    return resp
    except (ConnectionResetError, ConnectionError):
        pass  # A browser disconnect does not interrupt the owned execution.
    return resp


def _feedback_call(body: dict) -> tuple[str, dict]:
    """Route a feedback POST to the right tool: helpful → log_hit, else log_miss.

    Carries the ask/search ``event_id``; an optional free-text note is forwarded
    as ``feedback`` when present (log_miss defaults it to '' server-side).
    """
    args: dict = {'event_id': body.get('event_id')}
    if body.get('feedback'):
        args['feedback'] = str(body['feedback'])
    tool = 'ariadne_log_hit' if body.get('helpful') else 'ariadne_log_miss'
    return tool, args


async def _feedback(request: web.Request) -> web.Response:
    """Record 👍/👎 on an answer via its event_id (ariadne_log_hit / log_miss)."""
    body = await _read_json(request)
    tool, args = _feedback_call(body)
    try:
        bridge = await _project_bridge(request.app, body.get('project_id'), {})
    except ValueError as exc:
        return web.json_response({'error': str(exc)}, status=400)
    status, resp = await dispatch_tool(bridge, tool, args)
    return web.json_response(resp, status=status)


def make_app(bridge=None, *, server_params=None, mcp_url=None, projects=None) -> web.Application:
    """Build the onboarding web app.

    Pass ``bridge`` to inject an MCP bridge (tests). Otherwise the app connects
    on startup — to a remote ``ariadne mcp --http`` server when ``mcp_url`` is
    given, else by spawning one over stdio (from ``server_params`` or the
    default) — and tears it down on cleanup.
    """
    app = web.Application()
    app['jobs'] = {}
    app['project_admissions'] = set()
    app['projects'] = projects
    app['runtime_actions'] = {'task': None, 'error': None, 'stopping': False}
    register_run_routes(app)
    app.router.add_get('/api/onboarding-models', _onboarding_models)
    app.router.add_get('/api/projects', _projects)
    app.router.add_get('/api/local-models', _local_models)
    app.router.add_post('/api/local-models', _local_model_action)
    app.on_cleanup.append(_cleanup_projects)
    app.router.add_post('/api/onboard', _onboard_start)
    app.router.add_get('/api/onboard/events', _onboard_events)
    app.router.add_post('/api/feedback', _feedback)
    app.router.add_get('/', _index)
    app.router.add_get('/api/browse', _browse)
    app.router.add_get('/api/pick-folder', _pick_folder)
    for path, tool in TOOL_ROUTES.items():
        app.router.add_post(path, _make_tool_handler(tool))
    if STATIC_DIR.exists():
        app.router.add_static('/static/', STATIC_DIR)

    if bridge is not None:
        app['bridge'] = bridge
    else:
        app['_server_params'] = server_params
        app['_mcp_url'] = mcp_url
        app.on_startup.append(_startup_connect)
        app.on_cleanup.append(_cleanup_disconnect)
    return app


async def _cleanup_projects(app):
    tasks = [job['task'] for job in app['jobs'].values() if 'task' in job]
    if task := app['runtime_actions']['task']:
        tasks.append(task)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if app.get('projects') is not None:
        await app['projects'].close()


async def _startup_connect(app: web.Application) -> None:
    """Open the MCP connection in one long-lived task (see ``_mcp_lifecycle``).

    aiohttp runs ``on_startup`` and ``on_cleanup`` in different tasks, but the
    MCP client opens an anyio cancel scope that must be exited in the task it
    was entered in. So the whole connection lifetime is owned by a single
    background task instead of being entered here and closed from the cleanup
    task — otherwise shutdown raises "Attempted to exit cancel scope in a
    different task than it was entered in" and the server can't stop cleanly.
    """
    app['_mcp_ready'] = asyncio.Event()
    app['_mcp_stop'] = asyncio.Event()
    app['_mcp_task'] = asyncio.create_task(_mcp_lifecycle(app))
    await app['_mcp_ready'].wait()
    error = app.get('_mcp_error')
    if error is not None:
        raise error


async def _mcp_lifecycle(app: web.Application) -> None:
    """Own the MCP connection for the server's lifetime, in this one task.

    Opens the bridge, publishes it on ``app['bridge']``, then parks until
    shutdown — so the connect context managers are entered and exited in the
    same task, never across the aiohttp startup/cleanup task boundary.
    """
    try:
        async with AsyncExitStack() as stack:
            url = app.get('_mcp_url')
            if url:
                app['bridge'] = await connect_http(stack, url)
            else:
                params = app['_server_params'] or stdio_server_params()
                app['bridge'] = await connect_stdio(stack, params)
            app['_mcp_ready'].set()
            await app['_mcp_stop'].wait()
    except Exception as exc:  # surface a startup failure to _startup_connect
        app['_mcp_error'] = exc
        app['_mcp_ready'].set()


async def _cleanup_disconnect(app: web.Application) -> None:
    """Signal the lifecycle task to close the MCP connection, then wait for it."""
    stop = app.get('_mcp_stop')
    task = app.get('_mcp_task')
    if stop is not None:
        stop.set()
    if task is not None:
        await task


def serve(host: str = '127.0.0.1', port: int = 8765, *,
          config_path: str | None = None, mcp_url: str | None = None) -> None:
    """Run the onboarding server.

    Connects to a remote ``ariadne mcp --http`` server when ``mcp_url`` is
    given; otherwise spawns the MCP server as a stdio child.
    """
    from config import Config
    from web.projects import ProjectStore

    projects = ProjectStore(Config(Path(config_path) if config_path else None))
    if mcp_url:
        app = make_app(mcp_url=mcp_url, projects=projects)
    else:
        app = make_app(server_params=stdio_server_params(config_path), projects=projects)
    web.run_app(app, host=host, port=port)
