"""aiohttp backend for the Ariadne onboarding wizard.

Serves the single-page wizard and bridges its requests to the Ariadne MCP
server. No business logic lives here — each ``/api`` endpoint forwards 1:1 to
an MCP tool and returns the structured result as JSON, so the browser drives
Ariadne entirely "over MCP".
"""
from __future__ import annotations

import sys
import asyncio
import uuid
from collections import deque
from contextlib import AsyncExitStack
from pathlib import Path

from aiohttp import web

from web.mcp_client import (
    MCPCallError,
    connect_stdio,
    stdio_server_params,
)

STATIC_DIR = Path(__file__).resolve().parent / 'static'
REPO_ROOT = Path(__file__).resolve().parent.parent

# Onboard build jobs keep a bounded tail of their subprocess output: the drain
# buffer caps at _JOB_OUTPUT_LINES, and status reports the last _JOB_TAIL_LINES.
_JOB_OUTPUT_LINES = 400
_JOB_TAIL_LINES = 12

# Browser endpoint → MCP tool. Onboarding "Generate" (ariadne_onboard) is
# added once that tool + its progress stream land.
TOOL_ROUTES = {
    '/api/source_add': 'ariadne_source_add',
    '/api/sources': 'ariadne_list_sources',
    '/api/discover': 'ariadne_discover',
    '/api/estimate': 'ariadne_estimate',
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
        status, body = await dispatch_tool(request.app['bridge'], tool, args)
        return web.json_response(body, status=status)

    return handler


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


def _onboard_command(body):
    """The ``ariadne onboard`` argv for a build request, or ``None`` when no
    source is given. ``--approve`` runs it non-interactively; scope + the
    per-format excludes are already persisted to ariadne.yaml."""
    source = body.get('source')
    if not source:
        return None
    args = ['onboard', '--source', str(source), '--approve']
    model = body.get('model')
    if model:
        args += ['--model', str(model)]
    args.append('--batch' if body.get('batch') else '--live')
    types = body.get('types')
    if types:
        joined = ','.join(types) if isinstance(types, (list, tuple)) else str(types)
        args += ['--types', joined]
    return args


async def _drain_job(job):
    """Stream the subprocess output into the job's capped buffer, then set the
    terminal status from its exit code."""
    proc = job['proc']
    if proc.stdout is not None:
        async for raw in proc.stdout:
            line = raw.decode('utf-8', 'replace').rstrip()
            if line:
                job['output'].append(line)
    job['returncode'] = await proc.wait()
    job['status'] = 'done' if job['returncode'] == 0 else 'error'


async def _onboard_start(request):
    """Spawn ``ariadne onboard`` as a detached background job and return its id;
    the browser stores it and polls ``/api/onboard/status``."""
    body = await _read_json(request)
    args = _onboard_command(body)
    if args is None:
        return web.json_response({'error': 'source is required'}, status=400)
    proc = await asyncio.create_subprocess_exec(
        'uv', 'run', 'ariadne', *args,
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    job_id = uuid.uuid4().hex[:12]
    job = {'id': job_id, 'proc': proc, 'status': 'running',
           'output': deque(maxlen=_JOB_OUTPUT_LINES), 'returncode': None}
    request.app['jobs'][job_id] = job
    job['task'] = asyncio.create_task(_drain_job(job))   # job holds the ref alive
    return web.json_response({'job_id': job_id, 'status': 'running'})


async def _onboard_status(request):
    """Report a build job's status + a tail of its output for the runline."""
    job_id = request.query.get('job_id')
    job = request.app['jobs'].get(job_id) if job_id else None
    if job is None:
        return web.json_response({'status': 'unknown'}, status=404)
    return web.json_response({
        'status': job['status'],
        'returncode': job['returncode'],
        'tail': list(job['output'])[-_JOB_TAIL_LINES:],
    })


def make_app(bridge=None, *, server_params=None) -> web.Application:
    """Build the onboarding web app.

    Pass ``bridge`` to inject an MCP bridge (tests). Otherwise the app spawns
    the MCP server over stdio on startup using ``server_params`` (or the
    default subprocess params) and tears it down on cleanup.
    """
    app = web.Application()
    app['jobs'] = {}
    app.router.add_post('/api/onboard', _onboard_start)
    app.router.add_get('/api/onboard/status', _onboard_status)
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
        app.on_startup.append(_startup_connect)
        app.on_cleanup.append(_cleanup_disconnect)
    return app


async def _startup_connect(app: web.Application) -> None:
    stack = AsyncExitStack()
    params = app['_server_params'] or stdio_server_params()
    app['bridge'] = await connect_stdio(stack, params)
    app['_mcp_stack'] = stack


async def _cleanup_disconnect(app: web.Application) -> None:
    stack = app.get('_mcp_stack')
    if stack is not None:
        await stack.aclose()


def serve(host: str = '127.0.0.1', port: int = 8765, *, config_path: str | None = None) -> None:
    """Run the onboarding server, spawning the MCP server as a stdio child."""
    app = make_app(server_params=stdio_server_params(config_path))
    web.run_app(app, host=host, port=port)
