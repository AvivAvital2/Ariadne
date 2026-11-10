"""Explicit compatibility smoke for supported user-managed model pairs."""

from __future__ import annotations

import asyncio
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import urlsplit

import httpx
import numpy as np
from aiohttp import web

from docgen.llm.local import LocalInferenceError, check_endpoints
from inference import InferenceContext
from local_models import (
    GEMMA_CHAT_MODEL,
    GEMMA_PROFILE_ID,
    MAC_ACCURACY_PROFILE_ID,
    managed_model_profile,
)
from model_config import identity_hash, normalize_url, resolve_models, tokenize_url

PROBE_TIMEOUT = 120
PROBE_OUTPUT_TOKENS = 2048
MAX_URL_LENGTH = 2048
MIN_REQUEST_TIMEOUT = 600
MAX_REQUEST_TIMEOUT = 14400
_FIELDS = {'chat_url', 'embedding_url', 'chat_api_key_env', 'embedding_api_key_env',
           'request_timeout_seconds', 'model_profile'}
_CHECK_LOCK = web.AppKey('external_model_check_lock', asyncio.Lock)
_CHECK_TASKS = web.AppKey('external_model_check_tasks', set)


def build_profile(body: dict) -> dict:
    """Take endpoint locations and credential references, never arbitrary model options."""
    if not isinstance(body, dict) or set(body) - _FIELDS:
        raise ValueError('Provide server URLs, optional credential variable names and request timeout only')
    selected = body.get('model_profile', MAC_ACCURACY_PROFILE_ID)
    if selected not in (MAC_ACCURACY_PROFILE_ID, GEMMA_PROFILE_ID):
        raise ValueError('Select a supported generation and embedding pair')
    profile = managed_model_profile(selected)
    if 'request_timeout_seconds' in body:
        seconds = body['request_timeout_seconds']
        if type(seconds) is not int or not MIN_REQUEST_TIMEOUT <= seconds <= MAX_REQUEST_TIMEOUT:
            raise ValueError('Request timeout must be an integer from 600 to 14400 seconds')
        for section in ('llm', 'embeddings'):
            profile[section]['timeout_seconds'] = seconds
    for prefix, section in (('chat', 'llm'), ('embedding', 'embeddings')):
        url = body.get(prefix + '_url')
        if not isinstance(url, str) or not url.strip() or len(url) > MAX_URL_LENGTH:
            raise ValueError(f'{prefix.title()} server URL is required')
        url = normalize_url(url.strip())
        # urlsplit defers validation of numeric/range port errors to this property.
        _port = urlsplit(url).port
        tokenize_url(url)
        credential = body.get(prefix + '_api_key_env')
        if credential == '':
            credential = None
        if credential is not None and (not isinstance(credential, str)
                                      or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}', credential) is None):
            raise ValueError('Credentials must be environment-variable names, not secret keys')
        profile[section].update(base_url=url, api_key_env=credential)
    resolve_models(profile, execution=True)
    return profile


def profile_id(profile: dict) -> str:
    """Deployment identity includes endpoints; artifact/library identity stays unchanged."""
    return 'external-' + identity_hash(profile)[:24]


def _validated_profile(profile: dict) -> dict:
    try:
        body = {
            'chat_url': profile['llm']['base_url'],
            'embedding_url': profile['embeddings']['base_url'],
            'chat_api_key_env': profile['llm'].get('api_key_env'),
            'embedding_api_key_env': profile['embeddings'].get('api_key_env'),
        }
        if profile['llm']['model'] == GEMMA_CHAT_MODEL:
            body['model_profile'] = GEMMA_PROFILE_ID
        # Legacy profiles omit this field and retain their original identity.
        if 'timeout_seconds' in profile['embeddings']:
            body['request_timeout_seconds'] = profile['embeddings']['timeout_seconds']
        expected = build_profile(body)
    except (KeyError, TypeError, AttributeError):
        raise ValueError('Invalid saved user-managed model profile') from None
    if expected != profile:
        raise ValueError('Saved user-managed model profile differs from the supported contract')
    return expected


def load_profiles(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('profiles'), dict):
            raise ValueError('Invalid user-managed profile registry')
        for key, profile in data['profiles'].items():
            if key != profile_id(_validated_profile(profile)):
                raise ValueError('Saved user-managed deployment identity differs')
        return data['profiles']
    except json.JSONDecodeError:
        raise ValueError('Cannot read the saved user-managed model profiles') from None


def save_profile(path: Path, profile: dict) -> str:
    profile = _validated_profile(profile)
    key = profile_id(profile)
    profiles = load_profiles(path)
    if profiles.get(key) == profile:
        return key
    profiles[key] = profile
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                prefix='.model-profiles-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({'version': 1, 'profiles': profiles}, stream, sort_keys=True, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return key


def _require_context(actual: int, required: int) -> None:
    if actual < required:
        raise ValueError('Server context is smaller than the saved profile')


async def probe_profile(profile: dict) -> dict:
    """No user library is opened; a successful probe is not factual qualification."""
    bounded = deepcopy(profile)
    bounded['execution'] = {**bounded.get('execution', {}), 'max_attempts': 1}
    settings = resolve_models(bounded, execution=True)
    settings.check_credentials()
    probe_timeout = bounded['embeddings'].get('timeout_seconds', PROBE_TIMEOUT)
    phase = 'Server readiness'
    try:
        async with asyncio.timeout(probe_timeout):
            readiness = await check_endpoints(settings)
            for name, required in (('chat', settings.chat.context_window),
                                   ('embeddings', settings.embeddings.max_input_tokens)):
                phase = f'{name.title()} context (requires at least {required} tokens)'
                _require_context(readiness[name]['server_context'], required)
            with TemporaryDirectory(prefix='ariadne-endpoint-check-') as temporary:
                directory = Path(temporary)
                context = InferenceContext(settings, directory / 'library.db', directory / 'staleness.db')
                try:
                    with context.write_session():
                        pass
                    phase = 'Chat'
                    await context.chat.call(
                        'This is a connection test. Reply briefly with READY.',
                        'Acknowledge this connection test.',
                        max_tokens=PROBE_OUTPUT_TOKENS, timeout=probe_timeout,
                    )
                    phase = 'Embedding'
                    vectors = await context.embeddings.embed_documents(['A connection test document.'])
                    vectors.append(await context.embeddings.embed_query('Find the connection test document.'))
                    if any(v.shape != (3072,) or v.dtype != np.float32
                           or not np.isclose(np.linalg.norm(v.astype(np.float64)), 1)
                           for v in vectors):
                        raise ValueError('Invalid stored vector contract')
                finally:
                    await context.aclose()
            return {
                'chat': {**readiness['chat'], 'completed': True,
                         'max_output_tokens': PROBE_OUTPUT_TOKENS},
                'embeddings': {**readiness['embeddings'], 'dimensions': 3072,
                               'native_dimensions': 4096, 'normalized': True,
                               'document_and_query': True},
            }
    except TimeoutError:
        raise ValueError(f'{phase} check timed out; check the server and try again') from None
    except (LocalInferenceError, ValueError, httpx.HTTPError, httpx.InvalidURL):
        # Upstream content and credential-bearing error details never go to the UI.
        raise ValueError(f'{phase} check failed; verify the supported models, server APIs and credentials') from None


async def _test_and_save(request: web.Request) -> web.Response:
    origin = request.headers.get('Origin')
    if origin and origin != f'{request.scheme}://{request.host}':
        return web.json_response({'error': 'Cross-origin model configuration rejected'}, status=403)
    store = request.app.get('projects')
    if store is None:
        return web.json_response({'error': 'Project onboarding is not configured'}, status=404)
    if request.content_type != 'application/json':
        return web.json_response({'error': 'Send model configuration as JSON'}, status=400)
    try:
        profile = build_profile(await request.json())
    except (ValueError, TypeError):
        return web.json_response({'error': 'Provide valid /v1 URLs, optional credential variable names, '
                                          'and an optional integer timeout from 600 to 14400 seconds'}, status=400)
    lock = request.app[_CHECK_LOCK]
    if lock.locked():
        return web.json_response({'error': 'A model connection check is already running'}, status=409)
    task = asyncio.current_task()
    request.app[_CHECK_TASKS].add(task)
    try:
        async with lock:
            checks = await probe_profile(profile)
            key = store.save_external_profile(profile)
            return web.json_response({'profile_id': key, 'models': store.choices(),
                                      'checks': checks, 'qualification': 'unqualified'}, status=201)
    except ValueError as error:
        return web.json_response({'error': str(error)}, status=400)
    except OSError:
        return web.json_response({'error': 'Could not save the model profile; check configuration storage'}, status=500)
    finally:
        request.app[_CHECK_TASKS].discard(task)


async def _cancel_checks(app: web.Application) -> None:
    tasks = list(app[_CHECK_TASKS])
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def register_routes(app: web.Application) -> None:
    app[_CHECK_LOCK] = asyncio.Lock()
    app[_CHECK_TASKS] = set()
    app.router.add_post('/api/external-models', _test_and_save)
    app.on_shutdown.append(_cancel_checks)
