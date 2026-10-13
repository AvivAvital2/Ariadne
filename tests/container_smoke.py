"""Run inside the installed image with only this directory mounted read-only.

docker run --rm --network none --mount type=bind,src=.../tests,dst=/checks,readonly \
    --entrypoint python ariadne:local-model-support /checks/container_smoke.py

No pytest or package installation is needed in the runtime image. Loopback HTTP
fixtures exercise local serving contracts; they are not real GPU/model servers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import config
from library import Library
from library.model_binding import inspect_binding
from local_model_server import DOCUMENT, deployments, local_config


def command(arguments, env, cwd, records):
    result = subprocess.run(  # noqa: S603 - fixed application commands in an isolated test container.
        ['/opt/ariadne/.venv/bin/ariadne', *arguments],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    records.append(
        {
            'arguments': arguments,
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    )
    if result.returncode:
        raise AssertionError(f'CLI failed: {arguments}\n{result.stdout}\n{result.stderr}')


async def mcp_session(env, directory):
    params = StdioServerParameters(command='ariadne', args=['mcp'], env=env, cwd=str(directory))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        assert {'ariadne_search', 'ariadne_ask'} <= {t.name for t in tools.tools}
        search = await session.call_tool(
            'ariadne_search',
            {
                'query': 'How does arithmetic work?',
                'source': 'demo',
                'limit': 2,
            },
        )
        assert not search.isError, search
        answer = await session.call_tool(
            'ariadne_ask',
            {
                'question': 'How does arithmetic work?',
                'source': 'demo',
            },
        )
        assert not answer.isError, answer
        assert 'Core arithmetic' in str(answer)
        return {
            'tools': len(tools.tools),
            'search': search.model_dump(mode='json'),
            'answer': answer.model_dump(mode='json'),
        }


def main():
    # The implementation must come from the image, not the mounted test directory.
    assert '/site-packages/' in config.__file__ or config.__file__ == '/opt/ariadne/config.py', config.__file__
    report = {'installed_module': config.__file__, 'uid': os.getuid(), 'runs': []}
    assert os.getuid() != 0
    assert os.environ['ARIADNE_CONFIG'] in {'/data/ariadne.yaml', '/opt/ariadne/examples/cloud.yaml'}
    report['bundled_presets'] = {}
    for name in ('cloud', 'local'):
        preset = Path(f'/opt/ariadne/examples/{name}.yaml')
        cfg = config.Config(preset)
        settings = cfg.model_settings()
        assert cfg.db_path == f'/data/{name}.db'
        assert cfg.staleness_db_path == f'/data/{name}_staleness.db'
        assert cfg.get_source_config('project').path == '/workspace'
        assert (settings.chat.provider == 'local') == (name == 'local')
        report['bundled_presets'][name] = hashlib.sha256(preset.read_bytes()).hexdigest()
    with tempfile.TemporaryFile(dir='/data') as state:
        state.write(b'writable persistent-state location')
    for cloud_keys in (False, True):
        with tempfile.TemporaryDirectory() as temporary, deployments() as (chat, embed):
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            (source / 'core.py').write_text('def add(a, b):\n    return a + b\n')
            (source / 'extra.py').write_text('def multiply(a, b):\n    return a * b\n')
            data = local_config(chat.base_url, embed.base_url)
            data.update(default_source='demo', sources={'demo': str(source)}, themes_enabled=False)
            path = root / 'ariadne.yaml'
            path.write_text(yaml.safe_dump(data))
            env = dict(os.environ, ARIADNE_CONFIG=str(path))
            for key in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
                env.pop(key, None)
                if cloud_keys:
                    env[key] = 'dummy-cloud-key-must-not-be-used'
            records = []
            command(['--help'], env, root, records)
            command(['config'], env, root, records)
            command(['doctor', '--models'], env, root, records)
            requests = len(chat.requests) + len(embed.requests)
            command(['generate', '--source', 'demo', '--dry-run'], env, root, records)
            assert len(chat.requests) + len(embed.requests) == requests
            command(
                [
                    'generate',
                    '--source',
                    'demo',
                    '--path',
                    'core.py',
                    '--types',
                    'explanation',
                    '--no-batch',
                    '--no-crossrefs',
                    '--yes',
                ],
                env,
                root,
                records,
            )
            command(
                [
                    'generate',
                    '--source',
                    'demo',
                    '--path',
                    'extra.py',
                    '--types',
                    'explanation',
                    '--batch',
                    '--no-crossrefs',
                    '--yes',
                ],
                env,
                root,
                records,
            )
            command(['batch', 'list'], env, root, records)
            command(['search', 'arithmetic', '--source', 'demo'], env, root, records)
            command(['search', 'arithmetic', '--source', 'demo', '--chunks'], env, root, records)
            command(['rebuild'], env, root, records)
            command(['list'], env, root, records)
            mcp = asyncio.run(mcp_session(env, root))
            cfg = config.Config(path)
            assert inspect_binding(Path(cfg.db_path)) is not None
            with Library(Path(cfg.db_path)) as library:
                docs = library.list_documents('explanation')
                assert len(docs) == 2
                for doc in docs:
                    assert doc.content.strip() == DOCUMENT.strip()
                    assert doc.source_name == 'demo'
                    assert doc.embedding.shape == (3072,)
                    assert library.get_chunks(doc.id) and library.get_sections(doc.id)
            with sqlite3.connect(cfg.db_path) as conn:
                jobs = conn.execute('SELECT state FROM spool_jobs').fetchall()
                assert jobs and all(state == 'applied' for (state,) in jobs), jobs
            assert all(
                'Authorization' not in headers for _, _, headers in chat.requests + embed.requests
            )
            report['runs'].append(
                {
                    'cloud_keys_present': cloud_keys,
                    'commands': records,
                    'mcp': mcp,
                    'requests': len(chat.requests) + len(embed.requests),
                    'documents': len(docs),
                    'spool_jobs': len(jobs),
                }
            )
    report['status'] = 'passed'
    print(json.dumps(report, indent=2))


def persistent_state(stage):
    """Seed a named volume, then verify it from a different container."""
    root = Path('/data/recreation-check')
    root.mkdir(exist_ok=True)
    marker = root / 'saved.json'
    assert marker.exists() == (stage == 'reopen')
    source = root / 'source'
    source.mkdir(exist_ok=True)
    (source / 'core.py').write_text('def add(a, b):\n    return a + b\n')
    with deployments() as (chat, embed):
        data = local_config(chat.base_url, embed.base_url)
        data.update(default_source='demo', sources={'demo': str(source)}, themes_enabled=False)
        path = root / 'ariadne.yaml'
        path.write_text(yaml.safe_dump(data))
        cfg = config.Config(path)
        env = dict(os.environ, ARIADNE_CONFIG=str(path))
        records = []
        if stage == 'seed':
            command(
                [
                    'generate',
                    '--source',
                    'demo',
                    '--types',
                    'explanation',
                    '--batch',
                    '--no-crossrefs',
                    '--yes',
                ],
                env,
                root,
                records,
            )
        fingerprints = {}
        for database in (cfg.db_path, cfg.staleness_db_path):
            assert Path(database).is_file()
            with sqlite3.connect(database) as conn:
                fingerprints[database] = hashlib.sha256(
                    '\n'.join(conn.iterdump()).encode()
                ).hexdigest()
        with sqlite3.connect(cfg.db_path) as conn:
            runs = conn.execute('SELECT run_id, state FROM spool_runs').fetchall()
            assert len(runs) == 1 and runs[0][1] == 'completed', runs
            assert conn.execute('SELECT state FROM spool_jobs').fetchall() == [('applied',)]
        saved = {'fingerprints': fingerprints, 'run_id': runs[0][0]}
        if stage == 'seed':
            marker.write_text(json.dumps(saved, indent=2))
        else:
            assert saved == json.loads(marker.read_text()), 'Persisted database content changed'
            command(['batch', 'resume', runs[0][0]], env, root, records)
            assert not chat.requests and not embed.requests, 'Completed job replay made requests'
            command(['search', 'arithmetic', '--source', 'demo'], env, root, records)
            asyncio.run(mcp_session(env, root))
        print(
            json.dumps({'status': 'passed', 'stage': stage, **saved, 'commands': records}, indent=2)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--persistent-state', choices=('seed', 'reopen'))
    arguments = parser.parse_args()
    if arguments.persistent_state:
        persistent_state(arguments.persistent_state)
    else:
        main()
