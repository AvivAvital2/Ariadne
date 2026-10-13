"""Verify local registry distribution in an empty, isolated Docker daemon.

python tests/container_distribution.py --output NEW_DIRECTORY
No paid APIs or external registry publication. Requires Docker Desktop/Engine;
downloads pinned registry and Docker-in-Docker test infrastructure if absent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = 'registry:3.1.2@sha256:ddf754342cfc8acc51a56d5d0ab6af06826461864460636d8bd5c546dab2a7b8'
DAEMON = (
    'docker:29.6.2-dind@sha256:bfec1f5159c63a81ca6fdedbd81404d2c0e16378ed0feec3bb3fbf3998847659'
)


def verify(image, output, *, compose_only=False):
    output.mkdir(parents=True, exist_ok=False)
    docker = shutil.which('docker')
    assert docker, 'Docker CLI is required'
    prefix = 'ariadne-distribution-' + uuid.uuid4().hex[:10]
    registry, daemon, producer = prefix + '-registry', prefix + '-daemon', prefix + '-producer'
    report = {
        'image': image,
        'registry_image': REGISTRY,
        'daemon_image': DAEMON,
        'scope': 'compose' if compose_only else 'full',
        'commands': [],
    }
    cleanup = []

    def run(name, *arguments, check=True):
        print(name, flush=True)
        result = subprocess.run(  # noqa: S603 - fixed Docker operations scoped to disposable test resources.
            [docker, *arguments],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        stem = f'{len(report["commands"]):02d}-{name}'
        (output / (stem + '.stdout')).write_text(result.stdout)
        (output / (stem + '.stderr')).write_text(result.stderr)
        report['commands'].append(
            {
                'name': name,
                'arguments': list(arguments),
                'exit_code': result.returncode,
                'output': stem,
            }
        )
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if check and result.returncode:
            raise RuntimeError(f'{name} failed: {result.stderr}')
        return result

    def inner(name, *arguments, **options):
        return run(name, 'exec', daemon, 'docker', *arguments, **options)

    def start_daemon(name):
        run(
            'start-' + name,
            'run',
            '-d',
            '--privileged',
            '--name',
            name,
            '--network',
            prefix,
            '--entrypoint',
            'dockerd',
            DAEMON,
            '--host=unix:///var/run/docker.sock',
            '--insecure-registry=registry:5000',
        )
        cleanup.append(('rm', '-f', '-v', name))
        for attempt in range(30):
            if (
                run(
                    f'{name}-ready-{attempt}', 'exec', name, 'docker', 'info', check=False
                ).returncode
                == 0
            ):
                return
            time.sleep(1)
        raise RuntimeError('Isolated daemon did not start')

    try:
        source = json.loads(run('source-image', 'image', 'inspect', image).stdout)[0]
        report['source_id'] = source['Id']
        run('private-network', 'network', 'create', '--internal', prefix)
        cleanup.append(('network', 'rm', prefix))
        run(
            'start-registry',
            'run',
            '-d',
            '--name',
            registry,
            '--network',
            prefix,
            '--network-alias',
            'registry',
            REGISTRY,
        )
        cleanup.append(('rm', '-f', '-v', registry))
        start_daemon(daemon)
        start_daemon(producer)
        assert not inner('empty-image-store', 'image', 'ls', '--quiet').stdout.strip()
        report['initial_images'] = []
        run('registry-ready', 'exec', producer, 'wget', '-qO-', 'http://registry:5000/v2/')
        archive = output / 'producer-image.tar'
        run('export-to-producer', 'image', 'save', '--output', str(archive), image)
        run('copy-to-producer', 'cp', str(archive), producer + ':/seed.tar')
        run('load-producer', 'exec', producer, 'docker', 'load', '--input', '/seed.tar')
        archive.unlink()
        target = 'registry:5000/ariadne:verification'
        run('tag', 'exec', producer, 'docker', 'tag', image, target)
        run('push', 'exec', producer, 'docker', 'push', target)
        published = json.loads(
            run('published-image', 'exec', producer, 'docker', 'image', 'inspect', target).stdout
        )[0]
        digest = next(
            d.split('@')[1]
            for d in published['RepoDigests']
            if d.startswith(target.rsplit(':', 1)[0] + '@')
        )
        reference = 'registry:5000/ariadne@' + digest
        report.update(registry_digest=digest, pulled_reference=reference)
        inner('clean-pull', 'pull', reference)
        pulled = json.loads(inner('pulled-image', 'image', 'inspect', reference).stdout)[0]
        # Classic and containerd image stores expose different ID representations.
        # Verify transported identity, every filesystem layer and runtime settings.
        assert pulled['Id'] == published['Id']
        assert pulled['RootFS']['Layers'] == source['RootFS']['Layers']
        for key in ('User', 'Env', 'Entrypoint', 'Cmd', 'WorkingDir'):
            assert pulled['Config'].get(key) == source['Config'].get(key), key
        report['pulled_id'] = pulled['Id']
        report['source_layers_and_runtime_match'] = True

        if not compose_only:
            checks = output / 'checks'
            checks.mkdir()
            for filename in ('container_smoke.py', 'local_model_server.py'):
                shutil.copyfile(ROOT / 'tests' / filename, checks / filename)
            run('copy-checks', 'cp', str(checks), daemon + ':/checks')
            base = ['run', '--rm', '--pull=never', '--network=none']
            assert (
                'Documentation library management'
                in inner('default-command', *base, reference).stdout
            )
            assert '/data/cloud.db' in inner('bundled-cloud', *base, reference, 'config').stdout
            assert (
                '/data/local.db'
                in inner(
                    'bundled-local',
                    *base,
                    '-e',
                    'ARIADNE_CONFIG=/opt/ariadne/examples/local.yaml',
                    reference,
                    'config',
                ).stdout
            )
            runner = [
                '--mount',
                'type=bind,src=/checks,dst=/checks,readonly',
                '--entrypoint',
                'python',
            ]
            smoke = inner(
                'installed-smoke', *base, *runner, reference, '/checks/container_smoke.py'
            )
            report['smoke'] = json.loads(smoke.stdout)
            for stage in ('seed', 'reopen'):
                result = inner(
                    'persistent-' + stage,
                    *base,
                    *runner,
                    '--mount',
                    'type=volume,src=ariadne-state,dst=/data',
                    reference,
                    '/checks/container_smoke.py',
                    '--persistent-state',
                    stage,
                )
                report[stage] = json.loads(result.stdout)

        # Only deployment configuration and synthetic source enter the consumer directory.
        consumer = output / 'consumer'
        for filename in (
            'deployments/app/compose.yaml',
            'deployments/app/cloud.yaml',
            'deployments/app/local.yaml',
            'deployments/local-v1/compose.yaml',
        ):
            path = consumer / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / filename, path)
        (consumer / 'project').mkdir()
        (consumer / 'project/core.py').write_text('def add(a, b):\n    return a + b\n')
        run('copy-consumer-config', 'cp', str(consumer), daemon + ':/consumer')
        for suite, database in (('app', 'cloud'), ('local-v1', 'local')):
            result = run(
                'compose-' + suite,
                'exec',
                '-e',
                'ARIADNE_IMAGE=' + reference,
                '-e',
                'ARIADNE_SOURCE_DIR=/consumer/project',
                daemon,
                'docker',
                'compose',
                '-f',
                f'/consumer/deployments/{suite}/compose.yaml',
                'run',
                '--rm',
                '--pull',
                'never',
                'ariadne',
                'config',
            )
            assert f'/data/{database}.db' in result.stdout
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        run('daemon-logs', 'logs', daemon, check=False)
        run('producer-logs', 'logs', producer, check=False)
        run('registry-logs', 'logs', registry, check=False)
        for i, arguments in enumerate(reversed(cleanup)):
            run(f'cleanup-{i}', *arguments, check=False)
        (output / 'producer-image.tar').unlink(missing_ok=True)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default='ariadne:local-model-support')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--compose-only',
        action='store_true',
        help='Check clean pull and Compose without repeating CLI/MCP/persistence checks',
    )
    args = parser.parse_args()
    verify(args.image, args.output.resolve(), compose_only=args.compose_only)
