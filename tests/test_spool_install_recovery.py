"""An abrupt installer exit must never register uncommitted library content."""
import asyncio
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from config import Config
from inference import InferenceContext
from local_models import mac_model_profile
from spool_pack import build_pack, install_pack
from spools import SpoolError, SpoolManifest, resolve_spools
from tests.local_model_server import DOCUMENT
from writer import LibraryWriter

ROOT = Path(__file__).resolve().parents[1]


def snapshot(library):
    with library._conn_provider.acquire() as connection:
        return tuple(connection.iterdump())


@pytest.mark.parametrize('reader', ['manifest', 'resolution'])
def test_cache_publication_during_manifest_read_cannot_bypass_pending_marker(tmp_path, monkeypatch, reader):
    import spools

    cache = tmp_path / 'cache'
    directory = cache / 'corpus'
    directory.mkdir(parents=True)
    manifest_path = directory / 'manifest.yaml'
    manifest_path.write_text(yaml.safe_dump(SpoolManifest(
        environment='corpus', version='1', target_runtime='fixture', checksum='sha256:test').to_dict()))
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump({'spools': {'corpus': {'runtime': 'fixture'}}}))
    cfg = Config(config_path)
    original = spools.load_yaml_mapping

    def publish_pending_then_read(path, error_cls):
        # Models a cache directory replacement after the first marker check.
        (directory / spools.PACK_INSTALL_PENDING).touch()
        return original(path, error_cls)

    monkeypatch.setattr(spools, 'load_yaml_mapping', publish_pending_then_read)
    if reader == 'manifest':
        with pytest.raises(SpoolError, match='incomplete'):
            SpoolManifest.load(manifest_path)
    else:
        resolution = resolve_spools(cfg, cache_dir=cache)
        assert not resolution.registered
        assert len(resolution.gaps) == 1 and 'incomplete' in resolution.gaps[0].message


@pytest.mark.parametrize('reinstall', [False, True])
@pytest.mark.parametrize('crash_point', ['before_commit', 'after_commit'])
async def test_process_death_after_cache_publish_fails_closed_and_reinstall_recovers(
        native_pair, tmp_path, reinstall, crash_point):
    context, _chat, embed = native_pair
    cache = tmp_path / 'cache'
    data = mac_model_profile()
    data.update(db_path=str(tmp_path / 'consumer.db'),
                staleness_db_path=str(tmp_path / 'consumer-staleness.db'),
                spools={'corpus': {'runtime': 'fixture'}})
    config_path = tmp_path / 'consumer.yaml'
    config_path.write_text(yaml.safe_dump(data))
    cfg = Config(config_path)
    target = InferenceContext.from_config(cfg)
    pack_path = tmp_path / 'pack.zip'
    try:
        with context.write_session(), context.open_library(operation='write') as builder:
            writer = LibraryWriter(builder)
            doc = await writer.add_document('explanation', 'Arithmetic', DOCUMENT, source_name='corpus')
            build_pack(builder, environment='corpus', version='1', target_runtime='fixture',
                       source_root=tmp_path, out_path=pack_path)
            with target.write_session(), target.open_library(operation='write') as consumer:
                if reinstall:
                    await asyncio.to_thread(install_pack, consumer, pack_path, cache_dir=cache)
                before = snapshot(consumer)
            await writer.replace_document(doc.id, content=DOCUMENT + '\nUpdated edition.')
            build_pack(builder, environment='corpus', version='2', target_runtime='fixture',
                       source_root=tmp_path, out_path=pack_path)

        script = textwrap.dedent('''
            import os, sys
            from pathlib import Path
            from config import Config
            from inference import InferenceContext
            from spool_pack import install_pack
            config, pack, cache = map(Path, sys.argv[1:4])
            crash_point = sys.argv[4]
            original = Path.replace
            def crash_after_publish(path, destination):
                result = original(path, destination)
                if crash_point == 'before_commit' and path.name == 'new' and Path(destination) == cache / 'corpus':
                    os._exit(73)
                return result
            Path.replace = crash_after_publish
            original_unlink = Path.unlink
            def crash_after_commit(path, **kwargs):
                if crash_point == 'after_commit' and path == cache / 'corpus' / '.install-pending':
                    os._exit(73)
                return original_unlink(path, **kwargs)
            Path.unlink = crash_after_commit
            context = InferenceContext.from_config(Config(config))
            with context.write_session(), context.open_library(operation='write') as library:
                install_pack(library, pack, cache_dir=cache)
        ''')
        env = {key: value for key, value in os.environ.items()
               if not key.endswith(('_API_KEY', '_TOKEN'))}
        env['PYTHON_DOTENV_DISABLED'] = '1'
        requests_before = len(embed.requests)
        process = await asyncio.to_thread(
            subprocess.run, [sys.executable, '-c', script, str(config_path), str(pack_path), str(cache), crash_point],
            cwd=ROOT, env=env, capture_output=True, text=True,
            timeout=20, check=False)
        assert process.returncode == 73, process.stderr
        with target.open_library(operation='inspect') as consumer:
            if crash_point == 'before_commit':
                assert snapshot(consumer) == before
            else:
                assert consumer.get_document(doc.id).content.endswith('Updated edition.')
        with pytest.raises(SpoolError, match='incomplete'):
            SpoolManifest.load(cache / 'corpus' / 'manifest.yaml')
        resolution = resolve_spools(cfg, cache_dir=cache)
        assert not resolution.registered
        assert len(resolution.gaps) == 1 and 'incomplete' in resolution.gaps[0].message

        with target.write_session(), target.open_library(operation='write') as consumer:
            await asyncio.to_thread(install_pack, consumer, pack_path, cache_dir=cache)
            restored = consumer.get_document(doc.id)
            assert restored.content.endswith('Updated edition.')
            assert restored.source_name == 'spool:corpus'
        assert SpoolManifest.load(cache / 'corpus' / 'manifest.yaml').version == '2'
        assert set(resolve_spools(cfg, cache_dir=cache).registered) == {'corpus'}
        assert len(embed.requests) == requests_before
    finally:
        await context.aclose()
        await target.aclose()
