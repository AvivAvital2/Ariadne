"""``cmd_index`` must skip the slow per-language SCIP indexers when the
source's merged ``.scip`` is already fresh (younger than the source's
``max_staleness_days``), reusing the artifact and letting the persist phase
(which walks every configured source) reload it. This is what makes a re-run
of ``ariadne onboard --approve`` *not* re-index. ``--force`` bypasses the skip.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        'source': None, 'all': False, 'dry_run': False,
        'kind': None, 'db': None, 'force': False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def restore_global_config():
    import config as config_module
    saved = config_module._global_config
    yield
    config_module._global_config = saved


class _RecordingAdapter:
    """Records each indexer invocation so the test can assert it was (not) run."""

    def __init__(self) -> None:
        self.runs: list = []

    def run(self, *, cwd, output, env_hints, entry_kind='package',
            excludes=(), progress_callback=None):
        self.runs.append(cwd)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'\x08\x01synthetic')
        from cli.core import IndexerResult
        return IndexerResult(success=True, indexer_version='fake/0.1')


def _setup(tmp_path: Path) -> Path:
    source_root = tmp_path / 'webapp'
    (source_root / 'pkg').mkdir(parents=True)
    (source_root / 'pkg' / '__init__.py').write_text('def f(): ...')
    adir = source_root / '.ariadne'
    adir.mkdir()
    (adir / 'manifest.json').write_text(json.dumps({
        'ariadne_version': '1', 'source_name': 'webapp',
        'indexers': [{'kind': 'python', 'cwd': '.'}],
    }), encoding='utf-8')
    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'db_path: {tmp_path / "ariadne.db"}\n'
        f'sources:\n  webapp:\n    path: {source_root}\n',
        encoding='utf-8',
    )
    import config as config_module
    config_module._global_config = config_module.Config(config_path=yaml_path)
    return source_root


def _patch_persist(monkeypatch) -> None:
    for name in (
        'persist_all_sources', 'persist_api_endpoints', 'persist_string_literals',
        'persist_akka_http_endpoints', 'persist_python_routes',
        'persist_express_routes', 'persist_python_http_clients',
        'persist_js_http_clients', 'persist_scala_http_clients',
        'persist_url_resolver',
    ):
        monkeypatch.setattr(f'docgen.scip_persist.{name}', lambda *a, **k: 0)


def test_cmd_index_skips_indexer_when_scip_is_fresh(tmp_path, monkeypatch):
    from cli.core import cmd_index
    source_root = _setup(tmp_path)
    # A fresh merged artifact already exists (just written -> age ~0).
    (source_root / '.ariadne' / 'index.scip').write_bytes(b'\x08\x01existing')
    _patch_persist(monkeypatch)

    adapter = _RecordingAdapter()
    rc = cmd_index(_make_args(source='webapp'),
                   indexer_registry={'python': adapter}, merger=None)
    assert rc == 0
    assert adapter.runs == [], 'a fresh .scip must skip the indexer subprocess'

    # --force re-indexes despite the fresh artifact.
    adapter2 = _RecordingAdapter()
    rc = cmd_index(_make_args(source='webapp', force=True),
                   indexer_registry={'python': adapter2}, merger=None)
    assert rc == 0
    assert adapter2.runs, '--force must re-run the indexer'
