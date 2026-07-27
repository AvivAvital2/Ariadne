"""``cmd_index`` reuses per-SCOPE: a scope whose intermediate ``.scip`` is fresh
(younger than the source's ``max_staleness_days``, or staleness-exempt) skips
its slow indexer; a scope whose intermediate is MISSING re-runs. This lets a
re-run of ``ariadne onboard --approve`` reuse the scopes that succeeded while
retrying the ones that failed — without ``--force`` re-indexing everything.
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
        from cli.index import IndexerResult
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


def test_cmd_index_reuses_scope_when_intermediate_fresh(tmp_path, monkeypatch):
    from cli.index import cmd_index
    source_root = _setup(tmp_path)
    # The scope's cached INTERMEDIATE exists (fresh, cwd='.' → scope 'python'),
    # so its slow indexer is skipped.
    inter = source_root / '.ariadne' / 'intermediate' / 'index-python.scip'
    inter.parent.mkdir(parents=True, exist_ok=True)
    inter.write_bytes(b'\x08\x01cached')
    (source_root / '.ariadne' / 'index.scip').write_bytes(b'\x08\x01merged')
    _patch_persist(monkeypatch)

    adapter = _RecordingAdapter()
    rc = cmd_index(_make_args(source='webapp'),
                   indexer_registry={'python': adapter}, merger=None)
    assert rc == 0
    assert adapter.runs == [], 'a fresh intermediate must skip the indexer'

    # --force re-indexes despite the fresh intermediate.
    adapter2 = _RecordingAdapter()
    rc = cmd_index(_make_args(source='webapp', force=True),
                   indexer_registry={'python': adapter2}, merger=None)
    assert rc == 0
    assert adapter2.runs, '--force must re-run the indexer'


def test_cmd_index_reruns_scope_when_intermediate_missing(tmp_path, monkeypatch):
    from cli.index import cmd_index
    source_root = _setup(tmp_path)
    # A merged index.scip exists, but the SCOPE's own intermediate is missing
    # (e.g. it failed last run) — so its indexer RE-RUNS instead of being
    # skipped. This is the per-scope reuse contract: reuse what's built, retry
    # what's missing, without --force re-indexing everything.
    (source_root / '.ariadne' / 'index.scip').write_bytes(b'\x08\x01merged')
    _patch_persist(monkeypatch)

    adapter = _RecordingAdapter()
    rc = cmd_index(_make_args(source='webapp'),
                   indexer_registry={'python': adapter}, merger=None)
    assert rc == 0
    assert adapter.runs, 'a missing intermediate must re-run the indexer'


def test_cmd_index_mixed_reuses_fresh_and_reruns_missing(tmp_path, monkeypatch):
    """The spool scenario: python's intermediate is cached (reused, indexer
    skipped) while java's is missing (re-run) — in ONE pass, no --force."""
    from cli.index import cmd_index
    source_root = tmp_path / 'webapp'
    (source_root / 'pkg').mkdir(parents=True)
    (source_root / 'pkg' / '__init__.py').write_text('def f(): ...')
    (source_root / 'build.sbt').write_text('name := "x"')
    adir = source_root / '.ariadne'
    adir.mkdir()
    (adir / 'manifest.json').write_text(json.dumps({
        'ariadne_version': '1', 'source_name': 'webapp',
        'indexers': [{'kind': 'python', 'cwd': '.'},
                     {'kind': 'java', 'cwd': '.'}],
    }), encoding='utf-8')
    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'db_path: {tmp_path / "ariadne.db"}\n'
        f'sources:\n  webapp:\n    path: {source_root}\n', encoding='utf-8')
    import config as config_module
    config_module._global_config = config_module.Config(config_path=yaml_path)
    # python cached, java missing
    (adir / 'intermediate').mkdir()
    (adir / 'intermediate' / 'index-python.scip').write_bytes(b'\x08\x01cached')
    _patch_persist(monkeypatch)

    class _Merger:
        def merge(self, inputs, output):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b'\x08\x01merged')
            return True

    py, jv = _RecordingAdapter(), _RecordingAdapter()
    rc = cmd_index(_make_args(source='webapp'),
                   indexer_registry={'python': py, 'java': jv}, merger=_Merger())
    assert rc == 0
    assert py.runs == [], 'cached python scope must be reused'
    assert jv.runs, 'missing java scope must be re-run'
