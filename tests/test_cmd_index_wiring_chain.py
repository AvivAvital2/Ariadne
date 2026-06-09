"""End-to-end wiring contract for ``cli_core.cmd_index``.

Pins the full sequence of ``persist_*`` invocations that
``cmd_index`` runs after the indexer + merge phases. This is the
single safety net that catches:

- Removal of any persist step (regression from the half-baked-extractors
  audit work that wired Tiers 1, 2, 3 string_literals, and 4)
- Reordering that would break dependencies — e.g., string_literals
  MUST be persisted before route extractors look up literal path
  values; HTTP clients + endpoints MUST both be persisted before the
  URL resolver joins them
- Wrong source list passed (each helper is called with the same
  source_pairs list, not a per-source loop in cmd_index itself)

Each persist_* helper has its own focused tests (test_persist_*.py
files); this test does not duplicate that coverage. It pins ONLY the
invocation chain — the contract that makes the audit work load-bearing
in production.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        'source': None, 'all': False, 'dry_run': False,
        'kind': None, 'db': None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def restore_global_config():
    import config as config_module
    saved = config_module._global_config
    yield
    config_module._global_config = saved


class _FakeAdapter:
    """Minimal adapter that writes a synthetic ``.scip`` payload so
    cmd_index reaches the persist phase. Accepts every kwarg cli_core
    plumbs for the python kind (``entry_kind``, ``excludes``,
    ``progress_callback``) plus the base set."""

    def run(
        self, *, cwd, output, env_hints,
        entry_kind: str = 'package',
        excludes: tuple = (),
        progress_callback=None,
    ):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'\x08\x01synthetic')
        from cli.core import IndexerResult
        return IndexerResult(
            success=True, indexer_version='fake/0.1',
        )


def _setup_python_source(tmp_path: Path) -> Path:
    """Set up a minimal Python source with a manifest that declares
    one python indexer entry. Returns the source root."""
    source_root = tmp_path / 'webapp'
    source_root.mkdir()
    (source_root / 'pkg').mkdir()
    (source_root / 'pkg' / '__init__.py').write_text('def f(): ...')
    manifest_dir = source_root / '.ariadne'
    manifest_dir.mkdir()
    (manifest_dir / 'manifest.json').write_text(
        json.dumps({
            'ariadne_version': '1',
            'source_name': 'webapp',
            'indexers': [{'kind': 'python', 'cwd': '.'}],
        }),
        encoding='utf-8',
    )
    return source_root


def _activate_yaml(yaml_path: Path) -> None:
    import config as config_module
    config_module._global_config = config_module.Config(config_path=yaml_path)


def test_cmd_index_invokes_all_persist_steps_in_dependency_order(
    tmp_path: Path, monkeypatch,
) -> None:
    """The 12-step persist chain runs in the order the dependency
    graph requires:

      1. persist_all_sources       (scip_symbols → enables steps 7–12)
      2. persist_api_endpoints     (Swagger → api_endpoints)
      3. persist_string_literals   (req'd by route extractors below)
      4. persist_config_values     (HOCON/YAML/dotenv key→value)
      5. persist_config_reads      (getter call sites; needs 3 + 4)
      6. persist_akka_http_endpoints
      7. persist_python_routes
      8. persist_express_routes
      9. persist_python_http_clients
      10. persist_js_http_clients
      11. persist_scala_http_clients
      12. persist_url_resolver     (joins clients to endpoints)

    Every wrapper records its name when called. The recorded order
    must match the contract.
    """
    from cli.core import cmd_index

    source_root = _setup_python_source(tmp_path)

    yaml_path = tmp_path / 'ariadne.yaml'
    db_path = tmp_path / 'ariadne.db'
    yaml_path.write_text(
        f'db_path: {db_path}\n'
        f'sources:\n'
        f'  webapp:\n'
        f'    path: {source_root}\n'
        f'    swagger_paths: [api/openapi.json]\n',
        encoding='utf-8',
    )
    _activate_yaml(yaml_path)

    invocation_log: list[str] = []

    def _record(name: str, return_value: int = 0):
        def _spy(*args, **kwargs):
            invocation_log.append(name)
            return return_value
        return _spy

    # Patch each wrapper at its module-level binding in scip_persist.
    # cmd_index does ``from docgen.scip_persist import persist_*``
    # inside the function body — so patching the module-level name
    # is what the inner import binds to.
    targets = [
        'persist_all_sources',
        'persist_api_endpoints',
        'persist_string_literals',
        'persist_config_values',
        'persist_config_reads',
        'persist_akka_http_endpoints',
        'persist_python_routes',
        'persist_express_routes',
        'persist_python_http_clients',
        'persist_js_http_clients',
        'persist_scala_http_clients',
        'persist_url_resolver',
    ]
    for name in targets:
        monkeypatch.setattr(
            f'docgen.scip_persist.{name}', _record(name),
        )

    rc = cmd_index(
        _make_args(source='webapp'),
        indexer_registry={'python': _FakeAdapter()},
        merger=None,
    )
    assert rc == 0

    assert invocation_log == targets, (
        f'cmd_index persist chain out of order or missing steps.\n'
        f'expected: {targets}\n'
        f'got:      {invocation_log}'
    )


def test_cmd_index_skips_persist_api_endpoints_when_no_swagger_declared(
    tmp_path: Path, monkeypatch,
) -> None:
    """``persist_api_endpoints`` is the one conditional step — only
    called when at least one source has ``swagger_paths`` declared.
    The other 9 steps run unconditionally because each is fail-soft
    on missing artifacts. Pairs with the in-order test so a fix
    that always-or-never runs api_endpoints fails one half."""
    from cli.core import cmd_index

    source_root = _setup_python_source(tmp_path)
    # Note: NO swagger_paths in YAML.

    yaml_path = tmp_path / 'ariadne.yaml'
    db_path = tmp_path / 'ariadne.db'
    yaml_path.write_text(
        f'db_path: {db_path}\n'
        f'sources:\n'
        f'  webapp:\n'
        f'    path: {source_root}\n',
        encoding='utf-8',
    )
    _activate_yaml(yaml_path)

    invocation_log: list[str] = []

    def _record(name: str):
        def _spy(*args, **kwargs):
            invocation_log.append(name)
            return 0
        return _spy

    targets = [
        'persist_all_sources',
        'persist_api_endpoints',
        'persist_string_literals',
        'persist_config_values',
        'persist_config_reads',
        'persist_akka_http_endpoints',
        'persist_python_routes',
        'persist_express_routes',
        'persist_python_http_clients',
        'persist_js_http_clients',
        'persist_scala_http_clients',
        'persist_url_resolver',
    ]
    for name in targets:
        monkeypatch.setattr(
            f'docgen.scip_persist.{name}', _record(name),
        )

    cmd_index(
        _make_args(source='webapp'),
        indexer_registry={'python': _FakeAdapter()},
        merger=None,
    )

    assert 'persist_api_endpoints' not in invocation_log, (
        'persist_api_endpoints must skip when no source declares '
        f'swagger_paths; got log={invocation_log}'
    )
    # The other 9 must still run.
    expected_without_swagger = [t for t in targets if t != 'persist_api_endpoints']
    assert invocation_log == expected_without_swagger


def test_cmd_index_dry_run_skips_entire_persist_chain(
    tmp_path: Path, monkeypatch,
) -> None:
    """``--dry-run`` short-circuits before adapter execution and
    therefore before all 10 persist steps. ``library_scip`` stays
    untouched. Pinned per the same per-step pattern that
    test_cmd_index_typescript_e2e proved for persist_all_sources
    alone — extended here to cover the full chain."""
    from cli.core import cmd_index

    source_root = _setup_python_source(tmp_path)

    yaml_path = tmp_path / 'ariadne.yaml'
    db_path = tmp_path / 'ariadne.db'
    yaml_path.write_text(
        f'db_path: {db_path}\n'
        f'sources:\n  webapp:\n    path: {source_root}\n',
        encoding='utf-8',
    )
    _activate_yaml(yaml_path)

    invocation_log: list[str] = []

    def _record(name: str):
        def _spy(*args, **kwargs):
            invocation_log.append(name)
            return 0
        return _spy

    targets = [
        'persist_all_sources',
        'persist_api_endpoints',
        'persist_string_literals',
        'persist_config_values',
        'persist_config_reads',
        'persist_akka_http_endpoints',
        'persist_python_routes',
        'persist_express_routes',
        'persist_python_http_clients',
        'persist_js_http_clients',
        'persist_scala_http_clients',
        'persist_url_resolver',
    ]
    for name in targets:
        monkeypatch.setattr(
            f'docgen.scip_persist.{name}', _record(name),
        )

    cmd_index(
        _make_args(source='webapp', dry_run=True),
        indexer_registry={'python': _FakeAdapter()},
        merger=None,
    )

    assert invocation_log == [], (
        f'dry-run must not call any persist_* step; got log={invocation_log}'
    )
