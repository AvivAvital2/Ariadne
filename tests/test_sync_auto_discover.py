"""Tests for sync's auto-discover-on-new-language behavior.

When ``ariadne sync`` sees changed files in a SCIP-routable language
that isn't yet declared in the source's ``index_kinds``, it should
invoke ``ariadne discover --config-only`` to update ``ariadne.yaml``
so the next ``ariadne index`` picks the new language up. The
expensive scip-X invocation stays a separate user-invoked step —
surfaced as a hint, not an automatic action.

These tests pin the contract for the cheap (config-only) auto-write,
the hint message, and the no-op cases (no new language, language
already declared, no yaml on disk).
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _touch(p: Path, content: str = '') -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')


@pytest.fixture(autouse=True)
def restore_global_config():
    import config as config_module
    saved = config_module._global_config
    yield
    config_module._global_config = saved


def _activate_yaml(yaml_path: Path) -> None:
    import config as config_module
    config_module._global_config = config_module.Config(config_path=yaml_path)


def test_new_language_in_changed_files_triggers_config_only_discover(
    tmp_path: Path,
) -> None:
    """A source declared with no SCIP block and changed files containing
    .ts → sync's helper sees the gap, invokes discover (config-only),
    YAML grows the auto-managed block."""
    from cli.generation import _maybe_auto_discover_for_new_language

    src = tmp_path / 'webapp'
    _touch(src / 'package.json', '{"name": "webapp"}')
    _touch(src / 'src' / 'app.ts', 'export const x = 1')

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'sources:\n  webapp:\n    path: {src}\n',
        encoding='utf-8',
    )
    _activate_yaml(yaml_path)

    from config import get_config
    cfg = get_config()

    # Pretend git diff returned this file
    _maybe_auto_discover_for_new_language(
        cfg, 'webapp', src, ['src/app.ts'],
    )

    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    data = yaml.load(yaml_path.read_text(encoding='utf-8'))
    assert data['sources']['webapp'].get('index_kinds') == {
        'javascript': 'scip',
    }


def test_already_declared_language_is_a_noop(tmp_path: Path) -> None:
    """When index_kinds.javascript:scip is already in YAML and a .ts
    file changes, nothing happens — sync stays fast."""
    from cli.generation import _maybe_auto_discover_for_new_language

    src = tmp_path / 'webapp'
    _touch(src / 'package.json', '{"name": "webapp"}')

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'sources:\n'
        f'  webapp:\n'
        f'    path: {src}\n'
        f'    index_kinds:\n'
        f'      javascript: scip\n'
        f'    scip:\n'
        f'      artifact_path: {src / ".ariadne" / "index.scip"}\n'
        f'      max_staleness_days: 7\n',
        encoding='utf-8',
    )
    _activate_yaml(yaml_path)

    from config import get_config
    cfg = get_config()

    initial_mtime = yaml_path.stat().st_mtime_ns
    _maybe_auto_discover_for_new_language(
        cfg, 'webapp', src, ['src/app.ts'],
    )
    assert yaml_path.stat().st_mtime_ns == initial_mtime, (
        'already-declared language should not rewrite yaml'
    )


def test_no_scip_routable_files_is_a_noop(tmp_path: Path) -> None:
    """Changed files in non-SCIP-routable languages (Python, JSON,
    Markdown) don't trigger discover. Pairs with the trigger test —
    a fix that always-or-never invokes discover fails one half."""
    from cli.generation import _maybe_auto_discover_for_new_language

    src = tmp_path / 'pyproject'
    _touch(src / 'pkg' / '__init__.py')

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'sources:\n  pyproject:\n    path: {src}\n',
        encoding='utf-8',
    )
    _activate_yaml(yaml_path)

    from config import get_config
    cfg = get_config()

    initial_mtime = yaml_path.stat().st_mtime_ns
    _maybe_auto_discover_for_new_language(
        cfg, 'pyproject', src,
        ['pkg/foo.py', 'README.md', 'data.json'],
    )
    assert yaml_path.stat().st_mtime_ns == initial_mtime


def test_no_config_path_is_silent_noop(tmp_path: Path) -> None:
    """When cfg has no on-disk yaml (in-memory Config), there's
    nothing to write back to. Helper returns silently rather than
    raising."""
    from cli.generation import _maybe_auto_discover_for_new_language
    from config import Config

    src = tmp_path / 'webapp'
    _touch(src / 'package.json', '{"name": "x"}')

    cfg = Config()  # no config_path
    cfg._config['sources'] = {'webapp': {'path': str(src)}}

    # Should not raise
    _maybe_auto_discover_for_new_language(
        cfg, 'webapp', src, ['src/app.ts'],
    )


def test_java_files_trigger_both_scala_and_java_index_kinds(
    tmp_path: Path,
) -> None:
    """A .java file in changed_files triggers index_kinds.java AND
    index_kinds.scala (scip-java emits one .scip covering both)."""
    from cli.generation import _maybe_auto_discover_for_new_language

    src = tmp_path / 'jvmproject'
    _touch(src / 'build.sbt', 'name := "jvm"')
    _touch(src / 'src' / 'main' / 'java' / 'Foo.java', 'class Foo {}')

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'sources:\n  jvmproject:\n    path: {src}\n',
        encoding='utf-8',
    )
    _activate_yaml(yaml_path)

    from config import get_config
    cfg = get_config()

    _maybe_auto_discover_for_new_language(
        cfg, 'jvmproject', src,
        ['src/main/java/Foo.java'],
    )

    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    data = yaml.load(yaml_path.read_text(encoding='utf-8'))
    index_kinds = data['sources']['jvmproject'].get('index_kinds') or {}
    assert index_kinds.get('java') == 'scip'
    assert index_kinds.get('scala') == 'scip', (
        'discover writes both scala AND java because scip-java covers '
        'both extensions in one .scip output'
    )
