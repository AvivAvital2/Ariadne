"""Tests for ``write_source_scip_config`` — the helper that lets
``ariadne discover`` author the auto-managed ``index_kinds`` +
``scip:`` block in ``ariadne.yaml``.

Per the UX principle that users should only author ``path`` /
``depends_on`` / ``exclude`` / ``exclude_dirs``, this helper closes
the gap between "what discover detects" and "what the rest of
Ariadne reads from YAML."
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


def _write_yaml(p: Path, content: str) -> None:
    p.write_text(dedent(content).lstrip('\n'), encoding='utf-8')


def _read_yaml(p: Path) -> dict:
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    return yaml.load(p.read_text(encoding='utf-8'))


def test_writes_index_kinds_and_scip_block_when_missing(tmp_path: Path) -> None:
    """A source with only user-authored fields (path) gains the
    auto-managed ``index_kinds`` + ``scip:`` block after the call."""
    from docgen.yaml_writer import write_source_scip_config

    yaml_path = tmp_path / 'ariadne.yaml'
    _write_yaml(yaml_path, '''
        sources:
          webapp:
            path: /path/to/webapp
    ''')

    artifact = tmp_path / '.ariadne' / 'index.scip'
    rewrote = write_source_scip_config(
        yaml_path,
        'webapp',
        catalog_scip_languages={'javascript'},
        artifact_path=artifact,
    )
    assert rewrote is True

    data = _read_yaml(yaml_path)
    src = data['sources']['webapp']
    assert src['path'] == '/path/to/webapp'  # user field preserved
    assert src['index_kinds'] == {'javascript': 'scip'}
    assert src['scip'] == {
        'artifact_path': str(artifact),
        'max_staleness_days': 7,
    }


def test_idempotent_when_state_already_matches(tmp_path: Path) -> None:
    """Re-running with the same arguments after a successful write
    returns False and does not rewrite the file. The mtime stays put,
    so file watchers and git diff stay quiet."""
    from docgen.yaml_writer import write_source_scip_config

    yaml_path = tmp_path / 'ariadne.yaml'
    artifact = tmp_path / '.ariadne' / 'index.scip'
    _write_yaml(yaml_path, f'''
        sources:
          webapp:
            path: /path/to/webapp
            index_kinds:
              javascript: scip
            scip:
              artifact_path: {artifact}
              max_staleness_days: 7
    ''')

    initial_mtime = yaml_path.stat().st_mtime_ns
    rewrote = write_source_scip_config(
        yaml_path,
        'webapp',
        catalog_scip_languages={'javascript'},
        artifact_path=artifact,
    )
    assert rewrote is False
    assert yaml_path.stat().st_mtime_ns == initial_mtime, (
        'idempotent re-run should not touch the file'
    )


def test_writes_when_languages_change(tmp_path: Path) -> None:
    """Adding a new detected language updates index_kinds. Pairs with
    the idempotent test so a fix that always-or-never rewrites fails
    one half."""
    from docgen.yaml_writer import write_source_scip_config

    yaml_path = tmp_path / 'ariadne.yaml'
    artifact = tmp_path / '.ariadne' / 'index.scip'
    _write_yaml(yaml_path, f'''
        sources:
          polyglot:
            path: /path/to/polyglot
            index_kinds:
              javascript: scip
            scip:
              artifact_path: {artifact}
              max_staleness_days: 7
    ''')

    # Now scala has been added to the source.
    rewrote = write_source_scip_config(
        yaml_path,
        'polyglot',
        catalog_scip_languages={'javascript', 'scala'},
        artifact_path=artifact,
    )
    assert rewrote is True

    data = _read_yaml(yaml_path)
    assert data['sources']['polyglot']['index_kinds'] == {
        'javascript': 'scip',
        'scala': 'scip',
    }


def test_writes_multiple_languages_in_sorted_order(tmp_path: Path) -> None:
    """``index_kinds`` keys land in sorted order so YAML diffs across
    re-runs are stable (no spurious churn from Python dict ordering
    on retries with the same input)."""
    from docgen.yaml_writer import write_source_scip_config

    yaml_path = tmp_path / 'ariadne.yaml'
    _write_yaml(yaml_path, '''
        sources:
          jvm_polyglot:
            path: /path/to/jvm
    ''')

    write_source_scip_config(
        yaml_path,
        'jvm_polyglot',
        catalog_scip_languages={'scala', 'java'},
        artifact_path=tmp_path / 'idx.scip',
    )

    raw = yaml_path.read_text(encoding='utf-8')
    java_pos = raw.index('java:')
    scala_pos = raw.index('scala:')
    assert java_pos < scala_pos, (
        'index_kinds keys should appear in sorted order'
    )


def test_handles_source_with_no_user_fields(tmp_path: Path) -> None:
    """A bare ``sources.X:`` (None value, shorthand for empty mapping)
    gets coerced into a real mapping and gains the auto-managed block."""
    from docgen.yaml_writer import write_source_scip_config

    yaml_path = tmp_path / 'ariadne.yaml'
    _write_yaml(yaml_path, '''
        sources:
          empty:
    ''')

    rewrote = write_source_scip_config(
        yaml_path,
        'empty',
        catalog_scip_languages={'javascript'},
        artifact_path=tmp_path / 'idx.scip',
    )
    assert rewrote is True

    data = _read_yaml(yaml_path)
    assert data['sources']['empty']['index_kinds'] == {'javascript': 'scip'}


def test_raises_when_yaml_missing(tmp_path: Path) -> None:
    from docgen.yaml_writer import write_source_scip_config

    with pytest.raises(FileNotFoundError):
        write_source_scip_config(
            tmp_path / 'nope.yaml',
            'webapp',
            catalog_scip_languages={'javascript'},
            artifact_path=tmp_path / 'idx.scip',
        )


def test_raises_when_source_missing(tmp_path: Path) -> None:
    from docgen.yaml_writer import write_source_scip_config

    yaml_path = tmp_path / 'ariadne.yaml'
    _write_yaml(yaml_path, '''
        sources:
          known:
            path: /path/to/known
    ''')

    with pytest.raises(KeyError):
        write_source_scip_config(
            yaml_path,
            'unknown',
            catalog_scip_languages={'javascript'},
            artifact_path=tmp_path / 'idx.scip',
        )


def test_preserves_user_comments_on_round_trip(tmp_path: Path) -> None:
    """ruamel.yaml's RoundTripLoader preserves user comments. Pinning
    this so the helper isn't accidentally swapped to a comment-eating
    loader (PyYAML)."""
    from docgen.yaml_writer import write_source_scip_config

    yaml_path = tmp_path / 'ariadne.yaml'
    _write_yaml(yaml_path, '''
        # Top-level comment about the project
        sources:
          webapp:
            # The frontend bundle
            path: /path/to/webapp
            depends_on: [shared]
    ''')

    write_source_scip_config(
        yaml_path,
        'webapp',
        catalog_scip_languages={'javascript'},
        artifact_path=tmp_path / 'idx.scip',
    )

    raw = yaml_path.read_text(encoding='utf-8')
    assert '# Top-level comment about the project' in raw
    assert '# The frontend bundle' in raw
