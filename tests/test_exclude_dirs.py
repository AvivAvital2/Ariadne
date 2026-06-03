"""Tests for ``exclude_dirs:`` — directory-name pruning in discovery.

Glob patterns via ``Path.match`` only match a single segment per ``**``,
so ``exclude: ['docs/**']`` doesn't catch deeply-nested files. The
``exclude_dirs:`` mechanism prunes the walk at directory level by name,
mirroring the built-in ``_EXCLUDED_DIRS`` set (node_modules, .git, etc.).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Config layer: SourceConfig.exclude_dirs + yaml parsing
# ---------------------------------------------------------------------------


def test_source_config_has_exclude_dirs_default_empty():
    from config import SourceConfig
    sc = SourceConfig(path='/tmp/x')
    assert sc.exclude_dirs == ()


def test_source_config_accepts_exclude_dirs_tuple():
    from config import SourceConfig
    sc = SourceConfig(
        path='/tmp/x',
        exclude_dirs=('docs', 'generated'),
    )
    assert sc.exclude_dirs == ('docs', 'generated')


def test_config_reads_exclude_dirs_from_yaml(tmp_path):
    from config import Config

    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        'sources:\n'
        '  myapp:\n'
        '    path: /tmp/myapp\n'
        '    exclude_dirs:\n'
        '      - docs\n'
        '      - generated-docs\n'
        '      - target\n',
        encoding='utf-8',
    )
    cfg = Config(config_path=yaml_path)
    sc = cfg.get_source_config('myapp')
    assert sc is not None
    assert sc.exclude_dirs == ('docs', 'generated-docs', 'target')


# ---------------------------------------------------------------------------
# Discovery layer: walk-pruning honors caller-provided dirs
# ---------------------------------------------------------------------------


def test_find_catalog_files_prunes_named_directory(tmp_path):
    """A directory named in ``exclude_dir_names`` and ALL of its
    descendants must be skipped, regardless of depth.
    """
    from docgen.staleness import find_catalog_files

    # Lay out:
    #   src/main.py        (kept)
    #   docs/README.md     (excluded)
    #   docs/sub/deep.md   (excluded — would slip past glob ** patterns)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'main.py').write_text('x = 1', encoding='utf-8')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'README.md').write_text('# readme', encoding='utf-8')
    (tmp_path / 'docs' / 'sub').mkdir()
    (tmp_path / 'docs' / 'sub' / 'deep.md').write_text(
        'deep', encoding='utf-8',
    )

    files = find_catalog_files(
        tmp_path,
        exclude_dir_names=('docs',),
    )
    rels = sorted(str(f.relative_to(tmp_path)) for f in files)
    assert 'src/main.py' in rels
    assert not any(r.startswith('docs/') for r in rels), (
        f'docs subtree leaked: {rels}'
    )


def test_find_python_files_prunes_named_directory(tmp_path):
    """``find_python_files`` accepts the same ``exclude_dir_names`` kwarg."""
    from docgen.staleness import find_python_files

    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'app.py').write_text('x = 1', encoding='utf-8')
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'private').mkdir()
    (tmp_path / 'scripts' / 'private' / 'secret.py').write_text(
        "TOKEN = 'real'", encoding='utf-8',
    )

    files = find_python_files(
        tmp_path,
        exclude_dir_names=('private',),
    )
    rels = sorted(str(f.relative_to(tmp_path)) for f in files)
    assert 'src/app.py' in rels
    assert not any('private' in r for r in rels)


# ---------------------------------------------------------------------------
# Orchestrator layer
# ---------------------------------------------------------------------------


def test_orchestrator_config_has_exclude_dir_names_default_empty():
    from docgen.orchestrator import OrchestratorConfig
    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
    )
    assert cfg.exclude_dir_names == ()
