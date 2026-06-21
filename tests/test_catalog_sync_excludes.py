"""Tests for catalog-sync honoring per-source exclude / exclude_dirs.

``iter_catalog_files`` historically had no way to filter beyond the
built-in ``_SKIP_DIR_NAMES`` set. This module pins the new behavior:
the function accepts ``exclude_patterns`` (per-file globs) and
``exclude_dir_names`` (whole-tree pruning by name), mirroring
``find_catalog_files``. ``sync_source_catalog`` and ``cmd_catalog_sync``
forward the user's yaml config through.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# iter_catalog_files: new exclude parameters
# ---------------------------------------------------------------------------


def test_iter_catalog_files_honors_exclude_patterns(tmp_path):
    """File-level glob excludes filter matching files."""
    from docgen.catalog_writer import iter_catalog_files

    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'main.py').write_text('x = 1', encoding='utf-8')
    (tmp_path / 'src' / 'secrets.py').write_text(
        "API_KEY = 'sk-real'", encoding='utf-8',
    )

    files = iter_catalog_files(
        tmp_path, exclude_patterns=('**/secrets.py',),
    )
    rels = sorted(str(f.relative_to(tmp_path)) for f in files)
    assert 'src/main.py' in rels
    assert 'src/secrets.py' not in rels


def test_iter_catalog_files_honors_exclude_dir_names(tmp_path):
    """Directory-name pruning skips an entire subtree at any depth."""
    from docgen.catalog_writer import iter_catalog_files

    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'main.py').write_text('x = 1', encoding='utf-8')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'README.md').write_text('# r', encoding='utf-8')
    (tmp_path / 'docs' / 'deep' / 'nested').mkdir(parents=True)
    (tmp_path / 'docs' / 'deep' / 'nested' / 'x.yaml').write_text(
        'key: val\n', encoding='utf-8',
    )

    files = iter_catalog_files(
        tmp_path, exclude_dir_names=('docs',),
    )
    rels = sorted(str(f.relative_to(tmp_path)) for f in files)
    assert 'src/main.py' in rels
    assert not any(r.startswith('docs/') for r in rels), (
        f'docs subtree leaked: {rels}'
    )


def test_iter_catalog_files_default_excludes_nothing_extra(tmp_path):
    """Without exclude params, behavior must match the historical default
    (only ``_SKIP_DIR_NAMES`` and dot-dirs are skipped).
    """
    from docgen.catalog_writer import iter_catalog_files

    (tmp_path / 'regular').mkdir()
    (tmp_path / 'regular' / 'x.py').write_text('x = 1', encoding='utf-8')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'y.md').write_text('# y', encoding='utf-8')

    # No exclude args => docs/ NOT skipped, both files appear.
    files = iter_catalog_files(tmp_path)
    rels = sorted(str(f.relative_to(tmp_path)) for f in files)
    assert 'regular/x.py' in rels
    assert 'docs/y.md' in rels


# ---------------------------------------------------------------------------
# sync_source_catalog: forwards excludes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_source_catalog_forwards_exclude_dir_names(
    tmp_path, monkeypatch,
):
    """``sync_source_catalog(exclude_dir_names=...)`` must reach
    ``iter_catalog_files`` so excluded dirs are never sync'd.
    """
    import numpy as np

    from docgen.catalog_writer import sync_source_catalog
    from library import Library
    from tests._scoped_config_fixture import install_test_config
    from writer import LibraryWriter

    install_test_config(monkeypatch, tmp_path, 'myapp')

    async def fake_embed(self, text):
        return np.zeros(3072, dtype=np.float32)

    async def fake_embed_batch(self, texts):
        return [np.zeros(3072, dtype=np.float32) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr(
        'embedding.EmbeddingService.embed_batch', fake_embed_batch,
    )
    monkeypatch.setattr(
        'embedding.EmbeddingService._get_client', fake_get_client,
    )
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)

    src_root = tmp_path / 'src_root'
    src_root.mkdir()
    (src_root / 'good.py').write_text(
        'class Foo:\n    pass\n', encoding='utf-8',
    )
    (src_root / 'secrets').mkdir()
    (src_root / 'secrets' / 'creds.py').write_text(
        "TOKEN = 'real'\n", encoding='utf-8',
    )

    lib = Library(tmp_path / 'test.db')
    try:
        async with LibraryWriter(lib) as writer:
            summaries = await sync_source_catalog(
                library=lib,
                writer=writer,
                source_name='myapp',
                source_root=src_root,
                exclude_dir_names=('secrets',),
            )
        files_seen = sorted(s.file for s in summaries)
        assert 'good.py' in files_seen
        assert not any('secrets' in s.file for s in summaries)
    finally:
        lib.close()
def test_default_policy_prunes_claude_tooling_dir(tmp_path):
    """`.claude` (Claude Code's project config: settings.local.json, hooks,
    skills) is tooling, not source. The default exclude policy must prune it —
    like .vscode/.idea — so the estimate walk (find_catalog_files) never scans
    it. Without this it leaks into the dry-run, as it did when documenting
    Ariadne itself."""
    from config import DEFAULT_EXCLUDE_POLICY
    from docgen.staleness import find_catalog_files

    (tmp_path / 'app.py').write_text('x = 1\n', encoding='utf-8')
    claude = tmp_path / '.claude'
    claude.mkdir()
    (claude / 'settings.local.json').write_text('{}', encoding='utf-8')
    (claude / 'skills').mkdir()
    (claude / 'skills' / 's.md').write_text('# skill\n', encoding='utf-8')

    found = list(find_catalog_files(
        tmp_path, exclude_patterns=(), exclude_dir_names=DEFAULT_EXCLUDE_POLICY))
    rels = sorted(str(f.relative_to(tmp_path)) for f in found)
    assert 'app.py' in rels
    assert not any('.claude' in r for r in rels), (
        f'.claude tooling dir leaked into the catalog: {rels}'
    )
