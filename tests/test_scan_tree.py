"""Tier 1 of the dry-run explorer — the instant directory/token model.

``scan_tree`` walks a source tree, keeps only the files Ariadne would
document (mapped extensions), counts their tokens, prunes policy dirs,
and rolls ``{mapped_files, content_tokens, total_bytes}`` up a nested
``ScanNode``. No UI, no cost, no writes.

These started as one evolving ``test_scan`` (the Tier-1 design's method)
and were split into the focused tests below once the behavior — and 100%
branch coverage of ``scan_tree.py`` — was nailed down. All fixtures are
synthetic: neutral names, tiny known-content files.
"""
from __future__ import annotations

import os

from docgen.pricing import CHARS_PER_TOKEN

from docgen.scan_tree import scan_tree


def _child(node, name):
    """The single child of ``node`` named ``name``."""
    matches = [c for c in node.children if c.name == name]
    assert len(matches) == 1, (
        f'expected one {name!r}, got {[c.name for c in node.children]}'
    )
    return matches[0]


def test_scan_builds_tree(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'mod_a.py').write_text('def a():\n    return 1\n')
    (src / 'mod_b.py').write_text('def b():\n    return 2\n')

    root = scan_tree(tmp_path, excluded_dirs=frozenset())

    assert root.is_dir
    assert root.rel_path == '.'
    assert root.mapped_files == 2
    assert root.content_tokens > 0

    src_node = _child(root, 'src')
    assert src_node.is_dir
    assert src_node.rel_path == 'src'
    assert src_node.mapped_files == 2
    assert src_node.content_tokens == root.content_tokens


def test_scan_excludes_unmapped(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'mod_a.py').write_text('def a():\n    return 1\n')
    (src / 'logo.png').write_bytes(b'\x89PNG' + b'\x00' * 500)

    src_node = _child(scan_tree(tmp_path, excluded_dirs=frozenset()), 'src')

    assert src_node.mapped_files == 1  # png is not a documented file
    png = _child(src_node, 'logo.png')
    assert png.mapped_files == 0
    assert png.content_tokens == 0
    assert png.total_bytes == 504  # but its bytes still count
    assert src_node.total_bytes >= 504


def test_scan_prunes_policy_dirs(tmp_path):
    (tmp_path / 'keep.py').write_text('keep = 1\n')
    vendored = tmp_path / 'node_modules'
    vendored.mkdir()
    (vendored / 'dep.js').write_text('export const x = 1;\n')

    root = scan_tree(tmp_path, excluded_dirs=frozenset({'node_modules'}))

    assert all(c.name != 'node_modules' for c in root.children)
    assert root.mapped_files == 1  # dep.js (.js, mapped) pruned, adds nothing


def test_scan_rolls_up_aggregates(tmp_path):
    src = tmp_path / 'src'
    api = src / 'api'
    api.mkdir(parents=True)
    (src / 'mod_a.py').write_text('def a():\n    return 1\n')
    (api / 'handler.py').write_text('def handle(req):\n    return req.ok\n')

    root = scan_tree(tmp_path, excluded_dirs=frozenset())
    src_node = _child(root, 'src')
    api_node = _child(src_node, 'api')

    assert src_node.mapped_files == 2
    assert src_node.content_tokens == sum(c.content_tokens for c in src_node.children)
    assert root.content_tokens == src_node.content_tokens
    assert src_node.content_tokens > api_node.content_tokens > 0


def test_scan_sorts_dirs_then_files(tmp_path):
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'inner.py').write_text('x = 1\n')
    (tmp_path / 'big.py').write_text('def big():\n' + '    x = 1\n' * 40)
    (tmp_path / 'small.py').write_text('def small():\n    return 0\n')
    (tmp_path / 'notes.txt').write_text('unmapped\n')  # 0 tokens, sorts last

    children = scan_tree(tmp_path, excluded_dirs=frozenset()).children

    # directories first, then files
    kinds = [c.is_dir for c in children]
    assert kinds == sorted(kinds, reverse=True)
    # files ordered by content_tokens descending
    files = [c for c in children if not c.is_dir]
    assert [f.name for f in files] == ['big.py', 'small.py', 'notes.txt']
    assert files[0].content_tokens > files[1].content_tokens > files[2].content_tokens == 0


def test_scan_token_fallback(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    mod_a = src / 'mod_a.py'
    mod_a.write_text('def a():\n    return 1\n')
    expected = int(mod_a.stat().st_size // CHARS_PER_TOKEN)

    def _none_counter(_path):
        return None  # tiktoken unavailable

    def _raising_counter(_path):
        raise RuntimeError('tiktoken exploded')  # unreadable / broken

    for counter in (_none_counter, _raising_counter):
        root = scan_tree(tmp_path, excluded_dirs=frozenset(), token_counter=counter)
        assert _child(_child(root, 'src'), 'mod_a.py').content_tokens == expected


def test_scan_empty_dir(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'data.bin').write_bytes(b'\x00' * 256)

    assets_node = _child(scan_tree(tmp_path, excluded_dirs=frozenset()), 'assets')

    assert assets_node.is_dir
    assert assets_node.mapped_files == 0
    assert assets_node.content_tokens == 0
    assert assets_node.total_bytes == 256


def test_scan_skips_symlinks_and_unreadable(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'mod_a.py').write_text('def a():\n    return 1\n')

    # symlink must not be followed
    os.symlink(src, tmp_path / 'src_link')
    # wholly unreadable dir: listed, but yields no children (iterdir fails)
    locked = tmp_path / 'locked'
    locked.mkdir()
    (locked / 'hidden.py').write_text('secret = 1\n')
    locked.chmod(0o000)
    # listable-but-not-traversable dir: names enumerate, entries can't be
    # statted, so each is skipped (neither dir nor file)
    readonly = tmp_path / 'readonly'
    readonly.mkdir()
    (readonly / 'inner.py').write_text('x = 1\n')
    readonly.chmod(0o400)
    try:
        root = scan_tree(tmp_path, excluded_dirs=frozenset())
        assert all(c.name != 'src_link' for c in root.children)
        assert _child(root, 'locked').children == ()
        assert _child(root, 'locked').mapped_files == 0
        assert _child(root, 'readonly').children == ()
    finally:
        locked.chmod(0o755)
        readonly.chmod(0o755)
