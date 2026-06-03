"""Shared helper for tests that exercise catalog_writer / notify_changed.

After the Phase 3+ migration, ``docgen/catalog_writer.py`` builds a
``ScopedLibrary`` via ``make_scoped_library(get_config(), library,
source_name)`` for its read operations. Tests that use a source name
not present in any real ``ariadne.yaml`` would otherwise fail-closed
because the global ``get_config()`` doesn't know about the test source.

``install_test_config(monkeypatch, tmp_path, source_name)`` writes a
yaml to a process-unique directory under the OS temp root so the file
never falls inside any test's source_root walk AND never collides
across parallel pytest workers.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import uuid
from pathlib import Path


_CREATED_DIRS: set[Path] = set()


def _cleanup_created_dirs() -> None:
    """Remove every cfg dir we created across this pytest session.

    Without this, ``$TMPDIR`` accumulates one dir per ``install_test_config``
    call — visible as hundreds of stale ``ariadne-test-cfg-*`` dirs on a
    long-running dev machine. atexit fires after pytest's own teardown, so
    the cleanup is best-effort but always runs."""
    for d in list(_CREATED_DIRS):
        shutil.rmtree(d, ignore_errors=True)
    _CREATED_DIRS.clear()


atexit.register(_cleanup_created_dirs)


def install_test_config(
    monkeypatch,
    tmp_path: Path,
    source_name: str | tuple[str, ...] | list[str],
) -> None:
    """Point ``get_config()`` at a throwaway config for the test.

    ``source_name`` may be a single name or an iterable — useful for
    tests that exercise multiple sources (e.g. graph_builder's
    cross-source semantic-edge tests).

    Implementation note: we patch ``config._global_config`` (the cached
    instance the original ``get_config`` returns), NOT ``config.get_config``
    itself. Many modules do ``from config import get_config`` (a bound
    name); replacing the *function* on the ``config`` module doesn't reach
    those copies, and worse, if such a module is first imported while the
    function is patched, its bound name permanently captures the test
    lambda and leaks into every later test. Patching the global the
    original function reads keeps all bound copies correct and leak-free.
    """
    import config as config_module
    from config import Config

    names = (
        (source_name,) if isinstance(source_name, str) else tuple(source_name)
    )
    # Place the yaml at a process-unique path under the OS temp root.
    # This avoids two problems we previously had:
    #   - ``tmp_path / '..' / 'ariadne-cfg-<name>'`` was a session-
    #     shared parent dir, racing under pytest-xdist.
    #   - ``tmp_path / 'subdir'`` was inside the test's source_root
    #     walk; the yaml leaked into catalog-file iteration in tests
    #     whose ``sync_source_catalog`` call bypassed the default
    #     exclude policy.
    # The uuid+pid combination gives per-call uniqueness without
    # cross-test interference; nothing in any source_root walks
    # under ``$TMPDIR`` other than tmp_path itself.
    cfg_dir = Path(tempfile.gettempdir()) / (
        f'ariadne-test-cfg-{os.getpid()}-{uuid.uuid4().hex}'
    )
    cfg_dir.mkdir(parents=True, exist_ok=True)
    _CREATED_DIRS.add(cfg_dir)
    cfg_path = cfg_dir / 'ariadne.yaml'
    body = 'sources:\n' + ''.join(
        f'  {name}:\n    path: {tmp_path}\n' for name in names
    )
    cfg_path.write_text(body, encoding='utf-8')
    cfg = Config(cfg_path)
    monkeypatch.setattr(config_module, '_global_config', cfg)
