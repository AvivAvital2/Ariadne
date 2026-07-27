"""The per-run log captures INFO to a FILE; the console must stay at WARNING by
default (no --debug), else per-request INFO logging floods stderr and shreds the
Rich progress bar. Regression: raising the root logger to INFO for the file
handler must not leak INFO to a pre-existing (NOTSET) console handler.
"""
from __future__ import annotations

import logging

from cli.generate import _run_log_handlers


def test_run_log_pins_console_to_warning_by_default(tmp_path):
    root = logging.getLogger()
    console_handler = logging.StreamHandler()  # mimics basicConfig's NOTSET handler
    root.addHandler(console_handler)
    saved_root = root.level
    try:
        assert console_handler.level == logging.NOTSET
        with _run_log_handlers(tmp_path / 'db.sqlite', verbose=False):
            # File capture needs root at INFO...
            assert root.level == logging.INFO
            # ...but the console handler must be held at WARNING so INFO
            # doesn't leak to stderr and corrupt the bar.
            assert console_handler.level >= logging.WARNING
        # Restored afterward.
        assert console_handler.level == logging.NOTSET
    finally:
        root.removeHandler(console_handler)
        root.setLevel(saved_root)
