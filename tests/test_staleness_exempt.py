"""Staleness-exempt sources (e.g. a pinned/immutable spool corpus) must not pay
the per-file content-hash cost. A file that already has a record is
definitively not-stale for an exempt source — regardless of content changes —
so ``is_stale`` must decide that WITHOUT reading and hashing the file.
"""

from __future__ import annotations

import docgen.staleness as staleness_mod
from docgen.staleness import StalenessTracker


def _boom(_path):
    raise AssertionError("is_stale computed a file hash for a staleness-exempt path")


def test_exempt_recorded_file_is_not_stale_without_hashing(tmp_path, monkeypatch):
    src = tmp_path / "mod.py"
    src.write_text("x = 1\n", encoding="utf-8")
    with StalenessTracker(tmp_path / "stale.db") as tracker:
        tracker.record_documentation(src, ["doc1"], base_path=tmp_path)
        # Content changes so the hash WOULD differ from the recorded one.
        src.write_text("x = 2  # changed\n", encoding="utf-8")
        # From here, any file-hash computation inside is_stale is the wasted
        # work we're eliminating for exempt sources.
        monkeypatch.setattr(staleness_mod, "_compute_file_hash", _boom)
        assert (
            tracker.is_stale(src, base_path=tmp_path, is_exempt=lambda rel: True)
            is False
        )


def test_exempt_undocumented_file_is_still_stale(tmp_path):
    """Exemption must NOT suppress new-file detection: an undocumented file is
    still stale (needs generation) even for an exempt source."""
    src = tmp_path / "new.py"
    src.write_text("y = 1\n", encoding="utf-8")
    with StalenessTracker(tmp_path / "stale.db") as tracker:
        assert (
            tracker.is_stale(src, base_path=tmp_path, is_exempt=lambda rel: True)
            is True
        )


def test_non_exempt_changed_file_is_stale(tmp_path):
    """Guard: the reorder must not weaken normal staleness — a non-exempt file
    whose content changed is still stale."""
    src = tmp_path / "mod.py"
    src.write_text("x = 1\n", encoding="utf-8")
    with StalenessTracker(tmp_path / "stale.db") as tracker:
        tracker.record_documentation(src, ["doc1"], base_path=tmp_path)
        src.write_text("x = 2  # changed\n", encoding="utf-8")
        assert tracker.is_stale(src, base_path=tmp_path) is True
