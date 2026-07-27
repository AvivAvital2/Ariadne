"""Unit tests for the SCIP index completion marker (docgen/scip_index_marker).

The marker is what makes "a .scip exists" safe to trust for reuse: it is written
atomically only after a successful index and records the corpus shas the index
was built from.
"""

from __future__ import annotations

from pathlib import Path

from docgen.scip_index_marker import (
    MARKER_NAME,
    current_corpus_shas,
    index_complete,
    invalidate_marker,
    read_marker,
    write_marker,
)


def _ariadne(tmp_path: Path) -> Path:
    d = tmp_path / '.ariadne'
    d.mkdir()
    return d


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    write_marker(d, indexer_versions={'a.scip': 'scip-python/1'}, corpus_shas={'r': 'abc'})
    data = read_marker(d)
    assert data is not None
    assert data['indexer_versions'] == {'a.scip': 'scip-python/1'}
    assert data['corpus_shas'] == {'r': 'abc'}


def test_write_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    write_marker(d, indexer_versions={}, corpus_shas={})
    assert (d / MARKER_NAME).exists()
    assert not (d / (MARKER_NAME + '.tmp')).exists()  # temp renamed away


def test_read_missing_returns_none(tmp_path: Path) -> None:
    assert read_marker(_ariadne(tmp_path)) is None


def test_read_corrupt_returns_none(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    (d / MARKER_NAME).write_text('{not valid json', encoding='utf-8')
    assert read_marker(d) is None  # fail-closed


def test_invalidate_removes_marker(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    write_marker(d, indexer_versions={}, corpus_shas={})
    invalidate_marker(d)
    assert not (d / MARKER_NAME).exists()
    invalidate_marker(d)  # idempotent — no error when already gone


def test_index_complete_true_when_present_and_shas_match(tmp_path: Path) -> None:
    (tmp_path / 'repo').mkdir()
    (tmp_path / 'repo' / '.ariadne-corpus-sha').write_text('sha1\n', encoding='utf-8')
    d = _ariadne(tmp_path)
    write_marker(d, indexer_versions={}, corpus_shas={'repo': 'sha1'})
    assert index_complete(d, tmp_path) is True


def test_index_complete_false_when_marker_absent(tmp_path: Path) -> None:
    assert index_complete(_ariadne(tmp_path), tmp_path) is False


def test_index_complete_false_when_shas_differ(tmp_path: Path) -> None:
    (tmp_path / 'repo').mkdir()
    (tmp_path / 'repo' / '.ariadne-corpus-sha').write_text('sha2\n', encoding='utf-8')
    d = _ariadne(tmp_path)
    write_marker(d, indexer_versions={}, corpus_shas={'repo': 'sha1'})  # pin moved
    assert index_complete(d, tmp_path) is False


def test_current_corpus_shas_scans_and_skips_unreadable(tmp_path: Path) -> None:
    (tmp_path / 'good').mkdir()
    (tmp_path / 'good' / '.ariadne-corpus-sha').write_text('deadbeef\n', encoding='utf-8')
    # A clone whose marker is a DIRECTORY (unreadable) is skipped, not fatal.
    (tmp_path / 'bad').mkdir()
    (tmp_path / 'bad' / '.ariadne-corpus-sha').mkdir()
    assert current_corpus_shas(tmp_path) == {'good': 'deadbeef'}
