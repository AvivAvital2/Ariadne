"""Unit tests for the generation-completion marker (docgen/generation_marker).

The marker records that a pinned/immutable spool corpus is fully generated at a
given sha, so the generate step can skip discovery entirely on a re-run.
"""

from __future__ import annotations

from pathlib import Path

from docgen.generation_marker import (
    MARKER_NAME,
    generation_complete,
    invalidate_marker,
    read_marker,
    write_marker,
)


def _ariadne(tmp_path: Path) -> Path:
    d = tmp_path / '.ariadne'
    d.mkdir()
    return d


def _pin(tmp_path: Path, sha: str, repo: str = 'repo') -> None:
    (tmp_path / repo).mkdir(exist_ok=True)
    (tmp_path / repo / '.ariadne-corpus-sha').write_text(sha + '\n', encoding='utf-8')


def test_write_read_round_trip(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    write_marker(d, corpus_shas={'repo': 'sha1'}, doc_types=('explanation', 'gotcha'))
    m = read_marker(d)
    assert m is not None
    assert m['corpus_shas'] == {'repo': 'sha1'}
    assert m['doc_types'] == ['explanation', 'gotcha']  # sorted


def test_write_is_atomic_no_tmp_left(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    write_marker(d, corpus_shas={}, doc_types=())
    assert (d / MARKER_NAME).exists()
    assert not (d / (MARKER_NAME + '.tmp')).exists()


def test_read_missing_returns_none(tmp_path: Path) -> None:
    assert read_marker(_ariadne(tmp_path)) is None


def test_read_corrupt_returns_none(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    (d / MARKER_NAME).write_text('{ not json', encoding='utf-8')
    assert read_marker(d) is None


def test_invalidate_removes_marker(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    write_marker(d, corpus_shas={}, doc_types=())
    invalidate_marker(d)
    assert not (d / MARKER_NAME).exists()
    invalidate_marker(d)  # idempotent


def test_complete_true_when_sha_matches_and_types_covered(tmp_path: Path) -> None:
    _pin(tmp_path, 'sha1')
    d = _ariadne(tmp_path)
    write_marker(d, corpus_shas={'repo': 'sha1'}, doc_types=('explanation', 'gotcha'))
    assert generation_complete(d, tmp_path, requested_doc_types=('explanation',)) is True


def test_complete_false_when_marker_absent(tmp_path: Path) -> None:
    _pin(tmp_path, 'sha1')
    assert generation_complete(_ariadne(tmp_path), tmp_path, requested_doc_types=('explanation',)) is False


def test_complete_false_when_sha_differs(tmp_path: Path) -> None:
    d = _ariadne(tmp_path)
    write_marker(d, corpus_shas={'repo': 'sha1'}, doc_types=('explanation',))
    _pin(tmp_path, 'sha2')  # corpus re-pinned to a new version
    assert generation_complete(d, tmp_path, requested_doc_types=('explanation',)) is False


def test_complete_false_when_requested_doc_type_not_covered(tmp_path: Path) -> None:
    _pin(tmp_path, 'sha1')
    d = _ariadne(tmp_path)
    write_marker(d, corpus_shas={'repo': 'sha1'}, doc_types=('explanation',))
    # architecture was never generated -> not complete for that request
    assert generation_complete(d, tmp_path, requested_doc_types=('explanation', 'architecture')) is False
