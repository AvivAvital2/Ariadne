"""Tests for the gotcha feature in Ariadne."""
from __future__ import annotations

from pathlib import Path

import pytest

from library import Library
from schema import CONTENT_TYPES


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / 'test_gotcha.db'


@pytest.fixture
def library(temp_db: Path) -> Library:
    lib = Library(temp_db)
    yield lib
    lib.close()


class TestGotcha:
    def test_gotcha_in_content_type(self) -> None:
        """The 'gotcha' value must be present in ContentType literals."""
        assert 'gotcha' in CONTENT_TYPES

    def test_get_gotchas_empty(self, library: Library) -> None:
        """get_gotchas returns an empty list when no gotchas match."""
        result = library.get_gotchas(['nonexistent.py'])
        assert result == []

    def test_deprecate_stale_gotchas_no_gotchas(self, library: Library) -> None:
        """deprecate_stale_gotchas returns 0 when no gotchas exist."""
        count = library.deprecate_stale_gotchas()
        assert count == 0
