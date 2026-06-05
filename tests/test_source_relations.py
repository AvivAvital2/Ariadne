from __future__ import annotations

from pathlib import Path

import pytest

from library import Library


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'relations-test.db')
    yield lib
    lib.close()


def test_source_relations_persist_and_read_back(library: Library) -> None:
    """The build-time relational graph (depends_on + parent + branches) round-trips
    through the DB, so a serving box can resolve scope from the database alone — no
    ariadne.yaml paths or depends_on required.

    This pins the raw store only. Per-field precedence (yaml-when-present-else-DB)
    is layered above it in the config. The store's own contract: what goes in comes
    back, and a rebuild REPLACES the prior graph wholesale rather than accreting
    stale edges.
    """
    # Nothing persisted yet for an unknown source.
    assert library.get_source_relations('app') is None

    library.set_source_relations(
        'app',
        depends_on=['core', 'shared'],
        parent='platform',
        branches=['main', 'release/*'],
    )
    assert library.get_source_relations('app') == {
        'depends_on': ['core', 'shared'],
        'parent': 'platform',
        'branches': ['main', 'release/*'],
    }

    # A second build with a smaller graph REPLACES the prior one (no stale edges).
    library.set_source_relations('app', depends_on=['core'], parent=None, branches=[])
    assert library.get_source_relations('app') == {
        'depends_on': ['core'],
        'parent': None,
        'branches': [],
    }

    # all_source_relations returns the whole graph — the basis for DB-only scope
    # resolution on a serving box that lists no depends_on in yaml.
    library.set_source_relations('core', depends_on=['shared'], parent=None, branches=[])
    assert library.all_source_relations() == {
        'app': {'depends_on': ['core'], 'parent': None, 'branches': []},
        'core': {'depends_on': ['shared'], 'parent': None, 'branches': []},
    }
