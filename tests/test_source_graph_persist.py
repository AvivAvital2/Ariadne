from __future__ import annotations

from pathlib import Path

from config import Config
from docgen.source_graph import persist_source_graph
from library import Library


def _yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body)
    return p


def test_persist_source_graph_snapshots_every_source(tmp_path: Path) -> None:
    """A build persists EVERY configured source's RAW relational fields
    (depends_on / parent / branches) — not the pre-merged effective deps — so a
    serving box can reproduce get_effective_dependencies from the DB alone, with
    no depends_on in its own ariadne.yaml.
    """
    cfg = Config(config_path=_yaml(tmp_path, '''
default_source: app
sources:
  app:
    path: /x/app
    depends_on: [core, shared]
    branches: ["main", "release/*"]
  core:
    path: /x/core
    depends_on: [shared]
  shared:
    path: /x/shared
  webapp:
    path: /x/app/web
    parent: app
'''))
    lib = Library(tmp_path / 'g.db')

    persist_source_graph(cfg, lib)

    assert lib.all_source_relations() == {
        'app': {'depends_on': ['core', 'shared'], 'parent': None, 'branches': ['main', 'release/*']},
        'core': {'depends_on': ['shared'], 'parent': None, 'branches': []},
        'shared': {'depends_on': [], 'parent': None, 'branches': []},
        'webapp': {'depends_on': [], 'parent': 'app', 'branches': []},
    }
    lib.close()
