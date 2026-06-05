from __future__ import annotations

from pathlib import Path

from config import Config
from docgen.source_graph import persist_source_graph
from library import Library


def _yaml(d: Path, body: str) -> Path:
    p = d / 'ariadne.yaml'
    p.write_text(body)
    return p


def test_serve_from_db_end_to_end(tmp_path: Path) -> None:
    """The whole point, stitched together: a build persists the graph; a serving
    box copies ONLY ariadne.db and lists source NAMES (no paths, no depends_on);
    scope then resolves the entire closure from the database.
    """
    build_dir = tmp_path / 'build'
    build_dir.mkdir()
    serve_dir = tmp_path / 'serve'
    serve_dir.mkdir()

    # BUILD: full config (paths + graph) → persist into the DB.
    build_cfg = Config(config_path=_yaml(build_dir, '''
sources:
  app:
    path: /build/app
    depends_on: [core]
  core:
    path: /build/core
    depends_on: [shared]
  shared:
    path: /build/shared
'''))
    db = Library(tmp_path / 'ariadne.db')
    persist_source_graph(build_cfg, db)

    # SERVE: only the DB travels; the yaml lists names — no paths, no depends_on.
    serve_cfg = Config(config_path=_yaml(serve_dir, '''
sources:
  app:
'''))
    serve_cfg.hydrate_relations(db.all_source_relations())

    assert serve_cfg.scope_closure('app') == frozenset({'app', 'core', 'shared'})
    assert serve_cfg.get_source_path('app') is None        # serve-only, path-less
    db.close()
