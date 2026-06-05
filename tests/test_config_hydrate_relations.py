from __future__ import annotations

from pathlib import Path

from config import Config


def _yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body)
    return p


def test_hydrate_merges_yaml_else_db_and_adds_db_only_sources(tmp_path: Path) -> None:
    """hydrate_relations layers the DB-persisted graph onto the yaml config:

    - yaml-when-present-else-DB, PER FIELD (a source that sets depends_on in
      yaml keeps it, but a field it omits is filled from the DB);
    - sources present only in the DB are ADDED, so a serving box that lists no
      depends_on (or doesn't list the dependency at all) still resolves the
      whole closure from the database.

    After hydration the existing closure/dep machinery works unchanged.
    """
    cfg = Config(config_path=_yaml(tmp_path, '''
default_source: app
sources:
  app:
    path: /x/app
    depends_on: [core]
  webapp:
    path: /x/web
'''))

    cfg.hydrate_relations({
        'app':    {'depends_on': ['core', 'shared'], 'parent': None, 'branches': []},     # yaml wins
        'webapp': {'depends_on': ['app'], 'parent': None, 'branches': ['main']},           # fills gaps
        'core':   {'depends_on': ['shared'], 'parent': None, 'branches': []},              # DB-only
        'shared': {'depends_on': [], 'parent': None, 'branches': []},                       # DB-only
    })

    # yaml-present wins: app keeps its declared [core], NOT the DB's wider set.
    assert cfg.get_source_dependencies('app') == ['core']
    # yaml gap filled from the DB: webapp declared no depends_on/branches.
    assert cfg.get_source_dependencies('webapp') == ['app']
    assert cfg.get_source_config('webapp').branches == ('main',)
    # DB-only sources are added so the closure can reach them.
    assert cfg.get_source_dependencies('core') == ['shared']
    assert 'shared' in cfg.sources
    # The whole graph now resolves from the DB: app -> core -> shared.
    assert cfg.scope_closure('app') == frozenset({'app', 'core', 'shared'})
