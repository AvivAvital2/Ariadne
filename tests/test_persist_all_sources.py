"""Tests for ``persist_all_sources`` — the data-path closer for SCIP.

Without this helper, ``library_scip.scip_symbols`` / ``scip_edges``
tables stay empty in production: every ``CrossSourceGraph.save_to``
call site that previously existed was in unit tests. ``ariadne index``
now invokes this helper at the end of its run so downstream readers
(``ariadne callers``, ``impact_radius``, ``improve --dead-code``, and
the catalog generator's architecture-prompt Dependents section) have
the data they need.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _write_manifest(source_root: Path, indexer_entries: list[dict]) -> None:
    """Drop a minimal ``.ariadne/manifest.json`` into ``source_root``."""
    manifest_dir = source_root / '.ariadne'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / 'manifest.json').write_text(
        json.dumps({'indexers': indexer_entries}),
        encoding='utf-8',
    )


def _synthetic_python_index_with_one_class(repo: str = 'svc'):
    """Return a minimal ScipIndex with one Python class definition.

    Used as the return value of an ``index_factory`` so tests can
    exercise ``persist_all_sources`` without touching scip-python or
    reading a real .scip file.
    """
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    cls_sym = f'scip-python python {repo} 0.1 service/Service#'
    doc = _ScipDoc(
        relative_path='service.py',
        occurrences=(
            _ScipOccurrence(
                symbol=cls_sym, range=(0, 6, 0, 13), is_definition=True,
            ),
        ),
        symbols=(_ScipSymbol(
            symbol=cls_sym, kind='Class', display_name='Service',
        ),),
    )
    return ScipIndex(documents=(doc,))


def test_persist_all_sources_no_manifests_returns_zero(tmp_path: Path) -> None:
    """Sources with no ``.ariadne/manifest.json`` get silently skipped.

    persist_all_sources is meant to run optimistically after every
    ``ariadne index`` — a missing manifest is "discover wasn't run
    here yet", not an error condition.
    """
    from docgen.scip_persist import persist_all_sources

    db_path = tmp_path / 'ariadne.db'
    a_root = tmp_path / 'a'
    a_root.mkdir()
    b_root = tmp_path / 'b'
    b_root.mkdir()

    persisted = persist_all_sources(
        db_path, [('a', a_root), ('b', b_root)],
    )
    assert persisted == 0


def test_persist_all_sources_writes_scip_symbols_from_manifest(
    tmp_path: Path,
) -> None:
    """Given a source with a manifest + injected index_factory, the
    helper materializes the graph and the symbols land in
    ``library_scip.scip_symbols``. Bites if ``save_to`` ever stops
    being called, or if the manifest walk misses an entry."""
    from docgen.scip_persist import persist_all_sources

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'svc'
    source_root.mkdir()

    # Manifest declares one python indexer entry. The scip_path
    # value is relative to .ariadne/; the index_factory we inject
    # ignores it (we return the synthetic ScipIndex directly).
    _write_manifest(source_root, [{
        'kind': 'python',
        'scip_path': 'intermediate/index-python.scip',
    }])

    def _factory(scip_path, *, repo, max_staleness_days):
        return _synthetic_python_index_with_one_class(repo='svc')

    persisted = persist_all_sources(
        db_path, [('svc', source_root)],
        index_factory=_factory,
    )
    assert persisted == 1

    # The helper wrote rows to scip_symbols. Read raw — Library opening
    # the same file would re-run init_scip_schema, but the table is
    # already present from the helper's save path.
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT source_name, qualified_name, kind "
            "FROM scip_symbols WHERE source_name = 'svc'",
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, f'expected one symbol row, got {rows}'
    source_name, qn, kind = rows[0]
    assert source_name == 'svc'
    assert 'Service' in qn
    assert kind == 'Class'


def test_persist_all_sources_skips_failing_source(tmp_path: Path) -> None:
    """One source raising during load_source_from_manifest must not
    forfeit persistence for the others. The healthy source still lands
    in scip_symbols."""
    from docgen.scip_persist import persist_all_sources

    db_path = tmp_path / 'ariadne.db'

    good_root = tmp_path / 'good'
    good_root.mkdir()
    _write_manifest(good_root, [{
        'kind': 'python',
        'scip_path': 'intermediate/index-python.scip',
    }])

    bad_root = tmp_path / 'bad'
    bad_root.mkdir()
    _write_manifest(bad_root, [{
        'kind': 'python',
        'scip_path': 'intermediate/index-python.scip',
    }])

    def _factory(scip_path, *, repo, max_staleness_days):
        if repo == 'bad':
            msg = 'simulated scip read failure'
            raise RuntimeError(msg)
        return _synthetic_python_index_with_one_class(repo=repo)

    persisted = persist_all_sources(
        db_path, [('good', good_root), ('bad', bad_root)],
        index_factory=_factory,
    )
    assert persisted == 1  # only good was successfully loaded

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT DISTINCT source_name FROM scip_symbols',
        ).fetchall()
    finally:
        conn.close()

    source_names = {r[0] for r in rows}
    assert source_names == {'good'}


# ---------------------------------------------------------------------------
# scip_index_state — per-source bookkeeping with SHA + indexed_at
# ---------------------------------------------------------------------------


def test_persist_all_sources_writes_scip_index_state_when_merged_exists(
    tmp_path: Path,
) -> None:
    """When ``<source>/.ariadne/index.scip`` exists on disk, persisting
    the source also writes one row into ``scip_index_state`` carrying
    the merged file's SHA256, the absolute path, and an
    ``indexed_at`` timestamp.

    This is the staleness contract: a DB-queryable surface for
    ``last_indexed_at`` per source. Bites if the INSERT ever stops
    being called or if the SHA doesn't reflect the actual file.
    """
    import hashlib

    from docgen.scip_persist import persist_all_sources

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'svc'
    source_root.mkdir()

    _write_manifest(source_root, [{
        'kind': 'python',
        'scip_path': 'intermediate/index-python.scip',
    }])

    # Drop a fake merged .scip so the staleness-row write path fires.
    merged_path = source_root / '.ariadne' / 'index.scip'
    merged_bytes = b'\x08\x01synthetic-merged-payload'
    merged_path.write_bytes(merged_bytes)
    expected_sha = hashlib.sha256(merged_bytes).hexdigest()

    def _factory(scip_path, *, repo, max_staleness_days):
        return _synthetic_python_index_with_one_class(repo=repo)

    persisted = persist_all_sources(
        db_path, [('svc', source_root)], index_factory=_factory,
    )
    assert persisted == 1

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT source_name, scip_path, file_sha256, '
            'indexed_at, indexer_version '
            "FROM scip_index_state WHERE source_name = 'svc'",
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    source_name, scip_path, file_sha, indexed_at, indexer_version = rows[0]
    assert source_name == 'svc'
    assert Path(scip_path) == merged_path.resolve() or Path(scip_path) == merged_path
    assert file_sha == expected_sha, (
        f'scip_index_state.file_sha256 must hash the actual merged '
        f'.scip; got {file_sha!r} expected {expected_sha!r}'
    )
    assert indexed_at, 'indexed_at should be a non-empty ISO timestamp'
    # indexer_version is left NULL — per-indexer versions live in
    # manifest.json since a polyglot source has multiple indexers.
    assert indexer_version is None


def test_persist_all_sources_upserts_scip_index_state_on_re_persist(
    tmp_path: Path,
) -> None:
    """Re-running persist with a different merged .scip updates the
    row in place (PRIMARY KEY = source_name). Bites if anyone removes
    ``INSERT OR REPLACE`` and the table grows duplicates across runs.
    """
    import hashlib

    from docgen.scip_persist import persist_all_sources

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'svc'
    source_root.mkdir()
    _write_manifest(source_root, [{
        'kind': 'python',
        'scip_path': 'intermediate/index-python.scip',
    }])
    merged_path = source_root / '.ariadne' / 'index.scip'

    def _factory(scip_path, *, repo, max_staleness_days):
        return _synthetic_python_index_with_one_class(repo=repo)

    # First persist — initial bytes
    merged_path.write_bytes(b'first-payload')
    persist_all_sources(
        db_path, [('svc', source_root)], index_factory=_factory,
    )

    # Second persist — different bytes
    merged_path.write_bytes(b'second-payload-different')
    persist_all_sources(
        db_path, [('svc', source_root)], index_factory=_factory,
    )

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT file_sha256 FROM scip_index_state "
            "WHERE source_name = 'svc'",
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, (
        f'expected exactly one row after upsert; got {len(rows)}'
    )
    expected_sha = hashlib.sha256(b'second-payload-different').hexdigest()
    assert rows[0][0] == expected_sha, (
        'second persist must overwrite the row with the new SHA'
    )


def test_persist_all_sources_skips_scip_index_state_when_no_merged_file(
    tmp_path: Path,
) -> None:
    """When the merged ``index.scip`` doesn't exist (e.g., index
    was --dry-run, or merge failed and discover wrote the manifest
    but no artifact), ``scip_symbols``/``scip_edges`` still fill from
    the per-indexer intermediates, but ``scip_index_state`` stays
    empty for that source. Better an absent row than a row with
    fabricated SHA.
    """
    from docgen.scip_persist import persist_all_sources

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'svc'
    source_root.mkdir()
    _write_manifest(source_root, [{
        'kind': 'python',
        'scip_path': 'intermediate/index-python.scip',
    }])
    # NOTE: no merged index.scip on disk.

    def _factory(scip_path, *, repo, max_staleness_days):
        return _synthetic_python_index_with_one_class(repo=repo)

    persisted = persist_all_sources(
        db_path, [('svc', source_root)], index_factory=_factory,
    )
    assert persisted == 1  # graph data still persists

    conn = sqlite3.connect(db_path)
    try:
        symbol_count = conn.execute(
            "SELECT COUNT(*) FROM scip_symbols WHERE source_name = 'svc'",
        ).fetchone()[0]
        state_count = conn.execute(
            "SELECT COUNT(*) FROM scip_index_state "
            "WHERE source_name = 'svc'",
        ).fetchone()[0]
    finally:
        conn.close()

    assert symbol_count == 1, (
        'graph symbols should still persist without a merged file'
    )
    assert state_count == 0, (
        'no merged index.scip → no scip_index_state row '
        '(refuse to fabricate a SHA)'
    )
