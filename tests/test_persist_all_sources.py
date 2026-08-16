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
import shutil
import sqlite3
from pathlib import Path
from docgen.scip_extractor import ScipIndex as _ExtractorScipIndex
from docgen.scip_persist import persist_all_sources
from docgen.scip_extractor import _ScipDoc, _ScipOccurrence, _ScipSymbol


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


def _spool_index_defining_spark():
    """A spool ScipIndex defining ``pyspark.sql.SparkSession``."""
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    d = 'scip-python python pyspark 0.1 `pyspark.sql`/SparkSession#'
    doc = _ScipDoc(
        relative_path='pyspark/sql/session.py',
        occurrences=(
            _ScipOccurrence(symbol=d, range=(0, 0, 5, 0), is_definition=True),
        ),
        symbols=(_ScipSymbol(symbol=d, kind='Class',
                             display_name='SparkSession'),),
    )
    return ScipIndex(documents=(doc,))


def _user_index_calling_spark():
    """A user ScipIndex whose ``run()`` references SparkSession via a
    different-version moniker (the canonical-id-mismatch wall)."""
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    run = 'scip-python python userrepo 0.1 `app.main`/run().'
    ref = 'scip-python python pyspark 3.5 `pyspark.sql`/SparkSession#'
    doc = _ScipDoc(
        relative_path='app/main.py',
        occurrences=(
            _ScipOccurrence(symbol=run, range=(10, 0, 20, 0),
                            is_definition=True),
            _ScipOccurrence(symbol=ref, range=(12, 4, 12, 30),
                            is_definition=False),
        ),
        symbols=(_ScipSymbol(symbol=run, kind='Function',
                             display_name='run'),),
    )
    return ScipIndex(documents=(doc,))


def _merged_multi_project_index():
    """One source's MERGED index: project B defines ``spark.sql.Dataset``,
    project A references it from inside ``DeltaTable`` via a moniker carrying a
    different package coordinate.

    This is the shape a merged multi-project corpus actually has — several
    ``.scip`` files combined under ONE source name, where a reference crossing a
    project boundary keeps that project's own package/version coordinates and so
    misses the canonical-id lookup.
    """
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    dataset_def = 'scip-java java spark-core 0.1 `spark.sql`/Dataset#'
    dataset_ref = 'scip-java java spark-core 3.5.0 `spark.sql`/Dataset#'
    table_def = 'scip-java java delta 0.1 `delta.tables`/DeltaTable#'

    spark_doc = _ScipDoc(
        relative_path='sql/core/src/main/scala/spark/sql/Dataset.scala',
        occurrences=(
            _ScipOccurrence(symbol=dataset_def, range=(0, 0, 40, 0),
                            is_definition=True),
        ),
        symbols=(_ScipSymbol(symbol=dataset_def, kind='Class',
                             display_name='Dataset'),),
    )
    delta_doc = _ScipDoc(
        relative_path='spark/src/main/scala/delta/tables/DeltaTable.scala',
        occurrences=(
            _ScipOccurrence(symbol=table_def, range=(10, 0, 30, 0),
                            is_definition=True),
            # DeltaTable's body uses Dataset — a real cross-project dependency
            _ScipOccurrence(symbol=dataset_ref, range=(15, 8, 15, 15),
                            is_definition=False),
        ),
        symbols=(_ScipSymbol(symbol=table_def, kind='Class',
                             display_name='DeltaTable'),),
    )
    return ScipIndex(documents=(spark_doc, delta_doc))


def test_persist_resolves_refs_within_a_bare_named_source(tmp_path: Path) -> None:
    """The resolvable set must be keyed on the sources actually loaded, not on a
    ``spool:`` name prefix.

    SCIP rows are written under the BARE source name, so a prefix test never
    matches anything the SCIP tier emits and external-reference resolution never
    fires in production. A merged multi-project corpus indexed under one bare
    name consequently drops every reference that crosses a project boundary,
    leaving the call graph as disconnected per-project islands.
    """
    from docgen.scip_persist import persist_all_sources

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'databricks'
    source_root.mkdir()
    _write_manifest(source_root, [{
        'kind': 'java',
        'scip_path': 'intermediate/index-java.scip',
    }])

    def _factory(scip_path, *, repo, max_staleness_days):
        return _merged_multi_project_index()

    persist_all_sources(
        db_path, [('databricks', source_root)], index_factory=_factory,
    )

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT caller_canonical_id, callee_canonical_id, confidence '
            'FROM scip_edges',
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, (
        'expected DeltaTable -> Dataset to survive as a resolved edge, '
        f'got {rows}'
    )
    caller, callee, confidence = rows[0]
    assert 'DeltaTable' in caller and 'Dataset' in callee
    assert confidence == 'resolved'


def test_persist_resolves_user_ref_into_enabled_spool(tmp_path: Path) -> None:
    """When a spool source is among those persisted, a user reference to the
    spool's API (dropped by decision #4 today) is resolved by qualified name
    and lands in ``scip_edges`` as a ``confidence='resolved'`` cross-source
    edge — the removed wall. Non-spool corpora are unaffected (no spool loaded
    => no resolution)."""
    from docgen.scip_persist import persist_all_sources

    db_path = tmp_path / 'ariadne.db'
    user_root = tmp_path / 'userrepo'
    user_root.mkdir()
    spool_root = tmp_path / 'spool'
    spool_root.mkdir()
    for root in (user_root, spool_root):
        _write_manifest(root, [{
            'kind': 'python',
            'scip_path': 'intermediate/index-python.scip',
        }])

    def _factory(scip_path, *, repo, max_staleness_days):
        if repo == 'spool:databricks':
            return _spool_index_defining_spark()
        return _user_index_calling_spark()

    persist_all_sources(
        db_path,
        [('userrepo', user_root), ('spool:databricks', spool_root)],
        index_factory=_factory,
    )

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT caller_canonical_id, callee_canonical_id, confidence '
            "FROM scip_edges WHERE confidence = 'resolved'",
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, f'expected one resolved cross-source edge, got {rows}'
    caller, callee, conf = rows[0]
    assert 'run' in caller and 'SparkSession' in callee
    assert conf == 'resolved'


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


class TestStalenessIsHonoured:
    """A source configured to ignore staleness must not hit the 7-day default.

    ``cli/index.py`` computes ``max_staleness_by_source`` from
    ``effective_scip_staleness_days`` and threads it into
    ``persist_data_model`` — but not into ``persist_all_sources``, which had no
    such parameter. So the load fell back to ``load_source_from_manifest``'s
    default of 7 days.

    Measured on the databricks spool corpus: 48 of 50 manifest entries were
    ~11 days old, the first one raised ``ScipTooStaleError``, the blanket
    ``except Exception: continue`` swallowed it, and the run reported
    "Persisted cross-source graph (7 source(s))" while silently dropping the
    source whose 638MB index had just been rebuilt. The configured intent
    never reached the code that enforces it.
    """

    def test_configured_staleness_reaches_the_loader(self, tmp_path):
        from docgen.scip_persist import persist_all_sources

        root = tmp_path / 'svc'
        root.mkdir()
        _write_manifest(root, [{'kind': 'python',
                                'scip_path': 'intermediate/svc.scip'}])
        seen = {}

        def factory(path, *, repo, max_staleness_days):
            seen[repo] = max_staleness_days
            return _synthetic_python_index_with_one_class()

        persist_all_sources(
            tmp_path / 'a.db', [('svc', root)], index_factory=factory,
            max_staleness_by_source={'svc': None},
        )
        assert seen == {'svc': None}, (
            'a source that opts out of staleness must reach the loader with '
            f'None, not the 7-day default; got {seen}'
        )

    def test_unlisted_source_keeps_the_default(self, tmp_path):
        """Absent configuration still gets the conservative default."""
        from docgen.scip_persist import persist_all_sources

        root = tmp_path / 'other'
        root.mkdir()
        _write_manifest(root, [{'kind': 'python',
                                'scip_path': 'intermediate/other.scip'}])
        seen = {}

        def factory(path, *, repo, max_staleness_days):
            seen[repo] = max_staleness_days
            return _synthetic_python_index_with_one_class()

        persist_all_sources(
            tmp_path / 'b.db', [('other', root)], index_factory=factory,
            max_staleness_by_source={'svc': None},
        )
        assert seen == {'other': 7}

    def test_a_skipped_source_is_announced(self, tmp_path, caplog):
        """A load failure must be audible, not swallowed.

        The blanket ``except Exception: continue`` is deliberate — one bad
        source must not forfeit persistence for the rest — but silence made it
        indistinguishable from success. A 20-minute scip-java rebuild was
        discarded this way while the run printed "Persisted cross-source graph
        (7 source(s))" about seven *other* sources.
        """
        import logging

        from docgen.scip_persist import persist_all_sources

        root = tmp_path / 'brokensrc'
        root.mkdir()
        _write_manifest(root, [{'kind': 'python',
                                'scip_path': 'intermediate/brokensrc.scip'}])

        def exploding(path, *, repo, max_staleness_days):
            raise RuntimeError('index unreadable')

        with caplog.at_level(logging.WARNING, logger='docgen.scip_persist'):
            n = persist_all_sources(
                tmp_path / 'c.db', [('brokensrc', root)],
                index_factory=exploding)
        assert n == 0
        messages = [r.getMessage() for r in caplog.records]
        assert any('brokensrc' in m for m in messages), \
            f'the skip must be logged; got {messages}'
        assert any('index unreadable' in m for m in messages), \
            'the cause must be reported, not just the fact of a skip'
class TestTheStoreConvergesOnDiskTruth:
    """Graph rows survive only while something on disk can still refresh them.

    ``clear_source`` and ``save_to`` are scoped to the sources registered in the
    current run -- correctly, since re-persisting one source must not wipe another.
    The consequence was that a source disk no longer backs could never be reached
    again. Measured on the live store: a corpus whose code had been deleted still
    owned 487 symbols and 563,601 edges, and ``local 0`` -- a local binding in one of
    its deleted files -- was the endpoint of 16,061 edges drawn from 5,976 files
    across three OTHER sources. No re-index could remove it, because removal only
    ever happens for a source that registers.

    So persistence reconciles what the store holds against what the run could find.
    A source with no manifest cannot be refreshed by anything, and its graph rows go.
    A source whose artifact IS present but failed to load -- stale, corrupt,
    unreadable -- is a different case, and its rows are KEPT: deleting them would
    turn a transient read failure into data loss.

    Scope is deliberate. Reconciliation clears the SCIP graph tables this function
    owns; it does not touch ``documents``, whose rows cost LLM spend and belong to
    the generation pipeline.
    """

    def _factory(self, *, broken=()):
        def factory(scip_path, *, repo, max_staleness_days):
            if repo in broken:
                msg = f'simulated unreadable index for {repo}'
                raise RuntimeError(msg)
            return _synthetic_python_index_with_one_class(repo=repo)
        return factory

    def _sources_in(self, db_path):
        conn = sqlite3.connect(db_path)
        try:
            return {r[0] for r in conn.execute(
                'SELECT DISTINCT source_name FROM scip_symbols')}
        finally:
            conn.close()

    def test_a_source_disk_no_longer_backs_is_cleared(self, tmp_path: Path) -> None:
        """The corpus is gone, so nothing can ever rewrite these rows -- they go.

        The source stays in the caller's list, which is what actually happens: a
        deleted corpus is usually still named in ``ariadne.yaml``, so the run is
        handed the pair and finds no manifest behind it.
        """
        from docgen.scip_persist import persist_all_sources

        db_path = tmp_path / 'a.db'
        gone_root = tmp_path / 'gone'
        live_root = tmp_path / 'live'
        for root in (gone_root, live_root):
            root.mkdir()
            _write_manifest(root, [{'kind': 'python',
                                    'scip_path': 'intermediate/i.scip'}])

        pairs = [('gone', gone_root), ('live', live_root)]
        assert persist_all_sources(
            db_path, pairs, index_factory=self._factory()) == 2
        assert self._sources_in(db_path) == {'gone', 'live'}

        # The corpus is deleted -- manifest and all -- but still configured.
        shutil.rmtree(gone_root)

        persist_all_sources(
            db_path, pairs, index_factory=self._factory(), reconcile=True)

        assert self._sources_in(db_path) == {'live'}, (
            'rows for a source disk can no longer back must be cleared, or no '
            'run can ever remove them'
        )

    def test_a_source_whose_index_failed_to_load_keeps_its_rows(
        self, tmp_path: Path,
    ) -> None:
        """Present but unreadable is not the same as gone.

        A stale or corrupt artifact is a transient, reportable condition. Clearing
        on it would delete a good graph because one read raised.
        """
        from docgen.scip_persist import persist_all_sources

        db_path = tmp_path / 'b.db'
        flaky_root = tmp_path / 'flaky'
        live_root = tmp_path / 'live'
        for root in (flaky_root, live_root):
            root.mkdir()
            _write_manifest(root, [{'kind': 'python',
                                    'scip_path': 'intermediate/i.scip'}])

        pairs = [('flaky', flaky_root), ('live', live_root)]
        assert persist_all_sources(
            db_path, pairs, index_factory=self._factory()) == 2

        # Manifest still on disk; the read explodes.
        persist_all_sources(
            db_path, pairs, index_factory=self._factory(broken=('flaky',)),
            reconcile=True)

        assert self._sources_in(db_path) == {'flaky', 'live'}, (
            'a load failure must not be treated as absence -- the rows stay and '
            'the skip is reported'
        )
