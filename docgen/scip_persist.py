"""Persistence steps for the cross-source SCIP graph.

Extracted from :mod:`docgen.scip_cross_source`, which retains the graph builder
(``CrossSourceGraph``), the resolution types, and the manifest loader. This
module holds the ``persist_*`` pipeline: each step consumes merged ``.scip``
artifacts and writes one slice of the cross-source picture into the
``library_scip`` tables, in dependency order. See ARCHITECTURE.md §10,
"The persist pipeline".
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from docgen.scip_cross_source import (
    CrossSourceGraph,
    load_source_from_manifest,
)
from progress_util import iter_with_progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path


def persist_all_sources(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
    *,
    index_factory=None,
    progress_callback: 'Callable[[str, int, int], None] | None' = None,
) -> int:
    """Materialize the cross-source SCIP graph for every source with a
    current manifest and write it to ``library_scip`` tables.

    The contract that closes the data path from ``.scip`` files to
    ``ariadne callers`` / ``impact_radius`` / ``improve --dead-code`` /
    the architecture-prompt Dependents section: every consumer reads
    these tables via ``CrossSourceGraph.load_from(conn)``, but nothing
    else writes them.

    Cross-source edges only resolve when both endpoints are registered
    in the same materialized graph, so this walks every source up front
    rather than persisting one at a time. Sources whose manifest is
    missing or whose ``.scip`` is stale/unreadable are silently skipped
    — this runs optimistically after ``ariadne index`` where some
    sources may not have been processed yet.

    Returns the number of sources whose data was registered before the
    save. ``index_factory`` is injectable for tests; defaults to the
    real disk loader.
    """
    import hashlib
    from datetime import UTC, datetime
    from pathlib import Path as _P

    from library import Library

    graph = CrossSourceGraph()
    loaded_pairs: list[tuple[str, _P]] = []

    for source_name, source_root in iter_with_progress(sources, progress_callback, ''):
        manifest = _P(source_root) / '.ariadne' / 'manifest.json'
        if not manifest.exists():
            continue
        try:
            load_source_from_manifest(
                graph,
                source_name,
                _P(source_root),
                index_factory=index_factory,
            )
        except FileNotFoundError:
            continue
        except Exception:
            # ScipUnavailableError / ScipTooStaleError / parser errors
            # for one source must not forfeit persistence for the rest.
            continue
        loaded_pairs.append((source_name, _P(source_root)))

    if not loaded_pairs:
        return 0
    from spools import is_spool_source
    _spool_srcs = frozenset(
        name for name, _ in loaded_pairs if is_spool_source(name))
    graph.materialize(resolve_external_to=_spool_srcs or None)

    library = Library(db_path)
    try:
        with library._conn_provider.acquire() as conn:
            graph.save_to(conn)
            # Record per-source bookkeeping in scip_index_state so
            # staleness checks have one DB-queryable surface. PRIMARY
            # KEY on source_name → INSERT OR REPLACE upserts on every
            # re-persist, keeping rows current with the graph.
            now = datetime.now(UTC).isoformat()
            for source_name, source_root in loaded_pairs:
                merged = source_root / '.ariadne' / 'index.scip'
                if not merged.exists():
                    # No merged artifact (e.g., index ran with --dry-run
                    # or the merge step failed earlier). Skip the row
                    # so staleness checks don't lie about a missing file.
                    continue
                try:
                    file_sha = hashlib.sha256(merged.read_bytes()).hexdigest()
                except OSError:
                    continue
                conn.execute(
                    'INSERT OR REPLACE INTO scip_index_state '
                    '(source_name, scip_path, file_sha256, '
                    'indexed_at, indexer_version) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (
                        source_name,
                        str(merged),
                        file_sha,
                        now,
                        None,  # per-indexer versions live in manifest.json
                    ),
                )
            
            for source_name, source_root in loaded_pairs:
                persist_rst_autodoc_index(conn, source_root, source_name, graph)
            conn.commit()
    finally:
        library.close()

    return len(loaded_pairs)


def persist_api_endpoints(
    db_path: 'Path',
    sources_with_swagger: 'Iterable[tuple[str, Path, list[str]]]',
) -> int:
    """For each ``(source_name, source_root, swagger_paths)``, ingest
    OpenAPI specs into ``library_scip.api_endpoints``.

    First wired step of Wave 4 Tier 2 (HTTP API surface). Closes the
    data path from ``ariadne.yaml``'s ``swagger_paths`` declaration to
    the ``api_endpoints`` table that ``ariadne_trace_flow`` joins
    against. Without this, the table stays empty and any HTTP-tier
    hop in a flow trace silently dead-ends.

    ``swagger_paths`` are interpreted relative to ``source_root`` —
    matches the README's ``swagger_paths: [api/openapi.yaml]`` shape.

    Returns total endpoints persisted across all sources. Sources
    without ``swagger_paths`` are skipped silently. ``ingest_swagger_for_source``
    is idempotent — clears the source's prior rows before re-inserting.
    """
    from pathlib import Path as _P

    from docgen.swagger_ingest import ingest_swagger_for_source
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root, swagger_paths in sources_with_swagger:
            if not swagger_paths:
                continue
            absolute_paths = [
                str((_P(source_root) / p).resolve())
                for p in swagger_paths
            ]
            with library._conn_provider.acquire() as conn:
                count = ingest_swagger_for_source(
                    source_name=source_name,
                    swagger_paths=absolute_paths,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_string_literals(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
    progress_callback: 'Callable[[str, int, int], None] | None' = None,
) -> int:
    """For each ``(source_name, source_root)``, ingest every string
    literal from ``<source_root>/.ariadne/index.scip`` into the
    ``string_literals`` table.

    Required by the route extractors below (Akka HTTP / Flask-FastAPI
    / Express) — they look up literal values by SCIP position via
    ``lookup_literal_at_position``. Without this populated, those
    extractors silently skip every endpoint that uses a literal path.

    Must run after ``persist_all_sources`` because owner-symbol
    resolution queries ``scip_symbols``. Sources without a merged
    ``.scip`` artifact are silently skipped (the underlying
    ``ingest_string_literals`` returns 0 in that case).

    Returns the total number of literal rows inserted across all
    sources.
    """
    from docgen.scip_string_literal_extractor import ingest_string_literals
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_string_literals(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                progress_callback = progress_callback)
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_config_values(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, ingest HOCON / YAML / dotenv
    config values from the source tree into the ``config_values`` table (Phase 2q).

    This is the key->value index that Layer C's ``resolve_arg_value`` resolves
    config-getter arguments against. It was previously never called, so
    ``config_values`` stayed empty; wiring it activates config resolution for the
    sink extractors. Sources with no config files contribute 0.

    Returns the total number of config-value rows inserted across all sources.
    """
    from docgen.scip_config_value_extractor import ingest_config_values
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_config_values(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total
def persist_config_reads(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, enumerate config-getter
    read sites (``extract_config_reads``) and persist them to the
    ``config_reads`` table.

    Depends on ``string_literals`` (the candidate set) and
    ``config_values`` (value resolution) already being populated, so this
    must run after ``persist_string_literals`` and ``persist_config_values``
    in the index loop. The extractor reads source files by the absolute
    paths recorded in ``string_literals``; ``source_root`` is unused but
    kept for a uniform per-source signature. Sources with no reads
    contribute 0 (and have their prior rows cleared).

    Returns the total number of config-read rows persisted across all sources.
    """
    from docgen.scip_config_index import persist_config_reads as persist_rows
    from docgen.scip_config_usage_extractor import extract_config_reads
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, _source_root in sources:
            with library._conn_provider.acquire() as conn:
                reads = extract_config_reads(
                    source_name=source_name, conn=conn,
                )
                total += persist_rows(
                    source_name=source_name, config_reads=reads, conn=conn,
                )
                conn.commit()
    finally:
        library.close()
    return total


def persist_url_resolver(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each source, resolve ``http_client_calls`` rows against
    ``api_endpoints`` and write matched edges into ``api_calls``.

    The closing step of Wave 4. Tier 4 client extractors fill
    ``http_client_calls`` with raw URL strings; Tier 2 server-side
    extractors + Swagger fill ``api_endpoints`` with path templates;
    this resolver joins them by URL pattern matching and writes one
    ``api_calls`` row per (consumer_symbol → endpoint) edge.

    Per-source semantics: deletes only ``resolution_source='http-client'``
    rows whose call_site is in the source's current
    ``http_client_calls``, so re-running for one source doesn't
    clobber another source's resolutions.

    Returns the total number of resolved edges across all sources.
    """
    from docgen.scip_url_resolver import resolve_urls_to_endpoints
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        with library._conn_provider.acquire() as conn:
            for source_name, _source_root in sources:
                count = resolve_urls_to_endpoints(
                    conn=conn, source_name=source_name,
                )
                total += count
            conn.commit()
    finally:
        library.close()
    return total


def persist_scala_http_clients(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, extract Scala HTTP
    client call sites from the source's merged ``.scip`` and persist
    raw URL strings to ``http_client_calls``.

    Patterns caught: Akka HTTP ``Http().singleRequest(HttpRequest(uri
    = ...))``, sttp ``basicRequest.get(uri"...")``, plus akka-http
    client builders' fluent chains. Same per-source isolation +
    string_literals prereq as the Python and JS client wrappers.
    """
    from docgen.scip_scala_http_client_extractor import ingest_scala_http_clients
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_scala_http_clients(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_js_http_clients(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, extract JavaScript /
    TypeScript HTTP client call sites from the source's merged
    ``.scip`` and persist raw URL strings to ``http_client_calls``.

    Patterns caught: ``fetch(url)``, ``axios.{verb}``,
    ``this.$http.{verb}`` (Vue 2 / Angular). Same per-source isolation
    + string_literals prereq as the Python and Scala client wrappers.
    """
    from docgen.scip_js_http_client_extractor import ingest_js_http_clients
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_js_http_clients(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_go_http_clients(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, extract Go ``net/http``
    client call sites from the source's merged ``.scip`` and persist raw
    URL strings to ``http_client_calls``.

    Patterns caught: ``http.Get``/``Post``/``Head``/``PostForm``,
    ``(*http.Client).Get``/``Post``/``Head``, and
    ``http.NewRequest``/``NewRequestWithContext`` (URL arg). Same
    per-source isolation as the JS/Python/Scala client wrappers; the
    extractor reads only ``.go`` documents.
    """
    from docgen.scip_go_http_client_extractor import ingest_go_http_clients
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_go_http_clients(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_python_http_clients(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, extract Python HTTP
    client call sites from the source's merged ``.scip`` and persist
    raw URL strings to ``http_client_calls``.

    Patterns caught: ``httpx.{verb}``, ``requests.{verb}``,
    ``urllib.urlopen``, plus a configurable list of project-local
    wrapper classes. URL resolution (matching against
    ``api_endpoints.path_template``) is a separate step
    (``persist_url_resolver``, blocked on this + JS + Scala client
    extractors all being wired first).

    Same prereq as the route extractors: ``string_literals`` must
    have been populated for each source first. Re-ingest semantics:
    clears prior rows for ``source_name``.
    """
    from docgen.scip_python_http_client_extractor import ingest_python_http_clients
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_python_http_clients(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_express_routes(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, extract Express / Koa
    routes from the source's merged ``.scip`` and persist to
    ``api_endpoints`` with ``resolution_source='pattern'``.

    Same prereq + coexistence semantics as the Akka and Python-web
    wrappers — needs ``string_literals`` populated first; preserves
    Swagger-resolved rows.

    Patterns caught: ``app.<verb>(path, handler)``,
    ``router.<verb>(...)``, ``app.use(prefix, subrouter)`` for path
    composition, plus Koa's ``router.<verb>``.
    """
    from docgen.scip_express_route_extractor import ingest_express_routes
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_express_routes(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_go_routes(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, extract Go HTTP routes
    from the source's merged ``.scip`` and persist to ``api_endpoints``
    with ``resolution_source='pattern'`` (preserves Swagger rows).

    Patterns caught: gin/echo verb methods (``r.GET``/``e.POST``/…), chi
    Title-case verbs (``r.Get``/…), and net/http ``HandleFunc``/``Handle``
    (method ``ANY``). Same coexistence semantics as the Akka/Express/Python
    route wrappers; the extractor reads only ``.go`` documents.
    """
    from docgen.scip_go_route_extractor import ingest_go_routes
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_go_routes(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_python_routes(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, extract Flask / FastAPI
    routes from the source's merged ``.scip`` and persist to
    ``api_endpoints`` with ``resolution_source='pattern'``.

    Same prereq as the Akka extractor — ``string_literals`` must be
    populated for each source first (route decorators carry the URL
    template as a string literal that the extractor looks up by SCIP
    position). ``persist_string_literals`` runs ahead of this in
    ``cli_core.cmd_index``.

    Preserves Swagger-resolved rows: ``ingest_python_routes`` deletes
    only ``resolution_source='pattern'`` rows for the source before
    re-inserting.
    """
    from docgen.scip_python_web_extractor import ingest_python_routes
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_python_routes(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_akka_http_endpoints(
    db_path: 'Path',
    sources: 'Iterable[tuple[str, Path]]',
) -> int:
    """For each ``(source_name, source_root)``, extract Akka HTTP routes
    from the source's merged ``.scip`` and persist to
    ``api_endpoints`` with ``resolution_source='pattern'``.

    Requires ``string_literals`` populated for the source first;
    ``persist_string_literals`` must run before this call.

    Preserves Swagger-resolved rows: ``ingest_akka_http_routes``
    deletes only ``resolution_source='pattern'`` rows for the source
    before re-inserting, so a Swagger-declared endpoint and a
    pattern-detected endpoint can coexist for the same source.
    """
    from docgen.scip_akka_http_extractor import ingest_akka_http_routes
    from library import Library

    library = Library(db_path)
    total = 0
    try:
        for source_name, source_root in sources:
            with library._conn_provider.acquire() as conn:
                count = ingest_akka_http_routes(
                    source_name=source_name,
                    source_root=source_root,
                    conn=conn,
                )
                conn.commit()
                total += count
    finally:
        library.close()
    return total


def persist_data_model(db_path, sources, *, index_factory=None, strategies=None,
                       schema_paths_by_source=None, dialect_by_source=None,
                       max_staleness_by_source=None, progress_callback=None):
    """Populate ``schema_symbols`` + ``data_access`` for every source — the
    §10 wiring that makes the SQL data model live on a real ``ariadne index``.

    Per source: load its SCIP indexes, run the ORM binders
    (``persist_schema_symbols`` Layer 1, ``persist_data_access_orm`` Layer 2),
    then the raw-SQL binder. Then promote against the schema witnesses — any
    configured ``CREATE TABLE`` dump (``persist_schema_ddl``), the source's own
    Django migrations (``persist_schema_from_migrations``, auto-discovered at
    ``<app>/migrations/*.py``), AND its Alembic migrations
    (``persist_schema_from_alembic``, auto-discovered at ``versions/*.py`` — the
    SQLAlchemy promotion path) — the design-faithful per-ORM witnesses. Each
    binder's surfaced gaps (undecodable forms, drift/typo) are COLLECTED and
    persisted to ``data_model_gaps`` (§3a/§5.0 "surface, don't guess"), not
    silently dropped. Idempotent. A source whose manifest/.scip is missing or
    stale is skipped optimistically (raw-SQL + schema still run).

    ``index_factory`` / ``strategies`` / ``schema_paths_by_source`` are
    injectable for tests. Returns the total schema + access rows written.
    """
    import json
    from pathlib import Path as _P

    from docgen.orm_bindings import DEFAULT_STRATEGIES, persist_schema_symbols
    from docgen.orm_bindings.access import persist_data_access_orm
    from docgen.sql_access import persist_data_access_rawsql
    from docgen.sql_schema import (
        persist_schema_ddl,
        persist_schema_from_alembic,
        persist_schema_from_migrations,
    )
    from library import Library

    if strategies is None:
        strategies = list(DEFAULT_STRATEGIES)

    total = 0
    library = Library(db_path)
    try:
        for source_name, source_root in iter_with_progress(sources, progress_callback, ''):
            graph = CrossSourceGraph()
            try:
                load_source_from_manifest(
                    graph, source_name, _P(source_root),
                    index_factory=index_factory,
                    max_staleness_days=(max_staleness_by_source or {}).get(source_name, 7),
                )
            except Exception:
                # Optimistic post-index persist (as persist_all_sources): a
                # missing/stale manifest or .scip skips ORM binding for this
                # source — raw-SQL + schema below still run off persisted data.
                graph = None
            with library._conn_provider.acquire() as conn:
                gaps = []
                if graph is not None:
                    # Surface, don't guess (§3a/§5.0): the load succeeded, so the
                    # manifest is present and valid. If it declares more indexers
                    # than were bound here, those .scip artifacts aren't built yet
                    # (discover ran, index didn't) — record a gap instead of a
                    # silent 0 rows that looks identical to "no data model".
                    loaded = len(graph._sources.get(source_name, ()))
                    declared = len(json.loads(
                        (_P(source_root) / '.ariadne' / 'manifest.json')
                        .read_text(encoding='utf-8')).get('indexers', ()))
                    if declared > loaded:
                        gaps.append(
                            f'{source_name}: only {loaded} of {declared} declared '
                            f'indexer(s) are registered in the manifest with a built '
                            f'.scip — the data model bound those; run '
                            f'`ariadne index --source {source_name}` to (re)build '
                            f'the rest'
                        )
                    for entry in graph._sources.get(source_name, ()):
                        sb = persist_schema_symbols(
                            conn, source_name, entry.index, strategies=strategies)
                        ao = persist_data_access_orm(
                            conn, source_name, entry.index, strategies=strategies)
                        total += sb.nodes_written + ao.rows_written
                        gaps.extend(sb.gaps)
                        gaps.extend(ao.gaps)
                rs = persist_data_access_rawsql(conn, source_name, dialect=(dialect_by_source or {}).get(source_name))
                total += rs.rows_written
                gaps.extend(rs.gaps)
                schema_sqls = [
                    f.read_text(encoding='utf-8')
                    for rel in (schema_paths_by_source or {}).get(source_name, ())
                    if (f := _P(source_root) / rel).exists()
                ]
                if schema_sqls:
                    gaps.extend(
                        persist_schema_ddl(
                            conn, source_name, '\n'.join(schema_sqls), dialect=(dialect_by_source or {}).get(source_name, 'postgres')).gaps)
                # Django migrations (committed output) — the design-faithful
                # per-ORM promotion witness; auto-discovered, no config needed.
                migrations = [
                    (mig.parent.parent.name, mig.read_text(encoding='utf-8'))
                    for mig in sorted(_P(source_root).glob('**/migrations/*.py'))
                    if mig.name != '__init__.py'
                ]
                if migrations:
                    gaps.extend(persist_schema_from_migrations(
                        conn, source_name, migrations).gaps)
                # Alembic migrations (op.create_table/add_column) — a distinct
                # migration witness for Python ORMs (SQLAlchemy); auto-discovered
                # at versions/*.py, no config needed.
                alembic = [
                    mig.read_text(encoding='utf-8')
                    for mig in sorted(_P(source_root).glob('**/versions/*.py'))
                    if mig.name != '__init__.py'
                ]
                if alembic:
                    gaps.extend(persist_schema_from_alembic(
                        conn, source_name, alembic).gaps)
                # Referential integrity (design §9a): every data_access row must reference a
                # declared schema_symbol. The binders resolve/create their refs, so an orphan
                # signals a regression, not normal operation -- surfaced as a gap here, never a
                # silent read-time skip.
                orphans = [
                    sid for (sid,) in conn.execute(
                        'SELECT DISTINCT da.schema_symbol_id FROM data_access da '
                        'WHERE da.source_name = ? AND NOT EXISTS (SELECT 1 FROM schema_symbols '
                        'ss WHERE ss.canonical_id = da.schema_symbol_id)', (source_name,))
                ]
                gaps.extend(
                    f'{source_name}: data_access references undeclared schema symbol {sid!r} '
                    f'-- referential-integrity orphan (no witness declared it)'
                    for sid in sorted(orphans))
                # Surface, don't discard (§3a/§5.0): persist this source's gaps
                # so the diagnostics are reachable, not dropped on the floor.
                conn.execute(
                    'DELETE FROM data_model_gaps WHERE source_name = ?',
                    (source_name,))
                conn.executemany(
                    'INSERT INTO data_model_gaps (source_name, detail) VALUES (?, ?)',
                    [(source_name, g) for g in gaps])
                conn.commit()
    finally:
        library.close()
    return total


__all__ = [
    'persist_akka_http_endpoints', 'persist_all_sources', 'persist_api_endpoints', 'persist_config_reads', 'persist_config_values', 'persist_express_routes', 'persist_go_http_clients', 'persist_go_routes', 'persist_js_http_clients', 'persist_python_http_clients', 'persist_python_routes', 'persist_scala_http_clients', 'persist_string_literals', 'persist_url_resolver', 'persist_data_model'
]


def persist_rst_autodoc_index(conn, source_root, source_name, graph) -> None:
    """Resolve a source's rst autodoc directives against the materialized
    SCIP graph and persist the (symbol -> rst section) reverse index to
    ``rst_autodoc_links``.

    Run at persist time, after ``save_to``, so the reverse index is in the
    DB before doc-gen's ``load_from`` reads it — code docs are reverse-
    enriched in the same run. rst is re-extracted from disk (cheap at rst
    volumes) rather than threading catalog state through the pipeline.
    """
    from pathlib import Path as _P

    from docgen.catalog_enrich import _resolve_autodoc_links
    from docgen.catalog_extractor import extract_elements
    from docgen.staleness import find_catalog_files
    from library.scip import persist_rst_autodoc_links

    root = _P(source_root)
    links = []
    for f in find_catalog_files(root):
        if f.suffix.lower() == '.rst':
            links.extend(_resolve_autodoc_links(extract_elements(f, root), graph))
    persist_rst_autodoc_links(conn, source_name, links)


def dangling_autodoc(source_root, graph, ignore_staleness=False):
    """rst autodoc targets that no longer resolve to a code symbol -- the rst
    references code that's gone or renamed (a stale-doc signal). Returns
    ``(rst_section_qualified_name, target)`` pairs. Files matching
    ``ignore_staleness`` are skipped (don't nag sources opted out of staleness).
    """
    from pathlib import Path as _P

    from config import ignore_staleness_matches
    from docgen.catalog_enrich import _resolve_autodoc_links
    from docgen.catalog_extractor import extract_elements
    from docgen.staleness import find_catalog_files

    root = _P(source_root)
    dangling = []
    for f in find_catalog_files(root):
        if f.suffix.lower() == '.rst':
            rel = str(f.relative_to(root))
            if not ignore_staleness_matches(ignore_staleness, rel):
                for link in _resolve_autodoc_links(extract_elements(f, root), graph):
                    if not link.resolved:
                        dangling.append((link.section_qualified_name, link.target))
    return dangling
