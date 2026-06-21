"""Dockerfile support — Phase 1 (recognition & file-index).

Evolving test: one growing assertion over the recognition gate — detection
(catalog + pricing), the `is_catalog_file` predicate, both file walks, and
pricing. See `designs/dockerfile-support.md`.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from docgen import catalog_writer
from docgen.catalog_extractor import _detect_language as detect_catalog_lang
from docgen.catalog_extractor import extract_elements
from docgen.pricing import _detect_language as detect_pricing_lang
from docgen.pricing import estimate_cost
from docgen.scip_config_index import persist_config_values, query_config_values_by_key
from docgen.scip_config_scanners import _scanner_for_file, scan_config_sources
from docgen.staleness import find_catalog_files

_CONFIG_VALUES_DDL = (
    'CREATE TABLE config_values ('
    '  id INTEGER PRIMARY KEY AUTOINCREMENT,'
    '  source_name TEXT NOT NULL, file TEXT NOT NULL, key TEXT NOT NULL,'
    '  value TEXT NOT NULL, line_start INTEGER NOT NULL,'
    '  UNIQUE(source_name, file, key, line_start))'
)


def test_dockerfile_is_recognized_and_priced(tmp_path):
    # 1. Detection — both detectors (catalog + pricing) agree, matching by
    #    NAME since a Dockerfile has no extension. Unrelated extensionless
    #    files are NOT Dockerfiles.
    for detect in (detect_catalog_lang, detect_pricing_lang):
        assert detect(Path('Dockerfile')) == 'dockerfile'
        assert detect(Path('Dockerfile.dev')) == 'dockerfile'
        assert detect(Path('app.dockerfile')) == 'dockerfile'
        assert detect(Path('Makefile')) is None
        assert detect(Path('notes')) is None

    # 2. The single recognition-gate predicate.
    assert catalog_writer.is_catalog_file(Path('Dockerfile')) is True
    assert catalog_writer.is_catalog_file(Path('app.dockerfile')) is True
    assert catalog_writer.is_catalog_file(Path('Makefile')) is False
    assert catalog_writer.is_catalog_file(Path('x.py')) is True  # extensions still pass

    # 3. Both walks collect a Dockerfile: the catalog walk AND the staleness
    #    discovery walk (the latter is what puts it in the staleness pipeline,
    #    closing the "documented but never tracked" gap).
    (tmp_path / 'Dockerfile').write_text('FROM python:3.12-slim\nEXPOSE 8080\n')
    (tmp_path / 'app.py').write_text('x = 1\n')
    catalog_walk = catalog_writer.iter_catalog_files(tmp_path)
    stale_walk = find_catalog_files(tmp_path)
    assert any(p.name == 'app.py' for p in catalog_walk)   # sanity: the walk works
    assert any(p.name == 'Dockerfile' for p in catalog_walk)
    assert any(p.name == 'Dockerfile' for p in stale_walk)

    # 4. It is now priced (was $0.00 — dropped before pricing).
    est = estimate_cost(
        files=((tmp_path / 'Dockerfile', 4000),),
        doc_types=('explanation',),
        model='gpt-5.4',
    )
    assert est.total_cost_usd > 0.0


def test_extract_dockerfile_resolves_runtime_stage(tmp_path):
    """Phase 2 — structured extraction. A multi-stage build yields stage +
    instruction elements with line ranges, and the *runtime* base resolves
    through the FINAL stage's lineage (``scratch``), never the builder.
    """
    df = tmp_path / 'Dockerfile'
    df.write_text(
        'FROM golang:1.22-bookworm AS build\n'   # 1
        'ENV CGO_ENABLED=0\n'                     # 2
        'RUN go build -o /worker .\n'             # 3
        '\n'                                      # 4
        'FROM scratch\n'                          # 5
        'EXPOSE 8080\n'                           # 6
        'COPY --from=build /worker /worker\n',    # 7
    )
    elements = extract_elements(df, tmp_path)
    by_sub: dict[str, list] = {}
    for e in elements:
        by_sub.setdefault(e.subtype, []).append(e)

    # Two FROM stages; the `AS build` alias is the build stage's name, and
    # line ranges point at the FROM lines.
    stages = {s.qualified_name: s for s in by_sub.get('dockerfile_stage', [])}
    assert 'build' in stages
    assert 'golang:1.22-bookworm' in stages['build'].signature
    assert stages['build'].line_start == 1
    final = [s for s in by_sub['dockerfile_stage'] if s.qualified_name != 'build']
    assert len(final) == 1
    assert 'scratch' in final[0].signature
    assert final[0].line_start == 5

    # ENV / EXPOSE become instruction elements scoped to their stage.
    assert any(
        e.qualified_name.endswith('CGO_ENABLED') and e.parent_qualified_name == 'build'
        for e in by_sub.get('dockerfile_env', [])
    )
    assert any('8080' in e.signature for e in by_sub.get('dockerfile_expose', []))

    # The runtime base resolves to `scratch` via the final stage's lineage —
    # NOT the golang builder. Attached to the final-stage element's
    # JSON-serializable `documentation`.
    assert final[0].documentation['runtime_base'] == 'scratch'


def test_extract_dockerfile_lineage_chain_and_fallback(tmp_path):
    """A final stage built `FROM <prior-stage>` chases the lineage to the
    external root and derives the OS family; a Dockerfile with no parseable
    stage degrades to [] (file-index/text fallback, no crash)."""
    df = tmp_path / 'Dockerfile'
    df.write_text(
        'FROM gcr.io/distroless/base-debian12 AS base\n'  # 1
        'FROM base\n'                                      # 2 (final, refs a stage)
        'COPY app /app\n',                                 # 3
    )
    stages = [e for e in extract_elements(df, tmp_path)
              if e.subtype == 'dockerfile_stage']
    final = stages[-1]
    # Runtime base is the external root the stage chain resolves to, not 'base'.
    assert final.documentation['runtime_base'] == 'gcr.io/distroless/base-debian12'
    assert final.documentation['os'] == 'debian (distroless)'

    # No FROM → no structured elements → file degrades to file-index ([]).
    df.write_text('RUN echo hi\nCOPY . .\n')
    assert extract_elements(df, tmp_path) == []


def test_dockerfile_env_resolves_through_config_bridge(tmp_path):
    """Phase 3 — the differentiator. A Dockerfile's ENV/ARG become config keys
    in the ``config_values`` table that Layer C / ``ariadne_config_usage``
    resolves against — so a Dockerfile env can be linked to its code read
    sites (the cross-representation join grep can't do)."""
    df = tmp_path / 'Dockerfile'
    df.write_text(
        'FROM python:3.12-slim\n'            # 1
        'ARG REGISTRY=ghcr.io\n'             # 2
        'ENV MODEL_PATH=/models DEBUG=1\n'   # 3
        'ENV LEGACY value here\n',           # 4
    )

    # The registry routes a Dockerfile (no extension) by name, and the scanner
    # emits ConfigValues for ENV + ARG with line numbers — including the
    # multi-var and legacy (space-separated) ENV forms, and an ARG default.
    scanner = _scanner_for_file(df)
    assert scanner is not None, 'a Dockerfile must route to a config scanner'
    by_key = {c.key: c for c in scanner(df)}
    assert by_key['MODEL_PATH'].value == '/models'
    assert by_key['MODEL_PATH'].line_start == 3
    assert by_key['DEBUG'].value == '1'
    assert by_key['REGISTRY'].value == 'ghcr.io'      # ARG default
    assert by_key['LEGACY'].value == 'value here'     # legacy `ENV K v` form

    # End-to-end: the tree scan picks the Dockerfile up, and the ENV lands in
    # the config_values table the bridge resolves against.
    all_cvs = scan_config_sources(tmp_path)
    assert any(c.key == 'MODEL_PATH' for c in all_cvs)

    conn = sqlite3.connect(':memory:')
    conn.executescript(_CONFIG_VALUES_DDL)
    persist_config_values(source_name='svc', config_values=all_cvs, conn=conn)
    rows = query_config_values_by_key(source_name='svc', key='MODEL_PATH', conn=conn)
    assert rows and rows[0].value == '/models'
