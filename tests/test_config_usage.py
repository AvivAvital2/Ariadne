"""config_usage — the Tier 1 config↔code bridge (Feature 2).

Maps a config key (as code uses it) to (a) its literal default from the catalog
and (b) the code sites that read it (string_literals). Confidence is always
'string-match' in Tier 1.

Fixtures are synthetic and repo/language-agnostic: the bridge must behave the
same for any key/source, independent of which projects or config keys exist in a
real database. See designs/config-code-bridge/tier1-string-join.md (Feature 2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def library(tmp_path):
    from library import Library
    from library.scip import init_scip_schema

    lib = Library(tmp_path / 'cfg.db')
    with lib._conn_provider.acquire() as conn:
        init_scip_schema(conn)
        conn.commit()
    yield lib
    lib.close()


def _add_config_element(library, source, qn, signature, file, line):
    from docgen.catalog_writer import _element_doc_id

    library.add_document(
        content_type='catalog',
        title=qn,
        content=f'hocon_key {qn} :: {signature}',
        source_files=[file],
        embedding=np.zeros(8, dtype=np.float32),
        metadata={
            'kind': 'element', 'source_name': source, 'subtype': 'hocon_key',
            'qualified_name': qn, 'signature': signature,
            'location': {'line_start': line, 'line_end': line, 'col_start': 0, 'col_end': 0},
        },
        doc_id=_element_doc_id(source, qn),
    )


def _add_literals(library, source, rows):
    """rows: list of (file, line, value)."""
    from docgen.scip_string_literal_index import StringLiteral, persist_string_literals

    with library._conn_provider.acquire() as conn:
        persist_string_literals(
            source_name=source,
            literals=[StringLiteral(file=Path(f), line_start=ln, col_start=0, value=v, owning_symbol_id=None)
                      for (f, ln, v) in rows],
            conn=conn,
        )
        conn.commit()


def test_config_usage_definition_and_read_site(library) -> None:
    from docgen.catalog_lookup import config_usage

    _add_config_element(library, 'src1', 'pkg.conf.app.svc.timeout', 'timeout = 30', 'conf/app.conf', 12)
    _add_config_element(library, 'src1', 'pkg.conf.app.svc.retries', 'retries = 3', 'conf/app.conf', 99)  # non-matching
    _add_literals(library, 'src1', [('reader_a', 53, 'svc.timeout')])

    out = config_usage(library, 'src1', 'svc.timeout')
    assert out['found'] is True
    assert out['confidence'] == 'string-match'
    assert len(out['definitions']) == 1
    assert out['definitions'][0]['default_value'] == 'timeout = 30'
    assert out['read_count'] == 1
    assert out['read_sites'][0]['line'] == 53


def test_no_read_sites_populates_notes(library) -> None:
    from docgen.catalog_lookup import config_usage

    _add_config_element(library, 'src1', 'pkg.conf.app.feature.enabled', 'enabled = true', 'conf/app.conf', 5)
    out = config_usage(library, 'src1', 'feature.enabled')
    assert out['found'] is True
    assert out['read_count'] == 0
    assert out['notes'] and 'split/relative' in out['notes'][0]


def test_key_not_in_catalog_found_false(library) -> None:
    from docgen.catalog_lookup import config_usage

    out = config_usage(library, 'src1', 'absent.key')
    assert out['found'] is False
    assert out['definitions'] == []


def test_excludes_other_sources_and_non_config_subtypes(library) -> None:
    from docgen.catalog_lookup import config_usage

    _add_config_element(library, 'src2', 'pkg.conf.app.svc.timeout', 'v', 'conf/other.conf', 1)  # wrong source
    library.add_document(  # right key-suffix but a code symbol, not a config key
        content_type='catalog', title='pkg.code.svc.timeout', content='c',
        source_files=['code_file'], embedding=np.zeros(8, dtype=np.float32),
        metadata={'kind': 'element', 'source_name': 'src1', 'subtype': 'function',
                  'qualified_name': 'pkg.code.svc.timeout', 'signature': 'def f()',
                  'location': {'line_start': 1}},
        doc_id='fn-1',
    )
    out = config_usage(library, 'src1', 'svc.timeout')
    assert out['found'] is False


def test_multiple_definitions_across_files(library) -> None:
    from docgen.catalog_lookup import config_usage

    _add_config_element(library, 'src1', 'pkg.web.conf.shared.cache.ttl', 'ttl = 60', 'web/app.conf', 4)
    _add_config_element(library, 'src1', 'pkg.api.conf.shared.cache.ttl', 'ttl = 30', 'api/app.conf', 3)
    out = config_usage(library, 'src1', 'cache.ttl')
    assert out['found'] is True
    assert len(out['definitions']) == 2


def test_get_element_body_reads_file_range(library, tmp_path) -> None:
    # get_element_body is what ariadne_body delegates to; cover its happy path
    # (previously untested) so the catalog_lookup file meets the coverage gate.
    from docgen.catalog_lookup import get_element_body

    f = tmp_path / 'sample.conf'
    f.write_text('a = 1\nb = 2\nc = 3\n', encoding='utf-8')
    _add_config_element(library, 'src1', 'pkg.sample.b', 'b = 2', str(f), 2)

    out = get_element_body(library, 'src1', None, 'pkg.sample.b')
    assert out['found'] is True
    assert out['body'] == 'b = 2'
    assert out['body_line_count'] == 1


def _add_config_reads(library, source, rows):
    """rows: list of (file, line, col, key, value, confidence)."""
    from docgen.scip_config_index import persist_config_reads
    from docgen.scip_config_usage_extractor import ConfigRead

    with library._conn_provider.acquire() as conn:
        persist_config_reads(
            source_name=source,
            config_reads=[
                ConfigRead(
                    file=Path(f), line=ln, col=c, key=k,
                    value=v, confidence=conf,
                )
                for (f, ln, c, k, v, conf) in rows
            ],
            conn=conn,
        )
        conn.commit()


def test_prefers_config_reads_over_string_match(library) -> None:
    from docgen.catalog_lookup import config_usage

    _add_config_element(
        library, 'src1', 'pkg.conf.app.svc.timeout', 'timeout = 30',
        'conf/app.conf', 12,
    )
    # The same site appears both as a bare string-literal match (Tier 1)
    # and as a resolved config_reads row (Tier 2). Tier 2 wins, no dup.
    _add_literals(library, 'src1', [('reader.scala', 53, 'svc.timeout')])
    _add_config_reads(
        library, 'src1',
        [('reader.scala', 53, 8, 'svc.timeout', '30', 'config-resolved')],
    )

    out = config_usage(library, 'src1', 'svc.timeout')
    assert out['found'] is True
    assert out['confidence'] == 'config-resolved'
    assert out['read_count'] == 1  # not duplicated across the two sources
    assert out['read_sites'][0]['confidence'] == 'config-resolved'
    assert out['read_sites'][0]['value'] == '30'


def test_dedup_read_sites_by_file_line(library) -> None:
    from docgen.catalog_lookup import config_usage

    _add_config_element(library, 'src1', 'pkg.conf.svc.k', 'k = 1', 'conf/app.conf', 1)
    # Two config_reads rows at the same (file, line) — e.g. a split-path
    # chain whose two segments are separate literals on one line.
    _add_config_reads(library, 'src1', [
        ('reader.scala', 7, 4, 'svc.k', '1', 'config-resolved'),
        ('reader.scala', 7, 30, 'svc.k', '1', 'config-resolved'),
    ])
    out = config_usage(library, 'src1', 'svc.k')
    assert out['read_count'] == 1  # collapsed by (file, line)


def test_mixed_per_site_confidence(library) -> None:
    from docgen.catalog_lookup import config_usage

    _add_config_element(library, 'src1', 'pkg.conf.svc.k', 'k = 1', 'conf/app.conf', 1)
    # One verified getter call + one unsupported-language string match.
    _add_config_reads(library, 'src1', [
        ('a.scala', 1, 0, 'svc.k', '1', 'config-resolved'),
        ('b.rb', 2, 0, 'svc.k', '1', 'string-match'),
    ])
    out = config_usage(library, 'src1', 'svc.k')
    assert out['confidence'] == 'config-resolved'  # config_reads is the source
    assert out['read_count'] == 2
    by_file = {s['file']: s['confidence'] for s in out['read_sites']}
    assert by_file['a.scala'] == 'config-resolved'
    assert by_file['b.rb'] == 'string-match'
