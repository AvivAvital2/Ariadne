"""ariadne_config_usage MCP tool — Tier 1 Feature 3.

A thin wrapper over docgen.catalog_lookup.config_usage: resolve the source
(fail-closed when absent), open the library, delegate, close. Synthetic,
repo/language-agnostic fixtures. See designs/config-code-bridge/tier1-string-join.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _build_db(tmp_path, *, with_data: bool):
    from docgen.catalog_writer import _element_doc_id
    from docgen.scip_string_literal_index import StringLiteral, persist_string_literals
    from library import Library
    from library.scip import init_scip_schema

    db = tmp_path / 'cfg.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        init_scip_schema(conn)
        conn.commit()
    if with_data:
        lib.add_document(
            content_type='catalog', title='pkg.conf.app.svc.timeout',
            content='hocon_key pkg.conf.app.svc.timeout :: timeout = 30',
            source_files=['conf/app.conf'], embedding=np.zeros(8, dtype=np.float32),
            metadata={'kind': 'element', 'source_name': 'src1', 'subtype': 'hocon_key',
                      'qualified_name': 'pkg.conf.app.svc.timeout', 'signature': 'timeout = 30',
                      'location': {'line_start': 12, 'line_end': 12, 'col_start': 0, 'col_end': 0}},
            doc_id=_element_doc_id('src1', 'pkg.conf.app.svc.timeout'),
        )
        with lib._conn_provider.acquire() as conn:
            persist_string_literals(
                source_name='src1',
                literals=[StringLiteral(file=Path('reader_a'), line_start=53, col_start=0,
                                        value='svc.timeout', owning_symbol_id=None)],
                conn=conn,
            )
            conn.commit()
    lib.close()
    return db


def _patch_config(monkeypatch, db_path, default_source):
    import config

    cfg = type('Cfg', (), {})()
    cfg.db_path = str(db_path)
    cfg.default_source = default_source
    monkeypatch.setattr(config, 'get_config', lambda: cfg)


@pytest.mark.asyncio
async def test_tool_returns_usage(monkeypatch, tmp_path):
    from ariadne_mcp.server import ariadne_config_usage

    _patch_config(monkeypatch, _build_db(tmp_path, with_data=True), 'src1')
    out = await ariadne_config_usage(key='svc.timeout', source='src1')
    assert out['found'] is True
    assert out['read_count'] == 1
    assert out['confidence'] == 'string-match'


@pytest.mark.asyncio
async def test_tool_fail_closed_without_source(monkeypatch, tmp_path):
    from ariadne_mcp.server import ariadne_config_usage

    _patch_config(monkeypatch, _build_db(tmp_path, with_data=False), None)
    out = await ariadne_config_usage(key='svc.timeout', source=None)
    assert out.get('error') == 'no_source'
