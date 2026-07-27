"""The environment bridge wired into the MCP mechanistic tools.

``impact_radius`` / ``trace_flow`` attach an "environment considerations"
section — the spool docs most relevant to the target — when a spool is
enabled, and are unchanged otherwise (no-harm). Wiring-level: the tool resolves
the anchor docs, calls the tested ``environment_considerations`` service
method, and renders (or omits) the section.

See ``designs/spool-anchored-retrieval.md``.
"""
from __future__ import annotations

from types import SimpleNamespace

from ariadne_mcp import server_knowledge


class _FakeSvc:
    """Minimal stand-in for AriadneService: a fixed impact result, one anchor
    doc, and env notes the test controls."""

    def __init__(self, env_notes):
        self._env = env_notes

    def impact_radius(self, _file_path):
        return {
            'scip_indexed': True,
            'direct_dependents': 1,
            'transitive_dependents': 0,
            'affected_docs': 0,
            'affected_tests': 0,
            'radius_score': 2,
            'dependents_by_source': {},
            'top_dependents': [],
        }

    def find_documents_by_source_files(self, _files):
        return [SimpleNamespace(id='anchor1')]

    async def environment_considerations(self, _ids, **_kw):
        return self._env


class TestImpactRadiusEnvBridge:
    async def test_env_section_rendered_when_relevant(self, monkeypatch):
        notes = [{
            'doc_id': 'g1', 'title': 'Delta concurrent writes',
            'source': 'spool:databricks', 'snippet': 'optimistic concurrency',
        }]
        monkeypatch.setattr(
            'ariadne_mcp.service.AriadneService.get', lambda: _FakeSvc(notes))
        resp = await server_knowledge.ariadne_impact_radius('agentune/x.py')
        assert 'Environment considerations' in resp.output
        assert 'Delta concurrent writes' in resp.output
        assert 'spool:databricks' in resp.output

    async def test_no_section_and_base_output_preserved_when_no_env(
        self, monkeypatch,
    ):
        monkeypatch.setattr(
            'ariadne_mcp.service.AriadneService.get', lambda: _FakeSvc([]))
        resp = await server_knowledge.ariadne_impact_radius('agentune/x.py')
        assert 'Environment considerations' not in resp.output
        assert 'Impact Radius' in resp.output          # base output intact
        assert 'Direct dependents: 1' in resp.output


class TestTraceFlowEnvBridge:
    async def test_env_attached_to_trace_response(self, monkeypatch, tmp_path):
        import sqlite3

        from ariadne_mcp import server_admin

        # Stub the trace itself (SCIP walk) — we test only the enrichment.
        monkeypatch.setattr('docgen.trace_flow.trace_flow', lambda **_k: object())
        monkeypatch.setattr('cli.trace.trace_result_to_dict', lambda _r: {'hops': []})

        # A DB whose scip_symbols resolves the start symbol to a file.
        dbp = tmp_path / 'trace.db'
        con = sqlite3.connect(dbp)
        con.execute('CREATE TABLE scip_symbols (canonical_id TEXT, file TEXT)')
        con.execute(
            'INSERT INTO scip_symbols VALUES (?, ?)', ('sym1', 'agentune/foo.py'))
        con.commit()
        con.close()
        monkeypatch.setattr(
            'config.get_config', lambda: SimpleNamespace(db_path=str(dbp)))

        class _FakeSvc:
            def find_documents_by_source_files(self, _files):
                return [SimpleNamespace(id='a1')]

            async def environment_considerations(self, _ids, **_kw):
                return [{
                    'doc_id': 'g1', 'title': 'Delta concurrent writes',
                    'source': 'spool:databricks', 'snippet': '.',
                }]

        monkeypatch.setattr(
            'ariadne_mcp.service.AriadneService.get', lambda: _FakeSvc())

        resp = await server_admin.ariadne_trace_flow('sym1', depth=2)
        assert 'environment_considerations' in resp
        assert resp['environment_considerations'][0]['title'] == (
            'Delta concurrent writes')
        assert resp['environment_considerations'][0]['source'] == 'spool:databricks'
