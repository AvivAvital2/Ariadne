"""Tests for graph-based crossref injection.

Neighbours come from ONE batched ``get_related_batch`` call (a single bulk
graph load + in-memory walks) instead of a per-doc ``get_related`` query
storm. The graph already has imports + documents/topic_member edges from
build_graph and semantic_neighbor edges from build_semantic_edges (run during
themes); reusing them in one batch is what makes crossrefs scale.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _test_config(monkeypatch, tmp_path):
    """Configure ``'mylib'`` and ``'someotherlib'`` so the chokepoint
    admits the orchestrator's docs. The contract under test is the
    crossref injection logic; source naming is environmental."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(
        monkeypatch, tmp_path, ('mylib', 'someotherlib'),
    )


@pytest.mark.asyncio
async def test_inject_crossrefs_uses_batch_get_related(tmp_path):
    """``_inject_crossrefs_scoped`` must fetch neighbours via ONE batched
    ``get_related_batch`` call for all scoped docs, not a per-doc query storm.
    """
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'x.db',
        staleness_db_path=tmp_path / 's.db',
    )

    async with DocGenOrchestrator(cfg) as orch:
        fake_docs = [
            MagicMock(
                id=f'd{i}', title=f'Doc {i}',
                source_files=[f'f{i}.py'], content=f'content {i}',
            )
            for i in range(3)
        ]
        fake_lib = MagicMock()
        fake_lib.list_documents.return_value = fake_docs
        fake_lib.get_related_batch.return_value = {}  # no neighbours → no updates
        orch._library = fake_lib

        await orch._inject_crossrefs_scoped()

        assert fake_lib.get_related_batch.call_count == 1, (
            f'expected ONE batched get_related_batch for all scoped docs; '
            f'got {fake_lib.get_related_batch.call_count} calls'
        )
        # No neighbours → no doc updates.
        assert fake_lib.update_document.call_count == 0


@pytest.mark.asyncio
async def test_inject_crossrefs_filters_neighbors_to_scope(tmp_path):
    """``get_related_batch`` may surface docs outside the configured scope;
    the crossref pass must drop them so only scoped docs are linked.
    """
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'x.db',
        staleness_db_path=tmp_path / 's.db',
        source_name='mylib',
    )

    async with DocGenOrchestrator(cfg) as orch:
        # d_in is the in-scope doc; d_other is from a different source
        d_in = MagicMock(
            id='d_in', title='In-scope',
            source_files=['mylib/f1.py'], content='content',
        )
        d_other = MagicMock(
            id='d_other', title='Out-of-scope',
            source_files=['someotherlib/f.py'], content='content',
        )
        fake_lib = MagicMock()
        fake_lib.list_documents.return_value = [d_in, d_other]
        # the in-scope doc's only neighbour is the out-of-scope doc
        fake_lib.get_related_batch.return_value = {
            'd_in': [{
                'id': 'd_other', 'title': 'Out-of-scope',
                'content_type': 'explanation', 'distance': 1.0,
            }],
        }
        orch._library = fake_lib
        # Bypass the chokepoint's SQL-backed closure filter; the contract
        # under test is the orchestrator's own scoping pass, not the
        # ScopedLibrary wrapper (which is tested elsewhere).
        orch._scoped = fake_lib

        await orch._inject_crossrefs_scoped()

        # update_document should NOT be called: the only neighbor was filtered out
        assert fake_lib.update_document.call_count == 0, (
            'out-of-scope neighbor leaked through scope filter'
        )


@pytest.mark.asyncio
async def test_inject_crossrefs_injects_related_section_when_neighbors_exist(tmp_path):
    """When scoped neighbours exist, the doc content gets a ``Related``
    section injected (via ``inject_related_section``) and the doc is updated.
    """
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    cfg = OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'x.db',
        staleness_db_path=tmp_path / 's.db',
    )

    async with DocGenOrchestrator(cfg) as orch:
        d1 = MagicMock(
            id='d1', title='Doc 1',
            source_files=['f1.py'],
            content='# Doc 1\n\nSome body.\n',
        )
        d2 = MagicMock(
            id='d2', title='Doc 2',
            source_files=['f2.py'],
            content='# Doc 2\n\nMore body.\n',
        )
        fake_lib = MagicMock()
        fake_lib.list_documents.return_value = [d1, d2]
        # d1 links to d2; d2 has no neighbours
        fake_lib.get_related_batch.return_value = {
            'd1': [{
                'id': 'd2', 'title': 'Doc 2',
                'content_type': 'explanation', 'distance': 1.0,
            }],
            'd2': [],
        }
        orch._library = fake_lib
        # Bypass the chokepoint's SQL-backed closure filter; the contract
        # under test is "neighbors → update_document", not the wrapper.
        orch._scoped = fake_lib

        await orch._inject_crossrefs_scoped()

        # d1 should have been updated; d2 (no neighbors) should not be
        assert fake_lib.update_document.called
        called_with_d1 = any(
            call.args and call.args[0] == 'd1'
            for call in fake_lib.update_document.call_args_list
        )
        assert called_with_d1, 'expected update_document(d1, ...) for the linked doc'
