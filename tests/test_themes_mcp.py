"""Tests for ariadne_themes MCP tool surface (Themes plan, Phase 6).

The MCP tool delegates to the standalone `themes_action` helper so we can
unit-test the response shape without spinning up the full MCP server.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


def _add_doc(
    library: Library,
    doc_id: str,
    *,
    content_type: str = 'theme',
    title: str | None = None,
    content: str = 'content',
    source_name: str | None = None,
) -> None:
    library.add_document(
        content_type=content_type,  # type: ignore[arg-type]
        title=title or f'doc {doc_id}',
        content=content,
        source_files=[],
        embedding=_unit([1.0, 0, 0, 0, 0, 0, 0, 0]),
        metadata={},
        doc_id=doc_id,
    )
    if source_name is not None:
        with library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE documents SET source_name = ? WHERE id = ?',
                (source_name, doc_id),
            )


def _bootstrap_theme(
    library: Library,
    cluster_id: str,
    *,
    members: list[str],
    title: str = 'Theme Title',
    coherent: bool = True,
    source_name: str | None = None,
) -> None:
    """Insert a placeholder theme + members for tests."""
    doc_id = f'theme-doc-{cluster_id}'
    _add_doc(library, doc_id, content_type='theme', title=title,
             content=f'# {title}\n\nbody', source_name=source_name)
    library.add_theme(
        cluster_id=cluster_id,
        doc_id=doc_id,
        member_count=len(members),
        resolution=1.0,
        summary_hash='h',
        coherent=coherent,
        dirty=False,
    )
    for mid in members:
        _add_doc(library, mid, content_type='catalog', title=mid)
    library.set_theme_members(
        cluster_id, [(mid, 1.0) for mid in members],
    )


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'themes-mcp-test.db')
    yield lib
    lib.close()


# ---------------------------------------------------------------------------
# action='list'
# ---------------------------------------------------------------------------


class TestThemesActionList:
    def test_list_returns_coherent_themes_by_default(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        _bootstrap_theme(library, 'c1', members=['el1', 'el2'], coherent=True)
        _bootstrap_theme(library, 'c2', members=['el3', 'el4'], coherent=False)

        result = themes_action(library, action='list')

        assert 'themes' in result
        ids = {t['cluster_id'] for t in result['themes']}
        assert ids == {'c1'}

    def test_list_with_coherent_only_false_includes_incoherent(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        _bootstrap_theme(library, 'c1', members=['el1'], coherent=True)
        _bootstrap_theme(library, 'c2', members=['el2'], coherent=False)

        result = themes_action(library, action='list', coherent_only=False)

        ids = {t['cluster_id'] for t in result['themes']}
        assert ids == {'c1', 'c2'}

    def test_list_filters_by_source(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        _bootstrap_theme(library, 'c1', members=['el1'], source_name='src1')
        _bootstrap_theme(library, 'c2', members=['el2'], source_name='src2')

        result = themes_action(library, action='list', source='src1')

        ids = {t['cluster_id'] for t in result['themes']}
        assert ids == {'c1'}

    def test_list_payload_includes_member_count_and_title(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        _bootstrap_theme(library, 'c1', members=['el1', 'el2', 'el3'], title='Retry Logic')

        result = themes_action(library, action='list')
        first = result['themes'][0]

        assert first['cluster_id'] == 'c1'
        assert first['member_count'] == 3
        # title is taken from the theme doc.
        assert 'Retry' in first['title'] or first['title'] == 'Retry Logic'


# ---------------------------------------------------------------------------
# action='get'
# ---------------------------------------------------------------------------


class TestThemesActionGet:
    def test_get_returns_theme_doc_content(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        _bootstrap_theme(library, 'c1', members=['el1'], title='Config Loading')

        result = themes_action(library, action='get', cluster_id='c1')

        assert result.get('cluster_id') == 'c1'
        assert 'content' in result
        assert 'Config Loading' in result['content']

    def test_get_unknown_cluster_returns_error(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        result = themes_action(library, action='get', cluster_id='missing')
        assert 'error' in result

    def test_get_without_cluster_id_returns_error(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        result = themes_action(library, action='get')
        assert 'error' in result


# ---------------------------------------------------------------------------
# action='members'
# ---------------------------------------------------------------------------


class TestThemesActionMembers:
    def test_members_returns_id_title_weight(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        _bootstrap_theme(library, 'c1', members=['alpha', 'beta', 'gamma'])

        result = themes_action(library, action='members', cluster_id='c1')

        assert 'members' in result
        member_ids = {m['element_id'] for m in result['members']}
        assert member_ids == {'alpha', 'beta', 'gamma'}
        # Each member entry has element_id + title + weight.
        for m in result['members']:
            assert 'element_id' in m
            assert 'title' in m
            assert 'weight' in m

    def test_members_unknown_cluster_returns_empty(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        result = themes_action(library, action='members', cluster_id='missing')
        # Empty member list, not an error — themes_action may return empty for nonexistent cluster.
        assert result.get('members') == []

    def test_members_without_cluster_id_returns_error(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        result = themes_action(library, action='members')
        assert 'error' in result


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class TestSearchVisibility:
    """Plan §5.6 acceptance: themes must be visible to ariadne_search.

    The search engine reads candidates from list_documents_lite(); for themes
    to surface in search, they need to live in the documents table with
    content_type='theme'. This test asserts the structural condition.
    """

    def test_themes_appear_in_list_documents_lite(self, library: Library) -> None:
        _bootstrap_theme(
            library, 'c1', members=['el1'],
            title='Retry Logic with Exponential Backoff',
        )
        docs = library.list_documents_lite()
        theme_docs = [d for d in docs if d.content_type == 'theme']
        assert any('Retry' in d.title for d in theme_docs)


class TestMcpRegistration:
    @pytest.mark.asyncio
    async def test_ariadne_themes_in_mcp_tool_registry(self) -> None:
        import ariadne_mcp.server as mcp_server
        tools = await mcp_server.mcp.list_tools()
        names = [t.name for t in tools]
        assert 'ariadne_themes' in names


class TestListLimit:
    def test_list_respects_limit(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action

        for i in range(5):
            _bootstrap_theme(library, f'c{i}', members=[f'el{i}'], title=f'Theme {i}')

        result = themes_action(library, action='list', limit=2)
        assert len(result['themes']) == 2
        assert result['total'] == 5


class TestMembersOrdering:
    def test_members_ordered_by_weight_descending(self, library: Library) -> None:
        """Plan §5.6 — `members` returns elements ordered by weight (strongest
        cluster ties first). Catches regressions in the sort.
        """
        from ariadne_mcp.service_themes import themes_action

        _bootstrap_theme(library, 'c1', members=[])  # use raw inserts below
        # Replace members with explicit weights.
        for eid in ('low', 'high', 'mid'):
            _add_doc(library, eid, content_type='catalog')
        library.set_theme_members(
            'c1', [('low', 0.1), ('high', 0.9), ('mid', 0.5)],
        )

        result = themes_action(library, action='members', cluster_id='c1')

        order = [m['element_id'] for m in result['members']]
        weights = [m['weight'] for m in result['members']]
        assert order == ['high', 'mid', 'low']
        assert weights == sorted(weights, reverse=True)


class TestThemesActionRouting:
    def test_unknown_action_returns_error(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        result = themes_action(library, action='bogus')  # type: ignore[arg-type]
        assert 'error' in result

    def test_default_action_is_list(self, library: Library) -> None:
        from ariadne_mcp.service_themes import themes_action
        _bootstrap_theme(library, 'c1', members=['el1'])
        result = themes_action(library)
        assert 'themes' in result
