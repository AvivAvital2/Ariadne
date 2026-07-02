"""Guardrail test for fix #4 — `explain` chunkable (grows across slices).

Slice ①: `kinds` filters the returned document types.
Later slices extend this same file: `sections_only`, size budget, back-compat.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ariadne_mcp.server_knowledge import ariadne_explain
from ariadne_mcp.service import AriadneService
from library import Library

_FILE = 'pkg/sub/mod.py'


@pytest.fixture
def library(tmp_path: Path) -> Library:
    lib = Library(tmp_path / 'explain_chunkable.db')
    yield lib
    lib.close()


def _seed_three_types(library: Library) -> None:
    """Document one file with three distinct content types."""
    library.add_document(content_type='architecture', title='Arch', content='arch body', source_files=[_FILE])
    library.add_document(content_type='explanation', title='Expl', content='expl body', source_files=[_FILE])
    library.add_document(content_type='gotcha', title='Gotcha', content='gotcha body', source_files=[_FILE])


class TestExplainKinds:
    """Slice ① — `kinds` restricts `explain` to the requested document types."""

    def test_no_filter_returns_all_types(self, library: Library) -> None:
        """Baseline / regression guard: no filter → every type for the file."""
        _seed_three_types(library)
        result = library.explain(_FILE)
        assert set(result['documents']) == {'architecture', 'explanation', 'gotcha'}
        assert result['total_documents'] == 3

    def test_kinds_filters_returned_types(self, library: Library) -> None:
        """`kinds` returns exactly the requested types and excludes the rest,
        with total_documents / types_found reflecting the filter.
        """
        _seed_three_types(library)

        only_arch = library.explain(_FILE, kinds=['architecture'])
        assert set(only_arch['documents']) == {'architecture'}
        assert 'explanation' not in only_arch['documents']
        assert 'gotcha' not in only_arch['documents']
        assert only_arch['types_found'] == ['architecture']
        assert only_arch['total_documents'] == 1

        arch_and_gotcha = library.explain(_FILE, kinds=['architecture', 'gotcha'])
        assert set(arch_and_gotcha['documents']) == {'architecture', 'gotcha'}
        assert arch_and_gotcha['total_documents'] == 2

    def test_kinds_threads_through_tool(self, library: Library) -> None:
        """The full MCP path (tool → service facade → library) honors `kinds`."""
        _seed_three_types(library)
        svc = AriadneService()
        svc._library = library  # inject synthetic library; .library property returns it as-is
        AriadneService._instance = svc  # the tool calls AriadneService.get()

        resp = ariadne_explain(_FILE, kinds=['architecture'])

        assert set(resp.documents) == {'architecture'}
        assert 'explanation' not in resp.documents
        assert resp.total_documents == 1


class TestExplainSectionsOnly:
    """Slice 2 — `sections_only` returns a headings outline, not full bodies."""

    def test_sections_only_strips_bodies_to_headings(self, library: Library) -> None:
        body = (
            '# Title\n'
            'intro prose paragraph.\n'
            '## Section A\n'
            'body of section A.\n'
            '## Section B\n'
            'body of section B.\n'
        )
        library.add_document(content_type='architecture', title='Arch', content=body, source_files=[_FILE])

        full = library.explain(_FILE)['documents']['architecture'][0]['content']
        assert 'intro prose paragraph.' in full  # baseline: full body present without the flag

        outline = library.explain(_FILE, sections_only=True)['documents']['architecture'][0]['content']
        assert '# Title' in outline
        assert '## Section A' in outline
        assert '## Section B' in outline
        assert 'intro prose paragraph.' not in outline  # body prose stripped
        assert 'body of section A.' not in outline

    def test_sections_only_threads_through_tool(self, library: Library) -> None:
        library.add_document(content_type='architecture', title='Arch',
                             content='# Heading\nprose body.', source_files=[_FILE])
        svc = AriadneService()
        svc._library = library
        AriadneService._instance = svc

        resp = ariadne_explain(_FILE, sections_only=True)
        doc = resp.documents['architecture'][0]
        assert '# Heading' in doc.content
        assert 'prose body.' not in doc.content


class TestExplainPagination:
    """Slice 3 — pagination: full-fidelity slices; the caller pages until satisfied,
    worst case paging through everything. Content is never blanked or truncated.
    """

    @staticmethod
    def _seed(library: Library, n: int) -> None:
        for i in range(n):
            library.add_document(content_type='explanation', title=f'Doc {i}',
                                 content=f'full body of doc {i}', source_files=[_FILE])

    def test_limit_returns_full_fidelity_page_and_signals_more(self, library: Library) -> None:
        self._seed(library, 5)
        page = library.explain(_FILE, limit=2)
        entries = [e for docs in page['documents'].values() for e in docs]
        assert len(entries) == 2
        assert all(e['content'] for e in entries)   # complete content — nothing blanked/truncated
        assert page['total_documents'] == 5          # full total still reported
        assert page['returned'] == 2
        assert page['next_offset'] == 2              # caller can continue

    def test_paging_through_all_covers_every_doc_exactly_once(self, library: Library) -> None:
        self._seed(library, 5)
        seen: list[str] = []
        offset: int | None = 0
        while offset is not None:
            page = library.explain(_FILE, offset=offset, limit=2)
            seen += [e['id'] for docs in page['documents'].values() for e in docs]
            offset = page['next_offset']
        assert len(seen) == 5 and len(set(seen)) == 5   # no loss, no duplication

    def test_default_returns_entire_response(self, library: Library) -> None:
        self._seed(library, 3)
        result = library.explain(_FILE)               # default / worst case = everything
        entries = [e for docs in result['documents'].values() for e in docs]
        assert len(entries) == 3
        assert result['next_offset'] is None

    def test_offset_without_limit_returns_remainder(self, library: Library) -> None:
        """Offset with no limit ('skip the first N, give me the rest') still works."""
        self._seed(library, 5)
        page = library.explain(_FILE, offset=2)
        entries = [e for docs in page['documents'].values() for e in docs]
        assert len(entries) == 3                       # 5 total, first 2 skipped
        assert page['total_documents'] == 5
        assert page['offset'] == 2
        assert page['next_offset'] is None

    def test_unknown_file_reports_nothing_found(self, library: Library) -> None:
        """A file with no docs returns an empty, well-formed response (not an error)."""
        result = library.explain('does/not/exist.py')
        assert result['total_documents'] == 0
        assert result['documents'] == {}
        assert 'No documentation found' in result['summary']
        assert result['next_offset'] is None

    def test_pagination_threads_through_tool(self, library: Library) -> None:
        self._seed(library, 4)
        svc = AriadneService()
        svc._library = library
        AriadneService._instance = svc

        resp = ariadne_explain(_FILE, limit=2)
        entries = [e for docs in resp.documents.values() for e in docs]
        assert len(entries) == 2
        assert resp.total_documents == 4
        assert resp.next_offset == 2


class TestPaginatedDocQuery:
    """Slice 3 (memory-bound) — find_documents_page_by_source_files pushes LIMIT/OFFSET (+ kinds)
    and COUNT(*) OVER () to SQL, so only the page's rows are loaded while the full total is reported.
    """

    def test_returns_bounded_page_with_full_total(self, library: Library) -> None:
        for i in range(5):
            library.add_document(content_type='explanation', title=f'D{i}',
                                 content=f'body {i}', source_files=[_FILE])
        page, total = library.find_documents_page_by_source_files([_FILE], offset=0, limit=2)
        assert len(page) == 2          # only the page loaded into memory
        assert total == 5              # full count via COUNT(*) OVER (), no extra query

    def test_kinds_pushed_into_sql(self, library: Library) -> None:
        library.add_document(content_type='architecture', title='A', content='a', source_files=[_FILE])
        library.add_document(content_type='explanation', title='E', content='e', source_files=[_FILE])
        page, total = library.find_documents_page_by_source_files(
            [_FILE], content_types=['architecture'], offset=0, limit=10)
        assert total == 1
        assert [d.content_type for d in page] == ['architecture']

    def test_offset_walks_without_overlap(self, library: Library) -> None:
        for i in range(5):
            library.add_document(content_type='explanation', title=f'D{i}', content=f'b{i}', source_files=[_FILE])
        ids: list[str] = []
        for off in (0, 2, 4):
            page, total = library.find_documents_page_by_source_files([_FILE], offset=off, limit=2)
            ids += [d.id for d in page]
            assert total == 5
        assert len(ids) == 5 and len(set(ids)) == 5

    def test_empty_file_paths_returns_empty(self, library: Library) -> None:
        assert library.find_documents_page_by_source_files([], limit=5) == ([], 0)

    def test_no_match_returns_empty(self, library: Library) -> None:
        library.add_document(content_type='explanation', title='E', content='e', source_files=[_FILE])
        page, total = library.find_documents_page_by_source_files(['other/absent.py'], limit=5)
        assert page == [] and total == 0
