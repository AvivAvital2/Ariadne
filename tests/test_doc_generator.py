"""Tests for on-demand documentation generation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_generator import DocGenerator, _normalize_heading, _parse_sections


@pytest.fixture
def mock_library():
    lib = MagicMock()
    lib.list_documents.return_value = []
    lib.list_documents_lite.return_value = []
    lib.get_graph_stats.return_value = {'nodes': 10, 'edges': 20}
    lib.explain.return_value = {'documents': {}, 'summary': 'test'}
    lib.impact_radius.return_value = {'neighbors': []}
    return lib


@pytest.fixture
def doc_gen(mock_library, tmp_path):
    return DocGenerator(mock_library, tmp_path / 'generated-docs')


class TestParseSections:
    def test_splits_on_h2_headings(self):
        md = '# Title\nIntro\n\n## Section A\nContent A\n\n## Section B\nContent B'
        sections = _parse_sections(md)
        assert len(sections) == 3
        assert sections[0][0] == ''  # preamble
        assert sections[1][0] == '## Section A'
        assert sections[2][0] == '## Section B'

    def test_empty_input(self):
        assert _parse_sections('') == [('', '')]

    def test_no_headings(self):
        sections = _parse_sections('Just some text\nNo headings')
        assert len(sections) == 1
        assert sections[0][0] == ''


class TestNormalizeHeading:
    def test_strips_markdown(self):
        assert _normalize_heading('## Architecture Overview') == 'architecture overview'

    def test_strips_special_chars(self):
        assert _normalize_heading('## Key Modules & Patterns!') == 'key modules patterns'

    def test_empty(self):
        assert _normalize_heading('') == ''


class TestDocGeneratorReadme:
    @pytest.mark.asyncio
    async def test_generates_readme_file(self, doc_gen, mock_library):
        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='# My Project\n\nGenerated README'):
            results = await doc_gen.generate(['readme'])

        assert 'readme' in results
        path = results['readme']
        assert path.exists()
        assert 'My Project' in path.read_text()

    @pytest.mark.asyncio
    async def test_creates_output_dir(self, doc_gen):
        assert not doc_gen.output_dir.exists()
        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='# Test'):
            await doc_gen.generate(['readme'])
        assert doc_gen.output_dir.exists()


class TestDocGeneratorApi:
    @pytest.mark.asyncio
    async def test_generates_api_index_and_package_pages(self, doc_gen, mock_library, tmp_path):
        # Create a real Python file for AST extraction
        pkg_dir = tmp_path / 'pkg'
        pkg_dir.mkdir()
        (pkg_dir / 'mod.py').write_text('"""A module."""\ndef hello(name: str) -> str:\n    """Greet."""\n    return name\n')

        mock_doc = MagicMock()
        mock_doc.source_files = [str(pkg_dir / 'mod.py')]
        mock_doc.source_name = None
        mock_library.list_documents_lite.return_value = [mock_doc]

        # Patch _to_relative_path to return a package-relative path
        with patch('doc_generator._to_relative_path', side_effect=lambda fp: f'pkg/{Path(fp).name}'):
            results = await doc_gen.generate(['api'])

        assert 'api' in results
        assert results['api'].name == 'api-reference.md'
        index_content = results['api'].read_text()
        assert 'pkg' in index_content  # Package listed in index

        # Per-package page should exist
        api_dir = doc_gen.output_dir / 'api'
        assert api_dir.exists()
        pkg_page = api_dir / 'pkg.md'
        assert pkg_page.exists()
        assert 'hello' in pkg_page.read_text()


class TestDocGeneratorArchitecture:
    @pytest.mark.asyncio
    async def test_generates_architecture_guide(self, doc_gen, mock_library):
        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='# Architecture\nOverview'):
            results = await doc_gen.generate(['architecture'])

        assert 'architecture' in results
        content = results['architecture'].read_text()
        assert 'Architecture' in content


class TestUpdateReadmeInPlace:
    @pytest.mark.asyncio
    async def test_preserves_unmatched_sections(self, doc_gen, tmp_path):
        readme = tmp_path / 'README.md'
        readme.write_text('# My Project\n\n## Custom Section\nManual content\n\n## Architecture\nOld arch')

        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='# Project\n\n## Architecture\nNew arch\n\n## Overview\nNew overview'):
            result = await doc_gen.update_readme_in_place(readme)

        content = result.read_text()
        assert 'Manual content' in content  # Preserved
        assert 'New arch' in content  # Updated
        assert 'New overview' in content  # Appended

    @pytest.mark.asyncio
    async def test_creates_readme_if_missing(self, doc_gen, tmp_path):
        readme = tmp_path / 'nonexistent.md'
        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='# Generated'):
            result = await doc_gen.update_readme_in_place(readme)
        assert result.exists()


class TestDocGeneratorDiagrams:
    @pytest.mark.asyncio
    async def test_generates_all_diagram_formats(self, doc_gen, mock_library):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [
            ('src/module_a.py', 'src/module_b.py'),
            ('src/module_b.py', 'src/module_c.py'),
        ]
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_library._conn_provider = MagicMock()
        mock_library._conn_provider.acquire.return_value = mock_cm

        results = await doc_gen.generate(['diagrams'])
        assert 'diagrams' in results
        md_path = results['diagrams']
        assert md_path.suffix == '.md'
        assert '![Dependency Diagram]' in md_path.read_text()

        # Excalidraw file also generated
        excalidraw_path = md_path.parent / 'dependency-diagram.excalidraw'
        assert excalidraw_path.exists()
        import json
        data = json.loads(excalidraw_path.read_text())
        assert data['type'] == 'excalidraw'

        # SVG file also generated
        svg_path = md_path.parent / 'dependency-diagram.svg'
        assert svg_path.exists()
        assert '<svg' in svg_path.read_text()

    @pytest.mark.asyncio
    async def test_empty_graph_produces_placeholder(self, doc_gen, mock_library):
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_conn)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_library._conn_provider = MagicMock()
        mock_library._conn_provider.acquire.return_value = mock_cm

        results = await doc_gen.generate(['diagrams'])
        assert 'diagrams' in results
        assert 'No import graph data' in results['diagrams'].read_text()


class TestSvgDiagram:
    def test_builds_svg_from_layers(self):
        from doc_generator import _build_svg_diagram
        layers = [['a.py', 'b.py'], ['c.py']]
        edges = [('a.py', 'c.py')]
        svg = _build_svg_diagram(layers, edges)
        assert '<svg' in svg
        assert 'a.py' in svg
        assert 'c.py' in svg
        assert '<line' in svg  # edge
        assert '<rect' in svg  # node

    def test_empty_layers(self):
        from doc_generator import _build_svg_diagram
        svg = _build_svg_diagram([], [])
        assert '<svg' in svg


class TestMkdocsConfig:
    @pytest.mark.asyncio
    async def test_generates_mkdocs_yml(self, doc_gen, mock_library):
        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='# Test'):
            await doc_gen.generate(['readme'])
        config_path = doc_gen.output_dir / 'mkdocs.yml'
        assert config_path.exists()
        import yaml
        config = yaml.safe_load(config_path.read_text())
        assert config['site_name']
        assert config['theme']['name'] == 'material'
        assert config['nav']


class TestExcalidrawLayout:
    def test_builds_elements_from_layers(self):
        from doc_generator import _build_excalidraw_elements
        layers = [['a.py', 'b.py'], ['c.py']]
        edges = [('a.py', 'c.py')]
        elements = _build_excalidraw_elements(layers, edges)
        # 3 nodes + 1 arrow
        rects = [e for e in elements if e['type'] == 'rectangle']
        arrows = [e for e in elements if e['type'] == 'arrow']
        assert len(rects) == 3
        assert len(arrows) == 1

    def test_handles_no_edges(self):
        from doc_generator import _build_excalidraw_elements
        layers = [['a.py']]
        elements = _build_excalidraw_elements(layers, [])
        assert len(elements) == 1
        assert elements[0]['type'] == 'rectangle'


class TestNotebookCellBuilders:
    def test_markdown_cell_structure(self):
        from doc_generator import _build_markdown_cell
        cell = _build_markdown_cell('# Title\nBody')
        assert cell['cell_type'] == 'markdown'
        assert '# Title\n' in cell['source']

    def test_code_cell_structure(self):
        from doc_generator import _build_code_cell
        cell = _build_code_cell('x = 1\nprint(x)')
        assert cell['cell_type'] == 'code'
        assert cell['outputs'] == []
        assert cell['execution_count'] is None
        assert len(cell['source']) == 2

    def test_assemble_notebook_format(self):
        from doc_generator import _assemble_notebook, _build_code_cell, _build_markdown_cell
        cells = [_build_markdown_cell('# Hi'), _build_code_cell('x = 1')]
        nb = _assemble_notebook(cells)
        assert nb['nbformat'] == 4
        assert len(nb['cells']) == 2
        assert nb['metadata']['kernelspec']['language'] == 'python'


class TestDocGeneratorNotebooks:
    @pytest.mark.asyncio
    async def test_generates_notebook_file(self, doc_gen, mock_library):
        mock_doc = MagicMock()
        mock_doc.source_files = ['src/module.py']
        mock_library.list_documents_lite.return_value = [mock_doc]
        mock_library.explain.return_value = {
            'documents': {'explanation': [{'title': 'Module', 'content': 'Does stuff'}]},
            'summary': 'A useful module',
        }

        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='import module\nmodule.run()'):
            results = await doc_gen.generate(['notebooks'])

        assert 'notebooks' in results
        path = results['notebooks']
        assert path.suffix == '.ipynb'
        import json
        nb = json.loads(path.read_text())
        assert nb['nbformat'] == 4
        assert len(nb['cells']) >= 2  # title + at least one example

    @pytest.mark.asyncio
    async def test_empty_docs_produces_title_only(self, doc_gen, mock_library):
        mock_library.list_documents_lite.return_value = []
        with patch.object(doc_gen, '_llm', new_callable=AsyncMock):
            results = await doc_gen.generate(['notebooks'])
        import json
        nb = json.loads(results['notebooks'].read_text())
        assert len(nb['cells']) == 1  # Just the title cell


class TestDocGeneratorFaq:
    @pytest.mark.asyncio
    async def test_generates_faq_from_gotchas(self, doc_gen, mock_library):
        mock_gotcha = MagicMock()
        mock_gotcha.title = 'Watch out for X'
        mock_gotcha.content = "X can cause Y if you don't Z"
        mock_library.list_documents.side_effect = lambda content_type=None, limit=None: (
            [mock_gotcha] if content_type == 'gotcha' else []
        )

        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='## Q: What about X?\n\nWatch out for Y.'):
            results = await doc_gen.generate(['faq'])

        assert 'faq' in results
        content = results['faq'].read_text()
        assert 'Frequently Asked Questions' in content

    @pytest.mark.asyncio
    async def test_empty_faq_when_no_gotchas(self, doc_gen, mock_library):
        mock_library.list_documents.return_value = []
        results = await doc_gen.generate(['faq'])
        assert 'No gotchas' in results['faq'].read_text()


class TestDocGeneratorPatterns:
    @pytest.mark.asyncio
    async def test_generates_patterns_from_gotchas(self, doc_gen, mock_library):
        mock_gotcha = MagicMock()
        mock_gotcha.title = 'Mutable default args'
        mock_gotcha.content = 'Never use [] as default — use () instead'
        mock_gotcha.source_name = None
        mock_library.list_documents.side_effect = lambda content_type=None, limit=None: (
            [mock_gotcha] if content_type == 'gotcha' else []
        )

        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='## Patterns\nUse immutable defaults'):
            results = await doc_gen.generate(['patterns'])

        assert 'patterns' in results
        content = results['patterns'].read_text()
        assert 'Notable Patterns' in content

    @pytest.mark.asyncio
    async def test_empty_patterns_when_no_docs(self, doc_gen, mock_library):
        mock_library.list_documents.return_value = []
        results = await doc_gen.generate(['patterns'])
        assert 'No patterns' in results['patterns'].read_text()


class TestDocGeneratorDecisions:
    @pytest.mark.asyncio
    async def test_generates_decisions_from_arch_docs(self, doc_gen, mock_library):
        mock_arch = MagicMock()
        mock_arch.title = 'Auth Architecture'
        mock_arch.content = 'We chose JWT over session cookies because of scalability. The decision was...'
        mock_library.list_documents.side_effect = lambda content_type=None, limit=None: (
            [mock_arch] if content_type == 'architecture' else []
        )

        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='## JWT over Session Cookies\n\nWe chose JWT...'):
            results = await doc_gen.generate(['decisions'])

        assert 'decisions' in results
        content = results['decisions'].read_text()
        assert 'Design Decisions' in content

    @pytest.mark.asyncio
    async def test_empty_decisions_when_no_rationale(self, doc_gen, mock_library):
        mock_arch = MagicMock()
        mock_arch.title = 'Module Overview'
        mock_arch.content = 'This module handles user input.'  # No decision keywords
        mock_library.list_documents.side_effect = lambda content_type=None, limit=None: (
            [mock_arch] if content_type == 'architecture' else []
        )
        results = await doc_gen.generate(['decisions'])
        assert 'No design rationale' in results['decisions'].read_text()


class TestDocGeneratorDiff:
    @pytest.mark.asyncio
    async def test_generates_diff_summary(self, doc_gen, mock_library):
        with patch('git_ops.get_current_branch', return_value='feat/new-feature'), \
             patch('git_ops.get_changed_files_vs_main', return_value=['src/module.py', 'tests/test_module.py']), \
             patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value='## What changed\nNew feature added'):
            mock_library.explain.return_value = {'summary': 'Core module'}
            mock_library.impact_radius.return_value = {'total_affected_files': 3, 'affected_tests': 1}
            results = await doc_gen.generate(['diff'])

        assert 'diff' in results
        content = results['diff'].read_text()
        assert 'feat/new-feature' in content
        assert '2 files changed' in content

    @pytest.mark.asyncio
    async def test_no_changes_produces_placeholder(self, doc_gen, mock_library):
        with patch('git_ops.get_current_branch', return_value='main'), \
             patch('git_ops.get_changed_files_vs_main', return_value=[]):
            # Also mock config to return empty source paths
            with patch('config.get_config') as mock_cfg:
                mock_cfg.return_value.get_all_source_paths.return_value = {}
                results = await doc_gen.generate(['diff'])

        assert 'No changes detected' in results['diff'].read_text()


class TestUserFlows:
    @pytest.mark.asyncio
    async def test_generates_user_flows(self, doc_gen, mock_library):
        mock_doc = MagicMock()
        mock_doc.title = 'Analyze Pipeline'
        mock_doc.content = 'The pipeline flow: Step 1: ingest data. Step 2: generate features. Then enrichment happens next. Finally returns results.'
        mock_doc.source_files = []
        mock_doc.source_name = None
        mock_library.list_documents.side_effect = lambda content_type=None, limit=None: (
            [mock_doc] if content_type == 'architecture' else []
        )
        mock_library.list_documents_lite.return_value = []

        with patch.object(doc_gen, '_llm', new_callable=AsyncMock, return_value=(
            'FLOW: Analyze Pipeline\n'
            'STEPS: Ingest -> Feature Gen -> Enrichment -> Selection\n'
            'DESCRIPTION: Main analysis workflow'
        )):
            results = await doc_gen.generate(['user_flows'])

        assert 'user_flows' in results
        index = results['user_flows'].read_text()
        assert 'User Flows' in index
        assert 'Analyze Pipeline' in index

    @pytest.mark.asyncio
    async def test_empty_user_flows(self, doc_gen, mock_library):
        mock_library.list_documents.return_value = []
        mock_library.list_documents_lite.return_value = []
        results = await doc_gen.generate(['user_flows'])
        assert 'No user flows' in results['user_flows'].read_text()


class TestParseFlowOutput:
    def test_parses_flow_output(self):
        from doc_generator import _parse_flow_output
        text = 'FLOW: My Flow\nSTEPS: A -> B -> C\nDESCRIPTION: Test flow'
        flows = _parse_flow_output(text)
        assert len(flows) == 1
        assert flows[0][0] == 'My Flow'
        assert flows[0][1] == ['A', 'B', 'C']

    def test_multiple_flows(self):
        from doc_generator import _parse_flow_output
        text = 'FLOW: First\nSTEPS: A -> B\nDESCRIPTION: D1\nFLOW: Second\nSTEPS: X -> Y -> Z\nDESCRIPTION: D2'
        flows = _parse_flow_output(text)
        assert len(flows) == 2


class TestArchDiagrams:
    @pytest.mark.asyncio
    async def test_generates_system_overview(self, doc_gen, mock_library):
        mock_arch = MagicMock()
        mock_arch.title = 'Core Architecture'
        mock_arch.content = 'Core module handles types and schema. Step 1: ingest data. Step 2: process.'
        mock_arch.source_files = ['/path/to/myproject/pythonproject/core/__init__.py']
        mock_arch.source_name = None
        mock_library.list_documents.side_effect = lambda content_type=None, limit=None: (
            [mock_arch] if content_type == 'architecture' else []
        )

        results = await doc_gen.generate(['arch_diagrams'])
        assert 'arch_diagrams' in results
        index = results['arch_diagrams'].read_text()
        assert 'Architecture Diagrams' in index

    @pytest.mark.asyncio
    async def test_empty_arch_diagrams(self, doc_gen, mock_library):
        mock_library.list_documents.return_value = []
        results = await doc_gen.generate(['arch_diagrams'])
        assert 'No architecture documentation' in results['arch_diagrams'].read_text()


class TestComponentExtraction:
    def test_extracts_components_from_source_files(self):
        from doc_generator import _extract_components_from_docs
        mock_doc = MagicMock()
        mock_doc.title = 'Core Module'
        mock_doc.content = 'The core module handles types.'
        mock_doc.source_files = ['/path/to/myproject/pythonproject/core/__init__.py']
        components, interactions = _extract_components_from_docs([mock_doc])
        assert 'pythonproject.core' in components

    def test_components_to_layers(self):
        from doc_generator import _components_to_layers
        comps = {'A': 'desc A', 'B': 'desc B', 'C': 'desc C'}
        edges = [('A', 'B'), ('B', 'C')]
        layers, valid_edges = _components_to_layers(comps, edges)
        assert len(layers) >= 2
        assert ('A', 'B') in valid_edges


class TestDocsRead:
    def test_reads_existing_doc(self, tmp_path):
        doc_dir = tmp_path / 'generated-docs'
        doc_dir.mkdir()
        (doc_dir / 'README.md').write_text('# Hello')

        from doc_generator import DOC_TYPE_TO_FILE
        assert DOC_TYPE_TO_FILE['readme'] == 'README.md'

    def test_doc_type_mapping_complete(self):
        from doc_generator import DOC_TYPE_TO_FILE, DOC_TYPES
        for dtype in DOC_TYPES:
            assert dtype in DOC_TYPE_TO_FILE, f'Missing mapping for {dtype}'


class TestUnknownDocType:
    @pytest.mark.asyncio
    async def test_skips_unknown_types(self, doc_gen):
        results = await doc_gen.generate(['nonexistent'])
        assert results == {}
