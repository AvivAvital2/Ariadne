"""Tests for graph_viewer HTML generation."""
from __future__ import annotations

from pathlib import Path

from graph_viewer import generate_graph_html


class TestGraphViewer:
    """Tests for the graph visualization generator."""

    def test_generate_empty_graph(self, tmp_path: Path) -> None:
        """Empty graph should produce valid HTML."""
        out = generate_graph_html({'nodes': [], 'edges': []}, tmp_path / 'graph.html')
        assert out.exists()
        content = out.read_text()
        assert '<!DOCTYPE html>' in content
        assert 'graphData' in content
        assert '"nodes": []' in content

    def test_generate_with_nodes(self, tmp_path: Path) -> None:
        """Graph with nodes should embed them in the HTML."""
        data = {
            'nodes': [
                {'id': 'a.py', 'type': 'file', 'title': 'a.py', 'doc_count': 0},
                {'id': 'doc1', 'type': 'explanation', 'title': 'Module A', 'doc_count': 1},
            ],
            'edges': [
                {'source': 'a.py', 'target': 'doc1', 'type': 'documents', 'weight': 1.0},
            ],
        }
        out = generate_graph_html(data, tmp_path / 'graph.html')
        content = out.read_text()
        assert 'a.py' in content
        assert 'Module A' in content
        assert 'documents' in content

    def test_output_is_self_contained(self, tmp_path: Path) -> None:
        """HTML should load D3.js and have inline CSS/JS."""
        out = generate_graph_html({'nodes': [], 'edges': []}, tmp_path / 'g.html')
        content = out.read_text()
        assert 'd3.v7.min.js' in content
        assert '<style>' in content
        assert '<script>' in content

    def test_custom_title(self, tmp_path: Path) -> None:
        """Custom title should appear in the HTML."""
        out = generate_graph_html({'nodes': [], 'edges': []}, tmp_path / 'g.html', title='My Custom Graph')
        content = out.read_text()
        assert 'My Custom Graph' in content

    def test_coverage_ring_colors(self, tmp_path: Path) -> None:
        """HTML should contain coverage color functions."""
        out = generate_graph_html({'nodes': [], 'edges': []}, tmp_path / 'g.html')
        content = out.read_text()
        assert 'coverageRatio' in content
        assert 'coverageColor' in content

    def test_filter_buttons_present(self, tmp_path: Path) -> None:
        """HTML should contain clickable filter buttons for node types."""
        out = generate_graph_html({'nodes': [], 'edges': []}, tmp_path / 'g.html')
        content = out.read_text()
        assert 'filter-btn' in content
        assert 'data-filter' in content
        assert 'Undocumented' in content
        assert 'Architecture' in content
