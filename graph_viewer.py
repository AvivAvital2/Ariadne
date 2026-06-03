"""Self-contained HTML graph visualization for Ariadne dependency graph.

Generates a single HTML file with embedded D3.js force-directed graph.
No server needed — opens directly in any browser.
"""
from __future__ import annotations

import json
from pathlib import Path


def generate_graph_html(graph_data: dict, output_path: Path, title: str = 'Ariadne Dependency Graph') -> Path:
    """Generate a self-contained HTML file with interactive graph visualization.

    Args:
        graph_data: Dict with 'nodes' and 'edges' arrays from Library.export_graph_json().
        output_path: Where to write the HTML file.
        title: Page title.

    Returns:
        Path to the generated HTML file.
    """
    graph_json = json.dumps(graph_data, indent=None)
    html = _HTML_TEMPLATE.replace('{{GRAPH_DATA}}', graph_json).replace('{{TITLE}}', title)
    output_path.write_text(html, encoding='utf-8')
    return output_path


_HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{TITLE}}</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; overflow: hidden; }
#container { display: flex; height: 100vh; }
#graph { flex: 1; }
#sidebar { width: 320px; background: #16213e; padding: 16px; overflow-y: auto; border-left: 1px solid #0f3460; }
#sidebar h2 { color: #e94560; margin-bottom: 12px; font-size: 14px; }
#sidebar h3 { color: #ccc; font-size: 12px; margin: 8px 0 4px; }
#sidebar p, #sidebar li { font-size: 12px; line-height: 1.5; color: #aaa; }
#sidebar ul { list-style: none; padding-left: 8px; }
#controls { padding: 12px 16px; background: #16213e; border-bottom: 1px solid #0f3460; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
#controls label { font-size: 11px; cursor: pointer; }
#controls input[type="checkbox"] { margin-right: 4px; }
#search { background: #0f3460; border: 1px solid #333; color: #e0e0e0; padding: 4px 8px; border-radius: 4px; font-size: 12px; width: 200px; }
.legend { display: flex; gap: 12px; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 4px; font-size: 11px; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; }
.filter-btn { cursor: pointer; padding: 2px 6px; border-radius: 4px; border: 1px solid transparent; user-select: none; }
.filter-btn:hover { border-color: #555; }
.filter-btn.active { border-color: #888; }
.filter-btn:not(.active) { opacity: 0.35; }
svg text { fill: #ccc; font-size: 10px; pointer-events: none; }
</style>
</head>
<body>
<div id="controls">
  <input type="text" id="search" placeholder="Search nodes...">
  <label><input type="checkbox" id="toggle-imports" checked> Imports</label>
  <label><input type="checkbox" id="toggle-documents" checked> Documents</label>
  <label><input type="checkbox" id="toggle-topic" checked> Topics</label>
  <div class="legend">
    <div class="legend-item filter-btn active" data-filter="file"><div class="legend-dot" style="background:#4a9eff"></div>File</div>
    <div class="legend-item filter-btn active" data-filter="undocumented"><div class="legend-dot" style="background:#e94560"></div>Undocumented</div>
    <div class="legend-item filter-btn active" data-filter="explanation"><div class="legend-dot" style="background:#4ecca3"></div>Explanation</div>
    <div class="legend-item filter-btn active" data-filter="architecture"><div class="legend-dot" style="background:#f0a500"></div>Architecture</div>
    <div class="legend-item filter-btn active" data-filter="topic"><div class="legend-dot" style="background:#a855f7"></div>Topic</div>
    <div class="legend-item filter-btn active" data-filter="finding"><div class="legend-dot" style="background:#fbbf24"></div>Finding</div>
    <span style="color:#555;margin:0 4px">|</span>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(220,60,60,0.3);border:1px solid rgba(220,60,60,0.5)"></div>Low coverage</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(220,220,0,0.3);border:1px solid rgba(220,220,0,0.5)"></div>Partial</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(0,200,60,0.3);border:1px solid rgba(0,200,60,0.5)"></div>Well covered</div>
  </div>
</div>
<div id="container">
  <svg id="graph"></svg>
  <div id="sidebar">
    <h2>Ariadne Dependency Graph</h2>
    <p>Click a node to see details. Scroll to zoom. Drag to pan.</p>
    <div id="node-details"></div>
  </div>
</div>
<script>
const graphData = {{GRAPH_DATA}};

const colorMap = {
  file: '#4a9eff',
  undocumented: '#e94560',
  explanation: '#4ecca3',
  architecture: '#f0a500',
  topic: '#a855f7',
  finding: '#fbbf24',
};

function nodeColor(d) {
  if (d.type === 'file' && d.doc_count === 0) return colorMap.undocumented;
  return colorMap[d.type] || colorMap.file;
}

function edgeCount(d) {
  return graphData.edges.filter(e => e.source.id === d.id || e.target.id === d.id || e.source === d.id || e.target === d.id).length;
}

function nodeRadius(d) {
  return Math.max(4, Math.min(20, 3 + Math.sqrt(edgeCount(d)) * 2));
}

// Coverage ratio: doc_count relative to edge count. 2 docs per file = fully covered.
// 0 = undocumented (red), 1 = fully covered (green)
function coverageRatio(d) {
  if (d.type !== 'file') return 1; // non-file nodes are "covered"
  const edges = edgeCount(d);
  if (edges === 0 && d.doc_count === 0) return 0.5; // isolated, no docs = neutral
  const needed = Math.max(1, edges * 0.15); // ~15% of edges should have doc coverage
  return Math.min(1, (d.doc_count || 0) / needed);
}

// Red (0) → Yellow (0.5) → Green (1)
function coverageColor(ratio) {
  if (ratio <= 0.5) {
    const t = ratio * 2; // 0..1 within red→yellow
    const r = 220, g = Math.round(60 + t * 160), b = Math.round(60 * (1 - t));
    return `${r},${g},${b}`;
  } else {
    const t = (ratio - 0.5) * 2; // 0..1 within yellow→green
    const r = Math.round(220 * (1 - t)), g = Math.round(220 - t * 20), b = Math.round(60 * t);
    return `${r},${g},${b}`;
  }
}

const edgeColorMap = { imports: '#335577', documents: '#2a5a3a', topic_member: '#5a3a6a' };

const svg = d3.select('#graph');
const width = window.innerWidth - 320;
const height = window.innerHeight - 48;
svg.attr('width', width).attr('height', height);

const g = svg.append('g');
svg.call(d3.zoom().scaleExtent([0.1, 8]).on('zoom', (event) => {
  g.attr('transform', event.transform);
  currentZoom = event.transform.k;
  // Font: starts at 14px apparent, shrinks proportionally when zooming in (max 2x = 28px apparent)
  const apparentSize = Math.min(28, Math.max(6, 14 / Math.max(1, currentZoom)));
  label.attr('font-size', (apparentSize / currentZoom) + 'px');
  // Rings: more visible when zoomed out, borderline transparent when zoomed in
  const ringOpacity = Math.min(0.25, Math.max(0.03, 0.15 / currentZoom));
  const strokeOpacity = Math.min(0.4, Math.max(0.05, 0.25 / currentZoom));
  ring.attr('fill', d => `rgba(${coverageColor(coverageRatio(d))},${ringOpacity})`)
      .attr('stroke', d => `rgba(${coverageColor(coverageRatio(d))},${strokeOpacity})`);
}));

const simulation = d3.forceSimulation(graphData.nodes)
  .force('link', d3.forceLink(graphData.edges).id(d => d.id).distance(60).strength(0.3))
  .force('charge', d3.forceManyBody().strength(-80))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 2));

// Coverage tier rings — drawn behind everything
let currentZoom = 1;
const ring = g.append('g').selectAll('circle').data(graphData.nodes.filter(d => d.type === 'file')).join('circle')
  .attr('r', d => nodeRadius(d) + 12)
  .attr('fill', d => `rgba(${coverageColor(coverageRatio(d))},0.12)`)
  .attr('stroke', d => `rgba(${coverageColor(coverageRatio(d))},0.25)`)
  .attr('stroke-width', 1)
  .attr('pointer-events', 'none');

const link = g.append('g').selectAll('line').data(graphData.edges).join('line')
  .attr('stroke', d => edgeColorMap[d.type] || '#333')
  .attr('stroke-width', d => d.type === 'imports' ? 1.5 : 0.8)
  .attr('stroke-opacity', 0.6);

const node = g.append('g').selectAll('circle').data(graphData.nodes).join('circle')
  .attr('r', d => nodeRadius(d))
  .attr('fill', d => nodeColor(d))
  .attr('stroke', '#fff')
  .attr('stroke-width', 0.5)
  .attr('cursor', 'pointer')
  .call(d3.drag()
    .on('start', (event, d) => { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
    .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
    .on('end', (event, d) => { if (!event.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
  );

const label = g.append('g').selectAll('text').data(graphData.nodes.filter(d => {
  const edges = graphData.edges.filter(e => e.source.id === d.id || e.target.id === d.id).length;
  return edges >= 5;
})).join('text')
  .text(d => d.title.length > 25 ? d.title.slice(0, 22) + '...' : d.title)
  .attr('dx', 12).attr('dy', 4).attr('font-size', '14px');

node.on('click', (event, d) => {
  const edges = graphData.edges.filter(e => e.source.id === d.id || e.target.id === d.id);
  const inbound = edges.filter(e => e.target.id === d.id);
  const outbound = edges.filter(e => e.source.id === d.id);
  document.getElementById('node-details').innerHTML = `
    <h3>${d.title}</h3>
    <p><strong>Type:</strong> ${d.type}</p>
    <p><strong>ID:</strong> <code style="font-size:10px;color:#888">${d.id.length > 40 ? '...' + d.id.slice(-40) : d.id}</code></p>
    <p><strong>Docs:</strong> ${d.doc_count || 0}</p>
    <p><strong>Edges:</strong> ${edges.length} (${inbound.length} in, ${outbound.length} out)</p>
    ${inbound.length ? '<h3>Inbound</h3><ul>' + inbound.slice(0,10).map(e => '<li>' + (e.source.title || e.source) + ' <span style="color:#666">(' + e.type + ')</span></li>').join('') + '</ul>' : ''}
    ${outbound.length ? '<h3>Outbound</h3><ul>' + outbound.slice(0,10).map(e => '<li>' + (e.target.title || e.target) + ' <span style="color:#666">(' + e.type + ')</span></li>').join('') + '</ul>' : ''}
  `;
  // Highlight connected
  node.attr('opacity', n => n.id === d.id || edges.some(e => e.source.id === n.id || e.target.id === n.id) ? 1 : 0.15);
  link.attr('opacity', e => e.source.id === d.id || e.target.id === d.id ? 1 : 0.05);
});

svg.on('click', (event) => {
  if (event.target.tagName === 'svg' || event.target.tagName === 'rect') {
    node.attr('opacity', 1);
    link.attr('opacity', 0.6);
  }
});

simulation.on('tick', () => {
  link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  ring.attr('cx', d => d.x).attr('cy', d => d.y);
  node.attr('cx', d => d.x).attr('cy', d => d.y);
  label.attr('x', d => d.x).attr('y', d => d.y);
});

// Search
document.getElementById('search').addEventListener('input', (e) => {
  const q = e.target.value.toLowerCase();
  if (!q) { node.attr('opacity', 1); link.attr('opacity', 0.6); return; }
  node.attr('opacity', d => d.title.toLowerCase().includes(q) || d.id.toLowerCase().includes(q) ? 1 : 0.1);
  link.attr('opacity', 0.05);
});

// Node type filter (legend buttons)
function effectiveType(d) {
  if (d.type === 'file' && (d.doc_count || 0) === 0) return 'undocumented';
  return d.type;
}

const activeFilters = new Set(['file', 'undocumented', 'explanation', 'architecture', 'topic', 'finding']);

function applyNodeFilters() {
  node.attr('display', d => activeFilters.has(effectiveType(d)) ? null : 'none');
  ring.attr('display', d => activeFilters.has(effectiveType(d)) ? null : 'none');
  label.attr('display', d => activeFilters.has(effectiveType(d)) ? null : 'none');
  // Hide edges where both endpoints are hidden
  link.attr('display', d => {
    const srcVis = activeFilters.has(effectiveType(d.source));
    const tgtVis = activeFilters.has(effectiveType(d.target));
    return (srcVis && tgtVis) ? null : 'none';
  });
}

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const filter = btn.dataset.filter;
    if (activeFilters.has(filter)) {
      activeFilters.delete(filter);
      btn.classList.remove('active');
    } else {
      activeFilters.add(filter);
      btn.classList.add('active');
    }
    applyNodeFilters();
  });
});

// Edge type toggles
['imports', 'documents', 'topic'].forEach(type => {
  const realType = type === 'topic' ? 'topic_member' : type;
  document.getElementById('toggle-' + type).addEventListener('change', (e) => {
    link.filter(d => d.type === realType).attr('display', e.target.checked ? null : 'none');
  });
});
</script>
</body>
</html>'''
