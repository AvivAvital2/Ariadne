"""Characterization test: ``get_related_batch`` must be byte-identical to
per-seed ``get_related`` (the oracle), while loading the graph once instead
of running per-node queries.

Performance must not cost accuracy, so this pins the exact contract on the
accuracy-sensitive cases: multi-hop distances, parallel edges (differing
weights between the same pair), a hub with more than ``limit`` equidistant
neighbours (the distance-tie cut), and a non-document file-path node that
must be traversed but never hydrated into results.

Synthetic graph only: source ``src1``, docs ``A``…``D``/``hub``/``n*``,
non-doc node ``f.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library

_EMB = np.zeros(4, dtype=np.float32)


@pytest.fixture
def lib(tmp_path: Path) -> Library:
    library = Library(tmp_path / 'g.db')
    yield library
    library.close()


def _doc(library: Library, doc_id: str) -> None:
    library.add_document(
        content_type='explanation', title=f'title-{doc_id}',
        content=f'content-{doc_id}', embedding=_EMB,
        doc_id=doc_id, source_name='src1',
    )


def _edge(library: Library, src: str, tgt: str, weight: float = 1.0,
          edge_type: str = 'imports') -> None:
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'INSERT OR REPLACE INTO doc_graph '
            '(source_id, target_id, edge_type, weight) VALUES (?, ?, ?, ?)',
            (src, tgt, edge_type, weight),
        )
        conn.commit()


def _build_graph(library: Library) -> list[str]:
    docs = ['iso', 'A', 'B', 'C', 'D', 'hub', 'fp_src', 'fp_dst', 'w0', 'w1']
    docs += [f'n{i}' for i in range(15)]
    for d in docs:
        _doc(library, d)
    # chain A-B-C-D, plus a parallel A-B edge with a different weight + type
    _edge(library, 'A', 'B')
    _edge(library, 'B', 'C')
    _edge(library, 'C', 'D')
    _edge(library, 'A', 'B', weight=2.0, edge_type='semantic_neighbor')
    # hub with 15 equidistant neighbours (> limit 10 → tie-break at the cut)
    for i in range(15):
        _edge(library, 'hub', f'n{i}')
    # non-document file-path node bridging fp_src -> f.py -> fp_dst
    _edge(library, 'fp_src', 'f.py')
    _edge(library, 'f.py', 'fp_dst')
    # reverse edge duplicating A-B@1.0 → exercises the adjacency dedupe;
    # zero-weight edge → exercises the (weight <= 0 -> 10.0) distance branch
    _edge(library, 'B', 'A', weight=1.0)
    _edge(library, 'w0', 'w1', weight=0.0)
    # every seed we query: the docs plus the non-doc file node itself
    return docs + ['f.py']


def test_get_related_batch_matches_get_related(lib: Library) -> None:
    node_ids = _build_graph(lib)
    batch = lib.get_related_batch(node_ids)
    for node in node_ids:
        assert batch[node] == lib.get_related(node), f'mismatch for seed {node!r}'
