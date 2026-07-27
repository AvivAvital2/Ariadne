"""TDD for the ask() balanced synthesis context.

The anchored search returns repo (anchor) + spool (ground) with the repo floor
first; a plain top-k truncation therefore skews to the repo and starves the
ground, so WITH-spool answers underperform. ``_balanced_ask_docs`` takes the
top of EACH half so the synthesis always sees both — the repo subject and
enough spool context to cross-reference.
"""
from __future__ import annotations

from types import SimpleNamespace

from ariadne_mcp.service_analysis import _balanced_ask_docs


def _doc(doc_id, source_name):
    return SimpleNamespace(id=doc_id, title=doc_id, source_name=source_name)


class TestBalancedAskDocs:
    def test_takes_top_of_both_halves(self):
        # Ranked results: 5 repo (anchor) first (the floor), then 3 spool.
        docs = (
            [_doc(f'repo{i}', 'ao-core') for i in range(5)]
            + [_doc(f'spool{i}', 'spool:databricks') for i in range(3)]
        )
        out = _balanced_ask_docs(
            docs, frozenset({'spool:databricks'}), anchor_n=3, ground_n=3,
        )
        ids = [d.id for d in out]
        # top-3 repo AND top-3 spool — not 5 repo + 0 spool (the old top-k skew)
        assert ids == ['repo0', 'repo1', 'repo2', 'spool0', 'spool1', 'spool2']

    def test_no_spool_returns_repo_only(self):
        # No ground admitted (e.g. a CONTROL question or no spool) => repo only,
        # so WITH-spool never injects irrelevant ground → no-harm preserved.
        docs = [_doc(f'repo{i}', 'ao-core') for i in range(6)]
        out = _balanced_ask_docs(
            docs, frozenset({'spool:databricks'}), anchor_n=4, ground_n=4,
        )
        assert [d.id for d in out] == ['repo0', 'repo1', 'repo2', 'repo3']

    def test_ground_ranking_preserved_within_half(self):
        # Whatever order the ranker produced within each half is kept.
        docs = [
            _doc('r0', 'ao-core'), _doc('s0', 'spool:databricks'),
            _doc('r1', 'ao-core'), _doc('s1', 'spool:databricks'),
        ]
        out = _balanced_ask_docs(
            docs, frozenset({'spool:databricks'}), anchor_n=1, ground_n=1,
        )
        assert [d.id for d in out] == ['r0', 's0']
