"""Theme-routed aisle selection — slice 1 of the expert-aisles architecture
(designs/spool-expert-aisles.md §3, §7).

The router is the cheap coarse tier: given a question and the enabled spool
aisles (each carrying its themes as embedding vectors), pick the aisle(s) whose
themes are relevant — so an unrelated question wakes NO aisle (no tax), and the
chosen aisle(s) then do their own precise retrieval. Pure math over precomputed
theme vectors; no live embedder.
"""
from __future__ import annotations

from spool_router import Aisle, route


class TestThemeRouting:
    def test_routes_to_the_matching_aisle_only(self) -> None:
        databricks = Aisle('databricks', ((1.0, 0.0, 0.0), (0.9, 0.1, 0.0)))
        terraform = Aisle('terraform', ((0.0, 1.0, 0.0),))
        picked = route((0.95, 0.05, 0.0), [databricks, terraform], threshold=0.5)
        assert [a.name for a in picked] == ['databricks']

    def test_unrelated_question_wakes_no_aisle(self) -> None:
        # the load-bearing property: enable aisles freely; an off-topic question
        # routes to nothing → answered from the library alone, zero aisle cost.
        databricks = Aisle('databricks', ((1.0, 0.0, 0.0),))
        terraform = Aisle('terraform', ((0.0, 1.0, 0.0),))
        assert route((0.0, 0.0, 1.0), [databricks, terraform],
                     threshold=0.5) == []

    def test_ranks_by_best_theme_and_caps_top_k(self) -> None:
        a = Aisle('a', ((1.0, 0.0),))
        b = Aisle('b', ((0.8, 0.2),))
        c = Aisle('c', ((0.6, 0.4),))
        picked = route((1.0, 0.0), [c, a, b], threshold=0.1, top_k=2)
        assert [x.name for x in picked] == ['a', 'b']   # best-first, capped

    def test_below_threshold_excluded(self) -> None:
        weak = Aisle('weak', ((0.3, 0.95),))   # ~0.30 cosine vs a (1,0) query
        assert route((1.0, 0.0), [weak], threshold=0.5) == []

    def test_best_of_several_themes_decides(self) -> None:
        # an aisle qualifies on its BEST-matching theme, not its average.
        mixed = Aisle('mixed', ((0.0, 1.0), (1.0, 0.0)))   # one far, one exact
        picked = route((1.0, 0.0), [mixed], threshold=0.5)
        assert [a.name for a in picked] == ['mixed']

    def test_no_aisles_returns_empty(self) -> None:
        assert route((1.0, 0.0), [], threshold=0.5) == []
