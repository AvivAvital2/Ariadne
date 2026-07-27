"""TDD for anchored-ground retrieval ranking.

See ``designs/spool-anchored-retrieval.md``. The user repo is the protected
anchor (the subject — where the question lives); the spool is subordinate
ground (context), admitted by BOTH query similarity AND similarity to the
anchored docs, diversified and relevance-gated. Pure ranking over synthetic
unit embeddings — no LLM, no DB.

Axes: 0 = query-topic, 1 = anchor/repo-topic, 2 = a complementary facet.
"""
from __future__ import annotations

import numpy as np

from library.anchored_retrieval import anchored_rank, select_ground


def _v(*xs: float) -> np.ndarray:
    a = np.zeros(6, dtype=np.float32)
    for i, x in enumerate(xs):
        a[i] = x
    n = float(np.linalg.norm(a))
    return a / n if n else a


Q = _v(1, 1)   # hybrid query: spans query-topic + repo-topic
A = _v(0, 1)   # the repo's anchored doc (anchor-topic)


class TestAnchoredRank:
    # ---- 1: the repo anchor floor is never displaced by the spool ----------
    def test_anchor_floor_never_displaced(self) -> None:
        g_hot = _v(1, 1)   # identical to the query — maximal query similarity
        out = anchored_rank(
            Q, anchor=[('A', A)], ground=[('G', g_hot)],
            limit=1, anchor_floor=1, w_q=0.5, w_a=0.5, gate=0.0, diversity=0.5,
        )
        assert out == ['A']   # ground cannot evict the floor-protected anchor

    # ---- 2: ground close to BOTH query and anchor is admitted as context ---
    def test_relevant_ground_admitted(self) -> None:
        g_rel = _v(1, 1)   # high query_sim; anchor_sim = cos((1,1),(0,1)) ~ .707
        out = anchored_rank(
            Q, anchor=[('A', A)], ground=[('G', g_rel)],
            limit=3, anchor_floor=1, w_q=0.5, w_a=0.5, gate=0.5, diversity=0.5,
        )
        assert out[0] == 'A' and 'G' in out

    # ---- 3: query-similar but anchor-distant ground is gated out -----------
    def test_query_only_ground_gated_out(self) -> None:
        g_q = _v(1, 0)   # query_sim ~ .707, anchor_sim = 0 -> combined ~ .354
        out = anchored_rank(
            Q, anchor=[('A', A)], ground=[('Gq', g_q)],
            limit=3, anchor_floor=1, w_q=0.5, w_a=0.5, gate=0.5, diversity=0.5,
        )
        assert out == ['A']   # the spool is ground for the repo, not free-floating

    # ---- 4: diversity prefers a complementary facet over a near-duplicate --
    def test_diversity_prefers_complementary_facet(self) -> None:
        g_top = _v(1, 1)       # highest combined -> picked first
        g_dup = _v(1, 1)       # near-duplicate of g_top
        g_facet = _v(0, 1, 1)  # complementary facet (axis 2), still anchor-close
        out = anchored_rank(
            Q, anchor=[('A', A)],
            ground=[('top', g_top), ('dup', g_dup), ('facet', g_facet)],
            limit=3, anchor_floor=1, w_q=0.5, w_a=0.5, gate=0.3, diversity=0.6,
        )
        assert out[0] == 'A'
        assert 'top' in out and 'facet' in out
        assert 'dup' not in out   # MMR drops the redundant near-duplicate

    # ---- 5: weights demote bloat catalog docs ------------------------------
    def test_weights_demote_bloat(self) -> None:
        g_sig = _v(1, 1)
        g_bloat = _v(1, 1)   # identical embedding, but marked low-quality
        out = anchored_rank(
            Q, anchor=[('A', A)],
            ground=[('bloat', g_bloat), ('sig', g_sig)],
            limit=2, anchor_floor=1, w_q=0.5, w_a=0.5, gate=0.3, diversity=0.0,
            weights={'bloat': 0.3},
        )
        assert out == ['A', 'sig']   # signal ground beats the demoted bloat doc

    # ---- 6: no spool => pure query-similarity ranking (regression guard) ---
    def test_no_spool_matches_query_ranking(self) -> None:
        a1 = _v(1, 1)   # closest to Q
        a2 = _v(0, 1)   # farther
        a3 = _v(1, 0)   # farther
        out = anchored_rank(
            Q, anchor=[('a2', a2), ('a3', a3), ('a1', a1)], ground=[],
            limit=3, anchor_floor=1, w_q=0.5, w_a=0.5, gate=0.5, diversity=0.5,
        )
        assert out == ['a1', 'a2', 'a3']   # == today's query-ranked top-k


class TestSelectGround:
    """The reusable ground-selection primitive behind both the anchored search
    and the environment bridge (impact_radius / trace_flow). ``query_emb=None``
    is the bridge's anchor-only mode: rank spool ground purely by relevance to
    the target's own docs (the anchor) — "what environment context bears on
    this file/symbol", with no natural-language query.
    """

    # ---- anchor-only mode (the environment bridge) -------------------------
    def test_admits_anchor_relevant_gates_distant(self) -> None:
        anchor = [_v(0, 1)]                       # the file's own doc (topic 1)
        g_rel = _v(0, 1)                          # spool doc on the same topic
        g_far = _v(1, 0)                          # unrelated spool doc
        out = select_ground(
            None, anchor, [('rel', g_rel), ('far', g_far)],
            limit=5, gate=0.5, diversity=0.0,
        )
        assert out == ['rel']                     # relevant in, anchor-distant out

    def test_empty_when_no_ground(self) -> None:
        assert select_ground(None, [_v(0, 1)], [], limit=5, gate=0.5) == []

    def test_gate_excludes_when_nothing_relevant(self) -> None:
        # An unrelated target admits ~no environment context (dynamic sizing).
        out = select_ground(
            None, [_v(0, 1)], [('far', _v(1, 0))], limit=5, gate=0.5,
        )
        assert out == []

    def test_weights_demote_bloat(self) -> None:
        anchor = [_v(0, 1)]
        out = select_ground(
            None, anchor, [('bloat', _v(0, 1)), ('sig', _v(0, 1))],
            limit=1, gate=0.2, diversity=0.0, weights={'bloat': 0.3},
        )
        assert out == ['sig']                     # demoted bloat loses the slot

    # ---- query+anchor mode (what anchored_rank delegates) ------------------
    def test_query_and_anchor_combined(self) -> None:
        out = select_ground(
            _v(1, 0), [_v(0, 1)], [('g', _v(1, 1))],
            limit=1, w_q=0.5, w_a=0.5, gate=0.4, diversity=0.0,
        )
        assert out == ['g']                       # blends query AND anchor sim

    def test_diversifies_over_near_duplicate(self) -> None:
        Qc = _v(1, 1)
        top = _v(1, 1)
        dup = _v(1, 1)                            # near-duplicate of top
        facet = _v(0, 1, 1)                       # complementary, still relevant
        out = select_ground(
            Qc, [_v(0, 1)], [('top', top), ('dup', dup), ('facet', facet)],
            limit=2, w_q=0.5, w_a=0.5, gate=0.3, diversity=0.6,
        )
        assert 'top' in out and 'facet' in out and 'dup' not in out
