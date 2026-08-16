#!/usr/bin/env python3
"""One table: Ariadne against the bare LLM, on every parameter actually measured.

Assembled from artifacts on disk rather than retyped, so a number cannot drift
from the run that produced it. Every cell is either a measurement or the reason
there isn't one — an unmeasured parameter prints ``not measured`` and never a
blank that could read as a zero.

Three kinds of row, and they are not interchangeable:

  BOTH   the same quantity for both arms, from the same fixed
         ``chain_requirements.json`` on the same de-breadcrumbed questions.
         Only these are a head-to-head.
  ARM    meaningful for one arm only. Chain connectivity is a property of a
         graph; answer-first derivation is a property of an agent that
         explores. Printed for context, never as a comparison.
  NOTE   a standing caveat that changes how the numbers should be read.

    python evaluation/spool-clean-room/scorecard.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

BARE = HERE / 'chain-answers-cleanroom-questions_debcrumb_ask.json'
ARIADNE = HERE / 'chain-answers-ariadne-arm.json'
STRUCTURAL = HERE / 'ariadne-structural.json'
REQS = HERE / 'chain_requirements.json'

MISSING = 'not measured'

# --- the bottom line -------------------------------------------------------
# Three pillars, weighted. Evidence carries the most because it is the only one
# that cannot be produced from memory: a model recalls a mechanism and a file
# name, but not the line a symbol occupies at a pinned revision.
#
# Correctness is EARNED, not raw. An answer that is right but shows no evidence
# contributes zero to it. That is not a penalty bolted on: on a corpus the model
# has memorised, the judge scores evidenced and unevidenced answers within about
# a point of each other, so raw correctness measures recall of Spark/Delta and
# little else. The run's actual gap is printed under NOTES rather than asserted
# here, so this rationale cannot go stale against the numbers it explains.
WEIGHTS = {
    'evidence': 0.40,      # of the required mechanisms, how many are quoted at
                           # a line that actually matches the file
    'completeness': 0.30,  # does the answer join first mechanism to last:
                           # 1.0 every hop quoted, 0.6 ends quoted + middle
                           # articulated, 0.0 neither
    'correctness': 0.30,   # judge score /10, counted ONLY where admissible
}


def _load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _chain_stats(rows) -> dict:
    """Grounding-gate figures, identical definition for either arm."""
    if not rows:
        return {}
    n = len(rows)
    cs = [r['chain_score'] for r in rows if r.get('chain_score') is not None]
    return {
        'n': n,
        'admissible': f"{sum(1 for r in rows if r['admissible'])}/{n}",
        'full_route': str(sum(1 for r in rows if r.get('complete_full'))),
        'endpoint_route': str(sum(1 for r in rows if r.get('complete_endpoints')
                                  and not r.get('complete_full'))),
        'chain_evidence': f'{st.mean(cs):.0%}' if cs else MISSING,
        'zero_quote': f"{sum(1 for r in rows if r['verified_quotes'] == 0)}/{n}",
        'ends_no_middle': f"{sum(1 for r in rows if r['verified_quotes'] >= 2 and r.get('missing_middle'))}/{n}",
        'answer_first': f"{sum(1 for r in rows if r.get('answer_first'))}/{n}",
        'correctness': (f"{st.mean([r['correctness'] for r in rows if r.get('correctness') is not None]):.2f}"
                        if any(r.get('correctness') is not None for r in rows) else MISSING),
    }


def _mean_or_none(xs):
    return st.mean(xs) if xs else None


def _gap(graded):
    """Evidenced-minus-unevidenced correctness, or None if one side is empty."""
    a = [r['correctness'] for r in graded if r['admissible']]
    b = [r['correctness'] for r in graded if not r['admissible']]
    return (st.mean(a) - st.mean(b)) if a and b else None


def _pillars(rows) -> dict | None:
    """The three pillars and the single weighted score, 0-100.

    Where correctness has not been graded the score is computed over the
    weights that ARE measured and flagged PROVISIONAL. Treating an ungraded
    pillar as zero would report a failure that was never observed.
    """
    if not rows:
        return None
    n = len(rows)
    evidence = st.mean([r.get('chain_score') or 0.0 for r in rows])
    completeness = st.mean([
        1.0 if r.get('complete_full')
        else 0.6 if r.get('complete_endpoints')
        else 0.0
        for r in rows])
    graded = [r for r in rows if r.get('correctness') is not None]
    correctness = (
        st.mean([(r['correctness'] / 10.0) if r['admissible'] else 0.0
                 for r in graded]) if graded else None)

    parts = {'evidence': evidence, 'completeness': completeness}
    if correctness is not None:
        parts['correctness'] = correctness
    weight = sum(WEIGHTS[k] for k in parts)
    score = 100.0 * sum(WEIGHTS[k] * v for k, v in parts.items()) / weight
    return {
        'n': n, 'score': score, 'provisional': correctness is None,
        'weight_measured': weight,
        'evidence': evidence, 'completeness': completeness,
        'correctness': correctness,
        'raw_correctness': (st.mean([r['correctness'] for r in graded]) / 10.0
                            if graded else None),
        # The judge's own blind spot, measured on this run: how differently it
        # scores answers that proved their chain against ones that showed
        # nothing. A small gap means correctness is reading memory.
        'corr_admissible': _mean_or_none(
            [r['correctness'] for r in graded if r['admissible']]),
        'corr_inadmissible': _mean_or_none(
            [r['correctness'] for r in graded if not r['admissible']]),
        'judge_gap': _gap(graded),
    }


def _structural_stats(rows) -> dict:
    if not rows:
        return {}
    res = [r['resolution'] for r in rows if r.get('resolution') is not None]
    con = [r['connectivity'] for r in rows if r.get('connectivity') is not None]
    return {
        'resolution': f'{st.mean(res):.0%}' if res else MISSING,
        'connectivity': f'{st.mean(con):.0%}' if con else MISSING,
        'fully_connected': (f"{sum(1 for x in con if x == 1.0)}/{len(con)}"
                            if con else MISSING),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bare', default=str(BARE))
    ap.add_argument('--ariadne', default=str(ARIADNE))
    ap.add_argument('--structural', default=str(STRUCTURAL))
    ap.add_argument('--hops', default='3', help='label only: hop bound used')
    args = ap.parse_args()

    bare = _chain_stats(_load(Path(args.bare)))
    ari = _chain_stats(_load(Path(args.ariadne)))
    _stru_raw = _load(Path(args.structural))
    if isinstance(_stru_raw, dict):
        hops = str(_stru_raw.get('max_hops', args.hops))
        _stru_raw = _stru_raw.get('rows')
    else:
        # legacy bare-list artifact: the bound was not recorded, so say so
        hops = '?'
    stru = _structural_stats(_stru_raw)
    reqs = _load(REQS) or []
    scored = sum(1 for r in reqs if not r['flag'])
    flagged = [r['id'] for r in reqs if r['flag']]

    def cell(d, k, why=MISSING):
        return d.get(k) or why

    pb, pa = _pillars(_load(Path(args.bare))), _pillars(_load(Path(args.ariadne)))

    W = 34

    def num(p, key, pct=True):
        if p is None or p.get(key) is None:
            return MISSING
        return f'{p[key] * 100:.0f}' if pct else f'{p[key]:.0f}'

    print(f'\n{"":<{W}} {"Ariadne":>16} {"bare LLM":>16}')
    print('=' * (W + 34))
    print(f'{"  SCORE  /100":<{W}} '
          f'{(f"{pa["score"]:.1f}" if pa else MISSING):>16} '
          f'{(f"{pb["score"]:.1f}" if pb else MISSING):>16}')
    for p, who in ((pa, 'Ariadne'), (pb, 'bare LLM')):
        if p and p['provisional']:
            print(f'    {who}: PROVISIONAL - correctness ungraded, scored over '
                  f'the {p["weight_measured"]:.0%} that is measured')
    print('=' * (W + 34))
    print(f'\nPILLARS  (weights: evidence {WEIGHTS["evidence"]:.0%}, '
          f'completeness {WEIGHTS["completeness"]:.0%}, '
          f'correctness {WEIGHTS["correctness"]:.0%})')
    for label, key in (('  evidence', 'evidence'),
                       ('  completeness', 'completeness'),
                       ('  correctness (earned)', 'correctness'),
                       ('  correctness (raw, for contrast)', 'raw_correctness')):
        print(f'{label:<{W}} {num(pa, key):>16} {num(pb, key):>16}')
    print('-' * (W + 34))

    print('\nGROUNDING GATE  (same questions, same fixed chain requirements)')
    ari_why = MISSING if not ari else None
    for label, key in (
        ('  admissible', 'admissible'),
        ('  chain evidence (mean)', 'chain_evidence'),
        ('  full-chain route', 'full_route'),
        ('  endpoint route', 'endpoint_route'),
        ('  answers with zero quotes', 'zero_quote'),
        ('  ends read, middle unnamed', 'ends_no_middle'),
    ):
        print(f'{label:<{W}} {cell(ari, key, ari_why or MISSING):>16} '
              f'{cell(bare, key):>16}')

    print('\nCORRECTNESS  (LLM judge, 0-10)')
    print(f'{"  mean":<{W}} {cell(ari, "correctness"):>16} '
          f'{cell(bare, "correctness"):>16}')

    print('\nSCIP GRAPH  (Ariadne only - a graph property, not a head-to-head)')
    for label, key in (('  symbol resolution', 'resolution'),
                       (f'  chain connectivity (<={hops} hops)', 'connectivity'),
                       ('  fully connected', 'fully_connected')):
        print(f'{label:<{W}} {cell(stru, key):>16} {"n/a":>16}')

    print('\nDERIVATION  (bare arm only - Ariadne does not explore)')
    print(f'{"  answer-first":<{W}} {"n/a":>16} {cell(bare, "answer_first"):>16}')

    print('\nNOTES')
    print(f'  scored questions           : {scored}   '
          f'(flagged out: {flagged} - flagged pre-session, '
          f'reason never recorded)')
    if not ari:
        print('  Ariadne grounding          : NOT RUN. Needs '
              'ariadne_arm.py then score_chain.py.')
    # Computed, never hardcoded: this line asserts a fact about the run being
    # printed, and a figure carried over from an earlier run is the exact drift
    # this script exists to stop.
    for p_, who in ((pa, 'Ariadne'), (pb, 'bare LLM')):
        if p_ and p_.get('judge_gap') is not None:
            print(f'  judge blindness ({who:<8})  : evidenced '
                  f'{p_["corr_admissible"]:.2f} vs unevidenced '
                  f'{p_["corr_inadmissible"]:.2f} '
                  f'(gap {p_["judge_gap"]:.2f} of 10) - raw correctness '
                  f'{p_["raw_correctness"] * 100:.0f} against earned '
                  f'{p_["correctness"] * 100:.0f}')
    print('  arms are not symmetric     : the bare arm reads the corpus; '
          'Ariadne answers from docs')
    print('                               plus the SCIP index.')
    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
