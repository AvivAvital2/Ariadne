"""How a required symbol is matched to a chain — defined once, so it cannot drift.

This module exists because it drifted. Both measurement scripts had their own copy of a
containment test, and containment credited symbols that were not there: ``DataSource`` matched
``DataSourceV2Relation``; ``MergeIntoTable`` matched 117 protobuf builder lines instead of the
Catalyst plan a question asked about. Two false credits were the entire apparent gain of a
prompt variant, which shipped on that basis and had to be reverted.

Every recall figure in ``evaluation/answer-path/`` now comes from here.
"""
from __future__ import annotations

from collections import defaultdict, deque


def hits(wanted, names) -> set:
    """Required symbols present among ``names``, matched at dot-segment boundaries.

    ``DataSource`` matches ``...datasources.DataSource`` and ``...DataSource.resolveRelation``;
    it does not match ``...DataSourceV2Relation``. Stricter than the eval harness's own
    containment rule, deliberately: this one cannot inflate.
    """
    matched = set()
    for symbol in wanted:
        for name in names:
            if symbol in name.split('.'):
                matched.add(symbol)
                break
    return matched


def ambiguous(wanted, names) -> dict:
    """Required symbols whose matches span several packages — reported, never guessed.

    The answer key records a repo, and both ``...plans.logical.MergeIntoTable`` and the
    generated ``...connect.proto.MergeIntoTable`` live in ``spark``, so the key cannot say
    which was meant. Each is returned with the packages it matched.
    """
    spread: dict = {}
    for symbol in wanted:
        packages = {name.split(f'.{symbol}')[0] for name in names
                    if symbol in name.split('.') and f'.{symbol}' in name}
        if len(packages) > 1:
            spread[symbol] = sorted(packages)[:4]
    return spread


def parents_and_children(citations):
    """The chain as a graph, from the parent each hop recorded when the walk followed it."""
    parents, children = defaultdict(set), defaultdict(list)
    for citation in citations:
        if citation.parent_qualified_name:
            parents[citation.qualified_name].add(citation.parent_qualified_name)
            children[citation.parent_qualified_name].append(citation)
    return parents, children


def spine_of(citations) -> set:
    """Ancestors of the chain's ``leaf`` hops — the candidate always-travelling structure."""
    parents, _ = parents_and_children(citations)
    kept, frontier = set(), deque(
        {c.qualified_name for c in citations if c.stop_reason == 'leaf'})
    while frontier:
        name = frontier.popleft()
        if name in kept:
            continue
        kept.add(name)
        frontier.extend(parents.get(name, ()))
    return kept


def rule_sets(citations) -> dict:
    """Each candidate selection rule's kept symbol set, for one chain.

    * ``spine``  — ancestors of leaf terminals
    * ``G``      — spine plus the non-plumbing **calls** those bodies make
    * ``H``      — spine plus every non-plumbing child, references included
    * ``chain``  — every distinct symbol the walk reached (what an unpruned menu offers)
    """
    _, children = parents_and_children(citations)
    spine = spine_of(citations)
    ring = lambda admit: {c.qualified_name for parent in spine
                          for c in children.get(parent, []) if admit(c)}
    return {
        'spine': spine,
        'G': spine | ring(lambda c: c.stop_reason != 'plumbing' and c.relation == 'calls'),
        'H': spine | ring(lambda c: c.stop_reason != 'plumbing'),
        'chain': {c.qualified_name for c in citations},
    }
def coverage(per_question, reached) -> int:
    """Required **slots** covered, summed per question — the aggregation every figure uses.

    ``hits`` returns a set, so a caller that flattens all questions' requirements into one
    list and calls ``hits`` once gets *distinct symbol names* instead: a symbol three
    questions require counts once rather than three times. Both are meaningful, neither is
    comparable to the other, and mixing them produced 68 and 76 for the same data in one
    session. So the aggregation lives here, beside the matcher, for the same reason the
    matcher itself does.
    """
    return sum(len(hits(symbols, reached)) for symbols in per_question.values())


def slot_index(per_question) -> dict:
    """``symbol -> {(question id, symbol)}``, so a big name set can be scored in one pass.

    Greedy selection scores hundreds of candidate sets and :func:`coverage` costs
    O(required x reached) per call, which is why a 155-candidate greedy pass had to be killed.
    Inverting makes each score O(reached).
    """
    index: dict = {}
    for qid, symbols in per_question.items():
        for symbol in symbols:
            index.setdefault(symbol, set()).add((qid, symbol))
    return index


def slots_covered(index, reached) -> set:
    """The ``(question, symbol)`` slots ``reached`` satisfies, matched at dot boundaries."""
    out: set = set()
    for name in reached:
        for segment in name.split('.'):
            found = index.get(segment)
            if found:
                out |= found
    return out
