"""Telling the caller a chain forks wider than it can usefully show, in words.

A dispatch with hundreds of implementations has hundreds of answers. Printing them is not
an answer, and silently keeping ten of them is the cap this project rejects. So the count
is stated, the measured shape travels with it, and the caller is asked to narrow.

**The count is what this index can see.** Whole modules of the live corpus were never
indexed — 615 code files measured present on disk and absent from the graph — and
implementations can also live in code the corpus does not contain at all. So the wording
says what it is: a floor, not a census.

The phrasing varies, because one sentence returned verbatim on every wide dispatch reads
like a machine and invites being ignored. It varies **by content**: the wording is chosen
from a stable digest of the symbol and the count, so a given fan-out always reads the same
way while different ones read differently. Chance would make eval diffs noise, break any
assertion on the text, and change a cached response between runs.

Synthetic fixtures only.
"""
from __future__ import annotations

import zlib

from library.chain_disclosure import OBSERVATIONS, describe_fan_out
from library.structural_assembly import FanOut


def _fan_out(qn='pkg.reader.Reader.read', n=30, packages=(('pkg.main', 20),
                                                          ('pkg.tests', 10))):
    return FanOut(qualified_name=qn, file='pkg/reader.py', line_start=7,
                  implementations=n, by_package=packages, tests=10)


def test_every_phrasing_states_the_count_names_the_symbol_and_asks_something():
    """The wording is free to vary; the facts are not.

    Every rendering must carry the number, the symbol it forked at, and an actual question
    -- otherwise the caller receives prose instead of something to act on.
    """
    for index in range(120):
        fan_out = _fan_out(qn=f'pkg.m{index}.Type{index}.method', n=11 + index)

        text = describe_fan_out(fan_out)

        assert str(11 + index) in text, f'count missing: {text}'
        assert f'Type{index}.method' in text, f'symbol missing: {text}'
        assert '?' in text, f'no question to answer: {text}'


def test_every_phrasing_says_the_count_is_what_the_index_can_see():
    """Not "there are 12" — the index is incomplete and the sentence must not overclaim."""
    for index in range(60):
        text = describe_fan_out(_fan_out(qn=f'pkg.q{index}.T{index}.m', n=11 + index))

        assert 'index' in text.lower(), f'the sentence overclaims: {text}'


def test_the_wording_is_chosen_by_content_not_by_chance():
    """Same fan-out, same sentence -- and not dependent on the process hash seed."""
    fan_out = _fan_out(qn='pkg.a.A.run', n=30)

    first = describe_fan_out(fan_out)

    assert first == describe_fan_out(fan_out)
    chosen = OBSERVATIONS[zlib.crc32(b'pkg.a.A.run:30') % len(OBSERVATIONS)]
    assert chosen.split('{')[0].strip()[:18] in first, (
        'the choice must come from a stable digest, not from hash() or random')


def test_different_fan_outs_do_not_all_read_alike():
    """The point of the bank: 200 wide dispatches must not return one sentence."""
    texts = {describe_fan_out(_fan_out(qn=f'pkg.p{i}.T{i}.m', n=11 + i))
             for i in range(200)}

    assert len(texts) >= 20, f'only {len(texts)} distinct phrasings'


def test_the_measured_shape_travels_so_the_caller_can_narrow():
    """"Can you share more information?" is a dead end unless it says how."""
    text = describe_fan_out(_fan_out())

    assert 'pkg.main' in text and 'pkg.tests' in text, (
        'the packages are the dimension the caller narrows by')
    assert '10' in text, 'the test/main split is measured, so it is stated'
def test_the_shape_names_a_few_areas_then_counts_the_rest():
    """38 packages in one sentence is the enumeration this module exists to replace.

    Measured on the live store: the widest fan-out (``UnaryLike.withNewChildInternal``, 529
    implementations) spreads across 38 packages, and listing them produced a ~1,800
    character sentence. Three areas show where the weight sits; the remainder is a number.
    """
    packages = tuple((f'org.example.p{index}', 40 - index) for index in range(38))

    text = describe_fan_out(_fan_out(n=529, packages=packages))

    assert 'org.example.p0' in text, 'the largest areas are named'
    assert 'org.example.p20' not in text, 'the list must not run to 38 entries'
    assert '35 more' in text, 'the rest is counted, not dropped in silence'
