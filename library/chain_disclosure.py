"""Saying, in words, that a chain forks wider than it can usefully show.

A dispatch with hundreds of implementations has hundreds of answers. Printing them is not
an answer, and silently keeping ten is the cap this project rejects. So the count is
stated, the measured shape travels with it, and the caller is asked to narrow.

Two rules hold every phrasing together:

* **The facts are not optional.** Every rendering carries the count, the symbol the chain
  forked at, the packages the implementations sit in, and a real question. :func:`verify_bank`
  runs at import so a fragment that drops one cannot ship.
* **The count is what the index can see.** Modules of a corpus can sit on disk and never
  reach the graph -- 615 code files measured on the live databricks corpus -- and
  implementations can live in code the corpus does not contain at all. Every observation
  says "index" for that reason: it is a floor, never a census.

The wording varies because one sentence returned verbatim on every wide dispatch reads like
a machine and invites being skipped. It varies **by content**, not by chance: each fragment
is chosen from a CRC of the symbol and the count, so one fan-out always reads the same way
while different ones read differently. Randomness would make eval diffs noise, break any
assertion on the text, and change a cached response between runs -- and ``hash()`` is salted
per process, so it cannot be used here either.
"""
from __future__ import annotations

import zlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from library.structural_assembly import FanOut

#: How wide the fork is. Each says "index", because that is the limit of what we know.
OBSERVATIONS = (
    'The chain forks at {symbol}: the index holds {count} implementations of it.',
    'This is where one route stops — {symbol} is implemented {count} times in this index.',
    'Following {symbol} means following {count} implementations, as far as this index sees.',
    '{symbol} is implemented {count} times in what is indexed here, so "which one runs" '
    'has {count} answers.',
    'The index carries {count} implementations of {symbol}, and any of them could be what '
    'runs.',
    'At {symbol} the chain stops being a single path: {count} implementations of it are '
    'indexed.',
    'Dispatch through {symbol} reaches {count} implementations that this index knows about.',
    'The walk arrived at {symbol}, which the index shows implemented {count} times.',
)

#: Where they sit — the dimension a caller can name back.
SHAPES = (
    'They sit in {packages}, and {tests} are test code.',
    'By package: {packages}. {tests} of them are tests.',
    'The spread is {packages} — {tests} under test paths.',
    'Location: {packages}, including {tests} in tests.',
)

#: Nothing was dropped, and the caller should know that before being asked to narrow.
OFFERS = (
    'I can list all {count} if you want them.',
    'Naming every one is possible, though it is more list than answer.',
    'Nothing is hidden here — say so and I will enumerate them.',
    'I can print the full set, at the cost of a very long reply.',
    'All of them are available; none has been discarded.',
    'I can walk every one if that is genuinely what you need.',
)

#: The ask. Each ends in a question mark, because a dead end is not an answer.
ASKS = (
    'Which area are you asking about?',
    'Can you say which package or module you mean?',
    'Which of these is closest to what you are working on?',
    'Point me at a package or a caller and I will trace that route properly?',
    'What is the calling context you have in mind?',
    'If you can name the concrete type involved, shall I follow just that one?',
    'Which one should I trace?',
    'Do you know roughly where this runs?',
)


def verify_bank() -> None:
    """Every fragment carries its facts. Runs at import so a bad one cannot ship.

    The failure this prevents is specific and has happened here before: a shared template
    gains a phrasing that quietly omits a required slot, and the caller receives fluent
    prose with the number missing.
    """
    for fragment in OBSERVATIONS:
        for slot in ('{symbol}', '{count}'):
            if slot not in fragment:
                raise ValueError(f'observation missing {slot}: {fragment!r}')
        if 'index' not in fragment:
            raise ValueError(f'observation overclaims what is known: {fragment!r}')
    for fragment in SHAPES:
        for slot in ('{packages}', '{tests}'):
            if slot not in fragment:
                raise ValueError(f'shape missing {slot}: {fragment!r}')
    for fragment in OFFERS:
        lowered = fragment.lower()
        if not any(phrase in lowered for phrase in
                   ('{count}', 'them', 'every one', 'full set', 'all of')):
            raise ValueError(f'offer says nothing about the whole set: {fragment!r}')
    for fragment in ASKS:
        if not fragment.rstrip().endswith('?'):
            raise ValueError(f'ask is not a question: {fragment!r}')


verify_bank()


#: How many areas the shape names before it counts the remainder.
#:
#: Three is enough to see where the weight sits. The live store's widest fan-out
#: (``UnaryLike.withNewChildInternal``, 529 implementations) spreads across 38 packages, and
#: naming them all produced a ~1,800 character sentence — the enumeration this module exists
#: to replace, one level down.
SHAPE_AREAS = 3


def _packages_phrase(by_package: tuple[tuple[str, int], ...]) -> str:
    """The widest areas by name, and the rest as a count. Nothing is silently dropped."""
    shown = by_package[:SHAPE_AREAS]
    phrase = ', '.join(f'{name} ({count})' for name, count in shown)
    remaining = len(by_package) - len(shown)
    if remaining > 0:
        phrase += f', and {remaining} more packages'
    return phrase


def _pick(options: tuple[str, ...], key: str, salt: str = '') -> str:
    """One fragment, chosen from the content so the choice is reproducible.

    ``zlib.crc32`` and not ``hash()``: the latter is salted per process, which would make
    the same fan-out read differently between runs and defeat every assertion on the text.
    """
    digest = zlib.crc32(f'{salt}:{key}'.encode() if salt else key.encode())
    return options[digest % len(options)]


def describe_fan_out(fan_out: 'FanOut') -> str:
    """A wide dispatch, described: how wide, where, what is on offer, and the question.

    The four parts are each chosen from the content, so a given fork always reads the same
    way and two different forks do not read alike.
    """
    key = f'{fan_out.qualified_name}:{fan_out.implementations}'
    slots = {
        'symbol': fan_out.qualified_name,
        'count': fan_out.implementations,
        'packages': _packages_phrase(fan_out.by_package),
        'tests': fan_out.tests,
    }
    return ' '.join(
        fragment.format(**slots) for fragment in (
            _pick(OBSERVATIONS, key),
            _pick(SHAPES, key, 'shape'),
            _pick(OFFERS, key, 'offer'),
            _pick(ASKS, key, 'ask'),
        )
    )
