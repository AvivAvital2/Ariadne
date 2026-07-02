"""The per-turn time policy, in one place.

Every knob of the slow-turn behaviour lives HERE — the defaults, the yaml
``pool:`` keys, the soft→extension derivation, and the "still working" notice
text. Nothing else in the bridge may hardcode a timeout number or recompute
the window arithmetic: handlers consume :class:`TurnBudget` fields verbatim.
(The policy used to be smeared across the config defaults, the ``from_env``
literals, the handler arithmetic and two docs — which is how a deployed yaml
kept the pre-notice 120s cap and silently zeroed the extension window.)
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from attrs import frozen

# The operator-facing yaml keys (under ``pool:``) — parsed ONLY by from_pool.
SOFT_TIMEOUT_KEY = 'soft_timeout_seconds'
TURN_TIMEOUT_KEY = 'turn_timeout_seconds'


@frozen
class TurnBudget:
    """When the notice posts (``soft_seconds``) and the hard cap (``total_seconds``).

    Validated at construction so the degenerate config that bit production is
    unrepresentable: the cap must exceed the soft point, otherwise the extension
    window the notice promises would be zero and every slow turn would die at
    the soft deadline right after saying "still working".
    """

    soft_seconds: float = 120.0
    total_seconds: float = 240.0

    def __attrs_post_init__(self) -> None:
        if self.soft_seconds <= 0:
            raise ValueError(
                f'pool.{SOFT_TIMEOUT_KEY} must be positive, got {self.soft_seconds:g}'
            )
        if self.total_seconds <= self.soft_seconds:
            raise ValueError(
                f'pool.{TURN_TIMEOUT_KEY} ({self.total_seconds:g}s) must exceed '
                f'pool.{SOFT_TIMEOUT_KEY} ({self.soft_seconds:g}s): the gap is the '
                'extension window the "still working" notice buys — at zero the '
                'bot gives up the instant it posts the notice.'
            )

    @property
    def extension_seconds(self) -> float:
        """The post-notice window; positive by construction."""
        return self.total_seconds - self.soft_seconds

    @classmethod
    def from_pool(cls, pool: Mapping[str, Any]) -> TurnBudget:
        """Parse the yaml ``pool:`` block; absent keys keep the class defaults."""
        kwargs: dict[str, float] = {}
        if SOFT_TIMEOUT_KEY in pool:
            kwargs['soft_seconds'] = float(pool[SOFT_TIMEOUT_KEY])
        if TURN_TIMEOUT_KEY in pool:
            kwargs['total_seconds'] = float(pool[TURN_TIMEOUT_KEY])
        return cls(**kwargs)


_SLOW_BEFORE = [
    "This is taking longer than I expected",
    "This one's a bit involved",
    "This is a meatier question than usual",
    "There's a fair bit to sift through here",
    "This is running a little long",
    "Bigger than it first looked",
    "This one needs some real digging",
    "Still piecing this together",
    "Lots of ground to cover here",
    "This is a deep one",
    "This is taking more time than usual",
    "There's a lot to work through",
    "This one's keeping me busy",
    "Turns out this is non-trivial",
    "Still chasing this down",
    "Following the thread deeper into the labyrinth",
    "This corner of the labyrinth is new to me",
    "The answer is playing hard to get",
    "I opened one doc and found three more",
    "Your question sent me on a proper quest",
]

_SLOW_AFTER = [
    "hang on, I'm still digging",
    "bear with me, I'm still on it",
    "give me a moment to finish",
    "still searching, almost there",
    "hang tight, nearly done",
    "I haven't forgotten you, still going",
    "stay with me, I'm getting there",
    "just need a little longer",
    "still pulling the pieces together",
    "won't be much longer",
    "let me keep at it",
    "I'm still on the case",
    "nearly there, thanks for waiting",
    "almost done now",
    "still crunching, hang on",
    "the exit is around here somewhere",
    "no Minotaur so far, which is nice",
    "the coffee is kicking in",
    "I've brought extra thread",
    "still beats reading the whole repo",
]


def slow_notice() -> str:
    """A varied 'still working' line for the soft-deadline notice, mixed locally
    from two phrase pools (no LLM) so the user sees the turn is still alive."""
    return f'⏳ {random.choice(_SLOW_BEFORE)} — {random.choice(_SLOW_AFTER)}.'
