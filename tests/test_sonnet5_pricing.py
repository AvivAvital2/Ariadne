"""claude-sonnet-5 is introductory-priced through 2026-08-31; the standard rate
(3.00, 15.00) applies afterward. The switch must be by date, not a comment, so a
cost estimate run after the cutoff can't silently under-report.
"""
from __future__ import annotations

from datetime import date

from docgen.pricing import LLM_PRICING, _sonnet_5_price


def test_sonnet_5_price_switches_at_intro_deadline():
    assert _sonnet_5_price(date(2026, 1, 1)) == (2.00, 10.00)    # intro window
    assert _sonnet_5_price(date(2026, 8, 31)) == (2.00, 10.00)   # last intro day
    assert _sonnet_5_price(date(2026, 9, 1)) == (3.00, 15.00)    # standard after
    assert _sonnet_5_price(date(2027, 1, 1)) == (3.00, 15.00)


def test_llm_pricing_table_carries_a_resolved_sonnet_5_rate():
    # The static table resolves to one of the two real rates (not a comment).
    assert LLM_PRICING['claude-sonnet-5'] in {(2.00, 10.00), (3.00, 15.00)}
