"""Guardrail: the ``ariadne usage --days`` CLI contract.

``--days`` is optional with a default of 30. It also accepts a *bare* ``--days``
(the flag named with no value) as shorthand for that default, so naming the flag
without a number resolves to 30 rather than raising an argparse error — while
``--days N`` still selects N.
"""
from __future__ import annotations

from cli.main import create_parser


def test_usage_days_flag_contract():
    parser = create_parser()

    # Omitted entirely → the default window.
    assert parser.parse_args(['usage']).days == 30

    # Explicit value (long and short) → that value.
    assert parser.parse_args(['usage', '--days', '7']).days == 7
    assert parser.parse_args(['usage', '-d', '7']).days == 7

    # Bare flag, no value → resolves to the default (not an argparse error).
    assert parser.parse_args(['usage', '--days']).days == 30
    assert parser.parse_args(['usage', '-d']).days == 30
