"""The dependency-detection confirmation accepts only y/Y/n/N; any other
response is invalid and the prompt is shown again."""
from __future__ import annotations

from unittest.mock import patch

from cli.generate_cost import _prompt_yes_no, console


def test_prompt_yes_no_accepts_yn_and_reprompts_on_invalid():
    # Invalid responses ('', 'yes', 'maybe') are re-prompted; the first
    # valid y/Y/n/N decides. Here three invalids then 'Y' → True. We patch
    # the console object's ``input`` directly (imported, not via a module
    # path) so the test doesn't depend on where the function lives.
    with patch.object(console, 'input', side_effect=['', 'yes', 'maybe', 'Y']):
        assert _prompt_yes_no('ok?') is True

    for valid, expected in [('y', True), ('Y', True), ('n', False), ('N', False)]:
        with patch.object(console, 'input', return_value=valid):
            assert _prompt_yes_no('ok?') is expected
