"""The dependency-detection confirmation accepts only y/Y/n/N; any other
response is invalid and the prompt is shown again."""
from __future__ import annotations


def test_prompt_yes_no_accepts_yn_and_reprompts_on_invalid(monkeypatch):
    from cli import generate

    # Invalid responses ('', 'yes', 'maybe') are re-prompted; the first
    # valid y/Y/n/N decides. Here three invalids then 'Y' → True.
    seq = iter(['', 'yes', 'maybe', 'Y'])
    monkeypatch.setattr(generate.console, 'input', lambda *a, **k: next(seq))
    assert generate._prompt_yes_no('ok?') is True

    for valid, expected in [('y', True), ('Y', True), ('n', False), ('N', False)]:
        monkeypatch.setattr(generate.console, 'input', lambda *a, _v=valid, **k: _v)
        assert generate._prompt_yes_no('ok?') is expected
