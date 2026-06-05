"""Markdown → Slack mrkdwn conversion for bot replies.

The agent emits GitHub-flavored Markdown, but Slack renders *mrkdwn*: bold is
``*one asterisk*`` (not ``**two**``), links are ``<url|label>`` (not
``[label](url)``), and there are no ``#`` headings. ``to_mrkdwn`` fixes the
constructs Slack mangles while leaving code spans/blocks — and already-correct
mrkdwn — untouched.
"""
from __future__ import annotations

from slack_bridge.format import to_mrkdwn


def test_bold_double_asterisk_becomes_single():
    assert to_mrkdwn('**weather.com**') == '*weather.com*'
    assert to_mrkdwn('a __b__ c') == 'a *b* c'


def test_links_become_slack_format():
    assert to_mrkdwn('[docs](https://x.com)') == '<https://x.com|docs>'


def test_headings_become_bold():
    assert to_mrkdwn('### Title') == '*Title*'
    assert to_mrkdwn('## Big\nbody') == '*Big*\nbody'


def test_code_is_left_untouched():
    # inline code: literal ** must survive
    assert to_mrkdwn('use `**kwargs` here') == 'use `**kwargs` here'
    # fenced block: nothing inside is converted
    src = '```\n**x** and [a](b)\n```'
    assert to_mrkdwn(src) == src


def test_already_correct_mrkdwn_preserved():
    assert to_mrkdwn('*already bold*') == '*already bold*'   # single-asterisk stays
    assert to_mrkdwn('_italic_') == '_italic_'                # underscore italic stays
    assert to_mrkdwn('- bullet one\n- bullet two') == '- bullet one\n- bullet two'


def test_realistic_mixed_reply():
    src = (
        'For weather try **weather.com** or [Google](https://google.com).\n'
        '- option one\n'
        'Run `pip install **x**` to test.'
    )
    out = to_mrkdwn(src)
    assert '*weather.com*' in out
    assert '<https://google.com|Google>' in out
    assert '- option one' in out
    assert '`pip install **x**`' in out      # code span untouched
    assert '**weather.com**' not in out
