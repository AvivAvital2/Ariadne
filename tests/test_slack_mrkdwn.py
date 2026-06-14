"""Markdown → Slack mrkdwn conversion for bot replies.

The agent emits GitHub-flavored Markdown, but Slack renders *mrkdwn*: bold is
``*one asterisk*`` (not ``**two**``), links are ``<url|label>`` (not
``[label](url)``), and there are no ``#`` headings. ``to_mrkdwn`` fixes the
constructs Slack mangles while leaving code spans/blocks — and already-correct
mrkdwn — otherwise untouched. It also escapes the three characters Slack treats
specially (``&``, ``<``, ``>``) everywhere, including inside code, since an
unescaped ``<host>`` makes Slack mis-parse the surrounding span.
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


def test_slack_special_chars_escaped():
    """Slack mrkdwn requires ``&``/``<``/``>`` as HTML entities, everywhere.

    Regression: a URL placeholder in a code span — ``https://<resource>...`` —
    arrived with raw ``<``/``>``. Slack then broke the span, keeping
    ``https://<resource>`` as a code chip and autolinking the ``.azure.com/``
    tail. Escaping the angle brackets keeps the whole URL inside one span.
    """
    # the screenshot case: angle brackets inside a code span are escaped,
    # and the span survives as a single unit (no split, no stray autolink)
    assert (
        to_mrkdwn('`https://<resource>.openai.azure.com/`')
        == '`https://&lt;resource&gt;.openai.azure.com/`'
    )
    # the same three characters are escaped in plain prose too
    assert to_mrkdwn('compare a < b && c > d') == 'compare a &lt; b &amp;&amp; c &gt; d'
    # escaping runs before link conversion: a query-string ``&`` becomes
    # ``&amp;`` inside the Slack link, and the ``<url|label>`` markup we emit
    # is itself never re-escaped
    assert to_mrkdwn('[docs](https://x.com?a=1&b=2)') == '<https://x.com?a=1&amp;b=2|docs>'
