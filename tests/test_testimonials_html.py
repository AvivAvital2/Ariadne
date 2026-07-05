"""Tests for the HTML showcase renderer — a self-contained, presentation page.

Pins the Slack-mrkdwn→HTML conversion (safe, no injection), the self-contained
page (inline CSS + base64 diagrams, no external files), and the per-card content.

Also locks in the presentation fixes seen in a real report:
- code fences carry a language tag (```python) that must NOT leak into the block;
- code blocks must never be nested inside <p> (invalid HTML);
- the question resolves ``` fences and keeps line breaks, but stays verbatim
  otherwise (no bold/italic/list transforms) so pasted logs aren't mangled;
- very long questions collapse into <details>;
- a DOT/graphviz block whose diagram was rendered to an image shows the image,
  not the DOT source — unless there is no image, when the source is kept.
"""
from __future__ import annotations

from pathlib import Path

import testimonials_html as th
from testimonials import Testimonial
from testimonials_html import _mrkdwn_to_html, _score_color, render_html


def _t(path: Path, **kw) -> Testimonial:
    base = dict(score=9, duration_seconds=2.0, asked_at='2026-06-14T14:30:05Z',
                question='Q?', answer='A.', permalink=None, source_ts=None,
                images=(), path=path)
    base.update(kw)
    return Testimonial(**base)


def test_mrkdwn_to_html_converts_slack_formatting():
    assert '<strong>b</strong>' in _mrkdwn_to_html('*b*')
    assert '<em>i</em>' in _mrkdwn_to_html('_i_')
    assert '<code>c</code>' in _mrkdwn_to_html('`c`')
    assert '<a href="https://x">lbl</a>' in _mrkdwn_to_html('<https://x|lbl>')
    assert '<a href="https://x">https://x</a>' in _mrkdwn_to_html('<https://x>')
    # a file path's underscores must NOT become italics
    assert '<em>' not in _mrkdwn_to_html('see slack_bridge/scan.py')
    # a raw angle-bracket tag in prose is escaped, never injected
    assert '<script>' not in _mrkdwn_to_html('a <script> tag')
    # already-escaped Slack entities are not double-escaped
    assert '&amp;lt;' not in _mrkdwn_to_html('a &lt; b')


def test_mrkdwn_to_html_code_block_and_bullets():
    html = _mrkdwn_to_html('```\nx = 1\n```')
    assert '<pre>' in html and 'x = 1' in html
    assert _mrkdwn_to_html('- one\n- two').count('<li>') == 2


def test_score_color_bands_differ():
    assert _score_color(10) == _score_color(8)          # high band
    assert len({_score_color(9), _score_color(6), _score_color(3)}) == 3   # high/mid/low distinct


def test_render_html_is_a_self_contained_presentation_page(tmp_path):
    img = tmp_path / 'd.png'
    img.write_bytes(b'\x89PNG-bytes')
    cards = [
        _t(tmp_path, question='How does caching work?',
           answer='It uses an *LRU* in `cache.py`.\nSee the doc.',
           score=9, permalink='https://acme.slack.com/p1', images=(img,)),
        _t(tmp_path, question='mid', answer='plain', score=6),     # other colour bands
        _t(tmp_path, question='low', answer='plain', score=3),
    ]
    html = render_html(cards)

    assert html.startswith('<!DOCTYPE html>') and '</html>' in html
    assert '<style>' in html                              # inline CSS → self-contained
    assert 'How does caching work?' in html
    assert '<strong>LRU</strong>' in html and '<code>cache.py</code>' in html
    assert 'data:image/png;base64,' in html              # diagram embedded inline
    assert 'https://acme.slack.com/p1' in html           # link back to Slack
    assert '9/10' in html                                # score badge
    assert 'file' in html                                # 'cache.py' → "1 file cited" chip


def test_render_html_empty_and_no_diagram(tmp_path):
    assert 'No testimonials' in render_html([])           # empty → friendly page
    html = render_html([_t(tmp_path, question='Q', answer='plain text', images=())])
    assert 'data:image' not in html                      # no diagram → no <img>
    assert 'View in Slack' not in html                   # no permalink → no link


# --- presentation fixes (report.html review) ----------------------------------

def test_answer_code_fence_strips_language():
    out = _mrkdwn_to_html('```python\nprint(1)\n```')
    assert '<pre><code>print(1)</code></pre>' in out
    assert 'python' not in out            # language token must not leak into the block


def test_answer_never_nests_pre_in_p():
    for src in ('intro\n\n```\ncode\n```\n\nmore', 'intro\n```\ncode\n```\nmore'):
        out = _mrkdwn_to_html(src)
        assert '<p><pre' not in out
        assert '<pre><code>code</code></pre>' in out


def test_question_resolves_fences_but_not_emphasis():
    out = th._question_to_html('note *foo*\n```\nls -la\n```')
    assert '<pre><code>ls -la</code></pre>' in out
    assert '<strong>' not in out and '<em>' not in out   # no emphasis in questions


def test_question_preserves_line_breaks():
    assert 'line one<br>line two' in th._question_to_html('line one\nline two')


def test_short_question_is_a_plain_block():
    out = th._question_block('How does X work?')
    assert '<div class="q">' in out and '<details' not in out
    assert 'How does X work?' in out


def test_long_question_collapses_into_details():
    out = th._question_block('ERROR: ' + 'stack frame line\n' * 30)   # long + many lines
    assert '<details class="q">' in out and '<summary>' in out


def test_rendered_diagram_dropped_when_image_present():
    src = 'Here is the flow:\n```dot\ndigraph{a->b}\n```\nDone.'
    assert 'digraph' not in th._strip_rendered_diagrams(src, has_images=True)
    assert 'diagram rendered below' in th._strip_rendered_diagrams(src, has_images=True)
    assert 'digraph' in th._strip_rendered_diagrams(src, has_images=False)   # no image → keep


def test_render_html_shows_image_not_dot_source(tmp_path):
    png = tmp_path / 'd.png'
    png.write_bytes(b'\x89PNG fake-bytes')
    t = _t(tmp_path, question='Show me a diagram',
           answer='The flow:\n```dot\ndigraph{a->b}\n```\nThat covers it.',
           images=(png,))
    out = render_html([t])
    assert 'digraph' not in out                           # DOT source not shown
    assert 'data:image/png;base64,' in out               # image embedded instead
    assert 'diagram rendered below' in out
