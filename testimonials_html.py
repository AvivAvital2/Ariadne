"""Render the best-of testimonials as a self-contained, presentation-ready page.

One HTML file: inline CSS, diagrams embedded as base64 — so it opens, shares, or
screenshots into a deck with no external assets. Each Q&A is a clean slide-style
card: the question as the headline, the answer rendered from Slack mrkdwn, the
diagram inline, a score + richness badge, and a link back to the Slack thread.
"""
from __future__ import annotations

import base64
import re
from collections.abc import Sequence

from testimonials import Testimonial, _count_file_refs

_ENTITY_RE = re.compile(r'&(?!(?:amp|lt|gt|quot|apos|#\d+);)')
_BULLET_RE = re.compile(r'^\s*(?:[-*]|\d+\.)\s+')


def _escape(text: str) -> str:
    """Escape raw ``&``, ``<``, ``>`` without double-escaping existing entities.

    Slack message text already escapes literals (``&lt;``); raw agent text may
    not. This keeps both safe without turning ``&lt;`` into ``&amp;lt;``.
    """
    return _ENTITY_RE.sub('&amp;', text).replace('<', '&lt;').replace('>', '&gt;')


def _escape_attr(url: str) -> str:
    return _escape(url).replace('"', '&quot;')


def _blocks_to_html(text: str) -> str:
    """Group ``\\n\\n``-separated blocks into <ul> (all-bullet) or <p> (with <br>)."""
    out: list[str] = []
    for block in re.split(r'\n{2,}', text.strip()):
        lines = block.split('\n')
        if all(_BULLET_RE.match(ln) for ln in lines):
            out.append('<ul>' + ''.join(
                f'<li>{_BULLET_RE.sub("", ln)}</li>' for ln in lines) + '</ul>')
        else:
            out.append('<p>' + '<br>'.join(lines) + '</p>')
    return ''.join(out)


def _mrkdwn_to_html(text: str) -> str:
    """Convert Slack mrkdwn to safe HTML (code, links, bold/italic, lists)."""
    stash: list[str] = []

    def keep(rendered: str) -> str:
        stash.append(rendered)
        return f'\x00{len(stash) - 1}\x00'

    # Code first, so nothing formats inside it.
    text = re.sub(r'```\n?(.*?)\n?```',
                  lambda m: keep(f'<pre><code>{_escape(m.group(1))}</code></pre>'),
                  text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', lambda m: keep(f'<code>{_escape(m.group(1))}</code>'), text)
    # Slack links: <url|label> then bare <url>.
    text = re.sub(r'<([^|>\s]+)\|([^>]+)>',
                  lambda m: keep(f'<a href="{_escape_attr(m.group(1))}">{_escape(m.group(2))}</a>'),
                  text)
    text = re.sub(r'<(https?://[^>\s]+)>',
                  lambda m: keep(f'<a href="{_escape_attr(m.group(1))}">{_escape(m.group(1))}</a>'),
                  text)
    # Escape remaining prose, then inline emphasis (italics must not fire on
    # underscores inside identifiers like ``slack_bridge``).
    text = _escape(text)
    text = re.sub(r'\*([^*\n]+)\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\w)_([^_\n]+)_(?!\w)', r'<em>\1</em>', text)
    text = _blocks_to_html(text)
    for i, rendered in enumerate(stash):
        text = text.replace(f'\x00{i}\x00', rendered)
    return text


def _score_color(score: int) -> str:
    if score >= 8:
        return '#1f9d55'      # high — green
    if score >= 5:
        return '#c08a1e'      # mid — amber
    return '#6b7785'          # low — slate


def _card(t: Testimonial, rank: int) -> str:
    chips = ''
    if t.images:
        chips += f'<span class="chip">&#128202; {len(t.images)} diagram{"s" if len(t.images) > 1 else ""}</span>'
    n_files = _count_file_refs(t.answer)
    if n_files:
        chips += f'<span class="chip">{n_files} file{"s" if n_files > 1 else ""} cited</span>'
    diagrams = ''.join(
        f'<figure class="diagram"><img alt="diagram" '
        f'src="data:image/png;base64,{base64.b64encode(img.read_bytes()).decode("ascii")}"></figure>'
        for img in t.images)
    src = (f'<div class="src"><a href="{_escape_attr(t.permalink)}">View in Slack &#8599;</a></div>'
           if t.permalink else '')
    return (
        '<section class="card">'
        '<div class="meta">'
        f'<span class="rank">#{rank}</span>'
        f'<span class="score" style="background:{_score_color(t.score)}">{t.score}/10</span>'
        f'{chips}'
        '</div>'
        f'<h2 class="q">{_escape(t.question)}</h2>'
        f'<div class="a">{_mrkdwn_to_html(t.answer)}</div>'
        f'{diagrams}{src}'
        '</section>'
    )


def render_html(testimonials: Sequence[Testimonial], *, title: str = 'Ariadne · Best of') -> str:
    """A self-contained HTML showcase page for the given testimonials (best first)."""
    if testimonials:
        body = ''.join(_card(t, i) for i, t in enumerate(testimonials, 1))
    else:
        body = '<p class="empty">No testimonials recorded yet.</p>'
    safe_title = _escape(title)
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{safe_title}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n'
        f'<main class="page">\n<h1>{safe_title}</h1>\n{body}\n</main>\n</body>\n</html>\n'
    )


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#eaeef3;color:#16222e;
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.page{max-width:900px;margin:0 auto;padding:56px 24px}
.page>h1{font-size:30px;font-weight:800;letter-spacing:-.02em;margin:0 0 36px;color:#0b1620}
.card{background:#fff;border:1px solid #e3e9f0;border-radius:18px;
  box-shadow:0 10px 30px rgba(16,40,70,.08);padding:44px;margin:0 0 32px}
@media print{.card{break-inside:avoid;page-break-inside:avoid}}
.meta{display:flex;align-items:center;gap:10px;margin-bottom:22px;flex-wrap:wrap}
.rank{font-weight:800;font-size:13px;color:#9aa7b4}
.score{font-weight:800;font-size:13px;color:#fff;border-radius:999px;padding:4px 13px}
.chip{font-size:12px;color:#46586a;background:#eef2f7;border-radius:999px;padding:4px 11px}
.q{font-size:23px;font-weight:750;line-height:1.32;margin:0 0 18px;color:#0b1620}
.a{font-size:16px;color:#27384a}
.a p{margin:0 0 14px}.a p:last-child{margin-bottom:0}
.a ul{margin:0 0 14px;padding-left:22px}.a li{margin:4px 0}
.a code{background:#eef2f7;border-radius:6px;padding:2px 6px;font-size:.9em;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.a pre{background:#0e1726;color:#e6edf6;border-radius:12px;padding:18px 20px;overflow:auto;
  font-size:13.5px;margin:0 0 14px}
.a pre code{background:none;padding:0;color:inherit}
.a a{color:#2563eb;text-decoration:none}.a a:hover{text-decoration:underline}
.diagram{margin:26px 0 0;text-align:center}
.diagram img{max-width:100%;border-radius:12px;border:1px solid #e3e9f0;
  box-shadow:0 4px 18px rgba(16,40,70,.12)}
.src{margin-top:22px;font-size:13px}
.src a{color:#2563eb;text-decoration:none;font-weight:600}
.empty{color:#6b7785;font-size:16px}
"""
