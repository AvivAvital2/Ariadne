"""Render the best-of testimonials as a self-contained, presentation-ready page.

One HTML file: inline CSS, diagrams embedded as base64 — so it opens, shares, or
screenshots into a deck with no external assets. Each Q&A is a clean slide-style
card: the question as the headline (collapsed into a <details> when it's a long
pasted log/stack trace), the answer rendered from Slack mrkdwn, the diagram
inline, a score + richness badge, and a link back to the Slack thread.

A DOT/graphviz block whose diagram was rendered to an image (``t.images``) is
replaced by a short marker so the picture shows instead of the raw DOT source;
if nothing rendered, the source is kept rather than silently dropped.
"""
from __future__ import annotations

import base64
import re
from collections.abc import Sequence

from diagram_format import DOT_BLOCK_RE
from testimonials import Testimonial, _count_file_refs

_ENTITY_RE = re.compile(r'&(?!(?:amp|lt|gt|quot|apos|#\d+);)')
_BULLET_RE = re.compile(r'^\s*(?:[-*]|\d+\.)\s+')
_CODE_FENCE_RE = re.compile(r'```(.*?)```', re.DOTALL)
_INFO_STRING_RE = re.compile(r'[ \t]*[A-Za-z][\w.+-]*[ \t]*')   # a bare ```lang tag line
_PLACEHOLDER_RE = re.compile(r'\x00\d+\x00')
# Mirrors slack_bridge.diagram.prepare_diagrams — the text a rendered diagram
# leaves behind in place of its DOT source.
_DIAGRAM_MARKER = '_📊 (diagram rendered below)_'
_LONG_Q_CHARS = 240      # a question past this length (or line count) collapses
_LONG_Q_LINES = 4
_PREVIEW_CHARS = 100     # collapsed-question summary preview length


def _escape(text: str) -> str:
    """Escape raw ``&``, ``<``, ``>`` without double-escaping existing entities.

    Slack message text already escapes literals (``&lt;``); raw agent text may
    not. This keeps both safe without turning ``&lt;`` into ``&amp;lt;``.
    """
    return _ENTITY_RE.sub('&amp;', text).replace('<', '&lt;').replace('>', '&gt;')


def _escape_attr(url: str) -> str:
    return _escape(url).replace('"', '&quot;')


def _code_block_html(inner: str) -> str:
    """Render the text between a ``` fence pair as ``<pre><code>``.

    Drops a leading language info-string line (```python → ``python``) but never a
    real first line of code — an info string is a *bare* token with no spaces, so
    a first line like ``INFO  Worker …`` or ``key = value`` is kept.
    """
    if '\n' in inner:
        head, tail = inner.split('\n', 1)
        if _INFO_STRING_RE.fullmatch(head):
            inner = tail
    return f'<pre><code>{_escape(inner.strip(chr(10)))}</code></pre>'


def _stash_code(text: str, keep) -> str:
    """Stash ``` fenced blocks then `inline` code as placeholders (so nothing
    formats inside them). Shared by the answer and question paths."""
    text = _CODE_FENCE_RE.sub(lambda m: keep(_code_block_html(m.group(1))), text)
    return re.sub(r'`([^`]+)`', lambda m: keep(f'<code>{_escape(m.group(1))}</code>'), text)


def _blocks_to_html(text: str, *, lists: bool = True) -> str:
    """Group ``\\n\\n``-separated blocks into <ul> (all-bullet, when ``lists``) or
    <p> (lines joined with <br>).

    A code-block placeholder is emitted at top level, never wrapped in <p> — that
    would be invalid ``<p><pre>…</pre></p>`` — even when it sits mid-paragraph.
    """
    out: list[str] = []
    for block in re.split(r'\n{2,}', text.strip()):
        lines = block.split('\n')
        if lists and lines and all(_BULLET_RE.match(ln) for ln in lines):
            out.append('<ul>' + ''.join(
                f'<li>{_BULLET_RE.sub("", ln)}</li>' for ln in lines) + '</ul>')
            continue
        run: list[str] = []
        for ln in lines:
            if _PLACEHOLDER_RE.fullmatch(ln.strip()):
                if run:
                    out.append('<p>' + '<br>'.join(run) + '</p>')
                    run = []
                out.append(ln.strip())          # code block → top-level, not in <p>
            else:
                run.append(ln)
        if run:
            out.append('<p>' + '<br>'.join(run) + '</p>')
    return ''.join(out)


def _unstash(text: str, stash: list[str]) -> str:
    for i, rendered in enumerate(stash):
        text = text.replace(f'\x00{i}\x00', rendered)
    return text


def _mrkdwn_to_html(text: str) -> str:
    """Convert Slack mrkdwn to safe HTML (code, links, bold/italic, lists)."""
    stash: list[str] = []

    def keep(rendered: str) -> str:
        stash.append(rendered)
        return f'\x00{len(stash) - 1}\x00'

    # Code first, so nothing formats inside it.
    text = _stash_code(text, keep)
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
    return _unstash(text, stash)


def _question_to_html(text: str) -> str:
    """Render a question: resolve ``` fences and `inline` code and keep line
    breaks, but apply NO emphasis/link/list transforms — a question is often a
    pasted log or stack trace that must stay verbatim (no ``*x*`` → bold, no
    ``1.`` → list item)."""
    stash: list[str] = []

    def keep(rendered: str) -> str:
        stash.append(rendered)
        return f'\x00{len(stash) - 1}\x00'

    text = _stash_code(text, keep)
    text = _escape(text)
    text = _blocks_to_html(text, lists=False)
    return _unstash(text, stash)


def _question_preview(question: str) -> str:
    """A one-line, plain-text, escaped preview for a collapsed long question."""
    first = next((ln.strip() for ln in question.splitlines() if ln.strip()), '')
    first = first.lstrip('`').strip()
    if len(first) > _PREVIEW_CHARS:
        first = first[:_PREVIEW_CHARS].rstrip() + '…'
    return _escape(first) or 'Question'


def _question_block(question: str) -> str:
    """The question as a heading-styled block; collapsed into <details> when it's
    long (many chars or many lines) so a pasted stack trace can't dominate the
    card. Short questions render inline as a plain block."""
    rendered = _question_to_html(question)
    is_long = len(question) > _LONG_Q_CHARS or question.count('\n') + 1 > _LONG_Q_LINES
    if is_long:
        return (f'<details class="q"><summary>{_question_preview(question)}</summary>'
                f'<div class="q-body">{rendered}</div></details>')
    return f'<div class="q">{rendered}</div>'


def _strip_rendered_diagrams(answer: str, *, has_images: bool) -> str:
    """Replace DOT/graphviz blocks with the 'rendered below' marker when the card
    has images (the diagram is shown as a picture, so its source shouldn't also
    appear as a code block). With no images, keep the source so a diagram that
    never rendered isn't silently lost."""
    if not has_images:
        return answer
    return DOT_BLOCK_RE.sub(_DIAGRAM_MARKER, answer)


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
    answer_html = _mrkdwn_to_html(
        _strip_rendered_diagrams(t.answer, has_images=bool(t.images)))
    return (
        '<section class="card">'
        '<div class="meta">'
        f'<span class="rank">#{rank}</span>'
        f'<span class="score" style="background:{_score_color(t.score)}">{t.score}/10</span>'
        f'{chips}'
        '</div>'
        f'{_question_block(t.question)}'
        f'<div class="a">{answer_html}</div>'
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
.q{margin:0 0 18px}
.q>p,.q>summary{font-size:23px;font-weight:700;line-height:1.32;color:#0b1620;margin:0}
.q>p:not(:last-child){margin-bottom:8px}
.q>summary{cursor:pointer;list-style:none;outline:none}
.q>summary::-webkit-details-marker{display:none}
.q>summary::before{content:"\\25B8  ";color:#9aa7b4;font-weight:800}
details[open].q>summary::before{content:"\\25BE  "}
.q-body{margin-top:14px;font-size:15px;font-weight:400;color:#33465a;
  border-left:3px solid #e3e9f0;padding-left:16px}
.q-body p{margin:0 0 10px}.q-body p:last-child{margin-bottom:0}
.a{font-size:16px;color:#27384a}
.a p{margin:0 0 14px}.a p:last-child{margin-bottom:0}
.a ul{margin:0 0 14px;padding-left:22px}.a li{margin:4px 0}
.a code,.q code,.q-body code{background:#eef2f7;border-radius:6px;padding:2px 6px;font-size:.9em;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.a pre,.q pre,.q-body pre{background:#0e1726;color:#e6edf6;border-radius:12px;padding:18px 20px;
  overflow:auto;font-size:13.5px;margin:0 0 14px;font-weight:400;line-height:1.5}
.a pre code,.q pre code,.q-body pre code{background:none;padding:0;color:inherit;font-size:inherit}
.a a{color:#2563eb;text-decoration:none}.a a:hover{text-decoration:underline}
.diagram{margin:26px 0 0;text-align:center}
.diagram img{max-width:100%;border-radius:12px;border:1px solid #e3e9f0;
  box-shadow:0 4px 18px rgba(16,40,70,.12)}
.src{margin-top:22px;font-size:13px}
.src a{color:#2563eb;text-decoration:none;font-weight:600}
.empty{color:#6b7785;font-size:16px}
"""
