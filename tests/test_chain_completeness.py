"""Completeness: an answer must show how A leads to B, not just land on B.

The original gate demanded a verified quote for EVERY hop, which scored "read
nothing and recited the architecture" identically to "read both endpoints and
knew the middle". Those are different epistemic states. So an answer is complete
either way:

* **full** -- every required hop carries a quote that matches its file at the
  cited line; or
* **endpoints** -- the first and last hop carry such quotes AND every
  intermediate mechanism is named, so the transition is articulated rather than
  jumped.

Both paths need at least two verified snippets, and on a cross-repo question the
two endpoint quotes must come from different repositories -- otherwise a chain
spanning delta and spark is satisfied by two quotes from one file.

The line numbers carry the load in the endpoint path. Naming intermediate
mechanisms is exactly what a model that memorised the corpus does well; citing
the line a symbol actually occupies *at the pinned revision* is what it cannot
do without reading. That is why a quote which does not match its cited line is
not evidence, and why the endpoint concession is safe.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    'score_chain',
    Path(__file__).resolve().parent.parent / 'evaluation' / 'spool-clean-room'
    / 'score_chain.py',
)
score_chain = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(score_chain)

WIDGET = """package alpha.core

class Widget(name: String) {
  def emit(): Unit = registry.publish(name, payload)
}
"""
MIDDLE = """package alpha.core

trait Middle {
  def relay(msg: Message): Unit = downstream.forward(msg)
}
"""
GADGET = """package beta.core

object Gadget {
  def consume(evt: Event): Result = transform(evt).persist()
}
"""

# Each entry: (basename, line to cite, the source line at that line)
SNIPPETS = {
    'Widget': ('Widget.scala', 4, 'def emit(): Unit = registry.publish(name, payload)'),
    'Middle': ('Middle.scala', 4, 'def relay(msg: Message): Unit = downstream.forward(msg)'),
    'Gadget': ('Gadget.scala', 4, 'def consume(evt: Event): Result = transform(evt).persist()'),
}
REPO_OF = {'Widget': 'alpha', 'Middle': 'alpha', 'Gadget': 'beta'}


@pytest.fixture
def corpus(tmp_path):
    for repo, name, body in (('alpha', 'Widget.scala', WIDGET),
                             ('alpha', 'Middle.scala', MIDDLE),
                             ('beta', 'Gadget.scala', GADGET)):
        d = tmp_path / repo / 'src'
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(body)
    return tmp_path


def _quote(sym: str, line: int | None = None) -> str:
    fname, real_line, text = SNIPPETS[sym]
    return f'{fname}:{line or real_line}\n```scala\n{text}\n```\n'


def _answer(quoted: list[str], prose: str = '', bad_line: dict | None = None) -> dict:
    bad_line = bad_line or {}
    body = '\n'.join(_quote(s, bad_line.get(s)) for s in quoted)
    return {
        'id': 1,
        'answer': f'{body}\n{prose}',
        'files_read': [f'/corpus/{REPO_OF[s]}/src/{SNIPPETS[s][0]}' for s in quoted],
        'tool_calls': [],
    }


def _req(symbols: list[str]) -> dict:
    return {
        'id': 1, 'family': 'synthetic', 'flag': False,
        'required': [{'symbol': s, 'repos': [REPO_OF[s]], 'defines_file': True}
                     for s in symbols],
        'repos_spanned': sorted({REPO_OF[s] for s in symbols}),
        'hops': len(symbols),
    }


def _score(rec, req, corpus):
    return score_chain.score_answer(
        rec, req, score_chain.index(corpus), corpus, {}, tol=3)


CHAIN = ['Widget', 'Middle', 'Gadget']


class TestFullPath:
    def test_every_hop_quoted_is_complete_and_admissible(self, corpus):
        r = _score(_answer(CHAIN), _req(CHAIN), corpus)
        assert r['complete_full'] is True
        assert r['complete'] is True
        assert r['admissible'] is True


class TestEndpointPath:
    def test_endpoints_quoted_and_middle_named_is_complete(self, corpus):
        """The concession: read both ends, articulate the transit between them."""
        rec = _answer(['Widget', 'Gadget'],
                      prose='Widget hands off through Middle, which relays '
                            'downstream to Gadget.')
        r = _score(rec, _req(CHAIN), corpus)
        assert r['complete_endpoints'] is True
        assert r['complete_full'] is False, 'the middle hop was never quoted'
        assert r['admissible'] is True

    def test_unnamed_middle_is_a_jump_not_a_chain(self, corpus):
        """Landing on B from A without saying how is the failure being caught."""
        rec = _answer(['Widget', 'Gadget'],
                      prose='Widget ultimately produces the Gadget result.')
        r = _score(rec, _req(CHAIN), corpus)
        assert r['complete_endpoints'] is False
        assert r['missing_middle'] == ['Middle']
        assert r['admissible'] is False

    def test_naming_the_middle_without_reading_the_ends_is_not_enough(self, corpus):
        """Prose alone never completes: at minimum the two ends must be read."""
        rec = _answer([], prose='Widget calls Middle which reaches Gadget.')
        r = _score(rec, _req(CHAIN), corpus)
        assert r['complete'] is False
        assert r['admissible'] is False


class TestTwoSnippetFloor:
    def test_a_single_snippet_never_completes(self, corpus):
        """Even a one-hop question needs two snippets: quote-then-recite is out."""
        rec = _answer(['Widget'], prose='Widget is the whole mechanism.')
        r = _score(rec, _req(['Widget']), corpus)
        assert r['verified_quotes'] == 1
        assert r['complete'] is False
        assert r['admissible'] is False


class TestCrossRepoEndpoints:
    def test_both_endpoint_quotes_from_one_repo_do_not_span(self, corpus):
        """A delta->spark chain evidenced twice inside delta has not crossed."""
        chain = ['Widget', 'Gadget', 'Middle']  # ends in alpha, spans alpha+beta
        rec = _answer(['Widget', 'Middle'],
                      prose='Widget reaches Middle by way of Gadget.')
        r = _score(rec, _req(chain), corpus)
        assert r['endpoint_repos'] == ['alpha'], 'both ends quoted inside alpha'
        assert r['complete_endpoints'] is False
        assert r['admissible'] is False


class TestQuoteFormats:
    """Three shapes appear in real answers; the line claim must be read from each.

    Measured on the archived run: of 22 answers, 3 numbered every line inside
    the fence and 2 opened the fence with a path header. All five scored zero
    verified quotes against a verifier that only ever compared the fence's first
    line to the file -- so a formatting choice read as fabricated evidence. The
    line number is the claim being checked, and these are three ways of stating
    it, not three levels of rigour.
    """

    def test_line_numbers_inside_the_fence(self, corpus):
        """``4 def emit(): ...`` -- the number is a per-line claim, so use it."""
        _, line, text = SNIPPETS['Widget']
        rec = {'id': 1, 'tool_calls': [],
               'files_read': ['/corpus/alpha/src/Widget.scala'],
               'answer': f'Widget.scala:{line}\n```scala\n{line} {text}\n```\n'}
        q = score_chain.verified_quotes(
            rec, score_chain.index(corpus), corpus, {}, 3)
        assert len(q) == 1, 'a numbered line must verify against its own number'

    def test_a_path_header_opens_the_fence(self, corpus):
        """The header carries both the file and the line; no prose cite needed."""
        _, line, text = SNIPPETS['Gadget']
        rec = {'id': 1, 'tool_calls': [],
               'files_read': ['/corpus/beta/src/Gadget.scala'],
               'answer': f'```scala\n/corpus/beta/src/Gadget.scala:{line}\n{text}\n```\n'}
        q = score_chain.verified_quotes(
            rec, score_chain.index(corpus), corpus, {}, 3)
        assert len(q) == 1, 'a path-header fence must resolve from its own header'
        assert q[0]['repo'] == 'beta'

    def test_a_wrong_inline_number_still_fails(self, corpus):
        """Format tolerance must not become line tolerance."""
        _, line, text = SNIPPETS['Widget']
        rec = {'id': 1, 'tool_calls': [],
               'files_read': ['/corpus/alpha/src/Widget.scala'],
               'answer': f'Widget.scala:40\n```scala\n40 {text}\n```\n'}
        q = score_chain.verified_quotes(
            rec, score_chain.index(corpus), corpus, {}, 3)
        assert q == [], 'the text is verbatim but sits nowhere near line 40'


class TestLineNumbersAreLoadBearing:
    def test_a_quote_citing_the_wrong_line_is_not_evidence(self, corpus):
        """The version-sensitivity guarantee: the number must be right, not near.

        Recall reproduces a path and a symbol name; it does not reproduce the
        line a symbol occupies at a pinned revision. Slack is `tol`, so a
        citation well outside it fails even though the text is verbatim.
        """
        rec = _answer(['Widget', 'Gadget'],
                      prose='Widget hands off through Middle to Gadget.',
                      bad_line={'Gadget': 40})
        r = _score(rec, _req(CHAIN), corpus)
        assert r['verified_quotes'] == 1, 'the mis-cited quote must not verify'
        assert r['complete'] is False
        assert r['admissible'] is False
