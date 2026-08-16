"""The prompt stage four sends: the chain is the evidence, and nothing else is.

Untested until now, which is how an instruction telling the model to lean on
documentation survived a directive excluding documentation.
"""
from __future__ import annotations

from ariadne_mcp.service_analysis import _chain_prompt, _menu_prompt

SPINE = ('m.run  [m.py:10]  called at c.py:3\n'
         '    def run(self):\n'
         '        return self.engine.start()')


def test_the_prompt_carries_the_chain_and_the_question():
    prompt = _chain_prompt('How does run work?', SPINE)

    assert 'How does run work?' in prompt
    assert SPINE in prompt
    assert 'file:line' in prompt, 'the model must be told to cite coordinates'


def test_the_prompt_carries_no_documentation_block():
    """No block of search-retrieved documents. What the chain selects is a different thing.

    The prompt used to append ``Documentation:\\n{context}`` — eight documents chosen by
    embedding similarity, concatenated, 15,754 tokens measured on a production question —
    and instruct the model to "use the documentation only to explain WHY a step exists".
    That is prose asserted as background authority, and it stays out.

    Per-hop descriptions do travel, inside the chain: each is the ``catalog`` entry for a
    symbol the walk actually reached, fetched by deterministic id rather than searched. The
    difference that matters is which artefact can be wrong — a description can, a
    ``file:line`` from the index cannot — and the prompt says which is which.
    """
    prompt = _chain_prompt('How does run work?', SPINE)

    assert 'Documentation:' not in prompt
    assert 'documentation' not in prompt.lower(), (
        'no instruction may send the model to prose')


def test_the_model_is_told_not_to_invent_a_location():
    """Stage five verifies locations, so the prompt must forbid inventing them."""
    prompt = _chain_prompt('q', SPINE)

    assert 'does not appear in the chain' in prompt
def test_the_prompt_states_where_the_chain_forks_too_widely_to_show():
    """A fork the walk declined to expand is evidence about the code, so the model gets it.

    Otherwise the model answers as if the chain were complete, and the caller never learns
    that the real answer is "this dispatches 529 ways, which area did you mean?".
    """
    note = ('The chain forks at Reader.read: the index holds 529 implementations of it. '
            'Which area are you asking about?')

    prompt = _chain_prompt('How does read work?', SPINE, notes=(note,))

    assert note in prompt
    assert 'forks' in prompt
def test_the_menu_prompt_states_the_reply_format_and_permits_choosing_nothing():
    """The wording is unsettled; the contract it must keep is not.

    An end-to-end variant looked 20 points better and the gain was two substring matches —
    ``DataSource`` inside ``DataSourceV2Relation``, ``MergeIntoTable`` inside 117 protobuf
    builder lines. So this pins what the prompt must always do rather than which persuasion
    it uses: state the numeric reply format, and allow an empty answer, because a model forced
    to choose something will choose noise.
    """
    prompt = _menu_prompt('How does MERGE decide?', 'DEFINITIONS\n  1. A — does a thing')

    assert 'numbers only' in prompt.lower()
    assert 'S-prefixed' in prompt
    assert 'How does MERGE decide?' in prompt
    assert 'choose nothing' in prompt.lower(), (
        'an empty selection must stay available; the chain still travels')
def test_the_prompt_distinguishes_a_call_from_a_type_reference():
    """A reference site is not proof of an invocation, and the prompt must not say it is.

    The chain now labels each hop the way SCIP recorded it — ``called at`` for a ``call``
    edge, ``referenced at`` for a ``type_ref``. Telling the model that the site "proves the
    edge" collapses that distinction at the last step, which is where it does the most
    damage: measured on the live store, an analyzer rule's only inbound edges from outside
    itself are the ``type_ref``s emitted where an extension registers it, so the rule would
    be described as invoked at its registration site — a false mechanism on a coordinate
    that verifies, which is the one error the location guard cannot see.
    """
    prompt = _chain_prompt('q', SPINE)

    assert 'called at' in prompt, 'the call label must be explained'
    assert 'referenced at' in prompt, 'the reference label must be explained'
    assert 'call site that proves the edge' not in prompt, (
        'not every site is a call site: a type reference establishes only that the body '
        'names the symbol')
