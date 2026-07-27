"""TDD for the ask() balanced synthesis context.

The anchored search returns repo (anchor) + spool (ground) with the repo floor
first; a plain top-k truncation therefore skews to the repo and starves the
ground, so WITH-spool answers underperform. ``_balanced_ask_docs`` takes the
top of EACH half so the synthesis always sees both — the repo subject and
enough spool context to cross-reference.
"""
from __future__ import annotations

from types import SimpleNamespace

from ariadne_mcp.service_analysis import _balanced_ask_docs


def _doc(doc_id, source_name):
    return SimpleNamespace(id=doc_id, title=doc_id, source_name=source_name)


class TestBalancedAskDocs:
    def test_takes_top_of_both_halves(self):
        # Ranked results: 5 repo (anchor) first (the floor), then 3 spool.
        docs = (
            [_doc(f'repo{i}', 'ao-core') for i in range(5)]
            + [_doc(f'spool{i}', 'spool:databricks') for i in range(3)]
        )
        out = _balanced_ask_docs(
            docs, frozenset({'spool:databricks'}), anchor_n=3, ground_n=3,
        )
        ids = [d.id for d in out]
        # top-3 repo AND top-3 spool — not 5 repo + 0 spool (the old top-k skew)
        assert ids == ['repo0', 'repo1', 'repo2', 'spool0', 'spool1', 'spool2']

    def test_no_spool_returns_repo_only(self):
        # No ground admitted (e.g. a CONTROL question or no spool) => repo only,
        # so WITH-spool never injects irrelevant ground → no-harm preserved.
        docs = [_doc(f'repo{i}', 'ao-core') for i in range(6)]
        out = _balanced_ask_docs(
            docs, frozenset({'spool:databricks'}), anchor_n=4, ground_n=4,
        )
        assert [d.id for d in out] == ['repo0', 'repo1', 'repo2', 'repo3']

    def test_ground_ranking_preserved_within_half(self):
        # Whatever order the ranker produced within each half is kept.
        docs = [
            _doc('r0', 'ao-core'), _doc('s0', 'spool:databricks'),
            _doc('r1', 'ao-core'), _doc('s1', 'spool:databricks'),
        ]
        out = _balanced_ask_docs(
            docs, frozenset({'spool:databricks'}), anchor_n=1, ground_n=1,
        )
        assert [d.id for d in out] == ['r0', 's0']


def _cdoc(doc_id, source_name, content='body text'):
    return SimpleNamespace(
        id=doc_id, title=doc_id, source_name=source_name, content=content)


class TestEnvironmentLabel:
    """The CONSIDERING header's '<env> (runtime <pin>)' label. The pin lives
    on registration.manifest.target_runtime — reading it off the
    registration object silently dropped the pin (regression)."""

    def test_reads_pin_through_registration(self):
        from ariadne_mcp.service_analysis import _environment_label
        registration = SimpleNamespace(
            manifest=SimpleNamespace(target_runtime='fake-17.3'))
        resolution = SimpleNamespace(registered={'env1': registration})
        assert _environment_label(resolution) == 'env1 (runtime fake-17.3)'

    def test_bare_manifest_and_missing_pin(self):
        from ariadne_mcp.service_analysis import _environment_label
        resolution = SimpleNamespace(registered={
            'a': SimpleNamespace(target_runtime='r1'),
            'b': SimpleNamespace(),
        })
        assert _environment_label(resolution) == 'a (runtime r1), b'

    def test_none_when_nothing_registered(self):
        from ariadne_mcp.service_analysis import _environment_label
        assert _environment_label(SimpleNamespace(registered={})) is None


class TestLabeledAssembly:
    """The design-§7 labeled synthesis context: two attributed streams —
    'GIVEN' (the project's own docs) and 'CONSIDERING' (the environment
    reference, authoritative-where-relevant + injection guard, per-item
    connection labels). The old per-doc 'UNTRUSTED' fence measurably made
    the synthesis discount certified docs; the injection guard survives the
    rewrite, the distrust framing does not."""

    def test_two_streams_authoritative_with_labels_and_pin(self):
        from ariadne_mcp.service_analysis import _assemble_ask_context
        out = _assemble_ask_context(
            [_cdoc('r0', 'src1'), _cdoc('s0', 'spool:env1')],
            frozenset({'spool:env1'}),
            connections={'s0': 'entity(Quantum Mesh)'},
            environment_label='env1 (runtime fake-17.3)',
        )
        assert 'UNTRUSTED' not in out
        assert 'GIVEN' in out and 'CONSIDERING' in out
        assert 'authoritative' in out.lower()
        assert 'instructions' in out.lower()          # the guard survives
        assert 'entity(Quantum Mesh)' in out
        assert 'fake-17.3' in out
        assert out.index('GIVEN') < out.index('CONSIDERING')

    def test_spool_primary_flips_the_streams(self):
        # Bidirectional lens: on spool-primary questions the ENVIRONMENT is
        # the given (authoritative, guard intact) and the project's context
        # is the considering — the repo lens docs carry their labels.
        from ariadne_mcp.service_analysis import _assemble_ask_context
        out = _assemble_ask_context(
            [_cdoc('s0', 'spool:env1'), _cdoc('r0', 'src1')],
            frozenset({'spool:env1'}),
            connections={'s0': 'entity(Quantum Mesh)', 'r0': 'repo(0.61)'},
            environment_label='env1 (runtime fake-17.3)',
            primary='spool',
        )
        assert 'GIVEN' in out and 'CONSIDERING' in out
        given_block = out[:out.index('CONSIDERING')]
        assert 'environment reference' in given_block
        assert 'fake-17.3' in given_block
        assert 'authoritative' in given_block.lower()
        assert 'instructions' in given_block.lower()   # guard rides the env
        assert 'repo(0.61)' in out                     # lens label survives
        assert out.index('## s0') < out.index('## r0')

    def test_no_spool_docs_means_plain_context(self):
        from ariadne_mcp.service_analysis import _assemble_ask_context
        out = _assemble_ask_context(
            [_cdoc('r0', 'src1'), _cdoc('r1', 'src1')],
            frozenset({'spool:env1'}),
            connections=None, environment_label='env1 (runtime fake-17.3)',
        )
        assert 'GIVEN' not in out and 'CONSIDERING' not in out
        assert 'UNTRUSTED' not in out
        assert '## r0' in out and '## r1' in out

    def test_expert_only_environment_stream_alone(self):
        from ariadne_mcp.service_analysis import _assemble_ask_context
        out = _assemble_ask_context(
            [_cdoc('s0', 'spool:env1'), _cdoc('s1', 'spool:env1')],
            frozenset({'spool:env1'}),
            connections={'s0': 'semantic(0.78)'},
            environment_label='env1 (runtime fake-17.3)',
        )
        assert 'CONSIDERING' in out and 'GIVEN' not in out
        assert 'semantic(0.78)' in out
