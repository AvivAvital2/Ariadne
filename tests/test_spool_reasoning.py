"""Slice (f1) of the Spool plugin: the reasoning-path machinery.

Decompose along the Spool's concern taxonomy → assemble the capability
profile (probes + spans-matched guardrails, all through the e1 signal
evaluator) → consistency-pass the selected methods (requires/provides;
unmet = consult point, never auto-resolved) → compose the serializable
construction plan. No LLM anywhere — selection is an explicit input.
Synthetic fixtures only. Design: §7 · §9 (#1/#2/#4) · IMPLEMENT.md (f1).
"""
import textwrap

import pytest

from guardrails import CatalogView, load_guardrails
from library import Library
from spool_reasoning import (
    ReasoningError,
    assemble_capability_profile,
    compose_plan,
    consistency_pass,
    load_taxonomy,
    render_plan_markdown,
    render_selection_request,
    select_methods,
)


@pytest.fixture
def view(tmp_path):
    module_file = tmp_path / 'engine.py'
    module_file.write_text('def compute(conn):\n    return conn\n')
    with Library(tmp_path / 'catalog.db') as lib:
        lib.add_document(
            'catalog', 'fakelib.api.attach', 'method fakelib.api.attach',
            source_files=[str(module_file)],
            metadata={
                'qualified_name': 'fakelib.api.attach',
                'signature': 'def attach(self, db: FakeDuckdbDatabase):',
                'subtype': 'method',
                'location': {'line_start': 1, 'line_end': 1},
            },
            source_name='fakelib',
        )
        yield CatalogView(lib, source_name='fakelib')


def _taxonomy(tmp_path):
    path = tmp_path / 'taxonomy.yaml'
    path.write_text(textwrap.dedent('''
        environment: fakebricks
        concerns:
          - name: engine-coupling
            description: how tightly the code binds its compute engine
            probes:
              - kind: signature_scan
                prefix: fakelib.api.
                needle: FakeDuckdb
                subtypes: [function, method, class]
          - name: distribution
            description: how work can spread across nodes
    '''))
    return path


def _guardrails(tmp_path):
    path = tmp_path / 'guardrails.yaml'
    path.write_text(textwrap.dedent('''
        guardrails:
          - name: g-coupling
            kind: antipattern
            recommendation: engine swap is re-engineering
            rationale: public API exposes engine types
            citation: doc.md
            signal_type: structural
            signal:
              kind: signature_scan
              prefix: fakelib.api.
              needle: FakeDuckdb
              subtypes: [function, method, class]
            provides: [engine-coupling-known]
            spans: [engine-coupling]
          - name: m-scatter
            kind: method
            recommendation: scatter feature batches
            rationale: features are independent
            citation: doc.md
            signal_type: structural
            signal: {kind: symbol_exists, pattern: 'fakelib.api.attach'}
            requires: [stable-identity]
            provides: [distributed-compute]
            spans: [distribution]
          - name: m-identity
            kind: method
            recommendation: stamp a stable identifier
            rationale: physical row ids drift
            citation: doc.md
            signal_type: structural
            signal: {kind: symbol_exists, pattern: 'fakelib.api.attach'}
            requires: [ingest-owns-canonical]
            provides: [stable-identity]
            spans: [distribution]
    '''))
    return path


class TestReasoningMachinery:
    def test_reasoning_pipeline(self, view, tmp_path):
        # Demand 1 — taxonomy schema: loads; concerns ordered; loud on
        # a concern without a name.
        taxonomy = load_taxonomy(_taxonomy(tmp_path))
        assert taxonomy.environment == 'fakebricks'
        assert [c.name for c in taxonomy.concerns] == [
            'engine-coupling', 'distribution',
        ]
        broken = tmp_path / 'broken.yaml'
        broken.write_text('environment: x\nconcerns:\n  - description: no name\n')
        with pytest.raises(ReasoningError) as excinfo:
            load_taxonomy(broken)
        assert 'name' in str(excinfo.value)

        # Demand 2 — capability profile: per-concern probe evidence plus
        # spans-matched guardrail results through the e1 evaluator.
        guardrails = load_guardrails(_guardrails(tmp_path))
        profile = assemble_capability_profile(taxonomy, guardrails, view)
        coupling = profile.finding('engine-coupling')
        assert coupling.probe_results[0].fired is True
        assert any(
            'fakelib.api.attach' in e
            for e in coupling.probe_results[0].evidence
        )
        assert [g.guardrail.name for g in coupling.guardrails] == ['g-coupling']
        distribution = profile.finding('distribution')
        assert distribution.probe_results == ()
        assert {g.guardrail.name for g in distribution.guardrails} == {
            'm-scatter', 'm-identity',
        }

        # Demand 3 — consistency pass: chained provides satisfy requires;
        # what nothing provides becomes a consult point (never silently
        # resolved); the baseline can satisfy it.
        selected = [g for g in guardrails if g.kind == 'method']
        report = consistency_pass(selected, baseline=frozenset())
        assert report.met == {'stable-identity': 'm-identity'}
        assert [(c.guardrail, c.requirement) for c in report.consult_points] == [
            ('m-identity', 'ingest-owns-canonical'),
        ]
        report = consistency_pass(
            selected, baseline=frozenset({'ingest-owns-canonical'}),
        )
        assert report.consult_points == ()
        assert report.met['ingest-owns-canonical'] == 'baseline'

        # Demand 4 — the plan: per-concern selection + why + citations +
        # consult points; serializable via to_dict.
        plan = compose_plan(
            goal='run fakelib on fakebricks, distributed',
            profile=profile,
            selections={'distribution': 'm-scatter', 'engine-coupling': None},
            guardrails=guardrails,
            baseline=frozenset({'ingest-owns-canonical', 'stable-identity'}),
        )
        step = plan.step('distribution')
        assert step.selected == 'm-scatter'
        assert 'features are independent' in step.why
        assert plan.consult_points == ()
        data = plan.to_dict()
        assert data['goal'].startswith('run fakelib')
        assert data['steps'][1]['selected'] == 'm-scatter'

        with pytest.raises(ReasoningError) as excinfo:
            compose_plan(
                goal='g', profile=profile,
                selections={'distribution': 'no-such-guardrail'},
                guardrails=guardrails, baseline=frozenset(),
            )
        assert 'no-such-guardrail' in str(excinfo.value)

        # Demand 5 — the one output shape (§19.1): goal, per-concern steps
        # with why + file:line evidence, Decisions needed, confidence
        # legend, provenance footer.
        # A method selected for TWO concerns must not duplicate its
        # consult point (found live in the e2e dry run).
        open_plan = compose_plan(
            goal='run fakelib on fakebricks, distributed',
            profile=profile,
            selections={'distribution': 'm-identity',
                        'engine-coupling': 'm-identity'},
            guardrails=guardrails,
            baseline=frozenset(),
        )
        assert len(open_plan.consult_points) == 1
        report = render_plan_markdown(
            open_plan, provenance='fakebricks spool 1.0 (runtime fake-17.3)',
        )
        assert 'run fakelib on fakebricks' in report
        assert 'm-identity' in report
        assert 'physical row ids drift' in report          # the why
        assert 'engine.py:1' in report                     # file:line evidence
        assert 'Decisions needed' in report
        assert 'ingest-owns-canonical' in report
        assert 'verified (structural)' in report           # confidence legend
        assert 'fakebricks spool 1.0' in report            # provenance footer

        # Demand 6 — the selection digest: candidates with their contracts
        # + the strict-JSON output instructions.
        request = render_selection_request(
            'run fakelib on fakebricks', profile,
        )
        assert 'engine-coupling' in request
        assert 'm-scatter' in request
        assert 'requires: stable-identity' in request
        assert 'JSON' in request

        # Demand 7 — the selection loop: valid JSON accepted; an invalid
        # reply gets ONE repair re-prompt carrying the error; persistent
        # garbage is loud. Catalog-only selection is the structural
        # row-sharding protection (§19.3).
        calls = []

        def good_llm(prompt):
            calls.append(prompt)
            return ('{"selections": {"distribution": "m-scatter"}, '
                    '"assumed_baseline": ["stable-identity"], "notes": ""}')

        selection = select_methods(
            'goal', profile, guardrails, llm=good_llm,
        )
        assert selection.selections == {'distribution': 'm-scatter'}
        assert selection.assumed_baseline == ('stable-identity',)
        assert len(calls) == 1

        calls.clear()
        replies = iter([
            '{"selections": {"distribution": "shard-all-the-rows"}}',
            '{"selections": {"distribution": "m-scatter"}}',
        ])

        def repairing_llm(prompt):
            calls.append(prompt)
            return next(replies)

        selection = select_methods(
            'goal', profile, guardrails, llm=repairing_llm,
        )
        assert selection.selections == {'distribution': 'm-scatter'}
        assert len(calls) == 2
        assert 'shard-all-the-rows' in calls[1]            # repair carries the error

        garbage_calls = []

        def garbage_llm(prompt):
            garbage_calls.append(prompt)
            return 'row sharding is great, just do it'

        with pytest.raises(ReasoningError) as excinfo:
            select_methods('goal', profile, guardrails, llm=garbage_llm)
        assert 'invalid' in str(excinfo.value).lower()
        assert len(garbage_calls) == 2      # initial + ONE repair, no waste


class TestReasoningResilience:
    def test_stale_guardrail_does_not_crash_profile(self, tmp_path):
        # CRIT-8 blast radius — one guardrail whose body_contains target
        # file is stale must NOT sink the whole capability profile; the
        # healthy guardrail on the same concern is still reported.
        open(tmp_path / 'taxonomy.yaml', 'w').write(textwrap.dedent('''
            environment: e
            concerns:
              - {name: distribution, description: d}
        '''))
        open(tmp_path / 'guardrails.yaml', 'w').write(textwrap.dedent('''
            guardrails:
              - name: healthy
                kind: method
                recommendation: r
                rationale: r
                citation: c
                signal_type: structural
                signal: {kind: symbol_exists, pattern: pkg.good}
                spans: [distribution]
              - name: stale-file
                kind: method
                recommendation: r
                rationale: r
                citation: c
                signal_type: structural
                signal: {kind: body_contains, symbol: pkg.moved, needle: z}
                spans: [distribution]
        '''))
        taxonomy = load_taxonomy(tmp_path / 'taxonomy.yaml')
        guardrails = load_guardrails(tmp_path / 'guardrails.yaml')
        with Library(tmp_path / 'catalog.db') as lib:
            lib.add_document(
                'catalog', 'pkg.good', 'c',
                source_files=[str(tmp_path / 'present.py')],
                metadata={'qualified_name': 'pkg.good',
                          'signature': 'def good():', 'subtype': 'function',
                          'location': {'line_start': 1, 'line_end': 1}},
                source_name='s',
            )
            lib.add_document(
                'catalog', 'pkg.moved', 'c',
                source_files=[str(tmp_path / 'DELETED.py')],
                metadata={'qualified_name': 'pkg.moved',
                          'signature': 'def moved():', 'subtype': 'function',
                          'location': {'line_start': 1, 'line_end': 1}},
                source_name='s',
            )
            (tmp_path / 'present.py').write_text('def good(): pass\n')
            view = CatalogView(lib, source_name='s')
            profile = assemble_capability_profile(taxonomy, guardrails, view)
        finding = profile.finding('distribution')
        by_name = {g.guardrail.name: g for g in finding.guardrails}
        assert set(by_name) == {'healthy', 'stale-file'}
        assert by_name['healthy'].result.fired is True
        # the stale one degraded, not crashed
        assert by_name['stale-file'].result.fired is False
        assert 'signal-error' in by_name['stale-file'].result.confidence
