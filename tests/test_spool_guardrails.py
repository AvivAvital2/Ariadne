"""Slice (e1) of the Spool plugin: guardrail schema + structural signals.

The signal DSL is the §6 "code-signal half": a closed set of predicates
evaluated against the catalog, every result carrying fired + evidence +
a confidence label. Caveats A (local-var over-capture → subtype filter)
and B (first-line-truncated signatures → body_contains) from the
2026-07-08 probe are regression-tested here. Synthetic fixtures only.
Design: designs/spool-environment-plugin.md §6 · §9 · IMPLEMENT.md (e1).
"""
import textwrap

import pytest

from guardrails import (
    CatalogView,
    GuardrailError,
    evaluate_guardrail,
    evaluate_signal,
    load_guardrails,
)
from library import Library


def _add_element(lib, *, qualified_name, signature, subtype, file, line_start, line_end):
    # Mirrors the LIVE catalog shape (validated 2026-07-08): the file path
    # lives in the document's source_files, NOT in metadata['file'].
    lib.add_document(
        'catalog', qualified_name, f'{subtype} {qualified_name}',
        source_files=[file],
        metadata={
            'qualified_name': qualified_name,
            'signature': signature,
            'subtype': subtype,
            'location': {'line_start': line_start, 'line_end': line_end},
        },
        source_name='fakelib',
    )


@pytest.fixture
def view(tmp_path):
    module_file = tmp_path / 'engine.py'
    module_file.write_text(textwrap.dedent('''
        def split_table(conn, table_name,
                        train_fraction=0.8,
                        identifier_col_name=None):
            return None
    ''').strip())
    with Library(tmp_path / 'catalog.db') as lib:
        _add_element(
            lib,
            qualified_name='fakelib.api.attach',
            signature='def attach(self, db: FakeDuckdbDatabase) -> Bound:',
            subtype='method',
            file=str(module_file),
            line_start=1, line_end=1,
        )
        _add_element(
            lib,
            qualified_name='fakelib.api.helper_var',
            signature='table = FakeDuckdbTable.load(x)',
            subtype='variable',           # caveat A: over-captured local
            file=str(module_file),
            line_start=1, line_end=1,
        )
        _add_element(
            lib,
            qualified_name='fakelib.ingest.split_table',
            signature='def split_table(conn, table_name,',  # caveat B: truncated
            subtype='function',
            file=str(module_file),
            line_start=1, line_end=5,
        )
        yield CatalogView(lib, source_name='fakelib')


class TestGuardrailSignals:
    def test_signal_evaluation(self, view, tmp_path):
        # Demand 2 — symbol_exists: exact and suffix resolution; evidence
        # names the match; an absent symbol does not fire.
        result = evaluate_signal(
            {'kind': 'symbol_exists', 'pattern': 'fakelib.ingest.split_table'},
            view,
        )
        assert result.fired is True
        assert result.confidence == 'verified (structural)'
        assert 'fakelib.ingest.split_table' in result.evidence[0]

        assert evaluate_signal(
            {'kind': 'symbol_exists', 'pattern': 'ingest.split_table'}, view,
        ).fired is True                                   # suffix match
        assert evaluate_signal(
            {'kind': 'symbol_exists', 'pattern': 'no.such.symbol'}, view,
        ).fired is False

        # Demand 3 — signature_contains vs body_contains: the needle sits
        # past the truncated first line, so only the body predicate fires.
        sig = evaluate_signal(
            {'kind': 'signature_contains',
             'symbol': 'fakelib.ingest.split_table',
             'needle': 'identifier_col_name'},
            view,
        )
        assert sig.fired is False                          # caveat B
        body = evaluate_signal(
            {'kind': 'body_contains',
             'symbol': 'fakelib.ingest.split_table',
             'needle': 'identifier_col_name'},
            view,
        )
        assert body.fired is True
        assert any('engine.py' in e for e in body.evidence)

        # Demand 4 — signature_scan: subtype filter keeps the over-captured
        # variable OUT of the evidence (caveat A).
        scan = evaluate_signal(
            {'kind': 'signature_scan', 'prefix': 'fakelib.api.',
             'needle': 'FakeDuckdb',
             'subtypes': ['function', 'method', 'class']},
            view,
        )
        assert scan.fired is True
        assert any('fakelib.api.attach' in e for e in scan.evidence)
        assert not any('helper_var' in e for e in scan.evidence)

        # Demand 5 — composition + loud unknown kind.
        both = evaluate_signal(
            {'kind': 'all_of', 'signals': [
                {'kind': 'symbol_exists', 'pattern': 'fakelib.api.attach'},
                {'kind': 'symbol_exists', 'pattern': 'no.such.symbol'},
            ]},
            view,
        )
        assert both.fired is False
        either = evaluate_signal(
            {'kind': 'any_of', 'signals': [
                {'kind': 'symbol_exists', 'pattern': 'fakelib.api.attach'},
                {'kind': 'symbol_exists', 'pattern': 'no.such.symbol'},
            ]},
            view,
        )
        assert either.fired is True
        with pytest.raises(GuardrailError) as excinfo:
            evaluate_signal({'kind': 'telepathy'}, view)
        assert 'telepathy' in str(excinfo.value)

    def test_body_contains_degrades_on_unreadable_file(self, tmp_path):
        # CRIT-8 — a body_contains whose target file is stale/missing (moved
        # or deleted since indexing) must NOT raise: it degrades to a
        # non-fired result carrying a signal-error note. A runtime read
        # fault is isolated at the signal; it never propagates. (Schema
        # faults — unknown signal kind — still raise loudly, tested above.)
        with Library(tmp_path / 'catalog.db') as lib:
            lib.add_document(
                'catalog', 'pkg.moved', 'c',
                source_files=[str(tmp_path / 'DELETED-since-index.py')],
                metadata={
                    'qualified_name': 'pkg.moved',
                    'signature': 'def moved():',
                    'subtype': 'function',
                    'location': {'line_start': 1, 'line_end': 1},
                },
                source_name='s',
            )
            view = CatalogView(lib, source_name='s')
            result = evaluate_signal(
                {'kind': 'body_contains', 'symbol': 'pkg.moved',
                 'needle': 'z'},
                view,
            )
            assert result.fired is False
            assert 'signal-error' in result.confidence
            assert any('DELETED-since-index' in e for e in result.evidence)


class TestGuardrailSchema:
    def test_schema_lifecycle(self, view, tmp_path):
        # Demand 1 — load, evaluate, honest-nl, loud-on-missing-field.
        catalog_path = tmp_path / 'guardrails.yaml'
        catalog_path.write_text(textwrap.dedent('''
            guardrails:
              - name: g-fake-coupling
                kind: antipattern
                recommendation: avoid swapping the engine casually
                rationale: public API exposes engine types
                citation: fake-doc.md
                signal_type: structural
                signal:
                  kind: signature_scan
                  prefix: fakelib.api.
                  needle: FakeDuckdb
                  subtypes: [function, method, class]
                requires: []
                provides: [engine-coupling-known]
                spans: [engine-coupling]
              - name: g-fake-semantic
                kind: method
                recommendation: judge row semantics
                rationale: needs semantic judgment
                citation: fake-doc.md
                signal_type: nl
                signal: {kind: nl, prompt: 'does this code shard rows?'}
                requires: []
                provides: [row-semantics-judged]
                spans: [distribution]
        '''))
        loaded = load_guardrails(catalog_path)
        assert [g.name for g in loaded] == ['g-fake-coupling', 'g-fake-semantic']
        assert loaded[0].spans == ('engine-coupling',)
        assert loaded[0].provides == ('engine-coupling-known',)

        fired = evaluate_guardrail(loaded[0], view)
        assert fired.result.fired is True
        assert fired.guardrail.name == 'g-fake-coupling'

        judged = evaluate_guardrail(loaded[1], view)
        assert judged.result.fired is False
        assert judged.result.confidence == 'unevaluated (nl)'

        catalog_path.write_text(textwrap.dedent('''
            guardrails:
              - name: broken
                kind: method
        '''))
        with pytest.raises(GuardrailError) as excinfo:
            load_guardrails(catalog_path)
        assert 'recommendation' in str(excinfo.value)
