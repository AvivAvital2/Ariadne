"""Contract for ``docgen.reverse_augment`` — Phase 3.

Two pure functions that prepare data for the reverse-augment
regeneration phase:

- ``consumed_files_for_source(graph, source_name) -> set[str]``: lists
  the files in ``source_name`` that contain symbols consumed by other
  sources. These are the files whose docs need regeneration.
- ``build_consumer_context(file, source_name, graph) -> str``:
  produces the markdown prompt block that gets injected into the
  regeneration prompt for ``file``, listing each consumer with
  ``source_name``, ``symbol``, and ``file:line``.

The orchestrator integration (calling the regen pipeline with this
context) is a follow-on slice; this slice is the data-prep layer.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _build_graph_with_consumers():
    """Helper: build a CrossSourceGraph where biggerproject's
    SessionManager.refresh calls scalaproject's LicenseService.validate_token.
    """
    from docgen.scip_cross_source import CrossSourceGraph
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    scalaproject_method = (
        'scip-java maven com.scalaproject scalaproject 1 '
        'com/scalaproject/licensing/LicenseService#validate_token().'
    )
    scalaproject_class = (
        'scip-java maven com.scalaproject scalaproject 1 '
        'com/scalaproject/licensing/LicenseService#'
    )
    sm_sym = (
        'scip-java maven com.biggerproject biggerproject 1 '
        'com/biggerproject/SessionManager#refresh().'
    )

    scalaproject_doc = _ScipDoc(
        relative_path='src/main/scala/com/scalaproject/LicenseService.scala',
        occurrences=(
            _ScipOccurrence(
                symbol=scalaproject_class, range=(0, 6, 0, 19),
                is_definition=True,
            ),
            _ScipOccurrence(
                symbol=scalaproject_method, range=(2, 6, 4, 0),
                is_definition=True,
            ),
        ),
        symbols=(
            _ScipSymbol(symbol=scalaproject_class, kind='Class',
                       display_name='LicenseService'),
            _ScipSymbol(symbol=scalaproject_method, kind='Method',
                       display_name='validate_token'),
        ),
    )

    biggerproject_doc = _ScipDoc(
        relative_path='src/main/scala/com/biggerproject/SessionManager.scala',
        occurrences=(
            _ScipOccurrence(
                symbol=sm_sym, range=(20, 6, 22, 0), is_definition=True,
            ),
            _ScipOccurrence(
                symbol=scalaproject_method,
                range=(21, 8, 21, 30),
                is_definition=False,
            ),
        ),
        symbols=(
            _ScipSymbol(
                symbol=sm_sym, kind='Method', display_name='refresh',
            ),
        ),
    )

    graph = CrossSourceGraph()
    graph.add_source(
        'scalaproject',
        index=ScipIndex(documents=(scalaproject_doc,)),
        language='scala',
    )
    graph.add_source(
        'biggerproject',
        index=ScipIndex(documents=(biggerproject_doc,)),
        language='scala',
    )
    graph.materialize()
    return graph


# ---------------------------------------------------------------------------
# consumed_files_for_source
# ---------------------------------------------------------------------------


class TestConsumedFiles:
    def test_returns_files_containing_consumed_symbols(self) -> None:
        from docgen.reverse_augment import consumed_files_for_source

        graph = _build_graph_with_consumers()
        files = consumed_files_for_source(graph, 'scalaproject')
        # validate_token lives in LicenseService.scala — that's the
        # file that needs regeneration with consumer context
        assert 'src/main/scala/com/scalaproject/LicenseService.scala' in files

    def test_returns_empty_when_no_consumers(self) -> None:
        """A source with no cross-source consumers (nothing references
        its symbols from another source) returns an empty set —
        nothing to regenerate."""
        from docgen.reverse_augment import consumed_files_for_source
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        # scalaproject defines a method, but no other source references it
        sym = (
            'scip-java maven com.scalaproject scalaproject 1 '
            'com/scalaproject/Foo#bar().'
        )
        doc = _ScipDoc(
            relative_path='src/main/scala/Foo.scala',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 0, 5), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='bar',
            ),),
        )
        graph = CrossSourceGraph()
        graph.add_source(
            'scalaproject',
            index=ScipIndex(documents=(doc,)),
            language='scala',
        )
        graph.materialize()

        assert consumed_files_for_source(graph, 'scalaproject') == set()

    def test_unknown_source_returns_empty(self) -> None:
        from docgen.reverse_augment import consumed_files_for_source

        graph = _build_graph_with_consumers()
        assert consumed_files_for_source(graph, 'ghost_source') == set()


# ---------------------------------------------------------------------------
# build_consumer_context
# ---------------------------------------------------------------------------


class TestBuildConsumerContext:
    def test_includes_consumer_source_name(self) -> None:
        from docgen.reverse_augment import build_consumer_context

        graph = _build_graph_with_consumers()
        ctx = build_consumer_context(
            'src/main/scala/com/scalaproject/LicenseService.scala',
            'scalaproject',
            graph,
        )
        assert 'biggerproject' in ctx

    def test_includes_call_site_file_line(self) -> None:
        """The rendered prompt block points at exact call sites so
        the LLM can describe usage without making things up."""
        from docgen.reverse_augment import build_consumer_context

        graph = _build_graph_with_consumers()
        ctx = build_consumer_context(
            'src/main/scala/com/scalaproject/LicenseService.scala',
            'scalaproject',
            graph,
        )
        # Reference at 0-indexed line 21 → 1-indexed line 22
        assert (
            'SessionManager.scala:22' in ctx
            or 'SessionManager.scala (line 22)' in ctx
        )

    def test_includes_callee_symbol_name(self) -> None:
        """The LLM needs to know which specific symbol(s) of `file`
        are being consumed — otherwise the prose can't tell `validate_token`
        apart from other methods on the same class."""
        from docgen.reverse_augment import build_consumer_context

        graph = _build_graph_with_consumers()
        ctx = build_consumer_context(
            'src/main/scala/com/scalaproject/LicenseService.scala',
            'scalaproject',
            graph,
        )
        assert 'validate_token' in ctx

    def test_empty_when_file_has_no_consumers(self) -> None:
        """A file in a source whose symbols nobody consumes returns
        an empty string — the orchestrator can use this as a signal
        to skip regeneration."""
        from docgen.reverse_augment import build_consumer_context
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        sym = 'scip-java maven g a 1 com/Foo#unused().'
        doc = _ScipDoc(
            relative_path='Foo.scala',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 0, 5), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='unused',
            ),),
        )
        graph = CrossSourceGraph()
        graph.add_source(
            'lonely',
            index=ScipIndex(documents=(doc,)),
            language='scala',
        )
        graph.materialize()

        ctx = build_consumer_context('Foo.scala', 'lonely', graph)
        assert ctx == ''

    def test_groups_consumers_by_caller_source(self) -> None:
        """When a file is consumed by multiple sources, the rendered
        block groups by consumer source for readability."""
        from docgen.reverse_augment import build_consumer_context
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        # scalaproject defines validate_token
        scalaproject_sym = (
            'scip-java maven com.scalaproject scalaproject 1 '
            'com/scalaproject/Foo#validate().'
        )
        scalaproject_doc = _ScipDoc(
            relative_path='Foo.scala',
            occurrences=(_ScipOccurrence(
                symbol=scalaproject_sym, range=(0, 0, 0, 5),
                is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=scalaproject_sym, kind='Method',
                display_name='validate',
            ),),
        )

        # Two consumers in different sources
        sm_a = 'scip-java maven com.x x 1 com/x/A#callA().'
        a_doc = _ScipDoc(
            relative_path='A.scala',
            occurrences=(
                _ScipOccurrence(
                    symbol=sm_a, range=(5, 0, 7, 0), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=scalaproject_sym, range=(6, 4, 6, 12),
                    is_definition=False,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sm_a, kind='Method', display_name='callA',
            ),),
        )
        sm_b = 'scip-java maven com.y y 1 com/y/B#callB().'
        b_doc = _ScipDoc(
            relative_path='B.scala',
            occurrences=(
                _ScipOccurrence(
                    symbol=sm_b, range=(10, 0, 12, 0), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=scalaproject_sym, range=(11, 4, 11, 12),
                    is_definition=False,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sm_b, kind='Method', display_name='callB',
            ),),
        )

        graph = CrossSourceGraph()
        graph.add_source(
            'scalaproject',
            index=ScipIndex(documents=(scalaproject_doc,)),
            language='scala',
        )
        graph.add_source(
            'src_x',
            index=ScipIndex(documents=(a_doc,)),
            language='scala',
        )
        graph.add_source(
            'src_y',
            index=ScipIndex(documents=(b_doc,)),
            language='scala',
        )
        graph.materialize()

        ctx = build_consumer_context('Foo.scala', 'scalaproject', graph)
        # Both consumer sources appear
        assert 'src_x' in ctx
        assert 'src_y' in ctx
        # Both call sites referenced
        assert 'callA' in ctx
        assert 'callB' in ctx


# ---------------------------------------------------------------------------
# reverse_augment_plan — orchestrator entry point
# ---------------------------------------------------------------------------


class TestReverseAugmentPlan:
    """The plan is what an orchestrator phase consumes: a list of
    ``(source_name, file, consumer_context)`` triples for every file
    that needs regeneration with consumer context.

    Planning is pure: same graph + sources → same plan. The
    orchestrator decides when to invoke regeneration; the planner
    just identifies what.
    """

    def test_plan_emits_one_entry_per_consumed_file(self) -> None:
        from docgen.reverse_augment import reverse_augment_plan

        graph = _build_graph_with_consumers()
        plan = reverse_augment_plan(graph, ['scalaproject', 'biggerproject'])

        assert len(plan) == 1
        source, file, ctx = plan[0]
        assert source == 'scalaproject'
        assert 'LicenseService.scala' in file
        # Context populated, not empty
        assert 'biggerproject' in ctx

    def test_plan_is_empty_when_no_cross_source_edges(self) -> None:
        from docgen.reverse_augment import reverse_augment_plan
        from docgen.scip_cross_source import CrossSourceGraph

        graph = CrossSourceGraph()
        graph.materialize()

        plan = reverse_augment_plan(graph, ['anything'])
        assert plan == []

    def test_plan_skips_sources_outside_input_list(self) -> None:
        """Only sources passed in the ``source_names`` argument are
        considered. If the graph has consumers for source X but X is
        not in the list, no entries for X."""
        from docgen.reverse_augment import reverse_augment_plan

        graph = _build_graph_with_consumers()
        # Only ask for 'biggerproject' — scalaproject is not in the list
        plan = reverse_augment_plan(graph, ['biggerproject'])
        # biggerproject has no consumers; scalaproject does but isn't in list
        assert plan == []

    def test_plan_is_deterministic(self) -> None:
        """Same inputs → same outputs across calls. Important so that
        ariadne generate's reverse-augment phase doesn't churn the
        regeneration order on each invocation."""
        from docgen.reverse_augment import reverse_augment_plan

        graph = _build_graph_with_consumers()
        a = reverse_augment_plan(graph, ['scalaproject'])
        b = reverse_augment_plan(graph, ['scalaproject'])
        assert a == b


# ---------------------------------------------------------------------------
# DocGenerator hook — optional extra prompt context
# ---------------------------------------------------------------------------


class TestGeneratorExtraContext:
    """``DocGenerator.generate_for_module`` accepts an optional
    ``extra_prompt_context`` that's injected into the LLM prompt.
    Reverse-augment uses this to pass consumer context on regen.

    These tests verify the parameter exists and reaches the prompt
    builder without invoking real LLM calls.
    """

    @pytest.mark.asyncio
    async def test_generate_for_module_accepts_extra_prompt_context(
        self, tmp_path: Path,
    ) -> None:
        """Signature contract: the parameter is named
        ``extra_prompt_context``, accepts a string, defaults to empty
        for backwards compat with existing callers."""
        from docgen.generator import DocGenerator
        import inspect

        sig = inspect.signature(DocGenerator.generate_for_module)
        assert 'extra_prompt_context' in sig.parameters
        param = sig.parameters['extra_prompt_context']
        # Default is empty string (backwards compat — existing callers
        # don't pass anything; nothing is appended to their prompts)
        assert param.default == ''


# ---------------------------------------------------------------------------
# run_reverse_augment_phase — orchestration
# ---------------------------------------------------------------------------


class FakeAnalyzer:
    """Records analyze_file calls and returns a stub metadata object."""

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def __call__(self, path):
        self.calls.append(path)
        # Minimal metadata-shaped object — enough for the generator
        # to act on; tests don't inspect it.
        from types import SimpleNamespace
        return SimpleNamespace(path=path, module_name=path.stem)


class FakeGenerator:
    """Records generate_for_module calls and returns canned docs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_for_module(
        self, metadata, doc_types=None, extra_prompt_context: str = '',
    ) -> list:
        self.calls.append({
            'metadata': metadata,
            'doc_types': doc_types,
            'extra_prompt_context': extra_prompt_context,
        })
        return [
            f'<generated doc for {metadata.module_name}>',
        ]


class TestRunReverseAugmentPhase:
    @pytest.mark.asyncio
    async def test_empty_plan_does_not_call_generator(self) -> None:
        from docgen.reverse_augment import run_reverse_augment_phase
        from docgen.scip_cross_source import CrossSourceGraph

        graph = CrossSourceGraph()
        graph.materialize()

        analyzer = FakeAnalyzer()
        generator = FakeGenerator()
        result = await run_reverse_augment_phase(
            graph=graph,
            source_paths={},
            analyzer=analyzer,
            generator=generator,
        )
        assert result == []
        assert analyzer.calls == []
        assert generator.calls == []

    @pytest.mark.asyncio
    async def test_processes_each_plan_entry_with_consumer_context(
        self, tmp_path: Path,
    ) -> None:
        """Each plan entry results in one generate_for_module call
        with the corresponding consumer context as extra_prompt_context."""
        from docgen.reverse_augment import run_reverse_augment_phase

        graph = _build_graph_with_consumers()
        # Simulate source roots — scalaproject's source root is tmp_path
        source_paths = {
            'scalaproject': tmp_path / 'scalaproject',
            'biggerproject': tmp_path / 'biggerproject',
        }
        # Create the file the plan will reference (absolute path lookup)
        file_path = (
            tmp_path / 'scalaproject'
            / 'src/main/scala/com/scalaproject/LicenseService.scala'
        )
        file_path.parent.mkdir(parents=True)
        file_path.write_text('class LicenseService {}', encoding='utf-8')

        analyzer = FakeAnalyzer()
        generator = FakeGenerator()
        result = await run_reverse_augment_phase(
            graph=graph,
            source_paths=source_paths,
            analyzer=analyzer,
            generator=generator,
        )
        # Plan had 1 entry; one generator call with consumer context
        assert len(generator.calls) == 1
        ctx = generator.calls[0]['extra_prompt_context']
        assert 'biggerproject' in ctx
        # Result carries (source, file, docs)
        assert len(result) == 1
        source_name, rel_file, docs = result[0]
        assert source_name == 'scalaproject'
        assert 'LicenseService.scala' in rel_file
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_propagates_generator_exception(
        self, tmp_path: Path,
    ) -> None:
        """If the generator raises (e.g., LLM API failure), the phase
        propagates it — no silent swallowing of regen errors."""
        from docgen.reverse_augment import run_reverse_augment_phase

        graph = _build_graph_with_consumers()
        source_paths = {
            'scalaproject': tmp_path / 'scalaproject',
            'biggerproject': tmp_path / 'biggerproject',
        }
        file_path = (
            tmp_path / 'scalaproject'
            / 'src/main/scala/com/scalaproject/LicenseService.scala'
        )
        file_path.parent.mkdir(parents=True)
        file_path.write_text('class LicenseService {}', encoding='utf-8')

        class FailingGenerator:
            async def generate_for_module(
                self, metadata, doc_types=None, extra_prompt_context='',
            ):
                raise RuntimeError('LLM rate limited')

        with pytest.raises(RuntimeError, match='LLM rate limited'):
            await run_reverse_augment_phase(
                graph=graph,
                source_paths=source_paths,
                analyzer=FakeAnalyzer(),
                generator=FailingGenerator(),
            )


# ---------------------------------------------------------------------------
# run_reverse_augment_for_source — top-level wrapper
# ---------------------------------------------------------------------------


class TestRunReverseAugmentForSource:
    """Top-level wrapper used by cmd_generate. Builds CrossSourceGraph
    from the target source's manifest + any related sources'
    manifests, then runs run_reverse_augment_phase for the target.

    Tests use synthetic indexes (via index_factory injection) so no
    real .scip files are needed.
    """

    @pytest.mark.asyncio
    async def test_loads_manifests_and_runs_phase(
        self, tmp_path: Path,
    ) -> None:
        import json

        from docgen.reverse_augment import run_reverse_augment_for_source
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        target_root = tmp_path / 'scalaproject'
        related_root = tmp_path / 'biggerproject'

        # Scalaproject's manifest + synthetic index
        scalaproject_method = (
            'scip-java maven com.scalaproject scalaproject 1 '
            'com/scalaproject/Foo#bar().'
        )
        cb_doc = _ScipDoc(
            relative_path='Foo.scala',
            occurrences=(_ScipOccurrence(
                symbol=scalaproject_method, range=(2, 6, 4, 0),
                is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=scalaproject_method, kind='Method',
                display_name='bar',
            ),),
        )

        # Biggerproject's index references scalaproject's bar()
        sm_sym = (
            'scip-java maven com.biggerproject biggerproject 1 '
            'com/biggerproject/SM#refresh().'
        )
        biggerproject_doc = _ScipDoc(
            relative_path='SM.scala',
            occurrences=(
                _ScipOccurrence(
                    symbol=sm_sym, range=(20, 6, 22, 0),
                    is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=scalaproject_method,
                    range=(21, 8, 21, 30),
                    is_definition=False,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=sm_sym, kind='Method', display_name='refresh',
            ),),
        )

        for root, indexers in (
            (target_root, [{'kind': 'java', 'cwd': '.',
                            'scip_path': 'index.scip'}]),
            (related_root, [{'kind': 'java', 'cwd': '.',
                             'scip_path': 'index.scip'}]),
        ):
            (root / '.ariadne').mkdir(parents=True)
            (root / '.ariadne' / 'manifest.json').write_text(
                json.dumps({
                    'ariadne_version': '1', 'source_name': root.name,
                    'indexers': indexers,
                }),
                encoding='utf-8',
            )

        # File the analyzer will try to read
        target_file = target_root / 'Foo.scala'
        target_file.write_text('class Foo {}', encoding='utf-8')

        index_map = {
            target_root / '.ariadne' / 'index.scip':
                ScipIndex(documents=(cb_doc,)),
            related_root / '.ariadne' / 'index.scip':
                ScipIndex(documents=(biggerproject_doc,)),
        }

        def fake_factory(path, *, repo, max_staleness_days):
            return index_map[path]

        analyzer = FakeAnalyzer()
        generator = FakeGenerator()
        result = await run_reverse_augment_for_source(
            target_source='scalaproject',
            target_source_root=target_root,
            related_sources={'biggerproject': related_root},
            analyzer=analyzer,
            generator=generator,
            index_factory=fake_factory,
        )
        assert len(result) == 1
        # The plan entry's consumer context mentions biggerproject
        ctx = generator.calls[0]['extra_prompt_context']
        assert 'biggerproject' in ctx

    @pytest.mark.asyncio
    async def test_missing_target_manifest_returns_empty(
        self, tmp_path: Path,
    ) -> None:
        """If the target source has no manifest, the phase produces
        no work — clean empty result, not a crash."""
        from docgen.reverse_augment import run_reverse_augment_for_source

        result = await run_reverse_augment_for_source(
            target_source='scalaproject',
            target_source_root=tmp_path / 'nonexistent',
            related_sources={},
            analyzer=FakeAnalyzer(),
            generator=FakeGenerator(),
        )
        assert result == []


# ---------------------------------------------------------------------------
# DocGenOrchestrator.run_reverse_augment — orchestrator integration
# ---------------------------------------------------------------------------


class TestOrchestratorMethod:
    """The public orchestrator method that cmd_generate calls after
    its main run() completes. Wraps run_reverse_augment_for_source
    using the orchestrator's existing _analyzer and _generator, then
    persists regenerated docs via the orchestrator's _store_document.
    """

    @pytest.mark.asyncio
    async def test_method_exists_with_expected_signature(self) -> None:
        """Signature contract: ``run_reverse_augment(related_sources)``
        is async, takes a dict of source_name → Path."""
        from docgen.orchestrator import DocGenOrchestrator
        import inspect

        assert hasattr(DocGenOrchestrator, 'run_reverse_augment')
        sig = inspect.signature(
            DocGenOrchestrator.run_reverse_augment,
        )
        assert 'related_sources' in sig.parameters
