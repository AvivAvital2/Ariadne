"""Reverse-augment data preparation — Phase 3.

Two pure functions that compute regeneration inputs for cross-source
reverse-augmentation. The orchestrator (a follow-on slice) calls them
to identify which files in a source need fresh docs and what consumer
context to inject into each regeneration prompt.

Per design decision #4, only SCIP-precise edges feed this — there is
no fallback prose-bridging tier. Consumers in unindexed sources are
invisible here; nothing pretends otherwise.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docgen.scip_cross_source import CrossSourceGraph


def consumed_files_for_source(
    graph: 'CrossSourceGraph', source_name: str,
) -> set[str]:
    """Files in ``source_name`` that contain at least one symbol
    consumed by another source.

    These are exactly the files whose docs benefit from consumer-aware
    regeneration. Files with no cross-source consumers are excluded —
    no need to regenerate something that nothing else uses.
    """
    consumers = graph.consumers_of_source(source_name)
    return {edge.callee.file for edge in consumers}


def build_consumer_context(
    file: str,
    source_name: str,
    graph: 'CrossSourceGraph',
) -> str:
    """Build the markdown prompt block describing ``file``'s
    cross-source consumers.

    Output shape:

        ## Consumer context (SCIP-derived)

        This file is consumed by:

        ### From <caller_source>:
        - `<caller>` calls `<callee>()` (<caller_file>:<line>)
        ...

    Returns ``''`` when no cross-source consumers reference any symbol
    in ``file`` — orchestrator uses this as a signal to skip the
    regeneration entirely.
    """
    consumers = graph.consumers_of_source(source_name)
    relevant = [e for e in consumers if e.callee.file == file]
    if not relevant:
        return ''

    # Group by the caller's source for readable per-consumer sections.
    by_source: dict[str, list] = defaultdict(list)
    for edge in relevant:
        by_source[edge.caller.source_name].append(edge)

    lines = [
        '## Consumer context (SCIP-derived)',
        '',
        'This file is consumed by:',
        '',
    ]
    for caller_source in sorted(by_source):
        edges = sorted(
            by_source[caller_source],
            key=lambda e: (e.file, e.line, e.caller.display_name),
        )
        lines.append(f'### From {caller_source}:')
        for edge in edges:
            caller = edge.caller.display_name or edge.caller.canonical_id
            callee = edge.callee.display_name or edge.callee.canonical_id
            lines.append(
                f'- `{caller}` calls `{callee}()` '
                f'({edge.file}:{edge.line})',
            )
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def reverse_augment_plan(
    graph: 'CrossSourceGraph',
    source_names: list[str],
) -> list[tuple[str, str, str]]:
    """Compute the regeneration plan for the reverse-augment phase.

    For every source in ``source_names``, find files with cross-source
    consumers and emit ``(source_name, file, consumer_context)`` —
    the input the orchestrator hands to ``DocGenerator.generate_for_module``
    via its ``extra_prompt_context`` parameter.

    Pure function: same graph + same input list → identical plan.
    Orchestrator can preview, count, schedule, or run the plan freely.

    Returned list is sorted by ``(source_name, file)`` for
    deterministic regeneration order.
    """
    plan: list[tuple[str, str, str]] = []
    for source in sorted(source_names):
        files = consumed_files_for_source(graph, source)
        for file in sorted(files):
            ctx = build_consumer_context(file, source, graph)
            if ctx:
                plan.append((source, file, ctx))
    return plan


async def run_reverse_augment_phase(
    *,
    graph: 'CrossSourceGraph',
    source_paths: dict,
    analyzer,
    generator,
) -> list[tuple[str, str, list]]:
    """Drive the reverse-augment regeneration phase end-to-end.

    For each ``(source_name, file, consumer_context)`` in the plan,
    re-run doc generation for that file with consumer_context injected
    into the LLM prompt. Returns ``(source_name, file, generated_docs)``
    tuples for the caller to persist.

    The phase is duck-typed against ``analyzer(file_abs) -> metadata``
    and ``await generator.generate_for_module(metadata,
    extra_prompt_context=ctx)`` so production wires the real
    SourceAnalyzer + DocGenerator and tests substitute fakes.

    File paths in the plan are relative to the source's root (per
    manifest convention); ``source_paths`` resolves them to absolute
    paths the analyzer can read. A source listed in ``graph`` but not
    in ``source_paths`` raises a KeyError — fail-loud per design.

    Errors from the generator (e.g., LLM rate limits, model errors)
    propagate without being swallowed: regen failures must surface
    so the user can address them, not silently produce stale output.
    """
    results: list[tuple[str, str, list]] = []
    plan = reverse_augment_plan(graph, list(source_paths.keys()))
    for source_name, file_rel, ctx in plan:
        source_root = source_paths[source_name]
        from pathlib import Path
        file_abs = Path(source_root) / file_rel
        metadata = analyzer(file_abs)
        docs = await generator.generate_for_module(
            metadata, extra_prompt_context=ctx,
        )
        results.append((source_name, file_rel, docs))
    return results


async def run_reverse_augment_for_source(
    *,
    target_source: str,
    target_source_root,
    related_sources: dict,
    analyzer,
    generator,
    max_staleness_days: int = 7,
    index_factory=None,
) -> list[tuple[str, str, list]]:
    """Top-level wrapper for the reverse-augment phase. Builds a
    CrossSourceGraph from manifests on disk, then runs the regen phase
    for ``target_source`` only.

    The integration entry point. ``cmd_generate`` calls this after
    its existing per-source generation completes:

        from docgen.reverse_augment import run_reverse_augment_for_source

        results = await run_reverse_augment_for_source(
            target_source=source_name,
            target_source_root=source_root,
            related_sources={
                name: cfg.get_source_path(name)
                for name in cfg.sources
                if name != source_name
            },
            analyzer=orchestrator._analyzer,
            generator=orchestrator._generator,
        )
        for source_name, file_rel, docs in results:
            for doc in docs:
                library.add_document(...)  # persistence per existing flow

    Missing manifests are tolerated — sources that haven't been
    discovered/indexed yet are skipped. If the target itself has no
    manifest, returns an empty list (nothing to augment).
    """
    from pathlib import Path

    from docgen.scip_cross_source import (
        CrossSourceGraph,
        load_source_from_manifest,
    )

    target_source_root = Path(target_source_root)

    graph = CrossSourceGraph()

    try:
        load_source_from_manifest(
            graph,
            target_source,
            target_source_root,
            max_staleness_days=max_staleness_days,
            index_factory=index_factory,
        )
    except FileNotFoundError:
        # No manifest for the target — nothing to augment.
        return []

    # Load related sources so their references INTO the target
    # become visible as cross-source consumers.
    for related_name, related_root in related_sources.items():
        try:
            load_source_from_manifest(
                graph,
                related_name,
                Path(related_root),
                max_staleness_days=max_staleness_days,
                index_factory=index_factory,
            )
        except FileNotFoundError:
            # Related source not yet indexed — skip; reverse-augment
            # for the target works with whatever consumers ARE
            # currently visible.
            continue

    graph.materialize()

    # Run the phase, scoped to the target source only. Even though
    # other sources are loaded into the graph, we only regen the
    # target's files — re-augmenting other sources is the job of
    # their own ariadne generate runs.
    return await run_reverse_augment_phase(
        graph=graph,
        source_paths={target_source: target_source_root},
        analyzer=analyzer,
        generator=generator,
    )


__all__ = [
    'build_consumer_context',
    'consumed_files_for_source',
    'reverse_augment_plan',
    'run_reverse_augment_for_source',
    'run_reverse_augment_phase',
]
