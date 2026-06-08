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
    allowed_files: set[str] | None = None,
) -> set[str]:
    """Files in ``source_name`` that contain at least one symbol
    consumed by another source.

    These are exactly the files whose docs benefit from consumer-aware
    regeneration. Files with no cross-source consumers are excluded —
    no need to regenerate something that nothing else uses.

    When ``allowed_files`` is given (the source's doc-gen set — i.e. what
    survived ``find_catalog_files`` after ``exclude_dirs``/``exclude``), the
    result is intersected with it. This makes reverse-augment honor the same
    excludes as the catalog walk: a file the source excludes from doc
    generation is never regenerated just because another source references it.
    ``None`` means no filtering (legacy behavior).
    """
    consumers = graph.consumers_of_source(source_name)
    files = {edge.callee.file for edge in consumers}
    if allowed_files is not None:
        files &= allowed_files
    return files


def build_consumer_context(
    file: str,
    source_name: str,
    graph: 'CrossSourceGraph',
    max_call_sites: int = 20,
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

    # Bound the prompt: a few heavily-referenced files (hundreds of call
    # sites) would otherwise push per-file input into the 100k+-token range.
    # Render at most ``max_call_sites`` in a deterministic order and summarize
    # the rest as an omitted count.
    total = len(relevant)
    relevant = sorted(
        relevant,
        key=lambda e: (
            e.caller.source_name, e.file, e.line,
            e.caller.display_name or e.caller.canonical_id,
        ),
    )[:max_call_sites]
    omitted = total - len(relevant)

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

    if omitted:
        lines.append(
            f'_(+{omitted} more call site(s) omitted to bound context)_',
        )
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def reverse_augment_plan(
    graph: 'CrossSourceGraph',
    source_names: list[str],
    allowed_files: set[str] | None = None,
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
        files = consumed_files_for_source(
            graph, source, allowed_files=allowed_files,
        )
        for file in sorted(files):
            ctx = build_consumer_context(file, source, graph)
            if ctx:
                plan.append((source, file, ctx))
    return plan


def augment_marker(source_bytes: bytes, consumer_context: str) -> str:
    """Freshness key for a reverse-augmented file: ``sha256(source + ctx)``.

    A file only needs re-augmenting when the regeneration prompt would
    differ — i.e. its source changed OR a cross-source caller changed (which
    changes the rendered ``consumer_context``). Hashing both means an
    unchanged marker guarantees an identical prompt, so reusing the prior
    output is safe. This is what lets a re-run skip the expensive second
    generation pass instead of re-billing every consumed file.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(source_bytes)
    h.update(b'\x00')  # separator so (src+ctx) can't alias across the boundary
    h.update(consumer_context.encode('utf-8'))
    return h.hexdigest()


async def run_reverse_augment_phase(
    *,
    graph: 'CrossSourceGraph',
    source_paths: dict,
    analyzer,
    generator,
    allowed_files: set[str] | None = None,
    marker_store=None,
    persist=None,
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

    ``marker_store`` (optional, duck-typed ``get_augment_marker(source,
    path)`` / ``set_augment_marker(source, path, marker)``) makes the pass
    reuse-aware: a file whose :func:`augment_marker` matches the stored one
    is SKIPPED (its source + consumer context are unchanged since the last
    augment), so a re-run doesn't re-bill it. Without a store, every planned
    file is regenerated (legacy behavior).

    ``persist`` (optional, ``await persist(source, file, docs)``) stores a
    file's docs as they're produced. The ordering per file is skip-check →
    generate → persist → mark, so a file is marked fresh ONLY after it is
    durably stored: an aborted pass resumes (completed files skip, the rest
    retry) instead of marking files whose docs never landed.

    Errors from the generator (e.g., LLM rate limits, model errors)
    propagate without being swallowed: regen failures must surface
    so the user can address them, not silently produce stale output.
    """
    from pathlib import Path

    results: list[tuple[str, str, list]] = []
    plan = reverse_augment_plan(
        graph, list(source_paths.keys()), allowed_files=allowed_files,
    )
    for source_name, file_rel, ctx in plan:
        source_root = source_paths[source_name]
        file_abs = Path(source_root) / file_rel

        marker = None
        if marker_store is not None:
            try:
                source_bytes = file_abs.read_bytes()
            except OSError:
                source_bytes = b''
            marker = augment_marker(source_bytes, ctx)
            if marker_store.get_augment_marker(source_name, file_rel) == marker:
                continue  # source + consumer context unchanged → reuse, no re-bill

        metadata = analyzer(file_abs)
        docs = await generator.generate_for_module(
            metadata, extra_prompt_context=ctx,
        )
        if persist is not None:
            await persist(source_name, file_rel, docs)
        results.append((source_name, file_rel, docs))
        # Mark fresh only after a successful generate (+ persist): a failure
        # above propagated before reaching here, leaving the file unmarked.
        if marker_store is not None:
            marker_store.set_augment_marker(source_name, file_rel, marker)
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
    allowed_files: set[str] | None = None,
    marker_store=None,
    persist=None,
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

    target_source_root = Path(target_source_root)

    from docgen.scip_cross_source import (
        CrossSourceGraph,
        load_source_from_manifest,
    )

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

    # Load related sources so their references INTO the target become visible
    # as cross-source consumers; sources without a manifest yet are skipped.
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
        allowed_files=allowed_files,
        marker_store=marker_store,
        persist=persist,
    )


__all__ = [
    'augment_marker',
    'build_consumer_context',
    'consumed_files_for_source',
    'reverse_augment_plan',
    'run_reverse_augment_for_source',
    'run_reverse_augment_phase',
]
