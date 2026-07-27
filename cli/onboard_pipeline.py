"""Headless onboarding pipeline — the paid phases with no prompts.

``run_onboard_pipeline`` is the code path both ``ariadne onboard`` (behind its
interactive preview + prompts) and the ``ariadne_onboard`` MCP tool run: the
three PAID phases — catalog-describe → generate → themes-build — with every
prompt removed and an optional progress callback. It builds each phase's
argument Namespace from scratch (the MCP caller has no argparse namespace to
inherit from, so every field a phase command reads is set explicitly here).

By default the free phases (discover / index / catalog-sync) and the cost
preview are the caller's job — the CLI runs them in its dry-run preview, so it
leaves them off here to avoid re-indexing. Callers with no preview of their own
(the ``ariadne_onboard`` MCP tool / web wizard) pass ``include_free_phases=True``
so the free phases run first and a SCIP-routed source has a built index before
generate reads it.
"""
from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from cli.catalog import cmd_catalog_describe
from cli.generate import cmd_generate
from cli.themes_cmd import cmd_themes_build

if TYPE_CHECKING:
    from library import Library

# A progress callback: (phase_label, current, total). May be sync or a
# coroutine function — the pipeline awaits it when it returns an awaitable.
ProgressCallback = Callable[[str, int, int], Any]


class OnboardError(RuntimeError):
    """A fatal onboarding phase failed — raised loud, never swallowed.

    Carries the failing phase's exit code so the CLI can propagate it.
    """

    def __init__(self, message: str, rc: int = 1) -> None:
        super().__init__(message)
        self.rc = rc


@dataclass
class OnboardResult:
    """What the paid pipeline produced, read from the Library afterwards."""

    docs_written: int
    themes_found: int
    themes_ok: bool = True
    phases_run: list[str] = field(default_factory=list)


def get_library(db_path=None) -> 'Library':
    """Open the Library (thin indirection so tests can substitute a double)."""
    from cli.core import get_library as _get_library

    return _get_library(db_path)


async def run_onboard_pipeline(
    source: str,
    model: str,
    doc_types: tuple[str, ...],
    *,
    mode: str = 'live',
    concurrency: int | None = None,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
    db_path: str | None = None,
    include_free_phases: bool = False,
) -> OnboardResult:
    """Run the paid onboarding phases for ``source`` and report the result.

    Args:
        source: Source name (already resolved — never a default lookup here).
        model: LLM model for the paid phases (already resolved).
        doc_types: Doc types the generate phase should produce.
        mode: ``'batch'`` for the ~50%-off Message Batches API, else ``'live'``.
        concurrency: Uniform parallel-call cap; ``None`` → each phase's default
            (catalog-describe 4, generate 3).
        verbose: Pass-through to the sub-phases (controls their quiet flag).
        progress: Optional ``(label, current, total)`` callback fired once
            before each phase; awaited if it returns an awaitable.
        db_path: Override the Library db path (else the configured default).

    Returns:
        OnboardResult with the post-run document + theme counts.

    Raises:
        OnboardError: a fatal phase (catalog-describe / generate) returned a
            non-zero rc. Themes is non-fatal — its failure sets
            ``themes_ok=False`` but keeps the completed docs/embeddings.
    """
    use_batch = mode == 'batch'
    cc = concurrency

    catalog_describe_args = argparse.Namespace(
        source=source, force=False, model=model,
        concurrency=cc if cc is not None else 4,
        max_calls=None, dry_run=False, batch=use_batch, resume=False,
        quiet=not verbose, db=db_path,
    )
    generate_args = argparse.Namespace(
        source=source, model=model, provider=None, api_key=None,
        types=','.join(doc_types),
        concurrency=cc if cc is not None else 3,
        force=False, dry_run=False, verbose=verbose,
        path=None, no_crossrefs=False,
        batch_mode='always' if use_batch else 'never',
        auto_batch_threshold=200, confirm_yes=True,
        quiet=not verbose, db=db_path,
    )
    themes_args = argparse.Namespace(
        source=source, themes_action='build', model=model,
        # Honor the run's batch choice for themes too — otherwise theme
        # summarization silently runs live at full price even when the user
        # picked batch (~50% off) for the rest of the onboard.
        batch=use_batch,
        quiet=not verbose, db=db_path, concurrency=cc,
    )

    # (label, command, args, fatal). A fatal phase stops the pipeline; themes
    # is a semantic-clustering augmentation, so its failure must not discard a
    # completed (and, for a spool, paid) generate + embed run.
    phases: list[tuple[str, Callable, argparse.Namespace, bool]] = [
        ('Describing catalog elements', cmd_catalog_describe, catalog_describe_args, True),
        ('Generating documentation', cmd_generate, generate_args, True),
        ('Building themes', cmd_themes_build, themes_args, False),
    ]

    if include_free_phases:
        # Free phases (discover -> index -> catalog-sync), run for real before the
        # paid phases so a SCIP-routed source has a built index to read and the
        # catalog / cross-source graph are populated. The CLI runs these in its
        # dry-run preview; the web tool has no such preview, so it runs them here.
        # Discover reruns at build time (after the wizard persisted excludes), so
        # index_kinds reflects the final scope. All fatal: a failed free phase must
        # stop before any LLM budget is spent.
        from cli.catalog import cmd_catalog_sync as _cmd_catalog_sync
        from cli.index import cmd_discover as _cmd_discover
        from cli.index import cmd_index as _cmd_index
        _cc_free = cc if cc is not None else 4
        free_phases: list[tuple[str, Callable, argparse.Namespace, bool]] = [
            ('Detecting languages', _cmd_discover, argparse.Namespace(
                source=source, all=False, dry_run=False, review=False,
                quiet=not verbose), True),
            ('Indexing symbols (SCIP)', _cmd_index, argparse.Namespace(
                source=source, all=False, dry_run=False, kind=None, force=False,
                best_effort=False, quiet=not verbose), True),
            ('Cataloging files', _cmd_catalog_sync, argparse.Namespace(
                source=source, db=db_path, allow_degraded=False,
                concurrency=_cc_free, force=False, quiet=not verbose), True),
        ]
        phases = free_phases + phases

    themes_ok = True
    phases_run: list[str] = []
    total = len(phases)
    for index, (label, command, ns, fatal) in enumerate(phases, start=1):
        if progress is not None:
            maybe = progress(label, index, total)
            if inspect.isawaitable(maybe):
                await maybe
        result = command(ns)
        rc = await result if inspect.isawaitable(result) else result
        if rc != 0:
            if fatal:
                raise OnboardError(
                    f'Onboard phase {label!r} failed (rc={rc}).', rc=rc)
            themes_ok = False
            continue
        phases_run.append(label)

    library = get_library(db_path)
    return OnboardResult(
        docs_written=library.count_documents(),
        themes_found=len(library.list_themes(coherent_only=False)),
        themes_ok=themes_ok,
        phases_run=phases_run,
    )
