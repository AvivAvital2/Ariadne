"""SCIP indexing CLI commands (``index``, ``discover``) + the indexer subsystem.

Extracted from cli/core.py. ``index`` runs the per-language indexers
(scip-*, ast-grep) and the persist chain; ``discover`` walks the source tree
to fill ariadne.yaml. Wired into the parser via this module's
``register_commands`` + ``HANDLERS`` (assembled in cli/main.py).
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path

from attrs import frozen as _frozen
from rich.console import Console
from rich.table import Table

from config import get_config

console = Console()


_LANGUAGE_LABELS = {
    'python': 'Python',
    'typescript': 'TypeScript',
    'java': 'Java',
}


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register SCIP indexing commands."""

    # discover — walk a source tree, identify SCIP indexer locations,
    # write .ariadne/manifest.json, ensure .gitignore excludes .ariadne/.
    discover_parser = subparsers.add_parser(
        'discover',
        help=(
            'Walk a source tree, detect SCIP indexer locations, write '
            'manifest.json'
        ),
    )
    discover_parser.add_argument(
        '--source', '-s', help='Source name to discover',
    )
    discover_parser.add_argument(
        '--all', action='store_true',
        help='Run discover on every source in ariadne.yaml',
    )
    discover_parser.add_argument(
        '--dry-run', action='store_true',
        help='Print the manifest without writing files',
    )
    discover_parser.add_argument(
        '--review', action='store_true',
        help=(
            'Walk suspect entries (vendor bundles, mass-duplicate dirs) '
            'and prompt y/N. Rejected entries get appended to the source\'s '
            '`exclude:` list in ariadne.yaml.'
        ),
    )
    discover_parser.add_argument(
        '--config-only', action='store_true',
        help=(
            'Write manifest + ariadne.yaml auto-managed block, but skip '
            'the indexer-run step. Used by sync when it detects a new '
            'language extension, so config stays current without paying '
            'the scip-X cost.'
        ),
    )

    # index — invoke per-language SCIP indexers, merge into one .scip
    index_parser = subparsers.add_parser(
        'index',
        help=(
            'Run SCIP indexers per the manifest, merge intermediates '
            'into <source>/.ariadne/index.scip'
        ),
    )
    index_parser.add_argument(
        '--source', '-s', help='Source name to index',
    )
    index_parser.add_argument(
        '--all', action='store_true',
        help='Run index on every source in ariadne.yaml',
    )
    index_parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would run without invoking indexers',
    )
    index_parser.add_argument(
        '--kind', choices=['python', 'typescript', 'java'],
        help='Run only entries of this indexer kind (skip the rest)',
    )
    index_parser.add_argument(
        '--force', '-f', action='store_true',
        help='Re-index even if the .scip artifact is still fresh (younger '
             'than the source max_staleness_days). Without this, a fresh '
             'index is reused so re-runs (e.g. onboard --approve) skip the '
             'slow indexers.',
    )


@_frozen
class IndexerResult:
    """What an IndexerAdapter returns from ``run``. Hard-fail semantics
    (decision #5) means any non-success aborts the index command.

    ``resolved_interpreter`` and ``resolution_source`` are populated by
    Python adapters (Phase 2n) so callers can log which interpreter was
    picked; they're optional / empty for non-Python indexers.

    ``vue_mapping_path`` is set by the TypeScript adapter when it ran the
    Vue extractor; ``cmd_index`` records it on the manifest entry as
    ``vue_mapping`` so the loader can translate companion paths back to
    ``.vue`` files. Empty for runs without Vue.
    """
    success: bool
    indexer_version: str = ''
    error_message: str = ''
    resolved_interpreter: Path | None = None
    resolution_source: str = ''
    vue_mapping_path: str = ''


class _SubprocessMerger:
    """Default merger — shells out to the ``scip`` CLI's ``merge``
    subcommand. Returns False if scip is not on PATH or the merge
    fails. Tests substitute a fake.

    Prints an actionable diagnostic on failure rather than swallowing
    it silently. ``scip merge failed`` with no context made debugging
    pointless — users now see whether the binary is missing (install
    hint), whether scip itself returned nonzero (with its stderr), or
    something else.
    """

    def merge(self, inputs: list[Path], output: Path) -> bool:
        import subprocess
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = (
            ['scip', 'merge', '--output', str(output)]
            + [str(p) for p in inputs]
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except FileNotFoundError:
            console.print(
                '[red]scip CLI not on PATH — install with '
                '`brew install sourcegraph/scip/scip` (or '
                'see https://github.com/sourcegraph/scip).[/red]',
            )
            return False
        except subprocess.CalledProcessError as e:
            stderr = e.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', errors='replace')
            console.print(
                f'[red]scip merge exited {e.returncode}.[/red]',
            )
            if stderr and stderr.strip():
                console.print(f'[red]{stderr.strip()}[/red]')
            return False


def _default_indexer_registry() -> dict:
    """Lazy-load adapters so cli.core's import doesn't pull docgen at
    module init (avoids circular imports — adapters import IndexerResult
    from this module). Tests bypass this by passing their own registry."""
    from docgen.scip_indexers import (
        JavaIndexerAdapter,
        PythonIndexerAdapter,
        TypescriptIndexerAdapter,
    )
    return {
        'java': JavaIndexerAdapter(),
        'python': PythonIndexerAdapter(),
        'typescript': TypescriptIndexerAdapter(),
    }


def _scope_label(entry: dict, kind: str) -> str:
    """Label for an intermediate .scip file. ``cwd='.'`` becomes just
    the kind; nested cwd flattens slashes for safe filenames."""
    cwd = entry.get('cwd', '.')
    if cwd == '.':
        return kind
    return f'{cwd.replace("/", "-")}-{kind}'


def _streams_file_progress(kind: str) -> bool:
    """True if the adapter for ``kind`` emits per-file progress events.

    Only scip-python streams parseable ``current/total`` ticks. scip-java
    and scip-typescript are opaque subprocesses (they compile/index in one
    shot), so their bars can't track files — we show an animated
    indeterminate bar for them instead of a counter frozen at 0/N.
    """
    return kind == 'python'


def _pulse_bar(kind: str) -> bool:
    """True if ``kind``'s bar should be an animated indeterminate (pulse)
    bar rather than a determinate file counter.

    Only scip-java qualifies: it compiles the whole project in one
    monolithic, progress-less shot, so a ``completed/total`` bar would
    sit frozen at 0/N for minutes. Python streams per-file progress, and
    scip-typescript advances per scope — both get a real counter.
    """
    return kind == 'java'


def _index_detail_text(file_total: int, completed: int, *, pulse: bool) -> str:
    """Detail text for a per-language bar.

    Determinate bars (Python, TypeScript/JS) show a ``done/total files``
    counter. The pulse bar (Java) can't track progress, so it shows just
    the total — ``N files`` — without a meaningless ``0/``.
    """
    if pulse:
        return f'{int(file_total)} files'
    return f'{int(completed)}/{int(file_total)} files'


@functools.lru_cache(maxsize=1)
def _indexer_kind_extensions() -> dict[str, frozenset[str]]:
    """Map each indexer kind to the source-file extensions it covers,
    derived from the language registry so it can't drift.

    Cached: the registry is static, and ``_count_source_files`` is called
    once per scope, so without this the whole map is rebuilt N times per
    ``index`` run. Callers only read the returned dict."""
    from docgen.scip_languages import LANGUAGES

    out: dict[str, set[str]] = {}
    for lang in LANGUAGES:
        out.setdefault(lang.indexer_kind, set()).update(
            lang.source_extensions,
        )
    return {k: frozenset(v) for k, v in out.items()}


def _count_source_files(
    cwd: Path, kind: str, excluded_dir_names: frozenset[str],
) -> int:
    """Count indexable source files of ``kind`` under ``cwd``.

    Used only to size and order the per-language progress bars, so it
    needs to be a consistent estimate, not byte-exact to what the
    indexer ultimately reports. Skips excluded dirs, ``node_modules``,
    and minified JS bundles (the same things discovery ignores)."""
    import os

    from docgen.scip_discovery import _is_minified_js

    exts = _indexer_kind_extensions().get(kind, frozenset())
    if not exts:
        return 0
    skip = set(excluded_dir_names) | {'node_modules', '.git', '.ariadne'}
    count = 0
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if Path(f).suffix.lower() in exts and not _is_minified_js(f):
                count += 1
    return count


def _plan_indexing(entries: list, count_fn) -> list[tuple]:
    """Group manifest indexer entries by language and order the groups
    by total file volume, smallest first.

    Returns ``[(kind, [entries...], total_files), ...]``. Entry order is
    preserved within each language. ``count_fn(entry) -> int`` is
    injected so this stays pure (no filesystem) for testing.
    """
    groups: dict[str, list] = {}
    totals: dict[str, int] = {}
    for entry in entries:
        kind = entry.get('kind')
        groups.setdefault(kind, []).append(entry)
        totals[kind] = totals.get(kind, 0) + count_fn(entry)
    # Smallest volume first; kind name breaks ties deterministically.
    ordered = sorted(groups, key=lambda k: (totals[k], k or ''))
    return [(k, groups[k], totals[k]) for k in ordered]


def cmd_index(
    args: argparse.Namespace,
    *,
    indexer_registry: dict | None = None,
    merger=None,
    phase_summary: list | None = None,
) -> int:
    """Run SCIP indexers per the manifest, merge intermediates into
    ``<source>/.ariadne/index.scip``, then run the 12-step persist
    chain that fills ``library_scip`` tables.

    Persist chain (dependency-correct order, end of every successful
    index):

    1. ``persist_all_sources`` → ``scip_symbols`` / ``scip_edges`` /
       ``scip_index_state`` (Tier 1 cross-source graph)
    2. ``persist_api_endpoints`` → ``api_endpoints`` from Swagger
       specs (only when source declares ``swagger_paths``)
    3. ``persist_string_literals`` → required by route extractors
       below to look up literal path values by SCIP position
    4. ``persist_config_values`` → ``config_values`` (HOCON / YAML /
       dotenv key→value defaults)
    5. ``persist_config_reads`` → ``config_reads`` (config-getter call
       sites resolved to their keys + values; needs steps 3 + 4)
    6. ``persist_akka_http_endpoints`` → ``api_endpoints`` (pattern,
       Scala Akka HTTP)
    7. ``persist_python_routes`` → ``api_endpoints`` (pattern,
       Flask / FastAPI)
    8. ``persist_express_routes`` → ``api_endpoints`` (pattern,
       Express / Koa)
    9. ``persist_python_http_clients`` → ``http_client_calls``
       (Tier 4 — ``httpx`` / ``requests`` / ``urllib``)
    10. ``persist_js_http_clients`` → ``http_client_calls``
        (``fetch`` / ``axios``)
    11. ``persist_scala_http_clients`` → ``http_client_calls``
        (Akka HTTP / sttp)
    12. ``persist_url_resolver`` → ``api_calls`` (joins client
        URLs to server endpoints, closing the cross-language tracing
        loop for ``ariadne_trace_flow``)

    Each persist_* helper is fail-soft on missing artifacts (a source
    without a current ``.scip`` returns 0 from its persist; the chain
    continues for the rest). The wiring contract is pinned in
    ``tests/test_cmd_index_wiring_chain.py``.

    Hard-fails on any indexer or merge error per decision #5. Updates
    each manifest entry with ``scip_path``, ``indexed_at``, and
    ``indexer_version`` so subsequent ``ariadne sync`` knows where to
    load from.

    ``indexer_registry`` and ``merger`` are dependency-injection
    points for tests; production uses the module-level defaults.
    """
    import json
    from datetime import datetime, timezone

    cfg = get_config()

    if args.all:
        sources_to_process = list(cfg.sources.keys())
        if not sources_to_process:
            console.print(
                '[red]No sources configured in ariadne.yaml[/red]',
            )
            return 1
    elif args.source:
        sources_to_process = [args.source]
    else:
        if cfg.default_source:
            sources_to_process = [cfg.default_source]
        else:
            console.print(
                '[red]No source specified — use --source X, --all, or '
                'set default_source in ariadne.yaml[/red]',
            )
            return 1

    if indexer_registry is None:
        indexer_registry = _default_indexer_registry()
    if merger is None:
        merger = _SubprocessMerger()

    for source_name in sources_to_process:
        sc = cfg.get_source_config(source_name)
        if sc is None:
            console.print(
                f'[red]Source not found in config: {source_name}[/red]',
            )
            return 1

        source_root = Path(sc.path).expanduser().resolve()

        # Freshness skip: if the merged .scip is still fresh (younger than
        # the source's max_staleness_days), reuse it rather than re-running
        # the slow per-language indexers. The artifact SHA only gates the
        # persist step, so the indexer skip is necessarily time-based. The
        # persist phase below walks every configured source, so it still
        # reloads this artifact — only the expensive indexer+merge is
        # skipped. ``--force`` bypasses. This is what makes a re-run of
        # ``onboard --approve`` (which re-enters the free phases) not
        # re-index when nothing has gone stale.
        if not getattr(args, 'force', False):
            merged_scip = source_root / '.ariadne' / 'index.scip'
            if merged_scip.exists():
                max_days = cfg.effective_scip_staleness_days(source_name)
                age_days = (
                    datetime.now(timezone.utc).timestamp()
                    - merged_scip.stat().st_mtime
                ) / 86400
                if max_days is None or age_days < max_days:
                    if not getattr(args, 'quiet', False):
                        detail = (
                            'staleness-exempt'
                            if max_days is None
                            else f'{age_days:.1f}d < {max_days}d'
                        )
                        console.print(
                            f'  [dim]Index - reusing SCIP for {source_name} '
                            f'({detail}); pass --force to re-index[/dim]',
                        )
                    continue

        manifest_path = source_root / '.ariadne' / 'manifest.json'
        if not manifest_path.exists():
            console.print(
                f'[red]No manifest at {manifest_path} — run '
                f'`ariadne discover --source {source_name}` first[/red]',
            )
            return 1

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        intermediates: list[Path] = []

        # Select the entries this invocation will touch (honor --kind and
        # skip kinds with no registered adapter).
        selected_entries = []
        for entry in manifest.get('indexers', []):
            kind = entry.get('kind')
            if args.kind and args.kind != kind:
                continue
            if indexer_registry.get(kind) is None:
                console.print(
                    f'[yellow]No adapter registered for kind={kind!r} '
                    f'— skipping[/yellow]',
                )
                continue
            selected_entries.append(entry)

        excluded_dir_names = frozenset(
            cfg.resolve_excluded_dirs(source_name),
        )

        def _entry_path(entry) -> Path:
            return (source_root / entry.get('cwd', '.')).resolve()

        # File counts size + order the per-language bars. Count once so
        # the volume sort and the bar totals agree, and we don't walk
        # each tree twice.
        entry_counts = {
            id(e): _count_source_files(
                _entry_path(e), e.get('kind'), excluded_dir_names,
            )
            for e in selected_entries
        }
        # Group by language, smallest-volume language first (so the quick
        # languages finish before the heavy JVM compile).
        plan = _plan_indexing(
            selected_entries, lambda e: entry_counts[id(e)],
        )

        if args.dry_run:
            for kind, kind_entries, total in plan:
                for entry in kind_entries:
                    output = (
                        source_root / '.ariadne' / 'intermediate'
                        / f'index-{_scope_label(entry, kind)}.scip'
                    )
                    console.print(
                        f'[bold]Would run {kind} adapter:[/bold] '
                        f'cwd={_entry_path(entry)} → {output} '
                        f'({entry_counts[id(entry)]} files)',
                    )
            continue

        # The source's effective exclusion set from ariadne.yaml — the
        # same set catalog-sync / discovery use. Directory names become
        # ``**/<name>`` globs so pyright matches them at any depth.
        user_excludes: list[str] = []
        for d in cfg.resolve_excluded_dirs(source_name):
            user_excludes.append(f'**/{d}')
        user_excludes.extend(sc.exclude)

        quiet = getattr(args, 'quiet', False)
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress_columns = (
            SpinnerColumn(),
            TextColumn('[bold cyan]{task.description}'),
            BarColumn(),
            TextColumn('[dim]{task.fields[detail]}'),
            TextColumn('·'),
            TimeElapsedColumn(),
        )

        # One Progress spanning all languages; one task (bar) per
        # language. Python advances smoothly via scip-python's per-file
        # stream; TypeScript/Java are opaque subprocesses, so their bar
        # bumps by a whole scope's file count on each scope's completion.
        with Progress(
            *progress_columns, console=console, transient=quiet,
        ) as progress:
            import time as _time

            for kind, kind_entries, kind_total in plan:
                label = _LANGUAGE_LABELS.get(kind, str(kind).title())
                lang_started = _time.monotonic()
                streams = _streams_file_progress(kind)
                pulse = _pulse_bar(kind)
                # Python streams per-file progress and TypeScript/JS bump
                # per scope — both get a determinate ``X/N files`` bar.
                # scip-java compiles in one progress-less shot, so it gets
                # an animated indeterminate (pulse) bar showing just ``N
                # files`` (a 0/N counter would sit frozen for minutes).
                task_id = progress.add_task(
                    f'  {label}',
                    total=None if pulse else (kind_total or None),
                    detail=_index_detail_text(kind_total, 0, pulse=pulse),
                )
                base = 0  # files completed by prior scopes of this lang

                for entry in kind_entries:
                    cwd = _entry_path(entry)
                    scope = _scope_label(entry, kind)
                    output = (
                        source_root / '.ariadne' / 'intermediate'
                        / f'index-{scope}.scip'
                    )
                    entry_count = entry_counts[id(entry)]
                    adapter = indexer_registry.get(kind)

                    run_kwargs: dict = {
                        'cwd': cwd,
                        'output': output,
                        'env_hints': dict(sc.env_hints),
                    }
                    if kind == 'python':
                        run_kwargs['entry_kind'] = entry.get(
                            'entry_kind', 'package',
                        )
                        run_kwargs['excludes'] = tuple(user_excludes)

                        def on_progress(
                            event, _base=base, _count=entry_count,
                        ) -> None:
                            if event.kind == 'tick' and event.total:
                                frac = min(
                                    1.0, event.current / event.total,
                                )
                                done = int(_base + frac * _count)
                                progress.update(
                                    task_id,
                                    completed=done,
                                    detail=_index_detail_text(
                                        kind_total, done, pulse=False,
                                    ),
                                )
                            elif event.kind == 'warning' and not quiet:
                                progress.console.print(
                                    f'[yellow]{event.text}[/yellow]',
                                )
                            # 'message'/'total' kinds: silenced — we own
                            # the totals via our own file count.

                        run_kwargs['progress_callback'] = on_progress

                    result = adapter.run(**run_kwargs)

                    if not result.success:
                        # Name the failing scope — a language can have
                        # many, and quiet mode hides per-scope detail.
                        progress.console.print(
                            f'[red]{kind} adapter failed (scope={scope}, '
                            f'cwd={cwd}): {result.error_message}[/red]',
                        )
                        return 1

                    if (
                        kind == 'python'
                        and result.resolved_interpreter is not None
                        and result.resolution_source
                        and not quiet
                    ):
                        progress.console.print(
                            f'  Python interpreter: '
                            f'{result.resolved_interpreter} '
                            f'(source: {result.resolution_source})',
                        )

                    base += entry_count
                    if not pulse:
                        # Determinate bars (Python, TypeScript/JS) advance
                        # to the running file total after each scope.
                        progress.update(
                            task_id,
                            completed=base,
                            detail=_index_detail_text(
                                kind_total, base, pulse=False,
                            ),
                        )

                    entry['scip_path'] = str(
                        output.relative_to(source_root / '.ariadne'),
                    )
                    entry['indexed_at'] = datetime.now(
                        timezone.utc,
                    ).isoformat()
                    entry['indexer_version'] = result.indexer_version
                    # Record the Vue companion→.vue mapping so the loader
                    # (and resolve_index for catalog extraction) can
                    # translate SCIP positions back to the .vue files.
                    if result.vue_mapping_path:
                        entry['vue_mapping'] = str(
                            Path(result.vue_mapping_path).relative_to(
                                source_root / '.ariadne',
                            ),
                        )
                    intermediates.append(output)

                # Finish as a full bar so a completed language reads as
                # done. For the pulse (Java) bar we set a real total now
                # to fill it, but keep the "N files" detail (no X/N).
                if kind_total:
                    progress.update(
                        task_id,
                        total=kind_total,
                        completed=kind_total,
                        detail=_index_detail_text(
                            kind_total, kind_total, pulse=pulse,
                        ),
                    )

                # Record a per-language summary so phase callers
                # (dry-run / onboard) can render it nested under their
                # "✓ Index" line.
                if phase_summary is not None:
                    phase_summary.append({
                        'language': label,
                        'files': kind_total,
                        'seconds': _time.monotonic() - lang_started,
                    })

        if intermediates:
            final = source_root / '.ariadne' / 'index.scip'
            if len(intermediates) == 1:
                # Single-language project (or a multi-language project
                # where only one indexer ran this invocation): no merge
                # needed. Copy the lone intermediate directly so the
                # external ``scip`` CLI isn't a dependency for the
                # common single-source case.
                import shutil
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(intermediates[0], final)
            else:
                ok = merger.merge(intermediates, final)
                if not ok:
                    # _SubprocessMerger already printed the diagnostic.
                    return 1
            manifest['merged_at'] = datetime.now(
                timezone.utc,
            ).isoformat()
            manifest['merged_scip'] = 'index.scip'
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding='utf-8',
            )
            if not getattr(args, 'quiet', False):
                console.print(f'[green]Wrote {final}[/green]')

    # End-of-index step: persist the cross-source graph into
    # ``library_scip`` so downstream readers (ariadne callers /
    # impact_radius / dead-code, plus the catalog generator's
    # architecture-prompt Dependents section) see the fresh data.
    # Walks every configured source — cross-source edges only resolve
    # when both endpoints sit in the same materialized graph.
    if not args.dry_run:
        from docgen.scip_persist import (
            persist_akka_http_endpoints,
            persist_all_sources,
            persist_api_endpoints,
            persist_config_reads,
            persist_config_values,
            persist_express_routes,
            persist_js_http_clients,
            persist_python_http_clients,
            persist_python_routes,
            persist_scala_http_clients,
            persist_string_literals,
            persist_url_resolver,
        )

        source_pairs: list[tuple[str, Path]] = []
        swagger_pairs: list[tuple[str, Path, list[str]]] = []
        for name in cfg.sources:
            other_sc = cfg.get_source_config(name)
            if other_sc is None:
                continue
            other_root = Path(other_sc.path).expanduser().resolve()
            source_pairs.append((name, other_root))
            if other_sc.swagger_paths:
                swagger_pairs.append((
                    name, other_root, list(other_sc.swagger_paths),
                ))

        persisted = persist_all_sources(Path(cfg.db_path), source_pairs)
        if persisted and not getattr(args, 'quiet', False):
            console.print(
                f'[green]Persisted cross-source graph '
                f'({persisted} source(s)) → library_scip[/green]',
            )

        # Wave 4 Tier 2 — Swagger / OpenAPI ingestion. Populates
        # ``api_endpoints`` so ``ariadne_trace_flow``'s HTTP-tier
        # join (api_calls → api_endpoints) actually returns rows
        # once the client side is wired in a follow-up commit.
        if swagger_pairs:
            endpoints = persist_api_endpoints(
                Path(cfg.db_path), swagger_pairs,
            )
            if endpoints and not getattr(args, 'quiet', False):
                console.print(
                    f'[green]Ingested {endpoints} API endpoint(s) '
                    f'from Swagger specs → library_scip[/green]',
                )

        # Layer C prerequisite — populate ``string_literals``. The
        # route extractors below look up literal path values by SCIP
        # position; without this, every literal-arg endpoint silently
        # vanishes from the route walks.
        literals = persist_string_literals(Path(cfg.db_path), source_pairs)
        if literals and not getattr(args, 'quiet', False):
            console.print(
                f'[green]Indexed {literals} string literal(s) → '
                f'library_scip[/green]',
            )

        # Phase 2q — populate ``config_values`` (HOCON / YAML / dotenv
        # key->value) so Layer C can resolve config-getter arguments to
        # their configured values. Previously never wired, so the table
        # stayed empty.
        config_vals = persist_config_values(Path(cfg.db_path), source_pairs)
        if config_vals and not getattr(args, 'quiet', False):
            console.print(
                f'[green]Indexed {config_vals} config value(s) → '
                f'library_scip[/green]',
            )
        # Config↔code bridge (Tier 2) — enumerate config-getter call
        # sites into ``config_reads``. Needs ``string_literals`` and
        # ``config_values`` (both persisted just above).
        config_reads = persist_config_reads(Path(cfg.db_path), source_pairs)
        if config_reads and not getattr(args, 'quiet', False):
            console.print(
                f'[green]Indexed {config_reads} config read(s) → '
                f'library_scip[/green]',
            )

        # Wave 4 Tier 2 step 2 — Akka HTTP route extraction. Walks
        # SCIP-classified call sites for ``path`` / ``pathPrefix`` /
        # HTTP-verb combinators and persists detected routes to
        # ``api_endpoints`` with ``resolution_source='pattern'``
        # (preserves Swagger rows).
        quiet = getattr(args, 'quiet', False)
        akka_routes = persist_akka_http_endpoints(
            Path(cfg.db_path), source_pairs,
        )
        if akka_routes and not quiet:
            console.print(
                f'[green]Extracted {akka_routes} Akka HTTP route(s) '
                f'→ api_endpoints[/green]',
            )

        # Wave 4 Tier 2 step 3 — Flask / FastAPI route extraction.
        # Walks SCIP-classified ``@app.route`` / ``@app.<verb>`` /
        # ``@router.<verb>`` decorators and persists matched routes
        # to ``api_endpoints``. Same coexistence semantics as Akka.
        python_routes = persist_python_routes(
            Path(cfg.db_path), source_pairs,
        )
        if python_routes and not quiet:
            console.print(
                f'[green]Extracted {python_routes} Flask/FastAPI '
                f'route(s) → api_endpoints[/green]',
            )

        # Wave 4 Tier 2 step 4 — Express / Koa route extraction.
        # Walks ``app.<verb>(path, handler)`` and
        # ``router.<verb>(...)`` call sites and persists matched
        # routes to ``api_endpoints``.
        express_routes = persist_express_routes(
            Path(cfg.db_path), source_pairs,
        )
        if express_routes and not quiet:
            console.print(
                f'[green]Extracted {express_routes} Express/Koa '
                f'route(s) → api_endpoints[/green]',
            )

        # Wave 4 Tier 4 step 1 — Python HTTP client extraction.
        # Walks ``httpx.{verb}`` / ``requests.{verb}`` /
        # ``urllib.urlopen`` call sites and persists raw URL strings
        # to ``http_client_calls``. URL→endpoint joining (Phase 8c)
        # waits until all three client extractors are wired.
        py_http = persist_python_http_clients(
            Path(cfg.db_path), source_pairs,
        )
        if py_http and not quiet:
            console.print(
                f'[green]Extracted {py_http} Python HTTP call(s) '
                f'→ http_client_calls[/green]',
            )

        # Wave 4 Tier 4 step 2 — JS / TS HTTP client extraction.
        # Walks ``fetch(...)`` / ``axios.{verb}`` /
        # ``this.$http.{verb}`` (Vue 2 / Angular) call sites.
        js_http = persist_js_http_clients(
            Path(cfg.db_path), source_pairs,
        )
        if js_http and not quiet:
            console.print(
                f'[green]Extracted {js_http} JS/TS HTTP call(s) '
                f'→ http_client_calls[/green]',
            )

        # Wave 4 Tier 4 step 3 — Scala HTTP client extraction.
        # Walks Akka-HTTP ``Http().singleRequest`` / sttp
        # ``basicRequest.<verb>`` call sites.
        scala_http = persist_scala_http_clients(
            Path(cfg.db_path), source_pairs,
        )
        if scala_http and not quiet:
            console.print(
                f'[green]Extracted {scala_http} Scala HTTP call(s) '
                f'→ http_client_calls[/green]',
            )

        # Wave 4 Phase 8c — URL→endpoint resolution. The closing
        # step: matches client URLs (http_client_calls) against
        # server templates (api_endpoints), writes resolved edges
        # to ``api_calls``. Once this lands, ariadne_trace_flow's
        # HTTP-tier hops finally return cross-language chains.
        resolved = persist_url_resolver(
            Path(cfg.db_path), source_pairs,
        )
        if resolved and not quiet:
            console.print(
                f'[green]Resolved {resolved} URL→endpoint edge(s) '
                f'→ api_calls[/green]',
            )

    return 0


def _ensure_gitignore_entry(source_path: Path, line: str) -> None:
    """Append ``line`` to ``<source_path>/.gitignore`` if not already
    present. Creates the file if it doesn't exist. Idempotent."""
    gitignore = source_path / '.gitignore'
    if gitignore.exists():
        existing = gitignore.read_text(encoding='utf-8')
        if line in existing.splitlines():
            return
        with gitignore.open('a', encoding='utf-8') as f:
            if existing and not existing.endswith('\n'):
                f.write('\n')
            f.write(line + '\n')
    else:
        gitignore.write_text(line + '\n', encoding='utf-8')


def cmd_discover(args: argparse.Namespace) -> int:
    """Walk a source tree, detect SCIP indexer locations, write
    ``<source>/.ariadne/manifest.json``, AND auto-author the SCIP
    fields in ``ariadne.yaml``.

    UX principle: users author only ``path``, ``depends_on``,
    ``exclude``, ``exclude_dirs`` — Ariadne fills the rest.
    ``cmd_discover`` writes ``index_kinds`` (one entry per detected
    SCIP-routable language: ``javascript`` for any TS/JS file,
    ``scala``/``java`` for JVM sources) and the ``scip:`` block
    pointing at ``<source>/.ariadne/index.scip`` and a default
    ``max_staleness_days``. ``cmd_sync`` later detects new languages
    in changed files and triggers ``discover --config-only`` to keep
    these fields current automatically.

    Re-runs are idempotent — ``write_source_scip_config`` returns
    False (no rewrite) when the YAML already matches the detection
    state. Manual edits to the auto-managed block are regenerated
    on the next ``discover``.

    Adds ``.ariadne/`` to ``<source>/.gitignore`` so the artifact
    directory doesn't accidentally get committed.

    Per design decision #1 (artifacts gitignored, per-machine) and #6
    (auto-discovery via marker files: ``__init__.py``, ``package.json``,
    ``build.sbt``/``pom.xml``/``build.gradle*``).
    """
    import json

    from config import DEFAULT_EXCLUDE_POLICY
    from docgen.scip_discovery import discover

    cfg = get_config()

    if args.all:
        sources_to_process = list(cfg.sources.keys())
        if not sources_to_process:
            console.print(
                '[red]No sources configured in ariadne.yaml[/red]',
            )
            return 1
    elif args.source:
        sources_to_process = [args.source]
    else:
        if cfg.default_source:
            sources_to_process = [cfg.default_source]
        else:
            console.print(
                '[red]No source specified — use --source X, --all, or '
                'set default_source in ariadne.yaml[/red]',
            )
            return 1

    global_excl = set(
        cfg._config.get('exclude_policy') or DEFAULT_EXCLUDE_POLICY,
    )

    for source_name in sources_to_process:
        sc = cfg.get_source_config(source_name)
        if sc is None:
            console.print(
                f'[red]Source not found in config: {source_name}[/red]',
            )
            return 1

        source_path = Path(sc.path).expanduser().resolve()
        if not source_path.exists():
            console.print(
                f'[red]Source path does not exist: {source_path}[/red]',
            )
            return 1

        # Effective exclude set: global policy + source-level extras,
        # minus exempt-dirs.
        effective_excl = (
            global_excl | set(sc.exclude_dirs)
        ) - set(sc.exempt_dirs)

        # Per-source file-glob exclusions (e.g., '**/*.min.js').
        exclude_patterns = frozenset(sc.exclude or ())

        entries = discover(
            source_path,
            exclude_dirs=frozenset(effective_excl),
            exempt_dirs=frozenset(sc.exempt_dirs),
            exclude_patterns=exclude_patterns,
        )

        # --review: classify suspects, prompt user, persist decisions to
        # ariadne.yaml, and filter rejected entries from the manifest.
        if getattr(args, 'review', False) and entries:
            from docgen.scip_review import (
                classify_suspects,
                prompt_keep_entry,
            )
            from docgen.yaml_writer import append_source_excludes

            suspects = classify_suspects(
                entries, source_root=source_path,
            )
            if suspects:
                console.print(
                    f'\n[bold]Reviewing {len(suspects)} suspect '
                    f'entry/entries for {source_name}[/bold]',
                )
                rejected_entry_ids: set[int] = set()
                rejected_patterns: list[str] = []
                for suspect in suspects:
                    keep = prompt_keep_entry(suspect)
                    if not keep:
                        rejected_entry_ids.add(id(suspect.entry))
                        rejected_patterns.append(
                            suspect.suggested_pattern,
                        )

                if rejected_patterns:
                    entries = [
                        e for e in entries
                        if id(e) not in rejected_entry_ids
                    ]
                    config_path = cfg.config_path
                    if config_path is not None:
                        try:
                            added = append_source_excludes(
                                Path(config_path),
                                source_name,
                                rejected_patterns,
                            )
                            console.print(
                                f'[green]Added {added} exclude '
                                f'pattern(s) to {config_path}[/green]',
                            )
                        except (FileNotFoundError, KeyError) as e:
                            console.print(
                                f'[yellow]Could not update yaml: '
                                f'{e}[/yellow]',
                            )
                    else:
                        console.print(
                            '[yellow]No ariadne.yaml path resolved; '
                            'patterns not persisted. Add manually:'
                            '[/yellow]',
                        )
                        for p in rejected_patterns:
                            console.print(f'  - {p!r}')

        def _rel(p: Path) -> str:
            try:
                rel = p.relative_to(source_path)
                return '.' if str(rel) == '.' else str(rel)
            except ValueError:
                return str(p)

        manifest = {
            'ariadne_version': '1',
            'source_name': source_name,
            'indexers': [
                {
                    'kind': e.kind,
                    'cwd': _rel(e.cwd),
                    'markers': [_rel(m) for m in e.markers],
                    # Phase 2j.b: 'package' (has __init__.py / package.json
                    # / build file) vs 'scripts' (orphan dir). The Python
                    # adapter (Phase 2n) dispatches the transient
                    # pyrightconfig include pattern on this field.
                    'entry_kind': e.entry_kind,
                }
                for e in entries
            ],
        }

        if args.dry_run:
            console.print(
                f'[bold]Manifest for {source_name} (dry-run):[/bold]',
            )
            console.print(json.dumps(manifest, indent=2))
            continue

        manifest_dir = source_path / '.ariadne'
        manifest_dir.mkdir(exist_ok=True)
        # Pre-create the staging dir indexers write into. Each adapter's
        # underlying tool (e.g., scip-python) opens the output file
        # directly without mkdir-p, so without this the first
        # ``ariadne index`` run hits ENOENT. Creating it during discovery
        # — alongside ``.ariadne/`` itself — keeps the responsibility for
        # layout setup in one place.
        (manifest_dir / 'intermediate').mkdir(exist_ok=True)
        manifest_path = manifest_dir / 'manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding='utf-8',
        )

        _ensure_gitignore_entry(source_path, '.ariadne/')

        if entries:
            table = Table(
                title=f'Discovered indexers for {source_name}',
            )
            table.add_column('Kind')
            table.add_column('cwd')
            table.add_column('Markers')
            for e in entries:
                table.add_row(
                    e.kind,
                    _rel(e.cwd),
                    ', '.join(_rel(m) for m in e.markers),
                )
            console.print(table)
        else:
            console.print(
                f'[yellow]No indexer-relevant clusters found in '
                f'{source_path}[/yellow]',
            )
        console.print(f'[green]Wrote {manifest_path}[/green]')

        # Auto-author the SCIP-related fields in ariadne.yaml so the
        # user only ever has to author path/depends_on/exclude/exclude_dirs.
        # Mapping: discovery kind → language(s) whose catalog extraction
        # routes through SCIP today (extract_elements supports
        # scala/java/javascript SCIP routing). Python is detected and
        # cross-source-indexed but its catalog stays on ast-grep — the
        # qualified_names already align with scip-python's, so no
        # routing layer is needed.
        catalog_scip_languages: set[str] = set()
        for e in entries:
            if e.kind == 'typescript':
                catalog_scip_languages.add('javascript')
            elif e.kind == 'java':
                # scip-java covers both Scala and Java in one indexer
                # output. Declare both index_kinds so catalog extraction
                # routes either file extension through scip_extractor.
                catalog_scip_languages.add('scala')
                catalog_scip_languages.add('java')

        config_path = cfg.config_path
        if catalog_scip_languages and config_path is not None:
            from docgen.yaml_writer import write_source_scip_config
            artifact_path = manifest_dir / 'index.scip'
            try:
                rewrote = write_source_scip_config(
                    Path(config_path),
                    source_name,
                    catalog_scip_languages=catalog_scip_languages,
                    artifact_path=artifact_path,
                )
                if rewrote:
                    console.print(
                        f'[green]Updated {config_path} '
                        f'(index_kinds: {sorted(catalog_scip_languages)})'
                        f'[/green]',
                    )
            except (FileNotFoundError, KeyError) as exc:
                console.print(
                    f'[yellow]Could not update yaml: {exc}[/yellow]',
                )

    return 0


HANDLERS = {
    'discover': lambda args: cmd_discover(args),
    'index': lambda args: cmd_index(args),
}
