"""``ariadne spools`` — Spool status, pack build, pack install.

Bare ``ariadne spools`` shows each enabled Spool's resolution: registered
(with the pack's target runtime) or a structured gap (missing pack /
runtime mismatch), exit 1 iff gaps exist so fail-closed stays loud in
scripts and CI. ``spools build`` packages a source into a pack zip;
``spools install`` verifies + installs one (checksum-gated, no re-embed).
Design: designs/spool-environment-plugin.md §17 · §18.2 · §18.6.
"""
import argparse
from pathlib import Path

from config import get_config
from spools import default_cache_dir, enabled_spools, resolve_spools


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``spools`` command with its subactions."""
    spools_parser = subparsers.add_parser(
        'spools',
        help='Spools: status / init / create / install / enable / disable / '
             'reconcile',
    )
    spools_parser.add_argument(
        '--cache-dir',
        default=None,
        help='Pack cache directory (default: <config dir>/.ariadne/spools)',
    )
    spools_sub = spools_parser.add_subparsers(
        dest='spools_action',
        help='Spools subcommand (default: status)',
    )

    build = spools_sub.add_parser(
        'build', help='Package a source into a Spool pack zip',
    )
    build.add_argument('--source', required=True,
        help='Source name = the pack environment id')
    build.add_argument('--version', required=True,
        help='Pack version stamp')
    build.add_argument('--runtime', required=True,
        help='Target runtime edition the pack is built for (the pin)')
    build.add_argument('--certify', action='append', default=[],
        metavar='DIR',
        help='Official-docs dir relative to the source root '
             '(repeatable) — docs under it are tagged provenance=official')
    build.add_argument('--out', required=True,
        help='Output pack zip path')

    install = spools_sub.add_parser(
        'install', help='Verify and install a Spool pack zip',
    )
    install.add_argument('pack', help='Path to the pack zip')

    acquire_parser = spools_sub.add_parser(
        'acquire',
        help='Fetch a pack build corpus (consent-gated clone-at-SHA)',
    )
    acquire_parser.add_argument('packfile', help='Path to the packfile YAML')
    acquire_parser.add_argument('--dest', required=True,
        help='Directory to clone the corpus repos into')
    acquire_parser.add_argument('--approve', action='store_true',
        help='Skip the consent prompt (CI); default asks before fetching')
    
    create_parser = spools_sub.add_parser(
        'create',
        help='Set up the spool recipe (which spool + versions) and build '
             'it: consent, fetch, index, onboard (cost prompt), pack',
    )
    create_parser.add_argument('environment', nargs='?', default=None,
        help="Spool to set up (e.g. 'databricks'); omit to be prompted")
    _mode = create_parser.add_mutually_exclusive_group()
    _mode.add_argument('--live', dest='batch_mode', action='store_const',
        const='live',
        help='Onboard embeddings live (full price); skips the live/batch toggle')
    _mode.add_argument('--batch', dest='batch_mode', action='store_const',
        const='batch',
        help='Onboard embeddings via the Batch API (~half price); skips the live/batch toggle')
    create_parser.set_defaults(batch_mode=None)
    create_parser.add_argument('--yes', '-y', action='store_true',
        help='Non-interactive: skip setup + prompts and build the existing spools.yaml (pair with --batch/--live for a fully unattended run)')
    create_parser.add_argument('--resume', action='store_true',
        help='Resume an interrupted build: reuse the existing spools.yaml + '
             'already-fetched corpus (skip setup), then continue — completed '
             'docs are skipped via staleness, so only the unfinished work '
             'runs. Shows the (now-smaller) cost preview unless paired with '
             '--yes.')
    create_parser.add_argument('--dest', default='spool-corpus',
        help='Corpus checkout directory (default: ./spool-corpus)')
    create_parser.add_argument('--out', default=None,
        help='Pack zip path (default: <name>-<runtime>.zip)')
    create_parser.add_argument('--allow-ungrounded', action='store_true',
        default=False,
        help="Build even when the corpus language has no SCIP indexer "
             "(e.g. Go) — a docs-only pack with no code-tier grounding. "
             "Default False: create REFUSES an ungrounded corpus.")

    enable_parser = spools_sub.add_parser(
        'enable',
        help='Cross-check a project against a spool (reconcile its themes)',
    )
    enable_parser.add_argument('spool', help='Enabled spool name')
    enable_parser.add_argument('--project', required=True,
        help='Project (configured source) to cross-check against the spool')

    disable_parser = spools_sub.add_parser(
        'disable',
        help='Stop cross-checking a project against a spool (delete its themes)',
    )
    disable_parser.add_argument('spool', help='Enabled spool name')
    disable_parser.add_argument('--project', required=True,
        help='Project to stop cross-checking against the spool')

    reconcile_parser = spools_sub.add_parser(
        'reconcile',
        help='Refresh cross-source themes for enabled spools (after base '
             'changes)',
    )
    reconcile_parser.add_argument('--spool', default=None,
        help='Reconcile only this spool (default: all enabled)')


def cmd_spools(args: argparse.Namespace) -> int:
    """Dispatch the spools subaction (default: status)."""
    actions = {
        'create': _create,
        'build': _build,
        'install': _install,
        'acquire': _acquire,
        'enable': _enable,
        'disable': _disable,
        'reconcile': _reconcile,
    }
    return actions.get(getattr(args, 'spools_action', None), _status)(args)


def _cache_dir(args, config) -> Path:
    cache = getattr(args, 'cache_dir', None)
    return Path(cache) if cache else default_cache_dir(config)


def _gaps_by_spool(args, config) -> dict:
    """Structured resolution gaps keyed by spool name (empty → all ready)."""
    return {g.spool: g for g in
            resolve_spools(config, cache_dir=_cache_dir(args, config)).gaps}


def _status(args: argparse.Namespace) -> int:
    """Print the spool resolution for the current project."""
    config = get_config()
    if not enabled_spools(config):
        print('No spools enabled (add a `spools:` mapping to ariadne.yaml).')
        return 0
    from docgen.extraction_coverage import EXTRACTION_COVERAGE_VERSION
    resolution = resolve_spools(config, cache_dir=_cache_dir(args, config))
    for name, registration in sorted(resolution.registered.items()):
        manifest = registration.manifest
        print(
            f'  registered  {name}  (kind={registration.kind}, '
            f'runtime {manifest.target_runtime}, version {manifest.version})'
        )
        # Advisory (not a gap — the pack is still usable, so this doesn't
        # affect the exit code): its SCIP intelligence predates a coverage
        # change, so a rebuild would refresh it.
        if manifest.extraction_coverage_version < EXTRACTION_COVERAGE_VERSION:
            print(
                f'              ⚠ built under older extraction coverage '
                f'(v{manifest.extraction_coverage_version}, current '
                f'v{EXTRACTION_COVERAGE_VERSION}) — rebuild the pack '
                f'(`ariadne spools create`) to refresh its SCIP intelligence.'
            )
    for gap in resolution.gaps:
        print(f'  gap         {gap.spool}  [{gap.reason}] {gap.message}')
    return 1 if resolution.gaps else 0


def _build(args: argparse.Namespace) -> int:
    """Package ``--source`` into a pack zip (fails loud on misconfig)."""
    from library import Library
    from spool_pack import build_pack

    config = get_config()
    source_root = config.get_source_path(args.source)
    if source_root is None:
        print(f"Error: source '{args.source}' has no configured path — "
              f'add it via `ariadne source add` first.')
        return 1
    with Library(config.db_path) as library:
        manifest = build_pack(
            library,
            environment=args.source,
            version=args.version,
            target_runtime=args.runtime,
            certified_docs=tuple(args.certify),
            source_root=source_root,
            out_path=args.out,
        )
    print(f'  built  {args.out}  (environment {manifest.environment}, '
          f'runtime {manifest.target_runtime}, {manifest.checksum})')
    return 0


def _install(args: argparse.Namespace) -> int:
    """Verify + install a pack zip into the store and the cache."""
    from library import Library
    from spool_pack import install_pack

    config = get_config()
    with Library(config.db_path) as library:
        manifest = install_pack(
            library, args.pack, cache_dir=_cache_dir(args, config),
        )
    print(f'  installed  {manifest.environment}  '
          f'(runtime {manifest.target_runtime}, version {manifest.version})')
    return 0


def _acquire(args: argparse.Namespace) -> int:
    """Consent-gated corpus fetch; prints the follow-on build steps."""
    from spool_acquire import acquire, load_packfile

    packfile = load_packfile(args.packfile)
    result = acquire(packfile, dest_dir=args.dest, approve=args.approve)
    if not result.accepted:
        print('Acquisition declined — nothing fetched.')
        return 1
    for name in result.cloned:
        print(f'  acquired  {name}  -> {args.dest}/{name}')
    print(
        'Next steps: `ariadne source add` each corpus repo, '
        '`ariadne discover` + `ariadne onboard` (cost preview there), '
        'then `ariadne spools build`.'
    )
    return 0


def _create(args: argparse.Namespace) -> int:
    """Unified `spools create`: set up the recipe (which spool + each
    version) then build it. ``--yes`` skips the interactive setup and
    builds an existing ./spools.yaml unattended (pair with --batch/--live).
    """
    from spool_acquire import _load_spoolfile, create_spool, setup_recipe

    spoolfile = 'spools.yaml'
    resume = getattr(args, 'resume', False)
    if resume or args.yes:
        if not Path(spoolfile).exists():
            flag = '--resume' if resume else '--yes'
            print(f'Error: {flag} needs an existing {spoolfile}; run `ariadne spools create` without {flag} to set one up.')
            return 1
        if resume:
            print(f'Resuming build from {spoolfile} — completed work is skipped (staleness); only the unfinished phases run.')
    else:
        setup_recipe(args.environment, out_path=spoolfile)

    name, _version, data = _load_spoolfile(spoolfile)
    out = args.out or f"{name}-{data['runtime']}.zip"
    result = create_spool(
        spoolfile, dest_dir=args.dest, out_path=out, approve=args.yes,
        allow_ungrounded=args.allow_ungrounded, batch_mode=args.batch_mode,
        resume=resume,
    )
    if not result.accepted:
        print('Creation declined — nothing fetched.')
        return 1
    print(f'  spool pack built: {result.pack_path}')
    print(f'Install it with: `ariadne spools install {result.pack_path}`')
    return 0


def _enable(args: argparse.Namespace) -> int:
    """Add a project to a spool's cross-check set and reconcile its themes."""
    from library import Library
    from spools import reconcile_spool_themes, resolve_spools

    config = get_config()
    setting = enabled_spools(config).get(args.spool)
    if setting is None:
        print(f"Error: spool '{args.spool}' is not enabled — add it to "
              f'ariadne.yaml first '
              f'(e.g. `spools: {{{args.spool}: {{runtime: <edition>}}}}`).')
        return 1
    projects = sorted(set(setting.projects) | {args.project})
    config.set_spool_projects(args.spool, projects)
    config = get_config()
    print(f'  enabled  {args.project} × {args.spool}')

    # MEDIUM-2: reconcile only if the spool is actually registered; otherwise
    # surface WHY (not installed / unpinned / mismatched) rather than silently
    # build nothing.
    gaps = _gaps_by_spool(args, config)
    if args.spool in gaps:
        print(f'  Spool not ready — no cross-source themes built: '
              f'{gaps[args.spool].message}')
        print(f'  Fix that, then: `ariadne spools reconcile --spool {args.spool}`.')
        return 0

    with Library(getattr(args, 'db', None) or config.db_path) as library:
        result = reconcile_spool_themes(library, config, args.spool)
    if result.dirty_theme_count:
        print(f'  {result.dirty_theme_count} cross-source theme(s) pending '
              f'summarization — run `ariadne themes build` to summarize them '
              f'(it shows the LLM cost and prompts before spending).')
    else:
        print('  No cross-source themes surfaced for this project + spool.')
    return 0


def _disable(args: argparse.Namespace) -> int:
    """Remove a project from a spool's cross-check set and delete its themes."""
    from library import Library
    from spools import reconcile_spool_themes

    config = get_config()
    setting = enabled_spools(config).get(args.spool)
    if setting is None:
        print(f"Error: spool '{args.spool}' is not enabled.")
        return 1
    projects = [p for p in setting.projects if p != args.project]
    config.set_spool_projects(args.spool, projects)
    with Library(getattr(args, 'db', None) or config.db_path) as library:
        result = reconcile_spool_themes(library, config, args.spool)
    if result.removed:
        print(f'  disabled  {args.project} × {args.spool}  '
              f'(deleted {len(result.removed)} theme partition(s))')
    else:
        print(f'  {args.project} was not cross-checked against '
              f'{args.spool} — nothing to remove.')
    return 0


def _reconcile(args: argparse.Namespace) -> int:
    """Refresh cross-source themes for enabled spools (after base changes).

    Re-clusters each enabled spool's opted-in projects, so themes track the
    current base corpus (their cross-source edges are transient and rebuilt
    here). Summarization stays gated behind `ariadne themes build`.
    """
    from library import Library
    from spools import reconcile_spool_themes, resolve_spools

    config = get_config()
    enabled = enabled_spools(config)
    if not enabled:
        print('No spools enabled (add a `spools:` mapping to ariadne.yaml).')
        return 0
    target = getattr(args, 'spool', None)
    if target is not None and target not in enabled:
        print(f"Error: spool '{target}' is not enabled.")
        return 1
    names = [target] if target else sorted(enabled)
    # MEDIUM-2: skip (loudly) any spool that isn't registered — reconciling a
    # not-installed / unpinned / mismatched spool would build nothing silently.
    gaps = _gaps_by_spool(args, config)
    total_dirty = 0
    with Library(getattr(args, 'db', None) or config.db_path) as library:
        for name in names:
            if name in gaps:
                print(f'  skipped  {name} — not ready: {gaps[name].message}')
                continue
            result = reconcile_spool_themes(library, config, name)
            total_dirty += result.dirty_theme_count
            print(f'  reconciled  {name}  '
                  f'({len(enabled[name].projects)} project(s), '
                  f'{result.dirty_theme_count} theme(s) pending)')
    if total_dirty:
        print(f'  Run `ariadne themes build` to summarize {total_dirty} '
              f'updated theme(s) (shows the LLM cost and prompts).')
    return 0


HANDLERS = {'spools': cmd_spools}
