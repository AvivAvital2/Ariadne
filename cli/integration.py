"""Integration and setup CLI commands."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def get_library(db_path=None):
    from cli.main import get_library as _get_library
    return _get_library(db_path)


def register_commands(subparsers):
    # manifest
    manifest_parser = subparsers.add_parser('manifest', help='Output filtered manifest for session hooks')
    manifest_parser.add_argument('--source', '-s', default=None,
                                  help='Source name (default from config)')
    manifest_parser.add_argument('--auto-scope', action='store_true',
                                  help='Auto-detect source scope based on cwd and branch')
    manifest_parser.add_argument('--branch', '-b', default=None,
                                  help='Override branch for filtering (default: auto-detect)')
    manifest_parser.add_argument('--no-branch-filter', action='store_true',
                                  help='Disable branch-based filtering')
    manifest_parser.add_argument('--limit', '-l', type=int, default=20,
                                  help='Max documents to show (default: 20)')

    # init
    init_parser = subparsers.add_parser('init', help='Initialize Ariadne integration for a project')
    init_parser.add_argument('--source', '-s', default=None,
                              help='Source name for docs path (default from config)')
    init_parser.add_argument('--target', '-t', default=None,
                              help='Target project directory (default: current directory)')
    init_parser.add_argument('--global', dest='global_mcp', action='store_true',
                              help='Register MCP server at user scope (available in all projects)')

    # config
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    config_parser.add_argument('--path', action='store_true',
                                help='Show only the config file path')

    # mcp
    mcp_parser = subparsers.add_parser('mcp', help='Start the MCP server (stdio transport)')
    mcp_parser.add_argument('--directory', help='Ariadne project directory (default: directory containing ariadne.yaml)')
    mcp_parser.add_argument('--http', action='store_true',
        help='Serve over streamable-http (remote) instead of stdio.')
    mcp_parser.add_argument('--host', default='127.0.0.1',
        help='Bind host for --http (default: 127.0.0.1)')
    mcp_parser.add_argument('--port', type=int, default=8000,
        help='Bind port for --http (default: 8000)')

    serve_parser = subparsers.add_parser(
        'serve', help='Start the web onboarding UI (connects to the MCP server)')
    serve_parser.add_argument(
        '--host', default='127.0.0.1', help='Bind host (default: 127.0.0.1)')
    serve_parser.add_argument(
        '--port', type=int, default=8765, help='Bind port (default: 8765)')
    serve_parser.add_argument('--mcp-url', default=None,
        help='Connect to a remote `ariadne mcp --http` server at this URL '
             'instead of spawning one over stdio.')

    # sync-claude-md (internal command for PostToolUse hook)
    sync_md_parser = subparsers.add_parser('sync-claude-md', help='Sync edited CLAUDE.md to Ariadne')
    sync_md_parser.add_argument('file', help='Path to the edited CLAUDE.md file')
    sync_md_parser.add_argument('--source', '-s', default=None,
                                 help='Source name (default from config)')

    # edit-instructions
    edit_parser = subparsers.add_parser('edit-instructions', help='Edit Ariadne CLAUDE.md in $EDITOR')
    edit_parser.add_argument('--source', '-s', default=None,
                              help='Source name (default from config)')

    # source — manage ariadne.yaml source entries from the CLI
    source_parser = subparsers.add_parser(
        'source', help='Add/list/remove sources in ariadne.yaml')
    source_sub = source_parser.add_subparsers(
        dest='source_action', help='Source subcommand')

    src_add = source_sub.add_parser(
        'add', help='Add or update a source entry in ariadne.yaml')
    src_add.add_argument('name', nargs='?', help='Source name')
    src_add.add_argument('--path', '-p', default=None,
                         help='Path to the source code directory')
    src_add.add_argument('--depends-on', default=None,
                         help='Comma-separated source names this depends on')
    src_add.add_argument('--parent', default=None,
                         help='Parent source name (for subdirectory sources)')
    src_add.add_argument('--branches', default=None,
                         help='Comma-separated git branch patterns where active')
    src_add.add_argument('--ref', default=None,
                         help='Pin the source to a specific git ref')
    src_add.add_argument('--exclude', default=None,
                         help='Comma-separated glob patterns to exclude')
    src_add.add_argument('--exclude-dirs', default=None,
                         help='Comma-separated directory names to exclude')
    src_add.add_argument('--ignore-staleness', action='store_true', default=None, help='Skip staleness checks for this source (opt-in)')
    src_add.add_argument('--skip-dependency-detection', action='store_true', default=None, help='Skip scanning this source for hidden cross-source dependencies (opt-in)')

    source_sub.add_parser('list', help='List configured sources')

    src_rm = source_sub.add_parser(
        'remove', help='Remove a source entry from ariadne.yaml')
    src_rm.add_argument('name', nargs='?', help='Source name to remove')
    src_rm.add_argument('--yes', '-y', action='store_true',
                        help='Skip the confirmation prompt')
    src_rm.add_argument('--purge', action='store_true',
                        help='Also delete the source data (documents, chunks, SCIP rows) from the library DB and rebuild the embedding matrix')
    src_purge = source_sub.add_parser(
        'purge',
        help="Delete a source's indexed data from the library DB "
             "(documents, chunks, SCIP tables) and rebuild the matrix — "
             "works even if the source is no longer in ariadne.yaml")
    src_purge.add_argument('name', nargs='?', help='Source name to purge')
    src_purge.add_argument('--yes', '-y', action='store_true',
                           help='Skip the confirmation prompt')


def cmd_manifest(args: argparse.Namespace) -> int:
    """Output a filtered manifest for session hooks.

    Supports branch-aware filtering and directory-scoped dependencies.
    """
    import fnmatch

    from config import get_config

    cfg = get_config()

    # Get current branch
    current_branch: str | None = None
    if args.branch:
        current_branch = args.branch
    elif not args.no_branch_filter:
        from git_ops import get_current_branch
        current_branch = get_current_branch() or ''

    # Determine source scope
    source_name: str | None = args.source
    scope_path: Path | None = None
    dependencies: list[str] = []

    if args.auto_scope:
        # Auto-detect source based on cwd and branch
        cwd = Path.cwd()
        source_name = cfg.get_source_scope(cwd, current_branch)
        if source_name:
            scope_path = cfg.get_source_path(source_name)
            dependencies = cfg.get_effective_dependencies(source_name, current_branch)
    elif not source_name:
        source_name = cfg.default_source
        if source_name:
            dependencies = cfg.get_effective_dependencies(source_name, current_branch)

    library = get_library(args.db)

    try:
        docs = library.list_documents()

        # Filter documents based on metadata
        filtered_docs: list = []
        for doc in docs:
            status = doc.metadata.get('status', 'stable')
            branches = doc.metadata.get('branches', [])

            # Include stable docs always
            if status in ('stable', None, ''):
                filtered_docs.append(doc)
                continue

            # For experimental/deprecated docs, check branch match
            if status == 'experimental' and current_branch and branches:
                # Check if current branch matches any pattern
                if isinstance(branches, list):
                    for pattern in branches:
                        if fnmatch.fnmatch(current_branch, pattern):
                            filtered_docs.append(doc)
                            break
                continue

            # If no branch filter or no branch patterns, exclude experimental
            if status != 'experimental':
                filtered_docs.append(doc)

        # Resolve conflicts (branch-specific docs win over base)
        source_precedence: list[str] | None = None
        if source_name:
            # Build precedence list: current source first, then dependencies
            source_precedence = [source_name] + dependencies
        filtered_docs = library.resolve_conflicts(
            filtered_docs,
            branch=current_branch,
            source_precedence=source_precedence,
        )

        # Output manifest header
        print('## Ariadne Knowledge Base Active')
        print()
        if source_name:
            print(f'Source: {source_name}')
        if current_branch:
            print(f'Branch: {current_branch}')
        if scope_path:
            print(f'Scope: {scope_path}')
        if dependencies:
            dep_strs = []
            for dep in dependencies:
                dep_config = cfg.get_source_config(dep)
                if dep_config and dep_config.ref:
                    dep_strs.append(f'{dep}@{dep_config.ref}')
                else:
                    dep_strs.append(dep)
            print(f'Dependencies: {", ".join(dep_strs)}')
        print('Documents:')

        # Group by type
        by_type: dict[str, list] = {}
        for doc in filtered_docs:
            ct = doc.content_type
            if ct not in by_type:
                by_type[ct] = []
            by_type[ct].append(doc)

        # Output document list (limited)
        count = 0
        max_docs = args.limit or 20
        for content_type in ['explanation', 'architecture', 'finding', 'qa', 'diagram']:
            type_docs = by_type.get(content_type, [])
            for doc in type_docs[:max_docs - count]:
                print(f'  - id: {doc.id}')
                print(f'    title: "{doc.title}"')
                if doc.metadata.get('status'):
                    print(f'    status: {doc.metadata["status"]}')
                count += 1
                if count >= max_docs:
                    break
            if count >= max_docs:
                break

        if len(filtered_docs) > max_docs:
            print(f'  ... and {len(filtered_docs) - max_docs} more')

        # Behavioral directive
        if cfg.mention_ariadne_enabled:
            print()
            print('## Behavioral Directive')
            print(cfg.mention_ariadne_message)

        # Usage tracking instructions
        print()
        print('## Usage Tracking')
        print('After calling an Ariadne tool, report whether it helped:')
        print('- If useful: call ariadne_log_hit(event_id) with optional feedback')
        print('- If not useful: call ariadne_log_miss(event_id, feedback="what was missing")')
        print('The event ID appears at the end of each tool result as [Usage event: <id>].')
        print('When asked about Ariadne coverage gaps, call ariadne_gaps to generate a miss report.')

        return 0

    finally:
        library.close()


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize Ariadne integration for a project.

    Creates:
    - .claude/settings.json with session hook
    - CLAUDE.md snippet (appended if exists)
    - MCP server config (--global for user scope, or .mcp.json per project)
    """
    import json as json_module

    from config import get_config

    cfg = get_config()
    source_name = args.source or cfg.default_source or 'myproject'
    ariadne_path = Path(__file__).parent.resolve()
    docs_path = cfg.resolve_docs_path(source_name) if source_name in cfg.sources else ariadne_path / 'docs' / source_name
    target_dir = Path(args.target) if args.target else Path.cwd()

    # Create .claude directory
    claude_dir = target_dir / '.claude'
    claude_dir.mkdir(parents=True, exist_ok=True)

    # Build session hook command - use --auto-scope for directory-aware filtering
    # Prefer bare `ariadne` (assumes PATH install via `uv tool install`),
    # fall back to `cd <path> && uv run ariadne` for development setups.
    hook_cmd = (
        f"ariadne manifest --auto-scope 2>/dev/null || "
        f"echo 'Ariadne manifest not available. Install: uv tool install {ariadne_path}'"
    )

    # Create or update settings.json
    settings_path = claude_dir / 'settings.json'
    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json_module.loads(settings_path.read_text())
        except json_module.JSONDecodeError:
            pass

    # Add hooks if not present
    if 'hooks' not in settings:
        settings['hooks'] = {}
    if 'SessionStart' not in settings['hooks']:
        settings['hooks']['SessionStart'] = []

    # Check if Ariadne hook already exists
    has_ariadne_hook = any(
        'Ariadne' in (h.get('hooks', [{}])[0].get('command', '') if h.get('hooks') else '')
        for h in settings['hooks']['SessionStart']
    )

    # Build PostToolUse hook command for syncing CLAUDE.md edits
    sync_cmd = f'ariadne sync-claude-md "$CLAUDE_FILE_PATH" --source {source_name} 2>/dev/null'

    # Add PostToolUse hook for CLAUDE.md sync
    if 'PostToolUse' not in settings['hooks']:
        settings['hooks']['PostToolUse'] = []

    # Check if sync hook already exists
    has_sync_hook = any(
        'sync-claude-md' in (h.get('hooks', [{}])[0].get('command', '') if h.get('hooks') else '')
        for h in settings['hooks']['PostToolUse']
    )

    hooks_updated = False

    if not has_ariadne_hook:
        settings['hooks']['SessionStart'].append({
            'matcher': 'startup|resume|clear',
            'hooks': [{'type': 'command', 'command': hook_cmd}]
        })
        hooks_updated = True

    if not has_sync_hook:
        settings['hooks']['PostToolUse'].append({
            'matcher': '(Edit|Write).*CLAUDE\\.md',
            'hooks': [{'type': 'command', 'command': sync_cmd}]
        })
        hooks_updated = True

    if hooks_updated:
        settings_path.write_text(json_module.dumps(settings, indent=2))
        console.print(f'[green]Created/updated {settings_path}[/green]')
    else:
        console.print(f'[yellow]Ariadne hooks already exist in {settings_path}[/yellow]')

    # Configure MCP server
    if args.global_mcp:
        # Register at user scope via claude CLI
        import shutil
        import subprocess

        claude_bin = shutil.which('claude')
        if not claude_bin:
            console.print('[red]claude CLI not found in PATH. Install Claude Code first.[/red]')
            return 1

        result = subprocess.run(
            [claude_bin, 'mcp', 'add', '-s', 'user', 'ariadne', '--',
             str(Path(shutil.which('uv') or 'uv')),
             'run', '--directory', str(ariadne_path), 'ariadne', 'mcp'],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            console.print('[green]Registered Ariadne MCP server at user scope (available in all projects)[/green]')
        else:
            console.print(f'[red]Failed to register MCP server: {result.stderr.strip()}[/red]')
            return 1
    else:
        # Create or update .mcp.json for the MCP server (project scope)
        mcp_json_path = target_dir / '.mcp.json'
        mcp_config: dict = {}
        if mcp_json_path.exists():
            try:
                mcp_config = json_module.loads(mcp_json_path.read_text())
            except json_module.JSONDecodeError:
                pass

        if 'mcpServers' not in mcp_config:
            mcp_config['mcpServers'] = {}

        if 'ariadne' not in mcp_config['mcpServers']:
            mcp_config['mcpServers']['ariadne'] = {
                'command': 'uv',
                'args': ['run', '--directory', str(ariadne_path), 'ariadne', 'mcp'],
            }
            mcp_json_path.write_text(json_module.dumps(mcp_config, indent=2) + '\n')
            console.print(f'[green]Created/updated {mcp_json_path}[/green]')
        else:
            console.print(f'[yellow]Ariadne MCP server already configured in {mcp_json_path}[/yellow]')

    # Create or append to CLAUDE.md (minimal pointer - full content comes from hook)
    target_claude_md_path = target_dir / 'CLAUDE.md'
    claude_md_snippet = f'''
## Ariadne Integration

Instructions for this project are managed by Ariadne and injected via session hook.
The hook uses `--auto-scope` to detect the current directory and git branch,
then injects the appropriate scoped documentation.

Authoritative source: `{docs_path}/CLAUDE.md`

To edit instructions: `ariadne edit-instructions`
'''

    if target_claude_md_path.exists():
        existing = target_claude_md_path.read_text()
        if 'Ariadne Integration' not in existing and 'Knowledge Base' not in existing:
            with open(target_claude_md_path, 'a') as f:
                f.write(claude_md_snippet)
            console.print(f'[green]Appended Ariadne section to {target_claude_md_path}[/green]')
        else:
            console.print(f'[yellow]Ariadne section already exists in {target_claude_md_path}[/yellow]')
    else:
        target_claude_md_path.write_text(f'# {target_dir.name}\n{claude_md_snippet}')
        console.print(f'[green]Created {target_claude_md_path}[/green]')

    console.print()
    console.print('[bold]Ariadne integration initialized![/bold]')
    console.print()
    console.print('Next steps:')
    console.print(f'  1. Install: uv tool install {ariadne_path}')
    console.print('  2. Generate docs: ariadne generate')
    console.print('  3. Export docs: ariadne export')
    console.print('  4. Start a Claude Code session to verify')

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Show current configuration."""
    from config import get_config

    cfg = get_config()

    if args.path:
        if cfg.config_path:
            console.print(str(cfg.config_path))
        else:
            console.print('[yellow]No config file found[/yellow]')
        return 0

    table = Table(title='Ariadne Configuration')
    table.add_column('Setting', style='bold')
    table.add_column('Value', style='cyan')

    table.add_row('Config file', str(cfg.config_path) if cfg.config_path else '[dim]not found[/dim]')
    table.add_row('Default model', cfg.model)
    table.add_row('Database path', cfg.db_path)
    table.add_row('Docs base', str(cfg.docs_base))
    table.add_row('Default source', cfg.default_source or '[dim]not set[/dim]')

    console.print(table)

    if cfg.sources:
        console.print()
        sources_table = Table(title='Configured Sources')
        sources_table.add_column('Name', style='green')
        sources_table.add_column('Path', style='cyan')
        sources_table.add_column('Options', style='dim')

        for name in cfg.sources:
            source_config = cfg.get_source_config(name)
            if source_config:
                path_str = source_config.path
                options = []
                if source_config.parent:
                    options.append(f'parent={source_config.parent}')
                if source_config.branches:
                    options.append(f'branches={",".join(source_config.branches)}')
                if source_config.ref:
                    options.append(f'ref={source_config.ref}')
                if source_config.depends_on:
                    options.append(f'deps={",".join(source_config.depends_on)}')
                sources_table.add_row(name, path_str, ', '.join(options) if options else '')

        console.print(sources_table)

    return 0


def _build_embedding_matrix_on_startup() -> None:
    """Build or refresh the shared embedding matrix once at serve startup.

    Off by default (see cmd_mcp); enabled via ARIADNE_BUILD_MATRIX_ON_STARTUP.
    Degrades to the SQLite ranking fallback (with a logged warning) if it cannot
    build — it never blocks serving.
    """
    import logging

    try:
        from ariadne_mcp.service import AriadneService
        from library.embedding_matrix import ensure_matrix

        ensure_matrix(AriadneService.get().library)
    except Exception:
        logging.getLogger(__name__).warning(
            'Embedding matrix unavailable at startup; serving with the SQLite fallback',
            exc_info=True,
        )


def _serve_mcp(mcp, *, http: bool, host: str, port: int) -> None:
    """Run the MCP server: stdio by default, streamable-http under ``--http``.

    Host/port apply only to the http transport; stdio is a local pipe.
    """
    if http:
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport='streamable-http')
    else:
        mcp.run(transport='stdio')


def cmd_mcp(args: argparse.Namespace) -> int:
    """Start the MCP server.

    Building the embedding matrix at startup is OFF by default — the server only
    loads a pre-generated matrix (see ``ariadne build-matrix``). Set
    ARIADNE_BUILD_MATRIX_ON_STARTUP=1 to build it on startup instead (not advised
    on small or pooled boxes — the build can spike ~2 GB RAM).
    """
    if args.directory:
        os.chdir(args.directory)
    else:
        # Use the directory containing this file (the Ariadne project root)
        # rather than config_dir, which falls back to cwd when no config is found.
        os.chdir(Path(__file__).resolve().parent)

    import ariadne_mcp.server as mcp_server

    if os.environ.get('ARIADNE_BUILD_MATRIX_ON_STARTUP', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
        _build_embedding_matrix_on_startup()
    _serve_mcp(
        mcp_server.mcp,
        http=getattr(args, 'http', False),
        host=getattr(args, 'host', '127.0.0.1'),
        port=getattr(args, 'port', 8000),
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the web onboarding UI.

    Runs an aiohttp server that connects to the Ariadne MCP server (spawned
    as a stdio subprocess) and serves the single-page onboarding wizard.
    Open the printed URL to add a source, discover its languages, preview
    cost, and generate docs — all driven over MCP.
    """
    from web.server import serve

    console.print(
        f'[green]Ariadne onboarding UI →[/green] '
        f'http://{args.host}:{args.port}',
    )
    serve(
        host=args.host,
        port=args.port,
        config_path=os.environ.get('ARIADNE_CONFIG'),
        mcp_url=getattr(args, 'mcp_url', None),
    )
    return 0


def cmd_sync_claude_md(args: argparse.Namespace) -> int:
    """Sync an edited CLAUDE.md back to Ariadne's authoritative location.

    This is called by PostToolUse hooks when Claude edits a CLAUDE.md file.
    """
    import shutil

    from config import get_config

    cfg = get_config()
    source_name = args.source or cfg.default_source

    if not source_name:
        # Silent fail - no source configured
        return 0

    # Get the source file path
    source_file = Path(args.file).resolve()

    # Skip if this IS the Ariadne CLAUDE.md (avoid infinite loop)
    ariadne_docs_path = cfg.resolve_docs_path(source_name)
    ariadne_claude_md = ariadne_docs_path / 'CLAUDE.md'

    if source_file == ariadne_claude_md.resolve():
        # Already the authoritative file, nothing to sync
        return 0

    # Check if the file exists and is a CLAUDE.md
    if not source_file.exists():
        return 0

    if not source_file.name.endswith('CLAUDE.md'):
        return 0

    # Copy to Ariadne's location
    ariadne_docs_path.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(source_file, ariadne_claude_md)
        print(f'Synced to Ariadne: {ariadne_claude_md}')
    except OSError as e:
        print(f'Failed to sync: {e}')
        return 1

    return 0


def cmd_edit_instructions(args: argparse.Namespace) -> int:
    """Open Ariadne's CLAUDE.md in $EDITOR for editing.

    After saving, re-exports to update the authoritative version.
    """
    import os
    import subprocess

    from config import get_config

    cfg = get_config()
    source_name = args.source or cfg.default_source

    if not source_name:
        console.print('[red]No source specified and no default_source in config.[/red]')
        return 1

    ariadne_docs_path = cfg.resolve_docs_path(source_name)
    claude_md_path = ariadne_docs_path / 'CLAUDE.md'

    if not claude_md_path.exists():
        console.print(f'[yellow]CLAUDE.md not found at {claude_md_path}[/yellow]')
        console.print('Run "ariadne export" first to generate it.')
        return 1

    # Get editor from environment
    editor = os.environ.get('EDITOR', os.environ.get('VISUAL', 'vi'))

    # Open in editor
    console.print(f'Opening {claude_md_path} in {editor}...')
    result = subprocess.run([editor, str(claude_md_path)])

    if result.returncode != 0:
        console.print(f'[red]Editor exited with code {result.returncode}[/red]')
        return result.returncode

    console.print('[green]Changes saved.[/green]')

    # Note: We don't re-export here since the CLAUDE.md IS the authoritative source
    # If user wants to regenerate from library docs, they should run 'ariadne export'

    return 0


def _resolve_writable_config():
    """Return a Config bound to a writable ariadne.yaml, creating an
    empty one if the project doesn't have a config file yet.

    A brand-new project has no ariadne.yaml, so ``get_config()`` comes
    back with ``config_path is None`` and the source-mutation primitives
    refuse to write. Bootstrap a minimal file at ``$ARIADNE_CONFIG`` (or
    ``./ariadne.yaml``) so `source add` works as a project's very first
    Ariadne command.

    An explicit ``$ARIADNE_CONFIG`` is authoritative for writes: when it is
    set but the file doesn't exist yet, bootstrap THERE rather than adopt
    whatever ``config_search_paths`` fell through to (e.g. the package-root
    ariadne.yaml that ships for the ``ariadne mcp`` / ``uv run --directory``
    case). Otherwise `source add` would silently mutate an unrelated config.
    """
    import config as config_module
    from config import Config, get_config

    env = os.environ.get('ARIADNE_CONFIG')
    cfg = get_config()
    # Reuse an already-resolved config UNLESS an explicit $ARIADNE_CONFIG
    # points elsewhere (set-but-missing falls through to a fallback rung).
    if cfg.config_path is not None and (
        env is None
        or Path(env).resolve() == Path(cfg.config_path).resolve()
    ):
        return cfg

    cfg_file = Path(env) if env else Path.cwd() / 'ariadne.yaml'
    if not cfg_file.exists():
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text('sources: {}\n')
    # Rebind the cached singleton so later commands in this process (and
    # the rest of this one) see the now-existing config rather than
    # re-bootstrapping over it.
    new_cfg = Config(config_path=cfg_file)
    config_module._global_config = new_cfg
    return new_cfg


def _prompt_if_missing(value, label):
    """Return ``value``, or prompt for it interactively on a TTY.

    Flags are the primary input; the prompt is a fallback so the command
    stays scriptable (non-TTY → no prompt). Returns None when the value
    is absent and we can't prompt, letting the caller fail loudly.
    """
    import sys

    if value:
        return value
    if sys.stdin.isatty():
        entered = input(f'{label}: ').strip()
        return entered or None
    return None


def cmd_source(args: argparse.Namespace) -> int:
    """Dispatcher for `ariadne source <action>`."""
    action = getattr(args, 'source_action', None)
    if action == 'add':
        return _cmd_source_add(args)
    if action == 'list':
        return _cmd_source_list(args)
    if action == 'remove':
        return _cmd_source_remove(args)
    if action == 'purge':
        return _cmd_source_purge(args)
    console.print(
        '[yellow]Usage: ariadne source <add|list|remove|purge> ...[/yellow]')
    return 1


def _confirm(question: str, *, assume_yes: bool = False) -> bool:
    """Interactive y/N confirmation (default No). ``assume_yes`` (from --yes)
    skips the prompt and proceeds; a non-TTY without --yes is not confirmed."""
    if assume_yes:
        return True
    answer = _prompt_if_missing('', f'{question} [y/N]')
    return (answer or '').strip().lower() in ('y', 'yes')


def _cmd_source_remove(args: argparse.Namespace) -> int:
    from config import get_config

    name = _prompt_if_missing(getattr(args, 'name', None), 'Source name')
    if not name:
        console.print('[red]A source name is required.[/red]')
        return 1

    cfg = get_config()
    if cfg.get_source_config(name) is None:
        console.print(f'[yellow]No source named {name!r} in config.[/yellow]')
        return 1

    if not _confirm(f"Remove source '{name}'?",
                    assume_yes=getattr(args, 'yes', False)):
        console.print('[dim]Aborted.[/dim]')
        return 1

    if not cfg.remove_source(name):
        console.print(f'[red]Failed to remove source {name!r}.[/red]')
        return 1

    console.print(f'[green]Removed source [bold]{name}[/bold].[/green]')
    if getattr(args, 'purge', False):
        from library import Library
        from library.embedding_matrix import recreate_matrix
        from library.purge import purge_source
        with Library(cfg.db_path) as library:
            summary = purge_source(library, name)
            if summary.total == 0:
                console.print(f'[dim]No indexed DB data for {name!r}.[/dim]')
            else:
                console.print(
                    f'[green]Purged {summary.total} DB row(s) for '
                    f'[bold]{name}[/bold][/green] '
                    f'({summary.counts.get("documents", 0)} documents).'
                )
                if recreate_matrix(library):
                    console.print('[dim]Rebuilt the embedding matrix.[/dim]')
    return 0


def _cmd_source_purge(args: argparse.Namespace) -> int:
    """Delete a source's indexed data from the library DB, whether or not
    it is still in ariadne.yaml. `source remove` is config-only, so a
    removed source's documents / chunks / SCIP rows linger (and still rank
    in semantic search); this clears them and rebuilds the matrix.
    """
    from config import get_config
    from library import Library
    from library.embedding_matrix import recreate_matrix
    from library.purge import purge_source

    name = _prompt_if_missing(getattr(args, 'name', None), 'Source name')
    if not name:
        console.print('[red]A source name is required.[/red]')
        return 1
    cfg = get_config()
    with Library(cfg.db_path) as library:
        preview = purge_source(library, name, dry_run=True)
    if preview.total == 0:
        console.print(f'[yellow]No indexed DB data for {name!r}.[/yellow]')
        return 0
    if not _confirm(f"Purge {preview.total} DB row(s) for '{name}'?",
                    assume_yes=getattr(args, 'yes', False)):
        console.print('[dim]Aborted.[/dim]')
        return 1
    with Library(cfg.db_path) as library:
        summary = purge_source(library, name)
        recreate_matrix(library)
    console.print(
        f'[green]Purged {summary.total} DB row(s) for [bold]{name}[/bold]'
        f'[/green] ({summary.counts.get("documents", 0)} documents).'
    )
    return 0


def _cmd_source_list(args: argparse.Namespace) -> int:
    from config import get_config

    cfg = get_config()
    if not cfg.sources:
        console.print('[dim]No sources configured.[/dim]')
        return 0

    table = Table(title='Configured Sources')
    table.add_column('Name', style='green')
    table.add_column('Path', style='cyan')
    table.add_column('Options', style='dim')

    for name in cfg.sources:
        sc = cfg.get_source_config(name)
        if not sc:
            continue
        default_mark = ' [yellow](default)[/yellow]' if name == cfg.default_source else ''
        options = []
        if sc.parent:
            options.append(f'parent={sc.parent}')
        if sc.depends_on:
            options.append(f'deps={",".join(sc.depends_on)}')
        if sc.exclude:
            options.append(f'exclude={",".join(sc.exclude)}')
        if sc.exclude_dirs:
            options.append(f'exclude_dirs={",".join(sc.exclude_dirs)}')
        table.add_row(name + default_mark, sc.path, ', '.join(options))

    console.print(table)
    return 0


def _cmd_source_add(args: argparse.Namespace) -> int:
    name = _prompt_if_missing(getattr(args, 'name', None), 'Source name')
    if not name:
        console.print('[red]A source name is required.[/red]')
        return 1

    def _csv(value):
        if value is None:
            return None
        return [item.strip() for item in value.split(',') if item.strip()]

    cfg = _resolve_writable_config()
    existed = cfg.get_source_config(name) is not None

    # Path is mandatory when creating a source, optional when updating an
    # existing one (so `source add foo --parent bar` just edits a field).
    path = getattr(args, 'path', None)
    if not existed:
        path = _prompt_if_missing(path, 'Source path')
        if not path:
            console.print('[red]A source path is required.[/red]')
            return 1

    if not cfg.set_source_config(
        name,
        path=path,
        depends_on=_csv(getattr(args, 'depends_on', None)),
        parent=getattr(args, 'parent', None),
        branches=_csv(getattr(args, 'branches', None)),
        ref=getattr(args, 'ref', None),
        exclude=_csv(getattr(args, 'exclude', None)),
        exclude_dirs=_csv(getattr(args, 'exclude_dirs', None)),
    ignore_staleness=getattr(args, 'ignore_staleness', None), skip_dependency_detection=getattr(args, 'skip_dependency_detection', None)):
        console.print(f'[red]Failed to write source {name!r} to config.[/red]')
        return 1

    # First source in a fresh project becomes the default so downstream
    # commands (discover/onboard) have something to act on without -s.
    if cfg.default_source is None:
        cfg.set_default_source(name)

    verb = 'Updated' if existed else 'Added'
    effective = cfg.get_source_config(name)
    shown_path = effective.path if effective else path
    console.print(
        f'[green]{verb} source [bold]{name}[/bold] → {shown_path}[/green]')
    console.print(f'[dim]Config: {cfg.config_path}[/dim]')
    console.print()
    console.print('Next steps:')
    console.print(f'  ariadne discover --source {name}')
    console.print(f'  ariadne onboard --source {name}')
    return 0


HANDLERS = {
    'manifest': lambda args: cmd_manifest(args),
    'init': lambda args: cmd_init(args),
    'config': lambda args: cmd_config(args),
    'mcp': lambda args: cmd_mcp(args),
    'serve': lambda args: cmd_serve(args),
    'sync-claude-md': lambda args: cmd_sync_claude_md(args),
    'edit-instructions': lambda args: cmd_edit_instructions(args),
    'source': lambda args: cmd_source(args),
}
