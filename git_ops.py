"""Git subprocess helpers for Ariadne.

Thin wrappers around git commands used by the MCP server.
No Ariadne-specific imports — only stdlib.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def get_current_branch(cwd: Path | None = None) -> str | None:
    """Get the current git branch name.

    Args:
        cwd: Directory to run git in. Defaults to process cwd.

    Returns:
        Branch name, or None if not in a git repo.
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_changed_files_vs_main(
    source_path: Path,
    main_branch: str = 'main',
) -> list[str]:
    """Get files changed between the current branch and main.

    Args:
        source_path: Repository directory.
        main_branch: Name of the main branch to compare against.

    Returns:
        List of changed file paths (relative to repo root).
    """
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', f'{main_branch}...HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=source_path,
        )
        return [f for f in result.stdout.strip().split('\n') if f]
    except subprocess.CalledProcessError:
        return []


def get_head_commit(cwd: Path) -> str | None:
    """Current HEAD commit sha, or None if ``cwd`` isn't a git repo."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_changed_files_since(commit: str, cwd: Path) -> list[str] | None:
    """Files changed between ``commit`` and HEAD, relative to ``cwd``.

    ``--relative`` scopes the diff to ``cwd`` and yields ``cwd``-relative
    paths, so a source rooted in a subdirectory of a larger repo still gets
    paths that match the orchestrator's source-relative file keys. Returns
    None when ``cwd`` isn't a git repo or the commit is unknown (caller falls
    back to a full pass) — distinct from an empty list (no files changed).
    """
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', '--relative', f'{commit}..HEAD'],
            capture_output=True, text=True, check=True, cwd=cwd,
        )
        return [f for f in result.stdout.strip().split('\n') if f]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_commit_message(git_hash: str, cwd: Path) -> str | None:
    """Get the commit message for a given hash.

    Args:
        git_hash: The git commit hash.
        cwd: Repository directory.

    Returns:
        Commit subject line, or None on failure.
    """
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%s', git_hash],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def run_ariadne_cli(args: list[str]) -> str:
    """Run an ariadne CLI command and return combined output.

    Args:
        args: CLI arguments (e.g. ['sync', '--source', 'pythonproject']).

    Returns:
        Combined stdout and stderr.
    """
    result = subprocess.run(
        ['uv', 'run', 'ariadne', *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    return result.stdout + result.stderr
