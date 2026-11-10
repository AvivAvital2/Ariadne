#!/usr/bin/env python3
"""Repair the two local-model snapshot sync commits in the public checkout.

Run without --apply to check the repair. Changes stay uncommitted and unstaged;
this script never resets, commits, or pushes. Unrelated working/index changes
are preserved. The adjacent patch contains the reviewed integration changes.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


SYNCED_HEAD = '8786aaaaf6dc1b761e47dbb4c15296cdf74c7b49'
PATCH = Path(__file__).with_suffix('.patch')


def git(repo, *args):
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--to', type=Path, required=True, help='Public checkout to repair')
    parser.add_argument('--apply', action='store_true', help='Apply the checked repair')
    args = parser.parse_args()
    repo = args.to.resolve()
    if not PATCH.is_file():
        parser.error(f'Repair patch is missing: {PATCH}')
    root = git(repo, 'rev-parse', '--show-toplevel')
    if root.returncode:
        parser.error(f'Not a Git working tree: {repo}')
    repo = Path(root.stdout.strip())
    head = git(repo, 'rev-parse', '--verify', 'HEAD')
    if head.returncode:
        parser.error(f'Not a Git checkout: {repo}')

    # Also recognize an already applied repair after the user has committed it.
    reverse = git(repo, 'apply', '--reverse', '--check', str(PATCH))
    if reverse.returncode == 0:
        print('Repair is already applied; no changes made.')
        return 0
    if head.stdout.strip() != SYNCED_HEAD:
        parser.error(
            f'This repair targets public commit {SYNCED_HEAD[:12]}; '
            f'found {head.stdout.strip()[:12]}. No changes made.'
        )

    names = git(repo, 'apply', '--numstat', '-z', str(PATCH))
    if names.returncode:
        parser.error(names.stderr.strip())
    paths = [record.split('\t', 2)[2] for record in names.stdout.split('\0') if record]
    dirty = git(repo, 'diff', '--name-only', 'HEAD', '--', *paths)
    if dirty.returncode:
        parser.error(dirty.stderr.strip())
    if dirty.stdout:
        parser.error('Repair would overlap your edits; no changes made:\n' + dirty.stdout)

    check = git(repo, 'apply', '--check', str(PATCH))
    if check.returncode:
        parser.error('Repair cannot apply cleanly; no changes made:\n' + check.stderr)
    stat = git(repo, 'apply', '--stat', str(PATCH))
    if stat.returncode:
        parser.error(stat.stderr.strip())
    print(stat.stdout, end='')
    if not args.apply:
        print('Check passed. Add --apply to write the repair without staging or committing.')
        return 0

    # git apply checks the whole patch before writing; no --reject/--index flags.
    applied = git(repo, 'apply', str(PATCH))
    if applied.returncode:
        print(applied.stderr, file=sys.stderr, end='')
        return applied.returncode
    print('Repair applied. Changes are unstaged and uncommitted. Nothing was pushed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
