#!/usr/bin/env bash
#
# sync-downstream.sh — mirror files changed since the last push to
# origin (Ariadne-private) into the downstream public repo (Ariadne),
# and print each commit message so you can recreate the commits there.
#
# The two repos do NOT share git history, so the sync window is
# "commits on this branch not yet pushed to origin":
#
#     <upstream-tracking-ref>..HEAD      (e.g. origin/main..HEAD)
#
# Run this BEFORE you `git push` to origin, while the new commits are
# still unpushed. If you already pushed, pass --since <ref> with the
# last commit you synced (a SHA or tag).
#
# Exclusions are delegated to the DOWNSTREAM repo's .gitignore: any
# changed file that the downstream would ignore is skipped. Files that
# would be NEWLY introduced to downstream (not tracked there and not
# ignored) are flagged loudly so internal/private files can't silently
# leak into the public repo.
#
# Usage:
#   scripts/sync-downstream.sh                 # dry run (preview only)
#   scripts/sync-downstream.sh --apply         # actually copy/delete
#   scripts/sync-downstream.sh --since <ref>   # custom base (else @{upstream})
#   scripts/sync-downstream.sh --to <dir>      # custom downstream path
#
# Env:
#   DOWNSTREAM=<dir>   default downstream repo path
#
set -euo pipefail

# --- defaults --------------------------------------------------------
DOWNSTREAM_DEFAULT="/Users/spark/git/Ariadne"
SINCE=""
TO="${DOWNSTREAM:-$DOWNSTREAM_DEFAULT}"
APPLY=0

# --- parse args ------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --since)   SINCE="${2:?--since needs a ref}"; shift 2 ;;
    --to)      TO="${2:?--to needs a dir}"; shift 2 ;;
    --apply)   APPLY=1; shift ;;
    --dry-run) APPLY=0; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//;/^set -euo/d'; exit 0 ;;
    *) echo "Unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# --- locate repos (independent of CWD) -------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$(cd "$SCRIPT_DIR/.." && git rev-parse --show-toplevel)"
cd "$UPSTREAM"

if [ ! -e "$TO/.git" ]; then
  echo "ERROR: downstream git repo not found at: $TO" >&2
  echo "       Pass --to <dir> or set DOWNSTREAM=<dir>." >&2
  exit 1
fi
TO="$(cd "$TO" && pwd)"   # normalize to absolute

# --- resolve the sync base (last push) -------------------------------
if [ -z "$SINCE" ]; then
  if ! SINCE="$(git rev-parse --verify --quiet '@{upstream}')"; then
    echo "ERROR: no upstream tracking ref for the current branch." >&2
    echo "       Pass --since <ref> (the last commit you synced)." >&2
    exit 1
  fi
fi

if ! BASE="$(git rev-parse --verify --quiet "${SINCE}^{commit}")"; then
  echo "ERROR: --since ref '$SINCE' is not a valid commit." >&2
  exit 1
fi

RANGE="$BASE..HEAD"
N_COMMITS="$(git rev-list --count "$RANGE")"

echo "Upstream:   $UPSTREAM"
echo "Downstream: $TO"
echo "Base:       $(git rev-parse --short "$BASE")  ($SINCE)"
echo "HEAD:       $(git rev-parse --short HEAD)"
echo "Commits:    $N_COMMITS in $RANGE"
echo

if [ "$N_COMMITS" -eq 0 ]; then
  echo "Nothing to sync — no commits in $RANGE."
  echo
  echo "If you already pushed to origin, '@{upstream}' has advanced past your"
  echo "new commits. Re-run pointing at the pre-push position, e.g.:"
  echo "    $(basename "$0") --since 'origin/$(git rev-parse --abbrev-ref HEAD)@{1}'"
  echo "or pass --since <the last commit you synced>."
  exit 0
fi

# --- classify changed files ------------------------------------------
# Net change across the whole range; renames are split into delete+add
# (--no-renames) so the old path is removed downstream and the new one
# copied. ACMT = added / copied / modified / type-changed.
copy_candidates=()
while IFS= read -r -d '' f; do copy_candidates+=("$f"); done \
  < <(git diff -z --no-renames --name-only --diff-filter=ACMT "$BASE" HEAD)

del_candidates=()
while IFS= read -r -d '' f; do del_candidates+=("$f"); done \
  < <(git diff -z --no-renames --name-only --diff-filter=D "$BASE" HEAD)

to_copy=()
to_delete=()
skipped_ignored=()
new_to_downstream=()

for p in ${copy_candidates[@]+"${copy_candidates[@]}"}; do
  [ -z "$p" ] && continue
  if git -C "$TO" check-ignore -q -- "$p"; then
    skipped_ignored+=("$p"); continue
  fi
  to_copy+=("$p")
  if ! git -C "$TO" ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
    new_to_downstream+=("$p")
  fi
done

for p in ${del_candidates[@]+"${del_candidates[@]}"}; do
  [ -z "$p" ] && continue
  if git -C "$TO" check-ignore -q -- "$p"; then
    skipped_ignored+=("$p"); continue
  fi
  # Only a deletion to mirror if the file actually exists downstream.
  if [ -e "$TO/$p" ]; then
    to_delete+=("$p")
  fi
done

# --- report plan -----------------------------------------------------
print_list() { # $1=label  $2..=items
  local label="$1"; shift
  echo "$label ($#)"
  local i
  for i in "$@"; do echo "    $i"; done
  echo
}

[ "$APPLY" -eq 1 ] && echo "=== APPLY ===" || echo "=== DRY RUN (nothing written) ==="
echo
print_list "Copy → downstream:"        ${to_copy[@]+"${to_copy[@]}"}
print_list "Delete from downstream:"   ${to_delete[@]+"${to_delete[@]}"}
print_list "Skipped (downstream .gitignore):" ${skipped_ignored[@]+"${skipped_ignored[@]}"}

if [ "${#new_to_downstream[@]}" -gt 0 ]; then
  echo "⚠  WARNING — files NOT yet tracked downstream and NOT gitignored."
  echo "   These will be ADDED to the public repo. Verify none are internal;"
  echo "   add them to ${TO}/.gitignore if they should stay private:"
  for p in "${new_to_downstream[@]}"; do echo "    + $p"; done
  echo
fi

# --- apply -----------------------------------------------------------
if [ "$APPLY" -eq 1 ]; then
  for p in ${to_copy[@]+"${to_copy[@]}"}; do
    [ -z "$p" ] && continue
    dest="$TO/$p"
    mkdir -p "$(dirname "$dest")"
    git show "HEAD:$p" > "$dest"          # committed content, not work-tree
    mode="$(git ls-tree HEAD -- "$p" | awk '{print $1}')"
    [ "$mode" = "100755" ] && chmod +x "$dest"
  done
  for p in ${to_delete[@]+"${to_delete[@]}"}; do
    [ -z "$p" ] && continue
    rm -f "$TO/$p"
  done
  echo "Applied: ${#to_copy[@]} copied, ${#to_delete[@]} deleted."
  echo
fi

# --- commit messages to replay downstream ----------------------------
echo "========================================================"
echo " Commit messages to recreate downstream (oldest first):"
echo "========================================================"
git log --reverse --format='%n----- %h -----%n%B' "$RANGE"

if [ "$APPLY" -eq 0 ]; then
  echo
  echo "DRY RUN — re-run with --apply to copy the files above."
fi
