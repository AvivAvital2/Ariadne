#!/usr/bin/env bash
#
# sync-downstream.sh — mirror commits made since the last push to origin
# (Ariadne-private) into the downstream public repo (Ariadne), replaying
# each upstream commit 1:1 (same files, same message), then showing the
# new commits so you can decide whether to push.
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
# For each upstream commit (oldest first) it copies that commit's changed
# files into the downstream working tree and creates a matching downstream
# commit with the same message. Exclusions are delegated to the DOWNSTREAM
# repo's .gitignore: any changed file the downstream would ignore is
# skipped. Files that would be NEWLY introduced to downstream (not tracked
# there and not ignored) are flagged loudly so internal/private files
# can't silently leak into the public repo. It never pushes — that stays
# your call after reviewing the log it prints.
#
# Usage:
#   scripts/sync-downstream.sh                 # replay + commit, then show the log (DEFAULT)
#   scripts/sync-downstream.sh --dry-run       # preview only — no writes, no commits
#   scripts/sync-downstream.sh --no-commit     # copy files into the work tree, don't commit
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
DO_WRITE=1   # write files into the downstream work tree
DO_COMMIT=1  # create a downstream commit per upstream commit

# --- parse args ------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --since)     SINCE="${2:?--since needs a ref}"; shift 2 ;;
    --to)        TO="${2:?--to needs a dir}"; shift 2 ;;
    --no-commit) DO_COMMIT=0; shift ;;
    --dry-run)   DO_WRITE=0; DO_COMMIT=0; shift ;;
    --apply)
      echo "--apply is gone: copy+commit is now the default. Use" >&2
      echo "--no-commit to copy without committing, or --dry-run to" >&2
      echo "preview." >&2
      exit 2 ;;
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

# --- commits to replay, oldest first ---------------------------------
commits=()
while IFS= read -r c; do
  [ -n "$c" ] && commits+=("$c")
done < <(git rev-list --reverse "$BASE..HEAD")
N_COMMITS=${#commits[@]}

echo "Upstream:   $UPSTREAM"
echo "Downstream: $TO"
echo "Base:       $(git rev-parse --short "$BASE")  ($SINCE)"
echo "HEAD:       $(git rev-parse --short HEAD)"
echo "Commits:    $N_COMMITS in $BASE..HEAD"
echo

if [ "$N_COMMITS" -eq 0 ]; then
  echo "Nothing to sync — no commits in $BASE..HEAD."
  echo
  echo "If you already pushed to origin, '@{upstream}' has advanced past"
  echo "your new commits. Re-run pointing at the pre-push position, e.g.:"
  echo "    $(basename "$0") --since 'origin/$(git rev-parse --abbrev-ref HEAD)@{1}'"
  echo "or pass --since <the last commit you synced>."
  exit 0
fi

if [ "$DO_WRITE" -eq 0 ]; then
  echo "=== DRY RUN — no files written, no commits made ==="
elif [ "$DO_COMMIT" -eq 0 ]; then
  echo "=== COPY ONLY — files copied, not committed ==="
else
  echo "=== REPLAY — one downstream commit per upstream commit ==="
fi
echo

# Warn if the downstream tree already has uncommitted changes (before we
# add to it), so the replayed commits don't get tangled with stray work.
if [ "$DO_WRITE" -eq 1 ] && [ -n "$(git -C "$TO" status --porcelain)" ]; then
  echo "⚠  Downstream working tree has uncommitted changes; the sync will"
  echo "   add commits on top of the current state. Review before pushing."
  echo
fi

# --- replay loop -----------------------------------------------------
new_to_downstream=()   # would be ADDED to downstream and isn't gitignored
committed=0
skipped_commits=0

for c in "${commits[@]}"; do
  subject="$(git log -1 --format='%h %s' "$c")"

  # Net change of THIS commit vs its parent; renames split into delete+add.
  copy_set=()
  while IFS= read -r -d '' f; do copy_set+=("$f"); done \
    < <(git diff -z --no-renames --name-only --diff-filter=ACMT "${c}~1" "$c")
  del_set=()
  while IFS= read -r -d '' f; do del_set+=("$f"); done \
    < <(git diff -z --no-renames --name-only --diff-filter=D "${c}~1" "$c")

  paths=()            # downstream pathspecs to commit for this commit
  n_copy=0; n_del=0; n_skip=0

  for f in ${copy_set[@]+"${copy_set[@]}"}; do
    [ -z "$f" ] && continue
    if git -C "$TO" check-ignore -q -- "$f"; then n_skip=$((n_skip + 1)); continue; fi
    if ! git -C "$TO" ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
      new_to_downstream+=("$f")
    fi
    if [ "$DO_WRITE" -eq 1 ]; then
      dest="$TO/$f"
      mkdir -p "$(dirname "$dest")"
      git show "$c:$f" > "$dest"
      mode="$(git ls-tree "$c" -- "$f" | awk '{print $1}')"
      [ "$mode" = "100755" ] && chmod +x "$dest"
      [ "$DO_COMMIT" -eq 1 ] && git -C "$TO" add -- "$f"
    fi
    paths+=("$f"); n_copy=$((n_copy + 1))
  done

  for f in ${del_set[@]+"${del_set[@]}"}; do
    [ -z "$f" ] && continue
    if git -C "$TO" check-ignore -q -- "$f"; then n_skip=$((n_skip + 1)); continue; fi
    [ -e "$TO/$f" ] || continue
    if [ "$DO_WRITE" -eq 1 ]; then
      rm -f "$TO/$f"
      [ "$DO_COMMIT" -eq 1 ] && git -C "$TO" add -- "$f"
    fi
    paths+=("$f"); n_del=$((n_del + 1))
  done

  if [ "${#paths[@]}" -eq 0 ]; then
    echo "  - skip  $subject  (only ignored / not-present files)"
    skipped_commits=$((skipped_commits + 1))
    continue
  fi

  # Nothing actually staged (content already identical downstream)? Don't
  # create an empty commit.
  if [ "$DO_COMMIT" -eq 1 ] \
     && git -C "$TO" diff --cached --quiet -- ${paths[@]+"${paths[@]}"}; then
    echo "  - skip  $subject  (no net change downstream)"
    skipped_commits=$((skipped_commits + 1))
    continue
  fi

  echo "  •       $subject  (+${n_copy} / -${n_del}, ${n_skip} ignored)"

  if [ "$DO_COMMIT" -eq 1 ]; then
    # Pipe the original message straight to commit via stdin (-F -), so
    # there's no temp file to create or clean up.
    git log -1 --format='%B' "$c" \
      | git -C "$TO" commit -q -F - -- ${paths[@]+"${paths[@]}"}
    committed=$((committed + 1))
  fi
done
echo

# --- leak guard ------------------------------------------------------
if [ "${#new_to_downstream[@]}" -gt 0 ]; then
  echo "⚠  WARNING — files NOT previously tracked downstream and NOT"
  echo "   gitignored there. These were ADDED to the public repo. Verify"
  echo "   none are internal; add to ${TO}/.gitignore and amend if needed:"
  printf '%s\n' "${new_to_downstream[@]}" | sort -u | sed 's/^/    + /'
  echo
fi

# --- outcome ---------------------------------------------------------
if [ "$DO_WRITE" -eq 0 ]; then
  echo "DRY RUN — re-run without --dry-run to replay + commit downstream."
  exit 0
fi

if [ "$DO_COMMIT" -eq 0 ]; then
  echo "Copied files into $TO (uncommitted)."
  echo "Commit messages to recreate, oldest first:"
  git log --reverse --format='%n----- %h -----%n%B' "$BASE..HEAD"
  exit 0
fi

if [ "$committed" -eq 0 ]; then
  echo "No downstream commits created — all changes were already present"
  echo "or excluded by the downstream .gitignore."
  exit 0
fi

echo "Replayed ${committed} commit(s) into ${TO} (skipped ${skipped_commits})."
echo
echo "Review the new commits below, then decide whether to push:"
echo
git -C "$TO" log -n "$committed" --stat
echo
echo "If they look right, push from the downstream repo:"
echo "    (cd $TO && git push)"
