---
name: ariadne-improve
description: |
  Run Ariadne's improvement cycle: analyze documentation gaps, show per-document
  usage stats, and generate docs for undocumented source files.
  Use when: the user asks to improve Ariadne docs, run a doc quality sweep,
  or check what documentation is missing.
  user_invocable: true
---

# Ariadne Improvement Cycle

Run the Ariadne documentation improvement cycle to analyze gaps and generate missing docs.

## Steps

### 1. Preview the improvement plan

Call the MCP tool to see what would be done:

```
ariadne_improve(dry_run=True)
```

This shows:
- Gap analysis (what topics users searched for but didn't find)
- Undocumented file count
- Top-served documents
- Files that would be documented

### 2. Show current stats

```
ariadne_project_stats()
ariadne_document_usage(days=30, limit=10)
```

Present the stats to the user:
- Per-source document counts and sizes
- Most-served documents (candidates for splitting if too large)
- Least-served documents (candidates for pruning)

### 3. Get user approval and run

After showing the preview, ask the user if they want to proceed. If yes:

```
ariadne_improve(dry_run=False, max_files=10)
```

### 4. Report results

After generation completes, show:
- How many new docs were created
- Updated stats via `ariadne_project_stats()`
- Suggest running `ariadne export` and `ariadne rebuild` if needed