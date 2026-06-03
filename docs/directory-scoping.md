# Directory-Scoped Dependencies

This guide explains how to configure Ariadne's directory-scoped dependencies feature, which enables automatic source detection based on your working directory and git branch.

## Overview

When working with monorepos or complex project structures, you often need different documentation sets depending on where you're working:

- Working in `benchmark/` should include benchmark-specific docs
- Working on a `feature/*` branch might need experimental docs
- Subdirectory projects need their parent's docs as context

Directory-scoped dependencies solve this by letting Ariadne automatically detect which documentation sources are relevant based on:

1. **Your current working directory** - Subdirectory sources activate when you're inside them
2. **Your current git branch** - Branch-specific sources only activate on matching branches

## Configuration Reference

### SourceConfig Fields

Each source in `ariadne.yaml` supports these fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | required | Path to the source code directory |
| `depends_on` | list | `[]` | Explicit dependencies on other sources |
| `parent` | string | `null` | Parent source name (for subdirectory sources) |
| `branches` | list | `[]` | Git branch patterns where source is active |
| `ref` | string | `null` | Pin external dependency to specific branch/tag |

### Simple vs Full Form

Sources can be defined in two ways:

```yaml
sources:
  # Simple form: just a path (no advanced features)
  mylib: /path/to/mylib

  # Full form: with all options
  mylib:
    path: /path/to/mylib
    depends_on: [otherlib]
    parent: parentlib
    branches: ["feature/*"]
    ref: main
```

### Field Details

#### `path` (required)
The absolute path to the source code directory. Supports `~` expansion.

```yaml
mylib:
  path: ~/projects/mylib/src
```

#### `depends_on`
List of source names whose documentation should be loaded alongside this source. Dependencies are loaded transitively.

```yaml
mylib:
  path: /path/to/mylib
  depends_on: [core, utils]  # mylib docs include core and utils docs
```

#### `parent`
Declares this source as a subdirectory of another source. The child's path must be a subdirectory of the parent's path.

```yaml
pythonproject:
  path: /path/to/myproject

benchmark:
  path: /path/to/myproject/benchmark
  parent: pythonproject  # benchmark is inside pythonproject
```

When you use `parent`:
- Ariadne validates that the child path is under the parent path
- The parent is automatically added as an implicit dependency
- Working from the child directory activates both sources

#### `branches`
List of git branch patterns (using glob syntax) where this source is active. If empty or not set, the source is active on all branches.

```yaml
experimental:
  path: /path/to/experimental
  branches: ["feature/*", "develop"]  # Only active on feature branches or develop
```

Branch patterns support:
- Exact matches: `"main"`, `"develop"`
- Glob patterns: `"feature/*"`, `"release/*"`, `"hotfix-*"`
- Wildcard for all branches: `"*"`

#### `ref`
Pin an external dependency to a specific git ref (branch or tag). Useful for ensuring consistent documentation from external repos.

```yaml
external-lib:
  path: /path/to/external-lib
  ref: v2.0.0  # Always use docs from v2.0.0 tag
```

## Use Cases

### Monorepo with Subdirectory Projects

A common setup where subdirectories have their own focused documentation:

```yaml
sources:
  pythonproject:
    path: /path/to/myproject

  benchmark:
    path: /path/to/myproject/benchmark
    parent: pythonproject
    depends_on: [pythonproject]

  tools:
    path: /path/to/myproject/tools
    parent: pythonproject
    depends_on: [pythonproject]
```

When working in `myproject/benchmark/`:
- Ariadne detects `benchmark` as the active source
- `pythonproject` docs are loaded as context (via `depends_on`)

When working in `myproject/` (root):
- Only `pythonproject` is active

### Feature Branch Experimental Docs

Documentation that only appears when working on specific branches:

```yaml
sources:
  pythonproject:
    path: /path/to/myproject

  experimental-api:
    path: /path/to/myproject/experimental
    parent: pythonproject
    branches: ["feature/*", "develop"]
    depends_on: [pythonproject]
```

When on `main` branch:
- Only `pythonproject` docs are available

When on `feature/new-api` branch:
- Both `pythonproject` and `experimental-api` docs are available

### External Dependencies with Version Pinning

Referencing external repos with stable documentation versions:

```yaml
sources:
  myproject:
    path: /path/to/myproject

  external-lib:
    path: /path/to/external-lib
    ref: main  # Always use docs from main branch
```

## How It Works

### Scope Resolution Algorithm

When you run `ariadne manifest --auto-scope`, Ariadne:

1. Gets your current working directory
2. Gets your current git branch
3. For each configured source:
   - Checks if the source is active on the current branch
   - Checks if cwd is within the source's path
4. Returns the most specific match (longest path that contains cwd)

```python
# Simplified logic from get_source_scope()
for source in sources:
    if not source.is_active_on_branch(current_branch):
        continue
    if cwd.is_under(source.path):
        candidates.append(source)

return most_specific(candidates)  # Longest path wins
```

### Branch Pattern Matching

Branch patterns use Python's `fnmatch` module:

| Pattern | Matches |
|---------|---------|
| `main` | Exactly `main` |
| `feature/*` | `feature/foo`, `feature/bar` |
| `release-*` | `release-1.0`, `release-2.0` |
| `*` | Any branch |

### Effective Dependencies

When loading docs, Ariadne computes "effective dependencies" which include:

1. The parent source (if defined and active)
2. All explicit `depends_on` sources (if active on current branch)

This means `parent` creates an implicit dependency—you don't need to list it in `depends_on`.

## Conflict Resolution

When multiple sources provide documentation about the same topic, conflicts can arise.

### What Causes Conflicts

- Same document title in multiple sources
- Overlapping file coverage (same source files documented in parent and child)
- Duplicate content in branch-specific vs base sources

### Resolution Precedence

Ariadne resolves conflicts using these rules (highest priority first):

1. **Scope-specific wins** - If you're in a subdirectory, its docs take precedence over parent
2. **Branch-specific wins** - Branch-filtered docs override base docs
3. **Most recently updated wins** - Tiebreaker based on modification time

### Using `supersedes` Metadata

You can explicitly mark documents that supersede others:

```yaml
# In a document's frontmatter
---
title: Updated Caching Design
supersedes: old-caching-design
---
```

This ensures the new document is preferred when both are available.

## CLI Commands

### `ariadne manifest --auto-scope`

Outputs the manifest for the auto-detected source based on your cwd and branch:

```bash
# From /path/to/myproject/benchmark on main branch
$ ariadne manifest --auto-scope
# Shows: benchmark source manifest with pythonproject dependencies
```

Options:
- `--branch BRANCH` - Override branch detection
- `--no-branch-filter` - Disable branch filtering entirely
- `--limit N` - Limit number of documents shown

### `ariadne search --source --include-all`

Search with scope awareness:

```bash
# Search only in current scope
$ ariadne search "caching" --source benchmark

# Search across all sources regardless of scope
$ ariadne search "caching" --include-all
```

## Complete Example

A realistic configuration for a monorepo:

```yaml
default_source: pythonproject

sources:
  # Main library
  pythonproject:
    path: /path/to/myproject

  # Benchmarking suite (subdirectory)
  benchmark:
    path: /path/to/myproject/benchmark
    parent: pythonproject
    depends_on: [pythonproject]

  # Feature-branch experimental docs
  experimental:
    path: /path/to/myproject/experimental
    parent: pythonproject
    branches: ["feature/*", "develop"]
    depends_on: [pythonproject]

  # External research repo
  ao-research:
    path: /path/to/ao-research
    depends_on: [pythonproject]  # Research depends on pythonproject context
    ref: main               # Pin to main branch docs

docs_base: ./docs
```

### Verification

Test your configuration:

```bash
# Check hierarchy validation
$ ariadne config

# Test scope resolution from different directories
$ cd /path/to/myproject && ariadne manifest --auto-scope
$ cd /path/to/myproject/benchmark && ariadne manifest --auto-scope

# Test branch filtering
$ git checkout feature/new-api
$ ariadne manifest --auto-scope  # Should include experimental
```
