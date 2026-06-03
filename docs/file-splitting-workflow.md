# Token-Efficient File Splitting Workflow

## Problem

When splitting a large file (1000+ lines) into domain modules, multiple agents each re-read the entire file to extract their portion. For a 3800-line file with 3 agents, that's 11,400 lines of redundant reads.

## Solution: Ariadne as Shared Knowledge Cache

Use a 4-phase workflow where Ariadne stores a structured **file map** after one full read, and all subsequent agents query the map instead of re-reading the file.

### Phase 0: Pre-Index (one agent, one full read)

One agent reads the file once and creates a file map — a structured index stored in Ariadne via `ariadne_contribute`. The map contains:

```
# File Map: library.py (3855 lines)

## Domain: Core CRUD → library_core.py
- add_document: lines 321-388, deps: [_insert_document]
- _insert_document: lines 390-409, deps: []
- get_document: lines 411-430, deps: [_row_to_document]
...

## Domain: Search → library_search.py
- search: lines 789-822, deps: [list_documents, batch_dot_similarity]
- search_chunks: lines 824-892, deps: [get_documents_batch]
...
```

Each entry: method name, line range, dependencies, target module.

**Cost**: One full file read. Produces a ~200-line index.

### Phase 1: Plan (uses Ariadne only, zero file reads)

Query Ariadne: `ariadne_search("file map library.py domains")`. Gets the complete domain grouping and line ranges. Plan the split, verify no circular imports, create Myproject tasks with precise line ranges per module.

**Cost**: ~200 tokens for Ariadne query. Zero file reads.

### Phase 2: Extract (parallel workers, targeted reads only)

Each Myproject worker gets a task like:

> "Create library_core.py. Extract methods: add_document (321-388), get_document (411-430), ..."

The worker uses `Read(file, offset=321, limit=67)` for each method — reading only the exact lines needed.

**Cost per worker**: Sum of method line ranges (~300-500 lines), not the full file.

### Phase 3: Compose (one worker, reads file header only)

One worker reads the first ~200 lines (imports, class definition, `__init__`) and writes the thin composer that imports all mixins.

**Cost**: ~200 lines.

## Token Savings

| Approach | Lines Read | Agents |
|----------|-----------|--------|
| Naive (each agent reads full file) | 3×3800 = 11,400 | 3 |
| Ariadne-cached | 3800 + 300 + 500 + 400 + 200 = 5,200 | 5 |
| **Savings** | **~54%** | |

Savings scale with file size and number of target modules.

## File Map Format

Contribute to Ariadne as:
- **Title**: `File map: {filename} ({line_count} lines, {method_count} methods)`
- **Content type**: `finding`
- **Source files**: `[original_file_path]`

Searchable by filename via `ariadne_search`.

## Myproject Worker Prompt Addition

Workers assigned to file-split tasks should:
1. Check Ariadne first: `ariadne_search("file map {filename}")`
2. If a map exists, use targeted `Read(offset, limit)` instead of reading the full file
3. If no map exists, create one and contribute it before proceeding

## When to Use

- File is 500+ lines and needs splitting into 3+ modules
- Multiple methods need redistribution
- Internal cross-dependencies need tracking

## When NOT to Use

- Simple extract of one function (just read the relevant lines)
- File under 500 lines (map overhead exceeds savings)
