# Claude Code Integration

Set up Ariadne to provide documentation context to Claude Code.

> **New to Ariadne?** See [new-project-onboarding.md](new-project-onboarding.md) for a step-by-step guide covering the full setup from scratch.

## Quick Start

```bash
cd /path/to/your-project
ariadne init
```

This creates:
- `.claude/settings.json` - Session hook to display docs at startup
- CLAUDE.md snippet - Instructions for Claude to check docs first

## What Gets Created

### Session Hook

Displays the document manifest when starting a Claude Code session:

```
## Ariadne Knowledge Base Active

Source: your-source
Documents:
  - id: abc123
    title: "Core Data Abstractions"
  ...
```

### CLAUDE.md Instructions

Adds this to your CLAUDE.md:

```markdown
## Knowledge Base
Before exploring code, check Ariadne docs:
- `/path/to/docs/manifest.yaml` - Document index
- `/path/to/docs/explanations/` - How systems work
```

## Branch-Aware Filtering

Experimental docs can be hidden from main branch sessions.

Tag documents with metadata:
```yaml
metadata:
  status: experimental
  branches: ["feat/138-*"]
```

The session hook will only include experimental docs when working on matching branches.

## Verification

1. Start a Claude Code session
2. See "Ariadne Knowledge Base Active" message
3. Ask about code - Claude reads docs before exploring source

## MCP Server (Alternative)

Ariadne also provides an MCP (Model Context Protocol) server for on-demand documentation retrieval. Unlike the session hook which displays a static manifest at startup, the MCP server enables dynamic search and retrieval during a session.

### Hooks vs MCP

| Aspect | SessionStart Hook | MCP Server |
|--------|-------------------|------------|
| When | Session start only | On-demand |
| What | Static manifest display | Dynamic search & retrieval |
| Use case | Quick context awareness | Deep doc exploration |
| Latency | None after start | Per-request |

**Recommendation:** Use both together. The hook provides immediate context awareness; MCP enables deeper exploration when needed.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `ariadne_search` | Search docs with semantic query, branch-aware filtering |
| `ariadne_branch_status` | Show docs affected by current branch changes |
| `ariadne_branch_sync` | Regenerate stale docs for current branch |
| `ariadne_list_all` | List all documents with metadata |
| `ariadne_log_hit` | Report that a result was useful, with optional feedback |
| `ariadne_log_miss` | Report that a result wasn't helpful, with required feedback |
| `ariadne_usage_stats` | Show usage statistics and effectiveness metrics |
| `ariadne_gaps` | Generate gap/miss report with optional LLM analysis |

### Setup

#### Global Setup (Recommended)

Register Ariadne once at user scope so it's available in every Claude Code session:

```bash
# One-time setup — works across all projects
ariadne init --global

# Or manually via claude CLI:
claude mcp add -s user ariadne -- uv run --directory /path/to/Ariadne ariadne mcp
```

Verify with `claude mcp list` — you should see `ariadne: ... Connected`.

#### Per-Project Setup

Alternatively, add a `.mcp.json` to a specific project:

```bash
ariadne init --target /path/to/your-project
```

This creates `.mcp.json` in the project directory:

```json
{
  "mcpServers": {
    "ariadne": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Ariadne", "ariadne", "mcp"]
    }
  }
}
```

Restart Claude Code. You should see `mcp__ariadne__*` tools available.

### Usage Examples

Once configured, Claude can use these tools:

```
# Search for docs about caching
mcp__ariadne__ariadne_search(query="caching strategy")

# Check which docs are affected by branch changes
mcp__ariadne__ariadne_branch_status()

# List all available docs
mcp__ariadne__ariadne_list_all()
```

### Response Size

Search results include full document content for best precision — Claude sees everything and can reason across all results. This can trigger Claude Code's "Large MCP response" warning for broad queries.

The warning is informational only. To suppress it, set `MAX_MCP_OUTPUT_TOKENS` in your project's `.claude/settings.local.json`:

```json
{
  "env": {
    "MAX_MCP_OUTPUT_TOKENS": "50000"
  }
}
```

### When to Use MCP

- **Deep dives**: When you need to search for specific documentation
- **Branch work**: To see which docs are affected by your changes
- **Debugging**: To retrieve full document content, not just titles
- **Multi-source**: To query across multiple documentation sources

## How MCP Instructions Work

The MCP server includes a built-in `instructions` field (set in `ariadne_mcp/server.py` via `FastMCP(..., instructions=...)`). When Claude Code connects to the server, this directive is injected as a system-level instruction telling Claude to:

1. Query Ariadne **before** using any other codebase exploration tool (Grep, Glob, Read, etc.)
2. Use search results to guide follow-up code reads
3. Report feedback via `ariadne_log_hit` / `ariadne_log_miss`

This means **even without CLAUDE.md guidance or session hooks**, Claude will prioritize Ariadne when MCP is connected. The hooks and CLAUDE.md provide additional context (the doc manifest at startup, instructions on saving findings), but the MCP instructions are the primary behavioral driver.

## Usage Tracking & Feedback

Ariadne tracks MCP tool usage to measure documentation effectiveness and identify gaps.

### How It Works

1. **Auto-logging**: Every MCP tool call is automatically recorded with a unique event ID
2. **Event ID in output**: Each tool result ends with `[Usage event: <id>]`
3. **Claude reports back**: After using results, Claude calls:
   - `ariadne_log_hit(event_id, feedback?)` — result was useful
   - `ariadne_log_miss(event_id, feedback)` — result wasn't helpful (feedback required)
4. **Query statistics**: Use `ariadne_usage_stats` (MCP) or `ariadne usage` (CLI)
5. **Gap analysis**: Use `ariadne_gaps` to see what documentation is missing

### Behavioral Directive

The `mention_ariadne` configuration option injects a behavioral directive into the session manifest, telling Claude to casually mention when Ariadne helped answer a question. This is enabled by default.

```yaml
# In ariadne.yaml
mention_ariadne:
  enabled: true
  # message: "custom directive text..."  # Optional override
```

The directive appears in the manifest output alongside the document list, so Claude sees it at the start of every session.
