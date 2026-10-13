# New Project Onboarding

End-to-end guide for connecting a new codebase to Ariadne so Claude Code can search your documentation on demand.

## Prerequisites

- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)** installed
- **Ariadne** cloned:
  ```bash
  git clone https://github.com/your-org/ariadne.git
  ```
- **OpenAI API key** (used for cloud doc generation and semantic search embeddings):
  ```bash
  export OPENAI_API_KEY=sk-...
  ```
  Tip: Put this in a `.env` file in the Ariadne directory — the CLI loads it automatically via `python-dotenv`, and Ariadne's `.gitignore` already excludes `.env` so the key is never committed. (Generating with Claude needs `ANTHROPIC_API_KEY` too; OpenAI generation needs only this key — see the README's API-keys section.) The local Qwen pair needs its two model endpoints, with no cloud-key requirement. See [supported models](supported-models.md) for the verified pairs, exact YAML and qualification limits.
- **Claude Code** installed and working

> **Running commands**: This guide uses bare `ariadne` commands, which requires installing Ariadne as a tool: `uv tool install /path/to/Ariadne`. If you haven't done that, prefix all commands with `uv run --directory /path/to/Ariadne` instead (e.g., `uv run --directory /path/to/Ariadne ariadne generate`).

## Step 1: Configure the Source

Add your project to `ariadne.yaml` in the Ariadne directory:

```yaml
default_source: myproject

sources:
  myproject: /path/to/myproject/src

docs_base: ./docs
```

Add the `llm`, `embeddings` and database settings from the [model configuration guide](supported-models.md). GPT-6 Astra and Claude Opus 5.5 use OpenAI embeddings; the local Qwen pair uses its own embedding endpoint. Give each pair separate main/staleness paths. `ARIADNE_CONFIG` selects a configuration outside the default `ariadne.yaml` search path; use it for both CLI commands and the MCP process.

The `path` should point to your source code root. For more advanced setups (monorepo subdirectories, branch filtering, dependencies), see the [Source Configuration Schema](../README.md#source-configuration-fields) in the README.

## Step 2: Generate Documentation

```bash
# Generate docs from source code
ariadne generate --source myproject

# Export to markdown files under docs/myproject/
ariadne export --source myproject

# Verify documents were created
ariadne list --source myproject
```

You should see a list of generated documents (explanations, architecture docs, etc.).

**What to expect:**
- Generation uses `llm.model` in the selected configuration, for example `claude-opus-5-5`, `gpt-6-astra` or the fixed local `Qwen/Qwen3.8-27B`. Legacy cloud `defaults.model` remains supported.
- Duration depends on codebase size — a small project takes a few minutes, a large one can take 10-30 minutes
- Hosted concurrency is configurable: `ariadne generate --concurrency 5` (default: 3). The local pair uses concurrency one.
- Resume interrupted local durable runs with `ariadne batch resume local:RUN_UUID`. For live generation, rerun the command; completed, current work is skipped.
- Use `--dry-run` to preview what would be generated without making API calls
- Use `--verbose` to see detailed validation output for any failures

## Step 3: Initialize the Integration

`ariadne init` is the all-in-one setup command. It creates session hooks, CLAUDE.md instructions, and registers the MCP server.

### Automated Setup (Recommended)

```bash
# Register MCP globally + create hooks in your project
ariadne init --source myproject --target /path/to/myproject --global
```

This does three things:
1. **Registers the MCP server** at user scope (available in all Claude Code sessions)
2. **Creates `.claude/settings.json`** in the target project with:
   - **SessionStart hook** — runs `ariadne manifest --auto-scope` to display the doc index at session start
   - **PostToolUse hook** — syncs CLAUDE.md edits back to Ariadne
3. **Appends to CLAUDE.md** — instructions telling Claude to check Ariadne docs before exploring source code

### Per-Project MCP (Alternative)

If you don't want global MCP registration, omit `--global`:

```bash
ariadne init --source myproject --target /path/to/myproject
```

This creates a `.mcp.json` in the target project instead of registering globally:

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

### Manual MCP Registration

If you prefer to register the MCP server manually without creating hooks:

```bash
claude mcp add -s user ariadne -- uv run --directory /path/to/Ariadne ariadne mcp
```

Note: With manual registration, you won't get session hooks or CLAUDE.md instructions. The MCP server's built-in `instructions` directive still tells Claude to use Ariadne tools first (see [How MCP Instructions Work](claude-code-integration.md#how-mcp-instructions-work)), but you miss the session-start manifest display.

### Verify MCP Connection

```bash
claude mcp list
```

You should see: `ariadne: ... Connected`

## Step 4: Grant MCP Permissions (Optional)

Without permissions, Claude will prompt for approval each time it calls an MCP tool. To auto-allow the most common read-only tools, add them to `.claude/settings.local.json` in your project:

```json
{
  "permissions": {
    "allow": [
      "mcp__ariadne__ariadne_search",
      "mcp__ariadne__ariadne_list_all",
      "mcp__ariadne__ariadne_sync_status",
      "mcp__ariadne__ariadne_log_hit"
    ]
  }
}
```

For the full list of available tools, see [Available MCP Tools](claude-code-integration.md#available-mcp-tools).

## Step 5: Verify End-to-End

1. **Restart Claude Code** in your target project
2. **Check the session banner** — you should see "Ariadne Knowledge Base Active" with a document list (requires session hooks from Step 3)
3. **Test MCP tools** — ask Claude something about your codebase; it should call `ariadne_search` before grepping source files
4. **Confirm connection** — run `claude mcp list` and verify `ariadne` shows as connected

## Keeping Docs Fresh

After code changes, sync documentation:

```bash
# Check what needs updating
ariadne sync --status --source myproject

# Sync with git changes
ariadne sync --source myproject
```

For automatic sync after every commit, set up a git post-commit hook. See [Automatic Sync via Git Hook](../README.md#automatic-sync-via-git-hook) in the README.

## Troubleshooting

**MCP server not connecting:**
- Verify `uv` is on your PATH: `which uv`
- Test the server manually: `uv run --directory /path/to/Ariadne ariadne mcp` (should hang waiting for stdin — Ctrl+C to exit)
- Check `claude mcp list` for error messages

**No documents returned from search:**
- Verify docs exist: `ariadne list --source myproject`
- Check the source path in `ariadne.yaml` points to the correct directory
- Ensure `ariadne generate` and `ariadne export` completed without errors
- Check if branch filtering is hiding docs: try `ariadne search "your query" --include-all`

**Generation fails or produces no output:**
- For cloud operation, ensure `OPENAI_API_KEY` is set in your environment or in Ariadne's `.env` file; Claude generation also needs `ANTHROPIC_API_KEY`
- If using `.env`, verify it's in the Ariadne directory (not the target project)
- For local operation, run `ariadne doctor --models` and verify both model and tokenization endpoints
- Check that `llm.provider`, `llm.model`, embeddings and artifact pins match the selected pair and the library binding; changing models needs fresh databases
- Run with `--verbose` to see detailed error output: `ariadne generate --source myproject --verbose`

**Session hook not firing:**
- Check `.claude/settings.json` exists in the target project with a `SessionStart` hook
- Verify `ariadne manifest --auto-scope` works from the target project directory
- The hook only fires on `startup`, `resume`, and `clear` events

**Claude has MCP tools but doesn't use them:**
- The MCP server includes an `instructions` directive that tells Claude to query Ariadne first — this works automatically
- If Claude still ignores Ariadne, ensure your CLAUDE.md includes the Ariadne integration section (created by `ariadne init`)
- Check that permissions in `.claude/settings.local.json` don't block the tools

## Local model onboarding

For the local pair, follow [local models](local-models.md) and use a fresh database pair. `ariadne onboard --source NAME` performs only deterministic preview; it does not discover/index into the target corpus or generate catalog embeddings. `--approve` binds the pair and executes structural/catalog work, descriptions, generation, reverse augmentation, themes and cross-references as a durable foreground run. Keep the printed `local:` ID and resume it with `ariadne batch resume ID`. Local runs have no provider batch discount or 24-hour SLA. The supplied deployment remains a candidate until hardware qualification is recorded.
