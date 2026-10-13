# Ariadne Improvement Backlog — Business Logic & UX Focus

## Objective
Make Ariadne a polished, secure, user-friendly library for humans and machines.

## Security
- [x] Hardcoded OpenAI API URL in embedding.py — should be configurable
- [ ] DELETE FROM doc_graph without WHERE in build_graph — could delete wrong data on failure
- [ ] Validate file paths in explain/coverage to prevent path traversal
- [ ] Sanitize user input in search queries before logging to DB
- [ ] API key not redacted in error messages

## User Experience (CLI)
- [ ] `ariadne search` with no query shows unhelpful usage error — should list recent/popular docs
- [ ] `ariadne generate` with no source hangs — should error immediately
- [ ] `ariadne get <bad-id>` should say "not found" not fail silently
- [ ] `ariadne delete` should require confirmation
- [ ] Progress bars for long operations (generate, rebuild, sync)
- [ ] Consistent error message format across all commands
- [ ] `ariadne doctor` command — health check for DB, config, embeddings, staleness
- [ ] Colorized diff output for `ariadne check` — show what's stale vs current

## User Experience (MCP)
- [ ] ariadne_explain should trim content like search does (Related Documents)
- [ ] MCP error responses should include actionable suggestions
- [ ] ariadne_search with no results should auto-try broader query

## Business Logic
- [ ] `ariadne explain` should order docs by type: explanation first, then architecture, findings
- [ ] `ariadne improve` should report estimated API cost before generating
- [ ] `ariadne sync` should auto-rebuild graph after syncing
- [ ] `ariadne check` should use graph priorities to highlight most important stale files
- [ ] Search should penalize stale documents (source changed since doc generated)
- [ ] `ariadne history <file>` — show how docs evolved over time for a file
- [ ] Source file validation — warn when source_files paths don't exist on disk
- [ ] Auto-detect when a new file is added to the source and suggest generating docs

## Data Integrity
- [ ] Validate document IDs are valid UUIDs
- [ ] Ensure chunks are consistent with their parent document
- [ ] Detect orphaned chunks (parent document deleted)
- [ ] Transaction safety: wrap multi-step operations in transactions
