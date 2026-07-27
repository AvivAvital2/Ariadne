# MCP Tools (for agents)

The MCP server exposes 40+ tools (some multi-action) to LLM agents.

**Pick the right tool:**

| Question type | Tool |
|---|---|
| "How does X work?" / "What pattern is used for Y?" | `ariadne_search` (conceptual, semantic) |
| "What is this specific class/function?" | `ariadne_symbol` (qualified-name lookup) |
| "What does this whole file do?" | `ariadne_explain` (per-file explanation) |
| "Find code related to this concept" | `ariadne_themes` (cross-cutting clusters) |
| "What docs exist for X?" | `ariadne_list_all` |
| "What changed in this diff?" | `ariadne_diff_explain` |
| "What might break if I change X?" | `ariadne_impact_radius` |
| "What tests cover this?" | `ariadne_find_tests` |
| "Why is this error happening?" | `ariadne_diagnose` |
| "Show me this file's source" | `ariadne_read` (file) or `ariadne_body` (one element) |

**Disambiguating the three doc-shape tools:**

| Tool | Use when |
|---|---|
| `ariadne_explain` | You want the canonical per-file explanation doc (generate if missing) |
| `ariadne_expand` | An existing doc is too brief and you want it elaborated on a specific topic |
| `ariadne_summarize` | An existing doc is too long and you want a tighter version |

**Full tool reference, by category:**

**Retrieval & search**
| Tool | Purpose |
|------|---------|
| `ariadne_search` | Semantic search; auto-scopes to current source + dependencies |
| `ariadne_ask` | Ask a question and get a synthesized answer |
| `ariadne_symbol` | Look up a catalog element by qualified name |
| `ariadne_read` | Read a file via Ariadne's path-aware reader |
| `ariadne_body` | Get the source body of a catalog element |
| `ariadne_source_path` | Resolve a logical source name to its filesystem path |
| `ariadne_explain` | Generate/retrieve an explanation for a file |
| `ariadne_expand` | Expand an existing doc on a specific topic |
| `ariadne_summarize` | Summarize an existing doc more tightly |
| `ariadne_list_all` | List all documents including branch-specific/experimental |
| `ariadne_themes` | List / get / show cross-cutting themes |
| `ariadne_docs_read` | Read a generated doc by ID |

**Code understanding**
| Tool | Purpose |
|------|---------|
| `ariadne_diagnose` | Diagnose a failure mode (error message → likely causes) |
| `ariadne_diff_explain` | Summarize what changed in a diff |
| `ariadne_impact_radius` | Find code affected by a proposed change |
| `ariadne_graph` | Query the doc relationship graph |
| `ariadne_find_tests` | Locate tests covering a given file/symbol |
| `ariadne_test_suggestions` | Suggest tests to add for uncovered behavior |
| `ariadne_refactor_plan` | Plan a refactor across the codebase |
| `ariadne_review` / `ariadne_review_checklist` | Pre-merge review of changed files |
| `ariadne_debug_context` | Assemble debugging context for an issue |
| `ariadne_analyze_issue` | Walk through an issue with code context |

**Branch & sync state**
| Tool | Purpose |
|------|---------|
| `ariadne_branch_status` | Which docs are affected by current-branch changes |
| `ariadne_branch_sync` | Regenerate stale docs for the current branch |
| `ariadne_sync_status` | Show staleness state across the source |

**Telemetry & feedback**
| Tool | Purpose |
|------|---------|
| `ariadne_log_hit` / `ariadne_log_miss` | Report whether a result was useful |
| `ariadne_usage_stats` | Last-N-day usage statistics |
| `ariadne_gaps` | Documentation gaps with optional LLM analysis |
| `ariadne_coverage` | Per-file documentation coverage report |
| `ariadne_project_stats` | High-level project health snapshot |
| `ariadne_document_usage` | Per-document read-frequency stats |
| `ariadne_self_improve` | Suggest improvements for Ariadne itself |
| `ariadne_list_issues` | List known issues / open questions |

**Generation & contribution** (write paths)
| Tool | Purpose |
|------|---------|
| `ariadne_generate` / `ariadne_generate_docs` | Generate docs for a specific file |
| `ariadne_improve` | Run the improvement cycle on a target |
| `ariadne_contribute` | Save a finding back to the knowledge base |
| `ariadne_notify_changed` | Trigger incremental catalog refresh for changed files |
| `ariadne_merge` | Post-merge doc reconciliation |

Retrieval tools are read-only. Tracking tools record feedback. Administrative actions (full generation, cleanup, migration) require the CLI.

### Role-aware responses (optional)

`ariadne_search` and `ariadne_ask` accept an optional `role` parameter (default `'developer'`). The default behavior is unchanged — developer-level docs are what `ariadne generate` produces and what queries return today.

When the calling LLM reads phrasing like "from a product perspective" / "for the stakeholder review" / "in plain English" from the user's natural-language query, it can pass `role='product_manager'`. Ariadne then:

1. Looks for a cached audience-adapted response for that exact (audience, question) pair. Cache hit → return directly, no LLM call.
2. Cache miss → calls an LLM to translate the developer-level docs into a PM-friendly response (no code, focus on user-facing behavior and business impact), persists it as a new `audience_response` document, returns it.
3. Subsequent identical PM questions hit the cache.

The "explain further" property: dev-level content is always retrievable via a separate query (default `role='developer'`), so a PM who wants to dig into the technical detail can follow up without re-running anything.

Cache invalidation is lazy — at the next cache-lookup, if any parent dev doc has been updated since the audience_response was created, the stale row is dropped and the adapter re-runs. Dev doc updates never block on this; the freshness cost lives entirely on the PM-query path.

Role taxonomy is currently `developer` (default) + `product_manager`. Extension to other audiences is a config change, not a code change — see `designs/role-aware-responses.md`.

### Environment spools in responses (optional)

When a spool is enabled (`spools:` in `ariadne.yaml` — see [building-a-databricks-spool.md](building-a-databricks-spool.md)):

- `ariadne_search` results can include the environment's docs, clearly labeled with their `spool:<name>` source. Routing is a bidirectional lens — repo-subject questions rank your code first with the spool as a capped lens; environment-subject questions flip the sides — and the response's `lens_primary` field names which side owned the ranking (`repo` or `spool`).
- `ariadne_ask` answers carry an environment header with the runtime pin and component versions, a "Pinned version facts" block (`@Since`/deprecation data from the corpus) when the question touches a versioned symbol, and a provenance line citing each corpus pin + detected license.

> **Tip:** Search returns full document content, which may trigger Claude Code's "Large MCP response" warning. Set `MAX_MCP_OUTPUT_TOKENS=50000` in your `.claude/settings.local.json` `env` section to suppress it. See [claude-code-integration.md](claude-code-integration.md) for details.

See [claude-code-integration.md](claude-code-integration.md) for setup details.


---

_Part of the [Ariadne documentation](../README.md)._
