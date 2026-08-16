"""Pydantic response models for the Ariadne MCP server.

Each model defines a structured output schema that the MCP SDK
exposes to clients alongside the tool definition.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SectionResult(BaseModel):
    """A single section from a document, returned in sections_only mode."""

    heading: str
    description: str
    content: str
    score: float | None = None


class DocumentResult(BaseModel):
    """A single document returned from a search."""

    id: str
    title: str
    content_type: str
    content: str
    source_files: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    score: float | None = None  # Similarity score (0-1) when ranked by embedding
    sections: list[SectionResult] | None = None  # Populated in sections_only mode
    source_name: str | None = None  # Origin source (spool vs user) — CRIT-6 framing


class SearchResponse(BaseModel):
    """Response from ariadne_search."""

    documents: list[DocumentResult] = Field(default_factory=list)
    event_id: int
    suggested_queries: list[str] | None = None  # Populated when few/no results found
    improvement_hint: str | None = None  # Hint when library has coverage gaps
    truncated: bool = False  # True when full content exceeded response_token_budget; call ariadne_expand(event_id) for full docs
    spool_connections: dict[str, str] | None = None  # doc_id -> lens connection label ('entity(<term>)' / 'semantic(<cosine>)') for spool docs admitted by the router
    # Bidirectional lens: which side got the full embedding ranking on a
    # routed question ('repo' | 'spool'); None when unrouted.
    lens_primary: str | None = None


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

class DocumentSummary(BaseModel):
    """Compact document info for listing."""

    id: str
    title: str
    content_type: str
    status: str = 'stable'
    branches: list[str] = Field(default_factory=list)
    expires_at: str | None = None
    is_expired: bool = False
    source_files: list[str] = Field(default_factory=list)


class ListResponse(BaseModel):
    """Response from ariadne_list_all."""

    total: int
    documents: list[DocumentSummary] = Field(default_factory=list)
    event_id: int


# ---------------------------------------------------------------------------
# Branch status
# ---------------------------------------------------------------------------

class AffectedDocument(BaseModel):
    """Document affected by branch changes."""

    id: str
    title: str
    content_type: str


class BranchStatusResponse(BaseModel):
    """Response from ariadne_branch_status."""

    branch: str
    comparing_against: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    affected_documents: list[AffectedDocument] = Field(default_factory=list)
    message: str | None = None
    event_id: int


# ---------------------------------------------------------------------------
# Sync status
# ---------------------------------------------------------------------------

class SyncStatusResponse(BaseModel):
    """Response from ariadne_sync_status."""

    source: str
    status: str  # 'synced' | 'never_synced' | 'error'
    git_hash: str | None = None
    synced_at: str | None = None
    commit_message: str | None = None
    event_id: int


# ---------------------------------------------------------------------------
# Usage stats
# ---------------------------------------------------------------------------

class ToolStats(BaseModel):
    """Per-tool usage breakdown."""

    calls: int
    hits: int
    misses: int
    hit_rate: float


class DayStats(BaseModel):
    """Per-day usage breakdown."""

    date: str
    calls: int
    hits: int
    misses: int


class FeedbackEntry(BaseModel):
    """A single feedback record."""

    timestamp: str
    tool_name: str
    query: str
    outcome: str
    feedback: str


class UsageStatsResponse(BaseModel):
    """Response from ariadne_usage_stats."""

    total_calls: int
    total_hits: int
    total_misses: int
    hit_rate: float
    avg_calls_per_day: float
    by_tool: dict[str, ToolStats] = Field(default_factory=dict)
    by_day: list[DayStats] = Field(default_factory=list)
    recent_feedback: list[FeedbackEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Gap report
# ---------------------------------------------------------------------------

class GapEntry(BaseModel):
    """A documentation gap identified from miss feedback."""

    feedback: str
    count: int
    example_queries: list[str] = Field(default_factory=list)
    last_seen: str


class MissEntry(BaseModel):
    """A recent miss event."""

    timestamp: str
    tool_name: str
    query: str
    feedback: str


class GapReportResponse(BaseModel):
    """Response from ariadne_gaps."""

    total_misses: int
    miss_rate: float
    top_gaps: list[GapEntry] = Field(default_factory=list)
    recent_misses: list[MissEntry] = Field(default_factory=list)
    analysis: str | None = None


# ---------------------------------------------------------------------------
# Simple responses
# ---------------------------------------------------------------------------

class FeedbackResponse(BaseModel):
    """Response from ariadne_log_hit / ariadne_log_miss."""

    success: bool
    message: str


class AdminActionResponse(BaseModel):
    """Response from admin tools (branch_sync, generate, merge)."""

    output: str


class GraphStatsResponse(BaseModel):
    """Response from ariadne_graph(action='stats')."""

    total_edges: int
    by_type: dict[str, int] = Field(default_factory=dict)
    source_nodes: int
    target_nodes: int


class PriorityEntry(BaseModel):
    """A file with its priority score."""

    file: str
    total_edges: int
    doc_count: int
    coverage_percent: float
    priority_score: float


class GraphResponse(BaseModel):
    """Response from ariadne_graph."""

    action: str
    output: str | None = None
    file_path: str | None = None
    stats: GraphStatsResponse | None = None
    priorities: list[PriorityEntry] | None = None


class IssueAnalysisResponse(BaseModel):
    """Response from ariadne_analyze_issue."""

    issue_title: str
    issue_number: int
    relevant_docs: list[str] = Field(default_factory=list)
    relevant_files: list[str] = Field(default_factory=list)
    proposal: str = ''
    confidence: str = 'medium'


class AskResponse(BaseModel):
    """Response from ariadne_ask."""

    answer: str
    sources: list[str] = Field(default_factory=list)  # Doc titles used
    confidence: str = 'medium'  # 'high', 'medium', 'low' based on doc match quality
    event_id: int = 0
    #: Chain hops behind the answer: qualified_name, file, line, relation, hop,
    #: call_site, stop_reason. Additive — `sources` keeps doc titles, so every
    #: existing caller is unaffected.
    citations: list[dict] = Field(default_factory=list)


class ExplainDocument(BaseModel):
    """A document in an explain response."""

    id: str
    title: str
    content: str


class ExplainResponse(BaseModel):
    """Response from ariadne_explain."""

    file: str
    summary: str
    total_documents: int
    types_found: list[str] = Field(default_factory=list)
    documents: dict[str, list[ExplainDocument]] = Field(default_factory=dict)
    graph_neighbors: list[dict[str, str]] = Field(default_factory=list)
    offset: int = 0
    returned: int = 0
    next_offset: int | None = None


class CoverageResponse(BaseModel):
    """Response from ariadne_coverage."""

    source: str
    total_files: int
    documented_count: int
    undocumented_count: int
    coverage_percent: float
    undocumented_files: list[str] = Field(default_factory=list)


class ContributeResponse(BaseModel):
    """Response from ariadne_contribute."""

    success: bool
    document_id: str | None = None
    title: str | None = None
    message: str


# ---------------------------------------------------------------------------
# Project stats
# ---------------------------------------------------------------------------

class SourceStats(BaseModel):
    """Per-source document statistics."""

    doc_count: int
    content_size: int
    embedding_size: int
    chunk_count: int = 0


class ProjectStatsResponse(BaseModel):
    """Response from ariadne_project_stats."""

    by_source: dict[str, SourceStats] = Field(default_factory=dict)
    total_documents: int
    total_content_size: int
    db_size_bytes: int


class DocumentServeStats(BaseModel):
    """Per-document serving statistics."""

    document_id: str
    title: str
    content_type: str
    serve_count: int


class DocumentUsageResponse(BaseModel):
    """Response from ariadne_document_usage."""

    documents: list[DocumentServeStats] = Field(default_factory=list)
    days: int | None = None


# ---------------------------------------------------------------------------
# Architecture Review (Code Inspector)
# ---------------------------------------------------------------------------


class FileReviewContext(BaseModel):
    """Per-file context for architecture review."""

    file_path: str
    summary: str = ''
    impact_score: float = 0.0
    direct_dependents: int = 0
    top_dependents: list[str] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    """Composite architecture review combining search, explain, impact, checklist."""

    relevant_docs: list[DocumentResult] = Field(default_factory=list)
    file_contexts: list[FileReviewContext] = Field(default_factory=list)
    checklist_items: list[dict[str, str]] = Field(default_factory=list)
    architectural_concerns: list[str] = Field(default_factory=list)
    event_id: int = 0
    design_alignment_verdict: str | None = None
    divergence_reasons: list[str] = Field(default_factory=list)


class FileExplanation(BaseModel):
    """Per-file explanation with relevant content chunks for task context."""

    file_path: str
    summary: str = ''
    total_documents: int = 0
    types_found: list[str] = Field(default_factory=list)
    chunks: list[dict[str, str]] = Field(default_factory=list)
    '''Relevant content chunks from Ariadne docs — each has 'title' and 'content'.'''


class FileTests(BaseModel):
    """Test files related to a source file."""

    file_path: str
    tests: list[dict[str, str]] = Field(default_factory=list)


class TaskContextResponse(BaseModel):
    """One-shot briefing bundling search, explain, checklist, and tests."""

    search_results: list[DocumentResult] = Field(default_factory=list)
    file_explanations: list[FileExplanation] = Field(default_factory=list)
    checklist_items: list[dict[str, str]] = Field(default_factory=list)
    file_tests: list[FileTests] = Field(default_factory=list)
    event_id: int = 0


# ---------------------------------------------------------------------------
# Onboarding — source add
# ---------------------------------------------------------------------------


class GitInfo(BaseModel):
    """Filesystem + git metadata for a candidate source directory.

    Drives the onboarding "Connect" screen's "Git repository detected"
    banner. ``file_count``/``size_bytes`` come from a working-tree walk and
    are always populated; ``branch``/``last_commit_relative`` are None when
    the path is not a git work tree.
    """

    is_repo: bool
    branch: str | None = None
    file_count: int | None = None
    size_bytes: int | None = None
    last_commit_relative: str | None = None


class SourceAddResponse(BaseModel):
    """Response from ariadne_source_add — the persisted source config."""

    source: str
    path: str | None = None
    created: bool  # True when a new source was created, False on update
    is_default: bool
    depends_on: list[str] = Field(default_factory=list)
    parent: str | None = None
    branches: list[str] = Field(default_factory=list)
    ref: str | None = None
    exclude: list[str] = Field(default_factory=list)
    exclude_dirs: list[str] = Field(default_factory=list)
    exempt_dirs: list[str] = Field(default_factory=list)
    ignore_staleness: bool | list[str] = False
    doc_types_by_language: dict[str, list[str]] = Field(default_factory=dict)
    git: GitInfo | None = None


# ---------------------------------------------------------------------------
# Onboarding — cost estimate (the "Preview" step)
# ---------------------------------------------------------------------------


class LanguageCount(BaseModel):
    """One bar of the detected-language histogram."""

    language: str
    files: int
    percent: float


class ModelPrice(BaseModel):
    """A selectable LLM with its per-million-token rates."""

    model: str
    input_per_million: float
    output_per_million: float


class DocTypeCostModel(BaseModel):
    """Cost of generating one doc type across the whole source."""

    doc_type: str
    count: int  # generation calls for this type
    cost_usd: float
    cost_batched_usd: float


class DirCostModel(BaseModel):
    """Per-directory (and per-file) cost node — the explorer tree."""

    rel_path: str
    docs: int
    total_usd: float
    ingestion_usd: float


class ExclusionSaving(BaseModel):
    """What one configured exclusion (or force-include) does to the cost.

    ``saved_usd`` is how much the setting removes from the generate scope:
    positive for an exclude glob / excluded dir (you pay that much less),
    negative for an exempt (force-included) dir (it ADDS that much).
    """

    pattern: str
    kind: str  # 'glob' | 'dir' | 'exempt'
    files: int
    saved_usd: float
    saved_batched_usd: float


class EstimateResponse(BaseModel):
    """Response from ariadne_estimate — a no-LLM cost preview."""

    source: str
    model: str
    input_per_million: float
    output_per_million: float
    file_count: int
    total_calls: int
    input_tokens: int
    output_tokens: int
    embedding_tokens: int
    total_cost_usd: float           # generate-phase only (unchanged meaning)
    total_cost_batched_usd: float   # generate-phase only, batched
    # catalog-describe phase (index-free element count). Separate line item so
    # the generate-only totals above keep their meaning; None when the model
    # has no pricing entry.
    catalog_describe_cost_usd: float | None = None
    catalog_describe_cost_batched_usd: float | None = None
    catalog_element_count: int = 0
    # generate + catalog-describe — the true minimum the run will cost.
    grand_total_cost_usd: float = 0.0
    grand_total_cost_batched_usd: float = 0.0
    cost_lower_bound: float
    cost_upper_bound: float
    embedding_cost_usd: float
    languages: list[LanguageCount] = Field(default_factory=list)
    by_doc_type: list[DocTypeCostModel] = Field(default_factory=list)
    by_directory: list[DirCostModel] = Field(default_factory=list)
    available_models: list[ModelPrice] = Field(default_factory=list)
    language_doc_types: dict[str, list[str]] = Field(default_factory=dict)
    exclusion_savings: list[ExclusionSaving] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Onboarding — discover (the "Discover" step)
# ---------------------------------------------------------------------------


class IndexerPlan(BaseModel):
    """One SCIP indexer the discover step detected and will run."""

    kind: str  # python | typescript | java
    cwd: str
    markers: list[str] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)  # real package name per marker
    entry_kind: str  # package | scripts


class DiscoverResponse(BaseModel):
    """Response from ariadne_discover — language histogram + index plan."""

    source: str
    file_count: int
    dir_count: int
    languages: list[LanguageCount] = Field(default_factory=list)
    indexers: list[IndexerPlan] = Field(default_factory=list)
    index_kinds: list[str] = Field(default_factory=list)
    manifest_written: bool


# ---------------------------------------------------------------------------
# Onboarding — list configured sources (the dependency picker)
# ---------------------------------------------------------------------------


class SourceEntry(BaseModel):
    """A configured source in ariadne.yaml."""

    name: str
    path: str | None = None
    is_default: bool = False
    depends_on: list[str] = Field(default_factory=list)


class SourceListResponse(BaseModel):
    """Response from ariadne_list_sources."""

    sources: list[SourceEntry] = Field(default_factory=list)
    default_source: str | None = None


class OnboardResponse(BaseModel):
    """Response from ariadne_onboard — the paid build's "ready" stats.

    ``ariadne_onboard`` runs the paid phases (catalog-describe → generate →
    themes-build) and reports what the Step-5 "ready" screen shows: files
    indexed, docs written, themes found, and the resulting coverage. Not
    idempotent — it spends LLM budget and writes documents.
    """

    source: str
    model: str
    mode: str  # 'live' | 'batch'
    files_indexed: int
    docs_written: int
    themes_found: int
    coverage_percent: float
    doc_types: list[str] = Field(default_factory=list)
    themes_ok: bool = True
