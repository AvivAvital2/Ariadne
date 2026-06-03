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


class SearchResponse(BaseModel):
    """Response from ariadne_search."""

    documents: list[DocumentResult] = Field(default_factory=list)
    event_id: int
    suggested_queries: list[str] | None = None  # Populated when few/no results found
    improvement_hint: str | None = None  # Hint when library has coverage gaps
    truncated: bool = False  # True when full content exceeded response_token_budget; call ariadne_expand(event_id) for full docs


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
