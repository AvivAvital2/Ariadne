"""Theme summarization (Themes plan, Phase 4).

Replaces the placeholder theme document Phase 3 created with real LLM-generated
markdown describing the cluster's cross-cutting concern. Coherence detection:
if the LLM returns a markdown doc starting with `#`, the theme is coherent;
if it returns `INCOHERENT`, the theme is marked coherent=0 and no doc is
written (the placeholder stays in place but is filtered from search by default).

**Operates below the closure chokepoint.** Theme docs are cross-source by
design (a cluster can span members from any indexed source); their
``source_name`` column is ``None`` and the LLM summary needs every
member's text regardless of source. Raw ``library.X(...)`` access here
is intentional — see ``designs/directional-closure-scoping.md`` §
"Library-internal modules — legitimately unscoped".
"""
from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

from docgen.cluster import cluster_themes
from docgen.prompts import THEME_SYSTEM_PROMPT, THEME_USER_TEMPLATE
from llm import chat_complete
from schema import _now_iso

if TYPE_CHECKING:
    from docgen.llm.batch import BatchStrategy
    from library import Library
    from writer import LibraryWriter


# Edge types whose weights count when scoring an element's neighbor clusters
# during local_reassign.
_REASSIGN_EDGE_TYPES = ('imports', 'documents', 'topic_member', 'semantic_neighbor')

# Default fan-out for live theme summarization. Each summarize_theme call is an
# LLM completion + embedding write — async network I/O, NOT CPU-bound — so this
# caps concurrent in-flight requests to the provider, bounded by its rate
# limits (hence a small fixed number, NOT multiprocessing.cpu_count()). Theme
# summaries are independent per cluster, so we always parallelize at this width;
# ``concurrency=None`` (the signal onboard threads through) resolves to it.
DEFAULT_SUMMARIZE_CONCURRENCY = 4


# ---------------------------------------------------------------------------
# Summary hash (per plan §4.5)
# ---------------------------------------------------------------------------


def compute_summary_hash(
    members: list[str],
    summaries: list[str],
    resolution: float,
) -> str:
    """Deterministic hash of the inputs that drive a theme summary.

    Sorted by member id so member-order shuffles don't change the hash;
    the hash changes when any member's summary text or the resolution changes.
    Drives the "needs re-summary?" gate.
    """
    h = hashlib.blake2b(digest_size=16)
    paired = sorted(zip(members, summaries), key=lambda p: p[0])
    for mid, summary in paired:
        h.update(mid.encode('utf-8'))
        h.update(b'\x00')
        h.update(summary.encode('utf-8'))
        h.update(b'\x00')
    h.update(f'{resolution:.4f}'.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gather_member_data(
    library: "Library", cluster_id: str,
) -> tuple[list[str], list[str]]:
    """Return (sorted_member_ids, summaries) — summaries aligned to ids."""
    members = library.get_theme_members(cluster_id)
    member_ids = sorted(eid for eid, _ in members)
    summaries: list[str] = []
    for mid in member_ids:
        doc = library.get_document(mid)
        if doc is None:
            summaries.append('')
            continue
        meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        desc = meta.get('description')
        if desc:
            summaries.append(str(desc)[:300])
        else:
            summaries.append((doc.content or '')[:300])
    return member_ids, summaries


def _format_member_list(member_ids: list[str], summaries: list[str]) -> str:
    lines: list[str] = []
    for mid, summary in zip(member_ids, summaries):
        if summary:
            lines.append(f'- `{mid}` — {summary}')
        else:
            lines.append(f'- `{mid}`')
    return '\n'.join(lines)


def _set_theme_coherent(library: "Library", cluster_id: str, coherent: bool) -> None:
    """Direct UPDATE — exposed as a helper since ThemesMixin doesn't have it."""
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE themes SET coherent = ?, dirty = 0 WHERE cluster_id = ?',
            (int(coherent), cluster_id),
        )


class _ThemeRequest(NamedTuple):
    """One theme's prompt plus the bookkeeping needed to apply the response.

    Built by read-only DB access (:func:`_build_theme_request`) so a whole
    run's worth can be assembled up front and dispatched as a single batch,
    then persisted by :func:`_apply_theme_response`. The live and batched
    paths share both halves so they store identically.
    """
    cluster_id: str
    system_prompt: str
    user_prompt: str
    member_ids: list[str]
    summaries: list[str]
    resolution: float
    doc_id: str


def _build_theme_request(
    library: "Library", cluster_id: str,
) -> "_ThemeRequest | None":
    """Assemble the (system, user) prompt + apply-time data for one theme.

    Returns ``None`` if the theme row is gone. Pure read — no LLM, no
    writes — so callers can build many up front for batch dispatch.
    """
    theme = library.get_theme(cluster_id)
    if theme is None:
        return None
    member_ids, summaries = _gather_member_data(library, cluster_id)
    member_list = _format_member_list(member_ids, summaries)
    user_prompt = THEME_USER_TEMPLATE.format(
        n_members=len(member_ids),
        k_shown=len(member_ids),
        member_list_with_summaries=member_list,
        code_snippets='(omitted)',
        impact_summaries='(omitted)',
    )
    return _ThemeRequest(
        cluster_id=cluster_id,
        system_prompt=THEME_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        member_ids=member_ids,
        summaries=summaries,
        resolution=theme.resolution,
        doc_id=theme.doc_id,
    )


async def _apply_theme_response(
    library: "Library",
    writer: "LibraryWriter",
    req: "_ThemeRequest",
    response: str,
) -> bool:
    """Persist one theme-summary response. Returns True if coherent (doc
    written, theme marked clean + coherent), False on an INCOHERENT verdict.

    This is the original ``summarize_theme`` persistence logic verbatim, so
    the live and batched paths store identically. Callers pass a real
    response string: the live path forwards ``chat_complete``'s return
    as-is (a ``None`` there raises, exactly as before), and the batched
    path filters errored (``None``) results out before calling this.
    """
    response = response.strip()
    if not response:
        # An empty (or whitespace-only) completion isn't a usable doc. Leave the
        # theme dirty — retried on the next run — rather than overwriting the
        # placeholder with empty content (which update_document rejects) or
        # crashing the summary.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            'summarize_theme: empty response for cluster %s — left pending',
            req.cluster_id,
        )
        return False
    if response.startswith('INCOHERENT'):
        _set_theme_coherent(library, req.cluster_id, coherent=False)
        return False

    new_hash = compute_summary_hash(
        req.member_ids, req.summaries, req.resolution,
    )

    existing_doc = library.get_document(req.doc_id)
    old_meta = (
        existing_doc.metadata
        if existing_doc and isinstance(existing_doc.metadata, dict)
        else {}
    )
    new_meta = dict(old_meta)
    new_meta['pending'] = False
    title_line = response.split('\n', 1)[0].lstrip('#').strip()
    if title_line:
        new_meta['theme_title'] = title_line[:120]

    library.update_document(
        req.doc_id,
        title=title_line[:120] or f'Theme {req.cluster_id[:12]}',
        content=response,
        metadata=new_meta,
    )
    await writer.update_document_embedding(req.doc_id)

    library.update_summary_hash(req.cluster_id, new_hash)
    _set_theme_coherent(library, req.cluster_id, coherent=True)
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def summarize_theme(
    library: "Library",
    writer: "LibraryWriter",
    cluster_id: str,
    *,
    model: str | None = None,
) -> bool:
    """Summarize one theme via the LLM.

    Returns True if the LLM produced a coherent doc (and it was persisted),
    False if it returned INCOHERENT or the theme was missing.

    On coherent: replaces the placeholder doc's content with the markdown,
    re-embeds, updates summary_hash, marks the theme clean (and coherent=1).

    On INCOHERENT: leaves the placeholder doc in place, marks coherent=0,
    dirty=0. The cluster row remains so graph traversal still sees it.
    """
    req = _build_theme_request(library, cluster_id)
    if req is None:
        return False

    response = await chat_complete(
        [
            {'role': 'system', 'content': req.system_prompt},
            {'role': 'user', 'content': req.user_prompt},
        ],
        model=model,
    )
    return await _apply_theme_response(library, writer, req, response)


def estimate_theme_summary_cost(
    prompt_texts, *, model, batched, out_tokens_per_theme: int = 350,
):
    """Estimated USD for summarizing the given ASSEMBLED prompts — the
    number disclosed before any spend (chars/4 heuristic + LLM_PRICING +
    the Batch API discount). Unknown models yield None, never a fake $0."""
    from docgen.pricing import _BATCH_DISCOUNT, LLM_PRICING

    rates = LLM_PRICING.get(model)
    if rates is None:
        return None
    rate_in, rate_out = rates
    tokens_in = sum(len(text) // 4 for text in prompt_texts)
    tokens_out = len(prompt_texts) * out_tokens_per_theme
    cost = (tokens_in * rate_in + tokens_out * rate_out) / 1e6
    return cost * _BATCH_DISCOUNT if batched else cost


async def generate_themes(
    library: "Library",
    writer: "LibraryWriter",
    *,
    model: str | None = None,
    concurrency: int | None = DEFAULT_SUMMARIZE_CONCURRENCY,
    max_calls: int | None = None,
    on_progress=None,
) -> dict:
    """Re-summarize every dirty theme.

    Args:
        on_progress: optional ``(completed, total, cluster_id|None) -> None``
            callback fired after each summarize attempt. ``cluster_id`` is
            the theme just processed (or ``None`` on the final completion
            tick). Used by the CLI to render a progress bar.

    Returns:
        {
            'summarized':  int,  # coherent docs written
            'incoherent':  int,  # marked incoherent (no doc written)
            'failed':      int,  # raised an exception
            'total_dirty': int,  # input population
        }
    """
    dirty = library.get_dirty_themes()
    total_dirty = len(dirty)

    if max_calls is not None and len(dirty) > max_calls:
        dirty = dirty[:max_calls]

    total_to_process = len(dirty)
    # ``concurrency=None`` is the intended "use the default parallelism" signal:
    # onboard forwards its own ``--concurrency`` (default None) straight through
    # cmd_themes_build. ``asyncio.Semaphore`` rejects None ("'<' not supported
    # between 'NoneType' and 'int'"), so fall back to the default fan-out here.
    sem = asyncio.Semaphore(DEFAULT_SUMMARIZE_CONCURRENCY
                            if concurrency is None else concurrency)
    summarized = 0
    incoherent = 0
    failed = 0
    completed = 0
    lock = asyncio.Lock()
    # A hard API cap (credit/quota or a maxed workspace usage limit) is fatal
    # for the whole phase — every further call fails identically. Stop on the
    # first one and surface ONE clear message instead of N per-cluster
    # tracebacks. ``aborted`` gates clusters still queued behind the semaphore.
    from docgen.llm.anthropic import QuotaExhaustedError
    aborted = asyncio.Event()
    quota_message: str | None = None

    async def process(cluster_id: str) -> None:
        nonlocal summarized, incoherent, failed, completed, quota_message
        if aborted.is_set():
            return
        async with sem:
            if aborted.is_set():
                return
            try:
                ok = await summarize_theme(library, writer, cluster_id, model=model)
                async with lock:
                    if ok:
                        summarized += 1
                    else:
                        incoherent += 1
            except QuotaExhaustedError as e:
                # Account cap — not a per-cluster content failure. Record the
                # message once, stop the phase; the caller surfaces it.
                async with lock:
                    if quota_message is None:
                        quota_message = str(e)
                        import logging as _logging
                        _logging.getLogger(__name__).error(
                            'Theme summarization stopped — Anthropic API usage '
                            'cap reached: %s', quota_message,
                        )
                aborted.set()
            except Exception as e:
                # Log with traceback so silent "Failed: N" stops being
                # diagnostic-free.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    'summarize_theme failed for cluster %s: %s',
                    cluster_id, e, exc_info=True,
                )
                async with lock:
                    failed += 1
        async with lock:
            completed += 1
            if on_progress is not None:
                on_progress(completed, total_to_process, cluster_id)

    if dirty:
        await asyncio.gather(*[process(cid) for cid in dirty])

    if on_progress is not None:
        on_progress(total_to_process, total_to_process, None)

    return {
        'summarized': summarized,
        'incoherent': incoherent,
        'failed': failed,
        'total_dirty': total_dirty,
        'quota_exhausted': quota_message is not None,
        'quota_message': quota_message,
    }


async def generate_themes_batched(
    library: "Library",
    writer: "LibraryWriter",
    strategy: object,
    *,
    model: str | None = None,
    max_calls: int | None = None,
    on_progress=None,
    on_stage=None,
) -> dict:
    """Batched twin of :func:`generate_themes`.

    Builds every dirty theme's prompt up front and submits them as a
    SINGLE provider batch — ≈50% cheaper than the live per-theme
    ``chat_complete`` path, and it doesn't run as a synchronous
    full-price spike after generation 'completes'. Each result is
    persisted with the same logic as the live path
    (:func:`_apply_theme_response`).

    ``strategy`` is a ``BatchStrategy`` (``AnthropicBatchStrategy`` /
    ``OpenAIBatchStrategy``, built by ``make_batch_strategy``) exposing
    ``submit_batch`` / ``poll_batch`` / ``fetch_batch_results``. Same
    return shape as :func:`generate_themes`.
    """
    from docgen.llm.batch import BatchRequest

    dirty = library.get_dirty_themes()
    total_dirty = len(dirty)
    if max_calls is not None and len(dirty) > max_calls:
        dirty = dirty[:max_calls]

    requests: list = []
    reqs_by_cid: dict[str, _ThemeRequest] = {}
    for cid in dirty:
        req = _build_theme_request(library, cid)
        if req is None:
            continue
        cid_str = str(len(requests))
        reqs_by_cid[cid_str] = req
        requests.append(
            BatchRequest(
                custom_id=cid_str,
                system_prompt=req.system_prompt,
                user_prompt=req.user_prompt,
            ),
        )

    summarized = incoherent = failed = 0
    if not requests:
        return {
            'summarized': 0, 'incoherent': 0, 'failed': 0,
            'total_dirty': total_dirty,
        }

    if on_stage is not None:
        on_stage('submit', 0, len(requests))
    submission = await strategy.submit_batch(requests)

    def _poll(processing: int, succeeded: int, errored: int) -> None:
        if on_stage is not None:
            on_stage('processing', succeeded + errored, len(requests))

    await strategy.poll_batch(submission.batch_id, on_progress=_poll)

    if on_stage is not None:
        on_stage('download', 0, len(requests))
    results = await strategy.fetch_batch_results(submission.batch_id)

    completed = 0
    for cid_str, req in reqs_by_cid.items():
        response = results.get(cid_str)
        if response is None:
            failed += 1
        elif await _apply_theme_response(library, writer, req, response):
            summarized += 1
        else:
            incoherent += 1
        completed += 1
        if on_progress is not None:
            on_progress(completed, len(reqs_by_cid), req.cluster_id)

    if on_progress is not None:
        on_progress(len(reqs_by_cid), len(reqs_by_cid), None)

    return {
        'summarized': summarized,
        'incoherent': incoherent,
        'failed': failed,
        'total_dirty': total_dirty,
    }


# ---------------------------------------------------------------------------
# Phase 5 — Incremental update wiring
# ---------------------------------------------------------------------------
#
# refresh_themes is the public, pull-based entry point. It discovers what's
# changed in the library on its own (joining cluster_history.created_at
# against documents.updated_at), so callers (orchestrator.run, cmd_sync,
# Phase-7 CLI) don't have to track changed_element_ids themselves and don't
# import update_semantic_edges_for / cluster_themes / generate_themes /
# local_reassign individually.
#
# Internal helpers below; use refresh_themes from outside this module.


def _base_never_clustered(library: "Library") -> bool:
    """True iff the base (global) themes pass has never produced a theme.

    Keys on themes in the base partition (association=''), NOT on
    cluster_history: a spool reconcile writes cluster_history for its own
    scoped runs, so a history-based check would mistake that for a completed
    base build and skip the first-build global semantic-edge rebuild — leaving
    base themes empty. A base cluster run ⟺ a base theme exists ⟺ (on a
    pre-`association` DB) the old cluster_history was non-empty, so this
    preserves the pre-hash-migration and incremental paths unchanged.
    """
    with library._conn_provider.acquire() as conn:
        row = conn.execute(
            "SELECT 1 FROM themes WHERE association = '' LIMIT 1"
        ).fetchone()
    return row is None


def _empty_summary(path: str) -> dict:
    return {
        'path': path,
        'changed': 0,
        'recluster_full': False,
        'summarized': 0,
        'incoherent': 0,
        'failed': 0,
        'total_dirty': 0,
    }


def local_reassign(
    library: "Library",
    changed_element_ids: set[str],
) -> set[str]:
    """Reassign each changed element to its highest-weight neighbor cluster.

    For every element in `changed_element_ids`:
      1. Read its doc_graph neighbors (imports/documents/topic_member/semantic_neighbor).
      2. For each neighbor, sum edge weights into that neighbor's cluster(s).
      3. Move the element to the cluster with the highest aggregate weight.

    Elements with no neighbors that map to an existing cluster are left alone.

    Returns the set of cluster_ids whose membership changed (these are
    auto-marked dirty so a subsequent generate_themes re-summarizes them).
    """
    affected_clusters: set[str] = set()

    for eid in changed_element_ids:
        with library._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT target_id, edge_type, weight FROM doc_graph WHERE source_id = ? '
                'UNION ALL '
                'SELECT source_id, edge_type, weight FROM doc_graph WHERE target_id = ?',
                (eid, eid),
            ).fetchall()

        cluster_scores: dict[str, float] = defaultdict(float)
        for nid, edge_type, weight in rows:
            if edge_type not in _REASSIGN_EDGE_TYPES:
                continue
            for n_cluster in library.get_themes_for_element(nid):
                cluster_scores[n_cluster] += float(weight)

        if not cluster_scores:
            continue

        best_cluster = max(cluster_scores, key=cluster_scores.get)
        old_clusters = set(library.get_themes_for_element(eid))

        # Already only in the best cluster — nothing to do.
        if old_clusters == {best_cluster}:
            continue

        with library._conn_provider.acquire() as conn:
            for old_c in old_clusters:
                conn.execute(
                    'DELETE FROM theme_members WHERE cluster_id = ? AND element_id = ?',
                    (old_c, eid),
                )
                affected_clusters.add(old_c)
            conn.execute(
                'INSERT OR REPLACE INTO theme_members '
                '(cluster_id, element_id, weight, joined_at) '
                'VALUES (?, ?, ?, ?)',
                (best_cluster, eid, 1.0, _now_iso()),
            )
        affected_clusters.add(best_cluster)

    for cid in affected_clusters:
        library.mark_theme_dirty(cid)

    return affected_clusters
def _changed_catalog_elements(library: "Library") -> set[str]:
    """Catalog element ids whose semantic edges themes must refresh: never
    recorded (new), or whose body hash (``metadata.sha_at_sync``) changed since
    the last sync.

    Hash-based, so cosmetic ``updated_at`` bumps from metadata-only catalog-sync
    refreshes don't count — only genuinely-changed elements get their edges
    rebuilt, leaving every other element's edges (and thus the deterministic
    Leiden partition) stable. Mirrors the generate-staleness model.
    """
    with library._conn_provider.acquire() as conn:
        rows = conn.execute(
            "SELECT d.id FROM documents d "
            "LEFT JOIN theme_synced_hashes t ON t.element_id = d.id "
            "WHERE d.content_type = 'catalog' "
            "AND (t.element_id IS NULL "
            "OR COALESCE(t.content_hash, '') != "
            "COALESCE(json_extract(d.metadata, '$.sha_at_sync'), ''))"
        ).fetchall()
    return {row[0] for row in rows}


def _theme_synced_hashes_empty(library: "Library") -> bool:
    """True when no element has ever been theme-synced — a fresh DB, or a
    pre-hash install awaiting a one-time baseline adoption."""
    with library._conn_provider.acquire() as conn:
        return conn.execute(
            "SELECT 1 FROM theme_synced_hashes LIMIT 1",
        ).fetchone() is None


def _record_theme_synced_hashes(
    library: "Library", element_ids: "set[str] | None" = None,
) -> None:
    """Stamp the current body hash of each (synced) catalog element so it is not
    re-detected next run. ``element_ids=None`` stamps every catalog element
    (first build / one-time baseline adoption)."""
    base = (
        "INSERT INTO theme_synced_hashes (element_id, content_hash) "
        "SELECT id, COALESCE(json_extract(metadata, '$.sha_at_sync'), '') "
        "FROM documents WHERE content_type = 'catalog'"
    )
    conflict = (
        " ON CONFLICT(element_id) DO UPDATE "
        "SET content_hash = excluded.content_hash"
    )
    ids = None
    if element_ids is not None:
        ids = list(element_ids)
        if not ids:
            return
    with library._conn_provider.acquire() as conn:
        if ids is None:
            conn.execute(base + conflict)
        else:
            # Chunk the id list so a large changed-element set never exceeds
            # SQLite's bind-variable limit.
            from library.sql_vars import chunk_ids
            for chunk in chunk_ids(ids):
                ph = ', '.join('?' * len(chunk))
                conn.execute(f"{base} AND id IN ({ph}){conflict}", chunk)
        conn.commit()


async def refresh_themes(
    library: "Library",
    writer: "LibraryWriter",
    *,
    enabled: bool = True,
    cluster_kwargs: dict | None = None,
    summarize_kwargs: dict | None = None,
    batch_strategy: "BatchStrategy | None" = None,
) -> dict:
    """Re-run the Leiden clustering EVERY time over the current semantic graph,
    then summarize only the themes whose membership changed.

    Edges are NOT rebuilt every run: the kNN index (HNSW) is approximate and
    non-deterministic, so a full rebuild would shift existing elements' edges,
    churn the partition, and spuriously re-summarize ~all themes. Instead edges
    are built once (first run) and refreshed incrementally only for elements
    whose body hash changed — every other element's edges stay put. So the
    seeded Leiden re-cluster over an unchanged graph reproduces the identical
    partition → nothing dirty → zero LLM cost; only genuinely-changed elements'
    themes are re-summarized.

    Callers gate this on the upstream generate step SUCCEEDING — a failed run
    must not recluster over a partial catalog. ``ariadne themes build`` calls it
    directly.
    """
    if not enabled:
        return _empty_summary("disabled")

    cluster_kwargs = cluster_kwargs or {}
    summarize_kwargs = summarize_kwargs or {}

    catalog_total = library.count_documents(content_type="catalog")
    if catalog_total == 0:
        return _empty_summary("no_catalog")

    from docgen.graph_builder import (
        build_semantic_edges,
        update_semantic_edges_for,
    )

    first_build = _base_never_clustered(library)
    if first_build:
        build_semantic_edges(library)
        _record_theme_synced_hashes(library)
        changed = catalog_total
    else:
        # Pre-hash DB (clusters but no recorded hashes): adopt the current
        # catalog as the edge baseline WITHOUT rebuilding every element's edges
        # (that would churn). Future content changes are picked up by hash.
        if _theme_synced_hashes_empty(library):
            _record_theme_synced_hashes(library)
        changed_ids = _changed_catalog_elements(library)
        if changed_ids:
            update_semantic_edges_for(library, list(changed_ids))
            _record_theme_synced_hashes(library, changed_ids)
        changed = len(changed_ids)

    cluster_themes(library, **cluster_kwargs)
    if batch_strategy is not None:
        # Batch has no per-call fan-out, so the live-only `concurrency`
        # knob is dropped; everything else forwards unchanged.
        _bk = {
            k: v for k, v in summarize_kwargs.items()
            if k != 'concurrency'
        }
        summary = await generate_themes_batched(
            library, writer, batch_strategy, **_bk,
        )
    else:
        summary = await generate_themes(library, writer, **summarize_kwargs)
    touched = summary["summarized"] + summary["incoherent"] + summary["failed"]
    summary["path"] = (
        "initial_build" if first_build else ("rebuilt" if touched else "noop")
    )
    summary["changed"] = changed
    summary["recluster_full"] = True
    return summary


__all__ = [
    'compute_summary_hash',
    'generate_themes',
    'local_reassign',
    'refresh_themes',
    'summarize_theme',
]
