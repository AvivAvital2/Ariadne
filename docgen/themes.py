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
is intentional — "Library-internal modules — legitimately unscoped".
"""
from __future__ import annotations

import asyncio
import hashlib
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

from docgen.cluster import cluster_themes
from docgen.graph_builder import update_semantic_edges_for
from docgen.prompts import THEME_SYSTEM_PROMPT, THEME_USER_TEMPLATE
from llm import chat_complete
from schema import _now_iso

if TYPE_CHECKING:
    from library import Library
    from writer import LibraryWriter


# Edge types whose weights count when scoring an element's neighbor clusters
# during local_reassign.
_REASSIGN_EDGE_TYPES = ('imports', 'documents', 'topic_member', 'semantic_neighbor')


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


async def generate_themes(
    library: "Library",
    writer: "LibraryWriter",
    *,
    model: str | None = None,
    concurrency: int = 4,
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
    sem = asyncio.Semaphore(concurrency)
    summarized = 0
    incoherent = 0
    failed = 0
    completed = 0
    lock = asyncio.Lock()

    async def process(cluster_id: str) -> None:
        nonlocal summarized, incoherent, failed, completed
        async with sem:
            try:
                ok = await summarize_theme(library, writer, cluster_id, model=model)
                async with lock:
                    if ok:
                        summarized += 1
                    else:
                        incoherent += 1
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
    }


async def generate_themes_batched(
    library: "Library",
    writer: "LibraryWriter",
    provider: object,
    *,
    model: str | None = None,
    max_calls: int | None = None,
    on_progress=None,
    on_stage=None,
) -> dict:
    """Batched twin of :func:`generate_themes`.

    Builds every dirty theme's prompt up front and submits them as a
    SINGLE Anthropic Message Batch — ≈50% cheaper than the live
    per-theme ``chat_complete`` path, and it doesn't run as a
    synchronous full-price spike after generation 'completes'. Each
    result is persisted with the same logic as the live path
    (:func:`_apply_theme_response`).

    ``provider`` must expose ``submit_batch`` / ``poll_batch`` /
    ``fetch_batch_results`` (AnthropicProvider's batch surface). Same
    return shape as :func:`generate_themes`.
    """
    from docgen.llm.anthropic import BatchRequest

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
    submission = await provider.submit_batch(requests)

    def _poll(processing: int, succeeded: int, errored: int) -> None:
        if on_stage is not None:
            on_stage('processing', succeeded + errored, len(requests))

    await provider.poll_batch(submission.batch_id, on_progress=_poll)

    if on_stage is not None:
        on_stage('download', 0, len(requests))
    results = await provider.fetch_batch_results(submission.batch_id)

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


def _latest_cluster_run_time(library: "Library") -> str | None:
    """Return the ISO timestamp of the most recent cluster_history row, or None."""
    with library._conn_provider.acquire() as conn:
        row = conn.execute(
            'SELECT MAX(created_at) FROM cluster_history'
        ).fetchone()
    return row[0] if row and row[0] is not None else None


def _catalog_elements_changed_since(
    library: "Library", since_iso: str,
) -> set[str]:
    """Return ids of catalog-content documents whose updated_at > since_iso."""
    with library._conn_provider.acquire() as conn:
        rows = conn.execute(
            "SELECT id FROM documents "
            "WHERE content_type = 'catalog' AND updated_at > ?",
            (since_iso,),
        ).fetchall()
    return {row[0] for row in rows}


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


async def refresh_themes(
    library: "Library",
    writer: "LibraryWriter",
    *,
    enabled: bool = True,
    recluster_threshold: float = 0.05,
    cluster_kwargs: dict | None = None,
    summarize_kwargs: dict | None = None,
) -> dict:
    """Bring themes up to date with current library state.

    Pull-based: the function discovers what's changed by joining
    cluster_history.created_at against documents.updated_at, so callers don't
    track changed_element_ids themselves. Updates semantic edges only for
    changed elements, decides between cheap (local_reassign) and full
    (cluster_themes) paths via the drift gate, and re-summarizes any dirty
    themes via generate_themes.

    Args:
        library: Library to refresh.
        writer: LibraryWriter used by generate_themes for embedding writes.
        enabled: Master switch (config.themes_enabled). When False, returns
            immediately with path='disabled'.
        recluster_threshold: drift_ratio (|changed| / |catalog|) at-or-above
            which a full recluster runs instead of local reassignment.
        cluster_kwargs: Forwarded to cluster_themes() when invoked.
        summarize_kwargs: Forwarded to generate_themes() when invoked.

    Returns:
        {
            'path':           'disabled' | 'no_catalog' | 'initial_build' |
                              'noop' | 'local_reassign' | 'full_recluster' |
                              'summarize_only',
            'changed':        int,
            'recluster_full': bool,
            'summarized':     int,
            'incoherent':     int,
            'failed':         int,
            'total_dirty':    int,
        }
    """
    if not enabled:
        return _empty_summary('disabled')

    cluster_kwargs = cluster_kwargs or {}
    summarize_kwargs = summarize_kwargs or {}

    catalog_total = library.count_documents(content_type='catalog')
    if catalog_total == 0:
        return _empty_summary('no_catalog')

    last_run_time = _latest_cluster_run_time(library)

    if last_run_time is None:
        # No prior clustering — full initial build. We must build semantic
        # edges BEFORE Leiden, otherwise load_hybrid_graph filters the
        # structural-only edges out (imports/documents endpoints aren't
        # catalog doc UUIDs), the resulting igraph has zero edges, and
        # leidenalg.find_partition crashes with KeyError on the missing
        # 'weight' edge attribute. The incremental path further down already
        # calls update_semantic_edges_for; this mirrors it for first runs.
        from docgen.graph_builder import build_semantic_edges
        build_semantic_edges(library)
        cluster_themes(library, **cluster_kwargs)
        summary = await generate_themes(library, writer, **summarize_kwargs)
        summary['path'] = 'initial_build'
        summary['changed'] = catalog_total
        summary['recluster_full'] = True
        return summary

    changed_ids = _catalog_elements_changed_since(library, last_run_time)

    if not changed_ids:
        # Catalog hasn't moved; only summarize if some themes are still dirty
        # (e.g., a prior summarize call failed for them).
        if not library.get_dirty_themes():
            return _empty_summary('noop')
        summary = await generate_themes(library, writer, **summarize_kwargs)
        summary['path'] = 'summarize_only'
        summary['changed'] = 0
        summary['recluster_full'] = False
        return summary

    update_semantic_edges_for(library, list(changed_ids))

    drift_ratio = len(changed_ids) / catalog_total
    if drift_ratio >= recluster_threshold:
        cluster_themes(library, **cluster_kwargs)
        path = 'full_recluster'
        recluster_full = True
    else:
        local_reassign(library, set(changed_ids))
        path = 'local_reassign'
        recluster_full = False

    summary = await generate_themes(library, writer, **summarize_kwargs)
    summary['path'] = path
    summary['changed'] = len(changed_ids)
    summary['recluster_full'] = recluster_full
    return summary


__all__ = [
    'compute_summary_hash',
    'generate_themes',
    'local_reassign',
    'refresh_themes',
    'summarize_theme',
]
