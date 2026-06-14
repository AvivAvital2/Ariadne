"""LLM description pass for catalog elements."""                                                                                                                        
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from llm import chat_complete

if TYPE_CHECKING:
    from library import Library
    from writer import LibraryWriter


_logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 4


class _EmptyDescriptionError(Exception):
    """Sentinel for the empty-response classification.

    Distinct from generic Exception so the failure summary can tell
    the user "the provider returned no content" (which is recoverable
    by re-running) apart from a transient network error or an API
    rejection.
    """

_PROMPT = (
    'You are a code documenter. Write a concise 1-2 sentence description of '                                                                                                        
    'what this code element does. Be specific. No filler, no preamble, no code blocks. '
    'Respond with ONLY the description.\n\n'                                                                                                                                       
    'Kind: {subtype}\n'
    'Name: {qualified_name}\n'                                                                                                                                                      
    'Parent: {parent}\n'                                                                                                                                                            
    'Signature: {signature}\n'
    'Location: {file}:{line_start}-{line_end}\n\n'                                                                                                                                 
    'Description:'                                                                                                                                                                   
)
def build_describe_prompt(metadata: dict) -> str:
    """The exact user prompt ``describe_element`` sends for one element.

    Pure (no LLM, no I/O) so the cost estimator can tokenize the real
    per-element input instead of guessing a flat per-call figure.
    """
    location = metadata.get('location') or {}
    return _PROMPT.format(
        subtype=metadata.get('subtype', ''),
        qualified_name=metadata.get('qualified_name', ''),
        parent=metadata.get('parent_qualified_name') or '(module-level)',
        signature=metadata.get('signature', ''),
        file=metadata.get('file', ''),
        line_start=location.get('line_start', 0),
        line_end=location.get('line_end', 0),
    )


async def describe_element(metadata: dict, *, model: str | None = None) -> str:
    """Generate a 1-2 sentence description for a single catalog element."""
    response = await chat_complete(
        [{'role': 'user', 'content': build_describe_prompt(metadata)}],
        model=model,
    )
    return response.strip()


async def describe_source_elements(
    library: "Library",
    writer: "LibraryWriter",
    source_name: str,                                                                                                                                                                
    *,
    force: bool = False,                                                                                                                                                             
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str | None = None,
    max_calls: int | None = None,
    on_progress: Callable[..., None] | None = None,
) -> dict:
    """Generate descriptions for catalog element docs of a source.                                                                                                                   
                                                                                                                                                                                     
    Returns a summary dict:
    {described: int, already_had_description: int, failed: int, total_candidates: int}                                                                                               
                                                                                                                                                                                     
    Idempotent by default (skip elements that already have a description).
    Pass force=True to regenerate all.                                                                                                                                               
    """                                                                                                                                                                              
    all_catalog = library.list_documents(content_type='catalog', limit=100_000)
    candidates = [                                                                                                                                                                   
        d for d in all_catalog
        if d.metadata.get('source_name') == source_name                                                                                                                              
        and d.metadata.get('kind') == 'element'
    ]                                                                                                                                                                                
    if force:   
        to_describe = list(candidates)
    else:                                                                                                                                                                            
        to_describe = [d for d in candidates if not d.metadata.get('description')]
                                                                                                                                                                                     
    already_had = len(candidates) - len(to_describe)

    if max_calls is not None and max_calls < len(to_describe):
        to_describe = to_describe[:max_calls]                                                                                                                                        
 
    if not to_describe:                                                                                                                                                              
        return {
            'described': 0,
            'already_had_description': already_had,
            'failed': 0,                                                                                                                                                             
            'total_candidates': len(candidates),
        }                                                                                                                                                                            
                
    sem = asyncio.Semaphore(concurrency)
    described = 0
    failed = 0
    # Reasons keyed by category so the summary tells the user what
    # actually went wrong (vs. opaque "1 failed"): empty_response, network,
    # http_error, other. Stored per-doc for diagnostics.
    failures: list[tuple[str, str, str]] = []
    first_failure_detail: list[str] = []
    lock = asyncio.Lock()
    total = len(to_describe)

    async def process(doc):
        nonlocal described, failed
        async with sem:
            failure_category = None
            failure_detail = ''
            try:
                from docgen.calibration import usage_context
                with usage_context(
                    phase='describe', doc_type='element',
                    language=doc.metadata.get('language'),
                ):
                    desc = await describe_element(doc.metadata, model=model)
                if not desc:
                    raise _EmptyDescriptionError('empty description')
                new_metadata = dict(doc.metadata)
                new_metadata['description'] = desc
                new_content = (doc.content or '').rstrip() + '\n\nDescription: ' + desc
                library.update_document(doc.id, content=new_content, metadata=new_metadata)
                await writer.update_document_embedding(doc.id)
                async with lock:
                    described += 1
            except _EmptyDescriptionError as e:
                failure_category = 'empty_response'
                failure_detail = str(e)
            except Exception as e:
                # Classify by exception type so the summary tells a
                # story (transient network blip vs. content refusal
                # vs. provider bug).
                name = type(e).__name__
                if name in ('TimeoutException', 'ConnectError',
                            'ReadError', 'RequestError',
                            'NetworkError'):
                    failure_category = 'network'
                elif name == 'HTTPStatusError':
                    failure_category = 'http_error'
                elif name == 'QuotaExhaustedError':
                    # Quota exhaustion is fatal — surface up.
                    raise
                else:
                    failure_category = 'other'
                failure_detail = f'{name}: {e}'
                # For HTTP errors, append the response body so the
                # user sees Anthropic's specific complaint (e.g.,
                # "system: Field required" / "max_tokens: too large").
                # Without this, "Client error '400 Bad Request'" is
                # opaque.
                if name == 'HTTPStatusError':
                    try:
                        body = e.response.text  # type: ignore[attr-defined]
                        if body:
                            failure_detail += (
                                f' | body: {body[:300]}'
                            )
                    except Exception:
                        pass
            if failure_category is not None:
                async with lock:
                    failed += 1
                    failures.append(
                        (doc.id, failure_category, failure_detail),
                    )
                    if not first_failure_detail:
                        # Truncate so a multi-line traceback doesn't
                        # swallow the bar.
                        snippet = failure_detail.replace('\n', ' ')[:120]
                        first_failure_detail.append(
                            f'{failure_category}: {snippet}',
                        )
            if on_progress is not None:
                try:
                    on_progress(
                        described, failed, total,
                        first_failure=(
                            first_failure_detail[0]
                            if first_failure_detail else None
                        ),
                    )
                except TypeError:
                    # Older callback signature without first_failure;
                    # fall back to the three-arg form.
                    try:
                        on_progress(described, failed, total)
                    except Exception:
                        pass
                except Exception:
                    pass

    await asyncio.gather(*[process(d) for d in to_describe], return_exceptions=False)

    return {
        'described': described,
        'already_had_description': already_had,
        'failed': failed,
        'failure_reasons': failures,
        'total_candidates': len(candidates),
    }
                                                                                                                                                                                     
 
def _catalog_describe_config_hash(source_name: str, model: str) -> str:
    """Hash key for ``pending_batches`` rows owned by this command.

    Distinct from ``generate``'s config_hash so the two batch flows
    don't collide on the same row. The format is human-readable; if
    you need to debug a stuck batch, the prefix tells you which
    command owns it.
    """
    return f'catalog-describe::{source_name}::{model}'


def _make_poll_callback(on_progress, on_stage):
    """Bridge the batch poll's ``(processing, succeeded, errored)``
    callback to both the legacy ``on_progress`` and the staged
    ``on_stage('processing', done, total)`` consumers."""
    if on_progress is None and on_stage is None:
        return None

    def _cb(processing: int, succeeded: int, errored: int) -> None:
        if on_progress is not None:
            # A detailed handler owns the processing-line text; don't let
            # on_stage overwrite it on every tick.
            on_progress(processing, succeeded, errored)
        elif on_stage is not None:
            total = processing + succeeded + errored
            on_stage('processing', succeeded + errored, total or None)

    return _cb


async def _poll_or_cancel(provider, batch_id: str, poll_cb) -> None:
    """Poll the batch; on abort (Ctrl-C) or task cancellation, CANCEL the
    in-flight batch at the provider so it stops processing — and being
    billed — then re-raise. Cancellation is best-effort and must never
    mask the original abort.
    """
    try:
        await provider.poll_batch(batch_id, on_progress=poll_cb)
    except (KeyboardInterrupt, asyncio.CancelledError):
        cancel = getattr(provider, 'cancel_batch', None)
        if cancel is not None:
            try:
                await cancel(batch_id)
            except Exception:
                pass
        raise


async def describe_source_elements_batched(
    library: "Library",
    writer: "LibraryWriter",
    source_name: str,
    *,
    strategy,
    model: str | None = None,
    force: bool = False,
    resume: bool = False,
    max_calls: int | None = None,
    staleness_db_path: "Path | str | None" = None,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_stage: "Callable[[str, int, int | None], None] | None" = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict:
    """Batch-mode variant of ``describe_source_elements``.

    Submits all element prompts as a single Anthropic Message Batch
    (one request per element), polls until the batch ends, then
    applies the returned descriptions to the catalog docs.

    **Auto-resume**: regardless of the ``resume`` flag, the function
    first checks ``pending_batches`` for an in-flight batch matching
    the current source+model. If one exists, it adopts that batch
    rather than submitting a new one — so a ctrl-C'd run is recovered
    transparently on the next invocation. ``resume=True`` is the
    explicit recovery contract: error out (with a structured note) if
    no pending batch is found, instead of submitting fresh.

    Args:
        provider: a ``BatchStrategy`` (``AnthropicBatchStrategy`` /
            ``OpenAIBatchStrategy``, built by ``make_batch_strategy``) exposing
            ``submit_batch`` / ``poll_batch`` / ``fetch_batch_results``; tests
            inject a fake.
        resume: when True, require a pending batch (return a "no
            pending" note instead of submitting fresh). The default
            (False) auto-adopts a pending batch when present and
            submits fresh otherwise.
        staleness_db_path: where to find the StalenessTracker DB.
            None falls back to the global config's
            ``staleness_db_path``.
    """
    from docgen.llm.anthropic import BatchRequest
    from docgen.staleness import StalenessTracker

    all_catalog = library.list_documents(content_type='catalog', limit=100_000)
    candidates = [
        d for d in all_catalog
        if d.metadata.get('source_name') == source_name
        and d.metadata.get('kind') == 'element'
    ]
    to_describe = (
        list(candidates) if force
        else [d for d in candidates if not d.metadata.get('description')]
    )
    if max_calls is not None and max_calls < len(to_describe):
        to_describe = to_describe[:max_calls]

    config_hash = _catalog_describe_config_hash(
        source_name, model or '',
    )

    # Resolve the staleness DB path. Provide a fallback so the
    # function is usable without passing it explicitly.
    if staleness_db_path is None:
        from config import get_config
        cfg = get_config()
        staleness_db_path = cfg.staleness_db_path

    # ---- Auto-resume probe --------------------------------------------
    # Always check for a pending batch matching this source+model,
    # regardless of the ``resume`` flag. If found, adopt it — a prior
    # ctrl-C'd run shouldn't pay for a second batch submission. When
    # ``resume=True`` and there's no pending row, surface the structured
    # "no pending" note (the explicit-recovery contract); the default
    # (resume=False) falls through to the fresh-submit path.
    tracker = StalenessTracker(staleness_db_path)
    try:
        pending = tracker.find_pending_batch(config_hash)
    finally:
        tracker.close()

    if pending is None and resume:
        return {
            'submitted': 0,
            'batch_id': None,
            'described': 0,
            'failed': 0,
            'total_candidates': len(candidates),
            'resumed': False,
            'note': 'no pending batch found for this source+model',
        }

    if pending is not None:
        # Pick up the in-flight batch without re-submitting.
        batch_id = pending.batch_id
        try:
            poll_cb = _make_poll_callback(on_progress, on_stage)
            if on_stage is not None:
                on_stage('processing', 0, None)
            await _poll_or_cancel(strategy, batch_id, poll_cb)
            if on_stage is not None:
                on_stage('download', 0, None)
            from docgen.calibration import usage_context as _uctx
            with _uctx(phase='describe', doc_type='element', language=None):
                results = await strategy.fetch_batch_results(batch_id)
            if on_stage is not None:
                on_stage('download', len(results), len(results))
        except Exception as e:
            # Permanent failure (expired batch, 404, server-side cancel)
            # would otherwise leave the pending row stuck forever. Clear it
            # so subsequent runs can re-submit; surface a structured result.
            tracker = StalenessTracker(staleness_db_path)
            try:
                tracker.clear_pending_batch(batch_id)
            finally:
                tracker.close()
            return {
                'submitted': 0,
                'batch_id': batch_id,
                'described': 0,
                'failed': 0,
                'total_candidates': len(candidates),
                'resumed': False,
                'note': f'pending batch could not be fetched ({type(e).__name__}: {e}); pending row cleared so the next run can re-submit',
            }
        return await _apply_batch_results_to_docs(
            library, writer, to_describe, candidates,
            results, batch_id, submitted=0, resumed=True,
            staleness_db_path=staleness_db_path,
            config_hash=config_hash,
            on_stage=on_stage, concurrency=concurrency,
        )

    # ---- Submit path ----------------------------------------------------
    if not to_describe:
        return {
            'submitted': 0,
            'batch_id': None,
            'total_candidates': len(candidates),
            'described': 0,
            'failed': 0,
        }

    # Build one BatchRequest per element. ``custom_id`` is the element's
    # doc_id so we can re-attach the description when results come back.
    requests = []
    for doc in to_describe:
        location = doc.metadata.get('location') or {}
        prompt = _PROMPT.format(
            subtype=doc.metadata.get('subtype', ''),
            qualified_name=doc.metadata.get('qualified_name', ''),
            parent=(
                doc.metadata.get('parent_qualified_name')
                or '(module-level)'
            ),
            signature=doc.metadata.get('signature', ''),
            file=doc.metadata.get('file', ''),
            line_start=location.get('line_start', 0),
            line_end=location.get('line_end', 0),
        )
        requests.append(BatchRequest(
            custom_id=doc.id,
            system_prompt='',
            user_prompt=prompt,
        ))

    if on_stage is not None:
        on_stage('submit', 0, len(requests))
    submission = await strategy.submit_batch(requests)
    if on_stage is not None:
        on_stage('submit', len(requests), len(requests))

    # Persist the pending batch BEFORE we poll, so a crash mid-poll
    # leaves a recoverable row for ``--resume`` to find.
    tracker = StalenessTracker(staleness_db_path)
    try:
        tracker.record_pending_batch(
            batch_id=submission.batch_id,
            prompts_json='[]',  # not needed for catalog-describe resume
            file_to_idxs_json='{}',
            config_hash=config_hash,
        )
    finally:
        tracker.close()

    poll_cb = _make_poll_callback(on_progress, on_stage)
    if on_stage is not None:
        on_stage('processing', 0, None)
    await _poll_or_cancel(strategy, submission.batch_id, poll_cb)

    if on_stage is not None:
        on_stage('download', 0, None)
    from docgen.calibration import usage_context as _uctx
    with _uctx(phase='describe', doc_type='element', language=None):
        results = await strategy.fetch_batch_results(submission.batch_id)
    if on_stage is not None:
        on_stage('download', len(results), len(results))

    return await _apply_batch_results_to_docs(
        library, writer, to_describe, candidates,
        results, submission.batch_id, submitted=len(requests),
        resumed=False, staleness_db_path=staleness_db_path,
        config_hash=config_hash,
        on_stage=on_stage, concurrency=concurrency,
    )


async def _apply_batch_results_to_docs(
    library, writer, to_describe, candidates,
    results, batch_id, *, submitted, resumed,
    staleness_db_path, config_hash,
    on_stage=None, concurrency: int = DEFAULT_CONCURRENCY,
):
    """Apply batch results to docs and clear the pending row.

    Returns a summary dict. On success, the pending_batches row is
    deleted so subsequent --resume invocations don't re-fetch.

    ``on_stage(stage, completed, total)`` (optional) reports apply/embed
    progress so the caller can show it — this stage re-embeds every
    described doc and is otherwise invisible.
    """
    from docgen.staleness import StalenessTracker

    by_id = {d.id: d for d in to_describe}
    # Rows we'll actually write (non-empty result that maps to a doc).
    applicable = [
        (cid, text.strip())
        for cid, text in results.items()
        if text and text.strip() and by_id.get(cid) is not None
    ]
    errored = sum(
        1 for _cid, text in results.items() if not text or not text.strip()
    )
    # Submitted elements with NO result row at all. Anthropic returns a row
    # per custom_id, but OpenAI's split output/error files can come back short
    # on a wholesale-batch failure ('failed'/'expired'). Count them as failures
    # so a partial provider outcome surfaces instead of being silently
    # under-applied.
    missing = [cid for cid in by_id if cid not in results]
    if missing:
        _logger.warning(
            'Batch %s: %d of %d submitted element(s) returned no result row; '
            'counting as failed (provider returned a short result set)',
            batch_id, len(missing), len(by_id),
        )
    failed = errored + len(missing)
    skipped_already_applied = sum(
        1 for cid, text in results.items()
        if text and text.strip() and by_id.get(cid) is None
    )

    total = len(applicable)
    if on_stage is not None:
        on_stage('apply', 0, total)

    # Apply concurrently: the per-doc re-embed (``update_document_embedding``)
    # is the slow, awaitable part, so a bounded gather overlaps those
    # calls instead of crawling one-at-a-time over tens of thousands of
    # docs. The synchronous ``library.update_document`` runs to
    # completion before each await, so DB writes serialize naturally on
    # the single connection — no concurrent-write hazard.
    import asyncio

    sem = asyncio.Semaphore(max(1, concurrency))
    described = 0

    async def _apply_one(cid: str, desc: str) -> None:
        nonlocal described
        async with sem:
            doc = by_id[cid]
            new_metadata = dict(doc.metadata)
            new_metadata['description'] = desc
            new_content = (
                (doc.content or '').rstrip() + '\n\nDescription: ' + desc
            )
            library.update_document(
                doc.id, content=new_content, metadata=new_metadata,
            )
            await writer.update_document_embedding(doc.id)
            described += 1
            if on_stage is not None:
                on_stage('apply', described, total)

    await asyncio.gather(*[_apply_one(cid, desc) for cid, desc in applicable])

    # Clear the pending row so a follow-up --resume doesn't try to
    # re-apply the same results.
    tracker = StalenessTracker(staleness_db_path)
    try:
        tracker.clear_pending_batch(batch_id)
    finally:
        tracker.close()

    return {
        'submitted': submitted,
        'batch_id': batch_id,
        'described': described,
        'failed': failed,
        'skipped_already_applied': skipped_already_applied,
        'total_candidates': len(candidates),
        'resumed': resumed,
    }


__all__ = [
    'DEFAULT_CONCURRENCY',
    'describe_element',
    'describe_source_elements',
    'describe_source_elements_batched',
]                                                                                                    
