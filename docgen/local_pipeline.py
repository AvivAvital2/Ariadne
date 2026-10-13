"""Fixed local phases and the four persisted application handlers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docgen.orchestrator import DocGenOrchestrator, PipelineResult
    from docgen.scip_config import SourceScipConfig
    from library import Library

import json
from pathlib import Path

from docgen.generator import GeneratedDoc
from docgen.llm.batch import BatchRequest
from docgen.spool import JobKind, LocalExecutor, SpoolJobSpec, SpoolStore, file_fingerprint
from library.prepared import ObsoleteInput
from model_config import identity_hash
from writer import LibraryWriter


def catalog_fingerprint(library: Library) -> str:
    return identity_hash(
        sorted((doc.id, doc.updated_at) for doc in library.list_documents('catalog'))
    )


def _scope(orchestrator: DocGenOrchestrator) -> dict:
    c = orchestrator.config
    return {
        'source_name': c.source_name,
        'source_path': str(c.source_path),
        'doc_types': list(c.doc_types),
        'dependencies': list(c.dependencies),
        'inject_crossrefs': c.inject_crossrefs,
        'themes_enabled': c.themes_enabled,
        'generation_scope': orchestrator._staleness.generation_scope,
        'pipeline': True,
        'files_skipped': 0,
    }


async def plan_documents(
    orchestrator: DocGenOrchestrator,
    files: list[Path],
    *,
    phase: str = 'base',
    consumer_contexts: dict[Path, str] | None = None,
    extra_snapshots: dict[str, str] | None = None,
) -> list[SpoolJobSpec]:
    specs = []
    consumer_contexts = consumer_contexts or {}
    context_versions = {
        doc.id: doc.updated_at for doc in orchestrator._library.list_documents('catalog')
    }
    structural = structural_fingerprint(orchestrator._library)
    artifacts = artifact_snapshots(
        orchestrator.config.source_path, orchestrator.config.source_config
    )
    for path in files:
        before = file_fingerprint(path)
        prompts = await orchestrator._build_prompts_for_file(path)
        if before != file_fingerprint(path):
            raise ObsoleteInput(f'Source changed during prompt construction: {path}')
        if not prompts:
            from docgen.staleness import StalenessTracker

            if not orchestrator._staleness._filter_requested_by_language(
                path, set(orchestrator.config.doc_types) - {'catalog'}
            ):
                continue
            raise ValueError(f'No supported document prompts could be built for {path}')
        for prompt in prompts:
            captured = getattr(prompt, 'source_snapshot', None) or before
            if captured != before:
                raise ObsoleteInput(f'Source changed during prompt construction: {path}')
            metadata = {
                **prompt.metadata,
                'generated': True,
                'processed_hash': captured,
                'generation_scope': orchestrator._staleness.generation_scope,
            }
            if orchestrator.config.generation_scope.startswith('branch:'):
                metadata.update(
                    status='experimental',
                    branches=[orchestrator.config.generation_scope.removeprefix('branch:')],
                )
            doc = GeneratedDoc(prompt.title, '', prompt.doc_type, (str(path),), metadata)
            target = orchestrator._compute_deterministic_doc_id(
                doc, orchestrator.config.source_name
            )
            old = orchestrator._library.get_document(target)
            custom_id = f'{phase}:{target}'
            user = prompt.user_prompt
            if path in consumer_contexts:
                user += '\n\n' + consumer_contexts[path]
            specs.append(
                SpoolJobSpec(
                    custom_id,
                    JobKind.DOCUMENT,
                    target,
                    {
                        'content_type': prompt.doc_type,
                        'title': prompt.title,
                        'source_files': [str(path)],
                        'source_name': orchestrator.config.source_name,
                        'metadata': metadata,
                        'relative_path': str(path.relative_to(orchestrator.config.source_path)),
                        'structural_fingerprint': structural,
                    },
                    request=BatchRequest(custom_id, prompt.system_prompt, user, max_tokens=8192),
                    snapshots={str(path): captured, **artifacts, **(extra_snapshots or {})},
                    context_versions=context_versions,
                    expected_version=old.updated_at if old else None,
                    phase=phase,
                )
            )
    return specs


def _reverse_inputs(orchestrator: DocGenOrchestrator) -> tuple[dict[Path, str], dict[str, str]]:
    from config import get_config
    from docgen.reverse_augment import reverse_augment_plan
    from docgen.scip_cross_source import CrossSourceGraph, load_source_from_manifest

    cfg = get_config()
    source = orchestrator.config.source_name
    graph = CrossSourceGraph()
    paths = {name: cfg.get_source_path(name) for name in cfg.sources}
    paths[source] = orchestrator.config.source_path
    snapshots = {}
    for name, root in paths.items():
        if root is None:
            continue
        manifest = root / '.ariadne' / 'manifest.json'
        if not manifest.exists():
            if name == source:
                return {}, {}
            continue
        # Include manifests and index bytes consumed by the consumer graph.
        snapshots[str(manifest)] = file_fingerprint(manifest)
        snapshots.update(artifact_snapshots(root, cfg.get_source_scip_config(name)))
        load_source_from_manifest(graph, name, root)
    graph.materialize()
    contexts = {
        orchestrator.config.source_path / relative: context
        for _, relative, context in reverse_augment_plan(graph, [source])
    }
    return contexts, snapshots


def plan_themes(library: Library) -> list[SpoolJobSpec]:
    from docgen.themes import _build_theme_request, compute_summary_hash

    specs = []
    for cluster_id in library.get_dirty_themes():
        req = _build_theme_request(library, cluster_id)
        if req is None:
            continue
        doc = library.get_document(req.doc_id)
        custom_id = 'theme:' + req.cluster_id
        specs.append(
            SpoolJobSpec(
                custom_id,
                JobKind.THEME,
                req.doc_id,
                {
                    'content_type': 'theme',
                    'title': doc.title,
                    'metadata': doc.metadata,
                    'source_files': doc.source_files,
                    'source_name': doc.source_name,
                    'cluster_id': req.cluster_id,
                    'resolution': req.resolution,
                    'summary_hash': compute_summary_hash(
                        req.member_ids, req.summaries, req.resolution
                    ),
                },
                request=BatchRequest(
                    custom_id, req.system_prompt, req.user_prompt, max_tokens=2048
                ),
                context_versions={
                    member: library.get_document(member).updated_at for member in req.member_ids
                },
                expected_version=doc.updated_at,
                phase='themes',
            )
        )
    return specs


def reconcile_staleness(store: SpoolStore, run_id: str, orchestrator: DocGenOrchestrator) -> None:
    from docgen.staleness import StalenessTracker

    groups = {}
    for job in store.jobs(run_id):
        payload = json.loads(job['payload_json'])
        if payload['kind'] != JobKind.DOCUMENT:
            continue
        data = payload['data']
        relative = data['relative_path']
        # Later reverse-augmentation jobs replace their base-document result.
        group = groups.setdefault(relative, {})
        group[data['content_type']] = (job, data)
    for relative, by_type in groups.items():
        path = orchestrator.config.source_path / relative
        required = orchestrator._staleness._filter_requested_by_language(
            path, set(orchestrator.config.doc_types) - {'catalog'}
        )
        if not required.issubset(by_type) or any(
            by_type[t][0]['state'] != 'applied' for t in required
        ):
            continue
        hashes = {by_type[t][1]['metadata']['processed_hash'] for t in required}
        if len(hashes) != 1:
            continue
        processed_hash = hashes.pop()
        ids = [by_type[t][0]['target_id'] for t in sorted(required)]
        docs = [orchestrator._library.get_document(doc_id) for doc_id in ids]
        if any(
            doc is None
            or doc.source_name != orchestrator.config.source_name
            or doc.metadata.get('processed_hash') != processed_hash
            for doc in docs
        ):
            continue
        orchestrator._staleness.record_documentation(
            path, ids, orchestrator.config.source_path, processed_hash=processed_hash
        )


def _done(store: SpoolStore, run_id: str, phase: str) -> bool:
    return (
        json.loads(store.get_run(run_id)['checkpoints_json']).get(phase, {}).get('state')
        == 'complete'
    )


def _incomplete(counts: dict) -> bool:
    return bool(
        counts['failed']
        or counts['obsolete']
        or counts['cancelled']
        or counts['state'] in {'failed', 'obsolete', 'cancelled'}
    )


async def complete_generation(
    store: SpoolStore, run_id: str, orchestrator: DocGenOrchestrator
) -> dict:
    checkpoints = json.loads(store.get_run(run_id)['checkpoints_json'])
    if checkpoints.get('graph', {}).get('state') == 'complete' and checkpoints['graph'][
        'fingerprint'
    ] != catalog_fingerprint(store.library):
        raise ObsoleteInput('Catalog changed after semantic phases were planned')
    executor = LocalExecutor(store)
    counts = await executor.drive(run_id, finalize=False)
    if _incomplete(counts):
        reconcile_staleness(store, run_id, orchestrator)
        return store.finish(run_id)
    if not _done(store, run_id, 'base'):
        store.checkpoint(
            run_id, 'base', identity_hash([j['request_hash'] for j in store.jobs(run_id)])
        )
    if not _done(store, run_id, 'reverse'):
        checkpoints = json.loads(store.get_run(run_id)['checkpoints_json'])
        if 'reverse' not in checkpoints:
            contexts, snapshots = _reverse_inputs(orchestrator)
            specs = await plan_documents(
                orchestrator,
                sorted(contexts),
                phase='reverse',
                consumer_contexts=contexts,
                extra_snapshots=snapshots,
            )
            store.add_phase(run_id, 'reverse', specs, identity_hash(snapshots))
        counts = await executor.drive(run_id, finalize=False)
        if _incomplete(counts):
            reconcile_staleness(store, run_id, orchestrator)
            return store.finish(run_id)
        store.checkpoint(
            run_id,
            'reverse',
            identity_hash(
                [
                    j['request_hash']
                    for j in store.jobs(run_id)
                    if json.loads(j['payload_json'])['phase'] == 'reverse'
                ]
            ),
        )
    fingerprint = catalog_fingerprint(store.library)
    if not _done(store, run_id, 'graph'):
        if not store.begin_phase(run_id, 'graph', fingerprint):
            return store.finish(run_id)
        from docgen.themes import refresh_themes

        await refresh_themes(
            store.library,
            executor.writer,
            enabled=orchestrator.config.themes_enabled,
            summarize_kwargs={'max_calls': 0, 'concurrency': 1},
        )
        store.checkpoint(run_id, 'graph', fingerprint)
        with store.library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE library_state SET graph_fingerprint=? WHERE singleton_id=1', (fingerprint,)
            )
    elif (
        json.loads(store.get_run(run_id)['checkpoints_json'])['graph']['fingerprint'] != fingerprint
    ):
        raise ObsoleteInput('Catalog changed after this run planned its semantic phases')
    if not _done(store, run_id, 'themes'):
        if 'themes' not in json.loads(store.get_run(run_id)['checkpoints_json']):
            specs = plan_themes(store.library) if orchestrator.config.themes_enabled else []
            store.add_phase(run_id, 'themes', specs, fingerprint)
        counts = await executor.drive(run_id, finalize=False)
        if _incomplete(counts):
            reconcile_staleness(store, run_id, orchestrator)
            return store.finish(run_id)
        store.checkpoint(run_id, 'themes', fingerprint)
        with store.library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE library_state SET themes_fingerprint=? WHERE singleton_id=1', (fingerprint,)
            )
    if not _done(store, run_id, 'crossrefs'):
        if 'crossrefs' not in json.loads(store.get_run(run_id)['checkpoints_json']):
            specs = plan_crossrefs(orchestrator) if orchestrator.config.inject_crossrefs else []
            store.add_phase(run_id, 'crossrefs', specs, fingerprint)
        counts = await executor.drive(run_id, finalize=False)
        if _incomplete(counts):
            return store.finish(run_id)
        store.checkpoint(run_id, 'crossrefs', fingerprint)
        with store.library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE library_state SET crossrefs_fingerprint=? WHERE singleton_id=1',
                (fingerprint,),
            )
    if not _done(store, run_id, 'reconcile'):
        reconcile_staleness(store, run_id, orchestrator)
        store.checkpoint(
            run_id, 'reconcile', identity_hash([j['job_id'] for j in store.jobs(run_id)])
        )
    return store.finish(run_id)


async def run_generation(
    orchestrator: DocGenOrchestrator, files: list[Path], files_skipped: int
) -> PipelineResult:
    from docgen.orchestrator import PipelineResult

    store = SpoolStore(orchestrator._library.inference_context, orchestrator._library)
    specs = await plan_documents(orchestrator, files)
    scope = _scope(orchestrator)
    scope['files_skipped'] = files_skipped
    run_id = store.create(specs, operation='generate', scope=scope)
    orchestrator._emit(
        f'Local spool {run_id}: foreground execution; interruption leaves work resumable'
    )
    result = await complete_generation(store, run_id, orchestrator)
    return PipelineResult(
        files_processed=len(files),
        files_skipped=files_skipped,
        docs_created=result['applied'],
        docs_failed=result['failed'] + result['obsolete'] + result['cancelled'],
        errors=() if result['state'] == 'completed' else (f'Local spool incomplete: {run_id}',),
    )


def plan_catalog_descriptions(
    library: Library, *, source_name: str, force: bool = False, max_calls: int | None = None
) -> tuple[list[SpoolJobSpec], int, int]:
    from docgen.catalog_describer import description_prompt

    specs = []
    candidates = [
        doc
        for doc in library.list_documents('catalog')
        if doc.source_name == source_name and doc.metadata.get('kind') == 'element'
    ]
    pending = [doc for doc in candidates if force or not doc.metadata.get('description')]
    if max_calls is not None:
        pending = pending[:max_calls]
    for doc in pending:
        custom_id = 'catalog:' + doc.id
        specs.append(
            SpoolJobSpec(
                custom_id,
                JobKind.CATALOG,
                doc.id,
                {
                    'content_type': 'catalog',
                    'title': doc.title,
                    'original_content': doc.content,
                    'metadata': doc.metadata,
                    'source_files': doc.source_files,
                    'source_name': source_name,
                },
                request=BatchRequest(
                    custom_id, '', description_prompt(doc.metadata), max_tokens=2048
                ),
                expected_version=doc.updated_at,
                phase='catalog',
            )
        )
    return specs, len(candidates), len(pending)


async def run_catalog_descriptions(
    library: Library, *, source_name: str, force: bool = False, max_calls: int | None = None
) -> dict:
    specs, total, pending = plan_catalog_descriptions(
        library, source_name=source_name, force=force, max_calls=max_calls
    )
    store = SpoolStore(library.inference_context, library)
    run_id = store.create(
        specs, operation='catalog-describe', scope={'source_name': source_name}, phase='catalog'
    )
    result = await LocalExecutor(store).drive(run_id)
    return {
        **result,
        'described': result['applied'],
        'already_had_description': total - pending,
        'total_candidates': total,
    }


async def run_theme_build(library: Library, *, enabled: bool = True) -> dict:
    store = SpoolStore(library.inference_context, library)
    run_id = store.create([], operation='themes', scope={'enabled': enabled}, phase='graph')
    return await resume_theme_build(store, run_id)


async def resume_theme_build(store: SpoolStore, run_id: str) -> dict:
    from docgen.themes import refresh_themes

    run = store.get_run(run_id)
    fingerprint = catalog_fingerprint(store.library)
    if not _done(store, run_id, 'graph'):
        if not store.begin_phase(run_id, 'graph', fingerprint):
            return store.finish(run_id)
        await refresh_themes(
            store.library,
            LibraryWriter(store.library),
            enabled=json.loads(run['scope_json'])['enabled'],
            summarize_kwargs={'max_calls': 0, 'concurrency': 1},
        )
        store.checkpoint(run_id, 'graph', fingerprint)
    elif json.loads(run['checkpoints_json'])['graph']['fingerprint'] != fingerprint:
        raise ObsoleteInput('Catalog changed after this theme run planned its semantic phases')
    if 'themes' not in json.loads(store.get_run(run_id)['checkpoints_json']):
        specs = plan_themes(store.library) if json.loads(run['scope_json'])['enabled'] else []
        store.add_phase(run_id, 'themes', specs, fingerprint)
    result = await LocalExecutor(store).drive(run_id, finalize=False)
    if result['applied'] == result['total'] and not _incomplete(result):
        store.checkpoint(run_id, 'themes', fingerprint)
        with store.library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE library_state SET graph_fingerprint=?,themes_fingerprint=? WHERE singleton_id=1',
                (fingerprint, fingerprint),
            )
    return store.finish(run_id)


async def resume_run(library: Library, run_id: str) -> dict:
    store = SpoolStore(library.inference_context, library)
    run = store.get_run(run_id)
    store.jobs(run_id)
    if run['state'] in {'completed', 'failed', 'cancelled', 'obsolete'}:
        return {**store.counts(run_id), 'state': run['state'], 'run_id': run_id}
    if run['operation_kind'] == 'themes':
        return await resume_theme_build(store, run_id)
    scope = json.loads(run['scope_json'])
    if not scope.get('pipeline'):
        return await LocalExecutor(store).drive(run_id)
    from config import get_config
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    context = library.inference_context
    config = OrchestratorConfig(
        source_path=Path(scope['source_path']),
        source_name=scope['source_name'],
        db_path=context.db_path,
        staleness_db_path=context.staleness_db_path,
        model=context.settings.chat.model,
        provider='local',
        inference_context=context,
        concurrency=1,
        doc_types=tuple(scope['doc_types']),
        dependencies=tuple(scope['dependencies']),
        inject_crossrefs=scope['inject_crossrefs'],
        themes_enabled=scope['themes_enabled'],
        source_config=get_config().get_source_scip_config(scope['source_name']),
        generation_scope=scope.get('generation_scope', 'stable'),
    )
    async with DocGenOrchestrator(config) as orchestrator:
        if run['operation_kind'] == 'onboard':
            return await complete_onboard(store, run_id, orchestrator)
        return await complete_generation(store, run_id, orchestrator)


def plan_catalog(
    library: Library,
    source_name: str,
    source_path: Path,
    *,
    source_config: SourceScipConfig | None = None,
    exclude_patterns: tuple[str, ...] = (),
    exclude_dir_names: tuple[str, ...] = (),
    force: bool = False,
) -> list[SpoolJobSpec]:
    """Capture structural replacements before the first embedding request."""
    from docgen.catalog_extractor import extract_elements
    from docgen.catalog_writer import (
        _detect_language,
        _element_content,
        _element_doc_id,
        _element_metadata,
        _file_index_doc_id,
        _file_index_metadata,
        iter_catalog_files,
    )

    specs = []
    for path in iter_catalog_files(source_path, exclude_patterns, exclude_dir_names):
        relative = str(path.relative_to(source_path))
        captured = file_fingerprint(path)
        index_id = _file_index_doc_id(source_name, relative)
        old_index = library.get_document(index_id)
        if not force and old_index and old_index.metadata.get('file_sha') == captured:
            continue
        elements = extract_elements(path, source_path, source_config=source_config)
        if file_fingerprint(path) != captured:
            raise ObsoleteInput(f'Source changed during extraction: {path}')
        snapshots = {str(path): captured}
        # Index artifacts also contribute to SCIP extraction.
        snapshots.update(artifact_snapshots(source_path, source_config))
        element_ids = []
        for element in elements:
            doc_id = _element_doc_id(source_name, element.qualified_name)
            element_ids.append(doc_id)
            old = library.get_document(doc_id)
            metadata = _element_metadata(element, source_name)
            content = _element_content(element)
            if (
                old
                and old.metadata.get('sha_at_sync') == element.body_sha
                and old.metadata.get('description')
            ):
                content += '\n\nDescription: ' + old.metadata['description']
                metadata.update(description=old.metadata['description'], generated=True)
            elif old and old.metadata.get('description'):
                metadata.update(description=None, generated=False)
            if (
                old
                and old.content == content
                and all(old.metadata.get(k) == v for k, v in metadata.items())
                and old.embedding is not None
            ):
                continue
            specs.append(
                SpoolJobSpec(
                    'catalog-embed:' + doc_id,
                    JobKind.EMBEDDING,
                    doc_id,
                    {
                        'content_type': 'catalog',
                        'title': element.qualified_name,
                        'content': content,
                        'metadata': metadata,
                        'source_files': [element.file],
                        'source_name': source_name,
                    },
                    snapshots=snapshots,
                    expected_version=old.updated_at if old else None,
                    phase='catalog-embed',
                )
            )
        removed = {}
        current_ids = set(element_ids)
        for doc_id in old_index.metadata.get('element_ids', []) if old_index else []:
            if doc_id not in current_ids:
                old = library.get_document(doc_id)
                if old:
                    removed[doc_id] = old.updated_at
        specs.append(
            SpoolJobSpec(
                'catalog-embed:' + index_id,
                JobKind.EMBEDDING,
                index_id,
                {
                    'content_type': 'catalog',
                    'title': f'file_index:{source_name}:{relative}',
                    'content': f'Catalog index for {relative} -- {len(element_ids)} elements.',
                    'metadata': _file_index_metadata(
                        source_name, _detect_language(path), captured, element_ids
                    ),
                    'source_files': [str(path)],
                    'source_name': source_name,
                    'delete_versions': removed,
                },
                snapshots=snapshots,
                expected_version=old_index.updated_at if old_index else None,
                phase='catalog-embed',
            )
        )
    return specs


async def run_catalog_sync(
    library: Library, source_name: str, source_path: Path, **kwargs: Any
) -> dict:
    specs = plan_catalog(library, source_name, source_path, **kwargs)
    store = SpoolStore(library.inference_context, library)
    run_id = store.create(
        specs, operation='catalog-sync', scope={'source_name': source_name}, phase='catalog-embed'
    )
    return await LocalExecutor(store).drive(run_id)


async def live_post_process(
    orchestrator: DocGenOrchestrator,
    crossref_progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """Reverse augmentation precedes the final semantic refresh in live mode."""
    from docgen.themes import refresh_themes

    contexts, snapshots = _reverse_inputs(orchestrator)
    specs = await plan_documents(
        orchestrator,
        sorted(contexts),
        phase='reverse',
        consumer_contexts=contexts,
        extra_snapshots=snapshots,
    )
    for spec in specs:
        data = spec.data
        response = await orchestrator._generator._call_llm(
            spec.request.system_prompt, spec.request.user_prompt
        )
        from docgen.validator import ContentValidator

        validation = ContentValidator().validate(response, data['content_type'], data['title'])
        if not validation.is_valid:
            raise ValueError(f'Reverse augmentation failed validation for {data["title"]}')
        for path, captured in spec.snapshots.items():
            if file_fingerprint(Path(path)) != captured:
                raise ObsoleteInput(f'Reverse augmentation input changed: {path}')
        await orchestrator._store_document(
            GeneratedDoc(
                data['title'],
                response,
                data['content_type'],
                tuple(data['source_files']),
                data['metadata'],
            )
        )
    summary = await refresh_themes(
        orchestrator._library,
        orchestrator._writer,
        enabled=orchestrator.config.themes_enabled,
        summarize_kwargs={'concurrency': 1},
    )
    if summary.get('failed'):
        raise ValueError(f'{summary["failed"]} required theme summaries failed')
    if orchestrator.config.inject_crossrefs:
        await orchestrator._inject_crossrefs_scoped(progress_callback=crossref_progress)
    fingerprint = catalog_fingerprint(orchestrator._library)
    with orchestrator._library._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE library_state SET graph_fingerprint=?, themes_fingerprint=?, crossrefs_fingerprint=? WHERE singleton_id=1',
            (fingerprint, fingerprint, fingerprint),
        )


async def refresh_incremental(library: Library, source_name: str, source_path: Path) -> None:
    """Finish incremental CLI writes with the same local phase owner as generate."""
    from config import get_config
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    context = library.inference_context
    cfg = get_config()
    config = OrchestratorConfig(
        source_path=source_path,
        source_name=source_name,
        db_path=context.db_path,
        staleness_db_path=context.staleness_db_path,
        model=context.settings.chat.model,
        provider=context.settings.chat.provider,
        inference_context=context,
        concurrency=1,
        dependencies=tuple(cfg.get_source_dependencies(source_name)),
        themes_enabled=cfg.themes_enabled,
        source_config=cfg.get_source_scip_config(source_name),
    )
    async with DocGenOrchestrator(config) as orchestrator:
        await live_post_process(orchestrator)


async def run_onboard(orchestrator: DocGenOrchestrator) -> dict:
    store = SpoolStore(orchestrator._library.inference_context, orchestrator._library)
    run_id = store.create([], operation='onboard', scope=_scope(orchestrator), phase='structural')
    orchestrator._emit(f'Local onboarding {run_id}; resume with ariadne batch resume {run_id}')
    return await complete_onboard(store, run_id, orchestrator)


async def complete_onboard(
    store: SpoolStore, run_id: str, orchestrator: DocGenOrchestrator
) -> dict:
    import argparse

    from cli.index import cmd_discover, cmd_index
    from config import get_config
    from docgen.catalog_writer import iter_catalog_files

    cfg = get_config()
    c = orchestrator.config
    if not _done(store, run_id, 'structural'):
        if not store.begin_phase(
            run_id, 'structural', identity_hash({'source_path': str(c.source_path)})
        ):
            return store.finish(run_id)
        args = argparse.Namespace(
            source=c.source_name,
            db=store.context.db_path,
            all=False,
            dry_run=False,
            review=False,
            config_only=False,
            kind=None,
            quiet=True,
            verbose=False,
        )
        if cmd_discover(args) != 0 or cmd_index(args) != 0:
            raise ValueError(
                f'Structural onboarding failed; resume {run_id} after fixing the index'
            )
        store.checkpoint(run_id, 'structural', identity_hash({'source_path': str(c.source_path)}))
    # Discovery may have changed language routing; indexing populated the graph
    # after this orchestrator was opened. Refresh both before building prompts.
    from attrs import evolve

    from docgen.scip_cross_source import CrossSourceGraph

    c = evolve(c, source_config=cfg.get_source_scip_config(c.source_name))
    orchestrator.config = c
    graph = CrossSourceGraph()
    with store.library._conn_provider.acquire() as conn:
        graph.load_from(conn)
    orchestrator._cross_source_graph = graph
    executor = LocalExecutor(store)
    for phase in ('catalog-embed', 'catalog-describe'):
        if _done(store, run_id, phase):
            continue
        if phase not in json.loads(store.get_run(run_id)['checkpoints_json']):
            if phase == 'catalog-embed':
                source = cfg.get_source_config(c.source_name)
                specs = plan_catalog(
                    store.library,
                    c.source_name,
                    c.source_path,
                    source_config=c.source_config,
                    exclude_patterns=source.exclude if source else (),
                    exclude_dir_names=cfg.resolve_excluded_dirs(c.source_name),
                )
            else:
                specs, _, _ = plan_catalog_descriptions(store.library, source_name=c.source_name)
            store.add_phase(run_id, phase, specs, identity_hash([spec.target_id for spec in specs]))
        counts = await executor.drive(run_id, finalize=False)
        if _incomplete(counts):
            return store.finish(run_id)
        store.checkpoint(
            run_id,
            phase,
            identity_hash(
                [
                    job['request_hash']
                    for job in store.jobs(run_id)
                    if json.loads(job['payload_json'])['phase'] == phase
                ]
            ),
        )
    if 'base' not in json.loads(store.get_run(run_id)['checkpoints_json']):
        source = cfg.get_source_config(c.source_name)
        files = iter_catalog_files(
            c.source_path,
            source.exclude if source else (),
            cfg.resolve_excluded_dirs(c.source_name),
        )
        files = orchestrator._staleness.get_stale_files(
            files, base_path=c.source_path, requested_types=c.doc_types, library=store.library
        )
        specs = await plan_documents(orchestrator, files)
        store.add_phase(run_id, 'base', specs, identity_hash([spec.target_id for spec in specs]))
    return await complete_generation(store, run_id, orchestrator)


def plan_crossrefs(orchestrator: DocGenOrchestrator) -> list[SpoolJobSpec]:
    from docgen.crossref import CrossReference, inject_related_section
    from git_ops import get_current_branch
    from library import filter_by_branch

    scoped = orchestrator._scoped_lib()
    docs = filter_by_branch(
        scoped.get_documents_batch([doc.id for doc in scoped.list_documents_lite()]),
        get_current_branch(orchestrator.config.source_path) or 'main',
    )
    titles = {doc.id: doc.title for doc in docs}
    specs = []
    for doc in docs:
        if doc.content_type not in {
            'explanation',
            'architecture',
            'qa',
            'gotcha',
            'diagram',
            'theme',
        } or not doc.metadata.get('generated'):
            continue
        related = scoped.get_related(doc.id, max_hops=2, limit=10)
        refs = [
            CrossReference(
                source_id=doc.id,
                target_id=row['id'],
                ref_type='graph',
                context=f'Related (graph distance {row["distance"]:.2f})',
            )
            for row in related
            if row['id'] in titles
        ]
        if not refs:
            continue
        content = inject_related_section(doc.content, refs, titles)
        if content == doc.content:
            continue
        specs.append(
            SpoolJobSpec(
                'crossrefs:' + doc.id,
                JobKind.EMBEDDING,
                doc.id,
                {
                    'content_type': doc.content_type,
                    'title': doc.title,
                    'content': content,
                    'metadata': doc.metadata,
                    'source_files': doc.source_files,
                    'source_name': doc.source_name,
                    'reference_titles': {ref.target_id: titles[ref.target_id] for ref in refs},
                },
                expected_version=doc.updated_at,
                phase='crossrefs',
            )
        )
    return specs


def structural_fingerprint(library: Library) -> str:
    from library.model_binding import table_exists

    rows = {}
    with library._conn_provider.acquire() as conn:
        for table in ('scip_symbols', 'scip_edges', 'scip_index_state'):
            if table_exists(conn, table):
                rows[table] = sorted(
                    conn.execute('SELECT * FROM ' + table).fetchall(),
                    key=json.dumps,
                )
    return identity_hash(rows)


def artifact_snapshots(root: Path, source_config: SourceScipConfig | None = None) -> dict[str, str]:
    snapshots = {str(path): file_fingerprint(path) for path in (root / '.ariadne').rglob('*.scip')}
    artifact = getattr(source_config, 'artifact_path', None)
    if artifact and Path(artifact).is_file():
        snapshots[str(artifact)] = file_fingerprint(Path(artifact))
    return snapshots
