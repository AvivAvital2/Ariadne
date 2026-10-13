"""Durable foreground local work, with per-document application boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inference import InferenceContext
    from library import Library

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from docgen.llm.batch import BatchRequest
from docgen.llm.local import LocalInferenceError, TransientInferenceError, build_chat_request
from library.model_binding import ModelBindingMismatch, readonly, table_exists
from library.prepared import ObsoleteInput
from model_config import canonical_json, identity_hash
from schema import Document, _now_iso
from writer import LibraryWriter

SPOOL_VERSION = 1
PAYLOAD_VERSION = 1
PARSER_VERSION = 1
TERMINAL_RUNS = frozenset({'completed', 'failed', 'obsolete', 'cancelled'})
TERMINAL_JOBS = frozenset({'applied', 'failed', 'obsolete', 'cancelled'})


class UnsupportedSpoolVersion(ValueError):  # noqa: N818 - public contract name.
    pass


class JobKind(StrEnum):
    DOCUMENT = 'per_file_document'
    CATALOG = 'catalog_description'
    THEME = 'theme_summary'
    EMBEDDING = 'embedding_application'


@dataclass(frozen=True)
class SpoolJobSpec:
    custom_id: str
    kind: JobKind
    target_id: str
    data: dict
    request: BatchRequest | None = None
    snapshots: dict[str, str] = field(default_factory=dict)
    context_versions: dict[str, str | None] = field(default_factory=dict)
    expected_version: str | None = None
    phase: str = 'base'
    prompt_version: int = 1
    parser_version: int = PARSER_VERSION

    def payload(self, context: InferenceContext) -> dict:
        if self.kind == JobKind.EMBEDDING and self.request is not None:
            raise ValueError('Embedding-only jobs must not contain a chat request')
        if self.kind != JobKind.EMBEDDING and self.request is None:
            raise ValueError('Chat jobs require their exact request')
        request = None
        if self.request is not None:
            if self.request.custom_id != self.custom_id:
                raise ValueError('Spool custom ID differs from request custom ID')
            request = build_chat_request(
                context.settings.chat,
                self.request.system_prompt,
                self.request.user_prompt,
                self.request.max_tokens,
            )
        return {
            'version': PAYLOAD_VERSION,
            'prompt_version': self.prompt_version,
            'parser_version': self.parser_version,
            'kind': self.kind.value,
            'target_id': self.target_id,
            'data': self.data,
            'request': request,
            'snapshots': self.snapshots,
            'context_versions': self.context_versions,
            'expected_version': self.expected_version,
            'phase': self.phase,
        }


def file_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dicts(cursor: sqlite3.Cursor) -> list[dict]:
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


class SpoolStore:
    def __init__(self, context: InferenceContext, library: Library | None = None) -> None:
        if not context.local:
            raise ValueError('Local spools require a local model profile')
        self.context = context
        self.library = library
        self.path = context.db_path

    def initialize(self) -> None:
        self.context.require_writer()
        with self.library._conn_provider.acquire() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS spool_runs (
                run_id TEXT PRIMARY KEY, library_uuid TEXT NOT NULL, profile_hash TEXT NOT NULL,
                schema_version INTEGER NOT NULL, operation_kind TEXT NOT NULL, scope_json TEXT NOT NULL,
                phase TEXT NOT NULL, phase_fingerprint TEXT, checkpoints_json TEXT NOT NULL,
                settings_json TEXT NOT NULL, state TEXT NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")  # noqa: Q001 - repository quote rule conflicts with the formatter.
            conn.execute("""CREATE TABLE IF NOT EXISTS spool_jobs (
                job_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES spool_runs(run_id) ON DELETE CASCADE,
                custom_id TEXT NOT NULL, task_kind TEXT NOT NULL, target_id TEXT NOT NULL,
                payload_json TEXT NOT NULL, request_hash TEXT NOT NULL, snapshots_json TEXT NOT NULL,
                state TEXT NOT NULL, request_attempts INTEGER NOT NULL DEFAULT 0,
                validation_attempts INTEGER NOT NULL DEFAULT 0, apply_attempts INTEGER NOT NULL DEFAULT 0,
                retry_at REAL, response_text TEXT, parsed_result_json TEXT, failure_stage TEXT,
                error_class TEXT, error_message TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(run_id,custom_id), UNIQUE(run_id,target_id,request_hash))""")  # noqa: Q001 - repository quote rule conflicts with the formatter.

    def list_runs(self) -> list[dict]:
        if not self.path.exists():
            return []
        with readonly(self.path) as conn:
            if not table_exists(conn, 'spool_runs'):
                return []
            return _dicts(
                conn.execute(
                    'SELECT run_id, operation_kind, phase, state, cancel_requested, created_at, updated_at FROM spool_runs ORDER BY created_at DESC'
                )
            )

    def get_run(self, run_id: str) -> dict:
        with readonly(self.path) as conn:
            if not table_exists(conn, 'spool_runs'):
                raise ValueError(f'Unknown local run {run_id}')
            rows = _dicts(conn.execute('SELECT * FROM spool_runs WHERE run_id=?', (run_id,)))
        if not rows:
            raise ValueError(f'Unknown local run {run_id}')
        run = rows[0]
        binding = self.context.binding
        if binding is None or (run['library_uuid'], run['profile_hash']) != (
            binding.library_uuid,
            binding.profile_hash,
        ):
            raise ModelBindingMismatch('Spool belongs to a different library or model profile')
        if run['schema_version'] != SPOOL_VERSION:
            raise UnsupportedSpoolVersion(
                'Unsupported spool schema; finish with the matching application version or explicitly clear and replan'
            )
        return run

    def jobs(self, run_id: str) -> list[dict]:
        return self._read_jobs(run_id)

    def _read_jobs(self, run_id: str, *, job_id: str | None = None) -> list[dict]:
        self.get_run(run_id)
        query = 'SELECT * FROM spool_jobs WHERE run_id=?'
        parameters = (run_id,)
        if job_id is not None:
            query += ' AND job_id=?'
            parameters += (job_id,)
        with readonly(self.path) as conn:
            rows = _dicts(conn.execute(query + ' ORDER BY rowid', parameters))
        for row in rows:
            payload = json.loads(row['payload_json'])
            if (
                payload.get('version') != PAYLOAD_VERSION
                or payload.get('parser_version') != PARSER_VERSION
                or payload.get('prompt_version') != 1
                or payload.get('kind') not in set(JobKind)
            ):
                raise UnsupportedSpoolVersion(
                    'Unsupported local job payload, prompt or parser version; spool was left intact'
                )
            if identity_hash(payload) != row['request_hash']:
                raise UnsupportedSpoolVersion(
                    'Spool payload hash does not match its immutable request'
                )
        return rows

    def create(
        self, specs: list[SpoolJobSpec], *, operation: str, scope: dict, phase: str = 'base'
    ) -> str:
        self.initialize()
        self.context.validate()
        run_id = 'local:' + str(uuid.uuid4())
        now = _now_iso()
        payloads = [spec.payload(self.context) for spec in specs]
        with self.library._conn_provider.acquire() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute(
                'INSERT INTO spool_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (
                    run_id,
                    self.context.binding.library_uuid,
                    self.context.settings.profile_hash,
                    SPOOL_VERSION,
                    operation,
                    canonical_json(scope),
                    phase,
                    None,
                    '{}',
                    canonical_json(asdict(self.context.settings)),
                    'running',
                    0,
                    now,
                    now,
                ),
            )
            self._insert_jobs(conn, run_id, specs, payloads)
        return run_id

    def _insert_jobs(
        self, conn: sqlite3.Connection, run_id: str, specs: list[SpoolJobSpec], payloads: list[dict]
    ) -> None:
        now = _now_iso()
        for spec, payload in zip(specs, payloads, strict=True):
            state = 'apply_pending' if spec.kind == JobKind.EMBEDDING else 'queued'
            parsed = canonical_json(spec.data) if spec.kind == JobKind.EMBEDDING else None
            conn.execute(
                """INSERT INTO spool_jobs
                (job_id,run_id,custom_id,task_kind,target_id,payload_json,request_hash,snapshots_json,state,parsed_result_json,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",  # noqa: Q001 - repository quote rule conflicts with the formatter.
                (
                    str(uuid.uuid4()),
                    run_id,
                    spec.custom_id,
                    spec.kind.value,
                    spec.target_id,
                    canonical_json(payload),
                    identity_hash(payload),
                    canonical_json(spec.snapshots),
                    state,
                    parsed,
                    now,
                    now,
                ),
            )

    def add_phase(
        self, run_id: str, phase: str, specs: list[SpoolJobSpec], fingerprint: str
    ) -> None:
        self.context.require_writer()
        run = self.get_run(run_id)
        if run['cancel_requested']:
            return
        payloads = [spec.payload(self.context) for spec in specs]
        checkpoints = json.loads(run['checkpoints_json'])
        if phase in checkpoints:
            return
        checkpoints[phase] = {'state': 'planned', 'fingerprint': fingerprint}
        with self.library._conn_provider.acquire() as conn:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute(
                'SELECT cancel_requested FROM spool_runs WHERE run_id=?', (run_id,)
            ).fetchone()[0]:
                return
            self._insert_jobs(conn, run_id, specs, payloads)
            conn.execute(
                'UPDATE spool_runs SET phase=?,phase_fingerprint=?,checkpoints_json=?,updated_at=? WHERE run_id=?',
                (phase, fingerprint, canonical_json(checkpoints), _now_iso(), run_id),
            )

    def checkpoint(
        self, run_id: str, phase: str, fingerprint: str, *, state: str = 'complete'
    ) -> None:
        self.context.require_writer()
        run = self.get_run(run_id)
        checkpoints = json.loads(run['checkpoints_json'])
        checkpoints[phase] = {
            **checkpoints.get(phase, {}),
            'state': state,
            'fingerprint': fingerprint,
        }
        with self.library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE spool_runs SET phase=?,phase_fingerprint=?,checkpoints_json=?,updated_at=? WHERE run_id=? AND cancel_requested=0',
                (phase, fingerprint, canonical_json(checkpoints), _now_iso(), run_id),
            )

    def begin_phase(self, run_id: str, phase: str, fingerprint: str) -> bool:
        """Persist a bounded attempt before a fixed command phase starts."""
        self.context.require_writer()
        run = self.get_run(run_id)
        if run['cancel_requested']:
            return False
        checkpoints = json.loads(run['checkpoints_json'])
        checkpoint = checkpoints.get(phase, {})
        used = checkpoint.get('attempts', 0)
        if used >= json.loads(run['settings_json'])['execution']['max_attempts']:
            self.checkpoint(run_id, phase, fingerprint, state='failed')
            self.finish(run_id)
            raise LocalInferenceError(f'Persisted phase attempt budget exhausted: {phase}')
        checkpoints[phase] = {
            **checkpoint,
            'state': 'running',
            'fingerprint': fingerprint,
            'attempts': used + 1,
        }
        with self.library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE spool_runs SET phase=?,phase_fingerprint=?,checkpoints_json=?,updated_at=? WHERE run_id=? AND cancel_requested=0',
                (phase, fingerprint, canonical_json(checkpoints), _now_iso(), run_id),
            )
        return True

    def update_job(self, job_id: str, **values: Any) -> None:
        self.context.require_writer()
        allowed = {
            'state',
            'request_attempts',
            'validation_attempts',
            'apply_attempts',
            'retry_at',
            'response_text',
            'parsed_result_json',
            'failure_stage',
            'error_class',
            'error_message',
        }
        if set(values) - allowed:
            raise ValueError('Unknown spool field')
        values['updated_at'] = _now_iso()
        with self.library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE spool_jobs SET '
                + ','.join(name + '=?' for name in values)
                + " WHERE job_id=? AND state NOT IN ('applied','cancelled') AND run_id IN (SELECT run_id FROM spool_runs WHERE cancel_requested=0)",
                (*values.values(), job_id),
            )

    def cancel(self, run_id: str) -> None:
        # The only corpus-lock exception: short control-state-only transaction.
        self.get_run(run_id)
        with sqlite3.connect(self.path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute(
                "UPDATE spool_runs SET cancel_requested=1,state='cancelled',updated_at=? WHERE run_id=? AND state!='completed'",
                (_now_iso(), run_id),
            )
            conn.execute(
                "UPDATE spool_jobs SET state='cancelled',updated_at=? WHERE run_id=? AND state NOT IN ('applied','failed','obsolete')",
                (_now_iso(), run_id),
            )

    def clear(self, run_id: str) -> None:
        self.context.require_writer()
        if self.get_run(run_id)['state'] not in TERMINAL_RUNS:
            raise ValueError('Only terminal local runs can be cleared; cancel active work first')
        with self.library._conn_provider.acquire() as conn:
            conn.execute('DELETE FROM spool_jobs WHERE run_id=?', (run_id,))
            conn.execute('DELETE FROM spool_runs WHERE run_id=?', (run_id,))

    def counts(self, run_id: str) -> dict:
        rows = self.jobs(run_id)
        states = Counter(row['state'] for row in rows)
        return {
            'total': len(rows),
            'responses': sum(row['response_text'] is not None for row in rows),
            **{
                state: states[state]
                for state in (
                    'queued',
                    'running',
                    'response_ready',
                    'apply_pending',
                    'applied',
                    'failed',
                    'obsolete',
                    'cancelled',
                )
            },
        }

    def finish(self, run_id: str) -> dict:
        counts = self.counts(run_id)
        run = self.get_run(run_id)
        if run['cancel_requested']:
            state = 'cancelled'
        elif counts['failed'] or any(
            checkpoint.get('state') == 'failed'
            for checkpoint in json.loads(run['checkpoints_json']).values()
        ):
            state = 'failed'
        elif counts['obsolete']:
            state = 'obsolete'
        elif counts['applied'] == counts['total']:
            state = 'completed'
        else:
            state = 'running'
        with self.library._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE spool_runs SET state=?,updated_at=? WHERE run_id=? AND cancel_requested=0',
                (state, _now_iso(), run_id),
            )
        return {**counts, 'state': state, 'run_id': run_id}


class LocalExecutor:
    """One worker. Stage attempts are durable before any work in that stage."""

    def __init__(self, store: SpoolStore) -> None:
        self.store = store
        self.context = store.context
        self.library = store.library
        self.writer = LibraryWriter(self.library)

    def check_inputs(self, payload: dict) -> None:
        structural = payload['data'].get('structural_fingerprint')
        if structural is not None:
            from docgen.local_pipeline import structural_fingerprint

            if structural_fingerprint(self.library) != structural:
                raise ObsoleteInput('Structural context changed after prompt construction')
        for doc_id, title in payload['data'].get('reference_titles', {}).items():
            doc = self.library.get_document(doc_id)
            if doc is None or doc.title != title:
                raise ObsoleteInput(f'Cross-reference target changed: {doc_id}')
        for path, captured in payload['snapshots'].items():
            try:
                matches = file_fingerprint(Path(path)) == captured
            except OSError:
                matches = False
            if not matches:
                raise ObsoleteInput(f'Captured source changed: {path}')
        for doc_id, expected in payload['context_versions'].items():
            doc = self.library.get_document(doc_id)
            if (doc.updated_at if doc else None) != expected:
                raise ObsoleteInput(f'Captured context changed: {doc_id}')
        old = self.library.get_document(payload['target_id'])
        if (old.updated_at if old else None) != payload['expected_version']:
            raise ObsoleteInput(f'Target changed: {payload["target_id"]}')
        if payload['kind'] == JobKind.THEME:
            from docgen.themes import _gather_member_data, compute_summary_hash

            data = payload['data']
            theme = self.library.get_theme(data['cluster_id'])
            if (
                theme is None
                or theme.doc_id != payload['target_id']
                or theme.resolution != data['resolution']
            ):
                raise ObsoleteInput('Theme configuration changed')
            members, summaries = _gather_member_data(self.library, data['cluster_id'])
            if compute_summary_hash(members, summaries, data['resolution']) != data['summary_hash']:
                raise ObsoleteInput('Theme membership changed')

    def parse(self, payload: dict, response: str) -> dict:
        data = dict(payload['data'])
        kind = payload['kind']
        if kind == JobKind.DOCUMENT:
            from docgen.validator import ContentValidator

            result = ContentValidator().validate(response, data['content_type'], data['title'])
            if not result.is_valid:
                raise LocalInferenceError(
                    'Generated document failed validation (local batches do not repair output)'
                )
            data['content'] = response
        elif kind == JobKind.CATALOG:
            response = response.strip()
            if not response:
                raise LocalInferenceError('Empty catalog description')
            data['content'] = (
                data['original_content'].split('\n\nDescription: ', 1)[0].rstrip()
                + '\n\nDescription: '
                + response
            )
            data['metadata'] = {**data['metadata'], 'description': response, 'generated': True}
        elif kind == JobKind.THEME:
            data['coherent'] = not response.strip().startswith('INCOHERENT')
            if data['coherent']:
                data['content'] = response.strip()
                data['title'] = (
                    response.split('\n', 1)[0].lstrip('#').strip()[:120] or data['title']
                )
                data['metadata'] = {
                    **data['metadata'],
                    'pending': False,
                    'generated': True,
                    'theme_title': data['title'],
                }
        else:
            raise UnsupportedSpoolVersion('Unsupported chat job kind')
        return data

    async def apply(self, job: dict, payload: dict) -> None:
        self.check_inputs(payload)
        data = json.loads(job['parsed_result_json'])
        if payload['kind'] == JobKind.THEME and not data['coherent']:
            with self.library._conn_provider.acquire() as conn:
                conn.execute('BEGIN IMMEDIATE')
                if conn.execute(
                    'SELECT cancel_requested FROM spool_runs WHERE run_id=?', (job['run_id'],)
                ).fetchone()[0]:
                    return
                self.check_inputs(payload)
                conn.execute(
                    'UPDATE themes SET coherent=0,dirty=0,summary_hash=?,last_summarized_at=? WHERE cluster_id=?',
                    (data['summary_hash'], _now_iso(), data['cluster_id']),
                )
                conn.execute(
                    "UPDATE spool_jobs SET state='applied',updated_at=? WHERE job_id=?",
                    (_now_iso(), job['job_id']),
                )
            return
        doc = Document(
            id=payload['target_id'],
            content_type=data['content_type'],
            title=data['title'],
            content=data['content'],
            source_files=data.get('source_files', []),
            source_name=data.get('source_name'),
            metadata=data.get('metadata', {}),
        )
        theme = (
            {
                'cluster_id': data['cluster_id'],
                'summary_hash': data['summary_hash'],
                'coherent': True,
            }
            if payload['kind'] == JobKind.THEME
            else None
        )
        with self.context.single_attempt():
            prepared = await self.writer.prepare_document(doc, theme_update=theme)
        self.check_inputs(payload)
        prepared = replace(prepared, expected_version=payload['expected_version'])
        self.library.apply_prepared_document(
            prepared,
            job_id=job['job_id'],
            check_inputs=lambda: self.check_inputs(payload),
            delete_versions=data.get('delete_versions'),
        )

    async def drive(
        self, run_id: str, *, inference_only: bool = False, finalize: bool = True
    ) -> dict:
        self.context.require_writer()
        self.context.validate()
        run = self.store.get_run(run_id)
        jobs = self.store.jobs(run_id)  # Refuse unsupported versions before recovery writes.
        if run['cancel_requested'] or run['state'] in TERMINAL_RUNS:
            return {**self.store.counts(run_id), 'state': run['state'], 'run_id': run_id}
        budget = json.loads(run['settings_json'])['execution']['max_attempts']
        for initial in jobs:
            if initial['state'] == 'running':
                self.store.update_job(
                    initial['job_id'],
                    state='response_ready' if initial['response_text'] is not None else 'queued',
                )
            while True:
                job = next(iter(self.store._read_jobs(run_id, job_id=initial['job_id'])))
                if self.store.get_run(run_id)['cancel_requested'] or job['state'] in TERMINAL_JOBS:
                    break
                state = job['state']
                if inference_only and state in {'response_ready', 'apply_pending'}:
                    break
                stage = {
                    'queued': 'request',
                    'response_ready': 'validation',
                    'apply_pending': 'apply',
                }[state]
                attempts_field = stage + '_attempts'
                used = job[attempts_field]
                if used >= budget:
                    self.store.update_job(
                        job['job_id'],
                        state='failed',
                        failure_stage=stage,
                        error_class='AttemptsExhausted',
                        error_message='Persisted attempt budget exhausted',
                    )
                    break
                if job['retry_at'] and job['retry_at'] > time.time():
                    await asyncio.sleep(min(job['retry_at'] - time.time(), 30))
                payload = json.loads(job['payload_json'])
                self.store.update_job(
                    job['job_id'],
                    **{attempts_field: used + 1},
                    state='running' if stage == 'request' else state,
                    retry_at=None,
                )
                try:
                    self.check_inputs(payload)
                    if stage == 'request':
                        response = await self.context.chat.call_payload(
                            payload['request'], attempts=1
                        )
                        self.store.update_job(
                            job['job_id'], state='response_ready', response_text=response
                        )
                    elif stage == 'validation':
                        parsed = self.parse(payload, job['response_text'])
                        self.store.update_job(
                            job['job_id'],
                            state='apply_pending',
                            parsed_result_json=canonical_json(parsed),
                        )
                    else:
                        await self.apply(job, payload)
                except ObsoleteInput as exc:
                    self.store.update_job(
                        job['job_id'],
                        state='obsolete',
                        failure_stage=stage,
                        error_class=type(exc).__name__,
                        error_message=str(exc),
                    )
                except (ModelBindingMismatch, UnsupportedSpoolVersion):
                    raise
                except Exception as exc:  # noqa: BLE001 - persist a bounded failure for every job handler.
                    transient = isinstance(exc, TransientInferenceError)
                    retry = transient and used + 1 < budget
                    self.store.update_job(
                        job['job_id'],
                        state=state if retry else 'failed',
                        failure_stage=stage,
                        error_class=type(exc).__name__,
                        error_message=str(exc)[:300],
                        retry_at=time.time() + min(2**used, 8) if retry else None,
                    )
        if finalize and not inference_only:
            return self.store.finish(run_id)
        return {
            **self.store.counts(run_id),
            'run_id': run_id,
            'state': self.store.get_run(run_id)['state'],
        }
