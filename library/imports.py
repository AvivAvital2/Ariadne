"""Versioned generated-document imports with whole-input provenance preflight."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference import InferenceContext
    from library import Library

from pathlib import Path

import yaml

from library.model_binding import ModelBindingMismatch
from library.prepared import PreparedDocument
from schema import CONTENT_TYPES, Document
from writer import LibraryWriter


def read_bound_export(path: Path, context: InferenceContext) -> Document:
    text = path.read_text()
    if not text.startswith('---\n') or '\n---\n' not in text[4:]:
        raise ModelBindingMismatch(
            f'Unidentified export {path}; use explicit authored ingestion for authored text'
        )
    header, body = text[4:].split('\n---\n', 1)
    data = yaml.safe_load(header)
    if not isinstance(data, dict):
        raise ModelBindingMismatch(f'Invalid export frontmatter: {path}')
    provenance = data.get('ariadne_export')
    if not isinstance(provenance, dict) or provenance.get('version') != 1:
        raise ModelBindingMismatch(f'Missing or unsupported generated-export provenance: {path}')
    models = provenance.get('models', {})
    identity = context.settings.identity
    if (
        models.get('profile_hash') != context.settings.profile_hash
        or models.get('generation') != identity['generation']
        or models.get('embedding') != identity['embedding']
    ):
        raise ModelBindingMismatch(f'Export {path} belongs to a different model pair')
    offset = provenance.get('body_prefix_length')
    if type(offset) is not int or not 0 <= offset < len(body):
        raise ValueError(f'Invalid body framing in {path}')
    metadata = data.get('metadata') or {}
    if not isinstance(metadata, dict) or not isinstance(data.get('source_files', []), list):
        raise ValueError(f'Invalid document metadata in {path}')
    from model_config import canonical_json

    canonical_json(metadata)
    if (
        not isinstance(data.get('id'), str)
        or not data['id']
        or data.get('type') not in CONTENT_TYPES
        or not isinstance(data.get('title'), str)
        or not data['title'].strip()
        or not body[offset:].strip()
        or any(not isinstance(item, str) for item in data.get('source_files', []))
        or not isinstance(data.get('source_name'), (str, type(None)))
    ):
        raise ValueError(f'Invalid document fields in {path}')
    row_provenance = metadata.get('model_provenance')
    if metadata.get('generated') and (
        not isinstance(row_provenance, dict)
        or row_provenance.get('profile_hash') != context.settings.profile_hash
    ):
        raise ModelBindingMismatch(
            f'Generated document has missing or mismatched provenance: {path}'
        )
    return Document(
        id=data['id'],
        content_type=data['type'],
        title=data['title'],
        content=body[offset:],
        source_files=data.get('source_files', []),
        source_name=data.get('source_name'),
        metadata=metadata,
        created_at=str(data['created_at']),
        updated_at=str(data['updated_at']),
    )


async def import_bound_markdown(library: Library, directory: Path, *, embed: bool = True) -> int:
    context = library.inference_context
    if context is None:
        raise ModelBindingMismatch('A bound context is required for this import')
    context.require_writer()
    context.validate()

    def read_candidates() -> list[Document]:
        return [
            read_bound_export(path, context)
            for path in sorted(directory.rglob('*.md'))
            if path.name not in {'README.md', 'INDEX.md', 'CLAUDE.md'}
        ]

    docs = await asyncio.to_thread(read_candidates)
    if len({doc.id for doc in docs}) != len(docs):
        raise ValueError('Import contains duplicate document IDs')
    # No document is applied until every candidate has passed provenance checks.
    writer = LibraryWriter(library)
    for doc in docs:
        if embed:
            prepared = await writer.prepare_document(doc)
        else:
            old = library.get_document(doc.id)
            prepared = PreparedDocument(
                doc,
                expected_version=old.updated_at if old else None,
                embedding_space_id=context.settings.embeddings.embedding_space_id,
            )
        library.apply_prepared_document(prepared)
    return len(docs)
