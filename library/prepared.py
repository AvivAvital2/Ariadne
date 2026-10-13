"""Prepared, per-document application using one caller-owned connection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from sqlite3 import Connection

import numpy as np
from attrs import evolve
from numpy.typing import NDArray

from model_config import EMBEDDING_DIM
from schema import CONTENT_TYPES, Chunk, Document, Section, _now_iso

VECTOR_NORM_TOLERANCE = 1e-5


class ObsoleteInput(ValueError):  # noqa: N818 - public contract name.
    """A captured target or source changed before application."""


def validate_vector(vector: NDArray[np.float32] | None) -> None:
    if vector is None:
        return
    array = np.asarray(vector)
    if (
        array.shape != (EMBEDDING_DIM,)
        or array.dtype != np.float32
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f'Expected a finite float32 vector of width {EMBEDDING_DIM}')
    if abs(float(np.linalg.norm(array.astype(np.float64))) - 1) > VECTOR_NORM_TOLERANCE:
        raise ValueError('Stored embedding must have unit norm')


@dataclass(frozen=True)
class PreparedDocument:
    document: Document
    chunks: tuple[Chunk, ...] = ()
    sections: tuple[Section, ...] = ()
    expected_version: str | None = None
    content_token_count: int | None = None
    embedding_space_id: str | None = None
    theme_update: dict = field(default_factory=dict)


def apply_document(conn: Connection, prepared: PreparedDocument) -> Document:
    """No HTTP, connection acquisition, commit, or parent delete/reinsert here."""
    doc = prepared.document
    if doc.content_type not in CONTENT_TYPES or not doc.title.strip() or not doc.content.strip():
        raise ValueError('Invalid document')
    for record in (doc, *prepared.chunks, *prepared.sections):
        validate_vector(record.embedding)
    if any(child.document_id != doc.id for child in (*prepared.chunks, *prepared.sections)):
        raise ValueError('Prepared children must belong to their document')
    previous = conn.execute(
        'SELECT updated_at, created_at, metadata, source_name, content, content_token_count FROM documents WHERE id=?',
        (doc.id,),
    ).fetchone()
    version = previous[0] if previous else None
    if version != prepared.expected_version:
        raise ObsoleteInput(f'Document {doc.id} changed since preparation')
    metadata = {**(json.loads(previous[2]) if previous else {}), **doc.metadata}
    source_name = (
        doc.source_name if doc.source_name is not None else (previous[3] if previous else None)
    )
    count = prepared.content_token_count
    if count is None and previous and previous[4] == doc.content:
        count = previous[5]
    if previous and not prepared.theme_update:
        stored = conn.execute(
            'SELECT content_type,title,content,source_files,embedding,metadata,source_name FROM documents WHERE id=?',
            (doc.id,),
        ).fetchone()
        values = (
            doc.content_type,
            doc.title,
            doc.content,
            doc.source_files_json(),
            doc.embedding_bytes(),
            json.dumps(metadata),
            source_name,
        )
        chunks = conn.execute(
            'SELECT chunk_index,content,embedding FROM chunks WHERE document_id=? ORDER BY chunk_index',
            (doc.id,),
        ).fetchall()
        sections = conn.execute(
            'SELECT idx,heading,description,content,embedding FROM sections WHERE document_id=? ORDER BY idx',
            (doc.id,),
        ).fetchall()
        if (
            stored == values
            and chunks == [(c.chunk_index, c.content, c.embedding_bytes()) for c in prepared.chunks]
            and sections
            == [
                (s.index, s.heading, s.description, s.content, s.embedding_bytes())
                for s in prepared.sections
            ]
        ):
            return evolve(
                doc,
                metadata=metadata,
                source_name=source_name,
                created_at=previous[1],
                updated_at=previous[0],
            )
    doc = evolve(
        doc,
        metadata=metadata,
        source_name=source_name,
        created_at=previous[1] if previous else doc.created_at,
        updated_at=_now_iso(),
    )
    conn.execute(
        """INSERT INTO documents
        (id,content_type,title,content,source_files,embedding,created_at,updated_at,metadata,source_name,content_token_count)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET content_type=excluded.content_type, title=excluded.title,
        content=excluded.content, source_files=excluded.source_files, embedding=excluded.embedding,
        updated_at=excluded.updated_at, metadata=excluded.metadata, source_name=excluded.source_name,
        content_token_count=excluded.content_token_count""",  # noqa: Q001 - repository quote rule conflicts with the formatter.
        (
            doc.id,
            doc.content_type,
            doc.title,
            doc.content,
            doc.source_files_json(),
            doc.embedding_bytes(),
            doc.created_at,
            doc.updated_at,
            doc.metadata_json(),
            doc.source_name,
            count,
        ),
    )
    conn.execute('DELETE FROM chunks WHERE document_id=?', (doc.id,))
    conn.executemany(
        'INSERT INTO chunks (id,document_id,chunk_index,content,embedding) VALUES (?,?,?,?,?)',
        [
            (c.id, c.document_id, c.chunk_index, c.content, c.embedding_bytes())
            for c in prepared.chunks
        ],
    )
    conn.execute('DELETE FROM sections WHERE document_id=?', (doc.id,))
    conn.executemany(
        'INSERT INTO sections (document_id,idx,heading,description,content,embedding) VALUES (?,?,?,?,?,?)',
        [
            (s.document_id, s.index, s.heading, s.description, s.content, s.embedding_bytes())
            for s in prepared.sections
        ],
    )
    if prepared.theme_update:
        t = prepared.theme_update
        if set(t) != {'cluster_id', 'summary_hash', 'coherent'}:
            raise ValueError('Unsupported prepared theme metadata')
        conn.execute(
            'UPDATE themes SET summary_hash=?, coherent=?, dirty=0, last_summarized_at=? WHERE cluster_id=? AND doc_id=?',
            (t['summary_hash'], int(t['coherent']), doc.updated_at, t['cluster_id'], doc.id),
        )
    # Only catalog content feeds the semantic graph; generated prose decoration
    # must not invalidate its own upstream graph/theme inputs.
    if doc.content_type == 'catalog':
        conn.execute(
            'UPDATE library_state SET graph_fingerprint=NULL, themes_fingerprint=NULL, crossrefs_fingerprint=NULL WHERE singleton_id=1'
        )
    return doc
