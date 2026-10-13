"""Inspect and reserve immutable model identity before schema initialization."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from library.locking import WriteSession
from model_config import IDENTITY_VERSION, ModelSettings, canonical_json


class ModelBindingMismatch(ValueError):  # noqa: N818 - public contract name.
    """The library, sidecar or producer belongs to another model profile."""


@dataclass(frozen=True)
class ModelBinding:
    library_uuid: str
    profile_hash: str
    generation_identity_json: str
    embedding_identity_json: str
    schema_version: int = IDENTITY_VERSION

    @property
    def manifest(self) -> dict:
        return {
            'version': self.schema_version,
            'library_uuid': self.library_uuid,
            'profile_hash': self.profile_hash,
            'generation': json.loads(self.generation_identity_json),
            'embedding': json.loads(self.embedding_identity_json),
        }


@contextmanager
def readonly(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            'SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?', ('table', name)
        ).fetchone()
        is not None
    )


def _read_binding(conn: sqlite3.Connection) -> ModelBinding | None:
    if not table_exists(conn, 'library_model_binding'):
        return None
    row = conn.execute(
        'SELECT library_uuid, profile_hash, generation_identity_json, embedding_identity_json, schema_version FROM library_model_binding WHERE singleton_id=1'
    ).fetchone()
    if row is None:
        return None
    binding = ModelBinding(*row)
    if binding.schema_version != IDENTITY_VERSION:
        raise ModelBindingMismatch(f'Unsupported library binding version {binding.schema_version}')
    return binding


def inspect_binding(path: Path) -> ModelBinding | None:
    if not path.exists() or not path.stat().st_size:
        return None
    with readonly(path) as conn:
        return _read_binding(conn)


def populated(conn: sqlite3.Connection, *, ignore: frozenset[str] = frozenset()) -> bool:
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (name,) in tables:
        if name in ignore:
            continue
        quoted = '"' + name.replace('"', '""') + '"'
        if conn.execute(f'SELECT 1 FROM {quoted} LIMIT 1').fetchone():
            return True
    return False


def check_binding(binding: ModelBinding | None, settings: ModelSettings, path: Path) -> None:
    if binding is not None and binding.profile_hash != settings.profile_hash:
        actual = binding.manifest
        changed = [
            key for key in ('generation', 'embedding') if actual[key] != settings.identity[key]
        ]
        raise ModelBindingMismatch(
            f'ModelBindingMismatch for {path}: {", ".join(changed)} identity differs. '
            'Use a separate fresh library and staleness database; force/rebuild cannot change models.'
        )


def inspect_pair(
    main: Path, sidecar: Path, settings: ModelSettings, *, for_write: bool = False
) -> ModelBinding | None:
    """Read-only checks, including structural-only databases and legacy sidecars."""
    binding = inspect_binding(main)
    check_binding(binding, settings, main)
    legacy_hosted = False
    if binding is None and main.exists() and main.stat().st_size:
        with readonly(main) as conn:
            if populated(conn):
                if settings.local:
                    raise ModelBindingMismatch(
                        f'Populated unbound library {main}; use a fresh pair, including sidecar {sidecar}'
                    )
                legacy_hosted = True
    if sidecar.exists() and sidecar.stat().st_size:
        with readonly(sidecar) as conn:
            row = (
                conn.execute(
                    'SELECT library_uuid, profile_hash FROM staleness_owner WHERE singleton_id=1'
                ).fetchone()
                if table_exists(conn, 'staleness_owner')
                else None
            )
            if row and (binding is None or row != (binding.library_uuid, binding.profile_hash)):
                raise ModelBindingMismatch(
                    f'Staleness ownership mismatch: library {main}, sidecar {sidecar}. Use a fresh database pair.'
                )
            if (
                row is None
                and not legacy_hosted
                and (settings.local or binding is not None or for_write)
                and populated(conn)
            ):
                raise ModelBindingMismatch(
                    f'Nonempty unbound staleness database {sidecar} cannot belong to {main}; use a fresh pair'
                )
    return binding


def initialize_pair(
    main: Path, sidecar: Path, settings: ModelSettings, session: WriteSession
) -> ModelBinding | None:
    """Bind main first, then claim sidecar; a crash retains the original UUID."""
    session.require(main)
    settings.validate(execution=True)
    binding = inspect_pair(main, sidecar, settings, for_write=True)
    main.parent.mkdir(parents=True, exist_ok=True)
    if binding is None:
        if not main.exists():
            os.close(os.open(main, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        with sqlite3.connect(main) as conn:
            conn.execute('BEGIN IMMEDIATE')
            if populated(conn):
                if settings.local:
                    raise ModelBindingMismatch(f'Populated unbound library {main}')
                return None  # Explicitly unverified legacy hosted state.
            conn.execute("""CREATE TABLE IF NOT EXISTS library_model_binding (
                singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
                schema_version INTEGER NOT NULL, library_uuid TEXT NOT NULL,
                generation_identity_json TEXT NOT NULL, embedding_identity_json TEXT NOT NULL,
                profile_hash TEXT NOT NULL, created_at TEXT NOT NULL)""")  # noqa: Q001 - repository quote rule conflicts with the formatter.
            binding = ModelBinding(
                str(uuid.uuid4()),
                settings.profile_hash,
                canonical_json(settings.chat.identity),
                canonical_json(settings.embeddings.identity),
            )
            conn.execute(
                "INSERT INTO library_model_binding VALUES (1,?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                (
                    binding.schema_version,
                    binding.library_uuid,
                    binding.generation_identity_json,
                    binding.embedding_identity_json,
                    binding.profile_hash,
                ),
            )
            conn.execute("""CREATE TABLE IF NOT EXISTS library_state (
                singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1),
                corpus_revision INTEGER NOT NULL DEFAULT 0,
                graph_fingerprint TEXT, themes_fingerprint TEXT, crossrefs_fingerprint TEXT)""")  # noqa: Q001 - repository quote rule conflicts with the formatter.
            conn.execute('INSERT OR IGNORE INTO library_state(singleton_id) VALUES (1)')
        # Binding is deliberately committed before sidecar ownership. If the
        # claim fails or the process dies, retry with this UUID, never a new one.
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    if not sidecar.exists():
        with suppress(FileExistsError):  # Another library may race to claim the sidecar.
            os.close(os.open(sidecar, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    with sqlite3.connect(sidecar, timeout=1.0) as conn:
        conn.execute('BEGIN IMMEDIATE')
        if not table_exists(conn, 'staleness_owner') and populated(conn):
            raise ModelBindingMismatch(f'Nonempty sidecar {sidecar} cannot be claimed by {main}')
        conn.execute("""CREATE TABLE IF NOT EXISTS staleness_owner (
            singleton_id INTEGER PRIMARY KEY CHECK(singleton_id=1), library_uuid TEXT NOT NULL,
            profile_hash TEXT NOT NULL, schema_version INTEGER NOT NULL)""")  # noqa: Q001 - repository quote rule conflicts with the formatter.
        conn.execute(
            'INSERT OR IGNORE INTO staleness_owner VALUES (1,?,?,?)',
            (binding.library_uuid, binding.profile_hash, IDENTITY_VERSION),
        )
        owner = conn.execute(
            'SELECT library_uuid, profile_hash FROM staleness_owner WHERE singleton_id=1'
        ).fetchone()
        if owner != (binding.library_uuid, binding.profile_hash):
            raise ModelBindingMismatch(
                f'Staleness ownership mismatch: library {main}, sidecar {sidecar}'
            )
    return binding


def corpus_revision(path: Path) -> int:
    with readonly(path) as conn:
        if not table_exists(conn, 'library_state'):
            return 0
        return conn.execute(
            'SELECT corpus_revision FROM library_state WHERE singleton_id=1'
        ).fetchone()[0]
