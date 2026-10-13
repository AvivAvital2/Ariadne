"""One nonblocking POSIX writer lock per canonical library path."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class LibraryBusy(RuntimeError):  # noqa: N818 - public contract name.
    """Another operation owns this library's writer lock."""


@dataclass(eq=False)
class WriteSession:
    path: Path
    descriptor: int
    active: bool = True

    def require(self, path: Path) -> None:
        if not self.active or self.path != path.resolve():
            raise LibraryBusy('A live write session for this library is required')


@contextmanager
def writer_session(path: Path, *, reuse: WriteSession | None = None) -> Iterator[WriteSession]:
    path = path.expanduser().resolve()
    if reuse is not None:
        reuse.require(path)
        yield reuse
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + '.lock')
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LibraryBusy(f'LibraryBusy: another writer owns {path}') from exc
        session = WriteSession(path, descriptor)
        try:
            yield session
        finally:
            session.active = False
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
