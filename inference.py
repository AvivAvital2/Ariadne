"""Operation-scoped inference resources and library write ownership."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config
    from docgen.llm.base import LLMProvider
    from embedding import EmbeddingService
    from library import Library

import asyncio
import inspect
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from library.locking import LibraryBusy, WriteSession, writer_session
from library.model_binding import (
    ModelBindingMismatch,
    corpus_revision,
    initialize_pair,
    inspect_pair,
)
from model_config import ModelSettings, resolve_paths

_current: ContextVar[InferenceContext | None] = ContextVar(
    'ariadne_inference_context', default=None
)


def current_context() -> InferenceContext | None:
    return _current.get()


def local_context(owner: object) -> InferenceContext | None:
    """Return a real local context, including through library/writer handles."""
    candidate = getattr(owner, 'inference_context', None)
    return candidate if isinstance(candidate, InferenceContext) and candidate.local else None


def raise_inference_error(error: Exception) -> None:
    """Keep local readiness/contract failures out of optional synthesis catches."""
    from docgen.llm.local import LocalInferenceError
    from library.prepared import ObsoleteInput
    from model_config import ModelConfigurationError

    if isinstance(
        error,
        (
            LocalInferenceError,
            ModelConfigurationError,
            ModelBindingMismatch,
            LibraryBusy,
            ObsoleteInput,
        ),
    ):
        raise error


def service_operation(
    method: Callable[..., Awaitable[Any]], *, write: bool = False
) -> Callable[..., Awaitable[Any]]:
    """Activate the service context for one request, with independent ownership."""
    signature = inspect.signature(method)

    @wraps(method)
    async def invoke(self: Any, *args: Any, **kwargs: Any) -> Any:
        context = self.inference_context
        if context is None:
            return await method(self, *args, **kwargs)
        arguments = signature.bind(self, *args, **kwargs).arguments
        is_write = write or (
            method.__name__ == 'ask' and arguments.get('role', 'developer') != 'developer'
        )
        with context.activate():
            with context.write_session(reuse=context.session) if is_write else nullcontext():
                if is_write:
                    context.open_library(operation='write').close()
                before = context.cache_identity
                result = await method(self, *args, **kwargs)
                after = context.cache_identity
                if not is_write and before != after:
                    self._query_cache.clear()
                    raise ModelBindingMismatch(
                        'Corpus changed during semantic retrieval; retry the request'
                    )
                return result

    return invoke


class EndpointLimit:
    """One shared slot for synchronous and asynchronous requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def __aenter__(self) -> EndpointLimit:
        # Do not offload a blocking acquire: a cancelled waiter could otherwise
        # acquire later in a thread and permanently leak the slot.
        while not self._lock.acquire(blocking=False):  # noqa: ASYNC110 - also coordinates sync callers.
            await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *_: object) -> None:
        self._lock.release()

    def __enter__(self) -> EndpointLimit:
        self._lock.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self._lock.release()


class InferenceContext:
    """One immutable configuration and client owner; locks belong to operations."""

    def __init__(self, settings: ModelSettings, db_path: Path, staleness_db_path: Path) -> None:
        self.settings = settings.validate()
        self.db_path = db_path.expanduser().resolve()
        self.staleness_db_path = staleness_db_path.expanduser().resolve()
        if self.db_path == self.staleness_db_path:
            raise ModelBindingMismatch('The library and staleness database must be separate')
        self.binding = inspect_pair(self.db_path, self.staleness_db_path, settings)
        self._session: ContextVar[WriteSession | None] = ContextVar(
            'ariadne_write_session', default=None
        )
        self._attempts: ContextVar[int | None] = ContextVar(
            'ariadne_transport_attempts', default=None
        )
        self._chat = None
        self._embeddings = None
        self.chat_limit = EndpointLimit()
        self.embedding_limit = EndpointLimit()
        self._libraries = []

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        db: Path | str | None = None,
        staleness_db: Path | str | None = None,
        **overrides: Any,
    ) -> InferenceContext:
        settings = config.model_settings(**overrides)
        main, sidecar = resolve_paths(
            config, db=db, staleness_db=staleness_db, local=settings.local
        )
        return cls(settings, main, sidecar)

    @property
    def local(self) -> bool:
        return self.settings.local

    @contextmanager
    def activate(self) -> Iterator[InferenceContext]:
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    @property
    def session(self) -> WriteSession | None:
        session = self._session.get()
        return session if session is not None and session.active else None

    def require_writer(self) -> WriteSession:
        session = self.session
        if session is None:
            raise LibraryBusy(f'A corpus write session is required for {self.db_path}')
        session.require(self.db_path)
        return session

    @property
    def transport_attempts(self) -> int | None:
        return self._attempts.get()

    @contextmanager
    def single_attempt(self) -> Iterator[None]:
        """Spool stage budgets own retries, including every embedding sub-batch."""
        token = self._attempts.set(1)
        try:
            yield
        finally:
            self._attempts.reset(token)

    @contextmanager
    def write_session(self, *, reuse: WriteSession | None = None) -> Iterator[WriteSession]:
        with writer_session(self.db_path, reuse=reuse) as session:
            token = self._session.set(session)
            try:
                self.binding = initialize_pair(
                    self.db_path, self.staleness_db_path, self.settings, session
                )
                with self.activate():
                    yield session
            finally:
                self._session.reset(token)

    @contextmanager
    def nested_write(self) -> Iterator[WriteSession]:
        """Explicit nesting for helpers already inside a corpus operation."""
        with self.write_session(reuse=self.session) as session:
            yield session

    def validate(self, *, credentials: bool = False) -> None:
        self.settings.validate(execution=True)
        binding = inspect_pair(self.db_path, self.staleness_db_path, self.settings)
        if self.local and binding is None:
            raise ModelBindingMismatch(
                f'Local library {self.db_path} is not initialized; run a corpus write first'
            )
        if self.binding is not None and (
            binding is None or binding.library_uuid != self.binding.library_uuid
        ):
            raise ModelBindingMismatch('Library identity changed while open')
        self.binding = binding
        if credentials:
            self.settings.check_credentials()

    def validate_producer(
        self, *, model: str, provider: str, embedding_space_id: str | None = None
    ) -> None:
        self.validate()
        if embedding_space_id is not None:
            if embedding_space_id != self.settings.embeddings.embedding_space_id:
                raise ModelBindingMismatch(
                    'Embedding producer does not match the bound embedding space'
                )
        elif (model, provider) != (self.settings.chat.model, self.settings.chat.provider):
            raise ModelBindingMismatch('Chat producer does not match the bound model')

    @property
    def cache_identity(self) -> tuple[str, str, int]:
        self.validate(credentials=True)
        return (
            self.binding.library_uuid if self.binding else str(self.db_path),
            self.settings.embeddings.embedding_space_id,
            corpus_revision(self.db_path),
        )

    def open_library(self, *, operation: str = 'semantic') -> Library:
        from library import Library

        if operation in {'inspect', 'plan'}:
            if not self.db_path.exists():
                raise FileNotFoundError(f'Library does not exist: {self.db_path}')
            library = Library(
                self.db_path, inference_context=self, initialize=False, read_only=True
            )
        elif operation == 'write':
            self.require_writer()
            self.validate()
            library = Library(self.db_path, inference_context=self, initialize=True)
        else:
            self.validate(credentials=True)
            library = Library(self.db_path, inference_context=self, initialize=False)
        self._libraries.append(library)
        return library

    @property
    def chat(self) -> LLMProvider:
        if self._chat is None:
            self.validate(credentials=True)
            from docgen.llm.factory import make_llm_provider

            settings = self.settings.chat
            self._chat = make_llm_provider(
                settings.provider,
                model=settings.model,
                api_key=settings.credential() or '',
                base_url=settings.base_url,
                timeout=settings.timeout_seconds,
                max_retries=self.settings.execution.max_attempts,
                context=self,
            )
        return self._chat

    @property
    def embeddings(self) -> EmbeddingService:
        if self._embeddings is None:
            self.validate(credentials=True)
            from embedding import EmbeddingConfig, EmbeddingService

            self._embeddings = EmbeddingService(
                EmbeddingConfig.from_settings(self.settings.embeddings), context=self
            )
        return self._embeddings

    async def aclose(self) -> None:
        await self.close_clients()
        for library in self._libraries:
            library.close()
        self._libraries.clear()

    async def close_clients(self) -> None:
        if self._chat is not None:
            await self._chat.aclose()
            self._chat = None
        if self._embeddings is not None:
            await self._embeddings.close()
            self._embeddings = None
