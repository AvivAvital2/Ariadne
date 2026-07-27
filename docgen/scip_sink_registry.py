"""SCIP sink registry (Phase 2r).

A *sink* is a call site whose first (or N-th) argument carries a value
the agent wants to know about — an HTTP URL today, a subprocess command
path tomorrow. The registry is the declarative catalog: each
``SinkSpec`` is one entry, each entry is data not code.

Why a registry instead of per-extractor constants:

- New language support = adding entries, not new modules.
- Cross-language consumers (Phase 9 ``trace_flow``) can iterate every
  sink uniformly without per-language case analysis.
- Phase 2s resolution traversal (when it ships) walks every sink site
  the registry declares; extractors become thin facades that pick one
  ``kind`` and write it to the appropriate table.

Match semantics: **suffix-only**. Real SCIP indexers emit canonical_ids
like ``scip-python python pypi-requests 2.31.0
requests/api/get.``. The leading project/version preamble is variable
and ignored; only the trailing descriptor matters. Same convention as
Phase 8a route extractors — keeps the mental model uniform.

Initial scope (Phase 2r): Python HTTP client primitives — ``requests``,
``httpx``, ``aiohttp``, ``urllib.request.urlopen``. Phase 8b.2 (JS/TS)
and 8b.3 (JVM) extend this same registry with their entries; the
matcher and filter helpers don't change. Phase 2t will add subprocess
entries the same way.
"""
from __future__ import annotations

from typing import Literal

from attrs import frozen


SinkKind = Literal['http_client', 'process_invocation']


@frozen
class SinkSpec:
    """One sink primitive declared in the registry.

    ``symbol_suffixes`` is a tuple because indexers occasionally emit
    slightly different canonical_id shapes for the same primitive
    (e.g. ``requests/api/get.`` vs ``requests/get.``). A spec matches
    if ANY of its suffixes is the trailing part of the symbol.
    """
    name: str
    symbol_suffixes: tuple[str, ...]
    kind: SinkKind
    language: str  # SCIP language name per LANGUAGES registry
    arg_index: int = 0
    http_method: str | None = None


@frozen
class SinkRegistry:
    sinks: tuple[SinkSpec, ...] = ()

    def for_language(self, language: str) -> tuple[SinkSpec, ...]:
        return tuple(s for s in self.sinks if s.language == language)

    def matching_symbol(
        self, symbol: str, *, language: str | None = None,
    ) -> SinkSpec | None:
        """Return the first ``SinkSpec`` whose ``symbol_suffixes``
        contains a suffix matching ``symbol``. ``language`` filters
        candidates before matching (so ``shared/fetch.`` can resolve
        to a Python or TypeScript spec depending on the indexer that
        emitted the symbol).
        """
        for spec in self.sinks:
            if language is not None and spec.language != language:
                continue
            for suf in spec.symbol_suffixes:
                if symbol.endswith(suf):
                    return spec
        return None


# ---------------------------------------------------------------------------
# Initial Python HTTP client entries
# ---------------------------------------------------------------------------


_STANDARD_VERBS: tuple[tuple[str, str], ...] = (
    ('get', 'GET'),
    ('post', 'POST'),
    ('put', 'PUT'),
    ('delete', 'DELETE'),
    ('patch', 'PATCH'),
    ('head', 'HEAD'),
    ('options', 'OPTIONS'),
)


def _suffix_pair(module_path: str, descriptor: str) -> tuple[str, str]:
    """Generate both with-``.py`` and without-``.py`` suffix variants
    for a Python module-path / descriptor pair. scip-python descriptor
    convention has historically varied across versions and wrappers —
    accepting both is more robust than picking one and missing
    real-world canonical_ids."""
    return (
        f'{module_path}/{descriptor}',
        f'{module_path}.py/{descriptor}',
    )


def _python_pkg_func_suffixes(
    pkg: str, descriptor: str,
) -> tuple[str, str, str]:
    """Suffix variants for a top-level function inside a Python
    *package* (a directory with ``__init__.py``). scip-python emits
    the canonical_id with the ``__init__.py`` segment intact for
    stdlib packages like ``subprocess``, ``os``, ``urllib``. Older /
    flattened / single-file wrappers may drop or alter that segment,
    so accept all three forms."""
    return (
        f'{pkg}/__init__.py/{descriptor}',
        f'{pkg}.py/{descriptor}',
        f'{pkg}/{descriptor}',
    )


def _build_python_http_client_sinks() -> tuple[SinkSpec, ...]:
    out: list[SinkSpec] = []

    for verb, method in _STANDARD_VERBS:
        # ``requests.get(url)`` — top-level dispatching API in
        # ``requests/api.py``. Wrapper packages occasionally re-export
        # via ``requests/<verb>``; cover that too.
        out.append(SinkSpec(
            name=f'requests.{verb}',
            symbol_suffixes=(
                *_suffix_pair('requests/api', f'{verb}.'),
                f'requests/{verb}.',
            ),
            kind='http_client',
            language='python',
            http_method=method,
        ))
        # ``requests.Session().get(url)`` — class methods carry ``#``
        # in scip-python descriptors. Class is in ``requests/sessions``.
        out.append(SinkSpec(
            name=f'requests.Session.{verb}',
            symbol_suffixes=_suffix_pair(
                'requests/sessions', f'Session#{verb}.',
            ),
            kind='http_client',
            language='python',
            http_method=method,
        ))
        # ``httpx.get(url)`` — module-level routes via ``_api`` in
        # modern httpx; older variants flatten to ``httpx/<verb>.``.
        out.append(SinkSpec(
            name=f'httpx.{verb}',
            symbol_suffixes=(
                *_suffix_pair('httpx/_api', f'{verb}.'),
                f'httpx/{verb}.',
            ),
            kind='http_client',
            language='python',
            http_method=method,
        ))
        # ``httpx.Client().get(url)`` and async variant.
        for class_name in ('Client', 'AsyncClient'):
            out.append(SinkSpec(
                name=f'httpx.{class_name}.{verb}',
                symbol_suffixes=_suffix_pair(
                    'httpx/_client', f'{class_name}#{verb}.',
                ),
                kind='http_client',
                language='python',
                http_method=method,
            ))
        # ``aiohttp.ClientSession().get(url)``.
        out.append(SinkSpec(
            name=f'aiohttp.ClientSession.{verb}',
            symbol_suffixes=_suffix_pair(
                'aiohttp/client', f'ClientSession#{verb}.',
            ),
            kind='http_client',
            language='python',
            http_method=method,
        ))

    # ``urllib.request.urlopen(url)`` — Python stdlib. Default to GET;
    # ``data=`` kwarg makes it POST, but resolving that's Phase 2s
    # territory.
    out.append(SinkSpec(
        name='urllib.request.urlopen',
        symbol_suffixes=_suffix_pair(
            'urllib/request', 'urlopen.',
        ),
        kind='http_client',
        language='python',
        http_method='GET',
    ))

    return tuple(out)


PYTHON_HTTP_CLIENT_SINKS: tuple[SinkSpec, ...] = (
    _build_python_http_client_sinks()
)


# ---------------------------------------------------------------------------
# Initial JavaScript / TypeScript HTTP client entries (Phase 8b.2)
# ---------------------------------------------------------------------------


def _build_javascript_http_client_sinks() -> tuple[SinkSpec, ...]:
    """JS/TS HTTP client primitives. ``language='typescript'`` covers
    both ``.js``/``.jsx`` and ``.ts``/``.tsx`` per the SCIP
    ``LANGUAGES`` registry — scip-typescript indexes both.

    scip-typescript canonical_id shape:
    ``scip-typescript npm <package> <version> <descriptor>``. The
    descriptor doesn't include the package name as a path prefix
    (different from scip-python), so suffixes here start at the file
    path *within* the package.
    """
    out: list[SinkSpec] = []

    # ``fetch(url, options?)`` — global from TypeScript's lib.dom.d.ts
    # (and Node 18+ globals via the same mechanism). Default method
    # is GET; the ``options.method`` override is Phase 2s territory.
    out.append(SinkSpec(
        name='fetch',
        symbol_suffixes=(
            'lib.dom.d.ts/fetch.',
            'lib.dom.d.ts/fetch().',
        ),
        kind='http_client',
        language='typescript',
        http_method='GET',
    ))

    # ``node-fetch`` — older Node HTTP package. Its types live at
    # ``<pkg>/lib/index.d.ts`` typically.
    out.append(SinkSpec(
        name='node-fetch',
        symbol_suffixes=(
            'node-fetch/lib/index.d.ts/fetch.',
            'lib/index.d.ts/fetch.',
        ),
        kind='http_client',
        language='typescript',
        http_method='GET',
    ))

    # axios verb methods. Modern axios types expose verbs on
    # ``AxiosInstance``; the default export is an AxiosInstance, and
    # ``axios.create()`` returns one too. Older types had ``Axios``
    # / ``AxiosStatic`` shapes.
    for verb, method in _STANDARD_VERBS:
        out.append(SinkSpec(
            name=f'axios.{verb}',
            symbol_suffixes=(
                f'index.d.ts/AxiosInstance#{verb}().',
                f'index.d.ts/AxiosInstance#{verb}.',
                f'index.d.ts/AxiosStatic#{verb}().',
                f'index.d.ts/Axios#{verb}.',
            ),
            kind='http_client',
            language='typescript',
            http_method=method,
        ))

    # got verb methods.
    for verb, method in _STANDARD_VERBS:
        out.append(SinkSpec(
            name=f'got.{verb}',
            symbol_suffixes=(
                f'dist/source/index.d.ts/Got#{verb}.',
                f'dist/source/index.d.ts/Got#{verb}().',
                f'index.d.ts/Got#{verb}.',
            ),
            kind='http_client',
            language='typescript',
            http_method=method,
        ))

    return tuple(out)


JAVASCRIPT_HTTP_CLIENT_SINKS: tuple[SinkSpec, ...] = (
    _build_javascript_http_client_sinks()
)


# ---------------------------------------------------------------------------
# Initial JVM HTTP client entries (Phase 8b.3)
# ---------------------------------------------------------------------------


def _build_jvm_http_client_sinks() -> tuple[SinkSpec, ...]:
    """JVM HTTP client primitives. ``language='jvm'`` covers Scala +
    Java + Kotlin per the SCIP ``LANGUAGES`` registry — scip-java is
    the single indexer for all three.

    v1 ships play-ws ``WSClient.url()``: the simplest fluent-builder
    case. Its ``http_method`` is ``None`` because the actual verb
    (GET/POST/...) comes from a chained call (``.get()`` /
    ``.post()`` / ...) that we don't track at the URL-capture site.
    Phase 2s with chained-call walking will fill that gap.

    sttp (``basicRequest.get(uri"...")``) and Akka HTTP client
    (``Http().singleRequest(HttpRequest(uri = ...))``) are deferred —
    they need either interpolation-aware parsing (sttp's ``uri""``
    StringContext) or named-arg navigation through nested calls
    (Akka HttpRequest), both Phase 2s territory.
    """
    return (
        SinkSpec(
            name='play.WSClient.url',
            symbol_suffixes=(
                'play/api/libs/ws/WSClient#url().',
            ),
            kind='http_client',
            language='jvm',
            http_method=None,
        ),
    )


SCALA_HTTP_CLIENT_SINKS: tuple[SinkSpec, ...] = (
    _build_jvm_http_client_sinks()
)


# ---------------------------------------------------------------------------
# Go HTTP client entries (net/http stdlib)
# ---------------------------------------------------------------------------


def _build_go_http_client_sinks() -> tuple[SinkSpec, ...]:
    """Go ``net/http`` client primitives. ``language='go'`` per the
    LANGUAGES registry (scip-go). The Go client extractor reads the URL as
    a direct string-literal argument; ``arg_index`` locates it.

    Package-level helpers (``http.Get`` etc.) take the URL first; the
    ``*http.Client`` methods mirror them (no ``PostForm`` on Client).
    ``NewRequest`` / ``NewRequestWithContext`` carry an explicit method arg
    we don't resolve yet, so ``http_method`` is None and ``arg_index``
    points at the URL (1 and 2). Suffixes follow scip-go descriptor
    conventions for the stdlib ``net/http`` package.
    """
    out: list[SinkSpec] = []
    for fn, method in (
        ('Get', 'GET'), ('Post', 'POST'), ('Head', 'HEAD'),
        ('PostForm', 'POST'),
    ):
        out.append(SinkSpec(
            name=f'net/http.{fn}',
            symbol_suffixes=(f'net/http/{fn}().', f'net/http/{fn}.'),
            kind='http_client', language='go', arg_index=0,
            http_method=method,
        ))
        if fn in ('Get', 'Post', 'Head'):
            out.append(SinkSpec(
                name=f'net/http.Client.{fn}',
                symbol_suffixes=(
                    f'net/http/Client#{fn}().', f'net/http/Client#{fn}.',
                ),
                kind='http_client', language='go', arg_index=0,
                http_method=method,
            ))
    out.append(SinkSpec(
        name='net/http.NewRequest',
        symbol_suffixes=('net/http/NewRequest().', 'net/http/NewRequest.'),
        kind='http_client', language='go', arg_index=1, http_method=None,
    ))
    out.append(SinkSpec(
        name='net/http.NewRequestWithContext',
        symbol_suffixes=(
            'net/http/NewRequestWithContext().',
            'net/http/NewRequestWithContext.',
        ),
        kind='http_client', language='go', arg_index=2, http_method=None,
    ))
    return tuple(out)


GO_HTTP_CLIENT_SINKS: tuple[SinkSpec, ...] = _build_go_http_client_sinks()


# ---------------------------------------------------------------------------
# Process invocation sinks (Phase 2t)
# ---------------------------------------------------------------------------


def _build_python_process_sinks() -> tuple[SinkSpec, ...]:
    """Python subprocess primitives. ``subprocess.Popen`` is a class
    constructor — scip-python may emit the symbol on the ``Popen``
    token as either the class form (``Popen#``) or the constructor
    form (``Popen#__init__.``); accept both."""
    return (
        SinkSpec(
            name='subprocess.run',
            symbol_suffixes=_python_pkg_func_suffixes(
                'subprocess', 'run.',
            ),
            kind='process_invocation',
            language='python',
        ),
        SinkSpec(
            name='subprocess.Popen',
            symbol_suffixes=(
                'subprocess/__init__.py/Popen#__init__.',
                'subprocess.py/Popen#__init__.',
                'subprocess/__init__.py/Popen#',
                'subprocess.py/Popen#',
            ),
            kind='process_invocation',
            language='python',
        ),
        SinkSpec(
            name='subprocess.call',
            symbol_suffixes=_python_pkg_func_suffixes(
                'subprocess', 'call.',
            ),
            kind='process_invocation',
            language='python',
        ),
        SinkSpec(
            name='subprocess.check_call',
            symbol_suffixes=_python_pkg_func_suffixes(
                'subprocess', 'check_call.',
            ),
            kind='process_invocation',
            language='python',
        ),
        SinkSpec(
            name='subprocess.check_output',
            symbol_suffixes=_python_pkg_func_suffixes(
                'subprocess', 'check_output.',
            ),
            kind='process_invocation',
            language='python',
        ),
        SinkSpec(
            name='os.system',
            symbol_suffixes=_python_pkg_func_suffixes(
                'os', 'system.',
            ),
            kind='process_invocation',
            language='python',
        ),
    )


PYTHON_PROCESS_SINKS: tuple[SinkSpec, ...] = (
    _build_python_process_sinks()
)


def _build_javascript_process_sinks() -> tuple[SinkSpec, ...]:
    """Node ``child_process`` primitives — typings live in
    ``@types/node/child_process.d.ts``. Both with-and-without the
    package-prefix variants for indexer flexibility."""
    out: list[SinkSpec] = []
    for fn in (
        'spawn', 'exec', 'execFile',
        'spawnSync', 'execSync', 'fork',
    ):
        out.append(SinkSpec(
            name=f'child_process.{fn}',
            symbol_suffixes=(
                f'@types/node/child_process.d.ts/{fn}.',
                f'child_process.d.ts/{fn}.',
            ),
            kind='process_invocation',
            language='typescript',
        ))
    return tuple(out)


JAVASCRIPT_PROCESS_SINKS: tuple[SinkSpec, ...] = (
    _build_javascript_process_sinks()
)


def _build_jvm_process_sinks() -> tuple[SinkSpec, ...]:
    """JVM subprocess primitives. ``Process#apply()`` is the Scala
    sys.process entry. ``Runtime#exec`` and ``ProcessBuilder``
    constructor are Java standard library."""
    return (
        SinkSpec(
            name='scala.sys.process.Process',
            symbol_suffixes=(
                'scala/sys/process/Process#apply().',
                'scala/sys/process/Process.apply().',
            ),
            kind='process_invocation',
            language='jvm',
        ),
        SinkSpec(
            name='java.lang.Runtime.exec',
            symbol_suffixes=(
                'java/lang/Runtime#exec().',
                'java/lang/Runtime#exec.',
            ),
            kind='process_invocation',
            language='jvm',
        ),
        SinkSpec(
            name='java.lang.ProcessBuilder',
            symbol_suffixes=(
                'java/lang/ProcessBuilder#`<init>`().',
                'java/lang/ProcessBuilder#<init>().',
                'java/lang/ProcessBuilder#__init__.',
            ),
            kind='process_invocation',
            language='jvm',
        ),
    )


JVM_PROCESS_SINKS: tuple[SinkSpec, ...] = _build_jvm_process_sinks()


DEFAULT_SINK_REGISTRY = SinkRegistry(
    sinks=(
        PYTHON_HTTP_CLIENT_SINKS
        + JAVASCRIPT_HTTP_CLIENT_SINKS
        + SCALA_HTTP_CLIENT_SINKS
        + GO_HTTP_CLIENT_SINKS
        + PYTHON_PROCESS_SINKS
        + JAVASCRIPT_PROCESS_SINKS
        + JVM_PROCESS_SINKS
    ),
)


__all__ = [
    'SinkSpec',
    'SinkRegistry',
    'SinkKind',
    'PYTHON_HTTP_CLIENT_SINKS',
    'JAVASCRIPT_HTTP_CLIENT_SINKS',
    'SCALA_HTTP_CLIENT_SINKS',
    'GO_HTTP_CLIENT_SINKS',
    'PYTHON_PROCESS_SINKS',
    'JAVASCRIPT_PROCESS_SINKS',
    'JVM_PROCESS_SINKS',
    'DEFAULT_SINK_REGISTRY',
]
