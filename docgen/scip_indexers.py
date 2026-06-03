"""Per-language SCIP indexer adapters (Phase 2l).

Each adapter shells out to its respective ``scip-X`` tool. Adapters
duck-type a single ``run(*, cwd, output, env_hints) -> IndexerResult``
method, so ``cmd_index`` (in ``cli_core``) can dispatch over a
``dict[kind, adapter]`` registry.

The subprocess invocation is dependency-injected via the ``runner``
constructor parameter so tests can substitute fakes without touching
real binaries.

Currently shipped adapters: ``PythonIndexerAdapter`` (scip-python),
``TypescriptIndexerAdapter`` (scip-typescript, with built-in Vue SFC
extraction for ``.vue`` files), and ``JavaIndexerAdapter`` (scip-java).
The registry that dispatches them lives in
``cli_core._build_indexer_registry``.
"""
from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from attrs import frozen

from cli.core import IndexerResult


# ---------------------------------------------------------------------------
# Live progress events from indexer subprocesses
# ---------------------------------------------------------------------------


@frozen
class IndexerProgress:
    """One progress event from an indexer's stdout/stderr stream.

    Adapters parse their tool's free-form output (scip-python's
    timestamped log lines, etc.) and emit these structured events so
    the CLI can drive a Rich ``Progress`` widget without each call
    site re-implementing the parser.

    Kinds:
    - ``total`` — the indexer has determined how many units it will
      process. ``total`` carries the count.
    - ``tick`` — incremental update. ``current`` and ``total`` carry
      the latest known counts.
    - ``warning`` — non-fatal warning surfaced from the tool (e.g.
      "Python 3.14 unsupported"). Useful to print above the bar.
    - ``message`` — informational line. Usually silenced behind the
      progress bar; surfaced only on failure.
    """
    kind: str  # 'total' | 'tick' | 'warning' | 'message'
    current: int = 0
    total: int = 0
    text: str = ''


# scip-python emits timestamped lines like ``(01:53:18)   134 / 3192``
# for per-file ticks and ``(01:53:07) Total Project Files 3192`` once
# upfront. Match both with anchored regexes so we don't accidentally
# treat embedded numbers (e.g. inside warning text) as progress.
_SCIP_PYTHON_TICK_RE = re.compile(
    r'^\(\d+:\d+:\d+\)\s+(\d+)\s*/\s*(\d+)\s*$',
)
_SCIP_PYTHON_TOTAL_RE = re.compile(
    r'^\(\d+:\d+:\d+\)\s+Total Project Files\s+(\d+)\s*$',
)


_PACKAGE_INFO_PREFIX = (
    'Warning: Could not find package information for:'
)


def _summarize_missing_package_warning(line: str) -> str | None:
    """If ``line`` is pyright's verbose "missing package info" warning
    (a comma-joined list of every package whose metadata pyright can't
    read), collapse it to a one-line summary. Returns the summarized
    text, or ``None`` if the line doesn't match the pattern.

    Surfaces the COUNT (the actionable signal — "your interpreter has
    too many packages pyright can't read") rather than every name.
    """
    if not line.startswith(_PACKAGE_INFO_PREFIX):
        return None
    rest = line[len(_PACKAGE_INFO_PREFIX):].strip()
    packages = [p.strip() for p in rest.split(',') if p.strip()]
    if len(packages) <= 5:
        # Short list — keep as-is. Probably a real, focused warning.
        return None
    sample = ', '.join(packages[:3])
    return (
        f'Warning: pyright could not read metadata for '
        f'{len(packages)} packages (sample: {sample}, ...). '
        'Type info for those imports will be incomplete.'
    )


def _parse_scip_python_line(line: str) -> IndexerProgress | None:
    """Parse one scip-python output line into a structured event.

    Returns ``None`` for blank lines. Unrecognized lines map to the
    generic ``message`` kind so the CLI can choose whether to display
    them. Warning detection is heuristic — scip-python doesn't tag
    them — but covers the common shapes (``Warning:`` prefix,
    ``unsupported`` substring). Pyright's "could not find package
    information for: <giant list>" is collapsed to a one-line summary
    via :func:`_summarize_missing_package_warning`.
    """
    line = line.rstrip()
    if not line:
        return None

    m = _SCIP_PYTHON_TICK_RE.match(line)
    if m:
        return IndexerProgress(
            kind='tick',
            current=int(m.group(1)),
            total=int(m.group(2)),
        )
    m = _SCIP_PYTHON_TOTAL_RE.match(line)
    if m:
        return IndexerProgress(kind='total', total=int(m.group(1)))

    summary = _summarize_missing_package_warning(line)
    if summary is not None:
        return IndexerProgress(kind='warning', text=summary)

    if line.startswith('Warning:') or 'unsupported' in line.lower():
        return IndexerProgress(kind='warning', text=line)

    return IndexerProgress(kind='message', text=line)


def _stream_progress(
    lines: Iterable[str],
    progress_callback: Callable[[IndexerProgress], None],
) -> list[str]:
    """Drain an iterable of subprocess output lines, feeding parsed
    events to ``progress_callback`` and returning the buffered raw
    lines (so callers can surface them on failure).

    Factored out of the Popen invocation so tests can drive it with
    canned input — no real subprocess required to exercise the parser
    + dispatch logic.
    """
    buffered: list[str] = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        buffered.append(line)
        event = _parse_scip_python_line(line)
        if event is not None:
            progress_callback(event)
    return buffered


@frozen
class VueExtractorResult:
    """Output of a Vue extractor run. Mirrors IndexerResult's shape so
    failures bubble up cleanly to the indexer adapter.

    ``output_path`` is where the extractor wrote ``vue-mapping.json`` —
    the adapter propagates it to ``IndexerResult.vue_mapping_path`` so
    ``cmd_index`` can record it on the manifest entry.
    """
    success: bool
    error_message: str = ''
    output_path: str = ''


class _DefaultVueExtractor:
    """Production Vue extractor — invokes the Node script that ships
    with Ariadne (``scripts/scip/extract-vue-scripts.js``).

    Tests substitute a fake extractor via the adapter's
    ``vue_extractor`` parameter; this implementation is only used in
    production runs.
    """

    def __init__(self, *, runner: Callable | None = None) -> None:
        self._runner = runner if runner is not None else subprocess.run

    def extract(
        self, *, cwd: Path, output_path: Path | None = None,
    ) -> VueExtractorResult:
        script_path = (
            Path(__file__).parent.parent
            / 'scripts' / 'scip' / 'extract-vue-scripts.js'
        )
        if not script_path.exists():
            return VueExtractorResult(
                success=False,
                error_message=(
                    f'Vue extractor script not found at {script_path} — '
                    f'expected to ship with Ariadne'
                ),
            )

        # Tell the script where to write vue-mapping.json. Default matches
        # the script's own fallback so the contract stays self-consistent
        # even if a caller doesn't specify a path.
        if output_path is None:
            output_path = cwd / '.ariadne' / 'intermediate' / 'vue-mapping.json'
        env = dict(os.environ)
        env['ARIADNE_VUE_MAPPING_OUTPUT'] = str(output_path)

        try:
            result = self._runner(
                ['node', str(script_path)],
                cwd=cwd,
                capture_output=True,
                env=env,
            )
        except FileNotFoundError:
            return VueExtractorResult(
                success=False,
                error_message=(
                    'node binary not found on PATH — install Node.js to '
                    'enable Vue indexing'
                ),
            )

        if result.returncode != 0:
            stderr = result.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', errors='replace')
            return VueExtractorResult(
                success=False,
                error_message=(
                    stderr.strip() if stderr
                    else 'Vue extractor returned nonzero exit'
                ),
            )

        return VueExtractorResult(success=True, output_path=str(output_path))


_PYTHON_INDEXER_VERSION = 'scip-python/unknown'


def _run_streaming_python_indexer(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    progress_callback: Callable[[IndexerProgress], None],
) -> tuple[int, str]:
    """Spawn scip-python via Popen, parse its stdout/stderr live, and
    feed structured events to ``progress_callback``.

    Returns ``(returncode, error_text)``. ``error_text`` is the
    accumulated raw output, surfaced only when the subprocess exited
    nonzero (so the caller can put it in ``IndexerResult.error_message``).
    On success the buffered output is discarded — the user already
    saw progress live and a transcript would be noise.
    """
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge so a single iterator covers both
        text=True,
        bufsize=1,  # line-buffered
    )
    assert process.stdout is not None  # narrowed by stdout=PIPE
    buffered = _stream_progress(process.stdout, progress_callback)
    process.wait()
    rc = process.returncode
    error_text = '\n'.join(buffered) if rc != 0 else ''
    return rc, error_text


# Transient pyrightconfig.json template. Pyright searches for this
# file at the project root and walks up — there is no scip-python flag
# to point at a config in another location. The transient file is
# written before the subprocess and cleaned up immediately after via
# try/finally, even on subprocess failure or exception.
#
# {include_json} is one of:
#   - ``["."]`` for entry_kind='package' — pyright walks the cwd as a
#     package root (canonical behavior for __init__.py-rooted dirs)
#   - ``["./*.py"]`` for entry_kind='scripts' — pyright treats each
#     .py file as a standalone module (used for orphan directories
#     emitted by Phase 2j.b's script-entry detection)
# Maintenance dirs that no project ever wants in pyright's analysis
# tree. Kept hardcoded — these are not user-config concerns; they're
# directories Ariadne itself manages or that universally aren't source.
# Anything else (``.venv``, ``docs``, ``node_modules``, project-specific
# generated dirs) flows in via the source's ``exclude_dirs`` /
# ``exclude_policy`` from ariadne.yaml — see ``Config.resolve_excluded_dirs``
# and the ``excludes`` parameter on :meth:`PythonIndexerAdapter.run`.
_MAINTENANCE_EXCLUDES: tuple[str, ...] = (
    '**/__pycache__',
    '**/.git',
    '.ariadne',
)


# ``useLibraryCodeForTypes: false`` caps pyright's import-following.
# With the default (``true``) pyright walks into every imported
# library's source to extract types — that's the right call for IDE
# usage but explodes index time when libraries are deep. Library
# symbols still appear in the SCIP index (just without resolved types).
_PYRIGHTCONFIG_TEMPLATE = '''{{
    "pythonPath": "{python_path}",
    "include": {include_json},
    "exclude": {exclude_json},
    "useLibraryCodeForTypes": false
}}
'''


class PythonIndexerAdapter:
    """Drives ``scip-python`` in a subprocess.

    Phase 2n: the adapter no longer requires ``env_hints['python_path']``
    from the user. It calls :func:`docgen.python_resolver.resolve_python_interpreter`
    to find an interpreter, surfacing the chosen path + provenance label
    on the returned :class:`IndexerResult`.

    On unresolved (no venv found and no system python on PATH), returns
    ``IndexerResult(success=False)`` with the resolver's error_message
    intact and does NOT invoke scip-python.

    On missing binary (FileNotFoundError from the runner), same hard-
    fail pattern with an install hint.
    """

    def __init__(
        self,
        *,
        runner: Callable | None = None,
        resolver_which: Callable[[str], str | None] | None = None,
        materializer: Callable | None = None,
        skip_materialize: bool = False,
    ) -> None:
        self._runner = runner if runner is not None else subprocess.run
        # Forwarded to the resolver as its `which` injection point so
        # tests can simulate "no system python" without monkeypatching
        # shutil. None means production default (shutil.which).
        self._resolver_which = resolver_which
        # Phase 2n.b: subprocess-runner-style callable used to drive
        # uv sync / poetry install / pip install when materializing a
        # local venv. Tests inject FakeRunner; production uses the
        # default in materialize_venv.
        self._materializer = materializer
        self._skip_materialize = skip_materialize

    def run(
        self,
        *,
        cwd: Path,
        output: Path,
        env_hints: dict[str, str],
        entry_kind: str = 'package',
        progress_callback: Callable[[IndexerProgress], None] | None = None,
        excludes: tuple[str, ...] = (),
    ) -> IndexerResult:
        from docgen.python_resolver import resolve_python_interpreter

        # Phase 2n: resolve the interpreter via probe order
        # (env_hints → cwd-venv → walk-up → system → unresolved).
        # The legacy "use env_hints['python_path'] or nothing" path
        # is gone; the resolver subsumes it.
        resolution = resolve_python_interpreter(
            cwd,
            env_hints=env_hints,
            which=self._resolver_which,
        )

        # Phase 2n.b: when resolver fell back to system Python AND the
        # project has a recognized deps file, materialize a local venv
        # before scip-python runs. After successful materialization,
        # re-resolve so the new local .venv becomes the picked
        # interpreter. On failure, fall through with the original
        # system resolution — degraded SCIP quality is better than no
        # indexing at all.
        if (
            not self._skip_materialize
            and resolution.source == 'system'
            and cwd.is_dir()
        ):
            from docgen.python_venv_materialize import (
                detect_deps_tool,
                materialize_venv,
            )
            if detect_deps_tool(cwd) is not None:
                mat = materialize_venv(
                    cwd,
                    runner=self._materializer,
                    system_python=(
                        str(resolution.path)
                        if resolution.path is not None
                        else None
                    ),
                )
                if mat.success:
                    # Re-resolve to pick up the freshly-materialized
                    # .venv (priority: cwd-venv).
                    resolution = resolve_python_interpreter(
                        cwd,
                        env_hints=env_hints,
                        which=self._resolver_which,
                    )

        if resolution.source == 'unresolved':
            # Hard-fail: we have nothing to point pyright at. Don't
            # invoke scip-python — surface the resolver's diagnostic
            # so the user can act (install python, create a venv,
            # set env_hints['python_path']).
            return IndexerResult(
                success=False,
                error_message=resolution.error_message,
                resolution_source='unresolved',
            )

        python_path = str(resolution.path)

        cmd = [
            'scip-python', 'index',
            '--project-name', cwd.name or 'unknown',
            '--project-version', '0.1',
            '--output', str(output),
        ]

        env = dict(os.environ)
        # Belt-and-suspenders: also expose via env in case some tooling
        # reads it. Pyright's actual interpreter resolution comes from
        # pyrightconfig.json (handled below).
        env['PYRIGHT_PYTHON_PATH'] = python_path

        # Zero-config: if the project lacks a pyrightconfig.json, write
        # a transient one for the duration of the subprocess. Cleaned
        # up via try/finally — never persists past the run, even on
        # failure / exception. Include pattern dispatches on entry_kind:
        # package dirs use ``["."]``, script dirs use ``["./*.py"]``.
        pyrightconfig_path = cwd / 'pyrightconfig.json'
        wrote_transient_config = False
        if not pyrightconfig_path.exists():
            import json as _json
            include_json = (
                '["./*.py"]' if entry_kind == 'scripts' else '["."]'
            )
            # Effective excludes: maintenance defaults + caller-supplied
            # user excludes. Caller (cli_core.cmd_index) is responsible
            # for resolving the user side from ariadne.yaml's
            # exclude_policy / exclude_dirs / exempt_dirs / exclude
            # patterns — keeping it the single source of truth.
            all_excludes = list(_MAINTENANCE_EXCLUDES) + list(excludes)
            exclude_json = _json.dumps(all_excludes)
            pyrightconfig_path.write_text(
                _PYRIGHTCONFIG_TEMPLATE.format(
                    python_path=python_path,
                    include_json=include_json,
                    exclude_json=exclude_json,
                ),
                encoding='utf-8',
            )
            wrote_transient_config = True

        try:
            try:
                if progress_callback is not None:
                    # Streaming path: parse scip-python's stdout/stderr
                    # line-by-line for progress events. The CLI uses
                    # this to drive a Rich Progress widget — chatty log
                    # output is suppressed behind the bar; warnings are
                    # still routed via the callback (kind='warning').
                    returncode, error_text = _run_streaming_python_indexer(
                        cmd, cwd=cwd, env=env,
                        progress_callback=progress_callback,
                    )
                else:
                    # Non-streaming path: defer to the injected runner
                    # (subprocess.run by default). Output streams live
                    # to the parent terminal; tests inject FakeRunner.
                    result = self._runner(
                        cmd,
                        cwd=cwd,
                        env=env,
                    )
                    returncode = result.returncode
                    stderr = getattr(result, 'stderr', None)
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode('utf-8', errors='replace')
                    error_text = (stderr or '').strip()
            except FileNotFoundError:
                return IndexerResult(
                    success=False,
                    indexer_version='',
                    error_message=(
                        'scip-python binary not found on PATH — install with '
                        '`npm install -g @sourcegraph/scip-python`'
                    ),
                    resolved_interpreter=resolution.path,
                    resolution_source=resolution.source,
                )

            if returncode != 0:
                return IndexerResult(
                    success=False,
                    indexer_version='',
                    error_message=(
                        error_text
                        if error_text
                        else (
                            f'scip-python exited {returncode} — '
                            'see output above'
                        )
                    ),
                    resolved_interpreter=resolution.path,
                    resolution_source=resolution.source,
                )

            return IndexerResult(
                success=True,
                indexer_version=_PYTHON_INDEXER_VERSION,
                resolved_interpreter=resolution.path,
                resolution_source=resolution.source,
            )
        finally:
            if wrote_transient_config:
                pyrightconfig_path.unlink(missing_ok=True)


def _failure_detail(result, fallback: str) -> str:
    """Build a failure message from a subprocess result, surfacing BOTH
    streams.

    scip-typescript (and friends) often write the actual reason — e.g.
    ``error: no files got indexed`` — to **stdout** while leaving stderr
    empty. Reading only stderr yields the useless ``returned nonzero
    exit`` fallback, hiding why. Combine both streams, preferring stderr
    ordering, and fall back only when genuinely empty.
    """
    def _decode(stream) -> str:
        if stream is None:
            return ''
        if isinstance(stream, bytes):
            return stream.decode('utf-8', errors='replace')
        return stream

    parts = [
        _decode(getattr(result, 'stderr', '')).strip(),
        _decode(getattr(result, 'stdout', '')).strip(),
    ]
    detail = '\n'.join(p for p in parts if p)
    return detail or fallback


_TS_INDEXER_VERSION = 'scip-typescript/unknown'


class TypescriptIndexerAdapter:
    """Drives ``scip-typescript`` in a subprocess. Discovery of
    ``tsconfig.json`` and ``node_modules`` is scip-typescript's
    concern; this adapter just runs the binary with ``--output``.

    Vue support: when ``.vue`` files are present anywhere in ``cwd``,
    the Vue extractor runs first to produce ``*.vue.script.{js,ts}``
    companions plus ``vue-mapping.json``. scip-typescript then sees
    the companions; the loader translates positions back to ``.vue``
    via ``vue-mapping.json`` (Phase 2h).
    """

    def __init__(
        self,
        *,
        runner: Callable | None = None,
        vue_extractor=None,
    ) -> None:
        self._runner = runner if runner is not None else subprocess.run
        # None means fall back to the default Node-script extractor at
        # production runtime; tests inject a FakeVueExtractor.
        self._vue_extractor = vue_extractor

    def run(
        self,
        *,
        cwd: Path,
        output: Path,
        env_hints: dict[str, str],
    ) -> IndexerResult:
        # Vue pre-step: if any .vue files exist anywhere under cwd,
        # extract their <script> blocks before scip-typescript runs.
        vue_mapping_path = ''
        has_vue = any(cwd.rglob('*.vue'))
        if has_vue:
            # Derive a per-scope mapping path beside the .scip output:
            # index-<scope>.scip → vue-mapping-<scope>.json. Per-scope so
            # multiple TS entries in one source don't clobber each other.
            mapping_out = output.with_name(
                'vue-mapping-' + output.stem.removeprefix('index-') + '.json'
            )
            extractor = self._vue_extractor or _DefaultVueExtractor()
            extract_result = extractor.extract(
                cwd=cwd, output_path=mapping_out,
            )
            if not extract_result.success:
                return IndexerResult(
                    success=False,
                    indexer_version='',
                    error_message=extract_result.error_message,
                )
            vue_mapping_path = extract_result.output_path

        cmd = [
            'scip-typescript', 'index',
            '--output', str(output),
        ]
        # Zero-config: if the project has no tsconfig.json, ask
        # scip-typescript to infer one from package.json + node_modules.
        # Per design decision #1, we don't author config files in the
        # consumed repo; --infer-tsconfig is scip-typescript's built-in
        # mechanism for this case.
        if not (cwd / 'tsconfig.json').exists():
            cmd.append('--infer-tsconfig')

        try:
            result = self._runner(
                cmd,
                cwd=cwd,
                capture_output=True,
                env=dict(os.environ),
            )
        except FileNotFoundError:
            return IndexerResult(
                success=False,
                indexer_version='',
                error_message=(
                    'scip-typescript binary not found on PATH — install '
                    'with `npm install -g @sourcegraph/scip-typescript`'
                ),
            )

        if result.returncode != 0:
            return IndexerResult(
                success=False,
                indexer_version='',
                error_message=_failure_detail(
                    result, 'scip-typescript returned nonzero exit',
                ),
            )

        return IndexerResult(
            success=True,
            indexer_version=_TS_INDEXER_VERSION,
            vue_mapping_path=vue_mapping_path,
        )


_JAVA_INDEXER_VERSION = 'scip-java/unknown'


class JavaIndexerAdapter:
    """Drives ``scip-java`` in a subprocess. Build-tool detection
    (sbt/Maven/Gradle) is scip-java's concern; the adapter just runs
    the binary in the source root with ``--output``.

    For first run on a project, scip-java compiles via the project's
    build tool — expect minutes-long latency on large repos. Subsequent
    runs benefit from incremental compilation if the build tool
    supports it.
    """

    def __init__(self, *, runner: Callable | None = None) -> None:
        self._runner = runner if runner is not None else subprocess.run

    def run(
        self,
        *,
        cwd: Path,
        output: Path,
        env_hints: dict[str, str],
    ) -> IndexerResult:
        cmd = [
            'scip-java', 'index',
            '--output', str(output),
        ]

        try:
            result = self._runner(
                cmd,
                cwd=cwd,
                capture_output=True,
                env=dict(os.environ),
            )
        except FileNotFoundError:
            return IndexerResult(
                success=False,
                indexer_version='',
                error_message=(
                    'scip-java binary not found on PATH — install via '
                    'Coursier: `cs install scip-java`'
                ),
            )

        if result.returncode != 0:
            return IndexerResult(
                success=False,
                indexer_version='',
                error_message=_failure_detail(
                    result, 'scip-java returned nonzero exit',
                ),
            )

        return IndexerResult(
            success=True,
            indexer_version=_JAVA_INDEXER_VERSION,
        )


__all__ = [
    'JavaIndexerAdapter',
    'PythonIndexerAdapter',
    'TypescriptIndexerAdapter',
    'VueExtractorResult',
]
