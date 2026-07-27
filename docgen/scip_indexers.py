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

from cli.index import IndexerResult


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
    parse: Callable[[str], 'IndexerProgress | None'] | None = None,
) -> list[str]:
    """Drain an iterable of subprocess output lines, feeding parsed
    events to ``progress_callback`` and returning the buffered raw
    lines (so callers can surface them on failure). ``parse`` is the
    per-tool line parser (defaults to scip-python's; scip-java passes
    :func:`_parse_scip_java_line`).

    Factored out of the Popen invocation so tests can drive it with
    canned input — no real subprocess required to exercise the parser
    + dispatch logic.
    """
    parse = parse or _parse_scip_python_line
    buffered: list[str] = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        buffered.append(line)
        event = parse(line)
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


_GO_INDEXER_VERSION = 'scip-go/unknown'


class GoIndexerAdapter:
    """Drives ``scip-go`` in a subprocess. scip-go type-checks the Go module
    rooted at ``cwd`` (a ``go.mod`` directory) via ``go/packages`` and writes a
    ``.scip`` index. Unlike scip-java there is no build tool to detect or
    orchestrate — the Go toolchain compiles directly — so this is the simplest
    adapter: one command, one fast pass per module.

    Requires ``scip-go`` on PATH and a working Go toolchain. Install with
    ``go install github.com/scip-code/scip-go/cmd/scip-go@latest``.
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
        # env_hints is accepted for adapter-contract uniformity; scip-go needs
        # no interpreter/build-tool/JDK hint — just the ambient Go toolchain.
        cmd = ['scip-go', '--output', str(output)]
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
                    'scip-go binary not found on PATH — install with '
                    '`go install github.com/scip-code/scip-go/cmd/'
                    'scip-go@latest` (needs a working Go toolchain)'
                ),
            )

        if result.returncode != 0:
            return IndexerResult(
                success=False,
                indexer_version='',
                error_message=_failure_detail(
                    result, 'scip-go returned nonzero exit',
                ),
            )
        # A zero exit with no .scip is still a failure — feeding a phantom
        # intermediate to the merge aborts it (the spark-java lesson).
        if not output.exists():
            return IndexerResult(
                success=False,
                indexer_version='',
                error_message=_failure_detail(
                    result,
                    f'scip-go exited 0 but produced no index at {output}',
                ),
            )
        return IndexerResult(
            success=True,
            indexer_version=_GO_INDEXER_VERSION,
        )


_JAVA_INDEXER_VERSION = 'scip-java/unknown'

# Maven's reactor position, e.g. ``Building Apache Spark Core 4.0.0  [12/34]``.
_MAVEN_REACTOR_RE = re.compile(r'Building\b.*?\[(\d+)/(\d+)\]')


def _parse_scip_java_line(line: str) -> 'IndexerProgress | None':
    """Parse one scip-java/Maven output line into a structured event.

    scip-java compiles via Maven, whose reactor prints ``Building <module>
    <ver> [N/M]`` per module — the one countable progress signal in an
    otherwise opaque build. That becomes a ``tick``; ``Warning:`` /
    ``unsupported`` lines become ``warning`` (like scip-python); everything
    else is a silent ``message``."""
    line = line.rstrip()
    if not line:
        return None
    m = _MAVEN_REACTOR_RE.search(line)
    if m:
        return IndexerProgress(
            kind='tick', current=int(m.group(1)), total=int(m.group(2)),
        )
    if line.startswith('Warning:') or 'unsupported' in line.lower():
        return IndexerProgress(kind='warning', text=line)
    return IndexerProgress(kind='message', text=line)


# A module's build output dir, e.g. ``…/core/target/scala-2.13/classes``. The
# directory right before ``/target/`` is the module — the one determinate signal
# both Maven and sbt print (sbt has no Maven-style reactor ``[N/M]``).
_MODULE_TARGET_RE = re.compile(r'/([A-Za-z0-9._-]+)/target/')


def _build_module_names(cwd) -> frozenset:
    """Leaf directory names of the corpus's build modules — one per ``pom.xml``
    (excluding the root aggregator and any under ``target/``). This is the valid
    set the module-count bar checks against, so spurious ``<root>/target`` and
    sbt ``project/target`` hits, and leaf-name collisions, can't push it past
    100%. Empty when no modules are found (bar then stays indeterminate)."""
    cwd = Path(cwd)
    names = set()
    for pom in cwd.rglob('pom.xml'):
        if 'target' in pom.parts or pom.parent == cwd:
            continue  # generated pom, or the root aggregator (not a module)
        names.add(pom.parent.name)
    return frozenset(names)


class _JavaBuildProgress:
    """Stateful scip-java line parser driving a DETERMINATE bar by module count.
    Both Maven and sbt print each module's ``<module>/target/`` path as they
    compile it; we count DISTINCT modules from the corpus's known set (``sbt``
    has no Maven ``[N/M]`` reactor). Maven's reactor position is still used
    directly when present. Warnings/messages behave as ``_parse_scip_java_line``.

    Keyed off a fixed module set (not raw hits) so the repo root, sbt's
    ``project/`` meta-build, and duplicate target subpaths never over-count."""

    def __init__(self, modules) -> None:
        self.modules = frozenset(modules)
        self.total = len(self.modules)
        self.seen: set = set()
        self.saw_reactor = False

    def __call__(self, line: str) -> 'IndexerProgress | None':
        line = line.rstrip()
        if not line:
            return None
        m = _MAVEN_REACTOR_RE.search(line)
        if m:
            self.saw_reactor = True
            return IndexerProgress(
                kind='tick', current=int(m.group(1)), total=int(m.group(2)))
        m = _MODULE_TARGET_RE.search(line)
        if m:
            # Maven's ordered reactor is authoritative once seen; the
            # <module>/target signal only drives the bar for sbt (no reactor),
            # else the two scales (reactor total vs module-set size) interleave.
            module = m.group(1)
            if (not self.saw_reactor and module in self.modules
                    and module not in self.seen):
                self.seen.add(module)
                return IndexerProgress(
                    kind='tick', current=len(self.seen),
                    total=self.total or None, text=module)
            return None  # dup, non-module, or reactor already driving
        if line.startswith('Warning:') or 'unsupported' in line.lower():
            return IndexerProgress(kind='warning', text=line)
        return IndexerProgress(kind='message', text=line)


def _run_streaming_java_indexer(cmd, cwd, env, progress_callback):
    """Spawn scip-java via Popen, parse its (Maven) output live for reactor
    progress, and feed ``tick`` events to ``progress_callback``. Returns
    ``(returncode, raw_output)`` — the raw output is kept so the caller can
    detect the "multiple build tools" error and surface failures."""
    process = subprocess.Popen(
        cmd, cwd=str(cwd), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    # Determinate bar by module count — tool-agnostic (Maven's [N/M] reactor OR
    # each module's <module>/target/ path), so sbt gets real progress too.
    buffered = _stream_progress(
        process.stdout, progress_callback,
        parse=_JavaBuildProgress(_build_module_names(cwd)),
    )
    process.wait()
    return process.returncode, '\n'.join(buffered)


# Preference order when scip-java can't pick among several build tools. sbt is
# FIRST because scip-java indexes SCALA through the sbt path (it drives
# semanticdb-scalac via its sbt plugin); its Maven support is effectively
# Java-only, so on a Scala repo that ships both (e.g. spark: a Maven pom.xml AND
# an sbt build) Maven compiles but emits no Scala SemanticDB → "produced no
# index". sbt-present ⇒ Scala project ⇒ sbt is the indexable build.
_BUILD_TOOL_PREFERENCE = ('sbt', 'maven', 'gradle', 'bazel', 'mill')


def _build_tool_from_error(text: str) -> str | None:
    """If ``text`` is scip-java's "Multiple build tools detected: A, B" error
    (spark ships both a Maven ``pom.xml`` and an sbt build), return the
    preferred build tool among the ones it named — so a retry can force
    ``--build-tool``. Returns None when the text is any other error."""
    m = re.search(r'Multiple build tools detected:\s*([^.\n]+)', text or '')
    if not m:
        return None
    named = {n.strip().lower()
             for n in re.split(r'[,\s]+', m.group(1)) if n.strip()}
    return next((t for t in _BUILD_TOOL_PREFERENCE if t in named), None)


def _normalize_java_major(value: str) -> int:
    """``'17'`` → 17, ``'1.8'`` → 8, ``'21.0.1'`` → 21 (JDK major version)."""
    parts = str(value).strip().split('.')
    if parts[0] == '1' and len(parts) > 1:
        return int(parts[1])
    return int(parts[0])


def _required_java_version(cwd) -> 'int | None':
    """Deduce the JDK major version the corpus's build DECLARES, so scip-java
    compiles with a matching JDK (Spark 4.0's pom.xml declares Java 17 — on a
    newer JDK the SemanticDB often isn't emitted → "produced no index"). Reads,
    in priority order: ``.java-version`` / ``.tool-versions`` (jenv/asdf), then
    Maven ``pom.xml`` compiler properties. None when nothing declares one."""
    cwd = Path(cwd)
    jvf = cwd / '.java-version'
    if jvf.is_file():
        m = re.search(r'(\d[\d.]*)', jvf.read_text(errors='replace'))
        if m:
            return _normalize_java_major(m.group(1))
    tvf = cwd / '.tool-versions'
    if tvf.is_file():
        m = re.search(r'^\s*java\s+\S*?(\d[\d.]*)',
                      tvf.read_text(errors='replace'), re.MULTILINE)
        if m:
            return _normalize_java_major(m.group(1))
    pom = cwd / 'pom.xml'
    if pom.is_file():
        text = pom.read_text(errors='replace')
        for tag in ('maven.compiler.release', 'java.version',
                    'maven.compiler.target', 'maven.compiler.source'):
            m = re.search(
                rf'<{re.escape(tag)}>\s*([\d.]+)\s*</{re.escape(tag)}>', text)
            if m:
                return _normalize_java_major(m.group(1))
    return None


# Where JDKs live on macOS (system JVMs, Homebrew, asdf, sdkman). Each home has
# a ``release`` file with ``JAVA_VERSION="..."`` — the authoritative version.
_JDK_HOME_GLOBS = (
    '/Library/Java/JavaVirtualMachines/*/Contents/Home',
    '/opt/homebrew/opt/openjdk@*/libexec/openjdk.jdk/Contents/Home',
    '/usr/local/opt/openjdk@*/libexec/openjdk.jdk/Contents/Home',
    '~/.asdf/installs/java/*',
    '~/.sdkman/candidates/java/*',
)


def _jdk_major(home) -> 'int | None':
    """Major version of the JDK at ``home`` from its ``release`` file, or None."""
    rel = Path(home) / 'release'
    if not rel.is_file():
        return None
    m = re.search(r'JAVA_VERSION="?([\d._]+)', rel.read_text(errors='replace'))
    if not m:
        return None
    try:
        return _normalize_java_major(m.group(1).split('_')[0])
    except (ValueError, IndexError):
        return None


def _locate_jdk(version: int, *, globs=_JDK_HOME_GLOBS) -> 'str | None':
    """Find an installed JDK of major ``version`` by reading each candidate's
    ``release`` file — ``/usr/libexec/java_home`` is unreliable for non-Oracle
    JDKs (it reports "no Java" even when JDKs are present). None if none match."""
    import glob as _glob
    for pattern in globs:
        for path in sorted(_glob.glob(os.path.expanduser(pattern))):
            home = Path(path)
            if (home / 'bin' / 'java').exists() and _jdk_major(home) == version:
                return str(home)
    return None


def _resolve_java_home(cwd, *, locate=None) -> 'str | None':
    """The JDK home scip-java should compile ``cwd`` with — the version deduced
    from the build config, located on the system. None when undeducible or no
    matching JDK is installed (scip-java then falls back to the ambient JDK)."""
    version = _required_java_version(cwd)
    if version is None:
        return None
    return (locate or _locate_jdk)(version)


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
        progress_callback: 'Callable[[IndexerProgress], None] | None' = None,
    ) -> IndexerResult:
        base = ['scip-java', 'index', '--output', str(output)]
        hint = env_hints.get('build_tool')
        env = dict(os.environ)
        # Compile with the JDK the corpus DECLARES (Spark 4.0 → Java 17). On a
        # mismatched JDK scip-java's SemanticDB often isn't emitted, which shows
        # up as "produced no index". env_hints['java_home'] overrides.
        java_home = env_hints.get('java_home') or _resolve_java_home(cwd)
        if java_home:
            env['JAVA_HOME'] = str(java_home)
            env['PATH'] = f"{java_home}/bin:{env.get('PATH', '')}"

        def _invoke(extra) -> tuple[int, str]:
            """Run scip-java once → (returncode, raw_output). Streams Maven
            reactor progress live when a callback is given (production);
            otherwise uses the injected runner (tests / capture)."""
            if progress_callback is not None:
                return _run_streaming_java_indexer(
                    base + extra, cwd, env, progress_callback,
                )
            result = self._runner(
                base + extra, cwd=cwd, capture_output=True, env=env,
            )
            return result.returncode, _failure_detail(result, '')

        try:
            rc, err = _invoke([f'--build-tool={hint}'] if hint else [])
            # scip-java can't auto-pick when a repo has several build tools
            # (spark ships both Maven and sbt): it exits nonzero listing them.
            # Parse that list and retry with the preferred one. An explicit
            # env_hints['build_tool'] was already forced above, so skip.
            if rc != 0 and not hint:
                tool = _build_tool_from_error(err)
                if tool:
                    rc, err = _invoke([f'--build-tool={tool}'])
        except FileNotFoundError:
            return IndexerResult(
                success=False,
                indexer_version='',
                error_message=(
                    'scip-java binary not found on PATH — install via '
                    'Coursier: `cs install --contrib scip-java`'
                ),
            )

        if rc != 0:
            return IndexerResult(
                success=False,
                indexer_version='',
                error_message=err or 'scip-java returned nonzero exit',
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
