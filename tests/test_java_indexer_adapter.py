"""Contract for ``JavaIndexerAdapter`` — Phase 2l fourth slice.

Mirrors Python/TypeScript: shell out to ``scip-java``, return
``IndexerResult``. Build-tool detection (sbt/Maven/Gradle) is
scip-java's concern; the adapter just runs the binary in cwd.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


class FakeRunner:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b'',
        stderr: bytes = b'',
        scip_bytes: bytes = b'\x08\x01synthetic',
        raise_on_call: Exception | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.scip_bytes = scip_bytes
        self.raise_on_call = raise_on_call
        self.calls: list[dict] = []

    def __call__(self, cmd, *, cwd=None, capture_output=True, env=None,
                 **kwargs):
        self.calls.append({
            'cmd': list(cmd), 'cwd': cwd,
        })
        if self.raise_on_call:
            raise self.raise_on_call
        if self.returncode == 0 and '--output' in cmd:
            out_path = Path(cmd[cmd.index('--output') + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(self.scip_bytes)
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


# ---------------------------------------------------------------------------
# Command shape
# ---------------------------------------------------------------------------


class TestCommandShape:
    def test_invokes_scip_java_index(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert result.success
        cmd = runner.calls[0]['cmd']
        assert cmd[0] == 'scip-java'
        assert 'index' in cmd

    def test_output_flag_uses_target_path(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        adapter = JavaIndexerAdapter(runner=runner)
        out = tmp_path / 'subdir' / 'out.scip'
        adapter.run(cwd=tmp_path, output=out, env_hints={})

        cmd = runner.calls[0]['cmd']
        assert '--output' in cmd
        assert cmd[cmd.index('--output') + 1] == str(out)

    def test_runs_in_specified_cwd(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        adapter = JavaIndexerAdapter(runner=runner)
        adapter.run(
            cwd=tmp_path / 'project_root',
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert runner.calls[0]['cwd'] == tmp_path / 'project_root'


# ---------------------------------------------------------------------------
# Success / failure
# ---------------------------------------------------------------------------


class TestSuccessFailure:
    def test_returns_success_on_zero_returncode(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner(returncode=0)
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is True
        assert (tmp_path / 'out.scip').exists()

    def test_indexer_version_starts_with_scip_java(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.indexer_version.startswith('scip-java')

    def test_nonzero_returncode_reports_failure(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner(
            returncode=1,
            stderr=b'sbt: compilation failed in subproject foo\n',
        )
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is False
        assert (
            'sbt' in result.error_message
            or 'compilation' in result.error_message
        )

    def test_binary_not_found_reports_failure(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner(
            raise_on_call=FileNotFoundError(
                "[Errno 2] No such file or directory: 'scip-java'",
            ),
        )
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is False
        assert 'scip-java' in result.error_message
        # Helpful hint pointing the user to the install path
        assert (
            'install' in result.error_message.lower()
            or 'not found' in result.error_message.lower()
            or 'coursier' in result.error_message.lower()
            or 'path' in result.error_message.lower()
        )


# ---------------------------------------------------------------------------
# Build-tool ambiguity (spark ships both Maven and sbt)
# ---------------------------------------------------------------------------


class TestBuildTool:
    """scip-java can't auto-pick when a repo has several build tools (spark: a
    Maven ``pom.xml`` AND an sbt build). The adapter parses the tools scip-java
    named and retries with the preferred one (maven > gradle > sbt > bazel >
    mill); ``env_hints['build_tool']`` overrides and skips the retry."""

    _AMBIGUOUS = (
        b'Picked up _JAVA_OPTIONS: -Xmx32G\n'
        b'error: Multiple build tools detected: Maven, sbt. To fix this '
        b"problem, use the '--build-tool=BUILD_TOOL_NAME' flag.\n"
    )

    def test_retries_with_preferred_on_ambiguity(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        class TwoStep:
            def __init__(self) -> None:
                self.calls: list[list] = []

            def __call__(self, cmd, *, cwd=None, capture_output=True,
                         env=None, **kw):
                self.calls.append(list(cmd))
                if len(self.calls) == 1:                    # first try: plain
                    return SimpleNamespace(
                        returncode=1, stdout=b'',
                        stderr=TestBuildTool._AMBIGUOUS)
                out = Path(cmd[cmd.index('--output') + 1])  # retry succeeds
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b'\x08\x01x')
                return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        runner = TwoStep()
        result = JavaIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'o.scip', env_hints={})
        assert result.success is True
        assert len(runner.calls) == 2
        assert not any('--build-tool' in a for a in runner.calls[0])
        # sbt is preferred over Maven: scip-java indexes SCALA via the sbt path
        # (semanticdb-scalac); its Maven support is Java-only, so on a Scala repo
        # like spark (Maven + sbt) Maven compiles but emits no Scala SemanticDB.
        assert '--build-tool=sbt' in runner.calls[1]

    def test_env_hint_forces_tool_without_retry(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        JavaIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'o.scip',
            env_hints={'build_tool': 'sbt'})
        assert len(runner.calls) == 1
        assert '--build-tool=sbt' in runner.calls[0]['cmd']

    def test_ordinary_failure_is_not_retried(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner(returncode=1, stderr=b'sbt: compilation failed')
        result = JavaIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'o.scip', env_hints={})
        assert result.success is False
        assert len(runner.calls) == 1


# ---------------------------------------------------------------------------
# Maven reactor progress (scip-java spins up Maven — parse its [N/M] position)
# ---------------------------------------------------------------------------


class TestMavenProgress:
    """scip-java compiles via Maven, which prints a reactor position
    ``Building <module> <ver> [N/M]`` per module. We parse that into ``tick``
    events so the Java bar shows real progress instead of an endless spinner."""

    def test_parses_maven_reactor_position(self) -> None:
        from docgen.scip_indexers import _parse_scip_java_line as p
        ev = p('[INFO] Building Apache Spark Project Core 4.0.0        [12/34]')
        assert ev is not None
        assert ev.kind == 'tick' and ev.current == 12 and ev.total == 34

    def test_plain_build_line_is_message(self) -> None:
        from docgen.scip_indexers import _parse_scip_java_line as p
        ev = p('[INFO] Compiling 450 Scala sources to target/classes')
        assert ev.kind == 'message'

    def test_bracketed_phase_line_is_not_a_tick(self) -> None:
        from docgen.scip_indexers import _parse_scip_java_line as p
        # the phase separator has brackets but no N/M — must not be a tick
        ev = p('[INFO] --------------------------------[ jar ]-------------')
        assert ev.kind != 'tick'

    def test_unsupported_line_is_warning(self) -> None:
        from docgen.scip_indexers import _parse_scip_java_line as p
        assert p('Java version 25 is unsupported by this build').kind == 'warning'

    def test_stream_dispatches_maven_ticks(self) -> None:
        from docgen.scip_indexers import (
            _parse_scip_java_line, _stream_progress,
        )
        seen: list = []
        lines = [
            '[INFO] Building Spark Parent 4.0.0 [1/34]',
            '[INFO] --- some compile noise ---',
            '[INFO] Building Spark Core 4.0.0 [2/34]',
        ]
        _stream_progress(iter(lines), seen.append, parse=_parse_scip_java_line)
        ticks = [(e.current, e.total) for e in seen if e.kind == 'tick']
        assert ticks == [(1, 34), (2, 34)]


# ---------------------------------------------------------------------------
# JDK deduction: scip-java must compile with the JDK the corpus declares
# ---------------------------------------------------------------------------


class TestJavaVersionDeduction:
    """Deduce the required JDK from the corpus's build config (Spark 4.0's
    pom.xml declares Java 17) so scip-java compiles with a matching JDK — a
    mismatched JDK typically emits no SemanticDB → 'produced no index'. An
    env_hints['java_home'] overrides the deduction."""

    def test_version_from_pom_java_version(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _required_java_version
        (tmp_path / 'pom.xml').write_text(
            '<project><properties><java.version>17</java.version>'
            '</properties></project>')
        assert _required_java_version(tmp_path) == 17

    def test_version_from_compiler_release(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _required_java_version
        (tmp_path / 'pom.xml').write_text(
            '<project><properties>'
            '<maven.compiler.release>21</maven.compiler.release>'
            '</properties></project>')
        assert _required_java_version(tmp_path) == 21

    def test_version_normalizes_1_8(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _required_java_version
        (tmp_path / 'pom.xml').write_text(
            '<project><properties>'
            '<maven.compiler.target>1.8</maven.compiler.target>'
            '</properties></project>')
        assert _required_java_version(tmp_path) == 8

    def test_version_from_java_version_file(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _required_java_version
        (tmp_path / '.java-version').write_text('17\n')
        assert _required_java_version(tmp_path) == 17

    def test_version_from_tool_versions(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _required_java_version
        (tmp_path / '.tool-versions').write_text(
            'nodejs 20.1.0\njava temurin-17.0.9+9\n')
        assert _required_java_version(tmp_path) == 17

    def test_no_build_config_returns_none(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _required_java_version
        assert _required_java_version(tmp_path) is None

    def test_resolve_uses_deduced_version(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _resolve_java_home
        (tmp_path / 'pom.xml').write_text(
            '<project><properties><java.version>17</java.version>'
            '</properties></project>')
        seen = {}

        def locate(v):
            seen['v'] = v
            return f'/jdks/{v}'

        assert _resolve_java_home(tmp_path, locate=locate) == '/jdks/17'
        assert seen['v'] == 17

    def test_resolve_none_when_undetected(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _resolve_java_home
        assert _resolve_java_home(tmp_path, locate=lambda v: 1 / 0) is None

    def test_adapter_sets_deduced_java_home(self, tmp_path: Path, monkeypatch) -> None:
        from docgen import scip_indexers
        from docgen.scip_indexers import JavaIndexerAdapter
        (tmp_path / 'pom.xml').write_text(
            '<project><properties><java.version>17</java.version>'
            '</properties></project>')
        monkeypatch.setattr(scip_indexers, '_resolve_java_home',
                            lambda cwd: '/jdks/17')
        captured = {}

        def runner(cmd, *, cwd=None, capture_output=True, env=None):
            captured['env'] = env
            out = Path(cmd[cmd.index('--output') + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b'\x08\x01x')
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        JavaIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'o.scip', env_hints={})
        assert captured['env']['JAVA_HOME'] == '/jdks/17'
        assert captured['env']['PATH'].startswith('/jdks/17/bin')

    def test_env_hint_overrides_deduction(self, tmp_path: Path, monkeypatch) -> None:
        from docgen import scip_indexers
        from docgen.scip_indexers import JavaIndexerAdapter
        monkeypatch.setattr(scip_indexers, '_resolve_java_home',
                            lambda cwd: '/jdks/deduced')

        captured = {}

        def runner(cmd, *, cwd=None, capture_output=True, env=None):
            captured['env'] = env
            out = Path(cmd[cmd.index('--output') + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b'\x08\x01x')
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        JavaIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'o.scip',
            env_hints={'java_home': '/jdks/pinned'})
        assert captured['env']['JAVA_HOME'] == '/jdks/pinned'
        assert captured['env']['PATH'].startswith('/jdks/pinned/bin')


class TestJdkLocator:
    """Locate an installed JDK by reading each candidate's ``release`` file
    (``/usr/libexec/java_home`` is unreliable for non-Oracle JDKs)."""

    def test_locate_matches_by_release_file(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _locate_jdk
        for name, ver in (('openjdk-17.jdk', '17.0.19'),
                          ('corretto-11.jdk', '11.0.21'),
                          ('corretto-8.jdk', '1.8.0_292')):
            home = tmp_path / name / 'Contents' / 'Home'
            (home / 'bin').mkdir(parents=True)
            (home / 'bin' / 'java').write_text('#!/bin/sh\n')
            (home / 'release').write_text(f'JAVA_VERSION="{ver}"\n')
        globs = (str(tmp_path / '*' / 'Contents' / 'Home'),)
        assert _locate_jdk(17, globs=globs).endswith('openjdk-17.jdk/Contents/Home')
        assert _locate_jdk(11, globs=globs).endswith('corretto-11.jdk/Contents/Home')
        assert _locate_jdk(8, globs=globs).endswith('corretto-8.jdk/Contents/Home')
        assert _locate_jdk(21, globs=globs) is None       # not installed


class TestModuleProgress:
    """The determinate Java bar is driven by MODULE COUNT, tool-agnostically:
    both Maven and sbt print each module's ``<module>/target/`` path as they
    compile it. Counting is keyed off the corpus's known module-name set, so
    spurious hits (the repo root, sbt's ``project/``) and leaf-name collisions
    can't push the bar past 100% or make it drift."""

    def test_build_module_names_from_pom_dirs(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import _build_module_names
        for rel in ('core', 'sql/api', 'sql/engine', 'common/net'):
            (tmp_path / rel).mkdir(parents=True)
            (tmp_path / rel / 'pom.xml').write_text('<project/>')
        (tmp_path / 'pom.xml').write_text('<project/>')          # root aggregator
        (tmp_path / 'core' / 'target').mkdir()                   # generated poms
        (tmp_path / 'core' / 'target' / 'pom.xml').write_text('x')
        # leaf names only; root aggregator and target/ poms excluded
        assert _build_module_names(tmp_path) == {'core', 'api', 'engine', 'net'}

    def test_tick_per_distinct_in_set_module(self) -> None:
        from docgen.scip_indexers import _JavaBuildProgress
        p = _JavaBuildProgress({'core', 'engine', 'net'})
        e1 = p('[info] compiling 12 Scala sources to /x/core/target/scala-2.13/classes')
        assert (e1.kind, e1.current, e1.total, e1.text) == ('tick', 1, 3, 'core')
        # same module again (another target subpath) → no second tick
        assert p('[info] writing /x/core/target/streams/foo') is None
        e2 = p('[info] compiling 3 sources to /x/engine/target/scala-2.13/classes')
        assert (e2.current, e2.total, e2.text) == (2, 3, 'engine')

    def test_module_outside_set_is_not_a_tick(self) -> None:
        from docgen.scip_indexers import _JavaBuildProgress
        p = _JavaBuildProgress({'core'})
        # the repo root and sbt's meta-build both match /X/target/ but aren't modules
        assert p('[info] done /x/spark/target/foo') is None
        assert p('[info] loading /x/project/target/config') is None

    def test_maven_reactor_still_ticks_directly(self) -> None:
        from docgen.scip_indexers import _JavaBuildProgress
        e = _JavaBuildProgress({'core'})('[INFO] Building Apache Spark Core 4.0.0 [12/34]')
        assert (e.kind, e.current, e.total) == ('tick', 12, 34)

    def test_maven_reactor_suppresses_module_target_ticks(self) -> None:
        from docgen.scip_indexers import _JavaBuildProgress
        p = _JavaBuildProgress({'core'})
        # Maven's ordered reactor (total 34) is authoritative once seen; the
        # <module>/target signal (total = module-set size) must NOT also fire,
        # or the bar interleaves two scales and jumps around.
        assert p('[INFO] Building Core 4.0.0 [3/34]').kind == 'tick'
        assert p('compiling to /x/core/target/scala-2.13/classes') is None

    def test_empty_set_leaves_total_indeterminate(self) -> None:
        from docgen.scip_indexers import _JavaBuildProgress
        # no modules discovered → module lines never tick (bar stays a pulse),
        # but Maven's own [N/M] still drives it if present
        p = _JavaBuildProgress(set())
        assert p('compiling to /x/core/target/classes') is None

    def test_stream_counts_modules_monotonically(self) -> None:
        from docgen.scip_indexers import _JavaBuildProgress, _stream_progress
        seen: list = []
        p = _JavaBuildProgress({'core', 'engine', 'net'})
        lines = [
            'compiling to /x/core/target/scala-2.13/classes',
            'noise line',
            'compiling to /x/core/target/scala-2.13/classes',   # dup → no tick
            'compiling to /x/net/target/scala-2.13/classes',
            'compiling to /x/ignored/target/classes',           # not in set → no tick
        ]
        _stream_progress(iter(lines), seen.append, parse=p)
        ticks = [(e.current, e.total, e.text) for e in seen if e.kind == 'tick']
        assert ticks == [(1, 3, 'core'), (2, 3, 'net')]
