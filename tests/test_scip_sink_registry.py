"""Contract for the SCIP sink registry (Phase 2r).

The registry is a declarative catalog of *sink primitives* — call
sites whose first argument carries a value the agent wants to know
about (an HTTP URL, later a subprocess command path). It's the data
layer underneath Phase 8b/8c (HTTP clients + URL→endpoint matching)
and the future Phase 2t (process invocations).

Architecture:

- ``SinkSpec`` declares one sink: SCIP-symbol *suffixes* (multiple,
  since indexers may emit slightly different canonical_id shapes for
  the same primitive), the language it applies to, the argument index
  that holds the interesting value, the sink kind, and an optional
  HTTP method.
- ``SinkRegistry`` aggregates ``SinkSpec`` entries and answers two
  queries the extractors need: filter by language; match a SCIP
  canonical_id to its spec.

Match semantics: **suffix-only**. The leading project/version preamble
that real SCIP indexers emit (``scip-python python pypi-requests
2.31.0 ...``) is variable; only the trailing descriptor matters. This
matches the convention Phase 8a (Akka, Flask/FastAPI, Express)
already uses, so the registry isn't a new mental model.

Language values: SCIP language names (``python`` / ``typescript`` /
``jvm``) per ``docgen/scip_languages.LANGUAGES``. ``typescript``
covers ``.js``/``.ts``/``.jsx``/``.tsx``/``.mjs``; ``jvm`` covers
Scala/Java/Kotlin. Sink primitives that exist in only one of those
sub-languages (e.g. play-ws is Scala-only) still register under
``jvm`` — the extractor running over JVM files will only encounter
the symbol on Scala input.

These tests are RED until ``docgen/scip_sink_registry.py`` exists.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# SinkSpec — the value-object shape
# ---------------------------------------------------------------------------


class TestSinkSpec:
    def test_default_arg_index_is_zero(self) -> None:
        from docgen.scip_sink_registry import SinkSpec

        s = SinkSpec(
            name='foo',
            symbol_suffixes=('foo.',),
            kind='http_client',
            language='python',
        )
        assert s.arg_index == 0

    def test_default_http_method_is_none(self) -> None:
        from docgen.scip_sink_registry import SinkSpec

        s = SinkSpec(
            name='foo',
            symbol_suffixes=('foo.',),
            kind='http_client',
            language='python',
        )
        assert s.http_method is None

    def test_explicit_fields(self) -> None:
        from docgen.scip_sink_registry import SinkSpec

        s = SinkSpec(
            name='requests.post',
            symbol_suffixes=('requests/api/post.',),
            kind='http_client',
            language='python',
            arg_index=0,
            http_method='POST',
        )
        assert s.name == 'requests.post'
        assert s.symbol_suffixes == ('requests/api/post.',)
        assert s.kind == 'http_client'
        assert s.language == 'python'
        assert s.arg_index == 0
        assert s.http_method == 'POST'

    def test_immutable(self) -> None:
        """``@frozen`` instances reject attribute reassignment so a
        SinkSpec passed around can't accidentally be mutated by a
        consumer."""
        from attrs.exceptions import FrozenInstanceError

        from docgen.scip_sink_registry import SinkSpec

        s = SinkSpec(
            name='foo', symbol_suffixes=('foo.',),
            kind='http_client', language='python',
        )
        with pytest.raises(FrozenInstanceError):
            s.name = 'changed'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SinkRegistry — match + filter helpers
# ---------------------------------------------------------------------------


class TestSinkRegistryMatching:
    def _spec(self, **overrides):
        from docgen.scip_sink_registry import SinkSpec

        defaults = dict(
            name='foo', symbol_suffixes=('foo.',),
            kind='http_client', language='python',
        )
        defaults.update(overrides)
        return SinkSpec(**defaults)

    def test_match_returns_spec_when_suffix_matches(self) -> None:
        from docgen.scip_sink_registry import SinkRegistry

        spec = self._spec(symbol_suffixes=('requests/api/get.',))
        reg = SinkRegistry(sinks=(spec,))
        assert reg.matching_symbol(
            'scip-python npm requests 2.31.0 requests/api/get.',
        ) is spec

    def test_match_against_any_of_multiple_suffixes(self) -> None:
        """A spec with multiple suffixes matches if ANY of them is
        the trailing part of the symbol — handles indexer variants."""
        from docgen.scip_sink_registry import SinkRegistry

        spec = self._spec(symbol_suffixes=(
            'requests/api/get.', 'requests/get.',
        ))
        reg = SinkRegistry(sinks=(spec,))
        assert reg.matching_symbol(
            'scip-python . . . requests/api/get.',
        ) is spec
        assert reg.matching_symbol(
            'scip-python . . . requests/get.',
        ) is spec

    def test_match_returns_none_for_unrelated_symbol(self) -> None:
        from docgen.scip_sink_registry import SinkRegistry

        spec = self._spec(symbol_suffixes=('requests/api/get.',))
        reg = SinkRegistry(sinks=(spec,))
        assert reg.matching_symbol(
            'scip-python . . . cachelib/Cache#get.',
        ) is None

    def test_match_filters_by_language(self) -> None:
        from docgen.scip_sink_registry import SinkRegistry

        py = self._spec(
            name='py-fetch',
            symbol_suffixes=('shared/fetch.',),
            language='python',
        )
        ts = self._spec(
            name='ts-fetch',
            symbol_suffixes=('shared/fetch.',),
            language='typescript',
        )
        reg = SinkRegistry(sinks=(py, ts))
        # With filter: returns the language-matched spec only
        assert reg.matching_symbol(
            'xx.shared/fetch.', language='python',
        ) is py
        assert reg.matching_symbol(
            'xx.shared/fetch.', language='typescript',
        ) is ts
        # No spec for that language → None
        assert reg.matching_symbol(
            'xx.shared/fetch.', language='jvm',
        ) is None

    def test_for_language_filters_subset(self) -> None:
        from docgen.scip_sink_registry import SinkRegistry

        py = self._spec(name='p', language='python')
        ts = self._spec(name='t', language='typescript')
        jvm = self._spec(name='j', language='jvm')
        reg = SinkRegistry(sinks=(py, ts, jvm))
        assert reg.for_language('python') == (py,)
        assert reg.for_language('typescript') == (ts,)
        assert reg.for_language('jvm') == (jvm,)
        assert reg.for_language('unknown') == ()


# ---------------------------------------------------------------------------
# Initial Python HTTP client entries
# ---------------------------------------------------------------------------


class TestPythonHttpClientEntries:
    """Pin the names + http_method assignments so accidentally
    dropping a sink during edits is caught immediately."""

    def test_requests_module_verbs_present(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in PYTHON_HTTP_CLIENT_SINKS}
        for verb in ('get', 'post', 'put', 'delete',
                     'patch', 'head', 'options'):
            assert f'requests.{verb}' in names

    def test_requests_session_methods_present(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in PYTHON_HTTP_CLIENT_SINKS}
        for verb in ('get', 'post', 'put', 'delete', 'patch'):
            assert f'requests.Session.{verb}' in names

    def test_httpx_module_and_classes(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in PYTHON_HTTP_CLIENT_SINKS}
        assert 'httpx.get' in names
        assert 'httpx.post' in names
        assert 'httpx.Client.get' in names
        assert 'httpx.AsyncClient.get' in names

    def test_aiohttp_client_session(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in PYTHON_HTTP_CLIENT_SINKS}
        assert 'aiohttp.ClientSession.get' in names
        assert 'aiohttp.ClientSession.post' in names

    def test_urllib_urlopen(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in PYTHON_HTTP_CLIENT_SINKS}
        assert 'urllib.request.urlopen' in names

    def test_http_method_set_for_named_verbs(self) -> None:
        """Sinks named after a verb (``requests.get``) must declare
        the corresponding HTTP method so downstream extractors don't
        have to re-derive it from the name."""
        from docgen.scip_sink_registry import (
            PYTHON_HTTP_CLIENT_SINKS,
        )
        verb_to_method = {
            'get': 'GET', 'post': 'POST', 'put': 'PUT',
            'delete': 'DELETE', 'patch': 'PATCH',
            'head': 'HEAD', 'options': 'OPTIONS',
        }
        for spec in PYTHON_HTTP_CLIENT_SINKS:
            tail = spec.name.rsplit('.', 1)[-1]
            if tail in verb_to_method:
                assert spec.http_method == verb_to_method[tail], (
                    f'{spec.name}: expected '
                    f'{verb_to_method[tail]}, got {spec.http_method}'
                )

    def test_all_python_entries_have_python_language(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_HTTP_CLIENT_SINKS,
        )
        for spec in PYTHON_HTTP_CLIENT_SINKS:
            assert spec.language == 'python'


# ---------------------------------------------------------------------------
# Initial JavaScript / TypeScript HTTP client entries (Phase 8b.2)
# ---------------------------------------------------------------------------


class TestJavaScriptHttpClientEntries:
    """Same shape as ``TestPythonHttpClientEntries`` — pin the JS/TS
    sink names + http_method assignments so the registry growth
    doesn't accidentally drop entries."""

    def test_fetch_present(self) -> None:
        from docgen.scip_sink_registry import (
            JAVASCRIPT_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in JAVASCRIPT_HTTP_CLIENT_SINKS}
        assert 'fetch' in names

    def test_node_fetch_present(self) -> None:
        from docgen.scip_sink_registry import (
            JAVASCRIPT_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in JAVASCRIPT_HTTP_CLIENT_SINKS}
        assert 'node-fetch' in names

    def test_axios_verbs_present(self) -> None:
        from docgen.scip_sink_registry import (
            JAVASCRIPT_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in JAVASCRIPT_HTTP_CLIENT_SINKS}
        for verb in (
            'get', 'post', 'put', 'delete', 'patch',
            'head', 'options',
        ):
            assert f'axios.{verb}' in names

    def test_got_verbs_present(self) -> None:
        from docgen.scip_sink_registry import (
            JAVASCRIPT_HTTP_CLIENT_SINKS,
        )
        names = {s.name for s in JAVASCRIPT_HTTP_CLIENT_SINKS}
        for verb in ('get', 'post', 'put', 'delete', 'patch'):
            assert f'got.{verb}' in names

    def test_all_js_entries_have_typescript_language(self) -> None:
        """Per the SCIP ``LANGUAGES`` registry, JS and TS share one
        indexer and one language name (``'typescript'``). Every JS
        client sink uses that name; the extractor filters by it."""
        from docgen.scip_sink_registry import (
            JAVASCRIPT_HTTP_CLIENT_SINKS,
        )
        for spec in JAVASCRIPT_HTTP_CLIENT_SINKS:
            assert spec.language == 'typescript'

    def test_http_method_set_for_named_verbs(self) -> None:
        from docgen.scip_sink_registry import (
            JAVASCRIPT_HTTP_CLIENT_SINKS,
        )
        verb_to_method = {
            'get': 'GET', 'post': 'POST', 'put': 'PUT',
            'delete': 'DELETE', 'patch': 'PATCH',
            'head': 'HEAD', 'options': 'OPTIONS',
        }
        for spec in JAVASCRIPT_HTTP_CLIENT_SINKS:
            tail = spec.name.rsplit('.', 1)[-1]
            if tail in verb_to_method:
                assert spec.http_method == verb_to_method[tail], (
                    f'{spec.name}: expected '
                    f'{verb_to_method[tail]}, got {spec.http_method}'
                )

    def test_fetch_default_method_is_get(self) -> None:
        """``fetch(url)`` defaults to GET. The Phase 8b.2 extractor
        records that; an ``options.method`` override is Phase 2s
        territory (kwarg-style resolution)."""
        from docgen.scip_sink_registry import (
            JAVASCRIPT_HTTP_CLIENT_SINKS,
        )
        fetch = next(
            s for s in JAVASCRIPT_HTTP_CLIENT_SINKS
            if s.name == 'fetch'
        )
        assert fetch.http_method == 'GET'


# ---------------------------------------------------------------------------
# Initial JVM HTTP client entries (Phase 8b.3)
# ---------------------------------------------------------------------------


class TestJvmHttpClientEntries:
    """JVM = Scala + Java + Kotlin per the SCIP LANGUAGES registry.
    v1 ships play-ws ``WSClient.url()`` only — sttp / Akka HTTP
    client / OkHttp builder are fluent patterns that need Phase 2s
    chained-call walking and are deferred."""

    def test_play_ws_url_present(self) -> None:
        from docgen.scip_sink_registry import SCALA_HTTP_CLIENT_SINKS
        names = {s.name for s in SCALA_HTTP_CLIENT_SINKS}
        assert 'play.WSClient.url' in names

    def test_play_ws_url_method_is_none(self) -> None:
        """The HTTP method for ``WSClient.url(...).get()`` is on the
        chained ``.get()`` we don't track in v1. ``http_method=None``
        signals 'not yet resolved' rather than mis-asserting GET."""
        from docgen.scip_sink_registry import SCALA_HTTP_CLIENT_SINKS
        ws_url = next(
            s for s in SCALA_HTTP_CLIENT_SINKS
            if s.name == 'play.WSClient.url'
        )
        assert ws_url.http_method is None

    def test_all_jvm_entries_have_jvm_language(self) -> None:
        from docgen.scip_sink_registry import SCALA_HTTP_CLIENT_SINKS
        for spec in SCALA_HTTP_CLIENT_SINKS:
            assert spec.language == 'jvm'


# ---------------------------------------------------------------------------
# Process invocation sinks (Phase 2t)
# ---------------------------------------------------------------------------


class TestProcessInvocationSinks:
    """Phase 2t adds subprocess primitives (``subprocess.run``,
    ``child_process.spawn``, ``sys.process.Process``, etc.) under
    ``kind='process_invocation'``. Same registry shape as HTTP
    client sinks; new ``SinkKind`` value."""

    def test_python_subprocess_run_present(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_PROCESS_SINKS,
        )
        names = {s.name for s in PYTHON_PROCESS_SINKS}
        assert 'subprocess.run' in names

    def test_python_subprocess_popen_present(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_PROCESS_SINKS,
        )
        names = {s.name for s in PYTHON_PROCESS_SINKS}
        assert 'subprocess.Popen' in names

    def test_python_os_system_present(self) -> None:
        from docgen.scip_sink_registry import (
            PYTHON_PROCESS_SINKS,
        )
        names = {s.name for s in PYTHON_PROCESS_SINKS}
        assert 'os.system' in names

    def test_javascript_child_process_spawn_present(self) -> None:
        from docgen.scip_sink_registry import (
            JAVASCRIPT_PROCESS_SINKS,
        )
        names = {s.name for s in JAVASCRIPT_PROCESS_SINKS}
        assert 'child_process.spawn' in names

    def test_javascript_child_process_exec_present(self) -> None:
        from docgen.scip_sink_registry import (
            JAVASCRIPT_PROCESS_SINKS,
        )
        names = {s.name for s in JAVASCRIPT_PROCESS_SINKS}
        assert 'child_process.exec' in names

    def test_jvm_scala_process_present(self) -> None:
        from docgen.scip_sink_registry import (
            JVM_PROCESS_SINKS,
        )
        names = {s.name for s in JVM_PROCESS_SINKS}
        assert 'scala.sys.process.Process' in names

    def test_jvm_runtime_exec_present(self) -> None:
        from docgen.scip_sink_registry import (
            JVM_PROCESS_SINKS,
        )
        names = {s.name for s in JVM_PROCESS_SINKS}
        assert 'java.lang.Runtime.exec' in names

    def test_all_process_entries_have_process_kind(self) -> None:
        """Verifies kind on every process sink. The non-empty assertion
        bites a stub where the buckets are ``()`` — without it,
        ``for spec in (): ...`` runs zero iterations and falsely passes."""
        from docgen.scip_sink_registry import (
            JAVASCRIPT_PROCESS_SINKS,
            JVM_PROCESS_SINKS,
            PYTHON_PROCESS_SINKS,
        )
        all_process = (
            *PYTHON_PROCESS_SINKS,
            *JAVASCRIPT_PROCESS_SINKS,
            *JVM_PROCESS_SINKS,
        )
        assert len(all_process) > 0, (
            'expected populated process sink buckets; got empty'
        )
        for spec in all_process:
            assert spec.kind == 'process_invocation', (
                f'{spec.name}: kind must be process_invocation'
            )

    def test_process_entry_languages_correct(self) -> None:
        """Each bucket must be non-empty AND every entry must carry
        the right language tag. Empty buckets fail the size check;
        wrong language fails the per-entry check."""
        from docgen.scip_sink_registry import (
            JAVASCRIPT_PROCESS_SINKS,
            JVM_PROCESS_SINKS,
            PYTHON_PROCESS_SINKS,
        )
        assert len(PYTHON_PROCESS_SINKS) > 0, 'PYTHON empty'
        assert len(JAVASCRIPT_PROCESS_SINKS) > 0, 'JAVASCRIPT empty'
        assert len(JVM_PROCESS_SINKS) > 0, 'JVM empty'
        for spec in PYTHON_PROCESS_SINKS:
            assert spec.language == 'python'
        for spec in JAVASCRIPT_PROCESS_SINKS:
            assert spec.language == 'typescript'
        for spec in JVM_PROCESS_SINKS:
            assert spec.language == 'jvm'

    def test_default_registry_includes_process_entries(self) -> None:
        """Process sinks must appear in the aggregated registry. The
        non-empty assertions bite empty source tuples — without them
        ``set(()) <= anything`` is trivially true."""
        from docgen.scip_sink_registry import (
            DEFAULT_SINK_REGISTRY,
            JAVASCRIPT_PROCESS_SINKS,
            JVM_PROCESS_SINKS,
            PYTHON_PROCESS_SINKS,
        )
        assert len(PYTHON_PROCESS_SINKS) > 0
        assert len(JAVASCRIPT_PROCESS_SINKS) > 0
        assert len(JVM_PROCESS_SINKS) > 0
        all_specs = set(DEFAULT_SINK_REGISTRY.sinks)
        assert set(PYTHON_PROCESS_SINKS) <= all_specs
        assert set(JAVASCRIPT_PROCESS_SINKS) <= all_specs
        assert set(JVM_PROCESS_SINKS) <= all_specs


# ---------------------------------------------------------------------------
# Default registry aggregation + invariants
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_default_registry_includes_python_entries(self) -> None:
        from docgen.scip_sink_registry import (
            DEFAULT_SINK_REGISTRY,
            PYTHON_HTTP_CLIENT_SINKS,
        )
        assert set(PYTHON_HTTP_CLIENT_SINKS) <= set(
            DEFAULT_SINK_REGISTRY.sinks,
        )

    def test_default_registry_uses_scip_language_names(self) -> None:
        """Languages must be the SCIP-language registry names so
        callers can pass the same key they use for indexing — keeps
        ``language`` orthogonal to file extensions and grammar
        dispatch."""
        from docgen.scip_languages import LANGUAGES
        from docgen.scip_sink_registry import DEFAULT_SINK_REGISTRY

        valid = {lang.name for lang in LANGUAGES}
        for spec in DEFAULT_SINK_REGISTRY.sinks:
            assert spec.language in valid, (
                f'{spec.name}: language {spec.language!r} not in '
                f'LANGUAGES'
            )

    def test_all_entries_have_non_empty_suffixes(self) -> None:
        """Empty ``symbol_suffixes`` would either match nothing (dead
        spec) or — worse — match everything if special-cased. Forbid."""
        from docgen.scip_sink_registry import DEFAULT_SINK_REGISTRY

        for spec in DEFAULT_SINK_REGISTRY.sinks:
            assert spec.symbol_suffixes, (
                f'{spec.name}: must declare at least one suffix'
            )
            for suf in spec.symbol_suffixes:
                assert suf, f'{spec.name}: empty suffix string'

    def test_sink_names_unique(self) -> None:
        """``name`` is the human-readable identifier; keep it unique
        across the registry."""
        from docgen.scip_sink_registry import DEFAULT_SINK_REGISTRY

        names = [s.name for s in DEFAULT_SINK_REGISTRY.sinks]
        assert len(names) == len(set(names)), (
            f'duplicate sink names: '
            f'{[n for n in names if names.count(n) > 1]}'
        )

    def test_arg_index_non_negative(self) -> None:
        from docgen.scip_sink_registry import DEFAULT_SINK_REGISTRY

        for spec in DEFAULT_SINK_REGISTRY.sinks:
            assert spec.arg_index >= 0, (
                f'{spec.name}: arg_index={spec.arg_index} is negative'
            )
