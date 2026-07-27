"""Slice 2 — the version_facts layer (north star §12).

The spool's value claim (b) — version-pinned — is unfalsifiable while the
pin lives only in the manifest: extract the corpus's structured version
markers into declarative facts (symbol → since/deprecated + version), store
them per source, and join them against the runtime's component map so
"is X available on MY runtime" resolves deterministically. Deterministic
markers ONLY (annotations/directives, never prose) — these facts will feed
the corrector, so precision beats recall. Synthetic fixtures.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import library.version_facts as version_facts
from library import Library


class TestExtractVersionFacts:
    """Pure extraction over one doc's content."""

    def test_scala_since_annotation(self):
        facts = version_facts.extract_version_facts(
            'scala_object pkg.Frobnicator [scala] /x.scala:12 :: '
            '@Since("3.1.0")\nobject Frobnicator')
        assert ('since', '3.1.0') in [(f.fact, f.version) for f in facts]

    def test_sphinx_versionadded(self):
        facts = version_facts.extract_version_facts(
            'def frobnicate(x):\n    .. versionadded:: 2.4.0\n    body')
        assert [(f.fact, f.version) for f in facts] == [('since', '2.4.0')]

    def test_scala_deprecated_with_version(self):
        facts = version_facts.extract_version_facts(
            '@deprecated("use Quantum Mesh instead", "3.1.0")\ndef old(): Unit')
        assert [(f.fact, f.version) for f in facts] == [('deprecated', '3.1.0')]

    def test_sphinx_deprecated(self):
        facts = version_facts.extract_version_facts(
            'class Old:\n    .. deprecated:: 3.0.0\n    Use New instead.')
        assert [(f.fact, f.version) for f in facts] == [('deprecated', '3.0.0')]

    def test_bare_java_deprecated_has_no_version(self):
        facts = version_facts.extract_version_facts(
            '@Deprecated\npublic void old() {}')
        assert [(f.fact, f.version) for f in facts] == [('deprecated', None)]

    def test_prose_never_extracts(self):
        # 'deprecated' in prose is NOT a structured marker — precision rule.
        assert version_facts.extract_version_facts(
            'This API is deprecated in spirit and generally discouraged.',
        ) == []

    def test_multiple_markers_and_evidence(self):
        facts = version_facts.extract_version_facts(
            '@Since("3.0.0")\n@deprecated("gone", "3.5.0")\ndef f(): Unit')
        got = {(f.fact, f.version) for f in facts}
        assert got == {('since', '3.0.0'), ('deprecated', '3.5.0')}
        assert all(f.evidence for f in facts)


class TestVersionFactsStore:
    def test_init_upsert_query_and_dedupe(self, tmp_path: Path):
        with Library(tmp_path / 'vf.db') as lib:
            with lib._conn_provider.acquire() as conn:
                version_facts.init_version_facts_schema(conn)
                rows = [
                    version_facts.VersionFactRow(
                        source_name='env1', qualified_name='pkg.Frobnicator',
                        fact='since', version='3.1.0',
                        evidence='@Since("3.1.0")', doc_id='d1'),
                    version_facts.VersionFactRow(
                        source_name='env1', qualified_name='pkg.Frobnicator',
                        fact='deprecated', version='3.5.0',
                        evidence='@deprecated', doc_id='d1'),
                    version_facts.VersionFactRow(
                        source_name='other', qualified_name='pkg.Frobnicator',
                        fact='since', version='1.0.0',
                        evidence='@Since("1.0.0")', doc_id='d2'),
                ]
                version_facts.upsert_version_facts(conn, rows)
                version_facts.upsert_version_facts(conn, rows)  # idempotent
            with lib._conn_provider.acquire() as conn:
                got = version_facts.query_version_facts(
                    conn, ['env1'], 'pkg.Frobnicator')
                assert {(f.fact, f.version) for f in got} == {
                    ('since', '3.1.0'), ('deprecated', '3.5.0')}
                n = conn.execute(
                    'SELECT COUNT(*) FROM version_facts').fetchone()[0]
                assert n == 3  # dedupe held across the double upsert
            with lib._conn_provider.acquire() as conn:
                # Bare @Deprecated facts carry version=None, and SQLite's
                # UNIQUE constraint treats NULLs as DISTINCT — without the
                # NULL-safe index every re-extraction/install pass multiplies
                # them (observed live: 8 facts -> x4 after build + install).
                bare = version_facts.VersionFactRow(
                    source_name='env1', qualified_name='pkg.Legacy',
                    fact='deprecated', version=None,
                    evidence='@Deprecated', doc_id='d3')
                version_facts.upsert_version_facts(conn, [bare, bare])
                version_facts.upsert_version_facts(conn, [bare])
                n = conn.execute(
                    'SELECT COUNT(*) FROM version_facts').fetchone()[0]
                assert n == 4
                # A store that pre-dates the NULL-safe index self-heals on
                # init: legacy duplicates collapse to the oldest row.
                conn.execute('DROP INDEX idx_version_facts_dedupe')
                conn.execute(
                    "INSERT INTO version_facts (source_name, qualified_name,"
                    " fact, version, evidence, doc_id, component)"
                    " VALUES ('env1', 'pkg.Legacy', 'deprecated', NULL,"
                    " '@Deprecated', 'd3', '')")
                version_facts.init_version_facts_schema(conn)
                n = conn.execute(
                    'SELECT COUNT(*) FROM version_facts').fetchone()[0]
                assert n == 4


class TestExtractForSource:
    """The store's element signatures are TRUNCATED at the annotation
    (measured: 2,861 element docs carry bare '@Since', zero carry
    '@Since(' — the version payload exists only in the corpus source
    files). So extraction is a BUILD/BACKFILL-time pass: the store's bare
    markers prefilter the candidates, the located source line supplies the
    version, and the pack ships the resulting FACTS — consumers never need
    the files."""

    def _element(self, lib, tmp_path, qn, doc_id, *, line, source_text=None,
                 source='env1', outer='spool:env1', repo='repoa'):
        src_file = tmp_path / repo / f'{doc_id}.scala'
        if source_text is not None:
            src_file.parent.mkdir(parents=True, exist_ok=True)
            src_file.write_text(source_text)
        lib.add_document(
            'catalog', qn,
            f'scala_object {qn} [scala] {src_file}:{line}-{line} :: @Since',
            source_files=[str(src_file)],
            metadata={'kind': 'element', 'source_name': source,
                      'qualified_name': qn, 'subtype': 'scala_object',
                      'location': {'line_start': line, 'line_end': line}},
            doc_id=doc_id, source_name=outer,
            _allow_reserved_source=outer.startswith('spool:'),
        )

    def test_versions_read_from_the_located_source_line(self, tmp_path: Path):
        with Library(tmp_path / 'vf.db') as lib:
            self._element(
                lib, tmp_path, 'pkg.Frobnicator', 'd-frob', line=3,
                source_text='package pkg\n\n@Since("3.1.0")\nobject Frobnicator',
            )
            self._element(   # marker in store, source file GONE -> honest skip
                lib, tmp_path, 'pkg.Ghost', 'd-ghost', line=2,
                source_text=None,
            )
            self._element(   # other corpus source: never extracted
                lib, tmp_path, 'other.Thing', 'd-other', line=3,
                source_text='package other\n\n@Since("9.9.9")\nobject Thing',
                source='otherenv', outer='otherenv',
            )
            n = version_facts.extract_facts_for_source(
                lib, outer_sources=['spool:env1'], corpus_source='env1',
                source_root=tmp_path)
            assert n == 1
            with lib._conn_provider.acquire() as conn:
                got = version_facts.query_version_facts(
                    conn, ['env1'], 'pkg.Frobnicator')
                assert [(f.fact, f.version) for f in got] == [
                    ('since', '3.1.0')]
                assert got[0].doc_id == 'd-frob'
                # The corpus holds several differently-versioned repos under
                # ONE source: the fact carries WHICH repo (its component),
                # derived from the file's first segment under the root.
                assert got[0].component == 'repoa'
                assert version_facts.query_version_facts(
                    conn, ['env1'], 'pkg.Ghost') == []
                assert version_facts.query_version_facts(
                    conn, ['env1'], 'other.Thing') == []


class TestVersionCompare:
    def test_numeric_segment_order(self):
        at_most = version_facts.version_at_most
        assert at_most('3.1.0', '3.5.2')
        assert at_most('3.5', '3.5.0')
        assert at_most('3.9', '3.10')       # numeric, not lexicographic
        assert not at_most('4.0.0', '3.5')

    def test_suffixes_compare_on_numeric_prefix(self):
        assert version_facts.version_at_most('4.0.0-preview', '4.0.0')
        assert not version_facts.version_at_most('4.1.0-rc1', '4.0.0')


class TestRuntimeComponentsFromResolution:
    """The availability join's right-hand side comes from the PIN — the
    registered manifests' runtime_components — never caller-supplied dicts.
    Handles all three shapes the resolution can hold: a registration with
    .manifest, a bare manifest object, or a dict."""

    def test_merges_registered_manifests(self):
        registration = SimpleNamespace(
            manifest=SimpleNamespace(runtime_components={'env1': '3.5.0'}))
        bare = SimpleNamespace(runtime_components={'env2': '2.0'})
        as_dict = SimpleNamespace(
            manifest={'runtime_components': {'env3': '1.0'}})
        resolution = SimpleNamespace(registered={
            'env1': registration, 'env2': bare, 'env3': as_dict,
        })
        got = version_facts.runtime_components_from_resolution(resolution)
        assert got == {'env1': '3.5.0', 'env2': '2.0', 'env3': '1.0'}

    def test_empty_when_nothing_registered(self):
        assert version_facts.runtime_components_from_resolution(
            SimpleNamespace(registered={})) == {}


class TestRuntimeAvailability:
    def test_join_against_runtime_components(self, tmp_path: Path):
        with Library(tmp_path / 'vf.db') as lib:
            with lib._conn_provider.acquire() as conn:
                version_facts.init_version_facts_schema(conn)
                version_facts.upsert_version_facts(conn, [
                    version_facts.VersionFactRow(
                        source_name='env1', qualified_name='pkg.Early',
                        fact='since', version='3.1.0', evidence='e',
                        doc_id='d', component='repoa'),
                    version_facts.VersionFactRow(
                        source_name='env1', qualified_name='pkg.Future',
                        fact='since', version='9.0.0', evidence='e',
                        doc_id='d', component='repoa'),
                    version_facts.VersionFactRow(
                        source_name='env1', qualified_name='pkg.Old',
                        fact='deprecated', version='3.0.0', evidence='e',
                        doc_id='d', component='repoa'),
                    # A second component at a DIFFERENT version: the join
                    # must use the fact's component, never one
                    # version-per-source.
                    version_facts.VersionFactRow(
                        source_name='env1', qualified_name='sdk.Fresh',
                        fact='since', version='0.9.0', evidence='e',
                        doc_id='d', component='repob'),
                ])
            runtime = {'repoa': '3.5.0', 'repob': '0.5.0'}
            avail = version_facts.runtime_availability
            with lib._conn_provider.acquire() as conn:
                early = avail(conn, ['env1'], 'pkg.Early', runtime)
                assert early['available'] is True
                assert early['since'] == '3.1.0'
                future = avail(conn, ['env1'], 'pkg.Future', runtime)
                assert future['available'] is False
                old = avail(conn, ['env1'], 'pkg.Old', runtime)
                assert old['deprecated'] == '3.0.0'
                unknown = avail(conn, ['env1'], 'pkg.Unknown', runtime)
                assert unknown['available'] is None   # no facts -> honest None
                # component-keyed: sdk.Fresh needs repob 0.9 but the runtime
                # ships repob 0.5 -> unavailable, regardless of repoa's 3.5.
                fresh = avail(conn, ['env1'], 'sdk.Fresh', runtime)
                assert fresh['available'] is False
