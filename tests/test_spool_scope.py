"""A spool serves the projects that asked for it, not every project.

``SpoolSetting.projects`` already records which of the user's sources a spool
cross-checks, and ``reconcile_spool_themes`` already honours it when building
theme partitions. The query path does not: ``scope_sources()`` takes no project
and returns every registered spool, so a question about one project is answered
from a candidate pool containing an environment it does not run on.

The asking project is available at query time (``_search_uncached(..., source=
...)``), so the gate is a filter on the resolution rather than a new mechanism.

Back-compat: an empty ``projects`` means "not declared yet" and admits every
project, so enabling scope cannot silently switch a spool off for an existing
config (designs/spool-scope.md §4.3 — the undeclared case surfaces as a gap
instead).

Synthetic fixtures only: fake spool names, fake runtimes, tmp-path caches.
"""
import textwrap

from config import Config
from spools import resolve_spools

MANIFEST = '''
    environment: fakebricks
    version: '1.0.0'
    target_runtime: fake-17.3
    checksum: abc123
'''


def _cache(tmp_path):
    cache_dir = tmp_path / 'spool-cache'
    (cache_dir / 'fakebricks').mkdir(parents=True)
    (cache_dir / 'fakebricks' / 'manifest.yaml').write_text(
        textwrap.dedent(MANIFEST))
    return cache_dir


def _resolution(tmp_path, yaml_body):
    path = tmp_path / 'ariadne.yaml'
    path.write_text(textwrap.dedent(yaml_body))
    return resolve_spools(Config(config_path=path), cache_dir=_cache(tmp_path))


class TestSpoolScopeFollowsDeclaredProjects:
    def test_declared_project_gets_the_spool_and_others_do_not(
        self, tmp_path,
    ) -> None:
        resolution = _resolution(tmp_path, '''
            spools:
              fakebricks:
                runtime: fake-17.3
                projects: [runs-on-it]
        ''')
        assert resolution.gaps == ()

        assert resolution.scope_sources(for_project='runs-on-it') == \
            frozenset({'spool:fakebricks'})
        assert resolution.scope_sources(for_project='unrelated') == frozenset()

    def test_unscoped_query_still_sees_every_registered_spool(
        self, tmp_path,
    ) -> None:
        """A query with no project named keeps today's union — the caller has
        not claimed a scope, so nothing is narrowed."""
        resolution = _resolution(tmp_path, '''
            spools:
              fakebricks:
                runtime: fake-17.3
                projects: [runs-on-it]
        ''')
        assert resolution.scope_sources() == frozenset({'spool:fakebricks'})
        assert resolution.scope_sources(for_project=None) == \
            frozenset({'spool:fakebricks'})

    def test_undeclared_projects_admits_everyone(self, tmp_path) -> None:
        """Empty ``projects`` is "not declared yet", not "nobody" — an existing
        config keeps working until its owner declares a scope."""
        resolution = _resolution(tmp_path, '''
            spools:
              fakebricks:
                runtime: fake-17.3
        ''')
        assert resolution.scope_sources(for_project='anyone') == \
            frozenset({'spool:fakebricks'})


class TestScopedLibraryHonoursDeclaredProjects:
    """The admission point: ``make_scoped_library`` unions registered spools
    into every project's closure. That union is where an out-of-scope project
    acquires the spool's documents as ranking candidates, so it is where the
    declared scope has to be enforced — narrowing here narrows every consumer
    (search, ask, admin) at once rather than one gate at a time.
    """

    def _project(self, tmp_path):
        """Two sources; the spool is declared for only one of them."""
        for name in ('runs-on-it', 'unrelated'):
            (tmp_path / name).mkdir()
        cache = tmp_path / '.ariadne' / 'spools' / 'fakebricks'
        cache.mkdir(parents=True)
        (cache / 'manifest.yaml').write_text(textwrap.dedent(MANIFEST))
        path = tmp_path / 'ariadne.yaml'
        path.write_text(textwrap.dedent(f'''
            default_source: runs-on-it
            sources:
              runs-on-it:
                path: {tmp_path / 'runs-on-it'}
              unrelated:
                path: {tmp_path / 'unrelated'}
            spools:
              fakebricks:
                runtime: fake-17.3
                projects: [runs-on-it]
        '''))
        return Config(config_path=path)

    def test_out_of_scope_project_never_gets_the_spool_as_a_candidate(
        self, tmp_path,
    ) -> None:
        from library import Library
        from scope_resolution import make_scoped_library

        config = self._project(tmp_path)
        with Library(tmp_path / 'lib.db') as library:
            in_scope = make_scoped_library(
                config, library, 'runs-on-it', use_cwd=False)
            out_of_scope = make_scoped_library(
                config, library, 'unrelated', use_cwd=False)

        assert 'spool:fakebricks' in in_scope.closure, (
            'the project that declared the spool must still see it')
        assert 'spool:fakebricks' not in out_of_scope.closure, (
            'a project that never declared the spool must not rank against '
            f'it; closure was {sorted(out_of_scope.closure)}')
        # the project's own sources are untouched by the gate
        assert 'unrelated' in out_of_scope.closure


class TestAskNarrowsToTheAskingProject:
    """Some spool content reaches an answer WITHOUT passing through retrieval.

    ``_version_facts_block`` derives its corpus straight from the resolution
    and injects pinned versions into the prompt, so the closure gate does not
    cover it: an out-of-scope project would still be told the Databricks
    component versions. The environment label and provenance line have the
    same shape. Narrowing the resolution once — rather than threading a
    project through each consumer — closes all of them together, and moves
    the fingerprint so a cached answer shaped by a spool cannot be replayed
    for a project that does not get that spool.
    """

    def _resolved(self, tmp_path):
        return _resolution(tmp_path, '''
            spools:
              fakebricks:
                runtime: fake-17.3
                projects: [runs-on-it]
        ''')

    def test_narrowing_is_per_project_and_moves_the_fingerprint(
        self, tmp_path,
    ) -> None:
        resolution = self._resolved(tmp_path)

        assert resolution.narrowed_to('runs-on-it').scope_sources() == \
            frozenset({'spool:fakebricks'})
        assert resolution.narrowed_to('unrelated').scope_sources() == frozenset()
        assert resolution.narrowed_to(None).scope_sources() == \
            frozenset({'spool:fakebricks'})

        assert resolution.narrowed_to('runs-on-it').fingerprint() == \
            resolution.fingerprint()
        assert resolution.narrowed_to('unrelated').fingerprint() != \
            resolution.fingerprint()

    def test_out_of_scope_project_gets_no_pinned_version_facts(
        self, tmp_path,
    ) -> None:
        from ariadne_mcp.service_analysis import AnalysisMixin

        resolution = self._resolved(tmp_path)
        question = 'Which DeltaTable version ships with this runtime?'

        # corpus is empty for a project the spool does not serve, so the block
        # is skipped before any lookup — no environment facts leak in.
        assert AnalysisMixin._version_facts_block(
            None, question, resolution.narrowed_to('unrelated')) is None
        # the declared project keeps a corpus, so the block is reached rather
        # than short-circuited (whether it finds a matching fact depends on
        # what is stored, which is not what this test is about)
        assert resolution.narrowed_to('runs-on-it').scope_sources() == \
            frozenset({'spool:fakebricks'})
