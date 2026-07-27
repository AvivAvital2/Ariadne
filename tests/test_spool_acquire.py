"""Slice (b3) of the Spool plugin: builder acquisition + consent-before-fetch.

USER REQUIREMENT (2026-07-08): before ANY acquisition, show exactly what
will be fetched — repos at pinned tags with SHAs — and ask to proceed
(`--approve` skips for CI). Clones are verified at the declared SHA and
a mismatch fails loud with the clone removed. Local git fixture repos
only — no network. Design: IMPLEMENT.md (b3).
"""
import contextlib
import io
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml
import spool_acquire

from spool_acquire import (
    _default_phases,
    acquire,
    consent_text,
    create_spool,
    load_packfile,
    setup_recipe,
)
from spools import SpoolError, is_scip_eligible, unsupported_corpus_language


def _git(cwd, *args):
    result = subprocess.run(
        ['git', '-c', 'user.email=t@t', '-c', 'user.name=t', *args],
        cwd=cwd, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.fixture
def fixture_repo(tmp_path):
    repo = tmp_path / 'upstream-fakelib'
    repo.mkdir()
    _git(repo, 'init', '-q')
    (repo / 'engine.py').write_text('VERSION = 1\n')
    # A permissive LICENSE, like every real corpus — so the license-admission
    # gate (§18.1) admits it as redistribution-safe.
    (repo / 'LICENSE').write_text('Apache License\nVersion 2.0, January 2004\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'v1')
    pinned_sha = _git(repo, 'rev-parse', 'HEAD')
    # ANNOTATED tag, like every real Spark/Delta release — resolution must
    # return the peeled commit sha, not the tag object (CRIT-1).
    _git(repo, 'tag', '-a', 'v1.0', '-m', 'release v1.0')
    (repo / 'engine.py').write_text('VERSION = 2\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'v2')
    return repo, pinned_sha


def _packfile(tmp_path, repo, sha):
    path = tmp_path / 'fakebricks.packfile.yaml'
    path.write_text(textwrap.dedent(f'''
        runtime: fake-17.3
        corpus:
          fakelib:
            url: {repo}
            tag: v1.0
            sha: {sha}
        certify: [docs/]
    '''))
    return path


def _git_repo_with(tmp_path, name, filename, content):
    """A tagged single-commit git repo containing one file (for driving
    create_spool's clone against a controlled corpus language)."""
    repo = tmp_path / name
    (repo / Path(filename).parent).mkdir(parents=True, exist_ok=True)
    _git(repo, 'init', '-q')
    (repo / filename).write_text(content)
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-q', '-m', 'c')
    sha = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'tag', '-a', 'v1.0', '-m', 'r')
    return repo, sha


class TestSpoolLanguageGate:
    """A Spool must be grounded through a registered SCIP indexer. A corpus
    in a language with no adapter (e.g. Go) is refused up front — never
    silently built from the raw-file/ast-grep fallback."""

    def test_scip_eligibility_from_registry(self):
        # Eligible = a language with a registered SCIP adapter (python /
        # typescript / jvm-family / go), derived from the SCIP registry — so
        # adding the scip-go entry flipped Go eligible with no change here.
        for ok in ('python', 'py', 'typescript', 'ts', 'javascript', 'js',
                   'scala', 'java', 'kotlin', 'jvm', 'go', 'golang',
                   '.py', '.scala', '.go'):
            assert is_scip_eligible(ok), ok
        for bad in ('ruby', 'rust', 'c', 'cpp', '', None):
            assert not is_scip_eligible(bad), bad

    def test_corpus_scan_flags_unsupported_only_when_ungrounded(self, tmp_path):
        ruby_only = tmp_path / 'ruby_only'
        (ruby_only / 'lib').mkdir(parents=True)
        (ruby_only / 'lib' / 'app.rb').write_text("puts 'hi'\n")
        (ruby_only / 'README.md').write_text('# x\n')
        assert unsupported_corpus_language(ruby_only) == 'Ruby'
        # any supported source present → grounded → gate does not fire
        mixed = tmp_path / 'mixed'
        mixed.mkdir()
        (mixed / 'app.rb').write_text("puts 'hi'\n")
        (mixed / 'util.py').write_text('x = 1\n')
        assert unsupported_corpus_language(mixed) is None
        # docs-only (no code at all) → out of scope for THIS gate
        docs = tmp_path / 'docs'
        docs.mkdir()
        (docs / 'guide.md').write_text('# g\n')
        assert unsupported_corpus_language(docs) is None

    def test_create_aborts_on_unsupported_declared_language(
        self, tmp_path, fixture_repo,
    ):
        repo, _sha = fixture_repo
        spoolfile = tmp_path / 'ruby-spools.yaml'
        spoolfile.write_text(textwrap.dedent(f'''
            name: rubybricks
            runtime: ruby-1.x
            version: 1.0.0
            languages: [ruby]
            corpus:
              rubycore:
                url: {repo}
                tag: v1.0
        '''))
        calls = []
        phases = {
            'source_add': lambda name, path: calls.append('source_add'),
            'index': lambda name: calls.append('index'),
            'onboard': lambda name: calls.append('onboard'),
            'build': lambda **kw: calls.append('build'),
        }
        with pytest.raises(SpoolError) as excinfo:
            create_spool(
                spoolfile, dest_dir=tmp_path / 'corpus',
                out_path=tmp_path / 'pack.zip', approve=True,
                confirm=lambda p: calls.append('consent') or 'y',
                phases=phases,
            )
        msg = str(excinfo.value).lower()
        assert 'ruby' in msg
        assert 'not supported' in msg or "can't" in msg or 'cannot' in msg
        # Refused BEFORE any fetch / consent / phase — no clone, no cost.
        assert calls == []
        assert not (tmp_path / 'corpus').exists()

    def test_create_backstop_aborts_on_ungrounded_corpus(self, tmp_path):
        # Undeclared case: recipe names no languages, but the cloned corpus
        # turns out to be Ruby — the post-clone backstop refuses BEFORE the
        # paid onboard step (never falls through to a hollow pack).
        repo, _sha = _git_repo_with(tmp_path, 'rubyupstream', 'lib/app.rb',
                                    "puts 'hi'\n")
        spoolfile = tmp_path / 'undeclared-spools.yaml'
        spoolfile.write_text(textwrap.dedent(f'''
            name: rubybricks
            runtime: ruby-1.x
            version: 1.0.0
            corpus:
              rubycore:
                url: {repo}
                tag: v1.0
        '''))
        calls = []
        phases = {
            'source_add': lambda name, path: calls.append('source_add'),
            'index': lambda name: calls.append('index'),
            'onboard': lambda name: calls.append('onboard'),
            'build': lambda **kw: calls.append('build'),
        }
        with pytest.raises(SpoolError) as excinfo:
            create_spool(
                spoolfile, dest_dir=tmp_path / 'gocorpus',
                out_path=tmp_path / 'pack.zip', approve=True,
                confirm=lambda p: 'y', phases=phases,
            )
        assert 'ruby' in str(excinfo.value).lower()
        # Clone may have happened (undeclared → can't know pre-clone), but no
        # phase ran — the paid onboard was never reached.
        assert calls == []

    def test_create_threads_taxonomy_recipe_to_build(self, tmp_path, fixture_repo):
        # The aisle's advisory taxonomy declared in the spoolfile must reach the
        # build phase (→ the pack manifest), so a built aisle carries its lens.
        repo, _sha = fixture_repo
        spoolfile = tmp_path / 'tax-spools.yaml'
        spoolfile.write_text(textwrap.dedent(f'''
            name: taxbricks
            runtime: r-1.x
            version: 1.0.0
            languages: [python]
            taxonomy: [serialization, parallelism]
            corpus:
              core:
                url: {repo}
                tag: v1.0
        '''))
        captured = {}
        phases = {
            'source_add': lambda name, path: None,
            'index': lambda name: None,
            'onboard': lambda name: None,
            'build': lambda **kw: captured.update(kw),
        }
        create_spool(
            spoolfile, dest_dir=tmp_path / 'corpus',
            out_path=tmp_path / 'p.zip', approve=True,
            confirm=lambda p: 'y', phases=phases,
        )
        assert captured['taxonomy'] == ('serialization', 'parallelism')

    def test_allow_ungrounded_bypasses_language_gate(
        self, tmp_path, fixture_repo,
    ):
        repo, _sha = fixture_repo
        spoolfile = tmp_path / 'ruby2-spools.yaml'
        spoolfile.write_text(textwrap.dedent(f'''
            name: rubybricks
            runtime: ruby-1.x
            version: 1.0.0
            languages: [ruby]
            corpus:
              rubycore:
                url: {repo}
                tag: v1.0
        '''))
        calls = []
        phases = {
            'source_add': lambda name, path: calls.append('source_add'),
            'index': lambda name: calls.append('index'),
            'onboard': lambda name: calls.append('onboard'),
            'theme': lambda name: calls.append('theme'),
            'build': lambda **kw: calls.append('build'),
        }
        result = create_spool(
            spoolfile, dest_dir=tmp_path / 'corpus2',
            out_path=tmp_path / 'pack2.zip', approve=True,
            confirm=lambda p: 'y', phases=phases, allow_ungrounded=True,
        )
        assert result.accepted is True
        # The spool's own theme pass runs between onboard and pack build, so
        # its corpus themes are tagged to spool:<name>, not the base '' pass.
        assert calls == ['source_add', 'index', 'onboard', 'theme', 'build']


class TestSpoolAcquire:
    def test_acquire_lifecycle(self, tmp_path, fixture_repo):
        repo, pinned_sha = fixture_repo

        # Demand 1 — packfile schema: round-trips; missing sha is loud
        # (tags are mutable; the SHA is the pin).
        packfile = load_packfile(_packfile(tmp_path, repo, pinned_sha))
        assert packfile.runtime == 'fake-17.3'
        assert packfile.corpus['fakelib'].sha == pinned_sha
        bad = tmp_path / 'bad.packfile.yaml'
        bad.write_text(textwrap.dedent(f'''
            runtime: fake-17.3
            corpus:
              fakelib: {{url: {repo}, tag: v1.0}}
        '''))
        with pytest.raises(SpoolError) as excinfo:
            load_packfile(bad)
        assert 'sha' in str(excinfo.value)

        # Demand 2 — the consent text names exactly what will be fetched.
        text = consent_text(packfile)
        assert 'fakelib@v1.0' in text
        assert pinned_sha[:12] in text
        assert 'fake-17.3' in text

        # Demand 3 — declined consent: nothing is cloned.
        dest = tmp_path / 'corpus'
        result = acquire(
            packfile, dest_dir=dest, approve=False, confirm=lambda _: 'n',
        )
        assert result.accepted is False
        assert result.cloned == ()
        assert not (dest / 'fakelib').exists()

        # Demand 4 — approved: cloned AND checked out at the EXACT sha
        # (not the branch tip — the fixture has a newer commit).
        result = acquire(packfile, dest_dir=dest, approve=True)
        assert result.accepted is True
        assert result.cloned == ('fakelib',)
        clone = dest / 'fakelib'
        # .git is dropped after checkout — we keep only the source tree (no
        # history/metadata to index), pinned by a marker recording the sha.
        assert not (clone / '.git').exists()
        assert (clone / '.ariadne-corpus-sha').read_text().strip() == pinned_sha
        assert (clone / 'engine.py').read_text() == 'VERSION = 1\n'

        # Demand 4b (CRIT-2) — re-run is idempotent: an existing checkout at
        # the pinned sha is REUSED, and anything inside it survives.
        (clone / 'USER-LOCAL-WORK.txt').write_text('precious')
        rerun = acquire(packfile, dest_dir=dest, approve=True)
        assert rerun.accepted is True and rerun.cloned == ('fakelib',)
        assert (clone / 'USER-LOCAL-WORK.txt').read_text() == 'precious'

        # Demand 4c — a DIFFERENT pinned sha re-fetches our own corpus (it's an
        # ariadne workspace artifact, marker-identified) instead of refusing;
        # the stale checkout, and the file left in it, is replaced. (A FOREIGN
        # dir IS protected — see test_acquire_reuse_refuses_without_marker.)
        newer_sha = _git(repo, 'rev-parse', 'HEAD')      # the fixture's 2nd commit
        refetched = acquire(load_packfile(_packfile(tmp_path, repo, newer_sha)),
                            dest_dir=dest, approve=True)
        assert refetched.accepted is True
        assert (clone / '.ariadne-corpus-sha').read_text().strip() == newer_sha
        assert not (clone / 'USER-LOCAL-WORK.txt').exists()  # stale corpus refreshed

        # Demand 5 — declared sha that doesn't exist upstream: loud, and
        # the half-made clone is removed (fail-closed, no debris).
        wrong = load_packfile(_packfile(
            tmp_path, repo, 'deadbeef' * 5,
        ))
        dest2 = tmp_path / 'corpus2'
        with pytest.raises(SpoolError) as excinfo:
            acquire(wrong, dest_dir=dest2, approve=True)
        assert 'sha' in str(excinfo.value).lower()
        assert not (dest2 / 'fakelib').exists()

    def test_clone_shallow_single_tag(self, tmp_path):
        # _clone_shallow fetches ONLY the pinned revision (--depth 1 at the tag),
        # never the whole history (apache/spark's history is gigabytes). A commit
        # BEFORE the tag, over file:// (local PATH clones ignore --depth), proves
        # the history was truncated to depth 1. (.git is inspected here; acquire
        # removes it afterward — see test_acquire_lifecycle demand 4.)
        repo = tmp_path / 'upstream'
        repo.mkdir()
        _git(repo, 'init', '-q')
        (repo / 'a.py').write_text('x = 1\n')
        _git(repo, 'add', '.')
        _git(repo, 'commit', '-q', '-m', 'c1')
        (repo / 'a.py').write_text('x = 2\n')
        _git(repo, 'add', '.')
        _git(repo, 'commit', '-q', '-m', 'c2')
        _git(repo, 'tag', '-a', 'v2.0', '-m', 'r')
        sha = _git(repo, 'rev-parse', 'v2.0^{}')
        work = tmp_path / 'work'
        work.mkdir()
        spool_acquire._clone_shallow(work, 'fakelib', f'file://{repo}', 'v2.0')
        clone = work / 'fakelib'
        assert (clone / '.git' / 'shallow').exists()          # truncated history
        assert _git(clone, 'rev-list', '--count', 'HEAD') == '1'
        assert _git(clone, 'rev-parse', 'HEAD') == sha
        assert (clone / 'a.py').read_text() == 'x = 2\n'

    def test_acquire_adopts_legacy_checkout(self, tmp_path, fixture_repo):
        # A clone left by an earlier (pre-marker) run — has .git, no marker — at
        # the pinned sha is ADOPTED: marker written, .git dropped, reused. This
        # reproduces the "no pin marker → refusing" bug hit on a corpus from a
        # previous run.
        repo, pinned_sha = fixture_repo
        dest = tmp_path / 'corpus'
        dest.mkdir()
        _git(dest, 'clone', '-q', str(repo), 'fakelib')
        clone = dest / 'fakelib'
        _git(clone, 'checkout', '-q', pinned_sha)
        assert (clone / '.git').exists()
        assert not (clone / '.ariadne-corpus-sha').exists()
        result = acquire(load_packfile(_packfile(tmp_path, repo, pinned_sha)),
                         dest_dir=dest, approve=True)
        assert result.accepted is True and result.cloned == ('fakelib',)
        assert not (clone / '.git').exists()                        # adopted
        assert (clone / '.ariadne-corpus-sha').read_text().strip() == pinned_sha
        assert (clone / 'engine.py').read_text() == 'VERSION = 1\n'

    def test_acquire_refetches_legacy_checkout_at_wrong_sha(
        self, tmp_path, fixture_repo,
    ):
        # A pre-marker clone (has .git, no marker) OF THIS REPO but at the wrong
        # sha is re-fetched, not refused — it's ours, just stale/interrupted.
        repo, pinned_sha = fixture_repo
        newer_sha = _git(repo, 'rev-parse', 'HEAD')      # the 2nd commit
        dest = tmp_path / 'corpus'
        dest.mkdir()
        _git(dest, 'clone', '-q', str(repo), 'fakelib')  # HEAD at the 2nd commit
        clone = dest / 'fakelib'
        assert _git(clone, 'rev-parse', 'HEAD') == newer_sha != pinned_sha
        result = acquire(load_packfile(_packfile(tmp_path, repo, pinned_sha)),
                         dest_dir=dest, approve=True)
        assert result.accepted is True and result.cloned == ('fakelib',)
        assert (clone / '.ariadne-corpus-sha').read_text().strip() == pinned_sha
        assert not (clone / '.git').exists()
        assert (clone / 'engine.py').read_text() == 'VERSION = 1\n'

    def test_acquire_refetches_stale_corpus(self, tmp_path, fixture_repo):
        # When the PIN CHANGES, our own corpus (identified by the marker) is
        # re-fetched to the new sha — not left stale and not a hard refuse. This
        # is the delta-v4.0.1→v4.0.0 case across runs.
        repo, pinned_sha = fixture_repo
        dest = tmp_path / 'corpus'
        acquire(load_packfile(_packfile(tmp_path, repo, pinned_sha)),
                dest_dir=dest, approve=True)
        clone = dest / 'fakelib'
        assert (clone / '.ariadne-corpus-sha').read_text().strip() == pinned_sha
        newer_sha = _git(repo, 'rev-parse', 'HEAD')      # the fixture's 2nd commit
        rf = acquire(load_packfile(_packfile(tmp_path, repo, newer_sha)),
                     dest_dir=dest, approve=True)
        assert rf.accepted is True and rf.cloned == ('fakelib',)
        assert (clone / '.ariadne-corpus-sha').read_text().strip() == newer_sha
        assert (clone / 'engine.py').read_text() == 'VERSION = 2\n'

    def test_acquire_replaces_unrecognized_workspace_dir(
        self, tmp_path, fixture_repo, capsys,
    ):
        # The build workspace is ariadne's own, so an unrecognised leftover dir
        # (no marker, no matching clone) is REPLACED with a loud warning — not a
        # hard refuse that wedges the build. This is the regression the user hit:
        # leftovers from earlier runs must not block `spools create`.
        repo, pinned_sha = fixture_repo
        dest = tmp_path / 'spool-corpus'
        (dest / 'fakelib').mkdir(parents=True)
        (dest / 'fakelib' / 'stray.txt').write_text('leftover')
        result = acquire(load_packfile(_packfile(tmp_path, repo, pinned_sha)),
                         dest_dir=dest, approve=True)
        assert result.accepted is True and result.cloned == ('fakelib',)
        assert 'not a recognised ariadne corpus' in capsys.readouterr().out
        clone = dest / 'fakelib'
        assert not (clone / 'stray.txt').exists()          # replaced
        assert (clone / '.ariadne-corpus-sha').read_text().strip() == pinned_sha
        assert (clone / 'engine.py').read_text() == 'VERSION = 1\n'

    def test_create_spool(self, tmp_path, fixture_repo):
        repo, pinned_sha = fixture_repo

        # create: a missing sha is resolved from the tag, SHOWN in the
        # consent prompt, and pinned back into the file (TOFU); accepted
        # consent drives the phases in order; declined runs nothing.
        spoolfile = tmp_path / 'fake-spools.yaml'
        spoolfile.write_text(textwrap.dedent(f'''
            name: fakebricks
            runtime: fake-17.3
            version: 1.0.0
            corpus:
              fakelib:
                url: {repo}
                tag: v1.0
            certify: [docs/]
        '''))
        calls = []
        phases = {
            'source_add': lambda name, path: calls.append(('source_add', name)),
            'index': lambda name: calls.append(('index', name)),
            'onboard': lambda name: calls.append(('onboard', name)),
            'build': lambda **kw: calls.append(('build', str(kw['out_path']))),
        }

        def consenting(prompt):
            calls.append(('consent', pinned_sha[:12] in prompt))
            return 'y'

        result = create_spool(
            spoolfile, dest_dir=tmp_path / 'corpus',
            out_path=tmp_path / 'pack.zip', approve=False,
            confirm=consenting, phases=phases,
        )
        assert result.accepted is True
        assert calls[0] == ('consent', True)            # resolved sha shown
        assert [c[0] for c in calls] == [
            'consent', 'source_add', 'index', 'onboard', 'build',
        ]
        assert calls[1] == ('source_add', 'fakebricks')
        assert pinned_sha in spoolfile.read_text()      # TOFU write-back
        assert (tmp_path / 'corpus' / 'fakelib' / 'engine.py').exists()

        # HIGH-4 — declined create must NOT mutate the spoolfile: the SHA
        # is shown at consent (resolved in memory) but only persisted AFTER
        # acceptance. Use a fresh, tag-only file so a write would be visible.
        fresh = tmp_path / 'fresh-spools.yaml'
        fresh.write_text(textwrap.dedent(f'''
            name: fakebricks
            runtime: fake-17.3
            version: 1.0.0
            corpus:
              fakelib:
                url: {repo}
                tag: v1.0
        '''))
        before = fresh.read_text()
        calls.clear()
        declined = create_spool(
            fresh, dest_dir=tmp_path / 'corpus2',
            out_path=tmp_path / 'pack2.zip', approve=False,
            confirm=lambda prompt: pinned_sha[:12] in prompt and 'n' or 'n',
            phases=phases,
        )
        assert declined.accepted is False
        assert calls == []                              # nothing ran
        assert not (tmp_path / 'corpus2').exists()
        assert fresh.read_text() == before              # file untouched

    def test_setup_recipe(self, tmp_path, monkeypatch):
        # Dependency ORDER is discovered up front; pin it to the recipe order
        # for the sub-tests below so the cascade is deterministic (discovery +
        # fail-soft get their own unit test, driven via this saved real ref).
        real_discover_order = spool_acquire._discover_order
        monkeypatch.setattr(spool_acquire, '_discover_order',
                            lambda environment, runtime, repos: list(repos))
        # Unified `create`'s interactive setup: the repo SET comes from the
        # environment automatically; the user specifies each version. Prompts
        # are answered by content (order-independent) via an injected seam.
        def answer(prompt):
            pl = prompt.lower()
            if 'runtime' in pl:
                return 'dbr17.3-lts'
            if 'spark' in pl:
                return 'v4.0.0'
            if 'sdk' in pl:
                return '0.44.1'
            if 'delta' in pl:
                return 'v4.0.0'
            return ''

        out = tmp_path / 'spools.yaml'
        setup_recipe('databricks', out_path=out, prompt=answer, tags_fn = lambda url: [], compat_fn = lambda *a: {})
        data = yaml.safe_load(out.read_text())
        assert data['name'] == 'databricks'
        assert data['runtime'] == 'dbr17.3-lts'
        # repo set + urls provided automatically from the built-in recipe
        assert 'apache/spark' in data['corpus']['spark']['url']
        assert data['languages'] == ['python', 'scala']
        # every version is what the user answered
        assert data['corpus']['spark']['tag'] == 'v4.0.0'
        assert data['corpus']['databricks-sdk-py']['tag'] == '0.44.1'
        assert data['corpus']['delta']['tag'] == 'v4.0.0'

        # environment omitted → "which spool?" is asked (empty answer takes
        # the sole available default). An existing recipe seeds the defaults,
        # so empty answers CONFIRM what's pinned rather than restarting.
        seeded = tmp_path / 'seeded.yaml'
        seeded.write_text(yaml.safe_dump({
            'name': 'databricks', 'runtime': 'dbr16.4-lts', 'version': '1.0.0',
            'languages': ['python'],
            'corpus': {'spark': {'url': 'https://github.com/apache/spark',
                                 'tag': 'v3.5.0'}},
        }, sort_keys=False))
        setup_recipe(out_path=seeded, prompt=lambda _p: '',
                     available=['databricks'], tags_fn = lambda url: [], compat_fn = lambda *a: {})
        kept = yaml.safe_load(seeded.read_text())
        assert kept['runtime'] == 'dbr16.4-lts'            # existing kept
        assert kept['corpus']['spark']['tag'] == 'v3.5.0'  # existing kept

        # an unknown environment now falls through to GitHub discovery; it is
        # loud only when NEITHER a repo match NOR a pasted URL is available
        # (search returns nothing, prompt gives no URL → nothing to build).
        with pytest.raises(SpoolError):
            setup_recipe('nosuchenv', out_path=tmp_path / 'x.yaml',
                         prompt=lambda _p: '', available=['databricks'],
                         tags_fn=lambda url: [], compat_fn=lambda *a: {},
                         search_fn=lambda n: [])
        # no available recipes at all -> loud, not a silent empty write
        with pytest.raises(SpoolError):
            setup_recipe(out_path=tmp_path / 'none.yaml',
                         prompt=lambda _p: '', available=[], tags_fn = lambda url: [], compat_fn = lambda *a: {})

        # an existing file with NO corpus falls back to the built-in template
        nocorpus = tmp_path / 'nocorpus.yaml'
        nocorpus.write_text('name: databricks\n')
        setup_recipe('databricks', out_path=nocorpus, prompt=lambda _p: '',
                     available=['databricks'], tags_fn = lambda url: [], compat_fn = lambda *a: {})
        assert 'apache/spark' in yaml.safe_load(
            nocorpus.read_text())['corpus']['spark']['url']
        # the "which spool" prompt is a clean example hint
        # (no raw list dump, no [default] bracket)
        prompts = []
        setup_recipe(out_path=tmp_path / 'which.yaml',
                     prompt=lambda p: prompts.append(p) or '',
                     available=['databricks'], tags_fn = lambda url: [], compat_fn = lambda *a: {})
        assert '[' not in prompts[0]
        # clean hint: names the built-in recipe(s), advertises the GitHub-name
        # fallback, and notes versions come later — no raw list dump / bracket.
        assert 'built-in' in prompts[0] and 'version' in prompts[0].lower()
        assert 'databricks' in prompts[0] and 'github' in prompts[0].lower()
        # Versions are SELECTED from each repo's COMPATIBLE tags: a numbered
        # pick resolves within the compatible list; a typed name is verbatim;
        # Enter keeps the blessed pin. compat narrows spark to the 4.0 line, so
        # the picker only ever offers v4.0.0 / v4.0.1 out of a mixed tag bag.
        picks = tmp_path / 'picks.yaml'
        picks.write_text(yaml.safe_dump({
            'name': 'databricks', 'runtime': 'r', 'version': '1.0.0',
            'languages': ['python'],
            'corpus': {'spark': {'url': 'https://x/spark', 'tag': 'v4.0.0'}},
        }, sort_keys=False))
        mixed = lambda url: ['v4.0.0', 'v4.0.1', 'v3.5.1', 'v3.4.0']
        setup_recipe('databricks', out_path=picks,
                     prompt=lambda p: '2' if p.lower().startswith('spark') else '',
                     available=['databricks'], tags_fn=mixed,
                     compat_fn=lambda *a: {'spark': '4.0'})
        assert yaml.safe_load(picks.read_text())['corpus']['spark']['tag'] == 'v4.0.1'
        setup_recipe('databricks', out_path=picks,
                     prompt=lambda p: 'v9.9.9' if p.lower().startswith('spark') else '',
                     available=['databricks'], tags_fn=mixed,
                     compat_fn=lambda *a: {'spark': '4.0'})
        assert yaml.safe_load(picks.read_text())['corpus']['spark']['tag'] == 'v9.9.9'
        # runtime is asked first (it drives compatibility), before the pickers
        order = []
        setup_recipe('databricks', out_path=tmp_path / 'ord.yaml',
                     prompt=lambda p: order.append(p) or '',
                     available=['databricks'], tags_fn=lambda url: [],
                     compat_fn=lambda *a: {})
        spark_i = next(i for i, p in enumerate(order) if p.startswith('spark'))
        runtime_i = next(i for i, p in enumerate(order) if 'Runtime' in p)
        assert runtime_i < spark_i

        # NO all-tags dump (user requirement): when compatibility can't be
        # resolved for a group, the picker offers ONLY the blessed pin (plus a
        # warning) — the unsupported v1.0.0 is NEVER shown. Enter keeps the pin.
        pin_only = tmp_path / 'pinonly.yaml'
        pin_only.write_text(yaml.safe_dump({
            'name': 'databricks', 'runtime': 'r', 'version': '1.0.0',
            'languages': ['python'],
            'corpus': {'spark': {'url': 'https://x/spark', 'tag': 'v4.0.0'}},
        }, sort_keys=False))
        pin_asked = []
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            setup_recipe('databricks', out_path=pin_only,
                         prompt=lambda p: pin_asked.append(p) or '',
                         available=['databricks'],
                         tags_fn=lambda url: ['v1.0.0', 'v2.0.0', 'v3.5.1'],
                         compat_fn=lambda *a: {})
        assert yaml.safe_load(
            pin_only.read_text())['corpus']['spark']['tag'] == 'v4.0.0'
        assert 'v1.0.0' not in buf.getvalue()
        # an UNRESOLVED (warn) group is NOT auto-selected even though it's a
        # single (pin) option — it still prompts so the escape hatch survives.
        assert any(p.lower().startswith('spark') for p in pin_asked)

        # CASCADE (steps 6-8): the chosen upstream version — not the runtime
        # alone — narrows the downstream group. delta's compatible line is
        # unlocked ONLY once spark v4.0.1 is picked; so delta is offered the
        # 4.0 tag instead of #1 of its raw list, proving the pick fed forward.
        def cascade_compat(environment, runtime, repos, chosen=None):
            chosen = chosen or {}
            lines = {'spark': '4.0'}
            if chosen.get('spark') == 'v4.0.1':
                lines['delta'] = '4.0'
            return {r: lines[r] for r in repos if r in lines}

        def cascade_tags(url):
            if 'spark' in url:
                return ['v4.0.1', 'v3.5.1']
            if 'delta' in url:
                return ['v1.2.0', 'v4.0.0']     # v1.2.0 is #1 of the raw list
            return []

        cascade_out = tmp_path / 'cascade.yaml'
        setup_recipe(
            'databricks', out_path=cascade_out, available=['databricks'],
            prompt=lambda p: ('1' if (p.lower().startswith('spark')
                                      or p.lower().startswith('delta')) else ''),
            tags_fn=cascade_tags, compat_fn=cascade_compat)
        picked = yaml.safe_load(cascade_out.read_text())['corpus']
        assert picked['spark']['tag'] == 'v4.0.1'
        # delta got v4.0.0 (the 4.0 line) — NOT v1.2.0 (#1 of its raw tags),
        # which a runtime-only / unfiltered pick would have returned.
        assert picked['delta']['tag'] == 'v4.0.0'

        # SINGLE COMPATIBLE VERSION → NO picker (the dbr17.3 case): when every
        # group narrows to one confident compatible tag, nothing is asked beyond
        # the runtime; the set is auto-chosen (the newest compatible tag, never
        # a stale 'main'/'master' pin) and merely NOTIFIED — the downstream
        # consent authorizes the build.
        asked = []
        default_out = tmp_path / 'default.yaml'
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            setup_recipe(
                'databricks', out_path=default_out, available=['databricks'],
                prompt=lambda p: asked.append(p) or '',
                tags_fn=lambda url: (['0.44.1', '0.30.0'] if 'sdk' in url
                                     else ['v4.0.0', 'v3.5.1']),
                compat_fn=lambda *a: {'spark': '4.0', 'delta': '4.0',
                                      'databricks-sdk-py': '0.44'})
        chose = yaml.safe_load(default_out.read_text())['corpus']
        assert chose['databricks-sdk-py']['tag'] == '0.44.1'   # not 'main'
        assert chose['spark']['tag'] == 'v4.0.0'
        assert chose['delta']['tag'] == 'v4.0.0'
        # nothing was asked beyond the runtime — no version picker at all
        assert not any('tag (# or name)' in p for p in asked)
        assert any('runtime' in p.lower() for p in asked)
        # the auto-chosen set was notified as the only compatible option
        assert 'compatible' in buf2.getvalue().lower()

        # A group with a REAL choice still opens the picker; the single-option
        # groups around it stay auto (no prompt).
        asked2 = []

        def mixed_tags(url):
            if 'sdk' in url:
                return ['0.44.1']
            if 'delta' in url:
                return ['v4.0.0']
            return ['v4.0.0', 'v4.0.1']          # spark: a real choice

        mixed_out = tmp_path / 'mixed.yaml'
        setup_recipe(
            'databricks', out_path=mixed_out, available=['databricks'],
            prompt=lambda p: (asked2.append(p) or
                              ('2' if p.lower().startswith('spark') else '')),
            tags_fn=mixed_tags,
            compat_fn=lambda *a: {'spark': '4.0', 'delta': '4.0',
                                  'databricks-sdk-py': '0.44'})
        gm = yaml.safe_load(mixed_out.read_text())['corpus']
        assert gm['spark']['tag'] == 'v4.0.1'                 # picked #2
        assert gm['databricks-sdk-py']['tag'] == '0.44.1'     # auto (single)
        assert gm['delta']['tag'] == 'v4.0.0'                 # auto (single)
        assert any(p.lower().startswith('spark') for p in asked2)
        assert not any(p.lower().startswith('databricks-sdk-py') for p in asked2)
        assert not any(p.lower().startswith('delta') for p in asked2)
        # compatibility is resolved by a real-time LLM query (works for any spool),
        # parsed from the model's JSON; a failed query falls back to {} (→ pin + warn)
        ask = lambda p: '{"spark": "4.0", "delta": "4.0", "databricks-sdk-py": null}'
        assert spool_acquire._query_compat(
            'databricks', 'dbr17.3-lts',
            ['spark', 'delta', 'databricks-sdk-py'], ask=ask) == {
            'spark': '4.0', 'delta': '4.0'}

        # the cascade conditions the query on the versions already chosen: they
        # are handed to the model so a downstream line depends on upstream picks.
        seen = {}

        def _cap(p):
            seen['prompt'] = p
            return '{"delta": "4.0"}'

        assert spool_acquire._query_compat(
            'databricks', 'dbr17.3-lts', ['delta'],
            {'spark': 'v4.0.0'}, ask=_cap) == {'delta': '4.0'}
        assert 'v4.0.0' in seen['prompt']

        def _boom(p):
            raise RuntimeError('offline')
        assert spool_acquire._query_compat('x', 'y', ['spark'], ask=_boom) == {}
        # _query_compat fails soft on every bad shape -> {} (all tags + warn)
        ok = lambda p: '{"spark": "4.0"}'
        assert spool_acquire._query_compat('x', 'y', [], ask=ok) == {}          # no repos
        assert spool_acquire._query_compat(
            'x', 'y', ['spark'], ask=lambda p: 'no json here') == {}            # no object
        assert spool_acquire._query_compat(
            'x', 'y', ['spark'], ask=lambda p: '{bad json}') == {}              # unparseable

        # DEPENDENCY ORDER is discovered up front (before any version prompt):
        # a JSON array from the model, validated to be a permutation of the
        # repos; a failure keeps the given order (fail-soft, never a bad root).
        order_ask = {}

        def _order_ask(p):
            order_ask['prompt'] = p
            return '["spark", "delta", "databricks-sdk-py"]'

        assert real_discover_order(
            'databricks', 'dbr17.3-lts',
            ['databricks-sdk-py', 'spark', 'delta'], ask=_order_ask) == [
            'spark', 'delta', 'databricks-sdk-py']
        assert 'spark' in order_ask['prompt']            # repos handed to model
        # a reply that isn't a permutation of the repos is rejected (fail-soft)
        assert real_discover_order(
            'x', 'y', ['spark', 'delta'],
            ask=lambda p: '["spark"]') == ['spark', 'delta']
        # unparseable / no-array keeps the given order
        assert real_discover_order(
            'x', 'y', ['spark', 'delta'], ask=lambda p: 'no array') == [
            'spark', 'delta']

        def _order_boom(p):
            raise RuntimeError('offline')

        assert real_discover_order(
            'x', 'y', ['spark', 'delta'], ask=_order_boom) == ['spark', 'delta']
        # a single repo needs no discovery — no query is made at all
        assert real_discover_order(
            'x', 'y', ['spark'], ask=lambda p: 1 / 0) == ['spark']

        # setup_recipe RESOLVES versions in the DISCOVERED order: an order_fn
        # that puts delta first makes delta's picker page come before spark's.
        seq = []
        setup_recipe(
            'databricks', out_path=tmp_path / 'ordered.yaml',
            available=['databricks'], prompt=lambda p: seq.append(p) or '',
            tags_fn=lambda url: [], compat_fn=lambda *a: {},
            order_fn=lambda e, r, repos: ['delta', 'spark', 'databricks-sdk-py'])
        delta_i = next(i for i, p in enumerate(seq) if p.startswith('delta'))
        spark_i = next(i for i, p in enumerate(seq) if p.startswith('spark'))
        assert delta_i < spark_i
    
    
    def test_repo_tags(self, monkeypatch):
        # _repo_tags parses `git ls-remote --tags` (peeled ^{} deduped,
        # newest first) and degrades to [] when git fails.
        tab = chr(9)
        rows = ['h1' + tab + 'refs/tags/v3.4.0',
                'h2' + tab + 'refs/tags/v4.0.0',
                'h3' + tab + 'refs/tags/v4.0.0^{}']
        monkeypatch.setattr(spool_acquire, '_git', lambda *a: chr(10).join(rows))
        assert spool_acquire._repo_tags('u') == ['v4.0.0', 'v3.4.0']

        def boom(*a):
            raise SpoolError('offline')
        monkeypatch.setattr(spool_acquire, '_git', boom)
        assert spool_acquire._repo_tags('u') == []
        # pre-release tags (rc / preview) are filtered out; real releases only
        monkeypatch.setattr(spool_acquire, '_git', lambda *a: chr(10).join([
            'h1' + chr(9) + 'refs/tags/v4.2.0-rc6',
            'h2' + chr(9) + 'refs/tags/v4.0.0',
            'h3' + chr(9) + 'refs/tags/v3.5.1-preview1']))
        assert spool_acquire._repo_tags('u') == ['v4.0.0']

    def test_create_cli_and_default_phases(
        self, tmp_path, fixture_repo, monkeypatch, capsys,
    ):
        from cli.main import create_parser
        import cli.spools_cmd as spools_cmd
        import spool_acquire
        from config import Config
        from library import Library
        from spool_acquire import CreateResult, _default_phases, _run_cli

        repo, pinned_sha = fixture_repo
        parser = create_parser()
        monkeypatch.chdir(tmp_path)          # create owns ./spools.yaml here

        # Unified `spools create`: interactive setup writes ./spools.yaml
        # (repo set from the env; versions from the user via input()), THEN
        # builds. The build is faked to assert what setup produced + wiring.
        seen = {}

        def fake_create(spoolfile, *, dest_dir, out_path, approve,
                        allow_ungrounded=False, allow_nonfree=False,
                        batch_mode=None, resume=False):
            seen.update(spoolfile=str(spoolfile), dest=str(dest_dir),
                        out=str(out_path), approve=approve,
                        allow_ungrounded=allow_ungrounded,
                        allow_nonfree=allow_nonfree, batch_mode=batch_mode,
                        resume=resume)
            return CreateResult(accepted=True, pack_path=str(out_path))

        monkeypatch.setattr(spool_acquire, 'create_spool', fake_create)
        monkeypatch.setattr('builtins.input',
                            lambda p: '0.44.1' if 'sdk' in p.lower() else '')
        monkeypatch.setattr(spool_acquire, '_repo_tags', lambda url: [])
        monkeypatch.setattr(spool_acquire, '_discover_order',
                            lambda e, r, repos: list(repos))
        args = parser.parse_args(['spools', 'create', 'databricks', '--batch'])
        assert spools_cmd.HANDLERS['spools'](args) == 0
        # setup wrote a pinned recipe (sdk from the answer)...
        written = yaml.safe_load((tmp_path / 'spools.yaml').read_text())
        assert written['corpus']['databricks-sdk-py']['tag'] == '0.44.1'
        # ...and the build ran with the derived out name + mode; an
        # interactive run leaves the fetch consent in place (approve False).
        assert seen['out'] == 'databricks-dbr17.3-lts.zip'
        assert seen['batch_mode'] == 'batch'
        assert seen['approve'] is False
        assert seen['allow_ungrounded'] is False
        assert seen['allow_nonfree'] is False

        # `--yes` on the now-existing recipe skips setup and approves; a mode
        # flag makes it fully unattended.
        seen.clear()
        assert spools_cmd.HANDLERS['spools'](
            parser.parse_args(['spools', 'create', '--yes', '--live'])) == 0
        assert seen['approve'] is True
        assert seen['batch_mode'] == 'live'

        # `--yes` with no recipe present is a loud error, not a build.
        (tmp_path / 'spools.yaml').unlink()
        seen.clear()
        assert spools_cmd.HANDLERS['spools'](
            parser.parse_args(['spools', 'create', '--yes'])) == 1
        assert seen == {}

        # The default phases, exercised for real where offline allows:
        # source_add persists to ariadne.yaml; build packs from the store;
        # index/onboard delegate to _run_cli (recorded).
        cfg_path = tmp_path / 'ariadne.yaml'
        cfg_path.write_text('sources: {}\n')
        cfg = Config(config_path=cfg_path)
        monkeypatch.setattr('config.get_config', lambda: cfg)
        phases = _default_phases()

        phases['source_add']('fakepack', str(repo))
        reloaded = Config(config_path=cfg_path)
        assert 'fakepack' in reloaded.sources
        added = reloaded.get_source_config('fakepack')
        assert added.path == str(repo)
        # a spool build IGNORES test code: test dirs are pruned from the walk
        # and Scala/Java test-file globs are excluded (the Python test globs are
        # already a global default), so the pack documents the library surface,
        # not the test suite.
        assert 'test' in added.exclude_dirs and 'tests' in added.exclude_dirs
        assert any(g.endswith('Suite.scala') for g in added.exclude)
        # and it resolves into the effective set the catalog walk prunes by
        assert {'test', 'tests'} <= set(reloaded.resolve_excluded_dirs('fakepack'))
        # a spool corpus is pinned/immutable, so it's staleness-exempt: the
        # SCIP index + docs are reused on every re-run, never rebuilt.
        assert added.ignore_staleness is True
        # a spool is an ISOLATED environment plugin: it must never be tied to
        # another source via dependency detection (that is how databricks once
        # picked up a `depends_on: [<consumer>]` edge). Register it with the
        # cross-source scan OFF so knowledge only ever flows environment→consumer
        # at enable time, never through a source-dependency edge.
        assert added.skip_dependency_detection is True
        assert not added.depends_on

        with Library(cfg.db_path) as lib:
            lib.add_document(
                'explanation', 'pack me', 'body',
                source_files=[str(repo / 'docs' / 'a.md')],
                source_name='fakepack',
            )
        pack_out = tmp_path / 'phase-pack.zip'
        phases['build'](
            source='fakepack', version='1.0', runtime='fake-17.3',
            certify=('docs/',), source_root=repo, out_path=pack_out,
        )
        assert pack_out.exists()

        ran = []
        monkeypatch.setattr(spool_acquire, '_run_cli',
                            lambda *argv: ran.append(argv))
        phases = _default_phases()
        phases['index']('fakepack')
        phases['onboard']('fakepack')
        # index runs --quiet (scip/pyright warnings are noise in the Spool
        # scope) and --best-effort (one language's indexer failing, e.g. spark's
        # JVM build, must not sink the whole spool) in the spool build.
        # A spool build always leaves the heavy doc types off by default
        # (opt-in in the picker); this pair rides on every onboard call.
        off = ('--doc-types-off', 'architecture,qa,diagram')
        assert ran == [
            ('discover', '--source', 'fakepack', '--quiet'),
            ('index', '--source', 'fakepack', '--quiet', '--best-effort'),
            ('onboard', '--source', 'fakepack', *off),
        ]
        # and these are real, wired flags on their commands
        idx_args = parser.parse_args(
            ['index', '--source', 'x', '--quiet', '--best-effort'])
        assert idx_args.quiet is True and idx_args.best_effort is True
        assert parser.parse_args(
            ['discover', '--source', 'x', '--quiet']).quiet is True
        assert parser.parse_args(
            ['onboard', '--doc-types-off', 'architecture,qa,diagram'],
        ).doc_types_off == 'architecture,qa,diagram'

        # batch_mode threads a mode flag into the onboard phase (the flag
        # form of the live-vs-batch toggle); default (None) passes nothing.
        ran.clear()
        _default_phases(batch_mode='batch')['onboard']('fakepack')
        assert ran == [('onboard', '--source', 'fakepack', '--batch', *off)]
        ran.clear()
        _default_phases(batch_mode='live')['onboard']('fakepack')
        assert ran == [('onboard', '--source', 'fakepack', '--live', *off)]

        # onboard_approve threads --approve into the onboard phase (the
        # unattended path); with a mode flag it is fully non-interactive.
        ran.clear()
        _default_phases(batch_mode='batch', onboard_approve=True)['onboard'](
            'fakepack')
        assert ran == [
            ('onboard', '--source', 'fakepack', '--batch', *off, '--approve'),
        ]

        # spools_model threads --model into the onboard phase so the build uses
        # that model; unset passes no --model (inherits the config model).
        ran.clear()
        _default_phases(spools_model='claude-sonnet-5')['onboard']('fakepack')
        assert ran == [
            ('onboard', '--source', 'fakepack',
             '--model', 'claude-sonnet-5', *off),
        ]

        # _run_cli itself: success (spools status, exit 0) and loud failure.
        _run_cli('spools')
        with pytest.raises(SpoolError) as excinfo:
            _run_cli('spools', 'install', str(tmp_path / 'nonexistent.zip'))
        assert 'failed' in str(excinfo.value)
        # a declined build (create_spool returns accepted=False) exits 1.
        (tmp_path / 'spools.yaml').write_text(yaml.safe_dump(
            {'name': 'databricks', 'runtime': 'r',
             'corpus': {'spark': {'url': 'u', 'tag': 't'}}}))
        monkeypatch.setattr(
            spool_acquire, 'create_spool',
            lambda *a, **k: CreateResult(accepted=False, pack_path=''))
        assert spools_cmd.HANDLERS['spools'](
            parser.parse_args(['spools', 'create', '--yes'])) == 1
        # source_add fails LOUD when the source can't be persisted.
        class _NoPersist:
            db_path = cfg.db_path
            def set_source_config(self, *a, **k):
                return False

        monkeypatch.setattr('config.get_config', lambda: _NoPersist())
        with pytest.raises(SpoolError):
            _default_phases()['source_add']('x', str(repo))

    def test_acquire_cli(self, tmp_path, fixture_repo, monkeypatch, capsys):
        # Demand 6 — `ariadne spools acquire PACKFILE --dest D --approve`.
        from cli.main import create_parser
        import cli.spools_cmd as spools_cmd

        repo, pinned_sha = fixture_repo
        packfile_path = _packfile(tmp_path, repo, pinned_sha)
        dest = tmp_path / 'cli-corpus'
        parser = create_parser()
        args = parser.parse_args([
            'spools', 'acquire', str(packfile_path),
            '--dest', str(dest), '--approve',
        ])
        exit_code = spools_cmd.HANDLERS['spools'](args)
        out = capsys.readouterr().out
        assert exit_code == 0
        assert (dest / 'fakelib' / 'engine.py').exists()
        assert 'next' in out.lower()          # prints the follow-on steps


def test_query_compat_wiring(monkeypatch, capsys):
    """Drives the REAL default path (asyncio.run + llm.chat_complete), mocking
    ONLY the network hop — not _query_compat/_default_compat_ask themselves —
    so this catches breakage the compat_fn-injecting tests can't."""
    async def _ok(messages, **kw):
        return '```json\n{"spark": "4.0", "delta": "4.0"}\n```'
    monkeypatch.setattr('llm.chat_complete', _ok)
    assert spool_acquire._query_compat(
        'databricks', 'dbr17.3-lts', ['spark', 'delta']) == {
        'spark': '4.0', 'delta': '4.0'}

    async def _boom(messages, **kw):
        raise ValueError('ANTHROPIC_API_KEY required')
    monkeypatch.setattr('llm.chat_complete', _boom)
    assert spool_acquire._query_compat('databricks', 'dbr17.3-lts', ['spark']) == {}
    out = capsys.readouterr().out.lower()
    assert 'compatibility' in out and 'anthropic_api_key' in out


def test_github_slug():
    slug = spool_acquire._github_slug
    assert slug('https://github.com/databricks/databricks-sdk-py') == 'databricks/databricks-sdk-py'
    assert slug('https://github.com/apache/spark.git') == 'apache/spark'
    assert slug('https://github.com/apache/spark/') == 'apache/spark'
    assert slug('http://github.com/o/r') == 'o/r'
    # non-github (local fixture urls, ssh, other hosts) → no slug, no API
    assert slug('file:///tmp/repo') is None
    assert slug('/local/path/repo') is None
    assert slug('https://gitlab.com/o/r') is None


def test_remote_file_count():
    count = spool_acquire._remote_file_count
    tree = {'truncated': False, 'tree': [
        {'type': 'blob'}, {'type': 'tree'}, {'type': 'blob'}, {'type': 'blob'}]}
    seen = {}

    def fetch(slug, sha):
        seen['slug'], seen['sha'] = slug, sha
        return tree

    # blobs are files; trees (directories) are not
    assert count('https://github.com/o/r', 'deadbeef', fetch=fetch) == 3
    assert seen == {'slug': 'o/r', 'sha': 'deadbeef'}
    # non-github → 0 and NO fetch attempted (the API is GitHub-only)
    assert count('file:///x', 'sha', fetch=lambda *a: 1 / 0) == 0
    # a truncated tree can't be trusted for a determinate total → 0
    assert count('https://github.com/o/r', 's',
                 fetch=lambda *a: {'truncated': True, 'tree': [{'type': 'blob'}]}) == 0

    # any API failure fails soft to 0 (the fetch still proceeds, just no bar total)
    def boom(slug, sha):
        raise RuntimeError('offline')
    assert count('https://github.com/o/r', 's', fetch=boom) == 0


def test_count_files(tmp_path):
    (tmp_path / 'a.py').write_text('x')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'b.py').write_text('y')
    (tmp_path / 'sub' / 'c.txt').write_text('z')
    assert spool_acquire._count_files(tmp_path) == 3
    assert spool_acquire._count_files(tmp_path / 'missing') == 0


def test_acquire_counts_files_before_clone(tmp_path, fixture_repo, monkeypatch):
    """The progress bar's total is the per-repo file count fetched BEFORE the
    clone: assert the count→clone ordering and that the count feeds the clone's
    progress total."""
    repo, pinned_sha = fixture_repo
    events = []

    def fake_count(url, sha):
        events.append('count')
        return 7

    monkeypatch.setattr(spool_acquire, '_remote_file_count', fake_count)
    real_clone = spool_acquire._clone_shallow

    def spy_clone(dest, name, url, tag, *, on_files=None, total=0):
        events.append(('clone', total))
        return real_clone(dest, name, url, tag, on_files=on_files, total=total)

    monkeypatch.setattr(spool_acquire, '_clone_shallow', spy_clone)
    result = acquire(load_packfile(_packfile(tmp_path, repo, pinned_sha)),
                     dest_dir=tmp_path / 'corpus', approve=True)
    assert result.accepted is True
    # count is fetched BEFORE the clone, and its value becomes the clone's total
    assert events == ['count', ('clone', 7)]


def test_setup_recipe_placeholder_pin_falls_back_to_real_tags(tmp_path):
    """When the compat query can't resolve a repo (the real case: the SDK's
    releases aren't tied to the DBR runtime, so the live LLM returns null for
    it) AND its blessed pin is a moving branch (the template's 'main'/'master'
    placeholders), the picker must offer the repo's real release TAGS — never
    pin the branch. A spool is a VERSIONED pack; 'main' is not a reproducible
    pin. Even hitting Enter takes the newest real release, not the placeholder."""
    import yaml
    out = tmp_path / 'ph.yaml'

    def tags_fn(url):
        if 'sdk' in url:
            return ['0.121.0', '0.120.0']         # real releases, newest-first
        return ['v4.0.0']

    # compat resolves spark + delta but NOT the sdk — the exact live-LLM
    # behaviour ({"databricks-sdk-py": null}) that produced `tag: main`.
    setup_recipe(
        'databricks', out_path=out, available=['databricks'],
        prompt=lambda p: '',                       # Enter everywhere
        tags_fn=tags_fn,
        compat_fn=lambda *a, **k: {'spark': '4.0', 'delta': '4.0'},
        order_fn=lambda *a, **k: ['spark', 'databricks-sdk-py', 'delta'])
    corpus = yaml.safe_load(out.read_text())['corpus']
    assert corpus['databricks-sdk-py']['tag'] == '0.121.0'   # newest real tag
    assert corpus['databricks-sdk-py']['tag'] != 'main'      # NEVER the branch
