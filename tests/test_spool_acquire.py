"""Slice (b3) of the Spool plugin: builder acquisition + consent-before-fetch.

USER REQUIREMENT (2026-07-08): before ANY acquisition, show exactly what
will be fetched — repos at pinned tags with SHAs — and ask to proceed
(`--approve` skips for CI). Clones are verified at the declared SHA and
a mismatch fails loud with the clone removed. Local git fixture repos
only — no network. Design: IMPLEMENT.md (b3).
"""
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

from spool_acquire import (
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
        # Eligible = a language with a registered SCIP adapter
        # (python / typescript / jvm-family), derived from the SCIP
        # registry so a future scip-go entry flips Go automatically.
        for ok in ('python', 'py', 'typescript', 'ts', 'javascript', 'js',
                   'scala', 'java', 'kotlin', 'jvm', '.py', '.scala'):
            assert is_scip_eligible(ok), ok
        for bad in ('go', 'golang', 'ruby', 'rust', 'c', 'cpp', '', None):
            assert not is_scip_eligible(bad), bad

    def test_corpus_scan_flags_unsupported_only_when_ungrounded(self, tmp_path):
        go_only = tmp_path / 'go_only'
        (go_only / 'pkg').mkdir(parents=True)
        (go_only / 'pkg' / 'main.go').write_text('package main\n')
        (go_only / 'README.md').write_text('# x\n')
        assert unsupported_corpus_language(go_only) == 'Go'
        # any supported source present → grounded → gate does not fire
        mixed = tmp_path / 'mixed'
        mixed.mkdir()
        (mixed / 'main.go').write_text('package main\n')
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
        spoolfile = tmp_path / 'go-spools.yaml'
        spoolfile.write_text(textwrap.dedent(f'''
            name: gobricks
            runtime: go-1.x
            version: 1.0.0
            languages: [go]
            corpus:
              gocore:
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
        assert 'go' in msg
        assert 'not supported' in msg or "can't" in msg or 'cannot' in msg
        # Refused BEFORE any fetch / consent / phase — no clone, no cost.
        assert calls == []
        assert not (tmp_path / 'corpus').exists()

    def test_create_backstop_aborts_on_ungrounded_corpus(self, tmp_path):
        # Undeclared case: recipe names no languages, but the cloned corpus
        # turns out to be Go — the post-clone backstop refuses BEFORE the
        # paid onboard step (never falls through to a hollow pack).
        repo, _sha = _git_repo_with(tmp_path, 'goupstream', 'pkg/main.go',
                                    'package main\n')
        spoolfile = tmp_path / 'undeclared-spools.yaml'
        spoolfile.write_text(textwrap.dedent(f'''
            name: gobricks
            runtime: go-1.x
            version: 1.0.0
            corpus:
              gocore:
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
        assert 'go' in str(excinfo.value).lower()
        # Clone may have happened (undeclared → can't know pre-clone), but no
        # phase ran — the paid onboard was never reached.
        assert calls == []

    def test_allow_ungrounded_bypasses_language_gate(
        self, tmp_path, fixture_repo,
    ):
        repo, _sha = fixture_repo
        spoolfile = tmp_path / 'go2-spools.yaml'
        spoolfile.write_text(textwrap.dedent(f'''
            name: gobricks
            runtime: go-1.x
            version: 1.0.0
            languages: [go]
            corpus:
              gocore:
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
        result = create_spool(
            spoolfile, dest_dir=tmp_path / 'corpus2',
            out_path=tmp_path / 'pack2.zip', approve=True,
            confirm=lambda p: 'y', phases=phases, allow_ungrounded=True,
        )
        assert result.accepted is True
        assert calls == ['source_add', 'index', 'onboard', 'build']


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
        assert _git(clone, 'rev-parse', 'HEAD') == pinned_sha
        assert (clone / 'engine.py').read_text() == 'VERSION = 1\n'

        # Demand 4b (CRIT-2) — re-run is idempotent: an existing checkout at
        # the pinned sha is REUSED, and anything inside it survives.
        (clone / 'USER-LOCAL-WORK.txt').write_text('precious')
        rerun = acquire(packfile, dest_dir=dest, approve=True)
        assert rerun.accepted is True and rerun.cloned == ('fakelib',)
        assert (clone / 'USER-LOCAL-WORK.txt').read_text() == 'precious'

        # Demand 4c (CRIT-2) — an existing checkout at a DIFFERENT sha is
        # refused loudly and NEVER deleted.
        _git(clone, 'checkout', '-q', 'v1.0~0')  # stay detached
        _git(repo, 'rev-parse', 'HEAD')
        other = _packfile(tmp_path, repo, _git(repo, 'rev-parse', 'HEAD'))
        mismatch_pack = load_packfile(other)
        with pytest.raises(SpoolError) as excinfo:
            acquire(mismatch_pack, dest_dir=dest, approve=True)
        assert 'refusing' in str(excinfo.value)
        assert clone.exists()
        assert (clone / 'USER-LOCAL-WORK.txt').exists()  # nothing deleted

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

    def test_setup_recipe(self, tmp_path):
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
        setup_recipe('databricks', out_path=out, prompt=answer)
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
                     available=['databricks'])
        kept = yaml.safe_load(seeded.read_text())
        assert kept['runtime'] == 'dbr16.4-lts'            # existing kept
        assert kept['corpus']['spark']['tag'] == 'v3.5.0'  # existing kept

        # an unknown environment is loud
        with pytest.raises(SpoolError):
            setup_recipe('nosuchenv', out_path=tmp_path / 'x.yaml',
                         prompt=lambda _p: '', available=['databricks'])
        # no available recipes at all -> loud, not a silent empty write
        with pytest.raises(SpoolError):
            setup_recipe(out_path=tmp_path / 'none.yaml',
                         prompt=lambda _p: '', available=[])

        # an existing file with NO corpus falls back to the built-in template
        nocorpus = tmp_path / 'nocorpus.yaml'
        nocorpus.write_text('name: databricks\n')
        setup_recipe('databricks', out_path=nocorpus, prompt=lambda _p: '',
                     available=['databricks'])
        assert 'apache/spark' in yaml.safe_load(
            nocorpus.read_text())['corpus']['spark']['url']

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
                        allow_ungrounded=False, batch_mode=None):
            seen.update(spoolfile=str(spoolfile), dest=str(dest_dir),
                        out=str(out_path), approve=approve,
                        allow_ungrounded=allow_ungrounded, batch_mode=batch_mode)
            return CreateResult(accepted=True, pack_path=str(out_path))

        monkeypatch.setattr(spool_acquire, 'create_spool', fake_create)
        monkeypatch.setattr('builtins.input',
                            lambda p: '0.44.1' if 'sdk' in p.lower() else '')
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
        assert 'fakepack' in Config(config_path=cfg_path).sources

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
        assert ran == [
            ('discover', '--source', 'fakepack'),
            ('index', '--source', 'fakepack'),
            ('onboard', '--source', 'fakepack'),
        ]

        # batch_mode threads a mode flag into the onboard phase (the flag
        # form of the live-vs-batch toggle); default (None) passes nothing.
        ran.clear()
        _default_phases(batch_mode='batch')['onboard']('fakepack')
        assert ran == [('onboard', '--source', 'fakepack', '--batch')]
        ran.clear()
        _default_phases(batch_mode='live')['onboard']('fakepack')
        assert ran == [('onboard', '--source', 'fakepack', '--live')]

        # onboard_approve threads --approve into the onboard phase (the
        # unattended path); with a mode flag it is fully non-interactive.
        ran.clear()
        _default_phases(batch_mode='batch', onboard_approve=True)['onboard'](
            'fakepack')
        assert ran == [
            ('onboard', '--source', 'fakepack', '--batch', '--approve'),
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
