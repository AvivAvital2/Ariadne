"""Spool pack container tests: build → install.

Evolutionary test per IMPLEMENT.md; synthetic fixtures only. A pack is ONE
zip = manifest.yaml (checksum of the DB inside) + pack.db — a genuine
Ariadne library DB carrying the spool source's docs WITH their embeddings,
so install never re-embeds.
Design: designs/spool-environment-plugin.md §9 · §17 · §18.6.1/.2.
"""
import hashlib
import sqlite3
import textwrap
import zipfile

import numpy as np
import pytest
import yaml

from config import Config
from library import Library, ScopedLibrary
from library.scip import init_scip_schema
from spool_pack import build_pack, install_pack
from spools import SpoolError, SpoolManifest, resolve_spools


def _vec(*values):
    return np.array(values, dtype=np.float32)


def _config(tmp_path, yaml_body):
    path = tmp_path / 'ariadne.yaml'
    path.write_text(textwrap.dedent(yaml_body))
    return Config(config_path=path)


class TestSpoolPack:
    def test_pack_lifecycle(self, tmp_path, monkeypatch):
        # This pack carries 2-dim fixture embeddings; make the consumer's
        # embedding identity match so the CRIT-10 compat check passes.
        import embedding
        monkeypatch.setattr(embedding, 'EMBEDDING_DIM', 2)

        source_root = tmp_path / 'fakebricks-repo'
        (source_root / 'docs').mkdir(parents=True)
        # A corpus clone under the build root, marked like a real fetch, so the
        # pack captures its upstream LICENSE/NOTICE for attribution (Demand 5).
        corpus_repo = source_root / 'corpuslib'
        corpus_repo.mkdir()
        (corpus_repo / '.ariadne-corpus-sha').write_text('abc123def456\n')
        (corpus_repo / 'LICENSE').write_text(
            'Mozilla Public License Version 2.0\n\n1. Definitions\n...\n')
        (corpus_repo / 'NOTICE').write_text(
            'corpuslib\nCopyright (c) 2026 The corpuslib Authors\n')

        with Library(tmp_path / 'builder.db') as builder:
            engine_doc = builder.add_document(
                'explanation', 'engine doc', 'code-derived body',
                source_files=[str(source_root / 'src' / 'engine.py')],
                embedding=_vec(1.0, 0.0),
                source_name='fakebricks',
            )
            guide_doc = builder.add_document(
                'explanation', 'guide doc', 'guide body',
                source_files=[str(source_root / 'docs' / 'guide.md')],
                embedding=_vec(0.0, 1.0),
                metadata={'provenance': 'human-doc'},
                source_name='fakebricks',
            )
            readme_doc = builder.add_document(
                'explanation', 'readme doc', 'readme body',
                source_files=[str(source_root / 'README.md')],
                embedding=_vec(0.6, 0.8),
                metadata={'provenance': 'human-doc'},
                source_name='fakebricks',
            )
            builder.add_document(
                'explanation', 'other doc', 'not in the pack',
                source_name='src1',
            )
            # HIGH-2 — sections travel with the pack (§9): seed one.
            from schema import Section
            builder.store_sections(guide_doc.id, [
                Section(document_id=guide_doc.id, heading='Overview',
                        description='d', content='overview body', index=0),
            ])

            # Demand 1 — build: one zip; pack.db holds exactly the spool
            # source's docs with embeddings; checksum stamped; an empty
            # source refuses loudly.
            pack_path = tmp_path / 'fakebricks-pack.zip'
            manifest = build_pack(
                builder,
                environment='fakebricks',
                version='1.0.0',
                target_runtime='fake-17.3',
                certified_docs=('docs/',),
                source_root=source_root,
                out_path=pack_path,
            )
            assert pack_path.exists()
            with zipfile.ZipFile(pack_path) as zf:
                assert set(zf.namelist()) == {
                    'manifest.yaml', 'pack.db',
                    'licenses/corpuslib/LICENSE', 'licenses/corpuslib/NOTICE',
                }
                db_blob = zf.read('pack.db')
                packed_manifest = SpoolManifest.from_dict(
                    yaml.safe_load(zf.read('manifest.yaml')),
                )
            assert packed_manifest == manifest
            assert manifest.checksum == (
                'sha256:' + hashlib.sha256(db_blob).hexdigest()
            )

            # Demand 5 — attribution: the corpus clone's LICENSE/NOTICE are
            # bundled and recorded with a per-file sha256, so open-source
            # provenance travels with the pack.
            assert len(manifest.attribution) == 1
            attr = manifest.attribution[0]
            assert attr['repo'] == 'corpuslib'
            assert attr['sha'] == 'abc123def456'
            with zipfile.ZipFile(pack_path) as zf:
                lic_bytes = zf.read('licenses/corpuslib/LICENSE')
            recorded = {f['name']: f['sha256'] for f in attr['files']}
            assert set(recorded) == {'LICENSE', 'NOTICE'}
            assert recorded['LICENSE'] == (
                'sha256:' + hashlib.sha256(lic_bytes).hexdigest()
            )

            extracted_db = tmp_path / 'extracted-pack.db'
            extracted_db.write_bytes(db_blob)
            with Library(extracted_db) as pack:
                packed = {doc.id: doc for doc in pack.list_documents()}
                packed_sections = pack.get_sections(guide_doc.id)
            assert set(packed) == {engine_doc.id, guide_doc.id, readme_doc.id}
            assert np.array_equal(
                packed[engine_doc.id].embedding, engine_doc.embedding,
            )
            # HIGH-2 — the section round-tripped into the pack.
            assert [s.heading for s in packed_sections] == ['Overview']

            with pytest.raises(SpoolError) as excinfo:
                build_pack(
                    builder,
                    environment='ghost',
                    version='1.0.0',
                    target_runtime='fake-17.3',
                    certified_docs=(),
                    source_root=source_root,
                    out_path=tmp_path / 'ghost.zip',
                )
            assert 'no documents' in str(excinfo.value)

            # Demand 2 — certified tagging: the doc under docs/ leaves the
            # build as `official`; uncertified prose stays `human-doc`; the
            # code doc keeps no provenance; the BUILDER store is unmutated.
            assert packed[guide_doc.id].metadata['provenance'] == 'official'
            assert packed[readme_doc.id].metadata['provenance'] == 'human-doc'
            assert 'provenance' not in packed[engine_doc.id].metadata
            builder_docs = {d.id: d for d in builder.list_documents()}
            assert builder_docs[guide_doc.id].metadata['provenance'] == 'human-doc'

        # Demand 3 — install: same ids, byte-identical embeddings (no
        # re-embed), manifest cached so resolve_spools registers; re-install
        # is idempotent.
        install_root = tmp_path / 'consumer'
        install_root.mkdir()
        cache_dir = install_root / 'spool-cache'
        with Library(install_root / 'target.db') as target:
            installed = install_pack(target, pack_path, cache_dir=cache_dir)
            assert installed == manifest
            docs = {d.id: d for d in target.list_documents()}
            assert set(docs) == {engine_doc.id, guide_doc.id, readme_doc.id}
            assert np.array_equal(
                docs[guide_doc.id].embedding, guide_doc.embedding,
            )
            assert docs[guide_doc.id].metadata['provenance'] == 'official'

            # Demand 5 (install) — attribution files travel to the cache dir,
            # under the same per-environment folder as manifest.yaml/pack.db.
            lic_cache = cache_dir / 'fakebricks' / 'licenses' / 'corpuslib'
            assert (lic_cache / 'LICENSE').exists()
            assert (lic_cache / 'NOTICE').exists()

            cfg = _config(install_root, '''
                spools:
                  fakebricks:
                    runtime: fake-17.3
            ''')
            resolution = resolve_spools(cfg, cache_dir=cache_dir)
            assert resolution.gaps == ()
            assert resolution.scope_sources() == frozenset({'spool:fakebricks'})

            install_pack(target, pack_path, cache_dir=cache_dir)
            assert len(target.list_documents()) == 3

        # Demand 4 — checksum gate: a tampered pack.db is refused loudly,
        # BEFORE anything lands in the store or the cache.
        tampered_path = tmp_path / 'tampered-pack.zip'
        with zipfile.ZipFile(pack_path) as zf:
            manifest_bytes = zf.read('manifest.yaml')
            db_bytes = zf.read('pack.db')
        with zipfile.ZipFile(tampered_path, 'w') as zf:
            zf.writestr('manifest.yaml', manifest_bytes)
            zf.writestr('pack.db', db_bytes + b'tampered')
        victim_cache = tmp_path / 'victim-cache'
        with Library(tmp_path / 'victim.db') as victim:
            with pytest.raises(SpoolError) as excinfo:
                install_pack(victim, tampered_path, cache_dir=victim_cache)
            assert 'checksum' in str(excinfo.value)
            assert victim.list_documents() == []
        assert not victim_cache.exists()

        # Demand (b2+) — pack format version: stamped at build; a pack
        # declaring an unknown future format is refused loudly BEFORE any
        # write (no reliance on implicit schema migrations).
        assert manifest.pack_format == 1
        packed_yaml = yaml.safe_load(manifest_bytes)
        assert packed_yaml['pack_format'] == 1
        packed_yaml['pack_format'] = 99
        future_path = tmp_path / 'future-pack.zip'
        with zipfile.ZipFile(future_path, 'w') as zf:
            zf.writestr('manifest.yaml', yaml.safe_dump(packed_yaml))
            zf.writestr('pack.db', db_bytes)
        future_cache = tmp_path / 'future-cache'
        with Library(tmp_path / 'future-victim.db') as victim:
            with pytest.raises(SpoolError) as excinfo:
                install_pack(victim, future_path, cache_dir=future_cache)
            assert 'pack format' in str(excinfo.value)
            assert victim.list_documents() == []
        assert not future_cache.exists()

        # HIGH-2 — sections install into the consumer store too.
        install_root = tmp_path / 'sections-consumer'
        install_root.mkdir()
        with Library(install_root / 'target.db') as target:
            install_pack(target, pack_path, cache_dir=install_root / 'c')
            assert [s.heading for s in target.get_sections(guide_doc.id)] == [
                'Overview',
            ]

        # Demand 5 (integrity) — a tampered LICENSE member is refused by its
        # recorded per-file sha256, BEFORE any store/cache write (fail-closed,
        # the same guarantee the pack.db checksum gives).
        tampered_lic = tmp_path / 'tampered-lic.zip'
        with zipfile.ZipFile(pack_path) as zf:
            members = {n: zf.read(n) for n in zf.namelist()}
        members['licenses/corpuslib/LICENSE'] = b'FORGED'
        with zipfile.ZipFile(tampered_lic, 'w') as zf:
            for name, blob in members.items():
                zf.writestr(name, blob)
        lic_victim_cache = tmp_path / 'lic-victim-cache'
        with Library(tmp_path / 'lic-victim.db') as victim:
            with pytest.raises(SpoolError) as excinfo:
                install_pack(victim, tampered_lic, cache_dir=lic_victim_cache)
            assert 'license' in str(excinfo.value).lower()
            assert victim.list_documents() == []
        assert not lic_victim_cache.exists()

    def test_install_namespaces_spool_docs_no_collision(self, tmp_path):
        # CRIT-9 — a spool whose name collides with a real source must NOT
        # pollute it. Pack docs land under a reserved 'spool:<name>' source
        # id, so the real source is untouched, disable removes them from
        # scope, and uninstall reclaims them — all without ever sharing rows.
        from spools import (
            SpoolSetting, enabled_spools, resolve_spools, spool_source_id,
        )
        from spool_pack import uninstall_pack

        root = tmp_path / 'repo'
        (root / 'docs').mkdir(parents=True)
        with Library(tmp_path / 'store.db') as store:
            # user's OWN real source named 'databricks'
            user_doc = store.add_document(
                'explanation', 'my notebook', 'user body',
                source_name='databricks',
            )
            # a spool pack whose environment is ALSO 'databricks'
            with Library(tmp_path / 'builder.db') as builder:
                builder.add_document(
                    'explanation', 'spark internals', 'pack body',
                    source_files=[str(root / 'docs' / 'a.md')],
                    source_name='databricks',
                )
                build_pack(
                    builder, environment='databricks', version='1.0',
                    target_runtime='fake-17.3', certified_docs=(),
                    source_root=root, out_path=tmp_path / 'pack.zip',
                )
            cache = tmp_path / 'cache'
            install_pack(store, tmp_path / 'pack.zip', cache_dir=cache)

            ns = spool_source_id('databricks')
            assert ns == 'spool:databricks'
            real = [d.title for d in store.list_documents()
                    if d.source_name == 'databricks']
            packed = [d.title for d in store.list_documents()
                      if d.source_name == ns]
            assert real == ['my notebook']           # real source UNPOLLUTED
            assert packed == ['spark internals']      # pack under spool:*

            cfg = _config(tmp_path, '''
                spools:
                  databricks:
                    runtime: fake-17.3
            ''')
            resolution = resolve_spools(cfg, cache_dir=cache)
            assert 'databricks' in resolution.registered   # user-facing name
            assert resolution.scope_sources() == frozenset({ns})  # namespaced

            # disable = drop from config -> pack docs leave scope (rows
            # remain but are unreachable by any query).
            cfg_off = _config(tmp_path, 'sources: {}\n')
            assert resolve_spools(cfg_off, cache_dir=cache).scope_sources() \
                == frozenset()

            # uninstall = physically remove exactly the spool:* rows; the
            # user's real source is left intact, cache gone.
            uninstall_pack(store, 'databricks', cache_dir=cache)
            assert [d.title for d in store.list_documents()
                    if d.source_name == ns] == []
            assert [d.title for d in store.list_documents()
                    if d.source_name == 'databricks'] == ['my notebook']
            assert not (cache / 'databricks').exists()
            assert user_doc.id  # (silence unused)

    def test_install_reembeds_on_model_mismatch(self, tmp_path, monkeypatch):
        # CRIT-10 (revised) — a pack's embeddings are only meaningful under
        # the model that produced them. When the consumer's embedding model
        # MATCHES, the shipped vectors are used verbatim (the perf win, no
        # re-embed). When it DIFFERS, install RE-EMBEDS the docs' content
        # with the consumer's model rather than refusing — so a pack is
        # portable across models and the dim-mismatch crash can't happen.
        import embedding
        root = tmp_path / 'repo'
        (root / 'docs').mkdir(parents=True)
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'explanation', 'spark doc', 'spark body',
                source_files=[str(root / 'docs' / 'a.md')],
                embedding=_vec(0.6, 0.8),  # 2-dim, model = builder default
                source_name='databricks',
            )
            manifest = build_pack(
                builder, environment='databricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        assert manifest.embedding_dim == 2
        assert manifest.embedding_model == embedding.DEFAULT_MODEL

        reembed_calls: list[list[str]] = []

        def fake_batch(texts, config=None):
            reembed_calls.append(list(texts))
            return [_vec(0.1, 0.2, 0.3) for _ in texts]  # consumer dim = 3

        monkeypatch.setattr(embedding, 'embed_batch_sync', fake_batch)

        # (1) MATCH (dim + model) -> shipped vectors used verbatim, no re-embed
        monkeypatch.setattr(embedding, 'EMBEDDING_DIM', 2)
        with Library(tmp_path / 'c_match.db') as target:
            install_pack(target, tmp_path / 'pack.zip',
                         cache_dir=tmp_path / 'cache_match')
            (doc,) = target.list_documents()
            assert np.array_equal(doc.embedding, _vec(0.6, 0.8))  # shipped
        assert reembed_calls == []                                 # not re-embedded

        # (2) DIMENSION mismatch -> re-embed with consumer model (no refuse)
        monkeypatch.setattr(embedding, 'EMBEDDING_DIM', 3)
        with Library(tmp_path / 'c_dim.db') as target:
            install_pack(target, tmp_path / 'pack.zip',
                         cache_dir=tmp_path / 'cache_dim')
            (doc,) = target.list_documents()
            assert np.array_equal(doc.embedding, _vec(0.1, 0.2, 0.3))  # re-embedded
        # Cycle-1 fix: the re-embed input must be the CANONICAL doc text
        # (title + truncated content), identical to what writer.py embeds —
        # not content alone, or spool vectors rank inconsistently vs native.
        assert reembed_calls[-1][0] == embedding.doc_embedding_text(
            'spark doc', 'spark body',
        )
        assert reembed_calls[-1][0].startswith('spark doc')  # title included

        # (3) same dim, DIFFERENT model -> also re-embed (silent-garbage guard)
        reembed_calls.clear()
        monkeypatch.setattr(embedding, 'EMBEDDING_DIM', 2)
        monkeypatch.setattr(embedding, 'DEFAULT_MODEL', 'some-other-embed-model')

        def fake_batch2(texts, config=None):
            reembed_calls.append(list(texts))
            return [_vec(0.7, 0.7) for _ in texts]  # consumer dim = 2, new model

        monkeypatch.setattr(embedding, 'embed_batch_sync', fake_batch2)
        with Library(tmp_path / 'c_model.db') as target:
            install_pack(target, tmp_path / 'pack.zip',
                         cache_dir=tmp_path / 'cache_model')
            (doc,) = target.list_documents()
            assert np.array_equal(doc.embedding, _vec(0.7, 0.7))  # re-embedded
        assert reembed_calls  # model mismatch triggered a re-embed

    def test_reembed_guards(self, tmp_path, monkeypatch):
        # Cycle-1 minors — the re-embed fallback must fail LOUD (SpoolError,
        # clean rollback), never silently misalign or leak a raw error:
        #  - a batch returning the wrong count would misalign zip() -> refuse
        #  - a missing API key (raw ValueError) -> wrapped as SpoolError
        import embedding
        root = tmp_path / 'repo'
        (root / 'docs').mkdir(parents=True)
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'explanation', 'd', 'body',
                source_files=[str(root / 'docs' / 'a.md')],
                embedding=_vec(0.6, 0.8), source_name='databricks',
            )
            build_pack(
                builder, environment='databricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        monkeypatch.setattr(embedding, 'EMBEDDING_DIM', 3)  # force re-embed

        # wrong count -> loud, store untouched
        monkeypatch.setattr(embedding, 'embed_batch_sync',
                            lambda texts, config=None: [])
        with Library(tmp_path / 'c1.db') as target:
            with pytest.raises(SpoolError) as e:
                install_pack(target, tmp_path / 'pack.zip', cache_dir=tmp_path / 'k1')
            assert 're-embed' in str(e.value).lower()
            assert target.list_documents() == []

        # raw embed error (e.g. no API key) -> wrapped, store untouched
        def boom(texts, config=None):
            raise ValueError('No API key provided')
        monkeypatch.setattr(embedding, 'embed_batch_sync', boom)
        with Library(tmp_path / 'c2.db') as target:
            with pytest.raises(SpoolError) as e:
                install_pack(target, tmp_path / 'pack.zip', cache_dir=tmp_path / 'k2')
            assert 'embed' in str(e.value).lower()
            assert target.list_documents() == []

    def test_install_rolls_back_on_failure(self, tmp_path):
        # CRIT-4 — an install that crashes mid-copy leaves the store as it
        # was AND leaves the spool UNREGISTERED (honest "not installed"),
        # never a partial store that reports healthy.
        root = tmp_path / 'repo'
        (root / 'docs').mkdir(parents=True)
        with Library(tmp_path / 'builder.db') as builder:
            for i in range(3):
                builder.add_document(
                    'explanation', f'doc{i}', 'body',
                    source_files=[str(root / 'docs' / f'{i}.md')],
                    source_name='fakebricks',
                )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )

        class Flaky(Library):
            n = 0

            def add_document(self, *a, **k):
                Flaky.n += 1
                if Flaky.n == 3:
                    raise RuntimeError('ctrl-c / disk full mid-install')
                return super().add_document(*a, **k)

        cache = tmp_path / 'cache'
        with Flaky(tmp_path / 'target.db') as target:
            with pytest.raises(RuntimeError):
                install_pack(target, tmp_path / 'pack.zip', cache_dir=cache)
            # store restored — no partial fakebricks docs left behind
            assert [d for d in target.list_documents()
                    if d.source_name == 'spool:fakebricks'] == []
        # cache not written -> resolve sees missing-pack, NOT registered
        assert not (cache / 'fakebricks' / 'manifest.yaml').exists()
        cfg = _config(tmp_path, 'spools:\n  fakebricks: true\n')
        resolution = resolve_spools(cfg, cache_dir=cache)
        assert resolution.registered == {}
        assert resolution.gaps[0].reason == 'missing-pack'

    def test_install_bounds_untrusted_inputs(self, tmp_path):
        # CRIT-7 — install must not materialize an unbounded entry. A
        # pack.db that streams past the byte cap is refused BEFORE the
        # store or cache is touched; an oversized manifest likewise.
        root = tmp_path / 'repo'
        (root / 'docs').mkdir(parents=True)
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'explanation', 'doc', 'body',
                source_files=[str(root / 'docs' / 'a.md')],
                source_name='fakebricks',
            )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        # A legitimate pack still installs when within the cap (streaming
        # preserves the round-trip + checksum).
        cache = tmp_path / 'cache'
        with Library(tmp_path / 'ok.db') as target:
            install_pack(target, tmp_path / 'pack.zip', cache_dir=cache,
                         max_pack_bytes=10 * 1024 * 1024)
            assert len(target.list_documents()) == 1

        # Oversized pack.db (streamed) -> refused loudly, nothing written.
        big = tmp_path / 'bomb.zip'
        with zipfile.ZipFile(big, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('manifest.yaml',
                        'environment: x\nversion: 1\ntarget_runtime: r\n'
                        'checksum: sha256:deadbeef\n')
            zf.writestr('pack.db', b'\0' * (2 * 1024 * 1024))  # 2 MB
        bomb_cache = tmp_path / 'bomb-cache'
        with Library(tmp_path / 'victim.db') as victim:
            with pytest.raises(SpoolError) as excinfo:
                install_pack(victim, big, cache_dir=bomb_cache,
                             max_pack_bytes=64 * 1024)  # 64 KB cap
            assert 'too large' in str(excinfo.value).lower()
            assert victim.list_documents() == []
        assert not bomb_cache.exists()

        # Oversized manifest -> refused before it is even parsed.
        big_manifest = tmp_path / 'bigman.zip'
        with zipfile.ZipFile(big_manifest, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('manifest.yaml', 'x: ' + 'a' * (2 * 1024 * 1024))
            zf.writestr('pack.db', b'')
        with Library(tmp_path / 'victim2.db') as victim:
            with pytest.raises(SpoolError) as excinfo:
                install_pack(victim, big_manifest,
                             cache_dir=tmp_path / 'bm-cache',
                             max_manifest_bytes=64 * 1024)
            assert 'manifest' in str(excinfo.value).lower()

    def test_install_rejects_path_traversal_environment(self, tmp_path):
        # HIGH-1 — manifest.environment is attacker-controlled and is NOT
        # part of the checksum (which covers pack.db only). It flows into
        # the cache path (Path(cache)/environment) and the spool:<env>
        # source id, so a traversal/absolute value is an arbitrary-write +
        # a namespace escape. It must be refused at manifest parse, before
        # any store or filesystem write.
        root = tmp_path / 'repo'
        root.mkdir()
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'explanation', 'doc', 'body',
                source_files=[str(root / 'a.md')], source_name='fakebricks',
            )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        # Tamper ONLY the environment field; pack.db (hence the checksum) is
        # untouched, so the tampered pack still passes checksum verification
        # — this is the actual attack, not a corrupt pack.
        with zipfile.ZipFile(tmp_path / 'pack.zip') as zf:
            man = yaml.safe_load(zf.read('manifest.yaml'))
            pack_db = zf.read('pack.db')
        man['environment'] = '../escape'
        evil = tmp_path / 'evil.zip'
        with zipfile.ZipFile(evil, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('manifest.yaml', yaml.safe_dump(man))
            zf.writestr('pack.db', pack_db)

        cache = tmp_path / 'cache'
        with Library(tmp_path / 'victim.db') as victim:
            with pytest.raises(SpoolError) as excinfo:
                install_pack(victim, evil, cache_dir=cache)
            assert 'environment' in str(excinfo.value).lower()
            assert victim.list_documents() == []
        # The write never escaped the cache directory.
        assert not (tmp_path / 'escape').exists()

        # from_dict rejects traversal, absolute, separator and bare-dot forms.
        for bad in ('../escape', '/abs/escape', 'a/b', '..', '.'):
            with pytest.raises(SpoolError):
                SpoolManifest.from_dict({
                    'environment': bad, 'version': '1',
                    'target_runtime': 'r', 'checksum': 'c',
                })

    def test_build_pack_ships_scip_graph(self, tmp_path):
        # The pack must carry the source's SCIP symbols +
        # edges so a consumer gets symbol-level cross-reference (callers/
        # callees/resolve/impact) without the repo, JDK, or indexer.
        root = tmp_path / 'repo'
        root.mkdir()
        session_sym = (
            'scip-python python pyspark 0.1 '
            'pyspark/sql/session.py/SparkSession#'
        )
        create_df_sym = (
            'scip-python python pyspark 0.1 '
            'pyspark/sql/session.py/SparkSession#createDataFrame().'
        )
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'explanation', 'session', 'SparkSession explanation',
                source_files=[str(root / 'session.py')],
                source_name='fakebricks',
            )
            with builder._conn_provider.acquire() as conn:
                init_scip_schema(conn)
                conn.execute(
                    'INSERT INTO scip_symbols (canonical_id, source_name, '
                    'language, file, line_start, line_end, kind, display_name, '
                    'qualified_name, parent_qualified_name) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (session_sym, 'fakebricks', 'python', 'session.py', 1, 9,
                     'Class', 'SparkSession',
                     'pyspark.sql.session.SparkSession', None),
                )
                conn.execute(
                    'INSERT INTO scip_edges (caller_canonical_id, '
                    'callee_canonical_id, edge_type, file, line, confidence) '
                    'VALUES (?,?,?,?,?,?)',
                    (create_df_sym, session_sym, 'call', 'session.py', 5,
                     'exact'),
                )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        with zipfile.ZipFile(tmp_path / 'pack.zip') as zf:
            (tmp_path / 'p.db').write_bytes(zf.read('pack.db'))
        conn = sqlite3.connect(tmp_path / 'p.db')
        try:
            syms = {r[0] for r in conn.execute(
                'SELECT canonical_id FROM scip_symbols '
                "WHERE source_name = 'fakebricks'").fetchall()}
            edge_count = conn.execute(
                'SELECT COUNT(*) FROM scip_edges').fetchone()[0]
        finally:
            conn.close()
        # The pack carries the source's SCIP graph, verbatim (canonical_ids
        # raw — the namespace rewrite happens at install, not build).
        assert session_sym in syms, 'pack must carry the source SCIP symbols'
        assert edge_count >= 1, 'pack must carry the source SCIP edges'

    def test_install_lands_scip_under_spool_namespace(self, tmp_path):
        # Installing a pack lands its SCIP under the
        # reserved spool source id — source_name rewritten from the build
        # source (like docs), canonical_id kept RAW — so the environment's
        # graph is queryable in the consumer store, self-contained.
        root = tmp_path / 'repo'
        root.mkdir()
        session_sym = (
            'scip-python python pyspark 0.1 '
            'pyspark/sql/session.py/SparkSession#'
        )
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'explanation', 'session', 'SparkSession explanation',
                source_files=[str(root / 'session.py')],
                source_name='fakebricks',
            )
            with builder._conn_provider.acquire() as conn:
                init_scip_schema(conn)
                conn.execute(
                    'INSERT INTO scip_symbols (canonical_id, source_name, '
                    'language, file, line_start, line_end, kind, display_name, '
                    'qualified_name, parent_qualified_name) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (session_sym, 'fakebricks', 'python', 'session.py', 1, 9,
                     'Class', 'SparkSession',
                     'pyspark.sql.session.SparkSession', None),
                )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        with Library(tmp_path / 'consumer.db') as consumer:
            install_pack(consumer, tmp_path / 'pack.zip', cache_dir=tmp_path / 'c')
            doc_source = consumer.list_documents()[0].source_name
            with consumer._conn_provider.acquire() as conn:
                scip = {
                    (r[0], r[1]) for r in conn.execute(
                        'SELECT canonical_id, source_name FROM scip_symbols',
                    ).fetchall()
                }
        # SCIP lands under the SAME reserved namespace as the docs, and the
        # canonical_id is verbatim (no moniker rewrite).
        assert doc_source.startswith('spool:')
        assert (session_sym, doc_source) in scip

    def test_full_scip_layers_travel(self, tmp_path):
        # The pack ships the FULL source-scoped SCIP layer set, not just
        # the call graph. A richer layer (config_values here) must travel and
        # install under the spool namespace — proving the copy generalizes
        # beyond scip_symbols/scip_edges.
        root = tmp_path / 'repo'
        root.mkdir()
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'explanation', 'conf', 'config explanation',
                source_files=[str(root / 'app.conf')], source_name='fakebricks',
            )
            with builder._conn_provider.acquire() as conn:
                init_scip_schema(conn)
                conn.execute(
                    'INSERT INTO config_values (source_name, file, key, value, '
                    'line_start) VALUES (?,?,?,?,?)',
                    ('fakebricks', 'app.conf', 'spark.sql.shuffle.partitions',
                     '200', 3),
                )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        with Library(tmp_path / 'consumer.db') as consumer:
            install_pack(consumer, tmp_path / 'pack.zip', cache_dir=tmp_path / 'c')
            doc_source = consumer.list_documents()[0].source_name
            with consumer._conn_provider.acquire() as conn:
                cfg = conn.execute(
                    'SELECT key, value, source_name FROM config_values',
                ).fetchall()
        assert ('spark.sql.shuffle.partitions', '200', doc_source) in cfg

    def test_spool_themes_travel_with_pack(self, tmp_path, monkeypatch):
        # Slice 3: the spool's OWN theme pass (association = the reserved
        # spool id, built by build_spool_internal_themes) ships in the pack —
        # docs + themes rows + members — and lands on the consumer, where
        # the T8 association rule admits them iff the spool is enabled.
        # Base-pass themes never ship (they're the leak's old home).
        import embedding
        monkeypatch.setattr(embedding, 'EMBEDDING_DIM', 2)  # match _vec dims

        from spools import spool_source_id

        root = tmp_path / 'corpus'
        root.mkdir()
        with Library(tmp_path / 'builder.db') as builder:
            member = builder.add_document(
                'explanation', 'Mesh Compaction Guide', 'body',
                embedding=_vec(1.0, 0.0), source_name='fakebricks',
            )
            theme_doc = builder.add_document(
                'theme', 'Mesh Compaction Patterns',
                'summary of the corpus compaction cluster',
                embedding=_vec(0.9, 0.1), source_name=None,
            )
            builder.add_theme(
                cluster_id='c-mesh', doc_id=theme_doc.id, member_count=1,
                resolution=1.0, summary_hash='h-mesh', dirty=False,
                association=spool_source_id('fakebricks'),
            )
            builder.set_theme_members('c-mesh', [(member.id, 1.0)])
            # A base-pass theme must NOT ship.
            stray = builder.add_document(
                'theme', 'Stray Base Theme', 'base pass summary',
                embedding=_vec(0.1, 0.9), source_name=None,
            )
            builder.add_theme(
                cluster_id='c-stray', doc_id=stray.id, member_count=1,
                resolution=1.0, summary_hash='h-stray', association='',
            )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )

        with Library(tmp_path / 'consumer.db') as consumer:
            install_pack(consumer, tmp_path / 'pack.zip',
                         cache_dir=tmp_path / 'c')
            from library import ScopedLibrary
            with_spool = ScopedLibrary(
                consumer, frozenset({'src1', 'spool:fakebricks'}))
            titles = {d.title for d in with_spool.list_documents_lite()}
            assert 'Mesh Compaction Patterns' in titles
            assert 'Stray Base Theme' not in titles
            without_spool = ScopedLibrary(consumer, frozenset({'src1'}))
            titles = {d.title for d in without_spool.list_documents_lite()}
            assert 'Mesh Compaction Patterns' not in titles

    def test_surface_tags_travel_with_pack(self, tmp_path):
        # Slice 3: build tags the corpus's doc-grade docs from the recipe's
        # surface vocabularies, the manifest carries the vocabularies (the
        # consumer matches QUESTIONS against them), and the tags land
        # corpus-keyed at install.
        import sqlite3

        from library.surface_tags import (
            docs_for_surfaces,
            surfaces_from_resolution,
        )

        root = tmp_path / 'corpus'
        root.mkdir()
        surfaces = {'serialization': ['serializ', 'pickle']}
        with Library(tmp_path / 'builder.db') as builder:
            tagged = builder.add_document(
                'explanation', 'Serialization Tuning', 'body',
                source_name='fakebricks',
            )
            builder.add_document(
                'explanation', 'Governance Notes', 'body',
                source_name='fakebricks',
            )
            manifest = build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
                surfaces=surfaces,
            )
        assert manifest.surfaces == surfaces
        with zipfile.ZipFile(tmp_path / 'pack.zip') as zf:
            packed = SpoolManifest.from_dict(
                yaml.safe_load(zf.read('manifest.yaml')))
        assert packed.surfaces == surfaces

        with Library(tmp_path / 'consumer.db') as consumer:
            install_pack(consumer, tmp_path / 'pack.zip',
                         cache_dir=tmp_path / 'c')
            with consumer._conn_provider.acquire() as conn:
                assert docs_for_surfaces(
                    conn, ['fakebricks'], ['serialization']) == {tagged.id}

        # The consumer reads the vocabularies off the resolution.
        from types import SimpleNamespace
        resolution = SimpleNamespace(registered={
            'fakebricks': SimpleNamespace(manifest=packed)})
        assert surfaces_from_resolution(resolution) == surfaces

    def test_version_facts_travel_with_pack(self, tmp_path):
        # Slice 2: build extracts the corpus's version markers from the
        # LOCATED source lines into version_facts, the pack ships them, and
        # install lands them corpus-keyed on the consumer — who never needs
        # the corpus files. The manifest carries the runtime→component map
        # (the availability join's right-hand side).
        import sqlite3

        from library.version_facts import (
            query_version_facts,
            runtime_availability,
        )

        root = tmp_path / 'corpus'
        (root / 'pkg').mkdir(parents=True)
        src = root / 'pkg' / 'frob.scala'
        src.write_text('package pkg\n\n@Since("3.1.0")\nobject Frobnicator')
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'catalog', 'pkg.Frobnicator',
                f'scala_object pkg.Frobnicator [scala] {src}:3-3 :: @Since',
                source_files=[str(src)],
                metadata={'kind': 'element', 'source_name': 'fakebricks',
                          'qualified_name': 'pkg.Frobnicator',
                          'subtype': 'scala_object',
                          'location': {'line_start': 3, 'line_end': 3}},
                source_name='fakebricks',
            )
            manifest = build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
                runtime_components={'fakebricks': '3.5.0'},
            )
        assert manifest.runtime_components == {'fakebricks': '3.5.0'}
        with zipfile.ZipFile(tmp_path / 'pack.zip') as zf:
            packed = SpoolManifest.from_dict(
                yaml.safe_load(zf.read('manifest.yaml')))
            (tmp_path / 'packed.db').write_bytes(zf.read('pack.db'))
        assert packed.runtime_components == {'fakebricks': '3.5.0'}
        conn = sqlite3.connect(tmp_path / 'packed.db')
        assert conn.execute(
            'SELECT COUNT(*) FROM version_facts').fetchone()[0] == 1
        conn.close()

        with Library(tmp_path / 'consumer.db') as consumer:
            install_pack(consumer, tmp_path / 'pack.zip',
                         cache_dir=tmp_path / 'c')
            with consumer._conn_provider.acquire() as conn:
                facts = query_version_facts(
                    conn, ['fakebricks'], 'pkg.Frobnicator')
                assert [(f.fact, f.version) for f in facts] == [
                    ('since', '3.1.0')]
                avail = runtime_availability(
                    conn, ['fakebricks'], 'pkg.Frobnicator',
                    packed.runtime_components)
                assert avail['available'] is True

    def test_pack_carries_no_builder_absolute_paths(self, tmp_path):
        # A real pack shipped the builder's machine-local absolute paths in
        # 100% of its docs (content + source_files) and in every
        # string_literals/config_values row — leaking the builder's username
        # and machine layout, and dead-ending consumer-side file reads (the
        # corpus isn't on the consumer's disk). Packs must carry
        # corpus-RELATIVE paths everywhere.
        import sqlite3

        root = tmp_path / 'corpus'
        (root / 'pkg').mkdir(parents=True)
        abs_file = str(root / 'pkg' / 'mod.py')
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'catalog', 'pkg.mod.helper_fn',
                f'function pkg.mod.helper_fn [python] {abs_file}:3-9 '
                ':: def helper_fn(',
                source_files=[abs_file],
                source_name='fakebricks',
                metadata={'kind': 'element',
                          'qualified_name': 'pkg.mod.helper_fn'},
            )
            with builder._conn_provider.acquire() as conn:
                init_scip_schema(conn)
                conn.execute(
                    'INSERT INTO string_literals (source_name, file, '
                    'line_start, col_start, value) VALUES (?,?,?,?,?)',
                    ('fakebricks', abs_file, 3, 0, 'lit'),
                )
                conn.execute(
                    'INSERT INTO config_values (source_name, file, key, '
                    'value, line_start) VALUES (?,?,?,?,?)',
                    ('fakebricks', abs_file, 'k', 'v', 1),
                )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        with zipfile.ZipFile(tmp_path / 'pack.zip') as zf:
            (tmp_path / 'extracted-pack.db').write_bytes(zf.read('pack.db'))
        conn = sqlite3.connect(tmp_path / 'extracted-pack.db')
        try:
            builder_marker = str(tmp_path)
            content, source_files = conn.execute(
                'SELECT content, source_files FROM documents',
            ).fetchone()
            assert builder_marker not in content
            assert builder_marker not in source_files
            assert source_files == '["pkg/mod.py"]'
            assert 'pkg/mod.py:3-9' in content
            assert [r[0] for r in conn.execute(
                'SELECT file FROM string_literals')] == ['pkg/mod.py']
            assert [r[0] for r in conn.execute(
                'SELECT file FROM config_values')] == ['pkg/mod.py']
        finally:
            conn.close()

    def test_disabled_spool_scip_not_surfaced(self, tmp_path):
        # A spool's SCIP edges surface only when its namespace is in the
        # query closure. Out of closure (spool disabled), scip_callers on a
        # spool symbol returns nothing — the disable-remediation extends to
        # the new SCIP surface, not just docs.
        root = tmp_path / 'repo'
        root.mkdir()
        parent = 'scip-python python pyspark 0.1 m.py/A#'
        child = 'scip-python python pyspark 0.1 m.py/A#f().'
        with Library(tmp_path / 'builder.db') as builder:
            builder.add_document(
                'explanation', 'a', 'A explanation',
                source_files=[str(root / 'm.py')], source_name='fakebricks',
            )
            with builder._conn_provider.acquire() as conn:
                init_scip_schema(conn)
                for sym, qn in ((parent, 'm.A'), (child, 'm.A.f')):
                    conn.execute(
                        'INSERT INTO scip_symbols (canonical_id, source_name, '
                        'language, file, line_start, line_end, kind, '
                        'display_name, qualified_name, parent_qualified_name) '
                        'VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (sym, 'fakebricks', 'python', 'm.py', 1, 2, 'Function',
                         qn.rsplit('.', 1)[-1], qn, None),
                    )
                conn.execute(
                    'INSERT INTO scip_edges (caller_canonical_id, '
                    'callee_canonical_id, edge_type, file, line, confidence) '
                    'VALUES (?,?,?,?,?,?)',
                    (child, parent, 'call', 'm.py', 2, 'exact'),
                )
            build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        with Library(tmp_path / 'consumer.db') as consumer:
            install_pack(consumer, tmp_path / 'pack.zip', cache_dir=tmp_path / 'c')
            ns = consumer.list_documents()[0].source_name        # spool:...
            in_scope = ScopedLibrary(consumer, frozenset({ns}))
            out_scope = ScopedLibrary(consumer, frozenset({'unrelated'}))
            # In closure: the spool's own call edge surfaces.
            assert in_scope.scip_callers(parent), (
                'in-closure spool SCIP edge should surface'
            )
            # Out of closure (disabled): it must not.
            assert out_scope.scip_callers(parent) == [], (
                'a disabled spool must not surface SCIP edges'
            )

    def test_build_is_source_scoped(self, tmp_path):
        # HIGH-3 — build must not materialize the whole library: it reads
        # only the target source's docs, never the load-everything path.
        root = tmp_path / 'repo'
        root.mkdir()

        class NoFullScan(Library):
            def list_documents(self, *a, **k):
                raise AssertionError(
                    'build_pack loaded ALL documents (memory bomb)',
                )

        with NoFullScan(tmp_path / 'builder.db') as builder:
            Library.add_document(
                builder, 'explanation', 'mine', 'body',
                source_files=[str(root / 'a.py')], source_name='fakebricks',
            )
            Library.add_document(
                builder, 'explanation', 'other', 'huge other-source body',
                source_name='src1',
            )
            manifest = build_pack(
                builder, environment='fakebricks', version='1.0',
                target_runtime='fake-17.3', certified_docs=(),
                source_root=root, out_path=tmp_path / 'pack.zip',
            )
        assert manifest.environment == 'fakebricks'
        with zipfile.ZipFile(tmp_path / 'pack.zip') as zf:
            (tmp_path / 'p.db').write_bytes(zf.read('pack.db'))
        with Library(tmp_path / 'p.db') as pack:
            titles = {d.title for d in pack.list_documents()}
        assert titles == {'mine'}
