"""Phase 2 evolutionary-TDD walk for ``ScopedLibrary``.

The wrapper makes filtering structural at the data-access layer: every
public data-returning method on ``Library`` becomes accessible only via
a ``ScopedLibrary`` that holds a closure (a ``frozenset[str]`` of source
names), and applies the closure on every result.

This file grows one demand at a time. Each cycle adds a new behavioral
demand to ``TestScopedLibrary``; the test file *is* the spec for the
chokepoint.

Fixture nomenclature: ``shared`` is a leaf shared library; ``product``
depends on ``shared``; ``extension`` depends on ``product``. The
closure under test is ``{'product', 'shared'}``.

See ``designs/directional-closure-scoping.md`` Phase 2 for the cycle plan.
"""
from __future__ import annotations

from pathlib import Path


class TestScopedLibrary:
    # ---- T1 -----------------------------------------------------------
    # The smallest possible demand: build ScopedLibrary(library, closure)
    # with two product docs and one extension doc; .list_documents_lite()
    # returns only the docs whose source_name is in the closure.
    def test_t1_list_documents_lite_filtered_by_closure(
        self, tmp_path: Path,
    ) -> None:
        from library import Library, ScopedLibrary

        with Library(tmp_path / 'library.db') as library:
            library.add_document(
                content_type='explanation', title='product-explain-a',
                content='auth integration in product',
                source_name='product',
            )
            library.add_document(
                content_type='explanation', title='product-explain-b',
                content='product feature details',
                source_name='product',
            )
            library.add_document(
                content_type='explanation', title='extension-explain',
                content='extension integration details',
                source_name='extension',
            )

            scoped = ScopedLibrary(
                library, frozenset({'product', 'shared'}),
            )
            docs = scoped.list_documents_lite()

            titles = {d.title for d in docs}
            sources = {d.source_name for d in docs}
            assert titles == {'product-explain-a', 'product-explain-b'}
            assert sources == {'product'}
            assert 'extension-explain' not in titles

    # ---- T2 -----------------------------------------------------------
    # Embedding lookup by ID is closure-filtered too. Even if a caller
    # somehow holds an out-of-closure doc id (via a leak in an earlier
    # wrapper or a stale cache), get_embeddings_for_ids must silently
    # drop it — raising would surface the existence of the out-of-closure
    # id, which is itself a leak.
    def test_t2_get_embeddings_for_ids_filtered(
        self, tmp_path: Path,
    ) -> None:
        import numpy as np
        from library import Library, ScopedLibrary

        with Library(tmp_path / 'library.db') as library:
            doc_product = library.add_document(
                content_type='explanation', title='product-doc',
                content='product content', source_name='product',
                embedding=np.ones(3072, dtype=np.float32),
            )
            doc_extension = library.add_document(
                content_type='explanation', title='extension-doc',
                content='extension content', source_name='extension',
                embedding=np.ones(3072, dtype=np.float32),
            )

            scoped = ScopedLibrary(
                library, frozenset({'product', 'shared'}),
            )
            embeddings = scoped.get_embeddings_for_ids(
                [doc_product.id, doc_extension.id],
            )

            # product's embedding present; extension's silently dropped.
            assert doc_product.id in embeddings
            assert doc_extension.id not in embeddings

            # T1 still holds.
            docs = scoped.list_documents_lite()
            assert {d.source_name for d in docs} == {'product'}

    # ---- T3 -----------------------------------------------------------
    # Three more closure-bound methods, in one go (each follows the same
    # discipline — caller may pass any input; only in-closure rows come
    # back).
    #
    #   - find_documents_by_source_files: file-path lookup must drop
    #     matches that live in an out-of-closure source.
    #   - get_documents_batch: full-doc batch fetch must drop out-of-
    #     closure rows (paired with T2's embedding-only variant).
    #   - get_related: graph-walk neighbors must be filtered by closure,
    #     including the requested doc itself — if the seed doc is out of
    #     closure, the result is empty.
    def test_t3_three_more_closure_bound_methods(
        self, tmp_path: Path,
    ) -> None:
        from library import Library, ScopedLibrary

        with Library(tmp_path / 'library.db') as library:
            # product side: a doc whose source_files reference 'foo.py'.
            doc_product = library.add_document(
                content_type='explanation', title='product-doc',
                content='product content', source_name='product',
                source_files=['product/path/foo.py'],
            )
            # extension side: a doc whose source_files reference the same
            # basename. find_documents_by_source_files matches by basename
            # — both rows would normally surface; the closure filter
            # drops extension.
            doc_extension = library.add_document(
                content_type='explanation', title='extension-doc',
                content='extension content', source_name='extension',
                source_files=['extension/path/foo.py'],
            )
            # A second product doc, neighbored to doc_product via
            # doc_graph.
            doc_product2 = library.add_document(
                content_type='explanation', title='product-doc-2',
                content='product content 2', source_name='product',
                source_files=['product/path/bar.py'],
            )
            # An extension doc, also neighbored to doc_product — would
            # leak via get_related without the closure filter.
            doc_extension2 = library.add_document(
                content_type='explanation', title='extension-doc-2',
                content='extension content 2', source_name='extension',
                source_files=['extension/path/bar.py'],
            )
            # Wire up the graph so doc_product is connected to
            # doc_product2 (product→product) and to doc_extension2
            # (product→extension).
            with library._conn_provider.acquire() as conn:
                conn.execute(
                    'INSERT INTO doc_graph(source_id, target_id, '
                    'edge_type, weight) VALUES (?, ?, ?, ?)',
                    (doc_product.id, doc_product2.id, 'related', 1.0),
                )
                conn.execute(
                    'INSERT INTO doc_graph(source_id, target_id, '
                    'edge_type, weight) VALUES (?, ?, ?, ?)',
                    (doc_product.id, doc_extension2.id, 'related', 1.0),
                )
                conn.commit()

            scoped = ScopedLibrary(
                library, frozenset({'product', 'shared'}),
            )

            # find_documents_by_source_files: only product's foo.py
            # match.
            hits = scoped.find_documents_by_source_files(['foo.py'])
            assert {d.id for d in hits} == {doc_product.id}
            assert all(d.source_name == 'product' for d in hits)

            # get_documents_batch: full-doc variant of T2's embedding
            # lookup; out-of-closure ids silently dropped.
            batch = scoped.get_documents_batch(
                [doc_product.id, doc_extension.id],
            )
            assert {d.id for d in batch} == {doc_product.id}

            # get_related: neighbors of doc_product are doc_product2
            # (in-closure) and doc_extension2 (out-of-closure). Only
            # product's neighbor surfaces.
            related = scoped.get_related(doc_product.id)
            assert {r['id'] for r in related} == {doc_product2.id}

            # T1 + T2 still hold under the elaborated fixture.
            import numpy as np
            doc_product_emb = library.add_document(
                content_type='explanation', title='product-doc-emb',
                content='product with embedding', source_name='product',
                embedding=np.ones(3072, dtype=np.float32),
            )
            doc_extension_emb = library.add_document(
                content_type='explanation', title='extension-doc-emb',
                content='extension with embedding',
                source_name='extension',
                embedding=np.ones(3072, dtype=np.float32),
            )
            embs = scoped.get_embeddings_for_ids(
                [doc_product_emb.id, doc_extension_emb.id],
            )
            assert doc_product_emb.id in embs
            assert doc_extension_emb.id not in embs
            docs = scoped.list_documents_lite()
            assert 'extension' not in {d.source_name for d in docs}

    # ---- T4 -----------------------------------------------------------
    # Themes — the tricky case for the closure rule. A theme is anchored
    # by a *summary* document (one ``source_name``) and points at any
    # number of *member* docs (potentially across many sources).
    #
    # The closure rule applied to themes:
    #   - ``list_themes()`` returns a theme iff its summary doc's
    #     source_name is in the closure. (The summary is the theme's
    #     identity — if you can't see the summary, you don't see the
    #     theme.)
    #   - ``get_theme_members()`` returns only the members whose own
    #     source_name is in the closure. Out-of-closure members are
    #     filtered from the displayed set.
    def test_t4_themes_filtered_and_members_filtered(
        self, tmp_path: Path,
    ) -> None:
        from library import Library, ScopedLibrary

        with Library(tmp_path / 'library.db') as library:
            # Theme summary docs: one for product, one for extension.
            theme_product_doc = library.add_document(
                content_type='theme', title='product-theme-summary',
                content='product-side theme', source_name='product',
            )
            theme_extension_doc = library.add_document(
                content_type='theme', title='extension-theme-summary',
                content='extension-side theme', source_name='extension',
            )
            # Member docs across both sources. Critically, the product
            # theme has BOTH a product member and an extension member —
            # the closure rule must surface the theme (via its product
            # summary) but drop the extension member from the displayed
            # member list.
            product_member = library.add_document(
                content_type='explanation', title='product-member',
                content='product member', source_name='product',
            )
            ext_member_of_product_theme = library.add_document(
                content_type='explanation',
                title='extension-member-of-product-theme',
                content='extension member of a product theme',
                source_name='extension',
            )
            ext_member_of_ext_theme = library.add_document(
                content_type='explanation',
                title='extension-member-of-extension-theme',
                content='extension member', source_name='extension',
            )

            library.add_theme(
                cluster_id='cluster-product',
                doc_id=theme_product_doc.id,
                member_count=2, resolution=1.0,
                summary_hash='hash-product',
                coherent=True, dirty=False,
            )
            library.set_theme_members(
                'cluster-product',
                [
                    (product_member.id, 1.0),
                    (ext_member_of_product_theme.id, 1.0),
                ],
            )

            library.add_theme(
                cluster_id='cluster-extension',
                doc_id=theme_extension_doc.id,
                member_count=1, resolution=1.0,
                summary_hash='hash-extension',
                coherent=True, dirty=False,
            )
            library.set_theme_members(
                'cluster-extension',
                [(ext_member_of_ext_theme.id, 1.0)],
            )

            scoped = ScopedLibrary(
                library, frozenset({'product', 'shared'}),
            )

            # The product theme is visible (summary in closure); the
            # extension theme is not.
            themes = scoped.list_themes()
            cluster_ids = {t.cluster_id for t in themes}
            assert 'cluster-product' in cluster_ids
            assert 'cluster-extension' not in cluster_ids

            # The product theme's members include the product-side
            # member only; the extension-side member is filtered out
            # for display.
            members = scoped.get_theme_members('cluster-product')
            member_ids = {m[0] for m in members}
            assert product_member.id in member_ids
            assert ext_member_of_product_theme.id not in member_ids

            # And asking for the extension theme's members from
            # product's scope returns nothing (the theme itself is out
            # of closure).
            members_ext = scoped.get_theme_members('cluster-extension')
            assert members_ext == []

            # T1-T3 demands still hold under the elaborated fixture: no
            # extension source ever shows up in list_documents_lite().
            assert 'extension' not in {
                d.source_name for d in scoped.list_documents_lite()
            }

    # ---- T5 -----------------------------------------------------------
    # SCIP graph readers — closure applies to BOTH endpoints of each
    # edge. An edge is in-closure iff both its caller and its callee
    # have a source_name in the closure. This is what enforces the
    # directional rule at the graph level:
    #
    #   - product's scope ({product, shared}): scip_callers(shared.X)
    #     returns the product → shared edge but not extension → shared.
    #   - shared's scope ({shared, product, extension}, reverse): both
    #     consumer edges surface — the leaf gets its full consumer
    #     panorama, which is exactly the directional flip's purpose.
    def test_t5_scip_callers_filtered_by_closure(
        self, tmp_path: Path,
    ) -> None:
        from library import Library, ScopedLibrary

        with Library(tmp_path / 'library.db') as library:
            with library._conn_provider.acquire() as conn:
                # Three symbols, one per source.
                symbols = [
                    ('shared:authenticate', 'shared', 'python',
                     'shared/auth.py', 10, 20,
                     'function', 'authenticate',
                     'shared.authenticate', None),
                    ('product:handle', 'product', 'python',
                     'product/main.py', 30, 40,
                     'function', 'handle',
                     'product.handle', None),
                    ('extension:bootstrap', 'extension', 'python',
                     'extension/init.py', 5, 15,
                     'function', 'bootstrap',
                     'extension.bootstrap', None),
                ]
                conn.executemany(
                    'INSERT INTO scip_symbols VALUES '
                    '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    symbols,
                )
                # Two edges: product → shared, extension → shared.
                edges = [
                    ('product:handle', 'shared:authenticate', 'call',
                     'product/main.py', 35, 'precise'),
                    ('extension:bootstrap', 'shared:authenticate', 'call',
                     'extension/init.py', 12, 'precise'),
                ]
                conn.executemany(
                    'INSERT INTO scip_edges VALUES '
                    '(?, ?, ?, ?, ?, ?)',
                    edges,
                )
                conn.commit()

            # product's scope: only product → shared surfaces.
            product_scoped = ScopedLibrary(
                library, frozenset({'product', 'shared'}),
            )
            edges = product_scoped.scip_callers('shared:authenticate')
            callers = {e.caller.source_name for e in edges}
            assert callers == {'product'}
            assert all(
                e.callee.source_name == 'shared' for e in edges
            )

            # shared's reverse closure: both consumer edges surface.
            # This is the directional-flip payoff — the leaf sees its
            # full consumer panorama.
            shared_scoped = ScopedLibrary(
                library,
                frozenset({'shared', 'product', 'extension'}),
            )
            edges = shared_scoped.scip_callers('shared:authenticate')
            callers = {e.caller.source_name for e in edges}
            assert callers == {'product', 'extension'}

            # Symmetry — callees from a caller perspective.
            edges = product_scoped.scip_callees('product:handle')
            callees = {e.callee.source_name for e in edges}
            assert callees == {'shared'}
            # The same call from an extension scope (which doesn't
            # include product) shouldn't surface — the edge has one
            # endpoint outside the closure.
            extension_only = ScopedLibrary(
                library, frozenset({'extension', 'product', 'shared'}),
            )
            edges = extension_only.scip_callees('product:handle')
            # extension_only has product in closure (extension forward-
            # closes to product+shared), so this surfaces.
            assert {e.callee.source_name for e in edges} == {'shared'}

    # ---- T6 -----------------------------------------------------------
    # An empty closure is a misconfiguration, not a legitimate
    # "scope to nothing" state. Constructing a ScopedLibrary with an
    # empty frozenset must raise at __init__ — silently building a
    # wrapper that drops every row would be a confusing failure mode
    # (every method "works" but every result is empty).
    #
    # The error message names the offense so the caller can locate
    # the misconfigured closure resolution path.
    #
    # (Cycle numbering note: the design doc's plan-T6 — physical rename
    # of public Library methods to ``_*_unscoped`` — is deferred to
    # Phase 3 where it can be done atomically with the cross-codebase
    # caller migration. Without that atomic pairing, a unilateral rename
    # leaves the repo in a partial-migration state with broken consumers
    # at every commit, which violates the "every commit is green"
    # invariant. The rename's structural enforcement is also given by
    # Phase 5's lint check.)
    def test_t6_empty_closure_rejected(self, tmp_path: Path) -> None:
        import pytest
        from library import Library, ScopedLibrary

        with Library(tmp_path / 'library.db') as library:
            with pytest.raises(ValueError) as exc:
                ScopedLibrary(library, frozenset())
            message = str(exc.value)
            assert 'closure' in message.lower()
            assert 'empty' in message.lower()

    # ---- T7.5 (regression) --------------------------------------------
    # ``get_related`` is called by ``mcp_service_search._search_uncached``
    # with a file-path seed (the user's ``context_file`` argument). The
    # underlying graph stores some edges with file paths as nodes (e.g.
    # ``edge_type='imports'`` edges populated by ``library_intelligence.
    # explain``). The wrapper must NOT pre-reject the seed just because
    # it doesn't appear in the ``documents.id`` column — it should
    # delegate to the underlying walker and only filter the RESULTS by
    # closure.
    def test_t7_5_get_related_with_non_doc_id_seed_still_walks(
        self, tmp_path: Path,
    ) -> None:
        from library import Library, ScopedLibrary

        with Library(tmp_path / 'library.db') as library:
            # A real doc the file is "about", linked to a file-path
            # node via doc_graph (mimics how imports edges are
            # inserted by other code).
            target_doc = library.add_document(
                content_type='explanation',
                title='product-target',
                content='target', source_name='product',
            )
            with library._conn_provider.acquire() as conn:
                conn.execute(
                    'INSERT INTO doc_graph(source_id, target_id, '
                    'edge_type, weight) VALUES (?, ?, ?, ?)',
                    ('product/feature.py', target_doc.id,
                     'imports', 1.0),
                )
                conn.commit()

            scoped = ScopedLibrary(
                library, frozenset({'product', 'shared'}),
            )
            related = scoped.get_related(
                'product/feature.py', max_hops=1, limit=10,
            )
            # The file-path seed is not in ``documents.id``, but the
            # walker finds the imports edge and surfaces target_doc
            # (which is in closure). Pre-fix the wrapper short-circuited
            # to [] before the walk could run.
            assert {r['id'] for r in related} == {target_doc.id}

    # ---- T7 -----------------------------------------------------------
    # Filter composition: closure is one filter among potentially many
    # (coherent_only on themes, branch filter on docs, role / audience
    # on future surfaces). When multiple filters apply, the result set
    # must satisfy ALL of them, not whichever one runs last. The closure
    # is applied AFTER the underlying method's own filters, so the
    # underlying filter cannot widen the result past the closure (and
    # the closure cannot widen the underlying filter).
    #
    # This cycle drives the demand with two concrete pairs:
    #   - list_themes(coherent_only=True) ∧ closure: incoherent themes
    #     are dropped *and* out-of-closure themes are dropped.
    #   - filter_by_branch ∘ list_documents_lite() ∧ closure: branch-
    #     gated docs respect both branch policy and closure.
    def test_t7_filters_compose_with_closure(
        self, tmp_path: Path,
    ) -> None:
        from library import Library, ScopedLibrary, filter_by_branch

        with Library(tmp_path / 'library.db') as library:
            # Theme docs: one coherent product, one incoherent product,
            # one coherent extension. With closure={product, shared}
            # and coherent_only=True, only the coherent product theme
            # should surface (closure AND coherence both must hold).
            coh_prod_doc = library.add_document(
                content_type='theme', title='coherent-product-theme',
                content='coherent product theme',
                source_name='product',
            )
            incoh_prod_doc = library.add_document(
                content_type='theme', title='incoherent-product-theme',
                content='incoherent product theme',
                source_name='product',
            )
            coh_ext_doc = library.add_document(
                content_type='theme', title='coherent-extension-theme',
                content='coherent extension theme',
                source_name='extension',
            )
            library.add_theme(
                cluster_id='c-coh-prod', doc_id=coh_prod_doc.id,
                member_count=1, resolution=1.0,
                summary_hash='h1', coherent=True, dirty=False,
            )
            library.add_theme(
                cluster_id='c-incoh-prod', doc_id=incoh_prod_doc.id,
                member_count=1, resolution=1.0,
                summary_hash='h2', coherent=False, dirty=False,
            )
            library.add_theme(
                cluster_id='c-coh-ext', doc_id=coh_ext_doc.id,
                member_count=1, resolution=1.0,
                summary_hash='h3', coherent=True, dirty=False,
            )

            # Branch-gated documents: stable docs always pass; non-
            # stable docs need a branch-pattern match. With closure
            # {product, shared} and branch='main', only product docs
            # that also satisfy branch policy should surface.
            library.add_document(
                content_type='explanation',
                title='product-stable',
                content='stable product doc',
                source_name='product',
                metadata={'status': 'stable'},
            )
            library.add_document(
                content_type='explanation',
                title='product-feature-on-main',
                content='product feature gated to main',
                source_name='product',
                metadata={
                    'status': 'experimental',
                    'branches': ['main'],
                },
            )
            library.add_document(
                content_type='explanation',
                title='product-feature-on-other',
                content='product feature gated to other branch',
                source_name='product',
                metadata={
                    'status': 'experimental',
                    'branches': ['feature/*'],
                },
            )
            library.add_document(
                content_type='explanation',
                title='extension-stable',
                content='stable extension doc',
                source_name='extension',
                metadata={'status': 'stable'},
            )

            scoped = ScopedLibrary(
                library, frozenset({'product', 'shared'}),
            )

            # Theme composition: coherent AND in-closure.
            themes = scoped.list_themes(coherent_only=True)
            cluster_ids = {t.cluster_id for t in themes}
            assert 'c-coh-prod' in cluster_ids
            assert 'c-incoh-prod' not in cluster_ids   # coherence drops
            assert 'c-coh-ext' not in cluster_ids       # closure drops

            # Theme composition (relaxed coherence): in-closure still
            # holds even when coherent_only is False.
            themes_all = scoped.list_themes(coherent_only=False)
            cluster_ids_all = {t.cluster_id for t in themes_all}
            assert 'c-coh-prod' in cluster_ids_all
            assert 'c-incoh-prod' in cluster_ids_all    # coherence relaxed
            assert 'c-coh-ext' not in cluster_ids_all    # closure still holds

            # Branch composition: closure first, then filter_by_branch.
            docs = scoped.list_documents_lite()
            on_main = filter_by_branch(docs, 'main')
            titles = {d.title for d in on_main}
            assert 'product-stable' in titles
            assert 'product-feature-on-main' in titles
            assert 'product-feature-on-other' not in titles   # branch
            assert 'extension-stable' not in titles            # closure

            # Branch composition on a feature branch: closure still
            # holds, branch policy flips to allow the feature/* doc.
            on_feature = filter_by_branch(docs, 'feature/x')
            titles_f = {d.title for d in on_feature}
            assert 'product-stable' in titles_f
            assert 'product-feature-on-main' not in titles_f  # branch
            assert 'product-feature-on-other' in titles_f
            assert 'extension-stable' not in titles_f          # closure

    # ---- T8 -----------------------------------------------------------
    # Spool theme isolation. Cross-cutting THEME summary docs carry
    # source_name=NULL (cross-source by design) and were admitted
    # unconditionally — so a spool's themes leaked into EVERY project's
    # results regardless of whether that spool was in scope. The rule for
    # a NULL-source theme:
    #   - scoped association ('|'-joined sorted source-name key, e.g.
    #     'product|spool:alpha'): admitted iff EVERY source it spans is in
    #     the closure (unchanged).
    #   - base pass (''): member-grounded — admitted iff at least one
    #     member element's source is in the closure. "Always visible" was
    #     the leak: the base global pass clusters the WHOLE store, so a
    #     spool corpus's themes surfaced in every project's results.
    #   - a theme doc with NO themes row (a stale orphan from an earlier
    #     rebuild) is never admitted — fail closed.
    def test_t8_theme_docs_gated_by_association_closure(
        self, tmp_path: Path,
    ) -> None:
        from library import Library, ScopedLibrary

        with Library(tmp_path / 'library.db') as library:
            # A normal in-closure doc — always visible, spool-independent.
            # Also the member element grounding the base theme in 'product'.
            product_doc = library.add_document(
                content_type='explanation', title='product-doc',
                content='product content', source_name='product',
            )
            # A spool-corpus doc: the member element grounding the leaked
            # base-pass corpus theme in 'spool:beta'.
            spool_member_doc = library.add_document(
                content_type='explanation', title='spool-member-doc',
                content='spool corpus content', source_name='spool:beta',
                _allow_reserved_source=True,
            )
            # Four NULL-source theme summary docs + one orphan.
            base_theme = library.add_document(
                content_type='theme', title='base-theme',
                content='user/base pass theme', source_name=None,
            )
            corpus_theme = library.add_document(
                content_type='theme', title='corpus-theme',
                content='base-pass theme whose members are all spool corpus',
                source_name=None,
            )
            xsrc_theme = library.add_document(
                content_type='theme', title='xsrc-theme',
                content='project x spool cross-source theme',
                source_name=None,
            )
            spool_theme = library.add_document(
                content_type='theme', title='spool-internal-theme',
                content='spool-internal corpus theme', source_name=None,
            )
            library.add_document(
                content_type='theme', title='orphan-theme',
                content='stale summary doc with no themes row',
                source_name=None,
            )
            library.add_theme(
                cluster_id='c-base', doc_id=base_theme.id,
                member_count=1, resolution=1.0, summary_hash='h1',
                association='',
            )
            library.set_theme_members('c-base', [(product_doc.id, 1.0)])
            library.add_theme(
                cluster_id='c-corpus', doc_id=corpus_theme.id,
                member_count=1, resolution=1.0, summary_hash='h4',
                association='',
            )
            library.set_theme_members('c-corpus', [(spool_member_doc.id, 1.0)])
            library.add_theme(
                cluster_id='c-xsrc', doc_id=xsrc_theme.id,
                member_count=1, resolution=1.0, summary_hash='h2',
                association='product|spool:alpha',
            )
            library.add_theme(
                cluster_id='c-spool', doc_id=spool_theme.id,
                member_count=1, resolution=1.0, summary_hash='h3',
                association='spool:beta',
            )

            def scoped_for(closure):
                return ScopedLibrary(library, frozenset(closure))

            def titles_for(closure):
                return {
                    d.title
                    for d in scoped_for(closure).list_documents_lite()
                }

            # Neither spool in scope: the product doc + the base theme
            # grounded by a product member. The base-pass CORPUS theme is
            # hidden (its members are all spool docs — the leak), the
            # scoped-association themes are hidden, the orphan is hidden.
            assert titles_for({'product', 'shared'}) == {
                'product-doc', 'base-theme',
            }

            # spool:alpha in scope (enabled): the product×alpha theme
            # joins; beta-grounded/associated themes stay hidden.
            assert titles_for({'product', 'shared', 'spool:alpha'}) == {
                'product-doc', 'base-theme', 'xsrc-theme',
            }

            # Both spool sources in scope: every tracked theme is admitted
            # (the corpus theme via its spool:beta member), plus the spool
            # member doc itself as a regular in-closure doc. The orphan
            # stays hidden — fail closed.
            assert titles_for(
                {'product', 'shared', 'spool:alpha', 'spool:beta'},
            ) == {
                'product-doc', 'spool-member-doc', 'base-theme',
                'corpus-theme', 'xsrc-theme', 'spool-internal-theme',
            }

            # Member grounding scopes cross-repo themes to where their
            # members actually live: a different project sees NEITHER the
            # product-grounded base theme NOR 'product|spool:alpha'.
            assert titles_for({'other', 'shared', 'spool:alpha'}) == set()

            # The id-filter path (get_documents_batch → _filter_ids_by_closure)
            # gates identically — an out-of-scope theme id is dropped even
            # when explicitly requested.
            all_theme_ids = [
                base_theme.id, corpus_theme.id, xsrc_theme.id, spool_theme.id,
            ]
            batch_none = scoped_for(
                {'product', 'shared'},
            ).get_documents_batch(all_theme_ids)
            assert {d.id for d in batch_none} == {base_theme.id}
            batch_beta = scoped_for(
                {'product', 'shared', 'spool:beta'},
            ).get_documents_batch(all_theme_ids)
            assert {d.id for d in batch_beta} == {
                base_theme.id, corpus_theme.id, spool_theme.id,
            }
