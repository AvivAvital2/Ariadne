"""Document-layer reserved-namespace guard (audit finding S1).

Only a verified spool install may write documents under the reserved `spool:`
source; every other ingestion path (`ariadne import`, a direct `add_document`)
is refused. The HIGH-2 guard was config-only (`set_source_config`), so
`ariadne import` could forge unsigned `spool:`-namespaced content — a full
bypass of pack signing via a side door. Synthetic fixtures only.
"""
import textwrap

import pytest

from library import Library


def test_add_document_refuses_reserved_spool_source(tmp_path):
    lib = Library(tmp_path / 'g.db')
    try:
        # The reserved spool: namespace is refused on the ordinary write path.
        with pytest.raises(ValueError):
            lib.add_document(
                content_type='catalog', title='t', content='c',
                source_files=[], source_name='spool:evil',
            )
        # The privileged install path (post-verification) is allowed.
        doc = lib.add_document(
            content_type='catalog', title='t', content='c',
            source_files=[], source_name='spool:ok',
            _allow_reserved_source=True,
        )
        assert doc.source_name == 'spool:ok'
    finally:
        lib.close()


def test_import_refuses_forged_spool_namespaced_docs(tmp_path):
    # The S1 attack: a doc-export archive attributing its docs to a spool:
    # source, imported via `ariadne import`, must be refused — otherwise it
    # forges unsigned spool content through a door that skips install's
    # verification.
    from export import import_from_markdown

    tree = tmp_path / 'evil_export'
    (tree / 'explanations').mkdir(parents=True)
    (tree / 'explanations' / 'evil.md').write_text(textwrap.dedent('''\
        ---
        title: Forged Databricks Doc
        type: explanation
        id: forged-1
        metadata:
          source_name: spool:databricks
          kind: element
        ---
        # Forged Databricks Doc

        Malicious content masquerading as installed spool knowledge.
    '''))

    lib = Library(tmp_path / 'store.db')
    try:
        with pytest.raises(ValueError):
            import_from_markdown(lib, tree)
        # Nothing forged landed under the reserved namespace.
        assert not [
            d for d in lib.list_documents_lite()
            if d.source_name == 'spool:databricks'
        ]
    finally:
        lib.close()
