"""Local-model integration with public CLI archive and indexing workflows."""

import sqlite3
import sys
import types
import zipfile

import pytest
import yaml

from library import Library
from tests.local_model_server import deployments, trap_inference
from tests.test_local_maintenance import invoke
from tests.test_local_workflows import fixture_config


@pytest.mark.parametrize('mismatched', [False, True])
def test_local_cli_archive_import_preserves_and_checks_provenance(
    tmp_path, monkeypatch, mismatched
):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        assert invoke(monkeypatch, 'topic', 'Arithmetic', '--source', 'alpha', '--files', 'core.py') == 0
        archive = tmp_path / 'export.zip'
        assert invoke(monkeypatch, 'export', str(archive), '--source', 'alpha') == 0
        with Library(cfg.db_path) as library:
            original, = library.list_documents()

        if mismatched:
            changed = tmp_path / 'mismatched.zip'
            changed_documents = 0
            with zipfile.ZipFile(archive) as source, zipfile.ZipFile(changed, 'w') as target:
                for member in source.infolist():
                    data = source.read(member)
                    if data.startswith(b'---\n'):
                        header, body = data.decode()[4:].split('\n---\n', 1)
                        metadata = yaml.safe_load(header)
                        if 'ariadne_export' in metadata:
                            metadata['ariadne_export']['models']['profile_hash'] = 'wrong-pair'
                            data = ('---\n' + yaml.safe_dump(metadata) + '---\n' + body).encode()
                            changed_documents += 1
                    target.writestr(member, data)
            assert changed_documents == 1
            archive = changed

        destination = tmp_path / 'imported.db'
        requests_before = len(embed.requests)
        result = invoke(
            monkeypatch, '--db', str(destination), '--staleness-db',
            str(tmp_path / 'imported-staleness.db'), 'import', str(archive), '--yes',
        )
        assert result == int(mismatched)
        with Library(destination) as library:
            if mismatched:
                assert library.count_documents() == 0
                assert len(embed.requests) == requests_before
            else:
                restored = library.get_document(original.id)
                assert restored.content == original.content
                assert restored.metadata == original.metadata
                assert restored.source_name == original.source_name
                assert restored.embedding is not None
                assert library.get_sections(restored.id)


def test_literal_detector_does_not_skip_cleanup_when_only_prior_calls_exist(tmp_path, monkeypatch):
    from docgen.scip_persist import persist_literal_url_clients

    path = tmp_path / 'catalog.db'
    with Library(path):
        pass
    with sqlite3.connect(path) as conn:
        conn.execute(
            'INSERT INTO http_client_calls '
            '(source_name, call_site_file, call_site_line, http_method, raw_url, sink_name, confidence) '
            "VALUES ('alpha', 'client.py', 1, 'GET', '/old', 'literal', 'high')"
        )
    visited = []

    def ingest_literal_url_clients(*, source_name, conn):
        visited.append(source_name)
        conn.execute('DELETE FROM http_client_calls WHERE source_name = ?', (source_name,))
        return 0

    module = types.ModuleType('docgen.scip_literal_url_extractor')
    module.ingest_literal_url_clients = ingest_literal_url_clients
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert persist_literal_url_clients(path, [('alpha', tmp_path), ('empty', tmp_path)]) == 0
    assert visited == ['alpha']
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM http_client_calls').fetchone()[0] == 0
