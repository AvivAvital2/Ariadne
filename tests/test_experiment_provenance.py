"""Runtime attribution: manifest coverage and database fingerprints.

A dirty checkout can only produce attributable measurements when every
imported runtime file — tracked or not — is in the manifest, and two
stores with identical counts must still be distinguishable by the strong
streamed fingerprint.
"""
from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPEC = importlib.util.spec_from_file_location(
    "exp_fingerprint",
    ROOT / "evaluation" / "chain-benchmark" / "exp_fingerprint.py")
exp_fingerprint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp_fingerprint)

from library.scip import init_scip_schema


class TestRuntimeManifest:
    def test_manifest_covers_runtime_and_marks_untracked_files(self):
        manifest = exp_fingerprint.runtime_manifest(ROOT)

        paths = {row["path"] for row in manifest["files"]}
        assert "library/chain_answer.py" in paths
        assert "ariadne_mcp/service_analysis.py" in paths
        assert "evaluation/chain-benchmark/shadow_eval.py" in paths
        by_path = {row["path"]: row for row in manifest["files"]}
        # This repo carries an untracked runtime module the dirty search
        # path imports; the manifest must expose it, not hide it.
        if "library/document_scope.py" in paths:
            assert by_path["library/document_scope.py"]["tracked"] is False
            assert ("library/document_scope.py"
                    in manifest["untracked_runtime_files"])
        assert manifest["manifest_sha256"]
        assert manifest["git_head"]

    def test_manifest_hash_changes_with_file_content(self, tmp_path):
        (tmp_path / "library").mkdir()
        (tmp_path / "library" / "a.py").write_text("x = 1\n")

        first = exp_fingerprint.runtime_manifest(tmp_path)
        (tmp_path / "library" / "a.py").write_text("x = 2\n")
        second = exp_fingerprint.runtime_manifest(tmp_path)

        assert first["manifest_sha256"] != second["manifest_sha256"]


def make_store(path, rows):
    connection = sqlite3.connect(path)
    init_scip_schema(connection)
    for canonical, qname, file in rows:
        connection.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (canonical, "src1", "x", file, 1, 9, "", "", qname, ""))
    connection.commit()
    connection.close()


class TestDatabaseFingerprints:
    def test_equal_counts_different_content_share_fast_but_not_strong(
            self, tmp_path):
        first = tmp_path / "a.db"
        second = tmp_path / "b.db"
        make_store(first, [("id1", "m.run", "m.py")])
        make_store(second, [("id2", "m.other", "o.py")])

        fast_first = exp_fingerprint.fast_db_fingerprint(first)
        fast_second = exp_fingerprint.fast_db_fingerprint(second)
        assert (fast_first["symbol_counts"]
                == fast_second["symbol_counts"])

        strong_first = exp_fingerprint.strong_db_fingerprint(
            first, cache_path=tmp_path / "cache.json")
        strong_second = exp_fingerprint.strong_db_fingerprint(
            second, cache_path=tmp_path / "cache.json")
        assert (strong_first["strong_sha256"]
                != strong_second["strong_sha256"])

    def test_strong_fingerprint_caches_by_fast_identity(self, tmp_path):
        store = tmp_path / "a.db"
        make_store(store, [("id1", "m.run", "m.py")])
        cache = tmp_path / "cache.json"

        first = exp_fingerprint.strong_db_fingerprint(
            store, cache_path=cache)
        second = exp_fingerprint.strong_db_fingerprint(
            store, cache_path=cache)

        assert first["cache"] == "miss"
        assert second["cache"] == "hit"
        assert first["strong_sha256"] == second["strong_sha256"]


class TestEffectiveConfiguration:
    def test_configuration_is_introspected_not_hardcoded(self):
        configuration = exp_fingerprint.effective_configuration()

        assert configuration["expansion"]["forward_depth"] == 0
        assert configuration["expansion"]["depth"] == 2
        assert (configuration["forward_traversal_enabled_by_default"]
                is False)
        assert configuration["body_plan_application"] == "diagnostic-only"
