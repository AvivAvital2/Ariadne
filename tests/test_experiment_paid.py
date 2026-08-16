"""The paid preparer validates everything and executes nothing.

Every prepared command must survive being pasted: framework-only flags
are consumed here, only measure-runner-supported arguments are forwarded
(via the budget wrapper), the rendered command is shell-safe, and the
preparer itself never touches subprocess execution.
"""
from __future__ import annotations

import importlib.util
import json
import shlex
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evaluation" / "chain-benchmark"
sys.path.insert(0, str(BENCH))

SPEC = importlib.util.spec_from_file_location(
    "experiment", BENCH / "experiment.py")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def load(name):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp_seal = load("exp_seal")
exp_certificate = load("exp_certificate")
exp_fingerprint = load("exp_fingerprint")

from library.scip import init_scip_schema


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "store.db"
    connection = sqlite3.connect(path)
    init_scip_schema(connection)
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def certificate_path(tmp_path, store):
    fingerprint = exp_fingerprint.fast_db_fingerprint(store)
    payload = exp_seal.seal({
        "schema": "ariadne-blind-shadow-v3",
        "mode": "blind-facet",
        "provenance": {
            "runtime_manifest": {
                "manifest_sha256": "m", "untracked_runtime_files": []},
            "database_fingerprint": {**fingerprint, "level": "strong",
                                     "strong_sha256": "s"},
            "questions_sha256": "q", "embedding_cache_sha256": "e",
            "command": {"questions": "questions.json"}},
        "questions": [{"id": 4, "resolution_census": {
            "exact": 1, "ambiguous": 0, "missing": 0}, "ledger": "x"}],
    })
    grade = {"stage_vector": {stage: 9 for stage in (
        "store", "raw", "menu", "retained", "materialized", "ledger",
        "final")}, "claims": []}
    certificate = exp_certificate.issue(
        certificate_type="paid-canary-eligibility",
        verdict={"passed": True, "reasons": ["all canary gates satisfied"]},
        grade_report=grade, artifacts_payload=payload,
        cost_projection={"usd_per_question": 0.2,
                         "calls_per_question": 2},
        issued_at="2026-08-12T00:00:00Z")
    path = tmp_path / "certificate.json"
    path.write_text(json.dumps(certificate))
    return path


def prepare(arguments, monkeypatch, *, nonce="fresh"):
    if nonce:
        monkeypatch.setenv("ARIADNE_PAID_RUN_NONCE", nonce)
    else:
        monkeypatch.delenv("ARIADNE_PAID_RUN_NONCE", raising=False)
    return experiment.prepare_paid(arguments)


class TestPreparerRefusals:
    def test_missing_allow_paid_is_refused(self, monkeypatch, capsys):
        assert prepare(["--max-usd", "2", "--only", "4"],
                       monkeypatch) == 2
        assert "--allow-paid" in capsys.readouterr().out

    def test_zero_or_negative_budget_is_refused(self, monkeypatch, capsys):
        assert prepare(
            ["--allow-paid", "--max-usd", "0", "--only", "4"],
            monkeypatch) == 2
        assert "positive dollar budget" in capsys.readouterr().out

    def test_empty_and_malformed_only_are_refused(
            self, monkeypatch, capsys):
        assert prepare(
            ["--allow-paid", "--max-usd", "2", "--only", ""],
            monkeypatch) == 2
        assert "explicit question ids" in capsys.readouterr().out
        assert prepare(
            ["--allow-paid", "--max-usd", "2", "--only", "4,six"],
            monkeypatch) == 2
        assert "malformed" in capsys.readouterr().out

    def test_missing_nonce_is_refused(self, monkeypatch, capsys):
        assert prepare(
            ["--allow-paid", "--max-usd", "2", "--only", "4"],
            monkeypatch, nonce="") == 2
        assert "NONCE" in capsys.readouterr().out

    def test_missing_certificate_is_refused(self, monkeypatch, capsys):
        assert prepare(
            ["--allow-paid", "--max-usd", "2", "--only", "4"],
            monkeypatch) == 2
        assert "--certificate" in capsys.readouterr().out

    def test_stale_database_is_refused(
            self, monkeypatch, capsys, certificate_path, store):
        store.write_bytes(store.read_bytes() + b"\\x00")

        code = prepare(
            ["--allow-paid", "--max-usd", "2", "--only", "4",
             "--certificate", str(certificate_path),
             "--db", str(store)],
            monkeypatch)

        assert code == 2
        assert "stale" in capsys.readouterr().out

    def test_tampered_certificate_is_refused(
            self, monkeypatch, capsys, certificate_path, store):
        certificate = json.loads(certificate_path.read_text())
        certificate["passed"] = True
        certificate["limits"]["max_usd_per_question"] = 99
        certificate_path.write_text(json.dumps(certificate))

        code = prepare(
            ["--allow-paid", "--max-usd", "2", "--only", "4",
             "--certificate", str(certificate_path),
             "--db", str(store)],
            monkeypatch)

        assert code == 2
        assert "certificate rejected" in capsys.readouterr().out


class TestPreparedCommand:
    def test_command_is_shell_safe_and_forwards_no_framework_flags(
            self, monkeypatch, capsys, certificate_path, store, tmp_path):
        monkeypatch.setattr(
            experiment.subprocess, "call",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("preparer executed a subprocess")))

        code = prepare(
            ["--allow-paid", "--max-usd", "2.5", "--only", "4,6",
             "--certificate", str(certificate_path),
             "--db", str(store),
             "--answers", str(tmp_path / "answers.json")],
            monkeypatch)

        output = capsys.readouterr().out
        assert code == 0
        line = next(l for l in output.splitlines()
                    if "paid_canary_runner.py" in l)
        tokens = shlex.split(line.strip())
        assert "--allow-paid" not in tokens
        assert "--db" not in tokens
        assert "--only" in tokens and "4,6" in tokens
        assert "--max-usd" in tokens
        assert "PREPARED (not executed)" in output

    def test_existing_output_paths_are_never_overwritten(
            self, monkeypatch, capsys, certificate_path, store, tmp_path):
        existing = tmp_path / "answers.json"
        existing.write_text("[]")

        code = prepare(
            ["--allow-paid", "--max-usd", "2", "--only", "4",
             "--certificate", str(certificate_path),
             "--db", str(store), "--answers", str(existing)],
            monkeypatch)

        assert code == 2
        assert "will not be overwritten" in capsys.readouterr().out
