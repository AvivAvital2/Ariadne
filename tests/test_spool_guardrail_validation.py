"""Guard: a spool's SHIPPED guardrails must be environment knowledge.

A structural guardrail signal may key only on the environment's own API
(the allowlist, e.g. pyspark/mlflow/delta for databricks); a signal keyed
on a caller's own symbols is consumer knowledge and must never ship in a
spool. Prose (`nl`) guardrails always pass —
they name no symbols. ``validate_spool_guardrails`` is the gate; the last
test pins that the shipped databricks seed passes it (and would have
failed on the old consumer-keyed seed). Synthetic fixtures only.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from guardrails import (
    Guardrail,
    GuardrailError,
    load_guardrails,
    validate_spool_guardrails,
)

# Canonical environment API packages for the databricks spool's seed.
DATABRICKS_ENV_PACKAGES = ('pyspark', 'mlflow', 'delta', 'databricks')

_SEED = Path(__file__).resolve().parent.parent / 'spool_content' / 'databricks-seed-guardrails.yaml'


def _gr(signal_type, signal, *, name='g'):
    return Guardrail(
        name=name, kind='antipattern', recommendation='r', rationale='why',
        citation='designs/spool-environment-plugin.md §6',
        signal_type=signal_type, signal=signal,
    )


class TestValidateSpoolGuardrails:
    def test_prose_guardrail_always_passes(self):
        # nl names no symbols → allowed regardless of the allowlist.
        validate_spool_guardrails(
            [_gr('nl', {'kind': 'nl'})], allowed_prefixes=('mlflow',)
        )

    def test_environment_keyed_structural_passes(self):
        gr = _gr('structural',
                 {'kind': 'symbol_exists', 'pattern': 'mlflow.sklearn.autolog'})
        validate_spool_guardrails([gr], allowed_prefixes=DATABRICKS_ENV_PACKAGES)

    def test_consumer_keyed_structural_rejected(self):
        gr = _gr('structural',
                 {'kind': 'body_contains',
                  'symbol': 'demoproj.core.database.DbManager.on_disk',
                  'needle': 'read_only'},
                 name='leaky')
        with pytest.raises(GuardrailError) as exc:
            validate_spool_guardrails([gr], allowed_prefixes=DATABRICKS_ENV_PACKAGES)
        # The error must name the offending reference and the guardrail.
        assert 'demoproj.core.database.DbManager.on_disk' in str(exc.value)
        assert 'leaky' in str(exc.value)

    def test_nested_signals_are_checked(self):
        # A consumer symbol hidden inside all_of/any_of is still caught.
        gr = _gr('structural', {
            'kind': 'all_of',
            'signals': [
                {'kind': 'symbol_exists', 'pattern': 'pyspark.sql.DataFrame'},
                {'kind': 'signature_scan', 'prefix': 'demoproj.api.',
                 'needle': 'Duckdb', 'subtypes': ['method']},
            ],
        })
        with pytest.raises(GuardrailError):
            validate_spool_guardrails([gr], allowed_prefixes=DATABRICKS_ENV_PACKAGES)

    def test_empty_prefix_scan_is_allowed(self):
        # A scan-all (prefix '') keys on no namespace → environment-general.
        gr = _gr('structural',
                 {'kind': 'signature_scan', 'prefix': '', 'needle': 'SparkSession',
                  'subtypes': ['method']})
        validate_spool_guardrails([gr], allowed_prefixes=DATABRICKS_ENV_PACKAGES)

    def test_shipped_databricks_seed_is_environment_clean(self):
        # The regression guard on shipped content: the real seed must pass.
        # (A consumer-keyed seed — structural signals on a project's own
        # symbols — would fail this same check.)
        guardrails = load_guardrails(_SEED)
        assert guardrails, 'seed should declare guardrails'
        validate_spool_guardrails(guardrails, allowed_prefixes=DATABRICKS_ENV_PACKAGES)
