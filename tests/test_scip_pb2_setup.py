"""Guard test: scip_pb2 import behavior (SCIP plan, Phase A.4).

Asserts that when ``docgen.scip.scip_pb2`` is present (after running
``tests/setup_scip_pb2.py``), it imports cleanly and exposes the
``Index``, ``Document``, ``SymbolInformation``, and ``Occurrence``
message types the extractor depends on. Skips when absent so the test
suite stays green for contributors who haven't run the setup yet.
"""
from __future__ import annotations

import importlib

import pytest


def _scip_pb2_present() -> bool:
    try:
        importlib.import_module('docgen.scip.scip_pb2')
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _scip_pb2_present(),
    reason=(
        'scip_pb2 not generated; run '
        '`uv run python tests/setup_scip_pb2.py` to enable Scala/Java extraction'
    ),
)
class TestScipPb2:
    def test_index_message_exists(self) -> None:
        from docgen.scip import scip_pb2

        # Smoke check the public message types the extractor needs.
        for name in ('Index', 'Document', 'SymbolInformation', 'Occurrence'):
            assert hasattr(scip_pb2, name), (
                f'scip_pb2 missing expected message {name!r}; regenerate '
                f'with tests/setup_scip_pb2.py?'
            )

    def test_index_can_parse_empty_payload(self) -> None:
        """An empty Index protobuf round-trips through serialize/parse —
        sanity check that the generated module is functional.
        """
        from docgen.scip import scip_pb2

        empty = scip_pb2.Index()
        wire = empty.SerializeToString()
        parsed = scip_pb2.Index()
        parsed.ParseFromString(wire)
        # No documents in an empty index.
        assert len(parsed.documents) == 0
