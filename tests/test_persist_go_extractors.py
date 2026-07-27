"""Tests for the Go persist wrappers (``persist_go_routes`` /
``persist_go_http_clients``) and their sink-registry wiring.

Wrapper-level dispatch + aggregation contract only; extractor logic is
covered in ``tests/test_scip_go_route_extractor.py`` and
``tests/test_scip_go_http_client_extractor.py``.
"""
from __future__ import annotations

from pathlib import Path

from docgen.scip_persist import persist_go_http_clients, persist_go_routes
from docgen.scip_sink_registry import DEFAULT_SINK_REGISTRY


def _make_db(tmp_path: Path) -> Path:
    from library import Library

    db_path = tmp_path / 'ariadne.db'
    lib = Library(db_path)
    lib.close()
    return db_path


def test_persist_go_routes_invokes_and_aggregates(tmp_path, monkeypatch):
    db_path = _make_db(tmp_path)
    seen: list[str] = []
    counts = iter([2, 0, 5])

    def _spy(*, source_name, source_root, conn, index_factory=None):
        seen.append(source_name)
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_go_route_extractor.ingest_go_routes', _spy,
    )
    total = persist_go_routes(
        db_path,
        [('a', tmp_path / 'a'), ('b', tmp_path / 'b'), ('c', tmp_path / 'c')],
    )
    assert seen == ['a', 'b', 'c']
    assert total == 7


def test_persist_go_http_clients_invokes_and_aggregates(tmp_path, monkeypatch):
    db_path = _make_db(tmp_path)
    counts = iter([1, 4])

    def _spy(*, source_name, source_root, conn, index_factory=None):
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_go_http_client_extractor.ingest_go_http_clients', _spy,
    )
    total = persist_go_http_clients(
        db_path, [('svc1', tmp_path / 's1'), ('svc2', tmp_path / 's2')],
    )
    assert total == 5


def test_go_sinks_registered_for_language_go():
    """The registry recognizes Go net/http client primitives under
    ``language='go'`` — the production classification path."""
    get = DEFAULT_SINK_REGISTRY.matching_symbol(
        'scip-go gomod std go1 net/http/Get().', language='go',
    )
    assert get is not None and get.http_method == 'GET' and get.arg_index == 0
    newreq = DEFAULT_SINK_REGISTRY.matching_symbol(
        'scip-go gomod std go1 net/http/NewRequest().', language='go',
    )
    assert newreq is not None and newreq.arg_index == 1
    # a Go symbol must not match under another language's filter
    assert DEFAULT_SINK_REGISTRY.matching_symbol(
        'scip-go gomod std go1 net/http/Get().', language='python',
    ) is None
