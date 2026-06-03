"""Swagger 2.0 spec ingestion (Wave 4, Phase 7c).

Reads OpenAPI/Swagger 2.0 specs (JSON or YAML), extracts endpoint
declarations, attempts to bind ``operationId`` back to scip_symbols
via a convention-based strategy, and persists rows to the
``api_endpoints`` table.

Scope:
- Swagger 2.0 ONLY for this slice (per scoping decision). OpenAPI 3
  has different structure (``paths.<p>.<method>.requestBody``,
  ``components/schemas``, etc.) and is deferred.
- Convention-based binding only: ``operationId`` matches a symbol's
  ``display_name`` directly, with a camelCase ↔ snake_case fallback.
  Annotation-based binding (parsing SCIP ``signature_documentation``)
  and manual override map are out of scope here; they're easy to add
  later as additional resolver tiers.

Endpoints whose operationId can't be bound are still persisted with
``producer_symbol_id=NULL`` — the agent layer (and ``ariadne gaps``)
can surface unbound endpoints as items needing attention.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen

if TYPE_CHECKING:
    from sqlite3 import Connection


_HTTP_METHODS = {'get', 'post', 'put', 'delete', 'patch', 'options', 'head'}


@frozen
class EndpointDecl:
    """A single endpoint declared in a Swagger spec."""
    method: str
    path: str
    operation_id: str | None = None
    summary: str = ''


def parse_swagger_spec(spec_path: Path) -> list[EndpointDecl]:
    """Parse a Swagger 2.0 spec file (YAML or JSON) into a flat list
    of endpoint declarations.

    Raises ``FileNotFoundError`` if the file is missing, and any
    YAML/JSON parse error if the contents can't be decoded.
    """
    if not spec_path.exists():
        raise FileNotFoundError(f'Swagger spec not found: {spec_path}')

    text = spec_path.read_text(encoding='utf-8')
    suffix = spec_path.suffix.lower()
    if suffix in ('.json',):
        spec = json.loads(text)
    else:
        # Default to YAML — handles .yaml, .yml, no extension, etc.
        import yaml
        spec = yaml.safe_load(text)

    if not isinstance(spec, dict):
        raise ValueError(
            f'Swagger spec at {spec_path} is not a top-level mapping',
        )

    paths = spec.get('paths', {}) or {}
    out: list[EndpointDecl] = []
    for path_template, path_obj in paths.items():
        if not isinstance(path_obj, dict):
            continue
        for method, op in path_obj.items():
            if method.lower() not in _HTTP_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            out.append(EndpointDecl(
                method=method.upper(),
                path=path_template,
                operation_id=op.get('operationId'),
                summary=op.get('summary', '') or '',
            ))
    return out


_CAMEL_TO_SNAKE_RE = re.compile(r'(?<!^)(?=[A-Z])')


def _camel_to_snake(name: str) -> str:
    """``validateToken`` → ``validate_token``."""
    return _CAMEL_TO_SNAKE_RE.sub('_', name).lower()


def bind_operation_id(
    operation_id: str,
    *,
    symbols_by_display: dict[str, str],
) -> str | None:
    """Resolve an ``operationId`` to a scip_symbols canonical_id via
    convention. Returns the canonical_id, or None if no match.

    Strategies, in order:

    1. Direct: ``symbols_by_display[operation_id]``
    2. Snake-case fallback: ``symbols_by_display[snake_case(operation_id)]``
       — handles the case where the producer's symbol is in a
       Python/Scala consumer's snake_case namespace.

    Future tiers (annotation hints, manual overrides) can be added as
    additional fallbacks without changing this contract.
    """
    direct = symbols_by_display.get(operation_id)
    if direct is not None:
        return direct

    snake = _camel_to_snake(operation_id)
    if snake != operation_id:
        return symbols_by_display.get(snake)

    return None


def _endpoint_id(source_name: str, method: str, path_template: str) -> str:
    """Construct a stable endpoint_id for the api_endpoints PK."""
    return f'{source_name}:{method} {path_template}'


def ingest_swagger_for_source(
    *,
    source_name: str,
    swagger_paths: Iterable[Path],
    conn: 'Connection',
) -> int:
    """Parse each Swagger spec under ``swagger_paths``, bind operationIds
    against the ``scip_symbols`` table, persist rows to ``api_endpoints``.

    Idempotent: prior endpoints for ``source_name`` are deleted before
    fresh insertion, so re-runs reflect the latest spec without ghosts.

    Returns the count of endpoints persisted.
    """
    # Build symbols_by_display from scip_symbols rows. We index by
    # display_name for O(1) lookup during binding.
    symbols_by_display: dict[str, str] = {}
    for canonical_id, display_name in conn.execute(
        'SELECT canonical_id, display_name FROM scip_symbols '
        'WHERE source_name = ?',
        (source_name,),
    ):
        if display_name:
            # Last writer wins — for ambiguous display names (e.g.,
            # two methods named 'login' in different classes), we
            # accept the first-stored mapping. Real disambiguation is
            # an annotation-tier concern.
            symbols_by_display.setdefault(display_name, canonical_id)

    # Wipe prior rows for this source before re-ingesting.
    conn.execute(
        'DELETE FROM api_endpoints WHERE source_name = ?',
        (source_name,),
    )

    count = 0
    for spec_path in swagger_paths:
        endpoints = parse_swagger_spec(Path(spec_path))
        for ep in endpoints:
            producer_id: str | None = None
            if ep.operation_id:
                producer_id = bind_operation_id(
                    ep.operation_id,
                    symbols_by_display=symbols_by_display,
                )
            endpoint_id = _endpoint_id(
                source_name, ep.method, ep.path,
            )
            conn.execute(
                'INSERT OR REPLACE INTO api_endpoints VALUES '
                '(?, ?, ?, ?, ?, ?)',
                (
                    endpoint_id, source_name, ep.method, ep.path,
                    producer_id, 'swagger',
                ),
            )
            count += 1
    return count


__all__ = [
    'EndpointDecl',
    'bind_operation_id',
    'ingest_swagger_for_source',
    'parse_swagger_spec',
]
