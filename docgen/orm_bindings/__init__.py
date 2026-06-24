"""ORM binding engine + per-ORM strategies (design §5).

A single shared engine (``engine.py``) discovers models from a SCIP index —
anchored on definition symbols, metadata read from source — and emits
``schema_symbols``; each ORM is a small strategy supplying only the per-ORM
delta (``django.py`` etc.). The Strategy pattern keeps the invariant pipeline
in one place (§5).

**To add an ORM:** write a strategy implementing the ``OrmStrategy`` contract
(``name`` + ``detect`` / ``discover`` / ``recognize``), returning ``Table`` /
``Col`` from ``discover``; then add it to ``DEFAULT_STRATEGIES`` below. The
engine needs no change.
"""
from docgen.orm_bindings.django import DjangoStrategy
from docgen.orm_bindings.engine import (
    Col,
    OrmStrategy,
    SchemaBindResult,
    Table,
    persist_schema_symbols,
)
from docgen.orm_bindings.slick import SlickStrategy
from docgen.orm_bindings.sqlalchemy import SQLAlchemyStrategy

# The registry — the single place to register a new ORM (design §5.0).
DEFAULT_STRATEGIES: tuple[OrmStrategy, ...] = (
    DjangoStrategy(),
    SQLAlchemyStrategy(),
    SlickStrategy(),
)

__all__ = [
    'Col',
    'DEFAULT_STRATEGIES',
    'DjangoStrategy',
    'OrmStrategy',
    'SQLAlchemyStrategy',
    'SchemaBindResult',
    'SlickStrategy',
    'Table',
    'persist_schema_symbols',
]
