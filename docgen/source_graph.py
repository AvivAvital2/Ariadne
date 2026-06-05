"""Persist the configured source dependency graph into the library DB.

At build time, Ariadne knows each source's relational fields (``depends_on`` /
``parent`` / ``branches``) from ``ariadne.yaml``. This module snapshots that
graph into the ``source_relations`` table so the database becomes
self-describing: a serving box can resolve scope (and reproduce
``get_effective_dependencies``) from the DB alone, without restating the graph —
or any source paths — in its own ``ariadne.yaml``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config
    from library import Library


def persist_source_graph(cfg: Config, library: Library) -> None:
    """Snapshot every configured source's RAW relational fields into the DB.

    Persists ``depends_on`` / ``parent`` / ``branches`` verbatim — NOT the
    pre-merged effective dependency list. Storing the raw fields is what lets a
    serving box reproduce ``get_effective_dependencies`` (parent injection +
    branch-filtering) from the database. The store uses REPLACE semantics, so
    each build records the current config rather than accreting stale edges.
    """
    for name in cfg.sources:
        sc = cfg.get_source_config(name)
        if sc is None:
            continue
        library.set_source_relations(
            name,
            depends_on=sc.depends_on,
            parent=sc.parent,
            branches=sc.branches,
        )
