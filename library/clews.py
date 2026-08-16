"""Clews: pre-generated routes through the call graph, embedded as their own vector surface.

``index -> fetch document -> curate bundle -> formulate with LLM -> return response``. This
module sits *before* the first stage, and exists because of a gap measured in that stage:
``scip_edges`` holds 2.78M edges and **no embedding column**, so a question can match a symbol
— every symbol has an embedded catalog document — but never a *path*. Retrieval therefore
chooses where a walk starts and nothing else, and ``chain_from_seeds`` has never seen the
question.

A clew closes that. It is a route the walk already found, stored with the text a question can
match:

    fit -> withTransformEvent -> listenerBus -> getOrCreate

Measured on the databricks pack: generating clews from three strategies and pooling them
covers **92.8%** of the symbols answer keys require, against 66.0% for one live walk — because
different anchors find different routes. Generation is local and takes minutes; embedding the
whole set costs about $0.14.

**Why not a document ``content_type``.** Twenty-two files enumerate content types, so a
``'clew'`` type would inherit provenance weighting, gap analysis, doc-type pickers and export,
and any filter that lists types would drop clews without a word. A route is not prose and must
not be ranked against prose. ``sections`` is the precedent for what this is instead: a
non-document table carrying its own ``embedding``, queried deliberately.

**What a clew is not.** It is not evidence. The route's symbols seed a walk, and the walk's
hops carry the documents an answer cites. A clew positions; it does not explain — measured,
clews alone contain 57.7% of required symbols against a walk's 66.0%, so replacing the walk
with them would lose recall. Their value is where the walk *starts*.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from attrs import field, frozen

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlite3 import Connection

    from numpy.typing import NDArray

#: One row per distinct route. ``steps`` is the embedded surface — the collapsed display names
#: a question is compared against — while ``route`` keeps the fully qualified symbols and
#: ``files`` keeps what a walk needs to seed from. ``hops`` is not stored: it is
#: ``len(route) - 1`` and a second copy could disagree with the first.
CLEWS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS clews (
    id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    entry_symbol TEXT NOT NULL,
    steps TEXT NOT NULL,
    route TEXT NOT NULL,
    files TEXT NOT NULL,
    strategy TEXT NOT NULL,
    question TEXT,
    embedding BLOB
)
'''

CLEWS_INDEXES = '''
CREATE INDEX IF NOT EXISTS idx_clews_source ON clews(source_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_clews_route ON clews(source_name, route);
'''


@frozen
class Clew:
    """A route, the text that retrieves it, and the files that seed a walk from it."""

    id: str
    source_name: str
    entry_symbol: str
    steps: list = field(factory=list)
    route: list = field(factory=list)
    files: list = field(factory=list)
    strategy: str = ''
    #: The generated question this route answers, when one was generated. It is the better
    #: embedding surface — matching question-to-question beats matching question-to-route —
    #: but it costs an LLM call per clew, so a pack may carry routes without one.
    question: str | None = None

    @property
    def hops(self) -> int:
        return max(len(self.route) - 1, 0)


def init_clews_schema(conn: 'Connection') -> None:
    """Create the clew table and its indexes. Idempotent, like every schema call here."""
    conn.execute(CLEWS_SCHEMA)
    conn.executescript(CLEWS_INDEXES)


def _clew_id(source_name: str, route: list) -> str:
    from schema import generate_deterministic_id
    return generate_deterministic_id('clew', f'{source_name}:{" -> ".join(route)}')


def add_clew(
    conn: 'Connection',
    *,
    source_name: str,
    entry_symbol: str,
    steps: list,
    route: list,
    files: list,
    strategy: str = '',
    question: str | None = None,
    embedding: 'NDArray[np.float32] | None' = None,
) -> str:
    """Store one route, replacing any earlier copy of the same route.

    Dedup belongs here rather than in a generator because pooling strategies is the point:
    measured, three strategies together cover 14 points more than the best one alone, and they
    overlap by construction. The id is derived from the route, so the same route from two
    strategies is one row however it arrives.
    """
    clew_id = _clew_id(source_name, route)
    conn.execute(
        'INSERT INTO clews (id, source_name, entry_symbol, steps, route, files, strategy, '
        'question, embedding) VALUES (?,?,?,?,?,?,?,?,?) '
        'ON CONFLICT(id) DO UPDATE SET steps=excluded.steps, files=excluded.files, '
        'strategy=excluded.strategy, '
        'question=COALESCE(excluded.question, clews.question), '
        'embedding=COALESCE(excluded.embedding, clews.embedding)',
        (clew_id, source_name, entry_symbol, json.dumps(steps), json.dumps(route),
         json.dumps(files), strategy, question,
         None if embedding is None else np.asarray(embedding, dtype=np.float32).tobytes()))
    return clew_id


def _row_to_clew(row) -> Clew:
    return Clew(
        id=row['id'], source_name=row['source_name'], entry_symbol=row['entry_symbol'],
        steps=json.loads(row['steps']), route=json.loads(row['route']),
        files=json.loads(row['files']), strategy=row['strategy'] or '',
        question=row['question'])


def clews_for(conn: 'Connection', *, source_name: str) -> list[Clew]:
    """Every clew owned by one source, in insertion order."""
    conn.row_factory = __import__('sqlite3').Row
    return [_row_to_clew(row) for row in conn.execute(
        'SELECT id, source_name, entry_symbol, steps, route, files, strategy, question '
        'FROM clews WHERE source_name = ?', (source_name,))]


def nearest_clews(
    conn: 'Connection',
    query: 'NDArray[np.float32]',
    *,
    source_name: str,
    top_k: int = 5,
) -> list[Clew]:
    """The ``top_k`` routes closest to ``query`` by cosine, embedded ones only.

    A clew with no embedding is skipped rather than treated as a zero vector. Generation is
    local and embedding needs a provider key, so a pack can legitimately hold routes that are
    stored but not yet embedded; scoring those as distance-zero would make them the answer to
    every question.
    """
    conn.row_factory = __import__('sqlite3').Row
    norm = float(np.linalg.norm(query))
    if norm == 0.0:
        return []
    scored: list[tuple[float, Clew]] = []
    for row in conn.execute(
            'SELECT id, source_name, entry_symbol, steps, route, files, strategy, question, '
            'embedding FROM clews WHERE source_name = ? AND embedding IS NOT NULL',
            (source_name,)):
        vector = np.frombuffer(row['embedding'], dtype=np.float32)
        if vector.size != query.size:
            continue
        magnitude = float(np.linalg.norm(vector))
        if magnitude == 0.0:
            continue
        scored.append((float(np.dot(vector, query)) / (magnitude * norm),
                       _row_to_clew(row)))
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [clew for _, clew in scored[:top_k]]


def seeds_from_clews(clews: list[Clew]) -> list[str]:
    """The files a matched clew hands to seeding, in route order and deduplicated.

    Files rather than symbols, because that is the currency ``seeds_from_documents`` already
    resolves — a clew positions the existing walk instead of replacing its seeding rule.
    """
    seen: list[str] = []
    for clew in clews:
        for path in clew.files:
            if path not in seen:
                seen.append(path)
    return seen
