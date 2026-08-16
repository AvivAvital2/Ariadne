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
import re

import json
from typing import TYPE_CHECKING

import numpy as np
from attrs import field, frozen, evolve

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
DEFAULT_MIN_CLEW_SIMILARITY = 0.25


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
@frozen
class ClewMatch:
    """One intact route and the similarity that retrieved it."""

    clew: Clew
    similarity: float
    obligations: tuple[int, ...] = ()
    obligation_texts: tuple[tuple[int, str], ...] = ()
    target_symbols: tuple[tuple[int, str], ...] = ()
    origin_rank: int | None = None
    structure_score: int = 0
@frozen
class ClewSymbolMenu:
    text: str = ""
    labels: dict = field(factory=dict)
    owner_entries: dict = field(factory=dict)


def clew_symbol_menu(conn, matches: list[ClewMatch], plan: str, *,
                     source_name: str, limit: int = 500) -> ClewSymbolMenu:
    """Build a compact endpoint menu from route neighbors and owner members."""
    route_names = tuple(sorted({name for match in matches for name in match.clew.route}))
    names = list(route_names)
    ids = {}
    parents = set()
    from library.structural_assembly import question_entity_symbol_seeds
    entity_ids = question_entity_symbol_seeds(
        conn, plan, source=source_name, limit=100)
    for start in range(0, len(entity_ids), 300):
        chunk = list(entity_ids[start:start + 300])
        marks = ",".join("?" * len(chunk))
        for canonical, qualified, parent in conn.execute(
                f"SELECT canonical_id,qualified_name,parent_qualified_name FROM scip_symbols "
                f"WHERE source_name=? AND canonical_id IN ({marks})",
                [source_name, *chunk]):
            ids[canonical] = qualified
            if qualified not in names:
                names.append(qualified)
            if parent:
                parents.add(parent)
    names.sort()
    for start in range(0, len(names), 300):
        chunk = names[start:start + 300]
        marks = ",".join("?" * len(chunk))
        for canonical, qualified, parent in conn.execute(
                f"SELECT canonical_id,qualified_name,parent_qualified_name FROM scip_symbols "
                f"WHERE source_name=? AND qualified_name IN ({marks})",
                [source_name, *chunk]):
            ids[canonical] = qualified
            if parent:
                parents.add(parent)
    candidates = set(names)
    # A selected route node may itself be an owner type. Include its exact SCIP
    # members just as we include siblings when the route node is already a member.
    parents.update(names)
    for start in range(0, len(parents), 300):
        chunk = sorted(parents)[start:start + 300]
        marks = ",".join("?" * len(chunk))
        candidates.update(row[0] for row in conn.execute(
            f"SELECT qualified_name FROM scip_symbols WHERE source_name=? "
            f"AND parent_qualified_name IN ({marks}) AND canonical_id NOT GLOB ?",
            [source_name, *chunk, "local *"]))
    for start in range(0, len(ids), 300):
        chunk = sorted(ids)[start:start + 300]
        marks = ",".join("?" * len(chunk))
        endpoint_ids = {row[0] for row in conn.execute(
            f"SELECT callee_canonical_id FROM scip_edges "
            f"WHERE caller_canonical_id IN ({marks}) "
            "AND edge_type IN ('call','type_ref','implements')", chunk)}
        endpoint_ids.update(row[0] for row in conn.execute(
            f"SELECT caller_canonical_id FROM scip_edges "
            f"WHERE callee_canonical_id IN ({marks}) "
            "AND edge_type IN ('call','type_ref','implements')", chunk))
        for endpoint_start in range(0, len(endpoint_ids), 300):
            endpoint_chunk = sorted(endpoint_ids)[endpoint_start:endpoint_start + 300]
            endpoint_marks = ",".join("?" * len(endpoint_chunk))
            candidates.update(row[0] for row in conn.execute(
                f"SELECT qualified_name FROM scip_symbols WHERE source_name=? "
                f"AND canonical_id IN ({endpoint_marks}) AND canonical_id NOT GLOB ?",
                [source_name, *endpoint_chunk, "local *"]))
    beam_tokens = set(_lexical_tokens(plan))
    def symbol_rank(name, tokens):
        segments = name.rsplit(".", 2)
        terminal_tokens = set(_lexical_tokens(segments[-1]))
        context_tokens = set(_lexical_tokens(".".join(segments[-2:])))
        qualified_tokens = set(_lexical_tokens(name))
        return (-len(context_tokens & tokens),
                -len(terminal_tokens & tokens),
                -len(qualified_tokens & tokens),
                len(context_tokens - tokens), name)
    candidates = (set(names) | set(sorted(
        candidates, key=lambda name: symbol_rank(name, beam_tokens))[:400]))
    frontier_ids = {}
    candidate_names = sorted(candidates)
    for start in range(0, len(candidate_names), 300):
        chunk = candidate_names[start:start + 300]
        marks = ",".join("?" * len(chunk))
        frontier_ids.update({row[0]: row[1] for row in conn.execute(
            f"SELECT canonical_id,qualified_name FROM scip_symbols WHERE source_name=? "
            f"AND qualified_name IN ({marks}) AND canonical_id NOT GLOB ?",
            [source_name, *chunk, "local *"])})
    for start in range(0, len(frontier_ids), 300):
        chunk = sorted(frontier_ids)[start:start + 300]
        marks = ",".join("?" * len(chunk))
        endpoint_ids = {row[0] for row in conn.execute(
            f"SELECT callee_canonical_id FROM scip_edges "
            f"WHERE caller_canonical_id IN ({marks}) "
            "AND edge_type IN ('call','type_ref','implements')", chunk)}
        endpoint_ids.update(row[0] for row in conn.execute(
            f"SELECT caller_canonical_id FROM scip_edges "
            f"WHERE callee_canonical_id IN ({marks}) "
            "AND edge_type IN ('call','type_ref','implements')", chunk))
        for endpoint_start in range(0, len(endpoint_ids), 300):
            endpoint_chunk = sorted(endpoint_ids)[endpoint_start:endpoint_start + 300]
            endpoint_marks = ",".join("?" * len(endpoint_chunk))
            candidates.update(row[0] for row in conn.execute(
                f"SELECT qualified_name FROM scip_symbols WHERE source_name=? "
                f"AND canonical_id IN ({endpoint_marks}) AND canonical_id NOT GLOB ?",
                [source_name, *endpoint_chunk, "local *"]))
    tokens = set(_lexical_tokens(plan))
    route_adjacent = set()
    route_ids = sorted(
        canonical for canonical, qualified in ids.items() if qualified in route_names)
    for canonical in route_ids[:64]:
        adjacent_ids = set()
        adjacent_ids.update(row[0] for row in conn.execute(
            "SELECT callee_canonical_id FROM scip_edges WHERE caller_canonical_id=? "
            "AND edge_type IN ('call','type_ref','implements') LIMIT 12", (canonical,)))
        adjacent_ids.update(row[0] for row in conn.execute(
            "SELECT caller_canonical_id FROM scip_edges WHERE callee_canonical_id=? "
            "AND edge_type IN ('call','type_ref','implements') LIMIT 12", (canonical,)))
        for start in range(0, len(adjacent_ids), 300):
            chunk = sorted(adjacent_ids)[start:start + 300]
            marks = ",".join("?" * len(chunk))
            route_adjacent.update(row[0] for row in conn.execute(
                f"SELECT qualified_name FROM scip_symbols WHERE source_name=? "
                f"AND canonical_id IN ({marks}) AND canonical_id NOT GLOB ?",
                [source_name, *chunk, "local *"]))
    # Keep every selected route legible before globally ranking neighbors. A
    # global cut can otherwise erase the one distinguishing symbol from a valid
    # route merely because another route is much longer.
    route_priority = []
    per_route = max(2, min(8, max(limit, 0) // max(len(matches), 1)))
    for match in matches:
        route = list(dict.fromkeys(match.clew.route))
        endpoints = route[:1] + route[-1:]
        middle = sorted(
            set(route) - set(endpoints),
            key=lambda name: symbol_rank(name, tokens))
        route_priority.extend((*endpoints, *middle[:max(per_route - len(endpoints), 0)]))
    selected_route_names = list(dict.fromkeys(route_priority))
    route_local = sorted(
        route_adjacent - set(selected_route_names),
        key=lambda name: symbol_rank(name, tokens))[:max(12, min(96, 12 * max(len(matches), 1)))]
    remaining_routes = sorted(
        set(names) - set(selected_route_names),
        key=lambda name: symbol_rank(name, tokens))
    supplemental = sorted(
        candidates - set(selected_route_names) - set(remaining_routes),
        key=lambda name: symbol_rank(name, tokens))
    ranked = list(dict.fromkeys(
        (*selected_route_names, *route_local, *remaining_routes, *supplemental)))[:max(limit, 0)]
    labels = {f"S{index}": name for index, name in enumerate(ranked, 1)}
    owner_entries = {}
    ranked_parents = {}
    for start in range(0, len(ranked), 300):
        chunk = ranked[start:start + 300]
        marks = ",".join("?" * len(chunk))
        for qualified, parent in conn.execute(
                f"SELECT qualified_name,parent_qualified_name FROM scip_symbols "
                f"WHERE source_name=? AND qualified_name IN ({marks})",
                [source_name, *chunk]):
            if parent:
                ranked_parents[qualified] = parent
    # Fetch the relevant owner subgraph once. Per-symbol recursive SQL is
    # prohibitively expensive on multi-million-edge indexes.
    member_by_id = {}
    members_by_parent = {}
    parent_names = sorted(set(ranked_parents.values()) | set(ranked_parents))
    for start in range(0, len(parent_names), 300):
        chunk = parent_names[start:start + 300]
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT canonical_id,qualified_name,parent_qualified_name,kind,"
                f"line_start,line_end,language,display_name FROM scip_symbols WHERE source_name=? "
                f"AND parent_qualified_name IN ({marks}) AND canonical_id NOT GLOB ?",
                [source_name, *chunk, "local *"]):
            member_by_id[row[0]] = row
            members_by_parent.setdefault(row[2], []).append(row)
    member_ids = set(member_by_id)
    callers_by_callee = {}
    ordered_ids = sorted(member_ids)
    for start in range(0, len(ordered_ids), 300):
        chunk = ordered_ids[start:start + 300]
        marks = ",".join("?" * len(chunk))
        for caller, callee in conn.execute(
                f"SELECT caller_canonical_id,callee_canonical_id FROM scip_edges "
                f"WHERE caller_canonical_id IN ({marks}) AND edge_type IN ('call','type_ref','implements')", chunk):
            if callee in member_ids:
                callers_by_callee.setdefault(callee, set()).add(caller)
    ids_by_qualified = {}
    for canonical, row in member_by_id.items():
        ids_by_qualified.setdefault(row[1], []).append(canonical)
    executable_kinds = {"Method", "Function", "Constructor"}
    for qualified, parent in ranked_parents.items():
        owner_parent = qualified if qualified in members_by_parent else parent
        frontier = [(canonical, 0) for canonical in ids_by_qualified.get(qualified, ())]
        visited = {canonical for canonical, _ in frontier}
        ancestors = []
        while frontier:
            canonical, depth = frontier.pop(0)
            if depth >= 6:
                continue
            for caller in callers_by_callee.get(canonical, ()):
                if caller in visited:
                    continue
                visited.add(caller)
                next_depth = depth + 1
                frontier.append((caller, next_depth))
                row = member_by_id.get(caller)
                if row and row[2] == owner_parent and row[3] in executable_kinds:
                    ancestors.append((next_depth, row))
        if ancestors:
            _, row = min(ancestors, key=lambda item: (
                -item[0], -(item[1][5] - item[1][4]), item[1][4], item[1][1]))
        else:
            siblings = [row for row in members_by_parent.get(owner_parent, ())
                        if row[3] in executable_kinds]
            # Scala Rule implementations are invoked through `apply`; generic rule
            # dispatch often leaves no concrete caller edge in SCIP. Treat that
            # compiler-level callable convention as structural metadata, then fall
            # back to body extent for languages without such a convention.
            scala_entries = [row for row in siblings
                             if row[6].lower() == "scala" and row[7] == "apply"]
            pool = scala_entries or siblings
            row = min(pool, key=lambda item: (
                -(item[5] - item[4]), item[4], item[1])) if pool else None
        if row:
            owner_entries[qualified] = row[1]
    lines = [
        "SCIP ENDPOINT SYMBOLS; map every obligation to its bridge endpoints."]
    lines.extend(f"  {label}. {name}" for label, name in labels.items())
    return ClewSymbolMenu(
        text="\n".join(lines), labels=labels, owner_entries=owner_entries)


def attach_symbol_targets(matches: list[ClewMatch], menu: ClewSymbolMenu,
                          reply: str) -> list[ClewMatch]:
    targets = {}
    for line in (reply or "").splitlines():
        obligations = [int(value) for value in re.findall(r"\bC(\d{1,2})\b", line, re.I)]
        chosen = [menu.labels[f"S{value}"] for value in
                  re.findall(r"\bS(\d{1,4})\b", line, re.I)
                  if f"S{value}" in menu.labels]
        symbols = []
        for symbol in chosen:
            entry = menu.owner_entries.get(symbol)
            for candidate in (entry, symbol):
                if candidate and candidate not in symbols:
                    symbols.append(candidate)
        for obligation in obligations:
            existing = targets.setdefault(obligation, [])
            for symbol in symbols:
                # Four selected roles may each contribute an enclosing executable
                # and the exact helper selected by the model.
                bound = 8 if menu.owner_entries else 4
                if symbol not in existing and len(existing) < bound:
                    existing.append(symbol)
    return [evolve(match, target_symbols=tuple(
        (obligation, symbol) for obligation in match.obligations
        for symbol in dict.fromkeys(targets.get(obligation, ())))) for match in matches]
@frozen
class ClewRejection:
    match: ClewMatch
    reason: str


@frozen
class ClewSelection:
    accepted: list[ClewMatch] = field(factory=list)
    rejected: list[ClewRejection] = field(factory=list)


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
def _lexical_tokens(text: str) -> tuple[str, ...]:
    stop = {"a", "an", "and", "are", "as", "at", "be", "by", "does",
            "for", "from", "how", "if", "in", "into", "is", "it", "of",
            "on", "or", "the", "this", "to", "was", "what", "when", "why",
            "with"}
    parts = re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+",
        text or "")
    normalized = []
    for part in parts:
        token = part.lower()
        if token in stop or len(token) < 2:
            continue
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        else:
            for suffix in ("ing", "ed", "ion", "es", "s"):
                if len(token) > len(suffix) + 3 and token.endswith(suffix):
                    token = token[:-len(suffix)]
                    break
        normalized.append(token)
    return tuple(normalized)


def lexical_clew_matches(conn: "Connection", question: str, *,
                         source_name: str, top_k: int = 12) -> list[ClewMatch]:
    """Recall clew routes locally when no query-embedding provider is available."""
    import math
    if top_k <= 0:
        return []
    conn.row_factory = __import__("sqlite3").Row
    question_tokens = set(_lexical_tokens(question))
    if not question_tokens:
        return []
    candidates = []
    frequencies = {token: 0 for token in question_tokens}
    for row in conn.execute(
            "SELECT id, source_name, entry_symbol, steps, route, files, strategy, "
            "question, embedding FROM clews WHERE source_name = ?",
            (source_name,)):
        clew = _row_to_clew(row)
        surface = " ".join((clew.question or "", *clew.steps, *clew.route))
        tokens = set(_lexical_tokens(surface))
        overlap = question_tokens.intersection(tokens)
        for token in overlap:
            frequencies[token] += 1
        candidates.append((clew, overlap))
    population = max(len(candidates), 1)
    scored = []
    denominator = sum(
        1.0 + math.log((population + 1) / (frequencies[token] + 1))
        for token in question_tokens)
    for clew, overlap in candidates:
        score = sum(
            1.0 + math.log((population + 1) / (frequencies[token] + 1))
            for token in overlap) / max(denominator, 1.0)
        scored.append(ClewMatch(clew=clew, similarity=score))
    scored.sort(key=lambda match: (-match.similarity, match.clew.id))
    return scored[:top_k]
def pseudo_semantic_clew_matches(conn: "Connection", question: str, *,
                                 source_name: str, top_k: int = 12,
                                 seed_k: int = 32) -> list[ClewMatch]:
    """Expand lexical seeds through the locally stored clew embedding space."""
    lexical = lexical_clew_matches(
        conn, question, source_name=source_name, top_k=max(seed_k, 0))
    weighted = []
    ids = [match.clew.id for match in lexical if match.similarity > 0]
    if ids:
        weights = {match.clew.id: match.similarity for match in lexical}
        for start in range(0, len(ids), 300):
            chunk = ids[start:start + 300]
            marks = ",".join("?" * len(chunk))
            for clew_id, blob in conn.execute(
                    f"SELECT id,embedding FROM clews WHERE id IN ({marks}) "
                    "AND embedding IS NOT NULL", chunk):
                vector = np.frombuffer(blob, dtype=np.float32)
                if vector.size:
                    weighted.append((weights[clew_id], vector))
    if not weighted:
        return lexical[:max(top_k, 0)]
    dimensions = {vector.size for _weight, vector in weighted}
    if len(dimensions) != 1:
        return lexical[:max(top_k, 0)]
    total = sum(weight for weight, _vector in weighted)
    centroid = sum((weight * vector for weight, vector in weighted),
                   np.zeros(next(iter(dimensions)), dtype=np.float32)) / max(total, 1e-9)
    semantic = nearest_clew_matches(
        conn, centroid, source_name=source_name, top_k=max(top_k, 0),
        min_similarity=-1.0)
    combined = []
    seen = set()
    for match in (*lexical, *semantic):
        if match.clew.id not in seen:
            seen.add(match.clew.id)
            combined.append(match)
        if len(combined) == max(top_k, 0):
            break
    return combined
_CLEW_EMBEDDING_CACHE = {}
_CLEW_EMBEDDING_CACHE_LIMIT = 2


def clear_clew_embedding_cache() -> None:
    """Drop process-local vector indexes, primarily for rebuilds and tests."""
    _CLEW_EMBEDDING_CACHE.clear()


def _clew_embedding_stamp(conn: "Connection", source_name: str,
                           dimensions: int) -> tuple:
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(rowid), 0), COALESCE(SUM(LENGTH(embedding)), 0) "
        "FROM clews WHERE source_name=? AND embedding IS NOT NULL "
        "AND LENGTH(embedding)=?",
        (source_name, dimensions * np.dtype(np.float32).itemsize)).fetchone()
    return tuple(int(value or 0) for value in row)


def _clew_database_identity(conn: "Connection") -> str:
    for _sequence, name, path in conn.execute("PRAGMA database_list"):
        if name == "main":
            return str(path) if path else f":memory:{id(conn)}"
    return f":connection:{id(conn)}"


def _build_clew_embedding_index(conn: "Connection", source_name: str,
                                dimensions: int, stamp: tuple) -> dict:
    """Read one source once and normalize its vectors in one NumPy operation."""
    expected = int(stamp[0])
    matrix = np.empty((expected, dimensions), dtype=np.float32)
    ids = []
    entries = []
    size = 0
    for clew_id, entry_symbol, blob in conn.execute(
            "SELECT id,entry_symbol,embedding FROM clews "
            "WHERE source_name=? AND embedding IS NOT NULL AND LENGTH(embedding)=? "
            "ORDER BY id",
            (source_name, dimensions * np.dtype(np.float32).itemsize)):
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size != dimensions:
            continue
        matrix[size] = vector
        ids.append(str(clew_id))
        entries.append(str(entry_symbol))
        size += 1
    matrix = matrix[:size]
    norms = np.linalg.norm(matrix, axis=1) if size else np.empty(0, dtype=np.float32)
    valid = norms > 0.0
    if size:
        np.divide(matrix, norms[:, None], out=matrix, where=valid[:, None])
        matrix[~valid] = 0.0
    return {
        "ids": tuple(ids), "entries": tuple(entries), "matrix": matrix,
        "valid": valid, "stamp": stamp,
    }


def _clew_embedding_index(conn: "Connection", source_name: str,
                          dimensions: int) -> dict:
    stamp = _clew_embedding_stamp(conn, source_name, dimensions)
    key = (_clew_database_identity(conn), source_name, dimensions, stamp)
    cached = _CLEW_EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached
    index = _build_clew_embedding_index(
        conn, source_name, dimensions, stamp)
    stale = [existing for existing in _CLEW_EMBEDDING_CACHE
             if existing[:3] == key[:3]]
    for existing in stale:
        _CLEW_EMBEDDING_CACHE.pop(existing, None)
    while len(_CLEW_EMBEDDING_CACHE) >= _CLEW_EMBEDDING_CACHE_LIMIT:
        _CLEW_EMBEDDING_CACHE.pop(next(iter(_CLEW_EMBEDDING_CACHE)))
    _CLEW_EMBEDDING_CACHE[key] = index
    return index
def nearest_clew_matches(
    conn: 'Connection',
    query: 'NDArray[np.float32]',
    *,
    source_name: str,
    top_k: int = 5,
    min_similarity: float = DEFAULT_MIN_CLEW_SIMILARITY,
) -> list[ClewMatch]:
    """Return scored routes above a threshold, diversified by entry point.

    The source vector surface is normalized once per database revision. Queries then use
    one matrix-vector operation; only the winning clew metadata is decoded from SQLite.
    """
    conn.row_factory = __import__('sqlite3').Row
    query = np.asarray(query, dtype=np.float32)
    norm = float(np.linalg.norm(query))
    if norm == 0.0 or top_k <= 0:
        return []
    index = _clew_embedding_index(conn, source_name, int(query.size))
    if not index["ids"]:
        return []
    similarities = index["matrix"] @ (query / norm)
    order = np.lexsort((np.asarray(index["ids"], dtype=str), -similarities))
    selected = []
    entries = set()
    for position in order:
        offset = int(position)
        if not bool(index["valid"][offset]):
            continue
        similarity = float(similarities[offset])
        if similarity < min_similarity:
            continue
        entry = index["entries"][offset]
        if entry in entries:
            continue
        entries.add(entry)
        selected.append((index["ids"][offset], similarity))
        if len(selected) == top_k:
            break
    rows = {}
    selected_ids = [clew_id for clew_id, _similarity in selected]
    for start in range(0, len(selected_ids), 300):
        chunk = selected_ids[start:start + 300]
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT id,source_name,entry_symbol,steps,route,files,strategy,question "
                f"FROM clews WHERE id IN ({marks})", chunk):
            rows[str(row["id"])] = row
    return [ClewMatch(clew=_row_to_clew(rows[clew_id]), similarity=similarity)
            for clew_id, similarity in selected if clew_id in rows]
def document_clew_matches(conn: "Connection", documents, question: str, *, source_name: str,
        limit: int = 48, indexer_cwds: tuple[str, ...] | None = None,
        source_root: str | None = None) -> list[ClewMatch]:
    """Fuse positioned documents with bounded SCIP routes without losing provenance.

    Document order is retrieval evidence.  Module/type identity is structural evidence.
    Keep both through route selection instead of flattening every document into one seed bag.
    """
    if indexer_cwds is None:
        from docgen.scip_paths import indexer_cwds as load_indexer_cwds
        indexer_cwds = load_indexer_cwds(source_root) if source_root else ()
    from library.structural_assembly import seeds_from_documents, question_ranked_seeds

    if limit <= 0:
        return []
    positioned = list(documents)
    seed_set = seeds_from_documents(
        conn, positioned, source=source_name, indexer_cwds=indexer_cwds,
        source_root=source_root)

    name_ranks: dict[str, int] = {}
    heading_mentions: dict[int, list[str]] = {}
    document_files: dict[int, list[str]] = {}
    for document_rank, document in enumerate(positioned):
        title = (document.get("title", "") if isinstance(document, dict)
                 else getattr(document, "title", ""))
        metadata = (document.get("metadata", {}) if isinstance(document, dict)
                    else getattr(document, "metadata", {})) or {}
        content = (document.get("content", "") if isinstance(document, dict)
                   else getattr(document, "content", "")) or ""
        files = (document.get("source_files", []) if isinstance(document, dict)
                 else getattr(document, "source_files", [])) or []
        document_files[document_rank] = [str(file) for file in files if file]
        mentions = heading_mentions.setdefault(document_rank, [])
        for line in str(content).splitlines():
            if not re.match(r"^#{1,6}\s+", line):
                continue
            for code_surface in re.findall(r"`([^`]+)`", line):
                names_in_surface = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code_surface)
                if names_in_surface and names_in_surface[-1] not in mentions:
                    mentions.append(names_in_surface[-1])
        identities = [title]
        if isinstance(metadata, dict):
            identities.extend((metadata.get("qualified_name"),
                               metadata.get("module_name")))
        for identity in identities:
            if identity:
                value = str(identity)
                name_ranks[value] = min(document_rank, name_ranks.get(value, document_rank))

    exact_rows = []
    names = list(name_ranks)
    for start in range(0, len(names), 300):
        chunk = names[start:start + 300]
        marks = ",".join("?" * len(chunk))
        exact_rows.extend(conn.execute(
            f"SELECT canonical_id,qualified_name FROM scip_symbols "
            f"WHERE source_name=? AND qualified_name IN ({marks}) "
            "AND canonical_id NOT GLOB ?",
            [source_name, *chunk, "local *"]))
    exact_rows.sort(key=lambda row: (name_ranks.get(str(row[1]), len(positioned)),
                                     str(row[1]), str(row[0])))
    origin_by_id = {
        str(canonical): name_ranks.get(str(qualified), len(positioned))
        for canonical, qualified in exact_rows
    }

    # A catalog module describes its owned methods.  Ownership is not a runtime call,
    # but it identifies the bodies whose call edges form the executable route.
    owner_names = {str(qualified): origin_by_id[str(canonical)]
                   for canonical, qualified in exact_rows}
    owner_by_origin: dict[int, str] = {}
    for qualified, origin in owner_names.items():
        current = owner_by_origin.get(origin)
        if current is None or qualified.count(".") < current.count("."):
            owner_by_origin[origin] = qualified
    member_rows = []
    owner_values = list(owner_names)
    for start in range(0, len(owner_values), 300):
        chunk = owner_values[start:start + 300]
        marks = ",".join("?" * len(chunk))
        member_rows.extend(conn.execute(
            f"SELECT canonical_id,qualified_name,parent_qualified_name,file,line_start,language "
            f"FROM scip_symbols WHERE source_name=? "
            f"AND parent_qualified_name IN ({marks}) AND canonical_id NOT GLOB ?",
            [source_name, *chunk, "local *"]))
    for canonical, _qualified, parent, _file, _line, _language in member_rows:
        origin_by_id[str(canonical)] = owner_names[str(parent)]
    member_name_by_id = {str(canonical): str(qualified).rsplit(".", 1)[-1]
                         for canonical, qualified, _parent, _file, _line, _language
                         in member_rows}
    member_details = {
        str(canonical): (str(qualified), str(file), int(line or 0), str(language or ""))
        for canonical, qualified, _parent, file, line, language in member_rows}

    # Spend root capacity across positioned documents before the aggregate fallback.
    by_origin: dict[int, list[str]] = {}
    for canonical, origin in origin_by_id.items():
        by_origin.setdefault(origin, []).append(canonical)
    roots: list[str] = []
    per_origin_limit = max(2, min(8, max(1, limit // max(len(by_origin), 1))))
    for origin in sorted(by_origin):
        mention_order = {name.lower(): index for index, name in enumerate(
            heading_mentions.get(origin, []))}
        highlighted = sorted(
            (canonical for canonical in by_origin[origin]
             if member_name_by_id.get(canonical, "").lower() in mention_order),
            key=lambda canonical: mention_order[
                member_name_by_id[canonical].lower()])
        ranked_origin = question_ranked_seeds(
            conn, by_origin[origin], question, source=source_name,
            limit=per_origin_limit)
        roots.extend((*highlighted, *ranked_origin)[:per_origin_limit])
    ranked_fallback = question_ranked_seeds(
        conn, seed_set.seeds, question, source=source_name, limit=limit)
    roots.extend(ranked_fallback)
    root_cap = min(24, max(1, limit // 2))
    roots = list(dict.fromkeys(roots))[:root_cap]

    located = {}
    for start in range(0, len(roots), 300):
        chunk = roots[start:start + 300]
        marks = ",".join("?" * len(chunk))
        for canonical, qualified, file, parent, owner in conn.execute(
                f"SELECT canonical_id,qualified_name,file,parent_qualified_name,source_name "
                f"FROM scip_symbols WHERE canonical_id IN ({marks})",
                chunk):
            if owner == source_name:
                located[str(canonical)] = (str(qualified), str(file), str(parent or ""))

    tokens = set(_lexical_tokens(question))

    def route_rank(name, edge_type, direction):
        parts = name.rsplit(".", 2)
        terminal = set(_lexical_tokens(parts[-1]))
        context = set(_lexical_tokens(".".join(parts[-2:])))
        return (-len(context & tokens), -len(terminal & tokens),
                0 if edge_type == "call" else 1,
                0 if direction == "incoming" else 1, name)

    edge_buckets: dict[tuple[str, str], list[tuple[str, str]]] = {}
    neighbor_ids: set[str] = set()
    for direction, root_column, endpoint_index in (
            ("incoming", "callee_canonical_id", "idx_scip_edges_callee"),
            ("outgoing", "caller_canonical_id", "idx_scip_edges_caller")):
        for start in range(0, len(roots), 300):
            chunk = roots[start:start + 300]
            marks = ",".join("?" * len(chunk))
            for caller, callee, edge_type in conn.execute(
                    f"SELECT caller_canonical_id,callee_canonical_id,edge_type "
                    f"FROM scip_edges INDEXED BY {endpoint_index} "
                    f"WHERE {root_column} IN ({marks}) "
                    "AND edge_type IN ('call','implements','type_ref')",
                    chunk):
                root = str(callee if direction == "incoming" else caller)
                neighbor = str(caller if direction == "incoming" else callee)
                if neighbor == root or neighbor.startswith("local "):
                    continue
                edge_buckets.setdefault((root, direction), []).append(
                    (neighbor, str(edge_type)))
                neighbor_ids.add(neighbor)

    # Some compilers encode a method passed as a callback as an identifier occurrence,
    # not a call edge. Recover only same-owner member references inside SCIP-owned bodies.
    if source_root:
        try:
            from pathlib import Path
            from ast_grep_py import SgRoot

            extension_language = {
                ".scala": "scala", ".java": "java", ".py": "python",
                ".js": "javascript", ".jsx": "javascript",
                ".ts": "typescript", ".tsx": "typescript", ".go": "go",
            }
            for origin, member_ids in sorted(by_origin.items()):
                direct_members = [canonical for canonical in member_ids
                                  if canonical in member_details]
                names: dict[str, list[str]] = {}
                for canonical in direct_members:
                    names.setdefault(member_name_by_id[canonical], []).append(canonical)
                unique_names = {name: ids[0] for name, ids in names.items()
                                if len(ids) == 1}
                spans = sorted((member_details[canonical][2], canonical)
                               for canonical in direct_members
                               if member_details[canonical][2] > 0)
                if not spans or not unique_names:
                    continue
                for relative in document_files.get(origin, []):
                    source_file = Path(source_root) / relative
                    if not source_file.is_file():
                        continue
                    language = extension_language.get(source_file.suffix.lower())
                    if not language:
                        continue
                    tree = SgRoot(source_file.read_text(errors="replace"), language).root()
                    for kind in ("identifier", "field_identifier", "simple_identifier"):
                        try:
                            nodes = tree.find_all(kind=kind)
                        except RuntimeError:
                            continue
                        for node in nodes:
                            target = unique_names.get(node.text())
                            if target is None:
                                continue
                            line = node.range().start.line + 1
                            caller = None
                            for index, (start_line, candidate) in enumerate(spans):
                                end_line = (spans[index + 1][0] - 1
                                            if index + 1 < len(spans) else 10 ** 9)
                                if start_line <= line <= end_line:
                                    caller = candidate
                            if caller is None or caller == target:
                                continue
                            edge_buckets.setdefault((target, "incoming"), []).append(
                                (caller, "reference"))
                            edge_buckets.setdefault((caller, "outgoing"), []).append(
                                (target, "reference"))
                            neighbor_ids.update((caller, target))
        except (ImportError, OSError, ValueError):
            pass

    neighbor_symbols = {}
    ordered_neighbor_ids = sorted(neighbor_ids)
    for start in range(0, len(ordered_neighbor_ids), 300):
        chunk = ordered_neighbor_ids[start:start + 300]
        marks = ",".join("?" * len(chunk))
        for canonical, qualified, file, parent, owner in conn.execute(
                f"SELECT canonical_id,qualified_name,file,parent_qualified_name,source_name "
                f"FROM scip_symbols WHERE canonical_id IN ({marks})", chunk):
            if owner == source_name:
                neighbor_symbols[str(canonical)] = (
                    str(qualified), str(file), str(parent or ""))

    per_root = []
    for root_index, canonical in enumerate(roots):
        located_root = located.get(canonical)
        if located_root is None:
            continue
        qualified, file, parent = located_root
        origin = origin_by_id.get(canonical, len(positioned) + root_index)
        document_owner = owner_by_origin.get(origin, "")
        root_direct = bool(document_owner and parent == document_owner)
        routes = [([qualified], int(root_direct))]
        neighbors = []
        for direction in ("incoming", "outgoing"):
            edge_rows = edge_buckets.get((canonical, direction), ())
            for neighbor_id, edge_type in edge_rows:
                neighbor_symbol = neighbor_symbols.get(neighbor_id)
                if neighbor_symbol is None:
                    continue
                neighbor, neighbor_file, neighbor_parent = neighbor_symbol
                normalized = f"/{neighbor_file.lower().strip('/')}"
                if any(part in normalized for part in (
                        "/test/", "/tests/", "/benchmark/", "/benchmarks/",
                        "/target/", "/generated/")):
                    continue
                if neighbor != qualified:
                    neighbors.append((neighbor, edge_type, direction, neighbor_parent))
        ordered = sorted(dict.fromkeys(neighbors), key=lambda row: route_rank(*row[:3]))
        incoming = [row for row in ordered if row[2] == "incoming"][:3]
        outgoing = [row for row in ordered if row[2] == "outgoing"][:3]
        for depth in range(max(len(incoming), len(outgoing))):
            for family in (incoming, outgoing):
                if depth >= len(family):
                    continue
                neighbor, edge_type, direction, neighbor_parent = family[depth]
                route = ([neighbor, qualified] if direction == "incoming"
                         else [qualified, neighbor])
                structure_score = (int(edge_type in ("call", "reference"))
                                   + int(root_direct)
                                   + int(bool(document_owner) and
                                         neighbor_parent == document_owner))
                if all(candidate != route for candidate, _score in routes):
                    routes.append((route, structure_score))
        per_root.append((canonical, file, routes, root_index))

    matches = []
    seen_routes: set[tuple[str, ...]] = set()
    depth = 0
    while len(matches) < limit and any(depth < len(routes)
                                       for _canonical, _file, routes, _index in per_root):
        for canonical, file, routes, root_index in per_root:
            if depth >= len(routes):
                continue
            route, structure_score = routes[depth]
            route_key = tuple(route)
            if route_key in seen_routes:
                continue
            seen_routes.add(route_key)
            matches.append(ClewMatch(clew=Clew(
                id=f"document:{canonical}:{depth}", source_name=source_name,
                entry_symbol=route[0], route=route, files=[file],
                strategy="document-scip"), similarity=1.0,
                origin_rank=origin_by_id.get(canonical,
                                             len(positioned) + root_index),
                structure_score=structure_score))
            if len(matches) == limit:
                break
        depth += 1
    return matches


def select_clew_matches(question: str, matches: list[ClewMatch]) -> ClewSelection:
    """Prefer routes compatible with explicit entities and account for every rejection.

    Sparse prose contains no trustworthy package constraint, so it keeps every scored route.
    Filtering activates only when at least one candidate overlaps a code-shaped question
    entity; otherwise retrieval remains conservative instead of rejecting all candidates.
    """
    production = []
    rejected = []
    for match in matches:
        normalized_files = [f"/{str(path).lower().strip('/')}" for path in match.clew.files]
        route_surface = " ".join(match.clew.route)
        nonproduction = any(
            segment in path
            for path in normalized_files
            for segment in ("/test/", "/tests/", "/src/test/", "/target/",
                            "/generated/", "/benchmark/", "/benchmarks/"))
        nonproduction = nonproduction or bool(re.search(
            r"(?:Suite|Spec|Tests?)(?:[.$#]|$)", route_surface))
        if nonproduction:
            rejected.append(ClewRejection(match=match, reason="non-production route"))
        else:
            production.append(match)
    matches = production
    entities = {
        token.strip('`').lower()
        for token in re.findall(
            r'`[^`]+`|(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*|[A-Z][A-Za-z0-9_]*[A-Z_][A-Za-z0-9_]*',
            question or '')
    }
    if not entities:
        return ClewSelection(accepted=list(matches), rejected=rejected)

    compatible: list[ClewMatch] = []
    incompatible: list[ClewMatch] = []
    for match in matches:
        surface = ' '.join([
            *match.clew.route,
            *match.clew.steps,
            match.clew.question or '',
        ]).lower()
        target = compatible if any(entity in surface for entity in entities) else incompatible
        target.append(match)
    if not compatible:
        return ClewSelection(accepted=list(matches), rejected=rejected)
    return ClewSelection(
        accepted=compatible,
        rejected=[*rejected, *[ClewRejection(match=match, reason='question entity mismatch')
                  for match in incompatible]],
    )
@frozen
class ClewFamilyMenu:
    text: str = ""
    labels: dict = field(factory=dict)
def clew_family_menu(matches: list[ClewMatch], *, question: str = "",
                     limit: int = 200) -> ClewFamilyMenu:
    """Rank semantic entry owners before exposing a bounded selection menu."""
    grouped = {}
    for match in matches:
        entry = match.clew.route[0] if match.clew.route else match.clew.entry_symbol
        owner = entry.rsplit(".", 1)[0] if "." in entry else entry
        grouped.setdefault(owner, []).append(match)
    question_tokens = set(_lexical_tokens(question))
    def family_evidence(item):
        owner, owner_matches = item
        surface = " ".join((
            owner, *(match.clew.question or "" for match in owner_matches[:3]),
            *(" ".join(match.clew.route) for match in owner_matches[:3])))
        overlap = len(question_tokens.intersection(_lexical_tokens(surface)))
        return overlap, max(match.similarity for match in owner_matches)
    families = sorted(grouped.items(), key=lambda item: (
        -(family_evidence(item)[0] >= 2),
        -family_evidence(item)[0],
        -family_evidence(item)[1],
        item[0]))
    labels = {}
    family_lines = []
    for owner, family in families[:max(limit, 0)]:
        label = f"F{len(labels) + 1}"
        labels[label] = tuple(family)
        skeletons = []
        representatives = sorted(family, key=lambda match: (
            not bool(match.clew.route and
                     match.clew.route[0].rsplit(".", 1)[-1] == "apply"),
            -len(match.clew.route), match.clew.id))
        for match in representatives[:3]:
            if not match.clew.route:
                continue
            entry = match.clew.route[0].rsplit(".", 1)[-1]
            terminal = match.clew.route[-1].rsplit(".", 1)[-1]
            skeleton = entry if entry == terminal else f"{entry} -> {terminal}"
            if skeleton not in skeletons:
                skeletons.append(skeleton)
        family_lines.append(
            f"  {label}. {owner} — {len(family)} route(s) — "
            + "; ".join(skeletons))
    lines = [
        "SCIP ROUTE FAMILIES; choose every owner needed by the fixed obligations.",
        *family_lines]
    return ClewFamilyMenu(text="\n".join(lines), labels=labels)
def complete_clew_family_selection(menu: ClewFamilyMenu, reply: str, *,
                                   minimum: int, limit: int = 6) -> list[ClewMatch]:
    """Preserve model choices but ensure distinct obligations see distinct owners."""
    labels = []
    for digits in re.findall(r"\bF(\d{1,4})\b", reply or "", re.I):
        label = f"F{digits}"
        if label in menu.labels and label not in labels:
            labels.append(label)
    for label in menu.labels:
        if len(labels) >= min(max(minimum, 0), max(limit, 0)):
            break
        if label not in labels:
            labels.append(label)
    return resolve_clew_families(menu, " ".join(labels), limit=limit)


def resolve_clew_families(menu: ClewFamilyMenu, reply: str, *,
                          limit: int = 6) -> list[ClewMatch]:
    """Interleave selected owners so one large family cannot starve the others."""
    families = []
    for digits in re.findall(r"\bF(\d{1,4})\b", reply or "", re.I):
        family = menu.labels.get(f"F{digits}")
        if family is not None and family not in families:
            families.append(family)
            if len(families) == max(limit, 0):
                break
    selected = []
    depth = 0
    while any(depth < len(family) for family in families):
        for family in families:
            if depth < len(family) and family[depth] not in selected:
                selected.append(family[depth])
        depth += 1
    return selected
@frozen
class ClewRouteMenu:
    text: str = ""
    labels: dict = field(factory=dict)
def clew_route_menu(matches: list[ClewMatch]) -> ClewRouteMenu:
    """Render selected clew skeletons with semantic symbols on every card."""
    labels = {}
    lines = [
        "SCIP ROUTE SKELETONS; choose every route required by the question."]
    for match in matches:
        label = f"K{len(labels) + 1}"
        labels[label] = match
        skeleton = " -> ".join(match.clew.route)
        lines.append(f"  {label}. {skeleton}")
    return ClewRouteMenu(text="\n".join(lines), labels=labels)
def resolve_clew_routes(menu: ClewRouteMenu, reply: str, *,
                        limit: int = 4) -> list[ClewMatch]:
    obligations_by_label = {}
    for line in (reply or "").splitlines():
        obligations = tuple(dict.fromkeys(
            int(value) for value in re.findall(r"\bC(\d{1,2})\b", line, re.I)))
        for digits in re.findall(r"\bK(\d{1,4})\b", line, re.I):
            label = f"K{digits}"
            obligations_by_label[label] = tuple(dict.fromkeys(
                (*obligations_by_label.get(label, ()), *obligations)))
    selected = []
    for digits in re.findall(r"\bK(\d{1,4})\b", reply or "", re.I):
        label = f"K{digits}"
        match = menu.labels.get(label)
        if match is not None and all(item.clew.id != match.clew.id for item in selected):
            selected.append(evolve(match, obligations=obligations_by_label.get(label, ())))
            if len(selected) == max(limit, 0):
                break
    return selected
def attach_obligation_texts(matches: list[ClewMatch], plan: str) -> list[ClewMatch]:
    """Attach the owner plan's natural-language target to every selected route."""
    descriptions = {}
    for line in (plan or "").splitlines():
        found = re.search(r"\bC(\d{1,2})\b", line, re.I)
        if found:
            descriptions[int(found.group(1))] = line.strip()
    return [evolve(match, obligation_texts=tuple(
        (obligation, descriptions.get(obligation, ""))
        for obligation in match.obligations)) for match in matches]
def covered_route_obligations(menu: ClewRouteMenu, reply: str) -> set[int]:
    """Require each obligation to own at least one non-reused valid route."""
    routes_by_obligation = {}
    obligations_by_route = {}
    for line in (reply or "").splitlines():
        obligations = {int(value) for value in
                       re.findall(r"\bC(\d{1,2})\b", line, re.I)}
        routes = {f"K{value}" for value in
                  re.findall(r"\bK(\d{1,4})\b", line, re.I)
                  if f"K{value}" in menu.labels}
        for obligation in obligations:
            routes_by_obligation.setdefault(obligation, set()).update(routes)
        for route in routes:
            obligations_by_route.setdefault(route, set()).update(obligations)
    return {obligation for obligation, routes in routes_by_obligation.items()
            if any(len(obligations_by_route[route]) == 1 for route in routes)}


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
def deterministic_clew_matches(
        question: str, matches: list[ClewMatch], *, limit: int = 8) -> list[ClewMatch]:
    """Select clause-covering semantic and positioned SCIP routes without an LLM."""
    if limit <= 0 or not matches:
        return []
    clauses = [set(_lexical_tokens(part)) for part in re.split(
        r"[?;:]|\b(?:and|but|while|whereas|versus|compared\s+with)\b",
        question or "", flags=re.I) if _lexical_tokens(part)]
    if not clauses:
        clauses = [set(_lexical_tokens(question))]
    question_tokens = set().union(*clauses) if clauses else set()

    def surface_tokens(match: ClewMatch) -> set[str]:
        return set(_lexical_tokens(" ".join((match.clew.entry_symbol,
                                             *match.clew.route,
                                             *match.clew.steps, ""))))

    tokens = {match.clew.id: surface_tokens(match) for match in matches}

    def rank(match: ClewMatch, clause: set[str] | None = None):
        route_tokens = tokens[match.clew.id]
        focused = clause if clause is not None else question_tokens
        overlap = len(focused.intersection(route_tokens))
        total = len(question_tokens.intersection(route_tokens))
        structural_relevance = (overlap + 2 * match.structure_score
                                if match.clew.strategy == "document-scip" else overlap)
        return (-structural_relevance, -overlap, -total,
                0 if len(match.clew.route) > 1 else 1,
                -match.similarity, len(match.clew.route), match.clew.id)

    selected: list[ClewMatch] = []
    semantic = [match for match in matches
                if match.clew.strategy != "document-scip"]
    for clause in clauses:
        if not semantic:
            break
        candidate = min(semantic, key=lambda match: rank(match, clause))
        if clause.intersection(tokens[candidate.clew.id]) and candidate not in selected:
            selected.append(candidate)

    positioned_by_origin: dict[int, list[ClewMatch]] = {}
    for match in matches:
        if match.clew.strategy == "document-scip":
            origin = match.origin_rank if match.origin_rank is not None else 10 ** 9
            positioned_by_origin.setdefault(origin, []).append(match)

    def stitch(candidate: ClewMatch, family: list[ClewMatch]) -> ClewMatch:
        """Join adjacent compiler edges from one positioned document."""
        route = list(candidate.clew.route)
        files = list(candidate.clew.files)
        used = {candidate.clew.id}
        structure_score = candidate.structure_score
        while route and len(route) < 6:
            predecessor = [item for item in family
                           if item.clew.id not in used and len(item.clew.route) > 1
                           and item.clew.route[-1] == route[0]]
            successor = [item for item in family
                         if item.clew.id not in used and len(item.clew.route) > 1
                         and item.clew.route[0] == route[-1]]
            changed = False
            if predecessor:
                item = min(predecessor, key=rank)
                route = [*item.clew.route[:-1], *route]
                files = list(dict.fromkeys((*item.clew.files, *files)))
                structure_score += item.structure_score
                used.add(item.clew.id)
                changed = True
            if successor and len(route) < 6:
                item = min(successor, key=rank)
                route = [*route, *item.clew.route[1:]]
                files = list(dict.fromkeys((*files, *item.clew.files)))
                structure_score += item.structure_score
                used.add(item.clew.id)
                changed = True
            if not changed:
                break
        if route == candidate.clew.route:
            return candidate
        return evolve(candidate, clew=evolve(
            candidate.clew, entry_symbol=route[0], route=route, files=files),
            structure_score=structure_score)

    document_slots = min(len(positioned_by_origin), max(1, limit // 2))
    for origin in sorted(positioned_by_origin)[:document_slots]:
        family = positioned_by_origin[origin]
        candidate = stitch(min(family, key=rank), family)
        if all(item.clew.id != candidate.clew.id for item in selected):
            selected.append(candidate)

    owners = {(match.clew.route[0] if match.clew.route else
               match.clew.entry_symbol).rsplit(".", 1)[0] for match in selected}
    for candidate in sorted(matches, key=rank):
        if len(selected) >= limit:
            break
        if any(item.clew.id == candidate.clew.id for item in selected):
            continue
        owner = (candidate.clew.route[0] if candidate.clew.route else
                 candidate.clew.entry_symbol).rsplit(".", 1)[0]
        if owner in owners and any(
                (item.clew.route[0] if item.clew.route else
                 item.clew.entry_symbol).rsplit(".", 1)[0] not in owners
                for item in matches if item not in selected):
            continue
        selected.append(candidate)
        owners.add(owner)
    if not selected:
        selected.append(min(matches, key=rank))
    return selected[:limit]
