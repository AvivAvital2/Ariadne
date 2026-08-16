#!/usr/bin/env python3
"""Read-only experiment for a SCIP-first claim ledger.

The build phase reads every exact, same-source ``call`` edge from the SCIP
index and writes a temporary SQLite FTS index.  It deliberately accepts no
question or reviewed-gold input.  Querying is a separate phase: a question is
reduced to identifiers/terms, then the FTS index returns direct, compiler-
verified claim units.  Optional gold assessment happens *after* that result
has been produced, so it cannot influence the ledger or ranking.

This is an evaluation probe, not an Ariadne runtime path and it never writes
to ariadne.db.  The ledger is intentionally limited to direct calls: it tests
whether a compact precomputed proof primitive can surface the two sides of a
comparison before we add behavioural source claims or any dynamic graph walk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable


_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for",
    "from", "how", "if", "in", "into", "is", "it", "of", "on", "or",
    "the", "this", "to", "was", "what", "when", "why", "with",
})
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+")


def _terms(text: str) -> tuple[str, ...]:
    """The same identifier splitting class used by lexical clew recall."""
    output: list[str] = []
    # Keep versioned/package identifiers such as ``V2`` intact.  The normal
    # camel-case splitter deliberately drops one-character fragments, which
    # would turn that discriminating role token into nothing.
    output.extend(
        value.lower() for value in re.findall(r"[A-Za-z]+[0-9]+", text or "")
        if len(value) > 1)
    for part in _PARTS.findall(text or ""):
        token = part.lower()
        if token in _STOP or len(token) < 2:
            continue
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        else:
            for suffix in ("ing", "ed", "ion", "es", "s"):
                if len(token) > len(suffix) + 3 and token.endswith(suffix):
                    token = token[:-len(suffix)]
                    break
        output.append(token)
    return tuple(output)


def _surface(*parts: str) -> str:
    return " ".join(_terms(" ".join(part for part in parts if part)))


def _is_production_path(path: str) -> bool:
    """Keep source-level proof units; test/generated code is not user behaviour."""
    normalized = "/" + str(path or "").replace("\\", "/").lower().strip("/") + "/"
    return not any(marker in normalized for marker in (
        "/test/", "/tests/", "/benchmark/", "/benchmarks/", "/generated/", "/target/",
    ))


def _source_fingerprint(conn: sqlite3.Connection, source: str) -> str:
    digest = hashlib.sha256()
    for row in conn.execute(
        "SELECT canonical_id, file, line_start, line_end FROM scip_symbols "
        "WHERE source_name=? ORDER BY canonical_id", (source,),
    ):
        digest.update("\x1f".join(map(str, row)).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _source_lines(root: Path, file: str) -> tuple[list[str], str] | None:
    """Read a single indexed source file without accepting a path escape."""
    raw = Path(file)
    candidates = [root / raw]
    if not raw.is_absolute():
        candidates.extend(child / raw for child in root.iterdir() if child.is_dir())
    resolved: list[Path] = []
    for candidate in candidates:
        try:
            absolute = candidate.resolve()
            absolute.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        if absolute.is_file() and absolute not in resolved:
            resolved.append(absolute)
    if len(resolved) != 1:
        return None
    try:
        data = resolved[0].read_bytes()
        return data.decode("utf-8").splitlines(), hashlib.sha256(data).hexdigest()
    except (OSError, UnicodeError):
        return None


def _build(source_db: Path, ledger: Path, source: str) -> dict[str, Any]:
    if ledger.exists():
        raise FileExistsError(f"refusing to overwrite existing ledger: {ledger}")
    started = time.monotonic()
    source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    target = sqlite3.connect(ledger)
    try:
        target.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            CREATE TABLE claim_units (
                id TEXT PRIMARY KEY,
                relation TEXT NOT NULL,
                caller_id TEXT NOT NULL,
                callee_id TEXT NOT NULL,
                caller_name TEXT NOT NULL,
                callee_name TEXT NOT NULL,
                caller_file TEXT NOT NULL,
                caller_line_start INTEGER NOT NULL,
                caller_line_end INTEGER NOT NULL,
                callee_file TEXT NOT NULL,
                callee_line_start INTEGER NOT NULL,
                callee_line_end INTEGER NOT NULL,
                site_file TEXT NOT NULL,
                site_line INTEGER NOT NULL,
                surface TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE claim_search USING fts5(
                surface, content='claim_units', content_rowid='rowid',
                tokenize='porter unicode61'
            );
        """)
        query = """
            SELECT e.caller_canonical_id, e.callee_canonical_id, e.file, e.line,
                   caller.qualified_name, caller.file, caller.line_start, caller.line_end,
                   callee.qualified_name, callee.file, callee.line_start, callee.line_end
            FROM scip_edges AS e
            JOIN scip_symbols AS caller ON caller.canonical_id=e.caller_canonical_id
            JOIN scip_symbols AS callee ON callee.canonical_id=e.callee_canonical_id
            WHERE e.edge_type='call' AND caller.source_name=? AND callee.source_name=?
            ORDER BY e.caller_canonical_id, e.callee_canonical_id, e.file, e.line
        """
        batch: list[tuple[Any, ...]] = []
        count = 0
        for row in source_conn.execute(query, (source, source)):
            (caller_id, callee_id, site_file, site_line, caller_name, caller_file,
             caller_start, caller_end, callee_name, callee_file, callee_start,
             callee_end) = row
            if not all(_is_production_path(value) for value in (
                    caller_file, callee_file, site_file)):
                continue
            unit_id = hashlib.sha256(
                "\x1f".join(map(str, (caller_id, callee_id, "call", site_file, site_line))).encode()
            ).hexdigest()
            batch.append((
                unit_id, "call", caller_id, callee_id, caller_name, callee_name,
                caller_file, caller_start, caller_end, callee_file, callee_start,
                callee_end, site_file, site_line,
                _surface(caller_name, callee_name, caller_file, callee_file, "call"),
            ))
            if len(batch) == 5_000:
                target.executemany(
                    "INSERT INTO claim_units VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                count += len(batch)
                batch.clear()
        if batch:
            target.executemany(
                "INSERT INTO claim_units VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            count += len(batch)
        target.execute("INSERT INTO claim_search(claim_search) VALUES('rebuild')")
        target.executescript("""
            CREATE INDEX claim_units_caller ON claim_units(caller_id);
            CREATE INDEX claim_units_callee ON claim_units(callee_id);
            CREATE INDEX claim_units_site ON claim_units(site_file, site_line);
        """)
        target.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        meta = {
            "schema": "scip-direct-call-claim-ledger/v1",
            "source_db": str(source_db.resolve()),
            "source": source,
            "source_fingerprint": _source_fingerprint(source_conn, source),
            "unit_count": str(count),
        }
        target.executemany("INSERT INTO meta VALUES (?,?)", sorted(meta.items()))
        target.commit()
        return {**meta, "elapsed_seconds": round(time.monotonic() - started, 3),
                "ledger_bytes": ledger.stat().st_size}
    finally:
        source_conn.close()
        target.close()


def _row_item(row: sqlite3.Row | tuple[Any, ...], position: int, *,
              origin: str, source_rank: int | None = None) -> dict[str, Any]:
    item = dict(row)
    haystack = " ".join(str(item[key]).lower() for key in (
        "caller_name", "callee_name", "caller_file", "callee_file"))
    item.update({
        "position": position,
        "origin": origin,
        "source_fact_rank": source_rank,
    })
    return item


def _match_query(conn: sqlite3.Connection, question: str, top_k: int) -> list[dict[str, Any]]:
    tokens = _terms(question)
    fts = " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
    rows = conn.execute("""
        SELECT c.id, c.relation, c.caller_id, c.callee_id, c.caller_name,
               c.callee_name, c.caller_file, c.caller_line_start, c.caller_line_end,
               c.callee_file, c.callee_line_start, c.callee_line_end,
               c.site_file, c.site_line, bm25(claim_search) AS rank
        FROM claim_search
        JOIN claim_units AS c ON c.rowid=claim_search.rowid
        WHERE claim_search MATCH ?
        ORDER BY rank, c.id
        LIMIT ?
    """, (fts, top_k)).fetchall()
    output = []
    for position, row in enumerate(rows, start=1):
        output.append(_row_item(row, position, origin="edge-identity"))
    return output


def _source_fact_scan(source_db: Path, claim_conn: sqlite3.Connection,
                      source_root: Path, source: str, question: str,
                      fact_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Controlled no-write ablation for source-derived symbol vocabulary.

    It has the same unit boundary intended for ``symbol_facts``—one caller
    definition and its source range—but keeps the token surface in memory for
    this single experiment.  The only graph operation afterwards is an exact
    lookup of precomputed direct claims by canonical caller id.
    """
    if fact_k <= 0:
        return [], {"scanned_symbols": 0, "matches": []}
    query_terms = set(_terms(question))
    source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        rows = source_conn.execute("""
            SELECT s.canonical_id, s.qualified_name, s.file, s.line_start, s.line_end
            FROM scip_symbols AS s
            WHERE s.source_name=? AND s.line_start >= 1 AND s.line_end >= s.line_start
              AND EXISTS (
                SELECT 1 FROM scip_edges AS e
                JOIN scip_symbols AS callee ON callee.canonical_id=e.callee_canonical_id
                WHERE e.caller_canonical_id=s.canonical_id AND e.edge_type='call'
                  AND callee.source_name=?
              )
            ORDER BY s.file, s.line_start, s.line_end, s.canonical_id
        """, (source, source))
        last_file = None
        loaded: tuple[list[str], str] | None = None
        matches: list[tuple[int, str, str, str, int, int, str]] = []
        scanned = 0
        for symbol_id, qualified_name, file, line_start, line_end in rows:
            if not _is_production_path(str(file)):
                continue
            if file != last_file:
                loaded = _source_lines(source_root.resolve(), str(file))
                last_file = file
            start, end = int(line_start), int(line_end)
            if loaded is None or end - start >= 512 or end > len(loaded[0]):
                continue
            scanned += 1
            tokens = set(_terms(str(qualified_name)))
            tokens.update(_terms("\n".join(loaded[0][start - 1:end])))
            score = len(query_terms & tokens)
            if score:
                matches.append((score, str(qualified_name), str(symbol_id), str(file),
                                start, end, loaded[1]))
    finally:
        source_conn.close()
    matches.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = matches[:fact_k]
    selected_ids = [item[2] for item in selected]
    fact_rows = [
        {
            "position": position,
            "score": item[0],
            "symbol_id": item[2],
            "qualified_name": item[1],
            "file": item[3],
            "line_start": item[4],
            "line_end": item[5],
            "source_sha256": item[6],
        }
        for position, item in enumerate(selected, start=1)
    ]
    if not selected_ids:
        return [], {"scanned_symbols": scanned, "matches": fact_rows}
    marks = ",".join("?" * len(selected_ids))
    outgoing = claim_conn.execute(f"""
        SELECT id, relation, caller_id, callee_id, caller_name, callee_name,
               caller_file, caller_line_start, caller_line_end, callee_file,
               callee_line_start, callee_line_end, site_file, site_line, 0.0 AS rank
        FROM claim_units WHERE caller_id IN ({marks})
        ORDER BY caller_id, site_file, site_line, id
    """, selected_ids).fetchall()
    positions = {row["symbol_id"]: row["position"] for row in fact_rows}
    claims = [
        _row_item(row, position, origin="caller-source-fact",
                  source_rank=positions[row["caller_id"]])
        for position, row in enumerate(outgoing, start=1)
    ]
    return claims, {"scanned_symbols": scanned, "matches": fact_rows}


def _build_symbol_facts(source_db: Path, ledger: Path, source_root: Path,
                        source: str) -> dict[str, Any]:
    """Persist source-derived callable surfaces once beside the call ledger.

    This is the ingestion-time form of ``_source_fact_scan``.  It writes only
    to the caller-supplied external ledger, never to ``ariadne.db``.  Every
    fact remains tied to one exact SCIP caller symbol and a source hash, so a
    later question can use FTS rather than rescan every source body.
    """
    if not ledger.is_file():
        raise FileNotFoundError(f"claim ledger does not exist: {ledger}")
    target = sqlite3.connect(ledger)
    source_conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    try:
        exists = target.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='symbol_facts'"
        ).fetchone()
        if exists:
            count = target.execute("SELECT count(*) FROM symbol_facts").fetchone()[0]
            if count:
                return {"reused": True, "fact_count": int(count)}
            target.executescript("DROP TABLE symbol_facts; DROP TABLE symbol_fact_search;")
        target.executescript("""
            CREATE TABLE symbol_facts (
                symbol_id TEXT PRIMARY KEY,
                qualified_name TEXT NOT NULL,
                file TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                source_sha256 TEXT NOT NULL,
                surface TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE symbol_fact_search USING fts5(
                surface, content='symbol_facts', content_rowid='rowid',
                tokenize='porter unicode61'
            );
        """)
        query = """
            SELECT s.canonical_id, s.qualified_name, s.file, s.line_start, s.line_end
            FROM scip_symbols AS s
            WHERE s.source_name=? AND s.line_start >= 1 AND s.line_end BETWEEN s.line_start AND s.line_start + 511
              AND EXISTS (
                SELECT 1 FROM scip_edges AS e
                JOIN scip_symbols AS callee ON callee.canonical_id=e.callee_canonical_id
                WHERE e.caller_canonical_id=s.canonical_id AND e.edge_type='call'
                  AND callee.source_name=?
              )
            ORDER BY s.file, s.line_start, s.line_end, s.canonical_id
        """
        last_file = None
        loaded: tuple[list[str], str] | None = None
        batch: list[tuple[str, str, str, int, int, str, str]] = []
        fact_count = 0
        for symbol_id, qualified_name, file, line_start, line_end in source_conn.execute(query, (source, source)):
            if not _is_production_path(str(file)):
                continue
            if file != last_file:
                loaded = _source_lines(source_root.resolve(), str(file))
                last_file = file
            start, end = int(line_start), int(line_end)
            if loaded is None or end > len(loaded[0]):
                continue
            # Store a token surface, not source text: source is re-read and
            # hash-verified only when a selected fact is materialized.
            surface = _surface(str(qualified_name), "\n".join(loaded[0][start - 1:end]))
            if not surface:
                continue
            batch.append((str(symbol_id), str(qualified_name), str(file), start, end,
                          loaded[1], surface))
            if len(batch) == 2_000:
                target.executemany("INSERT INTO symbol_facts VALUES (?,?,?,?,?,?,?)", batch)
                fact_count += len(batch)
                batch.clear()
        if batch:
            target.executemany("INSERT INTO symbol_facts VALUES (?,?,?,?,?,?,?)", batch)
            fact_count += len(batch)
        target.execute("INSERT INTO symbol_fact_search(symbol_fact_search) VALUES('rebuild')")
        target.execute("CREATE INDEX symbol_facts_file ON symbol_facts(file, line_start)")
        target.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        target.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", (
            ("symbol_fact_schema", "scip-symbol-facts/v1"),
            ("symbol_fact_source", source),
            ("symbol_fact_source_fingerprint", _source_fingerprint(source_conn, source)),
        ))
        target.commit()
        return {"reused": False, "fact_count": fact_count,
                "ledger_bytes": ledger.stat().st_size}
    except BaseException:
        target.rollback()
        raise
    finally:
        source_conn.close()
        target.close()


def _source_fact_lookup(claim_conn: sqlite3.Connection, question: str,
                        fact_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Query persisted exact-symbol source surfaces without a source rescan."""
    if fact_k <= 0:
        return [], {"precomputed": True, "matches": []}
    exists = claim_conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='symbol_facts'"
    ).fetchone()
    if not exists:
        raise RuntimeError("symbol facts are absent from the external claim ledger")
    tokens = _terms(question)
    fts = " OR ".join(f'"{token}"' for token in dict.fromkeys(tokens))
    rows = claim_conn.execute("""
        SELECT f.symbol_id, f.qualified_name, f.file, f.line_start, f.line_end,
               f.source_sha256, bm25(symbol_fact_search) AS rank
        FROM symbol_fact_search
        JOIN symbol_facts AS f ON f.rowid=symbol_fact_search.rowid
        WHERE symbol_fact_search MATCH ?
        ORDER BY rank, f.symbol_id
        LIMIT ?
    """, (fts, fact_k)).fetchall()
    fact_rows = [
        {
            "position": position, "score": -float(row[6]),
            "symbol_id": str(row[0]), "qualified_name": str(row[1]),
            "file": str(row[2]), "line_start": int(row[3]), "line_end": int(row[4]),
            "source_sha256": str(row[5]),
        }
        for position, row in enumerate(rows, start=1)
    ]
    if not fact_rows:
        return [], {"precomputed": True, "matches": []}
    selected_ids = [row["symbol_id"] for row in fact_rows]
    marks = ",".join("?" * len(selected_ids))
    outgoing = claim_conn.execute(f"""
        SELECT id, relation, caller_id, callee_id, caller_name, callee_name,
               caller_file, caller_line_start, caller_line_end, callee_file,
               callee_line_start, callee_line_end, site_file, site_line, 0.0 AS rank
        FROM claim_units WHERE caller_id IN ({marks})
        ORDER BY caller_id, site_file, site_line, id
    """, selected_ids).fetchall()
    positions = {row["symbol_id"]: row["position"] for row in fact_rows}
    claims = [
        _row_item(row, position, origin="caller-source-fact",
                  source_rank=positions[str(row["caller_id"])])
        for position, row in enumerate(outgoing, start=1)
    ]
    return claims, {"precomputed": True, "matches": fact_rows}


def _review_edges(path: Path, question_id: int) -> set[tuple[str, str, str, str, int]]:
    """Evaluation-only: extract direct reviewed transitions after retrieval."""
    payload = json.loads(path.read_text())
    questions: Iterable[dict[str, Any]] = payload.get("questions", payload)
    target = next(item for item in questions if int(item["id"]) == question_id)
    output = set()
    for claim in target["claims"]:
        selected = set(claim.get("review", {}).get("selected_path_ids", ()))
        for candidate in claim.get("candidate_paths", ()):
            if selected and candidate.get("id") not in selected:
                continue
            for edge in candidate.get("edges", ()):
                output.add((edge["caller_canonical_id"], edge["callee_canonical_id"],
                            edge["edge_type"], edge["file"], int(edge["line"])))
    return output


def _assess(candidates: list[dict[str, Any]], gold: Path, question_id: int) -> dict[str, Any]:
    expected = _review_edges(gold, question_id)
    found: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (candidate["caller_id"], candidate["callee_id"], candidate["relation"],
               candidate["site_file"], candidate["site_line"])
        if key in expected:
            found.append({
                "position": candidate["position"],
                "caller": candidate["caller_name"],
                "callee": candidate["callee_name"],
                "site": f'{candidate["site_file"]}:{candidate["site_line"]}',
            })
    return {
        "reviewed_direct_transitions": len(expected),
        "retrieved_direct_transitions": len(found),
        "all_reviewed_direct_transitions_retrieved": len(found) == len(expected),
        "matches": found,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("ariadne.db"))
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--source-root", type=Path, default=Path("spool-corpus"))
    parser.add_argument("--fact-k", type=int, default=32)
    parser.add_argument("--source-fact-scan", action="store_true",
                        help="no-write source-token ablation over exact callable symbols")
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--question-id", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if bool(args.gold) != bool(args.question_id is not None):
        parser.error("--gold and --question-id must be supplied together")

    if args.ledger.exists():
        built: dict[str, Any] = {"reused": True}
    else:
        built = _build(args.db, args.ledger, args.source)
    conn = sqlite3.connect(f"file:{args.ledger}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        candidates = _match_query(conn, args.question, args.top_k)
        source_scan: dict[str, Any] | None = None
        if args.source_fact_scan:
            source_claims, source_scan = _source_fact_scan(
                args.db, conn, args.source_root, args.source, args.question, args.fact_k)
            seen = {item["id"] for item in candidates}
            for item in source_claims:
                if item["id"] not in seen:
                    seen.add(item["id"])
                    item["position"] = len(candidates) + 1
                    candidates.append(item)
    finally:
        conn.close()
    result: dict[str, Any] = {
        "schema": "scip-direct-call-claim-ledger-probe/v1",
        "build": built,
        "question": args.question,
        "question_terms": _terms(args.question),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "source_fact_scan": source_scan,
    }
    if args.gold:
        result["evaluation_only"] = _assess(candidates, args.gold, args.question_id)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.out}")
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "ledger": str(args.ledger), "output": str(args.out), "candidate_count": len(candidates),
        "evaluation_only": result.get("evaluation_only"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
