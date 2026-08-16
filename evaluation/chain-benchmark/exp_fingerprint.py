"""Build attribution: runtime manifest, database fingerprints, config.

A result cannot be attributed to a build when imported code is absent
from provenance — and this checkout is dirty, so the manifest covers
every runtime Python file (tracked or not) rather than a hand-picked
list. The database gets two fingerprint levels: a fast one for
development runs and a strong streamed row digest — cached outside Git,
keyed by the fast fingerprint — that paid-canary certification requires.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

RUNTIME_GLOBS = (
    "library/**/*.py",
    "ariadne_mcp/**/*.py",
    "docgen/**/*.py",
)
INVOKED_EVALUATION = (
    "evaluation/chain-benchmark/shadow_eval.py",
    "evaluation/chain-benchmark/offline_earliest_failure.py",
    "evaluation/chain-benchmark/reach_gaps.py",
    "evaluation/chain-benchmark/exp_seal.py",
    "evaluation/chain-benchmark/exp_compare.py",
    "evaluation/chain-benchmark/exp_fingerprint.py",
    "evaluation/chain-benchmark/experiment.py",
)
CONFIGURATION_FILES = ("ariadne.yaml",)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments) -> "str | None":
    probe = subprocess.run(
        ["git", *arguments], cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return probe.stdout if probe.returncode == 0 else None


def runtime_manifest(root: Path) -> dict:
    """Every runtime file with its hash and tracked status."""
    root = Path(root)
    tracked_listing = _git(root, "ls-files", "-z")
    tracked = (set(tracked_listing.split("\0")) - {""}
               if tracked_listing is not None else set())
    candidates: set = set()
    for pattern in RUNTIME_GLOBS:
        candidates.update(
            str(path.relative_to(root))
            for path in root.glob(pattern) if path.is_file())
    for relative in (*INVOKED_EVALUATION, *CONFIGURATION_FILES):
        if (root / relative).is_file():
            candidates.add(relative)
    files = [
        {"path": relative,
         "sha256": _sha256_file(root / relative),
         "tracked": relative in tracked}
        for relative in sorted(candidates)]
    head = _git(root, "rev-parse", "HEAD")
    diff = _git(root, "diff", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "-z")
    manifest_digest = hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "files": files,
        "file_count": len(files),
        "untracked_runtime_files": [
            row["path"] for row in files if not row["tracked"]],
        "manifest_sha256": manifest_digest,
        "git_head": (head or "not-a-git-checkout").strip(),
        "tracked_diff_sha256": (
            hashlib.sha256(diff.encode()).hexdigest()
            if diff is not None else "not-a-git-checkout"),
        "status_sha256": (
            hashlib.sha256(status.encode()).hexdigest()
            if status is not None else "not-a-git-checkout"),
        "python_version": sys.version,
    }


def fast_db_fingerprint(db_path) -> dict:
    """Cheap identity: stat plus schema and per-kind counts."""
    import sqlite3

    path = Path(db_path).resolve()
    stat = path.stat()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        schema_version = connection.execute(
            "PRAGMA schema_version").fetchone()[0]
        master_rows = sorted(connection.execute(
            "SELECT type, name, COALESCE(sql, '') FROM sqlite_master"))
        master_digest = hashlib.sha256(json.dumps(
            master_rows, separators=(",", ":")).encode()).hexdigest()
        has_index_state = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'scip_index_state'"
        ).fetchone() is not None
        index_state_digest = "absent"
        if has_index_state:
            state_rows = sorted(connection.execute(
                "SELECT * FROM scip_index_state"))
            index_state_digest = hashlib.sha256(json.dumps(
                state_rows, separators=(",", ":"),
                default=str).encode()).hexdigest()
        edge_counts = dict(connection.execute(
            "SELECT edge_type, COUNT(*) FROM scip_edges GROUP BY edge_type"
        ).fetchall())
        symbol_counts = dict(connection.execute(
            "SELECT source_name, COUNT(*) FROM scip_symbols "
            "GROUP BY source_name").fetchall())
    finally:
        connection.close()
    return {
        "level": "fast",
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device_inode": f"{stat.st_dev}:{stat.st_ino}",
        "schema_version": schema_version,
        "sqlite_master_sha256": master_digest,
        "scip_index_state_sha256": index_state_digest,
        "edge_counts": edge_counts,
        "symbol_counts": symbol_counts,
    }


def _stream_rows(digest, cursor) -> None:
    while True:
        rows = cursor.fetchmany(5000)
        if not rows:
            return
        for row in rows:
            for field in row:
                data = str(field).encode()
                digest.update(f"{len(data)}:".encode())
                digest.update(data)
            digest.update(b";")


def strong_db_fingerprint(db_path, *, cache_path=None) -> dict:
    """Streamed row digest over symbols and edges in primary-key order.

    Cached outside Git keyed by the fast fingerprint, because the strong
    digest is expensive and the fast fingerprint changes whenever the
    file does.
    """
    import sqlite3

    fast = fast_db_fingerprint(db_path)
    cache_key = hashlib.sha256(json.dumps(
        fast, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache_file = Path(cache_path) if cache_path else (
        Path(db_path).resolve().parent / ".ariadne"
        / "db-strong-digest-cache.json")
    cached = {}
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text())
    if cache_key in cached:
        return {**fast, "level": "strong",
                "strong_sha256": cached[cache_key], "cache": "hit"}

    digest = hashlib.sha256()
    connection = sqlite3.connect(
        f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        _stream_rows(digest, connection.execute(
            "SELECT canonical_id, source_name, language, file, "
            "line_start, line_end, kind, display_name, "
            "qualified_name, parent_qualified_name "
            "FROM scip_symbols ORDER BY canonical_id"))
        _stream_rows(digest, connection.execute(
            "SELECT caller_canonical_id, callee_canonical_id, "
            "edge_type, file, line, confidence "
            "FROM scip_edges ORDER BY caller_canonical_id, "
            "callee_canonical_id, edge_type, file, line"))
    finally:
        connection.close()
    strong = digest.hexdigest()
    cached[cache_key] = strong
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cached, indent=1, sort_keys=True))
    return {**fast, "level": "strong",
            "strong_sha256": strong, "cache": "miss"}


def effective_configuration() -> dict:
    """The configuration production actually runs with, introspected —
    never a hand-maintained constants dict that can drift."""
    import inspect

    from library.chain_menu import definition_body_selection_requires_llm
    from library.structural_assembly import obligation_seeded_expansion

    def defaults(function, names):
        signature = inspect.signature(function)
        return {
            name: signature.parameters[name].default
            for name in names if name in signature.parameters}

    expansion = defaults(
        obligation_seeded_expansion,
        ("depth", "forward_depth", "per_seed_limit", "reserve_limit"))
    body = defaults(
        definition_body_selection_requires_llm, ("auto_select_max",))
    return {
        "expansion": expansion,
        "body_selection": body,
        "forward_traversal_enabled_by_default": bool(
            expansion.get("forward_depth")),
        "body_plan_application": "diagnostic-only",
    }
