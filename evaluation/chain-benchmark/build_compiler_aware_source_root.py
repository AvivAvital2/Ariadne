#!/usr/bin/env python3
"""Build the minimal DBR 17.3 source root for the public comparison verifier.

This is deliberately not a full Spark or Delta checkout.  It downloads only
the source files referenced by the public twelve-question comparison panel,
checks each against the committed SHA-256, and writes the directory layout
consumed by ``verify_compiler_aware_comparison.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
UPSTREAM = {
    "spark": "https://raw.githubusercontent.com/apache/spark/v4.0.0",
    "delta": "https://raw.githubusercontent.com/delta-io/delta/v4.0.0",
}


def _load_sources(record: Path) -> dict[tuple[str, str], str]:
    with record.open() as handle:
        payload = json.load(handle)
    sources: dict[tuple[str, str], str] = {}
    for question in payload.get("questions") or ():
        for claim in question.get("claims") or ():
            for source in claim["reviewed_source_provenance"]["source_files"]:
                relative = str(source["file"])
                digest = str(source["sha256"])
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError(f"invalid source hash for {relative}")
                # DBR's Delta source lives below delta/spark/, whereas ordinary
                # sql/ and core/ paths are from Apache Spark.  A new path shape
                # must be mapped explicitly rather than guessed.
                if relative.startswith("spark/"):
                    repository = "delta"
                elif relative.startswith(("sql/", "core/")):
                    repository = "spark"
                else:
                    raise ValueError(f"unmapped panel source path: {relative}")
                old = sources.setdefault((relative, digest), repository)
                if old != repository:
                    raise ValueError(f"conflicting repository mapping: {relative}")
    if len(sources) != 46:
        raise ValueError(f"expected 46 distinct panel source files, got {len(sources)}")
    return sources


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Ariadne-panel-verifier"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.URLError as error:
        raise RuntimeError(f"download failed: {url}: {error}") from error


def _write_one(destination: Path, url: str, digest: str, *, dry_run: bool) -> int:
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"refusing to overwrite hash-mismatched file: {destination}")
        return destination.stat().st_size
    if dry_run:
        print(f"would fetch {url} -> {destination}")
        return 0
    content = _fetch(url)
    actual = hashlib.sha256(content).hexdigest()
    if actual != digest:
        raise ValueError(f"upstream content hash differs for {url}: {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return len(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path,
                        help="new or hash-compatible source-root directory")
    parser.add_argument("--record", type=Path,
                        default=HERE / "compiler-aware-comparison-record.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the exact 46 downloads without writing or using the network")
    args = parser.parse_args()

    sources = _load_sources(args.record)
    total = 0
    for (relative, digest), repository in sorted(sources.items()):
        encoded = urllib.parse.quote(relative, safe="/")
        total += _write_one(
            args.dest / repository / relative,
            f"{UPSTREAM[repository]}/{encoded}", digest, dry_run=args.dry_run)
    if args.dry_run:
        print(f"would create {len(sources)} hash-verified panel source files")
    else:
        print(f"created or verified {len(sources)} source files ({total:,} bytes) at {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
