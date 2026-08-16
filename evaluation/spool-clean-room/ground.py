#!/usr/bin/env python3
"""Grounding helper for cross-repo question generation.

Usage:
    python3 ground.py "ClassNameA" "ClassNameB" ...
    python3 ground.py --theme "MERGE INTO"        # dump a theme summary doc
    python3 ground.py --list "merge"              # list doc titles matching a substring

Prints real doc titles + content excerpts from the databricks spool pack so
generated questions cite actual this-corpus methods/fields (not hallucinated).
Read-only; no network, no API key.
"""
import sqlite3, sys, json

PACK = "/Users/spark/git/Ariadne.orig/.ariadne/spools/databricks/pack.db"
con = sqlite3.connect(f"file:{PACK}?mode=ro", uri=True)
c = con.cursor()

def repo_of(sf):
    try:
        files = json.loads(sf) if sf else []
    except Exception:
        files = []
    seg = files[0].split("/", 1)[0] if files else "?"
    return seg, (files[0] if files else "?")

if not sys.argv[1:]:
    print(__doc__); sys.exit(0)

if sys.argv[1] == "--list":
    sub = sys.argv[2]
    rows = c.execute(
        "SELECT title, content_type, source_files FROM documents "
        "WHERE title LIKE ? AND content_type IN ('explanation','gotcha','theme') "
        "ORDER BY content_token_count DESC LIMIT 60", (f"%{sub}%",)).fetchall()
    for t, ct, sf in rows:
        repo, f = repo_of(sf)
        print(f"[{repo:16}] ({ct}) {t}   <- {f}")
    sys.exit(0)

if sys.argv[1] == "--theme":
    sub = sys.argv[2]
    row = c.execute(
        "SELECT title, content FROM documents WHERE content_type='theme' AND title LIKE ? "
        "ORDER BY content_token_count DESC LIMIT 1", (f"%{sub}%",)).fetchone()
    if row:
        print("#", row[0], "\n"); print(row[1][:6000])
    else:
        print("no theme matching", sub)
    sys.exit(0)

for name in sys.argv[1:]:
    rows = c.execute(
        "SELECT title, content_type, source_files, content FROM documents "
        "WHERE (title LIKE ? OR content LIKE ?) "
        "AND content_type IN ('explanation','gotcha') "
        "ORDER BY (title LIKE ?) DESC, content_token_count DESC LIMIT 4",
        (f"%{name}%", f"%{name}%", f"%{name}%")).fetchall()
    print(f"\n===== {name} ({len(rows)} docs) =====")
    for t, ct, sf, content in rows:
        repo, f = repo_of(sf)
        print(f"\n--- [{repo}] ({ct}) {t}  <- {f}")
        print(content[:1800])
