"""Drive the rebuilt ingest on real indexes into a scratch DB, then gate it.

Read-only against the live store (for the comparison); every write goes to a scratch file.
The 669 MB databricks index is skipped by default — it needs several GB of RAM and this is
meant to be a safe check, not a memory experiment. Pass --include-databricks to add it.
"""
import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, '/Users/spark/git/Ariadne.orig')

from docgen.scip_graph import build_edges, build_symbols          # noqa: E402
from docgen.scip_index import ScipIndex                            # noqa: E402
from docgen.scip_store import save_rows                            # noqa: E402
from docgen.scip_graph import GraphRows                            # noqa: E402
from docgen.scip_wiring import wiring_report                       # noqa: E402
from library.scip import init_scip_schema                          # noqa: E402

LIVE = '/Users/spark/git/Ariadne.orig/ariadne.db'
BIG = 'databricks'

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/private/tmp/claude-501/-Users-spark-git-Ariadne-orig/'
                                'f60881d4-915f-4c2b-9c7e-75757e96cf0b/scratchpad/rebuilt.db')
ap.add_argument('--include-databricks', action='store_true')
args = ap.parse_args()

live = sqlite3.connect(f'file:{LIVE}?mode=ro', uri=True)
state = live.execute(
    'SELECT source_name, scip_path FROM scip_index_state ORDER BY source_name').fetchall()
languages = {name: (live.execute(
    'SELECT language, COUNT(*) c FROM scip_symbols WHERE source_name=? '
    'GROUP BY 1 ORDER BY c DESC LIMIT 1', (name,)).fetchone() or ['python'])[0]
    for name, _ in state}

out = Path(args.out)
if out.exists():
    out.unlink()
scratch = sqlite3.connect(out)
init_scip_schema(scratch)

# Flushed per source: the databricks index is 670 MB and takes minutes, so an
# unflushed run is indistinguishable from a hang when stdout is not a terminal.
print(f'{"source":14s} {"index":>10s} {"symbols":>9s} {"edges":>9s} '
      f'{"unresolved":>11s} {"unattrib":>9s} {"secs":>6s}', flush=True)
for name, scip_path in state:
    path = Path(scip_path)
    if not path.exists():
        print(f'{name:14s} {"MISSING":>10s}')
        continue
    size_mb = path.stat().st_size / 1e6
    if name == BIG and not args.include_databricks:
        print(f'{name:14s} {size_mb:9.0f}M  (skipped — pass --include-databricks)')
        continue
    print(f'{name:14s} {size_mb:9.1f}M  loading ...', flush=True)
    started = time.time()
    index = ScipIndex.load(path, repo=name, max_staleness_days=None)
    symbols = build_symbols(index, source_name=name, language=languages[name])
    edges, unresolved, unattributed = build_edges(
        index, source_name=name, language=languages[name], symbols=symbols)
    save_rows(scratch, GraphRows(symbols=symbols, edges=edges), source_name=name)
    print(f'{name:14s} {size_mb:9.1f}M {len(symbols):9d} {len(edges):9d} '
          f'{unresolved:11d} {unattributed:9d} {time.time()-started:6.1f}', flush=True)

print('\n=== the gate, on the rebuilt store ===')
report = wiring_report(scratch)
print(f'WIRING: {"OK" if report.ok else "BROKEN"}  '
      f'({len(report.failures())}/{len(report.checks)} failing)')
for check in report.checks:
    print(f'  [{"PASS" if check.ok else "FAIL"}] {check.name:28s} {check.measured}')

print('\n=== the same four checks on the LIVE store, for comparison ===')
live_report = wiring_report(live)
print(f'WIRING: {"OK" if live_report.ok else "BROKEN"}  '
      f'({len(live_report.failures())}/{len(live_report.checks)} failing)')
for check in live_report.checks:
    print(f'  [{"PASS" if check.ok else "FAIL"}] {check.name:28s} {check.measured}')

print('\n=== rebuilt store: edge types and extents ===')
for row in scratch.execute(
        'SELECT edge_type, COUNT(*) FROM scip_edges GROUP BY 1 ORDER BY 2 DESC'):
    print('   ', row)
named = scratch.execute("SELECT COUNT(*) FROM scip_symbols "
                        "WHERE canonical_id NOT LIKE 'local %'").fetchone()[0]
multi = scratch.execute("SELECT COUNT(*) FROM scip_symbols "
                        "WHERE canonical_id NOT LIKE 'local %' "
                        'AND line_end > line_start').fetchone()[0]
print(f'    named symbols with a body extent: {multi}/{named}')
scratch.close()
live.close()
print(f'\nscratch store: {out}  ({out.stat().st_size / 1e6:.1f} MB) — live store untouched')
