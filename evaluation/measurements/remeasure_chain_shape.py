"""designs/answer-path.md §2.6 -- re-measured with three corrections.

  1. SEEDS       a required CamelCase type has no outgoing `call` edge; its members do.
                 Seed = the type PLUS its declared members (parent_qualified_name, to fixpoint).
  2. SOURCE GUARD  scip_edges is global and `local N` ids are unnamespaced, so 76,823 call edges
                 (9.5%) join symbols attributed to different sources. Traversal admits an edge
                 only when BOTH endpoints are databricks symbols.
  3. SHAPE       "assembles the whole chain" = the slot graph is CONNECTED (A->B->C is a chain;
                 A->C need not be a direct edge). All-pairs is reported alongside as the strict form.

Exactness: display_name EQUALITY (not substring). File-granular target, leaning on the eval's own
defines_file, as the original did. production = path is not a test path.
"""
import sqlite3, json, os, re, collections, time

DB = "/Users/spark/git/Ariadne.orig/ariadne.db"
REQ = "/Users/spark/git/Ariadne.orig/evaluation/spool-clean-room/chain_requirements.json"
CORPUS = "/Users/spark/git/Ariadne.orig/spool-corpus"
REPO_DIRS = ("spark", "delta", "databricks-sdk-py")
SRC = "databricks"
VISIT_CAP = 250_000

con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
TEST_RE = re.compile(r"(^|/)(test|tests|it)/|Suite\.scala$|(^|/)test_[^/]+\.py$|_test\.py$")
is_test = lambda f: bool(TEST_RE.search(f))

print("loading the databricks id set for the source guard ...", flush=True)
DBX = {r[0] for r in con.execute(
    "SELECT canonical_id FROM scip_symbols WHERE source_name=? "
    "AND canonical_id NOT LIKE 'local %'", (SRC,))}
foreign_locals = con.execute(
    "SELECT COUNT(*) FROM scip_symbols WHERE source_name<>? AND canonical_id LIKE 'local %'",
    (SRC,)).fetchone()[0]
print(f"  {len(DBX):,} databricks symbols · guard may drop <= {foreign_locals} "
      f"local nodes owned by another source (last-writer-wins on the PK)")

_repo_cache: dict[str, frozenset] = {}
def repos_of(f):
    r = _repo_cache.get(f)
    if r is None:
        r = frozenset(d for d in REPO_DIRS if os.path.exists(os.path.join(CORPUS, d, f)))
        _repo_cache[f] = r
    return r

def resolve(sym):
    rows = con.execute("SELECT canonical_id, file, kind, qualified_name FROM scip_symbols "
                       "WHERE source_name=? AND display_name=?", (SRC, sym)).fetchall()
    prod = [r for r in rows if not is_test(r[1])]
    return prod or rows

def members(qns, levels=2):
    out, frontier = set(), set(qns)
    for _ in range(levels):
        if not frontier:
            break
        got, f = [], list(frontier)
        for i in range(0, len(f), 800):
            c = f[i:i + 800]
            qs = ",".join("?" * len(c))
            got += con.execute(f"SELECT canonical_id, qualified_name FROM scip_symbols "
                               f"WHERE source_name=? AND parent_qualified_name IN ({qs})",
                               [SRC] + c).fetchall()
        new = {g[0] for g in got} - out
        out |= new
        frontier = {g[1] for g in got}
    return out

_fc: dict[str, str] = {}
def files_of(ids):
    out, todo = set(), []
    for i in ids:
        (out.add(_fc[i]) if i in _fc else todo.append(i))
    for i in range(0, len(todo), 800):
        c = todo[i:i + 800]
        qs = ",".join("?" * len(c))
        for cid, f in con.execute(
                f"SELECT canonical_id, file FROM scip_symbols WHERE canonical_id IN ({qs})", c):
            _fc[cid] = f
            out.add(f)
    return out

def walk(seeds, depth, types, reverse=False):
    """Source-guarded BFS: a neighbour is admitted only if it is a databricks symbol."""
    src_col, dst_col = (("callee_canonical_id", "caller_canonical_id") if reverse
                        else ("caller_canonical_id", "callee_canonical_id"))
    tq = ",".join("?" * len(types))
    seeds = {s for s in seeds if s in DBX}
    seen, frontier, capped, dropped = set(seeds), set(seeds), False, 0
    for _ in range(depth):
        nxt = set()
        f = list(frontier)
        for i in range(0, len(f), 400):
            c = f[i:i + 400]
            qs = ",".join("?" * len(c))
            for (n,) in con.execute(
                    f"SELECT DISTINCT {dst_col} FROM scip_edges "
                    f"WHERE {src_col} IN ({qs}) AND edge_type IN ({tq})", c + list(types)):
                if n in DBX:
                    nxt.add(n)
                else:
                    dropped += 1
        frontier = nxt - seen
        seen |= frontier
        if len(seen) > VISIT_CAP:
            capped = True
            break
        if not frontier:
            break
    return seen, capped, dropped

questions = json.load(open(REQ))
slot, unresolved = {}, []
for qq in questions:
    for s in qq["required"]:
        sym = s["symbol"]
        if sym in slot:
            continue
        rows = resolve(sym)
        if not rows:
            unresolved.append(sym)
            slot[sym] = None
            continue
        ids, qns = {r[0] for r in rows}, {r[3] for r in rows}
        files = {r[1] for r in rows}
        slot[sym] = {"type_ids": ids, "seeds": ids | members(qns), "files": files,
                     "kinds": collections.Counter(r[2] for r in rows),
                     "repos": set().union(*(repos_of(f) for f in files)) if files else set()}

TYPE_KINDS = {"Class", "Trait", "Object", "Interface", "Enum"}
res = {k: v for k, v in slot.items() if v}
print("\n" + "=" * 76)
print(f"§2.6 RE-MEASURED — {time.strftime('%Y-%m-%d')}, source-guarded, member-seeded")
print("=" * 76)
print(f"25 questions · 97 slots · 75 distinct symbols · unresolved {unresolved}")
print(f"  CamelCase: {sum(1 for k in res if re.match(r'^[A-Z][A-Za-z0-9]*$', k))}/{len(res)}"
      f" · resolve to a TYPE kind: {sum(1 for v in res.values() if set(v['kinds']) & TYPE_KINDS)}/{len(res)}"
      f" · member expansion adds {sum(len(v['seeds']) - len(v['type_ids']) for v in res.values())} symbols")

def classify(depth, types, seed_key):
    cache, cap_n, drop_n = {}, 0, 0
    shapes, strict, rows = collections.Counter(), collections.Counter(), []
    for qq in questions:
        names = [s["symbol"] for s in qq["required"]
                 if slot[s["symbol"]] and slot[s["symbol"]][seed_key]]
        if len(names) < 2:
            shapes["UNMEASURABLE"] += 1
            strict["UNMEASURABLE"] += 1
            rows.append((qq["id"], "unmeasurable", ""))
            continue
        for n in names:
            if n not in cache:
                seen, cap, dr = walk(slot[n][seed_key], depth, types)
                cap_n += cap; drop_n += dr
                cache[n] = files_of(seen)
        adj, n_edges = collections.defaultdict(set), 0
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if (cache[a] & slot[b]["files"]) or (cache[b] & slot[a]["files"]):
                    adj[a].add(b); adj[b].add(a); n_edges += 1
        comp, stack = {names[0]}, [names[0]]
        while stack:
            for nb in adj[stack.pop()]:
                if nb not in comp:
                    comp.add(nb); stack.append(nb)
        n_pairs = len(names) * (len(names) - 1) // 2
        shapes["CONNECTIVE" if len(comp) == len(names) else "PARTIAL" if n_edges else "COMPARATIVE"] += 1
        strict["CONNECTIVE" if n_edges == n_pairs else "PARTIAL" if n_edges else "COMPARATIVE"] += 1
        rows.append((qq["id"], f"{len(comp)}/{len(names)} slots joined", f"{n_edges}/{n_pairs} pairs"))
    return shapes, strict, rows, cap_n, drop_n

keep = None
for types, label in [(("call",), "call only"), (("call", "type_ref"), "call+type_ref")]:
    for depth in (3, 4, 5):
        line = []
        for seed_key, sl in [("type_ids", "type-seeded"), ("seeds", "member-seeded")]:
            sh, st, rows, cap, dr = classify(depth, types, seed_key)
            tot = sum(sh.values())
            desc = "  ".join(f"{k[:4]} {sh.get(k,0):2d} ({100*sh.get(k,0)/tot:3.0f}%)"
                             for k in ("CONNECTIVE", "PARTIAL", "COMPARATIVE", "UNMEASURABLE"))
            line.append(f"    {sl:14s} {desc}   strict-allpairs CONN={st.get('CONNECTIVE',0)}"
                        + (f"  [{cap} capped]" if cap else "")
                        + (f"  guard dropped {dr:,} neighbours" if dr else ""))
            if types == ("call",) and depth == 3 and seed_key == "seeds":
                keep = rows
        print(f"\n--- {label}, depth {depth}")
        print("\n".join(line))

print("\n" + "=" * 76)
print('CLAIM 2: "of 19 multi-repo questions, 0 have a production-code node reaching')
print('          both sides within 3 hops; the only nodes reaching both are test files"')
print("=" * 76)
nmulti = sum(1 for q in questions
             if len({r for s in q["required"] for r in s.get("repos", [])}) >= 2)
for types, label in [(("call",), "call only"), (("call", "type_ref"), "call+type_ref")]:
    tally, detail = collections.Counter(), []
    for qq in questions:
        if len({r for s in qq["required"] for r in s.get("repos", [])}) < 2:
            continue
        sides = collections.defaultdict(set)
        for s in qq["required"]:
            if slot[s["symbol"]]:
                for r in s.get("repos", []):
                    sides[r] |= slot[s["symbol"]]["seeds"]
        sides = {k: v for k, v in sides.items() if v & DBX}
        if len(sides) < 2:
            tally["a side does not resolve"] += 1
            continue
        ancs = [walk(v, 3, types, reverse=True)[0] - v for v in sides.values()]
        shared = set.intersection(*ancs)
        if not shared:
            tally["no node reaches both"] += 1
            continue
        prod = {f for f in files_of(shared) if not is_test(f)}
        if prod:
            tally["PRODUCTION node reaches both"] += 1
            detail.append((qq["id"], len(prod), sorted(prod)[0]))
        else:
            tally["test-file nodes only"] += 1
    print(f"\n--- {label}, depth 3, member-seeded, source-guarded "
          f"(multi-repo questions per answer key: {nmulti})")
    for k, v in tally.most_common():
        print(f"      {v:3d}  {k}")
    for qid, n, ex in detail[:6]:
        print(f"        q{qid}: {n} production files reach both, e.g. {ex}")

print("\nper-question (call only, depth 3, member-seeded, guarded):")
for r in keep:
    print("   ", r)
con.close()
