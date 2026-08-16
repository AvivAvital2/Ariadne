#!/usr/bin/env python3
"""Compute per-question connection load from Ariadne's answer `sources`.

connection load = how much cross-repo dot-connecting Ariadne did to answer:
  - n_repos      : distinct repos among the sources the answer used (2 or 3 = cross-repo)
  - xp_themes    : how many of those sources are genuine cross-project themes
  - n_sources    : breadth (distinct docs cited)
Composite (weights emphasize the two the user named — file refs + theme connections):
  load = 3*(n_repos-1) + 2*min(xp_themes,2) + 0.4*min(n_sources,10)

Run after collected_sg*.json exist. Writes xrepo_connection.json.
"""
import json, glob, collections

tmap = json.load(open("title_repo_map.json"))
REPOS = {"databricks-sdk-py", "delta", "spark"}

rows = []
for f in sorted(glob.glob("collected_sg*.json")):
    rows.extend(json.load(open(f)))

out = []
for r in rows:
    if "answer" not in r or r.get("error"):
        out.append({**{k: r.get(k) for k in ("id", "family", "hinges_on")},
                    "n_repos": 0, "xp_themes": 0, "n_sources": 0,
                    "connection_load": 0.0, "repos": [], "error": r.get("error", "no answer")})
        continue
    sources = r.get("sources") or []
    repos = set()
    xp = 0
    for s in sources:
        info = tmap.get(s)
        if not info:
            continue
        repos |= (set(info["repos"]) & REPOS)
        if info.get("cross_project"):
            xp += 1
    n_repos = len(repos)
    n_sources = len(sources)
    load = 3 * max(0, n_repos - 1) + 2 * min(xp, 2) + 0.4 * min(n_sources, 10)
    out.append({"id": r["id"], "family": r.get("family"), "hinges_on": r.get("hinges_on"),
                "repos": sorted(repos), "n_repos": n_repos, "xp_themes": xp,
                "n_sources": n_sources, "connection_load": round(load, 2)})

out.sort(key=lambda x: x["id"])
json.dump(out, open("xrepo_connection.json", "w"), indent=2)

n = len(out)
xrepo = sum(1 for o in out if o["n_repos"] >= 2)
tri = sum(1 for o in out if o["n_repos"] >= 3)
withxp = sum(1 for o in out if o["xp_themes"] >= 1)
print(f"questions: {n}")
print(f"cross-repo sources (>=2 repos): {xrepo}  tri-repo: {tri}  with cross-project theme: {withxp}")
loads = sorted((o["connection_load"] for o in out), reverse=True)
print(f"connection_load: max={loads[0]} median={loads[n//2]} min={loads[-1]}")
fam = collections.Counter(o["family"] for o in out if o["n_repos"] >= 2)
print("cross-repo by family:", dict(sorted(fam.items())))
