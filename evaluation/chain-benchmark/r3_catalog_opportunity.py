"""R3: the measured catalog opportunity for the no-clew anchor cohort.

For every anchor frozen in the clew_in_store cohort (no clew route
mentions it), this probe asks — BEFORE any retrieval change — whether
the exact source-instance identity is reachable through the bounded
top-28 catalog seed pool the single-vector handoff would introduce:

- is a catalog/explanation document whose ``qualified_name`` resolves
  to the anchor's exact (qualified name, file, extent) present in the
  vector-ranked top-28 for that question, and at what rank;
- does that resolution stay inside the right source and module;
- does the symbol carry a usable SCIP definition row;
- which recorded obligation's requirement it lexically supports.

Zero provider calls: cached question vectors, the real embedding
matrix, and a read-only store.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import numpy as np

    from ariadne_mcp.service import AriadneService
    from library.chain_answer import catalog_positioning_documents
    from library.clews import _lexical_tokens

    service = AriadneService.get()
    vectors = np.load(str(HERE / "question-embeddings.npz"))
    cohorts = json.loads(
        (HERE / "anchor-cohorts-baseline.json").read_text())
    gold = json.loads(
        (HERE / "gold-chain-reviewed-compact.json").read_text())
    by_id = {int(question["id"]): question
             for question in gold["questions"]}

    def identity_for(question_id, anchor, qname):
        for claim in by_id[question_id].get("claims", ()):
            selected = (claim.get("review") or {}).get(
                "selected_candidate_by_anchor", {})
            for entry in claim.get("anchors", ()):
                if str(entry.get("anchor")) != anchor:
                    continue
                for candidate in entry.get("candidates", ()):
                    if (candidate.get("canonical_id")
                            == selected.get(anchor)
                            and candidate["qualified_name"] == qname):
                        return candidate
        return None

    no_clew = [row for row in cohorts["anchors"]
               if row["original_first_loss"] == "clew_in_store"]
    print(f"no-clew cohort: {len(no_clew)} anchor rows")

    pool_cache: dict = {}
    rows = []
    for row in no_clew:
        question_id = int(row["question"])
        if question_id not in pool_cache:
            trace = json.loads(gzip.decompress(
                (HERE / "live22-diagnostic-answers-traces" /
                 f"q{question_id}.json.gz").read_bytes()))
            question = str(trace["question"])
            plan = next(
                completion["response"]
                for completion in trace["llm_completions"]
                if completion["phase"] == "scip-obligation-plan")
            obligations = re.findall(
                r"(?m)^\s*C(\d{1,2})\s*:\s*(.+)$", plan)[:5]
            documents = catalog_positioning_documents(
                service.library, question,
                sources=("databricks", "spool:databricks"), limit=28,
                query_embedding=np.asarray(
                    vectors[f"q{question_id}"], dtype=np.float32),
                matrix_provider=service._get_embedding_matrix)
            resolved = []
            with service.library._conn_provider.acquire() as conn:
                for rank, document in enumerate(documents or ()):
                    metadata = getattr(document, "metadata", None)
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except ValueError:
                            metadata = None
                    name = (metadata or {}).get("qualified_name") if (
                        isinstance(metadata, dict)) else None
                    if not name:
                        resolved.append((rank, None, ()))
                        continue
                    hits = conn.execute(
                        "SELECT qualified_name, file, line_start, "
                        "line_end, kind FROM scip_symbols WHERE "
                        "source_name = 'databricks' AND "
                        "qualified_name = ? AND canonical_id "
                        "NOT GLOB 'local *'", (str(name),)).fetchall()
                    resolved.append((rank, str(name), hits))
            pool_cache[question_id] = (question, obligations, resolved)
        question, obligations, resolved = pool_cache[question_id]

        identity = identity_for(
            question_id, row["anchor"], row["qualified_name"])
        entry = {
            "question": question_id, "anchor": row["anchor"],
            "qualified_name": row["qualified_name"],
            "in_top28": False, "catalog_rank": None,
            "exact_identity": False, "right_source_module": False,
            "scip_definition": False, "supports_obligation": None}
        if identity is None:
            entry["note"] = "gold identity unresolved"
            rows.append(entry)
            continue
        for rank, name, hits in resolved:
            if name != identity["qualified_name"]:
                continue
            entry["in_top28"] = True
            entry["catalog_rank"] = rank
            for qname, file, line_start, line_end, kind in hits:
                entry["scip_definition"] = True
                if (file == identity["file"]
                        and int(line_start) == int(
                            identity["line_start"])
                        and int(line_end) == int(identity["line_end"])):
                    entry["exact_identity"] = True
                    entry["right_source_module"] = True
                    entry["kind"] = kind
            break
        anchor_tokens = set(_lexical_tokens(
            identity["qualified_name"]))
        best = None
        for number, text in obligations:
            overlap = len(anchor_tokens.intersection(
                _lexical_tokens(text)))
            if best is None or overlap > best[1]:
                best = (f"O{int(number)}", overlap)
        if best and best[1] > 0:
            entry["supports_obligation"] = best[0]
        rows.append(entry)

    r3 = [row for row in rows
          if row["in_top28"] and row["exact_identity"]]
    summary = {
        "cohort_rows": len(rows),
        "in_top28": sum(1 for row in rows if row["in_top28"]),
        "exact_identity_in_top28": len(r3),
        "distinct_r3_identities": len({
            (row["question"], row["qualified_name"]) for row in r3}),
    }
    payload = {"schema": "r3-catalog-opportunity-v1",
               "summary": summary, "rows": rows}
    (HERE / "r3-catalog-opportunity.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True))
    print("summary:", json.dumps(summary))
    for row in rows:
        print(f"  q{row['question']} {row['anchor']}: top28="
              f"{row['in_top28']} rank={row['catalog_rank']} "
              f"exact={row['exact_identity']} "
              f"scip={row['scip_definition']} "
              f"obligation={row['supports_obligation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
