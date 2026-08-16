"""Item-lifecycle audit: per reviewed item, one record tracing DB presence
through retrieval, menus, selection, retention, materialization, ledger,
and the final answer — with fail-closed nulls where v1 traces cannot prove
a stage, and byte-stable output.

All fixtures are synthetic (pkg.Owner.member style).
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


FLAG_NAMES = (
    "db_present", "retrieval_pool_present",
    "symbol_menu_present", "symbol_selected",
    "component_menu_present", "component_selected",
    "route_menu_present", "route_selected",
    "body_menu_present", "body_selected",
    "hydrated", "projected",
    "story_present", "ledger_present", "answer_present",
)


def _module():
    path = (Path(__file__).parents[1] / "evaluation/chain-benchmark" /
            "audit_item_lifecycle.py")
    spec = importlib.util.spec_from_file_location(
        "audit_item_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
def _make_db(path: Path, symbols: list[tuple], edges: list[tuple] = ()):
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE scip_symbols (canonical_id TEXT, source_name TEXT,"
        " language TEXT, file TEXT, line_start INTEGER, line_end INTEGER,"
        " kind TEXT, display_name TEXT, qualified_name TEXT,"
        " parent_qualified_name TEXT)")
    connection.execute(
        "CREATE TABLE scip_edges (caller_canonical_id TEXT,"
        " callee_canonical_id TEXT, edge_type TEXT, file TEXT,"
        " line INTEGER, confidence REAL)")
    for row in symbols:
        canonical_id, file, line_start, line_end, qualified_name = row[:5]
        source_name = row[5] if len(row) > 5 else "src1"
        connection.execute(
            "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
            (canonical_id, source_name, "python", file, line_start, line_end,
             "", qualified_name.rsplit(".", 1)[-1], qualified_name, None))
    for row in edges:
        caller_id, callee_id, file, line = row[:4]
        edge_type = row[4] if len(row) > 4 else "call"
        connection.execute(
            "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
            (caller_id, callee_id, edge_type, file, line, 1.0))
    connection.commit()
    connection.close()
    return path


def _default_db(tmp_path: Path) -> Path:
    return _make_db(
        tmp_path / "lifecycle.db",
        symbols=[
            ("cid-widget-run", "module_a/widget.py", 10, 20,
             "pkg_a.Widget.run"),
            ("cid-helper-emit", "module_a/helper.py", 5, 10,
             "pkg_a.Helper.emit"),
            ("cid-widget-render", "module_b/widget.py", 30, 50,
             "pkg_b.Widget.render"),
        ],
        edges=[
            ("cid-widget-run", "cid-helper-emit", "module_a/widget.py", 12),
        ])


_SYMBOL_MENU = "\n".join([
    "Map every obligation to compiler-derived endpoint symbols.",
    "C1: Prove the widget chain.",
    "SCIP ENDPOINT SYMBOLS; map every obligation to its bridge endpoints.",
    "  S1. pkg_a.Widget.run",
    "  S2. pkg_b.Widget.render",
])
_COMPONENT_MENU = "\n".join([
    "You select evidence. Reply with numbers only.",
    "SCIP CONNECTED COMPONENTS; choose every component needed.",
    "  G1. entry Widget.run; terminal Helper.emit",
    "  G2. entry Widget.render; terminal Widget.render",
])
_ROUTE_MENU = "\n".join([
    "You select exact compiler-derived routes. Reply with IDs only.",
    "SCIP ROUTES   expanded from selected graph components.",
    "  R1. pkg_a.Widget.run -> pkg_a.Helper.emit",
    "  R2. pkg_b.Widget.render",
])
_BODY_MENU = "\n".join([
    "DEFINITION BODY CARDS  \x14 choose only required bodies.",
    "  B1. Widget.run  \x14 route root; 11-line definition",
    "  B2. Widget.render  \x14 route root; 21-line definition",
    "  B3. Helper.emit  \x14 calls from Widget.run; 6-line definition"
    " [route transition]",
])

_MATERIALIZED = (
    "/corpus/module_a/widget.py:10\n```\n"
    + "\n".join([
        "def run():",
        "    emit(value)",
        "    helper.emit()",
        *(["    pass"] * 6),
        "    dropped_from_ledger",
        "    return value",
    ])
    + "\n```\n")

_FORMULATION_PROMPT = """EVIDENCE IR
Nodes:
  {{N1}}: pkg_a.Widget.run [module_a/widget.py:10]; called at module_a/widget.py:3
Transitions:
  {{E1}}: {{N1}} calls {{N1}} at module_a/widget.py:12
SOURCE CHUNKS
  {{X1}}: module_a/widget.py:10-12 [causal]
    10 | def run():
    11 |     emit(value)
    12 |     helper.emit()
"""


def _pool() -> dict:
    return {
        "R1": [
            ["pkg_a.Widget.run", "module_a/widget.py", 10, 20, "",
             "module_a/widget.py", 10, "localized", 0, "question_symbol"],
            ["pkg_a.Helper.emit", "module_a/helper.py", 5, 10,
             "pkg_a.Widget.run", "module_a/widget.py", 12, "calls", 1,
             "leaf"],
        ],
        "R2": [
            ["pkg_b.Widget.render", "module_b/widget.py", 30, 50, "",
             "module_b/widget.py", 30, "localized", 0, "question_symbol"],
        ],
    }


def _trace(*, symbol_reply: str = "C1: S1", component_reply: str = "C1: G1",
           route_reply: str = "C1: R1", body_menu: str | None = _BODY_MENU,
           body_reply: str = "B1", materialized: str = _MATERIALIZED,
           formulation_prompt: str = _FORMULATION_PROMPT,
           symbol_menu: str = _SYMBOL_MENU,
           component_menu: str = _COMPONENT_MENU,
           route_menu: str = _ROUTE_MENU) -> dict:
    completions = [
        {"phase": "scip-symbol-select",
         "messages": [{"role": "user", "content": symbol_menu}],
         "response": symbol_reply},
        {"phase": "scip-component-select",
         "messages": [{"role": "user", "content": component_menu}],
         "response": component_reply},
        {"phase": "scip-exact-route-select",
         "messages": [{"role": "user", "content": route_menu}],
         "response": route_reply},
    ]
    if body_menu is not None:
        completions.append(
            {"phase": "scip-body-select",
             "messages": [{"role": "user", "content": body_menu}],
             "response": body_reply})
    completions.append(
        {"phase": "completion",
         "messages": [{"role": "system", "content": "grounded"},
                      {"role": "user", "content": formulation_prompt}],
         "response": "{{N1}} {{E1}} {{X1}}"})
    return {
        "schema": "ariadne-live-diagnostic-v1",
        "id": 7,
        "question": "how does the widget emit?",
        "source": "src1",
        "corpus": "/corpus",
        "service_answer": "final",
        "benchmark_answer": "final with evidence",
        "response_diagnostics": {},
        "materialized_evidence": {
            "text": materialized,
            "files_read": ["/corpus/module_a/widget.py"],
            "file_hashes": {"/corpus/module_a/widget.py": "aa11"},
        },
        "llm_completions": completions,
        "usage_rows": [],
    }


def _answer(*, pool: dict | None = None,
            selected_routes: list[str] | None = None,
            completed: list[str] | None = None,
            hydrated: list[str] | None = None) -> dict:
    pool = pool if pool is not None else _pool()
    completed = completed if completed is not None else [
        "pkg_a.Widget.run", "pkg_a.Helper.emit"]
    return {
        "id": 7,
        "route_candidates": {
            label: [occurrence[0] for occurrence in occurrences]
            for label, occurrences in pool.items()},
        "route_candidate_occurrences": pool,
        "selected_route_ids": (
            selected_routes if selected_routes is not None else ["R1"]),
        "selected_body_symbols": list(completed),
        "selected_symbols": [],
        "hydrated_symbols": list(
            completed if hydrated is None else hydrated),
        "files_read": ["/corpus/module_a/widget.py"],
        "total_cost_usd": 0.25,
    }


def _claim(identifier: str, *, symbol: str, file: str, line_start: int,
           line_end: int, fragment: str | None = None,
           witness_file: str | None = None, witness_line: int | None = None,
           path_edge: dict | None = None) -> dict:
    witnesses = []
    if fragment is not None:
        witnesses.append({
            "id": f"{identifier}-w",
            "file": witness_file or file,
            "line_start": witness_line or line_start,
            "line_end": witness_line or line_start,
            "contains": [fragment],
        })
    claim = {
        "id": identifier,
        "assertion": identifier,
        "review": {
            "status": "accepted",
            "selected_candidate_by_anchor": {"entry": symbol},
            "selected_path_ids": [],
        },
        "anchors": [{
            "anchor": "entry",
            "candidates": [{
                "qualified_name": symbol,
                "canonical_id": symbol,
                "file": file,
                "line_start": line_start,
                "line_end": line_end,
            }],
        }],
        "candidate_paths": [],
        "witnesses": witnesses,
    }
    if path_edge is not None:
        claim["candidate_paths"] = [
            {"id": "p1", "nodes": [], "edges": [path_edge]}]
        claim["review"]["selected_path_ids"] = ["p1"]
    return claim


def _final(identifier: str, *, passed: bool,
           missing_symbols: list[str] | None = None,
           missing_definitions: list[str] | None = None,
           missing_edges: list[str] | None = None,
           missing_witness_fragments: list[str] | None = None) -> dict:
    return {
        "id": identifier,
        "passed": passed,
        "missing_symbols": missing_symbols or [],
        "missing_definitions": missing_definitions or [],
        "missing_edges": missing_edges or [],
        "missing_witness_fragments": missing_witness_fragments or [],
    }


def _default_question() -> dict:
    return {"id": 7, "claims": [
        _claim("entry-claim", symbol="pkg_a.Widget.run",
               file="module_a/widget.py", line_start=10, line_end=20,
               fragment="emit(value)", witness_line=11),
        _claim("helper-claim", symbol="pkg_a.Helper.emit",
               file="module_a/helper.py", line_start=5, line_end=10,
               fragment="unmaterialized_marker", witness_line=6),
        _claim("ghost-claim", symbol="pkg_c.Ghost.walk",
               file="module_c/ghost.py", line_start=3, line_end=9),
        _claim("ledger-claim", symbol="pkg_a.Widget.run",
               file="module_a/widget.py", line_start=10, line_end=20,
               fragment="dropped_from_ledger", witness_line=19),
        _claim("edge-claim", symbol="pkg_a.Widget.run",
               file="module_a/widget.py", line_start=10, line_end=20,
               path_edge={"file": "module_a/widget.py", "line": 12,
                          "caller": "pkg_a.Widget.run",
                          "callee": "pkg_a.Helper.emit"}),
        _claim("render-claim", symbol="pkg_b.Widget.render",
               file="module_b/widget.py", line_start=30, line_end=50),
    ]}


def _default_final_question() -> dict:
    return {"id": 7, "passed": False, "claims": [
        _final("entry-claim", passed=True),
        _final("helper-claim", passed=False,
               missing_definitions=[
                   "pkg_a.Helper.emit@module_a/helper.py:5-10"],
               missing_witness_fragments=[
                   "helper-claim-w:unmaterialized_marker"]),
        _final("ghost-claim", passed=False,
               missing_symbols=["pkg_c.Ghost.walk"],
               missing_definitions=[
                   "pkg_c.Ghost.walk@module_c/ghost.py:3-9"]),
        _final("ledger-claim", passed=False,
               missing_witness_fragments=[
                   "ledger-claim-w:dropped_from_ledger"]),
        _final("edge-claim", passed=True),
        _final("render-claim", passed=False,
               missing_symbols=["pkg_b.Widget.render"],
               missing_definitions=[
                   "pkg_b.Widget.render@module_b/widget.py:30-50"]),
    ]}


def _audit_default(module, tmp_path: Path) -> dict:
    db = module.LifecycleDb(_default_db(tmp_path))
    return module.audit_question_lifecycle(
        _answer(), _default_question(), _trace(),
        _default_final_question(), db)


def _items_by_key(question_report: dict) -> dict:
    return {
        (claim["id"], item["key"]): item
        for claim in question_report["claims"]
        for item in claim["items"]}


def test_recorded_lifecycle_traces_each_item_to_its_earliest_failure(
        tmp_path):
    module = _module()
    report = _audit_default(module, tmp_path)

    claims = {claim["id"]: claim for claim in report["claims"]}
    assert claims["entry-claim"]["earliest_failure"] == "pass"
    assert claims["edge-claim"]["earliest_failure"] == "pass"
    assert claims["helper-claim"]["earliest_failure"] == "materialization"
    assert claims["ghost-claim"]["earliest_failure"] == "retrieval"
    assert claims["ledger-claim"]["earliest_failure"] == "ledger construction"
    assert claims["render-claim"]["earliest_failure"] == (
        "selection truncation")

    items = _items_by_key(report)
    entry = items[("entry-claim", "pkg_a.Widget.run")]
    assert set(entry["flags"]) == set(FLAG_NAMES)
    assert entry["flags"]["db_present"] is True
    assert entry["canonical_ids"] == ["cid-widget-run"]
    assert entry["flags"]["symbol_menu_present"] is True
    assert entry["flags"]["symbol_selected"] is True
    assert entry["flags"]["component_menu_present"] is True
    assert entry["flags"]["component_selected"] is True
    assert entry["flags"]["route_selected"] is True
    assert entry["flags"]["body_selected"] is True
    assert entry["flags"]["answer_present"] is True
    assert entry["earliest_failure"] == "pass"

    helper_definition = items[(
        "helper-claim", "pkg_a.Helper.emit@module_a/helper.py:5-10")]
    assert helper_definition["flags"]["hydrated"] is True
    assert helper_definition["flags"]["projected"] is False
    assert helper_definition["first_false_flag"] == "projected"
    assert helper_definition["earliest_failure"] == "materialization"

    ghost = items[("ghost-claim", "pkg_c.Ghost.walk")]
    assert ghost["flags"]["db_present"] is False
    assert ghost["flags"]["retrieval_pool_present"] is False
    assert ghost["first_false_flag"] == "db_present"
    assert ghost["earliest_failure"] == "retrieval"

    ledger_witness = items[(
        "ledger-claim", "ledger-claim-w:dropped_from_ledger")]
    assert ledger_witness["flags"]["db_present"] is None
    assert ledger_witness["flags"]["projected"] is True
    assert ledger_witness["flags"]["story_present"] is False
    assert ledger_witness["flags"]["ledger_present"] is False
    assert ledger_witness["earliest_failure"] == "ledger construction"

    edge = items[(
        "edge-claim",
        "pkg_a.Widget.run->pkg_a.Helper.emit@module_a/widget.py:12")]
    assert edge["flags"]["db_present"] is True
    assert edge["earliest_failure"] == "pass"


def test_component_membership_elision_is_null_never_false(tmp_path):
    module = _module()
    report = _audit_default(module, tmp_path)
    items = _items_by_key(report)

    render = items[("render-claim", "pkg_b.Widget.render")]
    # Named only in the unselected G2 card: membership of selected
    # components is elided, so selection is unknowable — never False.
    assert render["flags"]["component_menu_present"] is True
    assert render["flags"]["component_selected"] is None
    assert render["flags"]["symbol_selected"] is False
    assert render["flags"]["route_selected"] is False
    assert render["first_false_flag"] == "route_selected"
    # Symbol/component menus steer the walk; per-item presence there is
    # not required, so the gating chain must not classify on them.
    assert render["earliest_failure"] == "selection truncation"

    # Witness stages that no v1 structure records are null, not guessed.
    entry_witness = items[("entry-claim", "entry-claim-w:emit(value)")]
    assert entry_witness["flags"]["symbol_menu_present"] is None
    assert entry_witness["flags"]["hydrated"] is None
    assert entry_witness["flags"]["ledger_present"] is True


def test_collapsed_body_identity_fails_closed_to_null(tmp_path):
    module = _module()
    db = module.LifecycleDb(_make_db(
        tmp_path / "collapsed.db",
        symbols=[
            ("cid-run-a", "module_a/widget.py", 10, 20, "pkg.Widget.run"),
            ("cid-run-b", "module_b/widget.py", 30, 40, "pkg.Widget.run"),
        ]))
    pool = {"R1": [
        ["pkg.Widget.run", "module_a/widget.py", 10, 20, "",
         "module_a/widget.py", 10, "localized", 0, "question_symbol"],
        ["pkg.Widget.run", "module_b/widget.py", 30, 40, "",
         "module_b/widget.py", 30, "localized", 0, "question_symbol"],
    ]}
    body_menu = "\n".join([
        "DEFINITION BODY CARDS  \x14 choose only required bodies.",
        "  B1. Widget.run  \x14 route root; 11-line definition",
    ])
    route_menu = "\n".join([
        "SCIP ROUTES   expanded from selected graph components.",
        "  R1. pkg.Widget.run",
    ])
    question = {"id": 7, "claims": [_claim(
        "which-module", symbol="pkg.Widget.run", file="module_b/widget.py",
        line_start=30, line_end=40)]}
    final_question = {"id": 7, "passed": False, "claims": [_final(
        "which-module", passed=False,
        missing_definitions=["pkg.Widget.run@module_b/widget.py:30-40"])]}
    answer = _answer(pool=pool, selected_routes=["R1"],
                     completed=["pkg.Widget.run"])
    trace = _trace(route_reply="C1: R1", body_menu=body_menu,
                   body_reply="B1", route_menu=route_menu)

    report = module.audit_question_lifecycle(
        answer, question, trace, final_question, db)

    item = _items_by_key(report)[(
        "which-module", "pkg.Widget.run@module_b/widget.py:30-40")]
    # Two same-name same-extent occurrences behind one card: identity is
    # unproven, so the body flags are null (fail closed), never a guess.
    assert item["flags"]["body_menu_present"] is None
    assert item["flags"]["body_selected"] is None
    assert "body_menu_present" in item["reasons"]
    assert item["flags"]["projected"] is False
    assert item["earliest_failure"] == "materialization"
    assert report["context"]["identity_unproven"] == ["B1"]


def test_oracle_and_unsteered_modes_are_explicit_stubs():
    module = _module()
    with pytest.raises(NotImplementedError, match="oracle"):
        module.main(["--mode", "oracle", "--answers", "a.json",
                     "--report", "r.json", "--out", "o.json"])
    with pytest.raises(NotImplementedError, match="unsteered"):
        module.main(["--mode", "unsteered", "--answers", "a.json",
                     "--report", "r.json", "--out", "o.json"])


def test_payload_reconciles_with_final_report_and_is_byte_stable(tmp_path):
    module = _module()
    meta = {"answers_file": "answers.json", "report_file": "report.json",
            "gold_file": "gold.json", "db_file": "lifecycle.db",
            "trace_dir": "traces", "mode": "recorded"}

    payloads = []
    for index in range(2):
        run_dir = tmp_path / f"run{index}"
        run_dir.mkdir()
        report = _audit_default(module, run_dir)
        payloads.append(module.build_payload([report], meta))
    first, second = (module.dumps_stable(payload) for payload in payloads)
    assert first == second

    summary = payloads[0]["summary"]
    assert summary["questions"] == 1
    assert summary["passed_questions"] == 0
    assert summary["claims"] == 6
    assert summary["passed_claims"] == 2
    matrix = payloads[0]["failing_claim_matrix"]
    assert [row["claim_id"] for row in matrix] == [
        "ghost-claim", "helper-claim", "ledger-claim", "render-claim"]
    assert all(row["question_id"] == 7 for row in matrix)
    stages = {row["claim_id"]: row["earliest_failure"] for row in matrix}
    assert stages["ghost-claim"] == "retrieval"
    assert "timestamp" not in first


def test_tampered_trace_is_rejected_by_hash(tmp_path):
    module = _module()
    payload = json.dumps(_trace()).encode()
    compressed = gzip.compress(payload)
    (tmp_path / "q7.json.gz").write_bytes(compressed)
    answer = {
        "id": 7,
        "diagnostic_trace": {
            "schema": "ariadne-live-diagnostic-v1",
            "file": "q7.json.gz",
            "sha256": hashlib.sha256(compressed).hexdigest(),
        },
    }

    loaded = module.load_question_trace(tmp_path, answer)
    assert loaded["id"] == 7

    (tmp_path / "q7.json.gz").write_bytes(
        gzip.compress(payload.replace(b"final", b"forged")))
    with pytest.raises(ValueError):
        module.load_question_trace(tmp_path, answer)
def test_db_lookups_are_source_isolated(tmp_path):
    module = _module()
    db = module.LifecycleDb(_make_db(
        tmp_path / "multi_source.db",
        symbols=[
            ("cid-a", "module_a/widget.py", 10, 20, "pkg.Widget.run",
             "src1"),
            ("cid-b", "module_b/widget.py", 30, 40, "pkg.Widget.run",
             "src2"),
            ("cid-t", "module_a/target.py", 5, 8, "pkg.Target.hit", "src1"),
            ("cid-t2", "module_b/target.py", 5, 8, "pkg.Target.hit",
             "src2"),
        ],
        edges=[
            ("cid-a", "cid-t", "module_a/widget.py", 12),
            ("cid-b", "cid-t2", "module_b/widget.py", 32),
        ]))

    rows = db.symbol_rows("pkg.Widget.run", "src1")
    assert [row["canonical_id"] for row in rows] == ["cid-a"]
    edge_rows = db.edge_rows("pkg.Widget.run", "pkg.Target.hit", "src1")
    assert [(row["caller_canonical_id"], row["file"])
            for row in edge_rows] == [("cid-a", "module_a/widget.py")]


def test_edge_rows_distinguish_relation_kinds(tmp_path):
    module = _module()
    db = module.LifecycleDb(_make_db(
        tmp_path / "relations.db",
        symbols=[
            ("cid-owner", "module_a/widget.py", 1, 40, "pkg.Owner"),
            ("cid-member", "module_a/widget.py", 10, 20,
             "pkg.Owner.member"),
        ],
        edges=[
            ("cid-owner", "cid-member", "module_a/widget.py", 12,
             "contains"),
            ("cid-owner", "cid-member", "module_a/widget.py", 15, "call"),
        ]))

    rows = db.edge_rows("pkg.Owner", "pkg.Owner.member", "src1")
    assert {row["edge_type"] for row in rows} == {"contains", "call"}
    call_rows = db.edge_rows(
        "pkg.Owner", "pkg.Owner.member", "src1", relation="call")
    assert [(row["edge_type"], row["line"]) for row in call_rows] == [
        ("call", 15)]


def test_pool_absence_is_not_reported_as_proven_retrieval():
    module = _module()
    flags = {name: None for name in module.FLAG_NAMES}
    flags.update({
        "db_present": True, "retrieval_pool_present": False,
        "answer_present": False})

    record = module._finish_item(
        {"key": "pool-absent", "item_type": "symbol", "canonical_ids": []},
        dict(flags), {})
    assert record["first_false_flag"] == "retrieval_pool_present"
    # The persisted pool is post-retention; absence there cannot prove a
    # retrieval failure.
    assert record["earliest_failure"] == "retrieval-or-early-selection"

    absent = {name: None for name in module.FLAG_NAMES}
    absent.update({
        "db_present": False, "retrieval_pool_present": False,
        "answer_present": False})
    record2 = module._finish_item(
        {"key": "db-absent", "item_type": "symbol", "canonical_ids": []},
        absent, {})
    assert record2["earliest_failure"] == "retrieval"


def test_final_presence_never_erases_an_earlier_contradiction():
    module = _module()
    flags = {name: None for name in module.FLAG_NAMES}
    flags.update({
        "db_present": True, "retrieval_pool_present": True,
        "hydrated": False, "answer_present": True})

    record = module._finish_item(
        {"key": "contradiction", "item_type": "symbol",
         "canonical_ids": []}, flags, {})

    assert record["first_false_flag"] == "hydrated"
    assert record["earliest_failure"] == "trace-validation-inconsistency"
    assert record["final_evidence_present"] is True
