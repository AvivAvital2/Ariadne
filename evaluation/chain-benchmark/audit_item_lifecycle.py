#!/usr/bin/env python3
"""Zero-cost per-item lifecycle audit over a recorded live run.

For every reviewed gold item (symbol, definition, edge, witness fragment)
this audit builds one lifecycle record with boolean stage flags — from
database presence through the retrieval pool, the persisted selection
menus, retention, materialization, the formulation story/ledger, and the
final scored answer — using only saved answers, hash-verified trace
sidecars, the reviewed gold, the scored report, and a read-only SQLite
connection. No API is ever called. Database lookups are source-isolated
(the recorded run's source), and edge lookups distinguish relation kinds
(call, type_ref, implements, contains, companion) whenever the reviewed
expectation carries one.

Flags are True, False, or null. Null means the recorded v1 trace cannot
prove the stage either way (for example component membership is elided on
the component menu), and is never guessed. The first false flag along the
gating chain classifies the item's earliest failure with the taxonomy:
retrieval, retrieval-or-early-selection, ranking, selection truncation,
retention, materialization, ledger construction, formulation omission,
validation-scoring. When final evidence is present despite an earlier
false recorded stage, the item is classified trace-validation-inconsistency
and both observations are preserved — a contradiction is never replaced by
a silent pass.

'retrieval-or-early-selection' is deliberate honesty: the persisted
route_candidate_occurrences is the RETAINED route pool (recorded after
component selection and scoping), so absence there cannot distinguish
never-retrieved from removed-before-the-persisted-pool. Exact attribution
needs a future trace schema recording raw hop identities, raw graph nodes
and edges, raw component cards, component selections, pre-scope route
cards, post-scope route cards, body cards, and provider completion
metadata. Only a database absence (db_present=False) is classified as a
proven retrieval gap.

Modes: 'recorded' (implemented) replays the saved live selections;
'oracle' and 'unsteered' are explicit stubs that raise NotImplementedError.
Output is byte-stable across runs: sorted keys, no timestamps.

Regenerating the live22 matrix (expected: 2/22 questions, 9/45 claims,
byte-identical across repeated runs):

    .venv/bin/python evaluation/chain-benchmark/audit_item_lifecycle.py \
        --answers evaluation/chain-benchmark/live22-diagnostic-answers.json \
        --report evaluation/chain-benchmark/live22-rescored-report.json \
        --gold evaluation/chain-benchmark/gold-chain-reviewed-compact.json \
        --out evaluation/chain-benchmark/live22-lifecycle-matrix.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import audit_live_run as live_audit
import measure_ariadne as reviewed_measure


MODES = ("recorded", "oracle", "unsteered")

_BODY_CARD = re.compile(
    r"(?m)^\s*(B\d{1,4})\.\s+(\S+)\s+\x14\s+(.+?); "
    r"(\d+)-line definition( \[route transition\])?\s*$")
_B_LABEL = re.compile(r"\bB(\d{1,4})\b", re.I)
_S_CARD = re.compile(r"(?m)^\s*(S\d{1,4})\.\s+(\S+)\s*$")
_S_LABEL = re.compile(r"\bS(\d{1,4})\b")
_G_CARD = re.compile(r"(?m)^\s*(G\d{1,4})\.\s+(.+?)\s*$")
_G_LABEL = re.compile(r"\bG(\d{1,4})\b")
_R_CARD = re.compile(r"(?m)^\s*(R\d{1,4})\.\s")
_R_LABEL = re.compile(r"\bR(\d{1,4})\b", re.I)
_MATERIALIZED_HEADER = re.compile(r"(?m)^/corpus/([^\n]+?):(\d+)$")
STAGES = (
    "retrieval", "retrieval-or-early-selection", "ranking",
    "selection truncation", "retention", "materialization",
    "ledger construction", "formulation omission", "validation-scoring",
    "trace-validation-inconsistency")

FLAG_NAMES = (
    "db_present", "retrieval_pool_present",
    "symbol_menu_present", "symbol_selected",
    "component_menu_present", "component_selected",
    "route_menu_present", "route_selected",
    "body_menu_present", "body_selected",
    "hydrated", "projected",
    "story_present", "ledger_present", "answer_present")

# Symbol/component menus steer the graph walk; per-item presence there is
# not required for an item to reach the answer, so those flags are
# contextual and the gating chain skips them.
GATING_FLAGS = (
    "db_present", "retrieval_pool_present",
    "route_menu_present", "route_selected",
    "body_menu_present", "body_selected",
    "hydrated", "projected",
    "story_present", "ledger_present", "answer_present")
STAGE_BY_FLAG = {
    "db_present": "retrieval",
    "retrieval_pool_present": "retrieval-or-early-selection",
    "route_menu_present": "ranking",
    "route_selected": "selection truncation",
    "body_menu_present": "ranking",
    "body_selected": "selection truncation",
    "hydrated": "retention",
    "projected": "materialization",
    "story_present": "ledger construction",
    "ledger_present": "ledger construction",
    "answer_present": "formulation omission",
}
APPROXIMATIONS = (
    "component membership is elided on the component menu, so"
    " component_menu_present/component_selected are True or null, never"
    " False, and never gate the earliest-failure classification",
    "symbol_menu_present/symbol_selected are contextual (non-gating):"
    " the symbol menu lists obligation endpoints only, and per-item"
    " presence there is not required by the pipeline",
    "definition/edge symbol-menu and hydrated flags are qualified-name"
    " approximations: v1 traces persist neither menu coordinates nor"
    " hydrated occurrence identities",
    "witness hydrated is null (hydration is name-based) and witness"
    " story_present is extent-only (story nodes/transitions carry no"
    " text)",
    "route_candidate_occurrences is the retained route pool, so pool"
    " absence is classified retrieval-or-early-selection: it cannot"
    " distinguish never-a-candidate from dropped-before-retention; exact"
    " attribution requires a future trace schema persisting raw hop"
    " identities, raw graph nodes/edges, raw component cards, component"
    " selections, pre-scope and post-scope route cards, body cards, and"
    " provider completion metadata",
    "db_present is measured against the current database, which may have"
    " been re-indexed since the recorded run; an item present in the"
    " recorded pool but absent from today's database does not gate",
    "a symbol absent from the pool but visible verbatim in a persisted"
    " selection prompt does not gate at retrieval (it was surfaced)",
)


def load_question_trace(trace_dir: Path | str, answer: dict) -> dict:
    """Load one answer's trace sidecar, hash-verified, failing closed."""
    return live_audit.load_trace(Path(trace_dir), answer)
class LifecycleDb:
    """Read-only lookups against the SQLite SCIP tables, source-isolated."""

    def __init__(self, path: Path | str):
        self._connection = sqlite3.connect(
            f"file:{Path(path)}?mode=ro", uri=True)

    def symbol_rows(self, qualified_name: str, source: str) -> list[dict]:
        rows = self._connection.execute(
            "SELECT canonical_id, file, line_start, line_end"
            " FROM scip_symbols"
            " WHERE qualified_name = ? AND source_name = ?"
            " ORDER BY canonical_id, file, line_start",
            (qualified_name, source)).fetchall()
        return [
            {"canonical_id": str(row[0]), "file": str(row[1] or ""),
             "line_start": int(row[2] or 0), "line_end": int(row[3] or 0)}
            for row in rows]

    def edge_rows(self, caller: str, callee: str, source: str,
                  relation: str | None = None) -> list[dict]:
        query = (
            "SELECT DISTINCT e.caller_canonical_id, e.callee_canonical_id,"
            " e.edge_type, e.file, e.line FROM scip_edges e"
            " JOIN scip_symbols c ON c.canonical_id = e.caller_canonical_id"
            " JOIN scip_symbols x ON x.canonical_id = e.callee_canonical_id"
            " WHERE c.qualified_name = ? AND x.qualified_name = ?"
            " AND c.source_name = ? AND x.source_name = ?")
        parameters = [caller, callee, source, source]
        if relation is not None:
            query += " AND e.edge_type = ?"
            parameters.append(relation)
        query += (" ORDER BY e.file, e.line, e.edge_type,"
                  " e.caller_canonical_id")
        rows = self._connection.execute(query, parameters).fetchall()
        return [
            {"caller_canonical_id": str(row[0]),
             "callee_canonical_id": str(row[1]),
             "edge_type": str(row[2] or ""),
             "file": str(row[3] or ""), "line": int(row[4] or 0)}
            for row in rows]


def _phase_call(trace: dict, phase: str) -> dict | None:
    for call in trace.get("llm_completions", ()) or ():
        if str(call.get("phase") or "") == phase:
            return call
    return None


def _call_text(call: dict) -> str:
    return "\n".join(
        live_audit._message_content(message.get("content"))
        for message in call.get("messages", ()) or ())


def _owner_short(qualified_name: str) -> str:
    parts = str(qualified_name or "").split(".")
    return ".".join(parts[-2:]) if len(parts) > 1 else str(qualified_name)


def _name_in_text(short_name: str, text: str) -> bool:
    return re.search(
        r"(?<![\w.])" + re.escape(short_name) + r"(?![\w])", text) is not None


def _covers(spans: list[dict], file: str, line_start: int,
            line_end: int) -> bool:
    return reviewed_measure._span_covers(spans, file, line_start, line_end)


def _text_covers(spans: list[dict], witness: dict) -> bool:
    fragment = reviewed_measure._normalize(witness["fragment"])
    return any(
        reviewed_measure._same_file(span["file"], witness["file"])
        and int(span["line_start"]) <= witness["line_end"]
        and int(span["line_end"]) >= witness["line_start"]
        and fragment in reviewed_measure._normalize(span["text"])
        for span in spans)


def parse_body_menu_prompt(text: str) -> dict:
    """Menu label -> the card exactly as the selector saw it."""
    cards = {}
    for label, owner, role, extent, required in _BODY_CARD.findall(text or ""):
        role = role.strip()
        relation = parent = None
        if role != "route root" and " from " in role:
            relation, parent = role.split(" from ", 1)
        cards[label] = {
            "owner": owner,
            "role": role,
            "relation": relation,
            "parent": parent,
            "extent": int(extent),
            "required": bool(required),
        }
    return cards


def _occurrence_row(row, route_id: str) -> dict | None:
    if not isinstance(row, (list, tuple)) or len(row) < 4:
        return None
    qualified_name = str(row[0] or "")
    if not qualified_name:
        return None

    def _field(index, default=""):
        return row[index] if len(row) > index and row[index] is not None \
            else default

    return {
        "qualified_name": qualified_name,
        "file": str(_field(1)),
        "line_start": int(_field(2, 0) or 0),
        "line_end": int(_field(3, 0) or 0),
        "parent_qualified_name": str(_field(4)),
        "call_site_file": str(_field(5)),
        "call_site_line": int(_field(6, 0) or 0),
        "relation": str(_field(7)),
        "route_id": str(route_id),
    }


def _pool_by_route(answer: dict) -> dict:
    routes = {}
    for route_id, occurrences in (
            answer.get("route_candidate_occurrences") or {}).items():
        rows = []
        for row in occurrences or ():
            occurrence = _occurrence_row(row, str(route_id))
            if occurrence is not None:
                rows.append(occurrence)
        routes[str(route_id)] = rows
    return routes


def _identity(occurrence: dict) -> tuple:
    return (occurrence["qualified_name"], occurrence["file"],
            occurrence["line_start"], occurrence["line_end"])


def _distinct(occurrences: list[dict]) -> list[dict]:
    seen = {}
    for occurrence in occurrences:
        seen.setdefault(_identity(occurrence), occurrence)
    return list(seen.values())


def _public_occurrence(occurrence: dict) -> dict:
    return {
        "qualified_name": occurrence["qualified_name"],
        "file": occurrence["file"],
        "line_start": occurrence["line_start"],
        "line_end": occurrence["line_end"],
    }


def _definition_spans(occurrences: list[dict]) -> list[dict]:
    return [
        {"file": occurrence["file"],
         "line_start": occurrence["line_start"],
         "line_end": occurrence["line_end"]}
        for occurrence in occurrences
        if occurrence["file"] and occurrence["line_start"] > 0
        and occurrence["line_end"] >= occurrence["line_start"]]


def _site_spans(occurrences: list[dict]) -> list[dict]:
    return [
        {"file": occurrence["call_site_file"],
         "line_start": occurrence["call_site_line"],
         "line_end": occurrence["call_site_line"]}
        for occurrence in occurrences
        if occurrence.get("call_site_file")
        and (occurrence.get("call_site_line") or 0) > 0]


def _card_matches(card: dict, pool: list[dict]) -> list[dict]:
    matches = [
        occurrence for occurrence in pool
        if occurrence["line_end"] > occurrence["line_start"]
        and occurrence["line_end"] - occurrence["line_start"] + 1
        == card["extent"]
        and _owner_short(occurrence["qualified_name"]) == card["owner"]]
    if card["parent"]:
        corroborated = [
            occurrence for occurrence in matches
            if _owner_short(occurrence["parent_qualified_name"])
            == card["parent"]]
        if corroborated:
            matches = corroborated
    return _distinct(matches)


def anchor_menu_cards(cards: dict, answer: dict) -> dict:
    """Anchor each card to exact occurrences; never guess between modules."""
    pool = [
        occurrence for occurrences in _pool_by_route(answer).values()
        for occurrence in occurrences]
    selected_routes = {
        str(route) for route in answer.get("selected_route_ids") or ()}
    selected_pool = [
        occurrence for occurrence in pool
        if occurrence["route_id"] in selected_routes]
    anchored = {}
    for label, card in cards.items():
        scope = "selected-routes"
        matches = _card_matches(card, selected_pool)
        if not matches:
            scope = "retained-routes"
            matches = _card_matches(card, pool)
        names = sorted({match["qualified_name"] for match in matches})
        if len(matches) == 1:
            status = "exact"
            qualified_name = matches[0]["qualified_name"]
            occurrence = _public_occurrence(matches[0])
        elif len(names) == 1 and matches:
            status = "collapsed"
            qualified_name = names[0]
            occurrence = None
        elif matches:
            status = "ambiguous"
            qualified_name = None
            occurrence = None
        else:
            status = "unmapped"
            qualified_name = None
            occurrence = None
        anchored[label] = {
            **card,
            "status": status,
            "qualified_name": qualified_name,
            "occurrence": occurrence,
            "candidates": [_public_occurrence(match) for match in matches],
            "scope": scope,
        }
    return anchored


def _persisted_card(card: dict, record: dict) -> dict:
    """A card whose identity was recorded at menu-build time: no guessing."""
    qualified_name = str(record.get("qualified_name") or "")
    occurrence = {
        "qualified_name": qualified_name,
        "file": str(record.get("file") or ""),
        "line_start": int(record.get("line_start") or 0),
        "line_end": int(record.get("line_end") or 0),
    }
    candidates = [
        {"qualified_name": qualified_name, "file": str(extent[0]),
         "line_start": int(extent[1]), "line_end": int(extent[2])}
        for extent in record.get("extents") or () if len(extent) >= 3]
    status = "persisted" if len(candidates) <= 1 else "persisted-collapsed"
    return {
        **card,
        "status": status,
        "qualified_name": qualified_name,
        "occurrence": occurrence,
        "candidates": candidates or [occurrence],
        "scope": "persisted-menu",
    }


def _raw_body_labels(cards: dict, reply: str) -> tuple[list[str], list[str]]:
    raw, unknown = [], []
    for digits in _B_LABEL.findall(reply or ""):
        label = f"B{digits}"
        if label in cards:
            if label not in raw:
                raw.append(label)
        elif label not in unknown:
            unknown.append(label)
    return raw, unknown


def body_selection_report(answer: dict, trace: dict) -> dict:
    """Raw model body choices kept separate from deterministic completion."""
    completed = [
        str(symbol) for symbol in answer.get("selected_body_symbols") or ()
        if symbol]
    call = _phase_call(trace, "scip-body-select")
    if call is None:
        return {
            "mode": "auto",
            "menu_size": None,
            "raw_labels": [],
            "unknown_labels": [],
            "cards": {},
            "raw_symbols": [],
            "completed_symbols": completed,
            "completion_added": [
                {"symbol": symbol, "label": None, "reason": "auto-no-menu"}
                for symbol in completed],
            "identity_unproven": [],
        }
    parsed = parse_body_menu_prompt(_call_text(call))
    persisted = (
        ((answer.get("graph_diagnostics") or {}).get("post_walk_selection")
         or {}).get("body_menu_occurrences") or {})
    if persisted:
        cards = {}
        for label, card in parsed.items():
            record = persisted.get(label)
            if record is None:
                raise ValueError(
                    f"menu label {label} missing from persisted"
                    " body-menu occurrences")
            cards[label] = _persisted_card(card, record)
    else:
        cards = anchor_menu_cards(parsed, answer)
    reply = str(call.get("response") or "")
    raw_labels, unknown_labels = _raw_body_labels(cards, reply)
    mode = "llm" if raw_labels else "fallback-all"
    raw_symbols = [
        cards[label]["qualified_name"] for label in raw_labels
        if cards[label]["qualified_name"]]
    label_by_symbol = {}
    for label, card in cards.items():
        if card["qualified_name"]:
            label_by_symbol.setdefault(card["qualified_name"], label)
    for symbol in completed:
        if symbol not in label_by_symbol and not any(
                card["owner"] == _owner_short(symbol)
                for card in cards.values()):
            raise ValueError(
                f"completed body symbol not on the parsed menu: {symbol}")
    raw_set = set(raw_symbols)
    completion_added = []
    for symbol in completed:
        if mode == "llm" and symbol in raw_set:
            continue
        label = label_by_symbol.get(symbol)
        card = cards.get(label) if label else None
        if mode == "fallback-all":
            reason = "fallback-all"
        elif card is None:
            reason = "unanchored-menu-card"
        elif card["required"]:
            reason = "route-transition"
        else:
            reason = "obligation-required"
        completion_added.append(
            {"symbol": symbol, "label": label, "reason": reason})
    selected_labels = set(raw_labels)
    if mode == "fallback-all":
        selected_labels = set(cards)
    identity_unproven = [
        label for label, card in cards.items()
        if card["status"] in (
            "collapsed", "ambiguous", "persisted-collapsed")
        and (label in selected_labels
             or (card["qualified_name"] or "") in set(completed))]
    return {
        "mode": mode,
        "menu_size": len(cards),
        "raw_labels": raw_labels,
        "unknown_labels": unknown_labels,
        "cards": cards,
        "raw_symbols": raw_symbols,
        "completed_symbols": completed,
        "completion_added": completion_added,
        "identity_unproven": sorted(identity_unproven),
    }


def symbol_menu_report(trace: dict) -> dict | None:
    call = _phase_call(trace, "scip-symbol-select")
    if call is None:
        return None
    cards = dict(_S_CARD.findall(_call_text(call)))
    reply = str(call.get("response") or "")
    selected_labels = {
        f"S{digits}" for digits in _S_LABEL.findall(reply)} & set(cards)
    return {
        "menu_size": len(cards),
        "names": set(cards.values()),
        "selected_names": {cards[label] for label in selected_labels},
    }


def component_menu_report(trace: dict) -> dict | None:
    call = _phase_call(trace, "scip-component-select")
    if call is None:
        return None
    cards = dict(_G_CARD.findall(_call_text(call)))
    cards = {
        label: text for label, text in cards.items()
        if _G_LABEL.fullmatch(label)}
    reply = str(call.get("response") or "")
    selected_labels = {
        f"G{digits}" for digits in _G_LABEL.findall(reply)} & set(cards)
    return {
        "menu_size": len(cards),
        "all_text": "\n".join(
            cards[label] for label in sorted(cards)),
        "selected_text": "\n".join(
            cards[label] for label in sorted(selected_labels)),
    }


def route_menu_labels(trace: dict) -> set[str] | None:
    call = _phase_call(trace, "scip-exact-route-select")
    if call is None:
        return None
    return set(_R_CARD.findall(_call_text(call)))


def parse_materialized_blocks(text: str) -> list[dict]:
    """Exact (file, line span, text) blocks out of the materialized record."""
    blocks = []
    headers = list(_MATERIALIZED_HEADER.finditer(text or ""))
    for index, match in enumerate(headers):
        end = (headers[index + 1].start()
               if index + 1 < len(headers) else len(text))
        lines = text[match.end():end].splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines and lines[0].strip() == "```":
            lines.pop(0)
        body = []
        for line in lines:
            if line.strip() == "```":
                break
            body.append(line)
        if not body:
            continue
        start = int(match.group(2))
        blocks.append({
            "file": match.group(1),
            "line_start": start,
            "line_end": start + len(body) - 1,
            "text": "\n".join(body),
        })
    return blocks


def _question_context(answer: dict, trace: dict,
                      final_question: dict) -> dict:
    pool_by_route = _pool_by_route(answer)
    pool = [
        occurrence for occurrences in pool_by_route.values()
        for occurrence in occurrences]
    body = body_selection_report(answer, trace)
    cards = body["cards"]
    anchored = [
        card for card in cards.values() if card.get("occurrence")]
    completed_set = set(body["completed_symbols"])
    formulation_call = live_audit.formulation_completion(trace)
    prompt = live_audit.formulation_prompt(formulation_call)
    evidence_ir = live_audit.parse_formulation_ir(prompt)
    story_spans = [
        {"file": value["file"], "line_start": int(value["line_start"]),
         "line_end": int(value["line_end"])}
        for value in (*evidence_ir["nodes"].values(),
                      *evidence_ir["edges"].values())]
    chunk_spans = [
        {"file": value["file"], "line_start": int(value["line_start"]),
         "line_end": int(value["line_end"]),
         "text": str(value.get("text") or "")}
        for value in evidence_ir["chunks"].values()]
    return {
        "source": str(trace.get("source") or ""), "pool_by_route": pool_by_route,
        "pool": pool,
        "pool_def_spans": _definition_spans(pool),
        "pool_site_spans": _site_spans(pool),
        "selection_text": live_audit._selection_prompt_text(trace),
        "symbol_menu": symbol_menu_report(trace),
        "component_menu": component_menu_report(trace),
        "route_menu_labels": route_menu_labels(trace),
        "retained_routes": {
            str(route) for route in answer.get("selected_route_ids") or ()},
        "body": body,
        "all_card_spans": _definition_spans(
            [card["occurrence"] for card in anchored]),
        "completed_card_spans": _definition_spans(
            [card["occurrence"] for card in anchored
             if card["qualified_name"] in completed_set]),
        "unanchored_shorts": {
            card["owner"] for card in cards.values()
            if card["status"] in ("collapsed", "ambiguous", "unmapped",
                                  "persisted-collapsed")},
        "completed_set": completed_set,
        "hydrated": {
            str(symbol) for symbol in answer.get("hydrated_symbols") or ()
            if symbol},
        "blocks": parse_materialized_blocks(
            (trace.get("materialized_evidence") or {}).get("text") or ""),
        "story_spans": story_spans,
        "node_symbols": {
            value["symbol"] for value in evidence_ir["nodes"].values()},
        "chunk_spans": chunk_spans,
        "final_by_id": {
            str(row.get("id")): row
            for row in final_question.get("claims") or ()},
    }


def _routes_for(context: dict, predicate) -> list[str]:
    return sorted(
        route_id for route_id, occurrences
        in context["pool_by_route"].items()
        if any(predicate(occurrence) for occurrence in occurrences))


def _route_flags(context: dict, route_ids: list[str],
                 flags: dict, reasons: dict) -> None:
    menu = context["route_menu_labels"]
    if menu is None:
        flags["route_menu_present"] = None
        reasons["route_menu_present"] = (
            "no scip-exact-route-select phase recorded")
    else:
        flags["route_menu_present"] = any(
            route_id in menu for route_id in route_ids)
    flags["route_selected"] = any(
        route_id in context["retained_routes"] for route_id in route_ids)


def _symbol_menu_flags(context: dict, names: list[str], flags: dict,
                       reasons: dict, *, approximate: bool) -> None:
    menu = context["symbol_menu"]
    if menu is None:
        flags["symbol_menu_present"] = None
        flags["symbol_selected"] = None
        reasons["symbol_menu_present"] = (
            "no scip-symbol-select phase recorded")
        return
    flags["symbol_menu_present"] = all(
        name in menu["names"] for name in names)
    flags["symbol_selected"] = all(
        name in menu["selected_names"] for name in names)
    if approximate:
        reasons["symbol_menu_present"] = (
            "qualified-name approximation; the symbol menu carries no"
            " coordinates")


def _component_flags(context: dict, shorts: list[str], flags: dict,
                     reasons: dict) -> None:
    menu = context["component_menu"]
    if menu is None:
        flags["component_menu_present"] = None
        flags["component_selected"] = None
        reasons["component_menu_present"] = (
            "no scip-component-select phase recorded")
        return
    named_any = all(
        _name_in_text(short, menu["all_text"]) for short in shorts)
    named_selected = all(
        _name_in_text(short, menu["selected_text"]) for short in shorts)
    flags["component_menu_present"] = True if named_any else None
    flags["component_selected"] = True if named_selected else None
    if not named_selected:
        reasons["component_selected"] = (
            "component membership is elided on the menu; absence of the"
            " name proves nothing")
    if not named_any:
        reasons["component_menu_present"] = reasons["component_selected"]


def _body_span_flags(context: dict, file: str, line_start: int,
                     line_end: int, shorts: list[str], flags: dict,
                     reasons: dict) -> None:
    body = context["body"]
    if body["mode"] == "auto":
        flags["body_menu_present"] = None
        flags["body_selected"] = None
        reasons["body_menu_present"] = (
            "no scip-body-select phase recorded; bodies were auto-selected")
        return
    blocked = [
        short for short in shorts
        if short in context["unanchored_shorts"]]
    if not shorts and context["unanchored_shorts"]:
        blocked = sorted(context["unanchored_shorts"])
    on_menu = _covers(context["all_card_spans"], file, line_start, line_end)
    if on_menu:
        flags["body_menu_present"] = True
    elif blocked:
        flags["body_menu_present"] = None
        reasons["body_menu_present"] = (
            "identity-collapsed menu card(s) prevent proof: "
            + ", ".join(sorted(set(blocked))))
    else:
        flags["body_menu_present"] = False
    selected = _covers(
        context["completed_card_spans"], file, line_start, line_end)
    if selected:
        flags["body_selected"] = True
    elif blocked:
        flags["body_selected"] = None
        reasons.setdefault("body_selected", (
            "identity-collapsed menu card(s) prevent proof: "
            + ", ".join(sorted(set(blocked)))))
    else:
        flags["body_selected"] = False


def _empty_flags() -> dict:
    return {name: None for name in FLAG_NAMES}


def _symbol_item(qualified_name: str, expected: dict, context: dict,
                 db: LifecycleDb, missing: dict) -> dict:
    flags = _empty_flags()
    reasons = {}
    short = _owner_short(qualified_name)
    rows = db.symbol_rows(qualified_name, context["source"])
    flags["db_present"] = bool(rows)
    if not rows:
        reasons["db_present"] = "qualified name not in scip_symbols"
    in_pool = any(
        occurrence["qualified_name"] == qualified_name
        or occurrence["parent_qualified_name"] == qualified_name
        for occurrence in context["pool"])
    flags["retrieval_pool_present"] = in_pool
    visible = qualified_name in context["selection_text"]
    _symbol_menu_flags(
        context, [qualified_name], flags, reasons, approximate=False)
    _component_flags(context, [short], flags, reasons)
    route_ids = _routes_for(
        context,
        lambda occurrence: occurrence["qualified_name"] == qualified_name
        or occurrence["parent_qualified_name"] == qualified_name)
    _route_flags(context, route_ids, flags, reasons)
    body = context["body"]
    if body["mode"] == "auto":
        flags["body_menu_present"] = None
        reasons["body_menu_present"] = (
            "no scip-body-select phase recorded; bodies were auto-selected")
    else:
        anchored_names = {
            card["qualified_name"] for card in body["cards"].values()
            if card["qualified_name"]}
        if qualified_name in anchored_names:
            flags["body_menu_present"] = True
        elif short in context["unanchored_shorts"]:
            flags["body_menu_present"] = None
            reasons["body_menu_present"] = (
                f"identity-collapsed menu card(s) prevent proof: {short}")
        else:
            flags["body_menu_present"] = False
    flags["body_selected"] = qualified_name in context["completed_set"]
    flags["hydrated"] = qualified_name in context["hydrated"]
    coordinates = [
        (file, line_start, line_end)
        for file, line_start, line_end, symbol in expected["definitions"]
        if symbol == qualified_name]
    if not coordinates:
        coordinates = [
            (row["file"], row["line_start"], row["line_end"])
            for row in rows if row["file"] and row["line_start"] > 0]
        if coordinates:
            reasons["projected"] = (
                "coordinates taken from the current database, not the"
                " reviewed gold")
    if coordinates:
        flags["projected"] = any(
            _covers(context["blocks"], file, line_start, line_end)
            for file, line_start, line_end in coordinates)
        flags["ledger_present"] = any(
            _covers(context["chunk_spans"], file, line_start, line_end)
            for file, line_start, line_end in coordinates)
    else:
        reasons["projected"] = "no definition coordinates available"
        reasons["ledger_present"] = reasons["projected"]
    flags["story_present"] = qualified_name in context["node_symbols"]
    flags["answer_present"] = qualified_name not in missing["symbols"]
    return _finish_item(
        {"key": qualified_name, "item_type": "symbol",
         "qualified_name": qualified_name,
         "canonical_ids": sorted({row["canonical_id"] for row in rows}),
         "selection_prompt_visible": visible},
        flags, reasons)


def _definition_item(file: str, line_start: int, line_end: int,
                     qualified_name: str, context: dict, db: LifecycleDb,
                     missing: dict) -> dict:
    flags = _empty_flags()
    reasons = {}
    short = _owner_short(qualified_name)
    rows = db.symbol_rows(qualified_name, context["source"])
    matching = [
        row for row in rows
        if row["file"]
        and reviewed_measure._same_file(row["file"], file)
        and row["line_start"] <= line_end and row["line_end"] >= line_start]
    flags["db_present"] = bool(matching)
    if rows and not matching:
        reasons["db_present"] = (
            "symbol known to the database only at other coordinates")
    elif not rows:
        reasons["db_present"] = "qualified name not in scip_symbols"
    flags["retrieval_pool_present"] = _covers(
        context["pool_def_spans"], file, line_start, line_end)
    _symbol_menu_flags(
        context, [qualified_name], flags, reasons, approximate=True)
    _component_flags(context, [short], flags, reasons)
    route_ids = _routes_for(
        context,
        lambda occurrence: occurrence["qualified_name"] == qualified_name
        and reviewed_measure._same_file(occurrence["file"], file)
        and occurrence["line_start"] <= line_end
        and occurrence["line_end"] >= line_start)
    _route_flags(context, route_ids, flags, reasons)
    _body_span_flags(
        context, file, line_start, line_end, [short], flags, reasons)
    if context["body"]["mode"] == "auto":
        flags["body_selected"] = qualified_name in context["completed_set"]
        reasons["body_selected"] = (
            "qualified-name approximation; no menu was recorded")
    flags["hydrated"] = qualified_name in context["hydrated"]
    reasons["hydrated"] = (
        "qualified-name approximation; v1 traces do not persist hydrated"
        " occurrence identities")
    flags["projected"] = _covers(
        context["blocks"], file, line_start, line_end)
    flags["story_present"] = _covers(
        context["story_spans"], file, line_start, line_end)
    flags["ledger_present"] = _covers(
        context["chunk_spans"], file, line_start, line_end)
    key = f"{qualified_name}@{file}:{line_start}-{line_end}"
    flags["answer_present"] = key not in missing["definitions"]
    return _finish_item(
        {"key": key, "item_type": "definition",
         "qualified_name": qualified_name, "file": file,
         "line_start": line_start, "line_end": line_end,
         "canonical_ids": sorted(
             {row["canonical_id"] for row in (matching or rows)})},
        flags, reasons)
def _edge_item(file: str, line: int, caller: str, callee: str,
               context: dict, db: LifecycleDb, missing: dict,
               relation: str | None = None) -> dict:
    flags = _empty_flags()
    reasons = {}
    shorts = [_owner_short(caller), _owner_short(callee)]
    rows = db.edge_rows(caller, callee, context["source"])
    matching = [
        row for row in rows
        if reviewed_measure._same_file(row["file"], file)
        and row["line"] == line
        and (relation is None or row["edge_type"] == relation)]
    flags["db_present"] = bool(matching)
    if rows and not matching:
        reasons["db_present"] = (
            "edge in scip_edges only as: " + ", ".join(sorted({
                f"{row['edge_type']}@{row['file']}:{row['line']}"
                for row in rows[:8]})))
    elif not rows:
        reasons["db_present"] = "caller->callee pair not in scip_edges"
    edge_spans = [*context["pool_def_spans"], *context["pool_site_spans"]]
    flags["retrieval_pool_present"] = _covers(edge_spans, file, line, line)
    _symbol_menu_flags(
        context, [caller, callee], flags, reasons, approximate=True)
    _component_flags(context, shorts, flags, reasons)
    route_ids = _routes_for(
        context,
        lambda occurrence: (
            occurrence["call_site_file"]
            and reviewed_measure._same_file(
                occurrence["call_site_file"], file)
            and occurrence["call_site_line"] == line)
        or (occurrence["file"]
            and reviewed_measure._same_file(occurrence["file"], file)
            and occurrence["line_start"] <= line
            and occurrence["line_end"] >= line))
    _route_flags(context, route_ids, flags, reasons)
    _body_span_flags(context, file, line, line, shorts, flags, reasons)
    if context["body"]["mode"] == "auto":
        flags["body_selected"] = all(
            name in context["completed_set"] for name in (caller, callee))
        reasons["body_selected"] = (
            "qualified-name approximation; no menu was recorded")
    flags["hydrated"] = all(
        name in context["hydrated"] for name in (caller, callee))
    reasons["hydrated"] = (
        "qualified-name approximation over both endpoints; v1 traces do"
        " not persist hydrated occurrence identities")
    flags["projected"] = _covers(context["blocks"], file, line, line)
    flags["story_present"] = _covers(
        context["story_spans"], file, line, line)
    flags["ledger_present"] = _covers(
        context["chunk_spans"], file, line, line)
    key = f"{caller}->{callee}@{file}:{line}"
    flags["answer_present"] = key not in missing["edges"]
    canonical_ids = sorted({
        identifier for row in matching
        for identifier in (
            row["caller_canonical_id"], row["callee_canonical_id"])})
    return _finish_item(
        {"key": key, "item_type": "edge", "caller": caller,
         "callee": callee, "file": file, "line_start": line,
         "line_end": line, "relation": relation,
         "matched_edge_types": sorted(
             {row["edge_type"] for row in matching}),
         "canonical_ids": canonical_ids},
        flags, reasons)


def _witness_item(witness: dict, context: dict, missing: dict) -> dict:
    flags = _empty_flags()
    reasons = {}
    file = witness["file"]
    line_start = witness["line_start"]
    line_end = witness["line_end"]
    flags["db_present"] = None
    reasons["db_present"] = "the database stores no source text"
    witness_spans = [
        *context["pool_def_spans"], *context["pool_site_spans"]]
    flags["retrieval_pool_present"] = _covers(
        witness_spans, file, line_start, line_end)
    flags["symbol_menu_present"] = None
    flags["symbol_selected"] = None
    reasons["symbol_menu_present"] = (
        "the symbol menu names symbols; a witness extent is not"
        " representable there")
    flags["component_menu_present"] = None
    flags["component_selected"] = None
    reasons["component_menu_present"] = reasons["symbol_menu_present"]
    route_ids = _routes_for(
        context,
        lambda occurrence: (
            occurrence["file"]
            and reviewed_measure._same_file(occurrence["file"], file)
            and occurrence["line_start"] <= line_end
            and occurrence["line_end"] >= line_start)
        or (occurrence["call_site_file"]
            and reviewed_measure._same_file(
                occurrence["call_site_file"], file)
            and line_start <= occurrence["call_site_line"] <= line_end))
    _route_flags(context, route_ids, flags, reasons)
    _body_span_flags(
        context, file, line_start, line_end, [], flags, reasons)
    flags["hydrated"] = None
    reasons["hydrated"] = (
        "hydration is recorded by qualified name only; a witness extent"
        " cannot be mapped to it")
    flags["projected"] = _text_covers(context["blocks"], witness)
    flags["story_present"] = _covers(
        context["story_spans"], file, line_start, line_end)
    reasons["story_present"] = (
        "extent-only; story nodes/transitions carry no text")
    flags["ledger_present"] = _text_covers(context["chunk_spans"], witness)
    key = f"{witness['id']}:{witness['fragment']}"
    flags["answer_present"] = key not in missing["witnesses"]
    return _finish_item(
        {"key": key, "item_type": "witness", "witness_id": witness["id"],
         "file": file, "line_start": line_start, "line_end": line_end,
         "fragment": witness["fragment"], "canonical_ids": []},
        flags, reasons)


def _first_false_flag(item: dict) -> str | None:
    flags = item["flags"]
    pool_present = flags["retrieval_pool_present"]
    visible = bool(item.get("selection_prompt_visible"))
    for flag in GATING_FLAGS:
        value = flags[flag]
        if value is not False:
            continue
        if flag == "db_present" and pool_present is True:
            # The current database may post-date the recorded run.
            continue
        if flag == "retrieval_pool_present" and visible:
            # Surfaced verbatim on a persisted selection prompt.
            continue
        return flag
    return None
def _finish_item(record: dict, flags: dict, reasons: dict) -> dict:
    record["flags"] = flags
    record["reasons"] = {key: reasons[key] for key in sorted(reasons)}
    first_false = _first_false_flag(
        {**record, "flags": flags})
    record["first_false_flag"] = first_false
    record["final_evidence_present"] = flags["answer_present"] is True
    if flags["answer_present"] is True:
        # Final evidence present while an earlier recorded stage is false
        # is a contradiction: preserve both observations, never a silent
        # pass.
        record["earliest_failure"] = (
            "trace-validation-inconsistency" if first_false else "pass")
    else:
        record["earliest_failure"] = STAGE_BY_FLAG[
            first_false or "answer_present"]
    return record
def _edge_relations(claim: dict) -> dict:
    """(file, line, caller, callee) -> unambiguous reviewed edge_type."""
    relations = {}
    for path in claim.get("candidate_paths", ()) or ():
        for edge in path.get("edges", ()) or ():
            key = (
                str(edge.get("file") or ""), int(edge.get("line") or 0),
                str(edge.get("caller")
                    or edge.get("caller_canonical_id") or ""),
                str(edge.get("callee")
                    or edge.get("callee_canonical_id") or ""))
            edge_type = str(edge.get("edge_type") or "")
            if not edge_type:
                continue
            if key in relations and relations[key] != edge_type:
                relations[key] = None
            elif key not in relations:
                relations[key] = edge_type
    return relations


def audit_question_lifecycle(answer: dict, question: dict, trace: dict,
                             final_question: dict,
                             db: LifecycleDb) -> dict:
    """Per reviewed item, one lifecycle record over one recorded trace."""
    context = _question_context(answer, trace, final_question)
    claims = []
    for claim in question.get("claims", ()) or ():
        if claim.get("review", {}).get("status") != "accepted":
            continue
        identifier = str(claim.get("id"))
        final_row = context["final_by_id"].get(identifier)
        if final_row is None:
            raise ValueError(
                f"no final score for reviewed claim {identifier}")
        expected = reviewed_measure._claim_expectations(claim)
        missing = {
            "symbols": set(final_row.get("missing_symbols") or ()),
            "definitions": set(final_row.get("missing_definitions") or ()),
            "edges": set(final_row.get("missing_edges") or ()),
            "witnesses": set(
                final_row.get("missing_witness_fragments") or ()),
        }
        edge_relations = _edge_relations(claim)
        items = []
        for qualified_name in expected["symbols"]:
            items.append(_symbol_item(
                qualified_name, expected, context, db, missing))
        for file, line_start, line_end, symbol in expected["definitions"]:
            items.append(_definition_item(
                file, line_start, line_end, symbol, context, db, missing))
        for file, line, caller, callee in expected["edges"]:
            items.append(_edge_item(
                file, line, caller, callee, context, db, missing,
                relation=edge_relations.get(
                    (file, line, caller, callee))))
        for witness in expected["witness_fragments"]:
            items.append(_witness_item(witness, context, missing))
        final_passed = bool(final_row.get("passed"))
        if final_passed:
            earliest = "pass"
        else:
            stages = [
                item["earliest_failure"] for item in items
                if item["flags"]["answer_present"] is False]
            earliest = (
                min(stages, key=STAGES.index) if stages
                else "validation-scoring")
        claims.append({
            "id": identifier,
            "final_passed": final_passed,
            "earliest_failure": earliest,
            "items": items,
        })
    body = context["body"]
    menu_labels = context["route_menu_labels"]
    return {
        "id": int(answer.get("id") or question.get("id") or 0),
        "final_passed": bool(final_question.get("passed")),
        "claims": claims,
        "context": {
            "pool_routes": len(context["pool_by_route"]),
            "pool_occurrences": len(_distinct(context["pool"])),
            "route_menu_routes": (
                len(menu_labels) if menu_labels is not None else None),
            "selected_routes": len(context["retained_routes"]),
            "body_mode": body["mode"],
            "body_menu_cards": body["menu_size"],
            "raw_body_labels": body["raw_labels"],
            "completed_bodies": len(body["completed_symbols"]),
            "completion_added": body["completion_added"],
            "identity_unproven": body["identity_unproven"],
            "hydrated_symbols": len(context["hydrated"]),
            "materialized_blocks": len(context["blocks"]),
            "story_nodes": len(context["node_symbols"]),
            "ledger_chunks": len(context["chunk_spans"]),
        },
    }


def build_payload(question_reports: list[dict], meta: dict) -> dict:
    """Assemble the byte-stable payload: summary, matrix, full records."""
    all_claims = [
        (report["id"], claim)
        for report in question_reports for claim in report["claims"]]
    all_items = [item for _qid, claim in all_claims
                 for item in claim["items"]]
    claim_failures = Counter(
        claim["earliest_failure"] for _qid, claim in all_claims
        if not claim["final_passed"])
    item_failures = Counter(
        item["earliest_failure"] for item in all_items
        if item["flags"]["answer_present"] is False)
    flag_counts = {}
    for flag in FLAG_NAMES:
        values = Counter(
            {True: "true", False: "false", None: "null"}[
                item["flags"][flag]]
            for item in all_items)
        flag_counts[flag] = {
            "true": values.get("true", 0),
            "false": values.get("false", 0),
            "null": values.get("null", 0),
        }
    matrix = [
        {
            "question_id": question_id,
            "claim_id": claim["id"],
            "earliest_failure": claim["earliest_failure"],
            "items": [
                {"key": item["key"], "item_type": item["item_type"],
                 "first_false_flag": item["first_false_flag"],
                 "earliest_failure": item["earliest_failure"]}
                for item in claim["items"]
                if item["flags"]["answer_present"] is False],
        }
        for question_id, claim in sorted(
            ((question_id, claim) for question_id, claim in all_claims
             if not claim["final_passed"]),
            key=lambda pair: (pair[0], str(pair[1]["id"])))
    ]
    summary = {
        "mode": str(meta.get("mode") or "recorded"),
        "questions": len(question_reports),
        "passed_questions": sum(
            report["final_passed"] for report in question_reports),
        "claims": len(all_claims),
        "passed_claims": sum(
            claim["final_passed"] for _qid, claim in all_claims),
        "items": len(all_items),
        "claim_earliest_failures": dict(sorted(claim_failures.items())),
        "item_earliest_failures": dict(sorted(item_failures.items())),
        "flag_counts": flag_counts,
        "approximations": list(APPROXIMATIONS),
    }
    return {
        "audit": "item-lifecycle",
        "mode": str(meta.get("mode") or "recorded"),
        "answers_file": str(meta.get("answers_file") or ""),
        "report_file": str(meta.get("report_file") or ""),
        "gold_file": str(meta.get("gold_file") or ""),
        "db_file": str(meta.get("db_file") or ""),
        "trace_dir": str(meta.get("trace_dir") or ""),
        "summary": summary,
        "failing_claim_matrix": matrix,
        "questions": sorted(
            question_reports, key=lambda report: report["id"]),
    }


def dumps_stable(payload: dict) -> str:
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="recorded")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--report", required=True,
                        help="scored report holding final claim results")
    parser.add_argument(
        "--gold", default=str(HERE / "gold-chain-reviewed.json"))
    parser.add_argument("--db", default=str(ROOT / "ariadne.db"))
    parser.add_argument("--trace-dir")
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.mode == "oracle":
        raise NotImplementedError(
            "mode 'oracle' (gold-steered re-walk) is not implemented;"
            " only 'recorded' replays the saved live selections")
    if args.mode == "unsteered":
        raise NotImplementedError(
            "mode 'unsteered' (fresh selector re-run) is not implemented;"
            " only 'recorded' replays the saved live selections")
    answers_path = Path(args.answers)
    answers = json.loads(answers_path.read_text())
    gold = json.loads(Path(args.gold).read_text())
    report = json.loads(Path(args.report).read_text())
    trace_dir = Path(
        args.trace_dir
        or answers_path.with_name(answers_path.stem + "-traces"))
    only = (
        {int(value) for value in args.only.split(",")}
        if args.only else None)
    questions = {
        int(question["id"]): question
        for question in gold.get("questions", ())}
    final_questions = {
        int(question["id"]): question
        for question in report.get("questions", ())}
    db = LifecycleDb(args.db)
    reports = []
    for answer in sorted(answers, key=lambda entry: int(entry["id"])):
        question_id = int(answer.get("id") or 0)
        if only is not None and question_id not in only:
            continue
        question = questions.get(question_id)
        if question is None:
            raise ValueError(f"question {question_id} missing from gold")
        final_question = final_questions.get(question_id)
        if final_question is None:
            raise ValueError(f"question {question_id} missing from report")
        trace = load_question_trace(trace_dir, answer)
        reports.append(audit_question_lifecycle(
            answer, question, trace, final_question, db))
    payload = build_payload(reports, {
        "mode": args.mode,
        "answers_file": args.answers,
        "report_file": args.report,
        "gold_file": args.gold,
        "db_file": args.db,
        "trace_dir": str(trace_dir),
    })
    Path(args.out).write_text(dumps_stable(payload))
    summary = payload["summary"]
    for report_row in payload["questions"]:
        failing = [
            f"{claim['id']}={claim['earliest_failure']}"
            for claim in report_row["claims"]
            if not claim["final_passed"]]
        print(
            f"Q{report_row['id']}: "
            f"{sum(claim['final_passed'] for claim in report_row['claims'])}"
            f"/{len(report_row['claims'])} claims"
            + (f" | {', '.join(failing)}" if failing else ""))
    print(
        "RECONCILIATION (from --report): "
        f"{summary['passed_claims']}/{summary['claims']} claims; "
        f"{summary['passed_questions']}/{summary['questions']} questions")
    print(json.dumps(
        {"claim_earliest_failures": summary["claim_earliest_failures"],
         "item_earliest_failures": summary["item_earliest_failures"]},
        indent=1, sort_keys=True))
    print(f"item lifecycle audit -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
