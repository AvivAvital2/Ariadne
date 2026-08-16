"""Compact-path canary free gate: recorded plans, grader cover, no spend.

Runs the production compact path (``AnalysisMixin._ask_compact``) against
the real store for the paid-canary questions, with three scripted slots:

- plan: the RECORDED live obligation-plan reply for the same question
  (exact-semantic mapping — the recorded cascade plan and the compact
  plan share the C-line obligation grammar; the recorded replies name no
  code identifiers, so menu reach measured here is a lower bound).
- selector: a GRADER-authored reply — the gold-aware minimum cover over
  the production card menu. Gold never touches production code paths;
  this scripted reply exists only to prove a legal selection within
  budget expands to every reviewed claim. The paid run gives the model
  the same menu with no gold anywhere.
- formulation: a scripted marker, because the graded surface is the
  formulation PROMPT the model would see, never a generated answer.

A second worst-case pass selects every family on the menu and verifies
the declared budgets hold (bounded expansion, refuse-never-truncate).
Q67/Q147 — the two live-passing questions — replay the same way, and
their recorded answers' file:line citations must remain supported by
the compact surface. Zero provider calls, zero embedding calls, and the
store is only read.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

CANARY = (15, 23, 62, 107)
REPLAY = (67, 147)
PINNED_PHASES = ["scip-obligation-plan", "scip-route-family-select",
                 "completion"]
FORBIDDEN_PHASES = {
    "scip-family-select", "scip-route-select", "scip-symbol-select",
    "scip-component-select", "scip-exact-route-select",
    "scip-body-select"}
SELECTOR_TOKEN_BUDGET = 8000
FORMULATION_TOKEN_BUDGET = 20000
TOTAL_TOKEN_BUDGET = 30000
CANARY_COST_CEILING_USD = 1.20
PLAN_MAX_TOKENS = 384
ANSWER_MAX_TOKENS = 4096


RUNTIME_MODULES = (
    "ariadne_mcp/service_analysis.py",
    "library/route_families.py",
    "library/structural_assembly.py",
    "library/clews.py",
    "library/chain_bundle.py",
    "library/chain_story.py",
    "library/chain_answer.py",
    "library/chain_menu.py",
    "library/selection_policy.py",
    "library/source_chunks.py",
)


def runtime_manifest() -> dict:
    """Git blob ids of the modules actually imported by this run.

    The gate must run a committed snapshot, never the dirty checkout:
    callers pass --head-manifest (from ``git ls-tree -r HEAD``) and any
    drift hard-fails the gate before a single question runs.
    """
    import hashlib

    manifest = {}
    for relative in RUNTIME_MODULES:
        content = (ROOT / relative).read_bytes()
        header = f"blob {len(content)}\0".encode()
        manifest[relative] = hashlib.sha1(header + content).hexdigest()
    return manifest


def enforce_snapshot(head_manifest_path: str) -> dict:
    observed = runtime_manifest()
    if not head_manifest_path:
        return {"verified_against_head": False, "modules": observed}
    expected = json.loads(Path(head_manifest_path).read_text())
    drifted = sorted(
        relative for relative, blob in observed.items()
        if expected.get(relative) not in (None, blob))
    missing = sorted(
        relative for relative in observed if relative not in expected)
    if drifted:
        raise SystemExit(
            "runtime drift vs committed HEAD: " + ", ".join(drifted))
    return {"verified_against_head": True, "modules": observed,
            "not_in_head": missing}


LIFECYCLE_STAGES = (
    "clew_in_store", "in_vector_pool", "past_filter", "shortlisted",
    "on_menu", "body_expanded", "in_ledger")


def anchor_lifecycle(identity: dict, compact: dict, conn,
                     source: str) -> dict:
    """Evaluator-side join: where along the compact boundary did this
    reviewed anchor fall out? Runtime trace stays gold-blind; the gold
    identity is joined here, after the run."""
    qname = identity["qualified_name"]
    trace = compact.get("trace") or {}
    clew_ids = {row[0] for row in conn.execute(
        "SELECT id FROM clews WHERE source_name = ? AND route LIKE ?",
        (source, f'%"{qname}"%'))}
    pool = trace.get("pool_order") or []
    pool_positions = [index for index, clew_id in enumerate(pool)
                      if clew_id in clew_ids]
    rejected = {entry["id"]: entry["reason"]
                for entry in trace.get("filter_rejected") or ()}
    rejection = next(
        (rejected[clew_id] for clew_id in clew_ids
         if clew_id in rejected), None)
    shortlists = {}
    lexical_ranks = {}
    for obligation_id, entry in trace.items():
        if not isinstance(entry, dict) or "shortlist" not in entry:
            continue
        shortlists[obligation_id] = any(
            clew_id in clew_ids for clew_id in entry["shortlist"])
        positions = [index for index, clew_id in enumerate(
            entry.get("lexical_order") or ()) if clew_id in clew_ids]
        if positions:
            lexical_ranks[obligation_id] = min(positions)
    dropped_reasons = sorted({
        drop["reason"]
        for entry in trace.values() if isinstance(entry, dict)
        for drop in entry.get("dropped") or ()
        if qname in (drop.get("entry"), drop.get("terminal"))})
    on_menu = any(
        card_covers(card, identity)
        for card in compact.get("cards") or ())
    bodies = trace.get("required_body_extents") or ()
    body_expanded = any(
        name == qname and file == identity["file"]
        and not (line_end < identity["line_start"]
                 or line_start > identity["line_end"])
        for name, file, line_start, line_end in bodies)
    chunks = trace.get("chunk_extents") or ()
    in_ledger = any(
        file == identity["file"]
        and not (line_end < identity["line_start"]
                 or line_start > identity["line_end"])
        for file, line_start, line_end in chunks)
    stages = {
        "clew_in_store": bool(clew_ids),
        "in_vector_pool": bool(pool_positions),
        "past_filter": bool(pool_positions) and rejection is None,
        "shortlisted": any(shortlists.values()),
        "on_menu": on_menu,
        "body_expanded": body_expanded,
        "in_ledger": in_ledger,
    }
    first_loss = next(
        (stage for stage in LIFECYCLE_STAGES if not stages[stage]),
        None)
    return {
        "anchor": identity["anchor"], "qualified_name": qname,
        "stages": stages, "first_loss": first_loss,
        "vector_pool_rank": (
            min(pool_positions) if pool_positions else None),
        "filter_rejection": rejection,
        "lexical_rank_by_obligation": lexical_ranks,
        "dropped_reasons": dropped_reasons,
    }


def tokens(text: str) -> int:
    return len(text or "") // 4


def normalized(text: str) -> str:
    return " ".join(str(text or "").split())


def load_trace(trace_dir: Path, question_id: int) -> dict:
    return json.loads(gzip.decompress(
        (trace_dir / f"q{question_id}.json.gz").read_bytes()))


def recorded_reply(trace: dict, phase: str) -> str:
    for completion in trace.get("llm_completions", ()):
        if str(completion.get("phase")) == phase:
            return str(completion.get("response") or "")
    raise SystemExit(f"no recorded {phase} reply in trace")


def anchor_identities(claim: dict) -> list:
    """Reviewer-selected exact identity per anchor of a reviewed claim."""
    selected = (claim.get("review") or {}).get(
        "selected_candidate_by_anchor", {})
    identities = []
    for anchor in claim.get("anchors", ()):
        anchor_id = str(anchor.get("anchor"))
        candidates = anchor.get("candidates") or ()
        canonical = selected.get(anchor_id)
        chosen = next(
            (candidate for candidate in candidates
             if candidate.get("canonical_id") == canonical), None)
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is None:
            continue
        identities.append({
            "anchor": anchor_id,
            "qualified_name": str(chosen["qualified_name"]),
            "file": str(chosen["file"]),
            "line_start": int(chosen["line_start"]),
            "line_end": int(chosen["line_end"])})
    return identities


def card_covers(card: dict, identity: dict) -> bool:
    for node in card.get("nodes", ()):
        name, file, line_start, line_end = node
        if (name == identity["qualified_name"]
                and file == identity["file"]
                and not (line_end < identity["line_start"]
                         or line_start > identity["line_end"])):
            return True
    return False


def minimum_cover(cards: list, identities: list):
    """Greedy gold-aware cover: fewest cards that reach every identity."""
    uncovered = set(range(len(identities)))
    chosen = []
    while uncovered:
        best, best_hits = None, set()
        for card in cards:
            hits = {index for index in uncovered
                    if card_covers(card, identities[index])}
            if len(hits) > len(best_hits) or (
                    hits and len(hits) == len(best_hits)
                    and best is not None
                    and card["bodies"] < best["bodies"]):
                best, best_hits = card, hits
        if best is None or not best_hits:
            break
        chosen.append(best)
        uncovered -= best_hits
    return chosen, sorted(
        identities[index]["anchor"] for index in uncovered)


def selector_reply(cards: list, chosen: list) -> str:
    """A legal reply: cover cards under their own obligations, plus the
    cheapest family for every obligation the cover left empty."""
    by_obligation: dict = {}
    for card in chosen:
        by_obligation.setdefault(
            card["obligation_id"], []).append(card["card_id"])
    for card in sorted(cards, key=lambda card: card["bodies"]):
        by_obligation.setdefault(card["obligation_id"], [card["card_id"]])
    return "\n".join(
        f"{obligation}: " + " ".join(dict.fromkeys(card_ids))
        for obligation, card_ids in sorted(by_obligation.items()))


class GateChat:
    """Scripted provider slots; records phases and full prompts."""

    def __init__(self, plan_reply: str, select_fn):
        self.plan_reply = plan_reply
        self.select_fn = select_fn
        self.phases: list = []
        self.prompts: dict = {}

    async def __call__(self, *, messages, **kwargs):
        phase = str(kwargs.get("phase", "completion"))
        self.phases.append(phase)
        self.prompts[phase] = str(messages[-1]["content"])
        sink = kwargs.get("usage_sink")
        if sink is not None:
            sink.append({"finish_reason": "stop", "output_tokens": 8,
                         "max_tokens": kwargs.get("max_tokens")})
        if phase == "scip-obligation-plan":
            return self.plan_reply
        if phase == "scip-route-family-select":
            return self.select_fn(self.prompts[phase])
        return "FREE-GATE-SCRIPTED-ANSWER"


def fragment_hits(witness: dict, surface: str, normalized_surface: str):
    missing = []
    for fragment in witness.get("contains", ()):
        if fragment in surface:
            continue
        if normalized(fragment) in normalized_surface:
            continue
        missing.append(fragment)
    return missing


_RANGE = re.compile(r"([\w./-]+\.\w+):(\d+)(?:-(\d+))?")


def surface_ranges(surface: str) -> list:
    ranges = []
    for match in _RANGE.finditer(surface or ""):
        file, start, end = match.group(1), int(match.group(2)), match.group(3)
        ranges.append((file, start, int(end) if end else start))
    return ranges


def citation_supported(file: str, line: int, ranges: list) -> bool:
    return any(
        (known.endswith(file) or file.endswith(known))
        and start - 3 <= line <= end + 3
        for known, start, end in ranges)


def grade_question(gold_question: dict, compact: dict, chat: GateChat):
    surface = chat.prompts.get("completion", "")
    normalized_surface = normalized(surface)
    cards = compact.get("cards", ())
    claims = []
    for claim in gold_question.get("claims", ()):
        identities = anchor_identities(claim)
        anchors_on_menu = {
            identity["anchor"]: any(
                card_covers(card, identity) for card in cards)
            for identity in identities}
        witnesses = []
        for witness in claim.get("witnesses", ()):
            missing = fragment_hits(witness, surface, normalized_surface)
            witnesses.append({
                "id": witness.get("id"),
                "file": witness.get("file"),
                "lines": [witness.get("line_start"),
                          witness.get("line_end")],
                "missing_fragments": missing,
                "materialized": not missing})
        menu_complete = bool(identities) and all(anchors_on_menu.values())
        fragments_ok = bool(witnesses) and all(
            entry["materialized"] for entry in witnesses)
        claims.append({
            "id": claim.get("id"),
            "anchors_on_menu": anchors_on_menu,
            "model_menu_complete": menu_complete,
            "witnesses": witnesses,
            "fragments_materialized": fragments_ok,
            "passes_free_gate": menu_complete and fragments_ok})
    return claims, surface


EMBEDDING_USD_PER_MTOK = 0.13  # text-embedding-3-large


def projected_cost_usd(chat: GateChat, price: dict, obligations: int,
                       question: str, embedding_calls: int) -> dict:
    input_tokens = sum(tokens(prompt) for prompt in chat.prompts.values())
    output_ceiling = (PLAN_MAX_TOKENS + max(64, 24 * obligations)
                      + ANSWER_MAX_TOKENS)
    embedding_tokens = tokens(question) * embedding_calls
    cost = (input_tokens * price["input_per_mtok"]
            + output_ceiling * price["output_per_mtok"]
            + embedding_tokens * EMBEDDING_USD_PER_MTOK) / 1_000_000
    return {"input_tokens": input_tokens,
            "output_token_ceiling": output_ceiling,
            "embedding_calls": embedding_calls,
            "embedding_tokens": embedding_tokens,
            "projected_usd_ceiling": round(cost, 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=str(
        HERE / "live22-diagnostic-answers-traces"))
    parser.add_argument("--gold", default=str(
        HERE / "gold-chain-reviewed-compact.json"))
    parser.add_argument("--price-config", default=str(
        HERE / "price-config-v1.json"))
    parser.add_argument("--model", default="claude-opus-4-8")
    parser.add_argument("--question-vectors", default=str(
        HERE / "question-embeddings.npz"))
    parser.add_argument("--cohorts", default=str(
        HERE / "anchor-cohorts-baseline.json"), help=(
        "frozen per-anchor first-loss cohorts; progression is reported "
        "against these and anchors are never re-bucketed"))
    parser.add_argument("--deep", action="store_true", help=(
        "record the full compact boundary trace and join reviewed "
        "anchors into per-anchor lifecycle rows"))
    parser.add_argument("--head-manifest", default="", help=(
        "JSON {path: git blob sha} from the committed HEAD; any runtime "
        "module drift aborts the gate"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    snapshot = enforce_snapshot(args.head_manifest)

    import numpy as np

    from ariadne_mcp.service import AriadneService

    service = AriadneService.get()
    question_vectors = np.load(args.question_vectors)

    class _CachedEmbedding:
        """The cached live question vector; embed() spends nothing here.

        The paid run embeds the question once instead — that single call
        is counted in the projected cost below. Recorded texts prove the
        ask path only ever embeds one unique string (the memo on the
        real EmbeddingService turns repeats into one provider call)."""

        def __init__(self, vector):
            self.vector = vector
            self.texts: list = []

        async def embed(self, text):
            self.texts.append(text)
            return self.vector
    gold = json.loads(Path(args.gold).read_text())
    price = json.loads(Path(args.price_config).read_text())[
        "models"][args.model]
    by_id = {int(question["id"]): question
             for question in gold["questions"]}
    trace_dir = Path(args.trace_dir)

    report = {"schema": "compact-free-gate-v2", "questions": [],
              "runtime": snapshot, "gate": {}}
    failures = []
    canary_cost = 0.0
    deep_compacts: dict = {}
    lifecycles_by_question: dict = {}

    for question_id in (*CANARY, *REPLAY):
        gold_question = by_id[question_id]
        trace = load_trace(trace_dir, question_id)
        question = str(trace["question"])
        source = str(trace["source"])
        fake_embedding = _CachedEmbedding(
            np.asarray(question_vectors[f"q{question_id}"],
                       dtype=np.float32))
        service._embedding_service = fake_embedding
        plan_reply = recorded_reply(trace, "scip-obligation-plan")
        identities = [identity
                      for claim in gold_question.get("claims", ())
                      for identity in anchor_identities(claim)]

        cover_note: dict = {}

        def select_fn(prompt, _identities=identities,
                      _note=cover_note, _diag_holder=[]):
            cards = _note["cards"]
            chosen, unmatched = minimum_cover(cards, _identities)
            _note["cover_card_ids"] = [card["card_id"] for card in chosen]
            _note["uncovered_anchors"] = unmatched
            return selector_reply(cards, chosen)

        diagnostics: dict = {}
        chat = GateChat(plan_reply, select_fn)

        # The selector callback needs the recorded cards; they are
        # written into diagnostics before the selector call fires. In
        # public-ask mode the compact diagnostics dict is discovered
        # through the interceptor's holder.
        compact_holder: dict = {}

        class _CardsProxy:
            """Live view of the CURRENT run's diagnostics: the selector
            callback fires mid-run, so the holder is pointed at each
            run's diagnostics dict before that run starts."""

            def _compact(self):
                live = compact_holder.get("diagnostics") or {}
                return live.get("compact") or {}

            def __getitem__(self, index):
                return self._compact()["cards"][index]

            def __iter__(self):
                return iter(self._compact().get("cards", ()))

            def __len__(self):
                return len(self._compact().get("cards", ()))

        cover_note["cards"] = _CardsProxy()

        # PUBLIC PATH: the question runs through service.ask() with the
        # recorded production flag; provider calls are intercepted at
        # llm.chat_complete, so any hidden cascade call would surface
        # as an unexpected phase.
        import llm as llm_module

        import ariadne_mcp.service_analysis as service_analysis_module

        service.config._config["ask_pipeline"] = "compact"
        original_chat_complete = llm_module.chat_complete
        original_compact = service_analysis_module._ask_compact

        async def intercepted_chat(messages, **kwargs):
            return await chat(messages=messages, **kwargs)

        async def traced_compact(self, *call_args, **call_kwargs):
            call_kwargs.setdefault("deep_trace", bool(args.deep))
            if isinstance(call_kwargs.get("diagnostics"), dict):
                compact_holder["diagnostics"] = call_kwargs[
                    "diagnostics"]
            result = await original_compact(
                self, *call_args, **call_kwargs)
            compact_holder["compact"] = (
                call_kwargs.get("diagnostics") or {}).get("compact")
            return result

        llm_module.chat_complete = intercepted_chat
        type(service)._ask_compact = traced_compact
        try:
            response = asyncio.run(service.ask(
                question, source=source))
        finally:
            llm_module.chat_complete = original_chat_complete
            type(service)._ask_compact = original_compact
            service.config._config.pop("ask_pipeline", None)
        compact = (compact_holder.get("compact")
                   or response.graph_diagnostics.get("compact") or {})

        # Equivalence: the direct compact invocation must expose the
        # same candidate surface the public path exposed.
        direct_diagnostics: dict = {}
        direct_chat = GateChat(plan_reply, select_fn)
        compact_holder["diagnostics"] = direct_diagnostics
        asyncio.run(original_compact(
            service, question, source=source, notes=(),
            diagnostics=direct_diagnostics, ask_chat=direct_chat,
            trace=lambda *a, **k: None, phase_timings={},
            deep_trace=bool(args.deep),
            question_vector=np.asarray(
                question_vectors[f"q{question_id}"], dtype=np.float32)))
        direct_compact = direct_diagnostics.get("compact", {})
        equivalent_surface = (
            direct_compact.get("cards") == compact.get("cards")
            and direct_chat.prompts.get("scip-route-family-select")
            == chat.prompts.get("scip-route-family-select"))

        claims, surface = grade_question(gold_question, compact, chat)
        selector_tokens = tokens(
            chat.prompts.get("scip-route-family-select", ""))
        formulation_tokens = tokens(surface)
        total_tokens = sum(
            tokens(prompt) for prompt in chat.prompts.values())
        obligations = len(compact.get("obligations", ()))
        cost = projected_cost_usd(
            chat, price, obligations, question,
            int(compact.get("embedding_calls") or 0))

        # Worst case: select every family the menu offers.
        worst_chat = GateChat(
            plan_reply,
            lambda prompt: selector_reply(
                list(diag_worst["compact"]["cards"]),
                list(diag_worst["compact"]["cards"])))
        diag_worst: dict = {}
        asyncio.run(service._ask_compact(
            question, source=source, notes=(), diagnostics=diag_worst,
            ask_chat=worst_chat,
            trace=lambda *args, **kwargs: None, phase_timings={}))
        worst = diag_worst.get("compact", {})
        worst_formulation = tokens(worst_chat.prompts.get("completion", ""))

        record = {
            "id": question_id,
            "role": "canary" if question_id in CANARY else "replay",
            "phases": chat.phases,
            "status": compact.get("status"),
            "obligations": obligations,
            "semantic_seed_count": compact.get("semantic_seed_count"),
            "families_shown": compact.get("cards_total"),
            "families_overflow": compact.get("overflow_by_obligation"),
            "cover_card_ids": cover_note.get("cover_card_ids"),
            "uncovered_anchors": cover_note.get("uncovered_anchors"),
            "unresolved_obligations": compact.get(
                "unresolved_obligations"),
            "route_nodes": compact.get("route_nodes"),
            "bodies": compact.get("required_bodies"),
            "chunks": compact.get("chunks"),
            "expansion_gaps": compact.get("expansion_gaps"),
            "selector_prompt_tokens": selector_tokens,
            "formulation_prompt_tokens": formulation_tokens,
            "total_prompt_tokens": total_tokens,
            "provider_calls": len(chat.phases),
            "cost": cost,
            "cards": compact.get("cards"),
            "worst_case": {
                "status": worst.get("status"),
                "formulation_prompt_tokens": worst_formulation,
                "expansion_gaps": worst.get("expansion_gaps"),
                "bodies": worst.get("required_bodies")},
            "claims": claims,
        }

        if args.deep:
            with service.library._conn_provider.acquire() as conn:
                rows = [anchor_lifecycle(identity, compact, conn, source)
                        for identity in identities]
            record["anchor_lifecycles"] = rows
            deep_compacts[question_id] = compact
            lifecycles_by_question[question_id] = rows

        checks = {
            "compact_invoked": compact.get("pipeline") == "compact",
            "public_ask_used": True,
            "single_question_embedding":
                len(set(fake_embedding.texts)) <= 1,
            "direct_public_equivalent": equivalent_surface,
            "phases_pinned": chat.phases == PINNED_PHASES,
            "no_forbidden_phase": not FORBIDDEN_PHASES.intersection(
                chat.phases),
            "selector_within_budget":
                selector_tokens <= SELECTOR_TOKEN_BUDGET,
            "formulation_within_budget":
                formulation_tokens <= FORMULATION_TOKEN_BUDGET,
            "total_within_budget": total_tokens <= TOTAL_TOKEN_BUDGET,
            "worst_case_bounded": (
                worst.get("status") in ("complete",
                                        "formulation-budget-exceeded")
                and (worst_formulation <= FORMULATION_TOKEN_BUDGET
                     or worst.get("status")
                     == "formulation-budget-exceeded")),
            "all_claims_pass": all(
                claim["passes_free_gate"] for claim in claims),
        }
        if question_id in REPLAY:
            recorded_answer = str(trace.get("service_answer") or "")
            ranges = surface_ranges(surface)
            cited = [(file, int(line)) for file, line in re.findall(
                r"([\w./-]+\.(?:scala|py|java|go|ts|js)):(\d+)",
                recorded_answer)]
            unsupported = [
                f"{file}:{line}" for file, line in cited
                if not citation_supported(file, line, ranges)]
            record["recorded_citations"] = len(cited)
            record["recorded_citations_unsupported"] = unsupported
            checks["recorded_answer_supported"] = not unsupported
        record["checks"] = checks
        report["questions"].append(record)
        if question_id in CANARY:
            canary_cost += cost["projected_usd_ceiling"]
        for name, passed in checks.items():
            if not passed:
                failures.append(f"q{question_id}:{name}")
        print(f"q{question_id}: status={record['status']} "
              f"families={record['families_shown']} "
              f"claims_pass={sum(c['passes_free_gate'] for c in claims)}"
              f"/{len(claims)} sel={selector_tokens}t "
              f"form={formulation_tokens}t "
              f"cost<=${cost['projected_usd_ceiling']}")

    if args.deep:
        def clews_for(conn, needle):
            return {row[0] for row in conn.execute(
                "SELECT id FROM clews WHERE source_name = 'databricks' "
                "AND route LIKE ?", (f"%{needle}%",))}

        with service.library._conn_provider.acquire() as conn:
            trace67 = (deep_compacts.get(67) or {}).get("trace") or {}
            pool67 = set(trace67.get("pool_order") or ())
            shortlisted67 = {
                clew_id for entry in trace67.values()
                if isinstance(entry, dict)
                for clew_id in entry.get("shortlist") or ()}
            q67_routes = {}
            for needle in ("DeltaSink.addBatchWithStatusImpl",
                           'DeltaSink.addBatch"',
                           "DeltaSink.PendingTxn.commit"):
                ids = clews_for(conn, needle)
                q67_routes[needle] = {
                    "clews": len(ids),
                    "in_pool": len(ids & pool67),
                    "shortlisted": len(ids & shortlisted67)}
            q67_repro = bool(q67_routes) and all(
                row["clews"] and row["in_pool"]
                and not row["shortlisted"]
                for row in q67_routes.values())

            q62_rows = lifecycles_by_question.get(62, [])
            q62_absent = [row["qualified_name"] for row in q62_rows
                          if not row["stages"]["clew_in_store"]]
            q62_in_pool = [row["qualified_name"] for row in q62_rows
                           if row["stages"]["in_vector_pool"]]
            q62_below_pool = [
                row["qualified_name"] for row in q62_rows
                if row["stages"]["clew_in_store"]
                and not row["stages"]["in_vector_pool"]]
            q62_repro = bool(
                q62_absent and q62_in_pool and q62_below_pool)

            # The originally observed q15 loss — DeltaFileFormatWriter
            # .write dropped among writeFiles' 32 forward callees — was
            # repaired by the committed fork + preference-retention
            # slices; the CURRENT q15 blockers are a missing clew for
            # the partition-columns option and writer-owner families
            # competing under the diversity cap.
            rows15 = lifecycles_by_question.get(15, [])
            partition_absent = any(
                not row["stages"]["clew_in_store"]
                for row in rows15
                if row["qualified_name"].endswith(
                    "WRITE_PARTITION_COLUMNS"))
            writer_diversity_drops = any(
                "diversity-cap:3-per-module-owner"
                in row["dropped_reasons"]
                for row in rows15
                if row["qualified_name"].endswith(
                    ("TransactionalWrite.writeFiles",
                     "DeltaFileFormatWriter.write")))
            q15_repro = partition_absent and writer_diversity_drops

        report["baseline_reproduction"] = {
            "q67_pool_but_not_shortlist": {
                "passed": q67_repro, "routes": q67_routes},
            "q62_mixture": {
                "passed": q62_repro, "absent_clew": q62_absent,
                "in_pool": q62_in_pool, "below_pool": q62_below_pool},
            "q15_current_blockers": {
                "passed": q15_repro,
                "partition_option_has_no_clew": partition_absent,
                "writer_families_hit_diversity_cap":
                    writer_diversity_drops,
                "note": "original writeFiles-callee drop resolved by "
                        "commits 60bff9e and 6402f49"},
        }
        print("baseline reproduction:", {
            key: value["passed"]
            for key, value in report["baseline_reproduction"].items()})

    if args.deep:
        frozen = json.loads(Path(args.cohorts).read_text())
        stage_position = {stage: index for index, stage
                          in enumerate(LIFECYCLE_STAGES)}
        stage_position["through"] = len(LIFECYCLE_STAGES)

        def anchor_key(question_id, row):
            return (int(question_id), str(row["anchor"]),
                    str(row["qualified_name"]))

        current = {}
        for question_id, rows in lifecycles_by_question.items():
            for row in rows:
                key = anchor_key(question_id, row)
                stage = row["first_loss"] or "through"
                current.setdefault(key, []).append(stage)
        frozen_keys: dict = {}
        for row in frozen["anchors"]:
            key = (int(row["question"]), str(row["anchor"]),
                   str(row["qualified_name"]))
            frozen_keys.setdefault(key, []).append(
                row["original_first_loss"])
        if {k: len(v) for k, v in frozen_keys.items()} != {
                k: len(v) for k, v in current.items()}:
            raise SystemExit(
                "cohort keys drifted from the frozen baseline — "
                "anchors must never be re-bucketed")
        progression: dict = {}
        for key, originals in frozen_keys.items():
            for original, now in zip(sorted(originals),
                                     sorted(current[key])):
                bucket = progression.setdefault(original, {
                    "advanced": 0, "blocked": 0, "regressed": 0,
                    "advanced_anchors": [], "regressed_anchors": []})
                delta = (stage_position[now]
                         - stage_position[original])
                if delta > 0:
                    bucket["advanced"] += 1
                    bucket["advanced_anchors"].append(
                        f"q{key[0]}:{key[1]}->{now}")
                elif delta < 0:
                    bucket["regressed"] += 1
                    bucket["regressed_anchors"].append(
                        f"q{key[0]}:{key[1]}->{now}")
                else:
                    bucket["blocked"] += 1
        report["cohort_progression"] = {
            "frozen_from": frozen.get("frozen_from_commit"),
            "cohorts": progression}
        print("cohort progression:", {
            cohort: {k: v for k, v in bucket.items()
                     if isinstance(v, int)}
            for cohort, bucket in sorted(progression.items())})

    claim_records = [
        claim for record in report["questions"]
        if record["role"] == "canary" for claim in record["claims"]]
    report["gate"] = {
        "canary_claims_pass": sum(
            claim["passes_free_gate"] for claim in claim_records),
        "canary_claims_total": len(claim_records),
        "canary_projected_usd_ceiling": round(canary_cost, 4),
        "cost_within_ceiling": canary_cost <= CANARY_COST_CEILING_USD,
        "failures": failures,
        "passed": not failures
                  and canary_cost <= CANARY_COST_CEILING_USD,
    }
    Path(args.out).write_text(
        json.dumps(report, indent=1, sort_keys=True))
    print(f"gate passed={report['gate']['passed']} "
          f"claims={report['gate']['canary_claims_pass']}"
          f"/{report['gate']['canary_claims_total']} "
          f"projected<=${report['gate']['canary_projected_usd_ceiling']}")
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
