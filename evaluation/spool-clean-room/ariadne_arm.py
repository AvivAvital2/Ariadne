#!/usr/bin/env python3
"""The Ariadne arm: answer the same questions through ``ask``, scored the same way.

Until now the two arms were never comparable. The bare arm is graded on
mechanical grounding — a quote that sits at the line it cites — while Ariadne's
answering path cited document titles and nothing else, so the grounding gate
would have returned 0/21 for every answer regardless of quality. Not a
measurement of Ariadne, a measurement of its output format.

With the code tier wired into ``ask`` (``VERIFIED CODE LOCATIONS``, resolved from
``scip_symbols``), an Ariadne answer can carry ``file:line`` and can therefore be
put through the identical gate: same de-breadcrumbed questions, same fixed
``chain_requirements.json``, same ``score_chain.py``.

What is deliberately NOT equalized: the bare arm reads the corpus and Ariadne
does not. Ariadne answers from documents plus the SCIP index. That is the
comparison — an index against a reader — not a handicap to correct for.

Needs BOTH keys: Anthropic synthesizes the answer, OpenAI embeds the query for
retrieval (Anthropic has no embeddings API). Runs 4-wide by default.

    ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \\
        python evaluation/spool-clean-room/ariadne_arm.py
    python evaluation/spool-clean-room/score_chain.py \\
        --answers evaluation/spool-clean-room/answers-ariadne-arm.json
"""
from __future__ import annotations
import os

import argparse
import asyncio
import hashlib
import json
def _response_diagnostics(response) -> dict:
    """Serializable final Ariadne state required for offline failure isolation."""
    return {
        "confidence_reasons": list(
            getattr(response, "confidence_reasons", None) or []),
        "chain_confidence": getattr(response, "chain_confidence", "low"),
        "formulation_confidence": getattr(
            response, "formulation_confidence", "low"),
        "scope_confidence": getattr(response, "scope_confidence", "low"),
        "route_candidates": dict(
            getattr(response, "route_candidates", None) or {}),
        "selected_route_ids": list(
            getattr(response, "selected_route_ids", None) or []),
        "selected_section_ids": list(
            getattr(response, "selected_section_ids", None) or []),
        "selected_symbols": list(
            getattr(response, "selected_symbols", None) or []),
        "selected_body_symbols": list(
            getattr(response, "selected_body_symbols", None) or []),
        "hydrated_symbols": list(
            getattr(response, "hydrated_symbols", None) or []),
        "hydrated_sections": list(
            getattr(response, "hydrated_sections", None) or []),
        "excluded_question_symbols": list(
            getattr(response, "excluded_question_symbols", None) or []),
        "cited_route_ids": list(
            getattr(response, "cited_route_ids", None) or []),
        "route_selection_status": getattr(
            response, "route_selection_status", "not-applicable"),
        "selection_complete": bool(
            getattr(response, "selection_complete", False)),
        "selection_reasons": list(
            getattr(response, "selection_reasons", None) or []),
        "route_candidate_occurrences": dict(
            getattr(response, "route_candidate_occurrences", None) or {}),
        "evidence_gaps": list(
            getattr(response, "evidence_gaps", None) or []),
        "chain_summary": dict(
            getattr(response, "chain_summary", None) or {}),
        "claims": list(getattr(response, "claims", None) or []),
        "chain_complete": bool(
            getattr(response, "chain_complete", False)),
        "completeness_reasons": list(
            getattr(response, "completeness_reasons", None) or []),
        "formulation_complete": bool(
            getattr(response, "formulation_complete", False)),
        "formulation_reasons": list(
            getattr(response, "formulation_reasons", None) or []),
        "scope_complete": bool(
            getattr(response, "scope_complete", False)),
        "scope_reasons": list(
            getattr(response, "scope_reasons", None) or []),
        "transition_claims": list(
            getattr(response, "transition_claims", None) or []),
        "route_scope_total": int(
            getattr(response, "route_scope_total", 0)),
        "route_scope_retained": int(
            getattr(response, "route_scope_retained", 0)),
        "section_candidates": int(
            getattr(response, "section_candidates", 0)),
        "graph_diagnostics": dict(
            getattr(response, "graph_diagnostics", None) or {}),
        "chain_citations": list(
            getattr(response, "chain_citations", None) or []),
        "unsupported_locations": list(
            getattr(response, "unsupported_locations", None) or []),
        "phase_timings": dict(
            getattr(response, "phase_timings", None) or {}),
        "llm_calls": int(getattr(response, "llm_calls", 0)),
    }
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

QUESTIONS = HERE / 'questions_debcrumb_ask.json'
REQS = HERE / 'chain_requirements.json'
def _resolve_corpus_file(corpus: Path, indexed_path: str) -> tuple[Path, str] | None:
    """Map a SCIP-relative path into the multi-repository clean corpus.

    The Databricks SCIP source combines Spark and Delta, so its paths omit the
    outer checkout name (``sql/...`` rather than ``spark/sql/...``). Resolve
    only a unique exact repo-prefix candidate; ambiguity fails closed.
    """
    rel = indexed_path.lstrip('/')
    direct = corpus / rel
    candidates = [(direct, rel)] if direct.is_file() else []
    for repo in ('spark', 'delta', 'databricks-sdk-py'):
        candidate = corpus / repo / rel
        if candidate.is_file():
            candidates.append((candidate, f'{repo}/{rel}'))
    unique = {(path.resolve(), shown) for path, shown in candidates}
    if len(unique) != 1:
        return None
    path, shown = unique.pop()
    return path, shown
def _materialize_citations(citations: list[dict], corpus: Path, *,
                           file_cache: dict | None = None) -> tuple[str, list[str], dict]:
    """Render cited coordinates while reading and hashing each source file once per run."""
    blocks, files, hashes = [], [], {}
    seen = set()
    snapshots = file_cache if file_cache is not None else {}
    for citation in citations:
        locations = [(str(citation.get('file') or ''),
                      int(citation.get('line') or 0))]
        call_site = str(citation.get('call_site') or '')
        if ':' in call_site:
            call_file, call_line = call_site.rsplit(':', 1)
            if call_line.isdigit():
                locations.append((call_file, int(call_line)))
        for indexed, line in locations:
            resolved = _resolve_corpus_file(corpus, indexed)
            if resolved is None:
                continue
            path, rel = resolved
            if line < 1 or (rel, line) in seen:
                continue
            seen.add((rel, line))
            key = path.resolve()
            snapshot = snapshots.get(key)
            if snapshot is None:
                raw = path.read_bytes()
                snapshot = (
                    raw.decode(errors='ignore').splitlines(),
                    hashlib.sha256(raw).hexdigest()[:16],
                )
                snapshots[key] = snapshot
            body, digest = snapshot
            if line > len(body):
                continue
            excerpt = body[line - 1:min(
                len(body), max(line + 5, int(citation.get("line_end") or line)))]
            claimed = f'/corpus/{rel}'
            blocks.append(f'{claimed}:{line}\n```\n' + '\n'.join(excerpt) + '\n```')
            files.append(claimed)
            hashes[claimed] = digest
    return '\n\n'.join(blocks), sorted(set(files)), hashes
async def _one(
        service, qid: int, question: str, source: str, corpus: Path,
        file_cache: dict | None = None,
        trace_dir: Path | None = None) -> dict:
    """Ask once and persist a replay-complete trace without changing the request."""
    from llm import capture_completion_trace, capture_completion_usage

    t0 = time.time()
    with capture_completion_usage() as usage_rows:
        with capture_completion_trace(
                enabled=trace_dir is not None) as completion_rows:
            import inspect

            parameters = inspect.signature(service.ask).parameters.values()
            ask_kwargs = {"source": source}
            if ("trace_id" in inspect.signature(service.ask).parameters
                    or any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                           for parameter in parameters)):
                ask_kwargs["trace_id"] = qid
            resp = await service.ask(question, **ask_kwargs)
    chain_files = list(getattr(resp, "chain_files", None) or [])
    citations = list(getattr(resp, "citations", None) or [])
    evidence, files, hashes = _materialize_citations(
        citations, corpus, file_cache=file_cache)
    diagnostics = _response_diagnostics(resp)
    row = {
        "id": qid,
        "question": question,
        "benchmark_source": source,
        "benchmark_corpus": str(corpus.resolve()),
        "answer": ((resp.answer or "")
                   + (f"\n\nVERBATIM SCIP EVIDENCE\n{evidence}"
                      if evidence else "")),
        "sources": list(getattr(resp, "sources", None) or []),
        "confidence": getattr(resp, "confidence", None),
        "files_read": files,
        "file_hashes": hashes,
        "chain_files": chain_files,
        "citations": citations,
        "tool_calls": [],
        "elapsed_s": round(time.time() - t0, 1),
        **diagnostics,
        **_usage_summary(usage_rows),
    }
    if trace_dir is not None:
        payload = {
            "schema": TRACE_SCHEMA,
            "id": qid,
            "question": question,
            "source": source,
            "corpus": str(corpus.resolve()),
            "service_answer": resp.answer or "",
            "benchmark_answer": row["answer"],
            "response_diagnostics": diagnostics,
            "llm_completions": completion_rows,
            "usage_rows": usage_rows,
            "materialized_evidence": {
                "text": evidence,
                "files_read": files,
                "file_hashes": hashes,
            },
        }
        row["diagnostic_trace"] = _write_diagnostic_trace(
            trace_dir, qid, payload)
    return row
def _write_checkpoint(path: Path, rows) -> None:
    """Atomically preserve every completed answer in deterministic order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    ordered = sorted(rows, key=lambda row: int(row["id"]))
    temporary.write_text(json.dumps(ordered, indent=2) + "\n")
    temporary.replace(path)
def _load_resumable_answers(
        path: Path, questions, *, source: str, corpus: Path,
        trace_dir: Path | None = None) -> dict[int, dict]:
    """Reuse only matching rows whose requested diagnostic trace is intact."""
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    if not isinstance(rows, list):
        return {}
    expected = {
        int(question["id"]):
            question.get("after") or question.get("question") or ""
        for question in questions}
    corpus_name = str(corpus.resolve())
    resumed = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            question_id = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        compatible = (
            question_id in expected
            and row.get("question") == expected[question_id]
            and row.get("benchmark_source") == source
            and row.get("benchmark_corpus") == corpus_name)
        if not compatible:
            continue
        if (trace_dir is not None
                and not _diagnostic_trace_valid(
                    trace_dir, row.get("diagnostic_trace") or {})):
            continue
        resumed[question_id] = row
    return dict(sorted(resumed.items()))
async def main() -> int:
    os.environ["ARIADNE_BENCHMARK_NO_REPAIR"] = "1"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='databricks')
    ap.add_argument('--corpus', default=str(REPO / 'spool-corpus'))
    ap.add_argument('--questions', default=str(QUESTIONS))
    ap.add_argument('--out', default=str(HERE / 'answers-ariadne-arm.json'))
    ap.add_argument('--trace-dir', default=None, help='directory for compressed per-question diagnostic traces')
    ap.add_argument('--only', default=None,
                    help='comma-separated ids, for a cheap smoke run')
    ap.add_argument('--concurrency', type=int, default=4,
                    help='questions in flight (default 4, matching the judge)')
    ap.add_argument('--timeout', type=float, default=900,
                    help='per-question wall ceiling in seconds (default 900)')
    ap.add_argument('--resume', action='store_true',
                    help='reuse compatible checkpointed answers and run only missing questions')
    args = ap.parse_args()

    from config import get_config
    configured_model = get_config().model
    expected_model = 'claude-opus-4-8'
    if configured_model != expected_model:
        print(f'benchmark requires model {expected_model}, configured model is '
              f'{configured_model}; refusing an unequal arm', file=sys.stderr)
        return 2

    scored = {r['id'] for r in json.loads(REQS.read_text()) if not r['flag']}
    qs = [q for q in json.loads(Path(args.questions).read_text())
          if q['id'] in scored]
    if args.only:
        keep = {int(x) for x in args.only.split(',')}
        qs = [q for q in qs if q['id'] in keep]
    output = Path(args.out)
    corpus = Path(args.corpus)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    resumed = (_load_resumable_answers(
        output, qs, source=args.source, corpus=corpus, trace_dir = trace_dir) if args.resume else {})
    pending = [question for question in qs if int(question['id']) not in resumed]
    print(f'Ariadne arm: {len(qs)} question(s), source={args.source}, '
          f'{args.concurrency}-wide; resumed={len(resumed)}, pending={len(pending)}',
          flush=True)

    service = None
    if pending:
        from ariadne_mcp.service import AriadneService
        service = AriadneService.get()
    sem = asyncio.Semaphore(max(1, args.concurrency))
    completed = dict(resumed)
    source_cache = {}

    async def guarded(q: dict) -> dict:
        assert service is not None
        text = q.get('after') or q.get('question') or ''
        async with sem:
            print(f'  id {q["id"]:>3}  starting', flush=True)
            row = await asyncio.wait_for(
                _one(service, q['id'], text, args.source, corpus, source_cache, trace_dir = trace_dir),
                timeout=args.timeout)
        completed[int(q['id'])] = row
        _write_checkpoint(output, completed.values())
        print(f'  id {q["id"]:>3}  {row["elapsed_s"]:>6}s  '
              f'{len(row["answer"]):>5} chars  '
              f'{len(row["chain_files"])} chain file(s), '
              f'{len(row["citations"])} cited coordinate(s), '
              f'{len(row["files_read"])} materialized  conf={row["confidence"]}',
              flush=True)
        return row

    settled = await asyncio.gather(*(guarded(q) for q in pending),
                                   return_exceptions=True)
    failed = []
    for q, result in zip(pending, settled):
        if isinstance(result, BaseException):
            failed.append((q['id'], f'{type(result).__name__}: {result}'))
    rows = sorted(completed.values(), key=lambda row: int(row['id']))
    _write_checkpoint(output, rows)
    if failed:
        print(f'\n{len(failed)} question(s) FAILED and are absent from the '
              f'output (not scored as zero):', file=sys.stderr)
        for qid, why in failed:
            print(f'  id {qid}: {why}', file=sys.stderr)
    print(f'\nanswers ({len(rows)}/{len(qs)}) -> {output}')
    print('diagnostic chain coverage (not the hidden continuation gate):\n'
          f'  python evaluation/spool-clean-room/score_chain.py --answers {output}')
    return 1 if failed else 0
def _usage_summary(rows) -> dict:
    from docgen.pricing import LLM_PRICING
    keys = ("input_tokens", "cache_creation_input_tokens",
            "cache_read_input_tokens", "output_tokens")
    totals = {key: sum(int(row.get(key, 0) or 0) for row in rows) for key in keys}
    cost = 0.0
    for row in rows:
        rates = LLM_PRICING.get(str(row.get("model") or ""))
        if rates is None:
            continue
        input_rate, output_rate = rates
        cost += (int(row.get("input_tokens", 0) or 0) * input_rate
                 + int(row.get("cache_creation_input_tokens", 0) or 0) * input_rate * 1.25
                 + int(row.get("cache_read_input_tokens", 0) or 0) * input_rate * 0.10
                 + int(row.get("output_tokens", 0) or 0) * output_rate) / 1_000_000
    return {"num_turns": len(rows), "token_usage": totals,
            "total_cost_usd": cost, "usage_by_phase": {
                    phase: {key: sum(int(row.get(key, 0) or 0) for row in rows
                                  if row.get("phase") == phase) for key in keys}
                    for phase in sorted({str(row.get("phase") or "completion") for row in rows})}}
TRACE_SCHEMA = "ariadne-live-diagnostic-v1"


def _write_diagnostic_trace(
        trace_dir: Path, qid: int, payload: dict) -> dict:
    """Atomically write one deterministic compressed trace and return a receipt."""
    import gzip

    trace_dir.mkdir(parents=True, exist_ok=True)
    name = f"q{int(qid)}.json.gz"
    path = trace_dir / name
    temporary = trace_dir / (name + ".tmp")
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()
    compressed = gzip.compress(raw, mtime=0)
    temporary.write_bytes(compressed)
    temporary.replace(path)
    return {
        "schema": TRACE_SCHEMA,
        "file": name,
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_bytes": len(compressed),
        "uncompressed_bytes": len(raw),
    }


def _diagnostic_trace_valid(trace_dir: Path, receipt: dict) -> bool:
    """A resumable row must retain the exact trace produced with it."""
    name = str((receipt or {}).get("file") or "")
    expected = str((receipt or {}).get("sha256") or "")
    if (receipt or {}).get("schema") != TRACE_SCHEMA:
        return False
    if not name or Path(name).name != name or not expected:
        return False
    path = trace_dir / name
    try:
        return (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
    except OSError:
        return False


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
