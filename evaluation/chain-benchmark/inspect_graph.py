#!/usr/bin/env python3
"""Inspect Ariadne's SCIP-derived occurrence graph without LLM synthesis."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
QUESTIONS = ROOT / "evaluation/spool-clean-room/questions_debcrumb_ask.json"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--only", help="comma-separated question ids")
    parser.add_argument("--source", default="databricks")
    parser.add_argument("--paid-route-selection", action="store_true", help="use the configured LLM to select SCIP owner families and routes")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    if args.only:
        wanted = {int(value) for value in args.only.split(",")}
        questions = [item for item in questions if int(item["id"]) in wanted]

    from ariadne_mcp.service import AriadneService
    service = AriadneService.get()
    previous = service.config._config.get("ask_synthesis")
    previous_route_selection = service.config._config.get("ask_route_selection")
    service.config._config["ask_synthesis"] = False
    service.config._config["ask_route_selection"] = args.paid_route_selection
    rows = []
    try:
        for item in questions:
            question = item.get("after") or item.get("question") or ""
            response = await service.ask(question, source=args.source)
            rows.append({"id": item["id"], "question": question,
                         "graph": response.graph_diagnostics})
            graph = response.graph_diagnostics
            origin_components = {}
            origin_symbols = {}
            for component in graph.get("components", []):
                for origin, symbols in component.get("seed_origins", {}).items():
                    origin_components[origin] = origin_components.get(origin, 0) + 1
                    origin_symbols[origin] = origin_symbols.get(origin, 0) + len(symbols)
            origins = ", ".join(
                f"{origin}={origin_components[origin]}c/{origin_symbols[origin]}s"
                for origin in sorted(origin_components)) or "none"
            clews = graph.get("clew_selection", {})
            clew_summary = (f"recall={clews.get('recalled', 0)}, "
                            f"accepted={clews.get('accepted', 0)}, "
                            f"selected={clews.get('selected', 0)}, "
                            f"families={clews.get('families', 0)}, "
                            f"obligations={clews.get('covered_obligations', 0)}/{clews.get('obligations', 0)}, "
                            f"missing={clews.get('missing_obligations', [])}, "
                            f"status={clews.get('status', 'missing')}, "
                            f"stage={clews.get('stage', 'missing')}, "
                            f"error={clews.get('error', '')}")
            print(f"id {item['id']}: {graph.get('node_count', 0)} nodes, "
                  f"{graph.get('edge_count', 0)} edges, "
                  f"{graph.get('component_count', 0)} components; origins: {origins}; "
                  f"clews: {clew_summary}", flush=True)
    finally:
        service.config._config["ask_synthesis"] = previous
        if previous_route_selection is None:
            service.config._config.pop("ask_route_selection", None)
        else:
            service.config._config["ask_route_selection"] = previous_route_selection
    Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
    print(f"graph report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
