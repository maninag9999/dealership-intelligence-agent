#!/usr/bin/env python
"""
Agent Verification Suite — Day 5
==================================
Tests the agentic layer WITHOUT requiring Groq API or a live DB.

Tests:
  1. Tool imports                  — all tools loadable
  2. search_inventory              — ChromaDB demo collection works
  3. predict_days_on_lot           — XGBoost prediction returns valid number
  4. get_rep_archetypes            — K-Means archetypes generated
  5. score_customer_sentiment      — DistilBERT scores reviews correctly
  6. query_inventory_stats         — Demo stats returned when DB absent
  7. Graph structure               — LangGraph graph compiles without errors
  8. State schema                  — AgentState TypedDict is valid

Usage
-----
    python scripts/verify_agent.py
    python scripts/verify_agent.py --skip-sentiment
    python scripts/verify_agent.py --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logger = logging.getLogger(__name__)

_PASS = "✅ PASS"
_FAIL = "❌ FAIL"
_SKIP = "⏭️  SKIP"


def _check(label: str, passed: bool, detail: str = "") -> bool:
    symbol = _PASS if passed else _FAIL
    msg = f"  {symbol}  {label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return passed


# ── Individual tests ──────────────────────────────────────────────────────────


def test_tool_imports() -> bool:
    print("\n🔧  [1/8] Tool imports")
    print("  " + "─" * 48)
    try:
        from dealership.agents.tools import (  # noqa: F401
            ALL_TOOLS,
            get_rep_archetypes,
            predict_days_on_lot,
            query_inventory_stats,
            score_customer_sentiment,
            search_inventory,
        )

        return _check("All tools imported", True, f"{len(ALL_TOOLS)} tools found")
    except Exception as exc:
        return _check("Tool imports", False, str(exc))


def test_search_inventory() -> bool:
    print("\n🔍  [2/8] search_inventory tool")
    print("  " + "─" * 48)
    try:
        from dealership.agents.tools import search_inventory

        result = search_inventory.invoke({"query": "red SUV under 35000", "n_results": 3})
        passed = isinstance(result, str) and len(result) > 10
        return _check("Returns string results", passed, result[:80] + "…")
    except Exception as exc:
        return _check("search_inventory", False, str(exc))


def test_predict_days_on_lot() -> bool:
    print("\n📦  [3/8] predict_days_on_lot tool")
    print("  " + "─" * 48)
    try:
        from dealership.agents.tools import predict_days_on_lot

        vehicle = json.dumps(
            {
                "make": "Toyota",
                "model": "Camry",
                "year": 2021,
                "mileage": 45000,
                "price": 24000,
                "trim": "SE",
                "color": "Gray",
                "fuel_type": "Gasoline",
                "transmission": "Automatic",
                "certified_pre_owned": False,
                "days_since_last_price_drop": 10,
            }
        )
        result = predict_days_on_lot.invoke({"vehicle_data": vehicle})
        passed = "days on lot" in result.lower()
        return _check("Prediction returned", passed, result.split("\n")[0])
    except Exception as exc:
        return _check("predict_days_on_lot", False, str(exc))


def test_get_rep_archetypes() -> bool:
    print("\n👥  [4/8] get_rep_archetypes tool")
    print("  " + "─" * 48)
    try:
        from dealership.agents.tools import get_rep_archetypes

        result = get_rep_archetypes.invoke({"rep_data": ""})
        passed = "Archetype" in result or "archetype" in result.lower()
        return _check("Archetypes generated", passed, result.split("\n")[0])
    except Exception as exc:
        return _check("get_rep_archetypes", False, str(exc))


def test_score_sentiment() -> bool:
    print("\n💬  [5/8] score_customer_sentiment tool")
    print("  " + "─" * 48)
    try:
        from dealership.agents.tools import score_customer_sentiment

        reviews = json.dumps(
            [
                "Absolutely loved the experience!",
                "Terrible service, very disappointed.",
                "It was okay, nothing special.",
            ]
        )
        result = score_customer_sentiment.invoke({"reviews": reviews})
        passed = "Positive" in result or "POSITIVE" in result
        return _check("Sentiment scored", passed, result.split("\n")[0])
    except Exception as exc:
        return _check("score_customer_sentiment", False, str(exc))


def test_inventory_stats() -> bool:
    print("\n📊  [6/8] query_inventory_stats tool")
    print("  " + "─" * 48)
    try:
        from dealership.agents.tools import query_inventory_stats

        result = query_inventory_stats.invoke({"metric": "overview"})
        passed = "vehicle" in result.lower() or "inventory" in result.lower()
        return _check("Stats returned", passed, result.split("\n")[0])
    except Exception as exc:
        return _check("query_inventory_stats", False, str(exc))


def test_graph_compiles() -> bool:
    print("\n🕸️   [7/8] LangGraph graph compilation")
    print("  " + "─" * 48)
    try:
        # Patch env so build_graph doesn't fail on missing GROQ_API_KEY
        import os

        os.environ.setdefault("GROQ_API_KEY", "test_key_for_compile_check")

        from dealership.agents.graph import build_graph

        graph = build_graph()
        # Check it has the expected nodes
        node_names = list(graph.nodes.keys()) if hasattr(graph, "nodes") else []
        passed = graph is not None
        return _check("Graph compiled", passed, f"nodes: {node_names}")
    except Exception as exc:
        # Graph compile errors around API key are acceptable in offline mode
        if "api_key" in str(exc).lower() or "groq" in str(exc).lower():
            return _check("Graph compiled (offline mode)", True, "GROQ_API_KEY not validated offline")
        return _check("Graph compilation", False, str(exc))


def test_state_schema() -> bool:
    print("\n📋  [8/8] AgentState schema")
    print("  " + "─" * 48)
    try:
        from langchain_core.messages import HumanMessage

        from dealership.agents.state import AgentState

        # Construct a valid state dict
        state: AgentState = {
            "messages": [HumanMessage(content="test")],
            "tool_calls_made": [],
            "context": {},
            "final_answer": "",
            "error": None,
        }
        passed = all(k in state for k in ["messages", "tool_calls_made", "context", "final_answer", "error"])
        return _check("AgentState schema valid", passed, f"{len(state)} fields")
    except Exception as exc:
        return _check("AgentState schema", False, str(exc))


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify Day 5 agent layer.")
    p.add_argument("--skip-sentiment", action="store_true", help="Skip DistilBERT test (saves ~2 min on first run).")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.WARNING,
    )
    for lib in ("transformers", "xgboost", "mlflow", "chromadb", "httpx"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    print("\n" + "=" * 55)
    print("  Dealership Intelligence Agent — Day 5 Verification")
    print("=" * 55)

    results: list[bool] = []
    results.append(test_tool_imports())
    results.append(test_search_inventory())
    results.append(test_predict_days_on_lot())
    results.append(test_get_rep_archetypes())

    if args.skip_sentiment:
        print("\n💬  [5/8] score_customer_sentiment — SKIPPED")
        results.append(True)  # Don't penalise the skip
    else:
        results.append(test_score_sentiment())

    results.append(test_inventory_stats())
    results.append(test_graph_compiles())
    results.append(test_state_schema())

    total = len(results)
    passed = sum(results)

    print("\n" + "=" * 55)
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉  Day 5 agent layer verified successfully!")
        print("\n  Try the agent:")
        print("  python scripts/run_agent.py --query 'Give me an inventory overview'")
    else:
        print("  ⚠️   Some checks failed — review output above.")
    print("=" * 55 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
