#!/usr/bin/env python
"""
Interactive CLI for the Dealership Intelligence Agent.

Usage
-----
    python scripts/run_agent.py
    python scripts/run_agent.py --query "Which vehicles are aging on the lot?"
    python scripts/run_agent.py --query "How is the team performing?" --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Load .env file for GROQ_API_KEY
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dealership.agents.graph import DealershipAgent  # noqa: E402

BANNER = """
╔══════════════════════════════════════════════════════╗
║     Dealership Intelligence Agent  🚗  Day 5         ║
║     Type your question or 'quit' to exit             ║
╚══════════════════════════════════════════════════════╝
"""

EXAMPLE_QUERIES = [
    "Which vehicles are at risk of aging on the lot?",
    "Give me an inventory overview",
    "What are the different sales rep archetypes on my team?",
    "Score this review: 'The staff was incredibly helpful and the process was smooth!'",
    "Predict days on lot for a 2021 Toyota Camry SE with 45000 miles priced at 24000",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Dealership Intelligence Agent.")
    p.add_argument("--query", "-q", type=str, default=None, help="Single query to run (non-interactive mode).")
    p.add_argument("--verbose", "-v", action="store_true", help="Show tool calls and debug info.")
    p.add_argument("--examples", action="store_true", help="Show example queries and exit.")
    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
    for lib in ("httpx", "httpcore", "groq", "chromadb", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def run_query(agent: DealershipAgent, query: str, verbose: bool) -> None:
    print(f"\n🤔 Question: {query}")
    print("─" * 55)
    print("⏳ Thinking...\n")

    result = agent.run(query)

    answer = result.get("final_answer", "No answer generated.")
    tools_used = result.get("tool_calls_made", [])

    print(answer)

    if verbose and tools_used:
        print(f"\n🔧 Tools used: {', '.join(tools_used)}")

    print("─" * 55)


def interactive_mode(agent: DealershipAgent, verbose: bool) -> None:
    print(BANNER)
    print("💡 Example questions:")
    for i, q in enumerate(EXAMPLE_QUERIES, 1):
        print(f"   {i}. {q}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        # Allow picking example by number
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(EXAMPLE_QUERIES):
                user_input = EXAMPLE_QUERIES[idx]
                print(f"Using: {user_input}")

        run_query(agent, user_input, verbose)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    if args.examples:
        print("Example queries:")
        for i, q in enumerate(EXAMPLE_QUERIES, 1):
            print(f"  {i}. {q}")
        return 0

    print("🚀 Loading Dealership Intelligence Agent…")
    try:
        agent = DealershipAgent()
    except OSError as exc:
        print(f"\n❌ {exc}")
        return 1

    if args.query:
        run_query(agent, args.query, args.verbose)
    else:
        interactive_mode(agent, args.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())
