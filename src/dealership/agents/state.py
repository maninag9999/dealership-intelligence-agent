"""
Agent State
-----------
Defines the shared state that flows through every node in the
LangGraph dealership agent graph.

LangGraph passes this TypedDict between nodes — each node reads
from it and returns a partial update dict.
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state for the Dealership Intelligence Agent.

    Fields
    ------
    messages : list
        Full conversation history. `add_messages` is a LangGraph
        reducer that appends new messages rather than replacing.
    tool_calls_made : list[str]
        Names of tools invoked in this turn — used for logging.
    context : dict[str, Any]
        Scratch-pad for tool results that nodes need to share
        (e.g. ML predictions, ChromaDB hits).
    final_answer : str
        Populated by the last node before END.
    error : str | None
        Set if any node encounters an unrecoverable error.
    """

    messages: Annotated[list, add_messages]
    tool_calls_made: list[str]
    context: dict[str, Any]
    final_answer: str
    error: str | None
