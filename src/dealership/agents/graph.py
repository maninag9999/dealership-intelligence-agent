"""
LangGraph Agent Graph
---------------------
Defines the ReAct-style agent graph for the Dealership Intelligence Agent.

Graph flow:
  START → agent_node → (tool_node | END)
              ↑               |
              └───────────────┘

The agent_node calls the LLM (Groq) which decides:
  - Call a tool  → route to tool_node → back to agent_node
  - No more tools → route to END with final_answer
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from dealership.agents.prompts import SYSTEM_PROMPT
from dealership.agents.state import AgentState
from dealership.agents.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


# ── LLM setup ─────────────────────────────────────────────────────────────────
def _build_llm():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise OSError("GROQ_API_KEY not set. Add it to your .env file:\n" "  GROQ_API_KEY=gsk_...")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,  # Low temp for factual, consistent answers
        max_tokens=2048,
        api_key=api_key,
    ).bind_tools(ALL_TOOLS)


# ── Nodes ─────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """
    Core reasoning node — calls the LLM with the full message history.
    The LLM either produces a final answer or requests a tool call.
    """
    logger.info("agent_node called | messages=%d", len(state["messages"]))

    llm = _build_llm()

    # Build message list: system prompt + conversation history
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    response: AIMessage = llm.invoke(messages)
    logger.info(
        "LLM response | tool_calls=%d | content_len=%d",
        len(response.tool_calls) if response.tool_calls else 0,
        len(str(response.content)),
    )

    # Track which tools were called this turn
    tool_calls_made = list(state.get("tool_calls_made", []))
    if response.tool_calls:
        tool_calls_made.extend([tc["name"] for tc in response.tool_calls])

    # If no tool calls, this is the final answer
    final_answer = ""
    if not response.tool_calls:
        final_answer = str(response.content)

    return {
        "messages": [response],
        "tool_calls_made": tool_calls_made,
        "final_answer": final_answer,
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge: route to tool_node if LLM requested tools,
    otherwise end the graph.
    """
    last_message = state["messages"][-1]

    # AIMessage with tool_calls → run tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "end"


# ── Graph builder ──────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    """Compile and return the LangGraph agent graph."""
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Entry point
    graph.add_edge(START, "agent")

    # Conditional routing after agent node
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    # After tools run, always go back to agent
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Public interface ───────────────────────────────────────────────────────────
class DealershipAgent:
    """
    High-level wrapper around the compiled LangGraph agent.

    Usage
    -----
    >>> agent = DealershipAgent()
    >>> result = agent.run("Which vehicles are at risk of aging on the lot?")
    >>> print(result["final_answer"])
    """

    def __init__(self) -> None:
        logger.info("Building DealershipAgent graph…")
        self.graph = build_graph()
        logger.info("DealershipAgent ready.")

    def run(self, user_message: str) -> dict:
        """
        Run the agent on a single user message.

        Returns
        -------
        dict with keys:
          - final_answer    : str — the agent's response
          - tool_calls_made : list[str] — tools invoked
          - messages        : full message history
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_message)],
            "tool_calls_made": [],
            "context": {},
            "final_answer": "",
            "error": None,
        }

        logger.info("Running agent | query=%r", user_message[:80])
        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": 10},
        )
        logger.info(
            "Agent finished | tools_used=%s",
            final_state.get("tool_calls_made", []),
        )
        return final_state

    def stream(self, user_message: str):
        """
        Stream agent events — useful for the Streamlit UI.
        Yields state updates as they happen.
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_message)],
            "tool_calls_made": [],
            "context": {},
            "final_answer": "",
            "error": None,
        }
        yield from self.graph.stream(
            initial_state,
            config={"recursion_limit": 10},
        )
