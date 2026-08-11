"""
agent/clinical_agent.py
Builds and runs the clinical decision agent graph.

Graph structure:
  [decide] ──(needs_retrieval=True)──> [retrieve] ──> [generate] ──> END
  [decide] ──(needs_retrieval=False)──────────────> [generate] ──> END

The agent skips retrieval for non-clinical queries — saving latency and tokens.
For clinical queries, it runs the full RAG pipeline.
"""
import logging
from typing import Optional
from langgraph.graph import StateGraph, END
from agent.state import ClinicalAgentState
from agent.nodes import decide_node, retrieve_node, generate_node, route_after_decision

logger = logging.getLogger(__name__)

# ── Singleton graph — compiled once, reused across all requests ───────────────
_compiled_agent = None


def _build_agent():
    """Build and compile the LangGraph clinical decision agent."""
    graph = StateGraph(ClinicalAgentState)

    # Register nodes
    graph.add_node("decide",   decide_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # Entry point
    graph.set_entry_point("decide")

    # Conditional routing after decide
    graph.add_conditional_edges(
        "decide",
        route_after_decision,
        {"retrieve": "retrieve", "generate": "generate"},
    )

    # After retrieval, always generate
    graph.add_edge("retrieve", "generate")

    # After generation, done
    graph.add_edge("generate", END)

    compiled = graph.compile()
    logger.info("Clinical agent graph compiled successfully")
    return compiled


def get_agent():
    """Return compiled agent — builds on first call, cached after."""
    global _compiled_agent
    if _compiled_agent is None:
        _compiled_agent = _build_agent()
    return _compiled_agent


async def run_clinical_agent(
    query:      str,
    patient_id: Optional[str] = None,
    max_results: int = 5,
) -> dict:
    """
    Run the clinical decision agent for one query.
    Returns structured response with answer, sources, and agent's reasoning.
    """
    agent = get_agent()

    initial_state: ClinicalAgentState = {
        "query":           query,
        "patient_id":      patient_id,
        "max_results":     max_results,
        "chunks":          [],
        "answer":          "",
        "sources":         [],
        "needs_retrieval": True,
        "reasoning":       "",
        "model_used":      "",
        "iterations":      0,
    }

    final_state = await agent.ainvoke(initial_state)

    logger.info(
        f"Agent complete: needs_retrieval={final_state['needs_retrieval']} | "
        f"chunks_used={len(final_state['chunks'])} | "
        f"reasoning='{final_state['reasoning']}'"
    )

    return {
        "answer":          final_state["answer"],
        "sources":         final_state["sources"],
        "model_used":      final_state["model_used"],
        "needs_retrieval": final_state["needs_retrieval"],
        "reasoning":       final_state["reasoning"],
        "chunks_used":     len(final_state["chunks"]),
    }