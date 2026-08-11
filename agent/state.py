"""
agent/state.py
LangGraph state definition for the clinical decision agent.

State is passed between every node in the graph.
Each node receives the full state and returns updated fields.
TypedDict enforces structure — LangGraph validates this at runtime.
"""
from typing import Optional, TypedDict
from models import ChunkResult


class ClinicalAgentState(TypedDict):
    query:           str
    patient_id:      Optional[str]
    max_results:     int
    chunks:          list[ChunkResult]    # populated by retrieve_node
    answer:          str                  # populated by generate_node
    sources:         list[str]            # populated by generate_node
    needs_retrieval: bool                 # set by decide_node
    reasoning:       str                  # agent's decision explanation
    model_used:      str
    iterations:      int                  # guard against infinite loops