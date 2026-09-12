"""
agent/nodes.py
The three nodes of the clinical decision agent.

decide_node   — agent reasons whether retrieval is needed
retrieve_node — calls existing QueryEngine (reuses pipeline)
generate_node — calls existing Generator (reuses pipeline)

Key design: nodes reuse existing pipeline components.
The agent adds decision-making ON TOP of the pipeline, not replacing it.
"""
import json
import logging
from typing import Optional
from groq import AsyncGroq
from config import settings
from query import QueryEngine
from generation.generator import Generator
from agent.state import ClinicalAgentState

logger = logging.getLogger(__name__)

# ── Singletons — built once, reused across all agent invocations ──────────────
_query_engine: Optional[QueryEngine] = None
_generator:    Optional[Generator]   = None
_groq_client:  Optional[AsyncGroq]   = None


def _get_query_engine() -> QueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine()
    return _query_engine


def _get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


def _get_groq() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


# ── Decision prompt ───────────────────────────────────────────────────────────
DECISION_PROMPT = """You are a clinical query router for a healthcare RAG system.

Decide if this query requires searching clinical documents to answer correctly.

Return ONLY valid JSON, no markdown, no explanation outside the JSON:
{{"needs_retrieval": true, "reasoning": "one sentence"}}

Query: {query}

Rules:
- needs_retrieval: true  → asks about clinical guidelines, symptoms, treatments,
                           medications, diagnoses, lab values, or protocols
- needs_retrieval: false → general greeting, non-medical topic, or simple
                           factual question answerable without clinical documents"""


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def decide_node(state: ClinicalAgentState) -> ClinicalAgentState:
    """
    Agent reasoning step — decides whether retrieval is needed.
    This is what separates an agent from a pipeline.
    A pipeline always retrieves. An agent decides first.
    """
    client = _get_groq()

    response = await client.chat.completions.create(
        model   = settings.llm_model,
        messages= [{"role": "user", "content": DECISION_PROMPT.format(query=state["query"])}],
        temperature = 0.0,
        max_tokens  = 80,
    )

    raw = response.choices[0].message.content.strip()

    try:
        raw_clean = raw.strip()
        # Strip markdown fences if model wraps in ```json
        if "```" in raw_clean:
            parts = raw_clean.split("```")
            for part in parts:
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    raw_clean = part
                    break
    # Find JSON object anywhere in the response
        start = raw_clean.find("{")
        end   = raw_clean.rfind("}") + 1
        if start != -1 and end > start:
            raw_clean = raw_clean[start:end]
        decision        = json.loads(raw_clean)
        needs_retrieval = bool(decision.get("needs_retrieval", True))
        reasoning       = decision.get("reasoning", "Decision made")
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        needs_retrieval = True
        reasoning       = "Parse failed — defaulting to retrieval for safety"
        logger.info(f"Agent decision: needs_retrieval={needs_retrieval} | {reasoning}")

    return {
        **state,
        "needs_retrieval": needs_retrieval,
        "reasoning":       reasoning,
        "iterations":      state.get("iterations", 0) + 1,
    }


async def retrieve_node(state: ClinicalAgentState) -> ClinicalAgentState:
    """
    Retrieval step — reuses the existing 3-stage QueryEngine.
    Agent chose to retrieve, so we run the full pipeline.
    """
    engine = _get_query_engine()
    chunks = await engine.run(
        query      = state["query"],
        patient_id = state.get("patient_id"),
        max_results= state.get("max_results", 5),
    )
    logger.info(f"Retrieved {len(chunks)} chunks")
    return {**state, "chunks": chunks}


async def generate_node(state: ClinicalAgentState) -> ClinicalAgentState:
    """
    Generation step — reuses the existing Generator.
    Runs whether or not retrieval happened.
    If no chunks retrieved, Generator returns 'insufficient context' response.
    """
    generator = _get_generator()
    result    = await generator.generate(
        query      = state["query"],
        chunks     = state.get("chunks", []),
        patient_id = state.get("patient_id"),
    )
    return {
        **state,
        "answer":     result["answer"],
        "sources":    result["sources"],
        "model_used": result["model_used"],
    }


def route_after_decision(state: ClinicalAgentState) -> str:
    """
    Conditional edge function — returns the name of the next node.
    LangGraph calls this after decide_node to determine routing.
    """
    return "retrieve" if state.get("needs_retrieval", True) else "generate"

CONFIDENCE_THRESHOLD = 3.5   # cross-encoder scores range ~0 to 10
                              # 3.5 = moderate clinical relevance
                              # below this = uncertain, flag for review

async def confidence_check_node(state: ClinicalAgentState) -> ClinicalAgentState:
    """
    Human-in-the-loop gate.

    Evaluates average cross-encoder score of retrieved chunks.
    High confidence → answer delivered directly to clinician.
    Low confidence  → answer flagged for pharmacist/clinical review.

    Why this matters in healthcare:
    AI should augment clinical judgment, never replace it.
    When the system is uncertain, a human must verify before
    the answer reaches a doctor.
    """
    chunks = state.get("chunks", [])

    if not chunks:
        confidence = 0.0
    else:
        scores     = [c.score for c in chunks if c.score is not None]
        confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

    requires_review = confidence < CONFIDENCE_THRESHOLD

    if requires_review:
        # Prepend clinical safety warning to the answer
        warning = (
            f"⚠️ CLINICAL REVIEW REQUIRED — Confidence score: {confidence:.2f} "
            f"(threshold: {CONFIDENCE_THRESHOLD}). "
            f"This response should be verified by a qualified clinician "
            f"before being used in patient care.\n\n"
        )
        answer = warning + state.get("answer", "")
        logger.warning(
            f"Low confidence answer flagged for review: "
            f"score={confidence:.4f} < threshold={CONFIDENCE_THRESHOLD}"
        )
    else:
        answer = state.get("answer", "")
        logger.info(f"Confidence check passed: score={confidence:.4f}")

    return {
        **state,
        "confidence_score": confidence,
        "requires_review":  requires_review,
        "answer":           answer,
    }