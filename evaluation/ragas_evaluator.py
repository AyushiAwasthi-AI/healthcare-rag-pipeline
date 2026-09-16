"""
evaluation/ragas_evaluator.py
Evaluates the RAG pipeline using RAGAS metrics.
Uses sequential execution to avoid Groq free-tier rate limits.
"""
import math
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

logger = logging.getLogger(__name__)


def _safe_score(val) -> float:
    """
    Handle RAGAS returning NaN, lists, or None on partial failures.
    Timeout errors produce NaN values — average valid scores, return 0.0 if all failed.
    """
    if isinstance(val, list):
        valid = [
            v for v in val
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        return round(sum(valid) / len(valid), 4) if valid else 0.0
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return 0.0


def _build_ragas_llm():
    return LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=settings.groq_api_key,
        )
    )


def _build_ragas_embeddings():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )


class RAGASEvaluator:

    def __init__(self):
        self.llm        = _build_ragas_llm()
        self.embeddings = _build_ragas_embeddings()
        # Sequential execution — prevents Groq rate limit timeouts
        self.run_config = RunConfig(
            timeout=180,
            max_retries=3,
            max_wait=60,
            max_workers=1,    # 1 = sequential, not concurrent
        )
        logger.info("RAGASEvaluator initialized")

    def evaluate(
        self,
        questions:     list[str],
        answers:       list[str],
        contexts:      list[list[str]],
        ground_truths: list[str],
    ) -> dict:

        dataset = Dataset.from_dict({
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts,
            "ground_truth": ground_truths,
        })

        result = evaluate(
            dataset          = dataset,
            metrics          = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm              = self.llm,
            embeddings       = self.embeddings,
            run_config       = self.run_config,
            raise_exceptions = False,    # partial failures return NaN, not crash
        )

        scores = {
            "faithfulness":      _safe_score(result["faithfulness"]),
            "answer_relevancy":  _safe_score(result["answer_relevancy"]),
            "context_precision": _safe_score(result["context_precision"]),
            "context_recall":    _safe_score(result["context_recall"]),
        }

        logger.info(f"RAGAS scores: {scores}")
        # Threshold alerting — production systems alert when quality degrades
        THRESHOLDS = {
            "faithfulness":      0.85,   # below = hallucination risk
            "answer_relevancy":  0.80,   # below = answers drifting off-topic
            "context_precision": 0.75,   # below = retrieval noise increasing
            "context_recall":    0.75,   # below = missing critical chunks
        }

        alerts = []
        for metric, threshold in THRESHOLDS.items():
            score = scores.get(metric, 0.0)
            if score < threshold:
                alert_msg = (
                    f"⚠️  RAGAS ALERT: {metric} = {score:.4f} "
                    f"(threshold: {threshold}) — investigate retrieval pipeline"
                )
                alerts.append(alert_msg)
                logger.warning(alert_msg)

        if alerts:
            logger.warning(
                f"RAGAS degradation detected: {len(alerts)} metric(s) below threshold. "
                f"In production: trigger PagerDuty, Slack alert, or block deployment."
            )
        else:
            logger.info("All RAGAS metrics within thresholds — pipeline healthy.")

        return scores
        
