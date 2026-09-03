"""
query/query_engine.py
Orchestrates the full 3-stage query pipeline.
This is the single entry point — FastAPI and LangGraph agent both call this.
"""
import asyncio
import logging
from typing import Optional

from query.processor import QueryProcessor
from query.retriever import Retriever
from query.reranker import Reranker
from models import ChunkResult
from langsmith import traceable   

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Full pipeline: QueryProcessor → Retriever → Reranker.
    All models load once at startup via singleton pattern.
    """

    def __init__(self):
        self.processor = QueryProcessor()
        self.retriever = Retriever()
        self.reranker = Reranker()
        logger.info("QueryEngine ready")

    """async def run(
        self,
        query: str,
        patient_id: Optional[str] = None,
        max_results: int = 5,
    ) -> list[ChunkResult]:"""


    # add decorator immediately before async def run:
    @traceable(name="rag_query_pipeline", run_type="chain")
    async def run(self, query, patient_id=None, max_results=5):
        # existing code unchanged
        
        #Run the full query pipeline.Returns max_results ranked ChunkResult objects for the generation layer.
        # Stage 1 — normalize query, extract filters
        processed = self.processor.process(query, patient_id)

        # Stage 2 — embed + search (retrieve 2x then rerank down)
        chunks = await self.retriever.retrieve(
            query=processed.clean_query,
            top_k=max_results * 2,
            filters=processed.filters,
        )

        if not chunks:
            logger.warning(f"No chunks found for: '{query[:50]}'")
            return []

        # Stage 3 — rerank is CPU-bound, run in thread pool
        ranked = await asyncio.to_thread(
            self.reranker.rerank,
            processed.clean_query,
            chunks,
            max_results,
        )

        logger.info(
            f"QueryEngine: '{query[:40]}' → {len(ranked)} chunks returned"
        )
        return ranked