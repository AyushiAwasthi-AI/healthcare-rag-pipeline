"""
query/retriever.py
Stage 2: embeds the query and searches Pinecone.
Returns top_k ChunkResult objects sorted by cosine similarity.
"""
import asyncio
import logging
from typing import Optional

from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import VectorStore
from models import ChunkResult
from langsmith import traceable   

logger = logging.getLogger(__name__)


class Retriever:
    """
    Converts query text → embedding → Pinecone search → ChunkResults.
    Critical: embedding model MUST match ingestion model.
    """

    def __init__(self):
        self.embedder    = DocumentEmbedder()
        self.vector_store = VectorStore()
        logger.info("Retriever initialized")

    '''async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[ChunkResult]:'''

    @traceable(name="pinecone_retriever", run_type="retriever")
    async def retrieve(self, query, top_k=10, filters=None):
        

        # Step 1 — embed query (CPU-bound → thread pool)
        embedding = await asyncio.to_thread(
            self.embedder.embed_single_query, query
        )

        # Step 2 — search Pinecone (sync client → thread pool)
        matches = await asyncio.to_thread(
            self.vector_store.similarity_search,
            embedding,
            top_k,
            filters or {},
        )
        # Step 3 — convert to ChunkResult objects
        chunks = []
        for match in matches:
            meta = match.get("metadata", {})

            # page_number: Pinecone may return int or float — normalise explicitly
            raw_page = meta.get("page_number")
            try:
                page_num = int(raw_page) if raw_page is not None else None
                if page_num is not None and page_num <= 0:
                    page_num = None
            except (ValueError, TypeError):
                page_num = None

            chunks.append(ChunkResult(
                chunk_id      = match["id"],
                text          = meta.get("text", ""),
                source        = meta.get("source", ""),
                score         = round(match["score"], 4),
                chunk_index   = int(meta.get("chunk_index", 0)),
                chunk_type    = meta.get("type", "text_pdf"),
                page_number   = page_num,
                section_header= meta.get("section_header") or None,
            ))

        logger.info(f"Retrieved {len(chunks)} chunks for: '{query[:50]}'")
        return chunks