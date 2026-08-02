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

logger = logging.getLogger(__name__)


class Retriever:
    """
    Converts query text → embedding → Pinecone search → ChunkResults.

    Critical: embedding model MUST match ingestion model.
    Ingestion used all-MiniLM-L6-v2 → query must also use all-MiniLM-L6-v2.
    Mixing models = cosine scores are meaningless.
    """

    def __init__(self):
        self.embedder = DocumentEmbedder()       # same model as ingestion
        self.vector_store = VectorStore()
        logger.info("Retriever initialized")

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[ChunkResult]:
        """
        Embed query and search Pinecone.
        Both steps run in thread pool — embedding is CPU-bound,
        Pinecone client is sync (blocking I/O).
        """
        # Embed — CPU-bound, must not block the event loop
        embedding = await asyncio.to_thread(
            self.embedder.embed_single_query, query
        )

        # Search Pinecone — sync client, must not block the event loop
        matches = await asyncio.to_thread(
            self.vector_store.similarity_search,
            embedding,
            top_k,
            filters or {},
        )

        chunks = [
        ChunkResult(
        chunk_id=match["id"],
        text=match["metadata"].get("text", ""),
        source=match["metadata"].get("source", ""),
        score=round(match["score"], 4),
        chunk_index=int(match["metadata"].get("chunk_index", 0)),
        chunk_type=match["metadata"].get("type", "text_pdf"),
    )
    for match in matches
]


        logger.info(f"Retrieved {len(chunks)} chunks for: '{query[:50]}'")
        return chunks