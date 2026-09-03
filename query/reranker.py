"""
query/reranker.py
Stage 3: re-ranks retrieved chunks using a cross-encoder.

Why two models?
  Bi-encoder (retrieval): fast, embeds query and chunk SEPARATELY.
    Searches all 706 vectors in milliseconds. Less precise.
  Cross-encoder (reranking): slow, looks at query+chunk TOGETHER.
    Much more accurate. Only feasible on top 10, not 706.

Interview answer: "retrieve 10 with bi-encoder, rerank to 5 with cross-encoder."
"""
import logging
from sentence_transformers import CrossEncoder
from models import ChunkResult
from langsmith import traceable   

logger = logging.getLogger(__name__)

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Scores (query, chunk) pairs together using a cross-encoder.
    Returns top_n chunks re-sorted by cross-encoder relevance.
    """

    def __init__(self):
        self.model = CrossEncoder(RERANK_MODEL)
        logger.info(f"Reranker loaded: {RERANK_MODEL}")

    '''def rerank(
        self,
        query: str,
        chunks: list[ChunkResult],
        top_n: int = 5,
    ) -> list[ChunkResult]:'''
    

    @traceable(name="cross_encoder_reranker", run_type="reranker")
    def rerank(self, query, chunks, top_n=5):
    # existing code unchanged
        """
        Score each (query, chunk_text) pair.
        Returns top_n sorted by cross-encoder score descending.
        Cross-encoder score replaces the original cosine score.
        """
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        results = [
        ChunkResult(
        chunk_id      = chunk.chunk_id,
        text          = chunk.text,
        source        = chunk.source,
        score         = round(float(score), 4),
        chunk_index   = chunk.chunk_index,
        chunk_type    = chunk.chunk_type,
        page_number   = chunk.page_number,        # ADD
        section_header= chunk.section_header,     # ADD
    )
    for score, chunk in ranked[:top_n]
]
        logger.info(
            f"Reranked {len(chunks)} → {len(results)} chunks. "
            f"Top score: {results[0].score:.4f}"
        )
        return results