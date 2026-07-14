import os
import logging
from typing import List, Dict
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages all vector DB operations.
    Handles: storing, searching, deleting, updating chunks.
    HIPAA compliant — supports deletion by source document.
    """

    def __init__(self):
        """
        Initialize Pinecone connection.
        Reads API key from .env file.
        """
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX")

        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file")
        if not index_name:
            raise ValueError("PINECONE_INDEX not found in .env file")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        # Connect to existing index
        self.index = self.pc.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

    def upsert_chunks(self, embedded_chunks: List[Dict]) -> int:
        """
        Store embedded chunks in vector DB.
        Uses upsert — updates if exists, inserts if new.
        Returns number of chunks stored.
        """
        if not embedded_chunks:
            logger.warning("Empty chunks list — nothing to upsert")
            return 0

        # Pinecone expects specific format
        vectors = []
        for chunk in embedded_chunks:
            vectors.append({
                "id": chunk["chunk_id"],
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "type": chunk["type"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"]
                }
            })

        # Batch upsert — more efficient than one at a time
        batch_size = 100
        total_upserted = 0

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            total_upserted += len(batch)
            logger.info(
                f"Upserted batch {i//batch_size + 1}: "
                f"{total_upserted}/{len(vectors)} chunks"
            )

        logger.info(f"Successfully stored {total_upserted} chunks")
        return total_upserted

    def similarity_search(self,
                         query_embedding: List[float],
                         top_k: int = 20) -> List[Dict]:
        """
        Search for similar chunks using query embedding.
        Returns top_k most similar chunks with metadata.
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Format results for re-ranker
        chunks = []
        for match in results.matches:
            chunks.append({
                "chunk_id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "source": match.metadata.get("source", ""),
                "type": match.metadata.get("type", ""),
                "chunk_index": match.metadata.get("chunk_index", 0)
            })

        logger.info(
            f"Retrieved {len(chunks)} chunks, "
            f"top score: {chunks[0]['score']:.4f}"
        )
        return chunks

    def delete_document(self, source_path: str) -> None:
        """
        Delete all chunks from a specific document.
        HIPAA compliance — right to erasure.
        Finds all chunks where source matches, deletes them.
        """
        # Pinecone supports deletion by metadata filter
        self.index.delete(
            filter={"source": {"$eq": source_path}}
        )
        logger.info(f"Deleted all chunks from: {source_path}")

    def get_index_stats(self) -> Dict:
        """
        Returns current index statistics.
        Useful for monitoring pipeline health.
        """
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimensions": stats.dimension,
            "index_name": self.index_name
        }