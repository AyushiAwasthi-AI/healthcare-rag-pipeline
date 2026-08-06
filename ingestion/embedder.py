import uuid
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
#from dotenv import load_dotenv

#load_dotenv()
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Converts text chunks into vector embeddings.
    Supports multiple embedding models for domain-specific use.
    Adds unique IDs for HIPAA-compliant deletion tracking.
    """

    # Model registry — maps document type to correct model
    MODEL_REGISTRY = {
        "clinical_notes":    "emilyalsentzer/Bio_ClinicalBERT",
        "research_papers":   "microsoft/BiomedNLP-PubMedBERT-base",
        "hl7_messages":      "dmis-lab/biobert-base-cased-v1.2",
        "general":           "all-MiniLM-L6-v2"
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with specific model.
        Defaults to general model if not specified.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Embedder initialized with model: {model_name}")

    def embed(self, chunks: List[Dict]) -> List[Dict]:
        """
        Main entry point.
        Receives chunks from DocumentChunker.
        Returns chunks with embeddings and unique IDs added.
        """
        if not chunks:
            logger.warning("Empty chunks list received")
            return []

        # Extract text from all chunks for batch processing
        texts = [chunk["text"] for chunk in chunks]

        # Batch embed — faster than one by one
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # Critical for cosine similarity
        )

        # Attach embedding and unique ID to each chunk
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = {
                **chunk,  # Keep all existing metadata
                "embedding": embeddings[i].tolist(),
                "chunk_id": self._generate_chunk_id(chunk),
                "model_used": self.model_name
            }
            embedded_chunks.append(embedded_chunk)

        logger.info(f"Successfully embedded {len(embedded_chunks)} chunks")
        return embedded_chunks

    #def _generate_chunk_id(self, chunk: Dict) -> str:
        """
        Generate unique, deterministic ID per chunk.
        Deterministic means same chunk always gets same ID.
        Critical for HIPAA deletion — can find chunk again later.
        """
        '''# Combine source + chunk_index for uniqueness
        unique_string = f"{chunk['source']}_{chunk['chunk_index']}"
        # UUID5 generates deterministic UUID from string
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))'''
    def _generate_chunk_id(self, chunk: dict) -> str:
        """
        Deterministic ID from source + page + position.
        Same document re-ingested = same IDs = safe upsert, no duplicates.
        """
        page = chunk.get("page_number", 0)
        unique_string = f"{chunk['source']}_p{page}_{chunk['chunk_index']}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

    @classmethod
    def for_document_type(cls, doc_type: str) -> "DocumentEmbedder":
        """
        Factory method — creates correct embedder for document type.
        Usage: embedder = DocumentEmbedder.for_document_type("clinical_notes")
        """
        model_name = cls.MODEL_REGISTRY.get(doc_type, "all-MiniLM-L6-v2")
        logger.info(f"Selected model {model_name} for {doc_type}")
        return cls(model_name)

    def embed_single_query(self, query: str) -> list[float]:
        """
        Embed a single query string for retrieval.
        MUST use same model as document ingestion - 
        mixing models breaks cosine similarity entirely
        """
        embedding = self.model.encode(query)
        return embedding.tolist()