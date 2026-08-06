"""
api/main.py
FastAPI application — Clinical Decision Support RAG API.

3 endpoints:
  GET  /health  — liveness check for Docker, Azure, load balancers
  POST /query   — full RAG pipeline: embed → search → rerank → generate
  POST /ingest  — ingest one document into the vector store

Singleton pattern: QueryEngine and Generator load once at startup.
Loading them per-request would add 3-5 seconds to every call.
"""
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from query import QueryEngine
from generation.generator import Generator
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import VectorStore
from models import (
    QueryRequest, QueryResponse,
    IngestionRequest, IngestionResponse,
    HealthCheckResponse,
)
from config import settings

logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
# Declared at module level — shared across all requests.
# Never instantiate these inside an endpoint.
_query_engine:  QueryEngine        = None
_generator:     Generator          = None
_embedder:      DocumentEmbedder   = None
_vector_store:  VectorStore        = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan — runs startup code before serving requests.
    Models load ONCE here. All requests reuse the same instances.
    Replaces deprecated @app.on_event("startup").
    """
    global _query_engine, _generator, _embedder, _vector_store

    logger.info("Starting up — loading models...")
    _query_engine = QueryEngine()         # loads bi-encoder + cross-encoder
    _generator    = Generator()           # initialises Groq client
    _embedder     = DocumentEmbedder()    # loads sentence-transformers
    _vector_store = VectorStore()         # connects to Pinecone
    logger.info("All models loaded. Server ready.")

    yield   # server runs here

    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Clinical Decision Support RAG API",
    description = "HIPAA-compliant RAG system for healthcare document Q&A",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Liveness probe for Docker, Kubernetes, and Azure Container Apps.
    Returns 200 if all components are operational.
    Every production deployment pings this endpoint.
    """
    try:
        stats       = _vector_store.get_index_stats()
        pinecone_ok = stats["total_vectors"] > 0
    except Exception:
        pinecone_ok = False

    status = "healthy" if (pinecone_ok and _query_engine is not None) else "degraded"

    return HealthCheckResponse(
        status                 = status,
        pinecone_connected     = pinecone_ok,
        embedding_model_loaded = _query_engine is not None,
        environment            = settings.environment,
    )


# ── POST /query ───────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Full RAG pipeline for one clinical query.
    processor → retriever → reranker → generator
    Returns answer with page-level source citations.
    """
    start = time.time()

    try:
        chunks = await _query_engine.run(
            query      = request.query,
            patient_id = request.patient_id,
            max_results= request.max_results,
        )

        result = await _generator.generate(
            query      = request.query,
            chunks     = chunks,
            patient_id = request.patient_id,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    processing_time = round((time.time() - start) * 1000, 2)

    return QueryResponse(
        answer                = result["answer"],
        query                 = request.query,
        patient_id            = request.patient_id,
        sources               = chunks if request.include_sources else [],
        retrieved_chunks_count= len(chunks),
        model_used            = result["model_used"],
        processing_time_ms    = processing_time,
    )


# ── POST /ingest ──────────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestionResponse)
async def ingest(request: IngestionRequest):
    """
    Ingest one document into the vector store via HTTP.
    Runs the full ingestion pipeline: load → chunk → embed → store.
    All CPU-bound steps run in thread pool — event loop stays free.
    """
    start = time.time()

    try:
        loader = DocumentLoader()
        pages  = await asyncio.to_thread(loader.load, request.file_path)

        valid_pages = [p for p in pages if p.get("text", "").strip()]
        if not valid_pages:
            raise HTTPException(
                status_code=400,
                detail="Document is empty or unreadable"
            )

        chunker    = DocumentChunker()
        all_chunks = []
        for page in valid_pages:
            all_chunks.extend(chunker.chunk(page))

        embedded = await asyncio.to_thread(_embedder.embed, all_chunks)
        stored   = await asyncio.to_thread(_vector_store.upsert_chunks, embedded)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    processing_time = round((time.time() - start) * 1000, 2)

    return IngestionResponse(
        status             = "success",
        source             = request.file_path,
        chunks_stored      = stored,
        processing_time_ms = processing_time,
    )