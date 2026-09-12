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
import uuid
import time
import os
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
from agent.clinical_agent import run_clinical_agent, get_agent
from models import AgentQueryRequest, AgentQueryResponse

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

    global _query_engine, _generator, _embedder, _vector_store

    # ── LangSmith tracing ──────────────────────────────────────────
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
        os.environ["LANGCHAIN_API_KEY"]     = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"]     = settings.langchain_project
        logger.info(f"LangSmith tracing enabled → {settings.langchain_project}")
    else:
        logger.info("LangSmith tracing disabled — LANGSMITH_API_KEY not set")

    # ── Load models ────────────────────────────────────────────────
    logger.info("Starting up — loading models...")
    _query_engine = QueryEngine()
    _generator    = Generator()
    _embedder     = DocumentEmbedder()
    _vector_store = VectorStore()
    _vector_store = VectorStore()
    get_agent()                          # ADD THIS LINE HERE
    logger.info("All models loaded. Server ready.")
    

    yield

    logger.info("Shutting down.")
    
    """
    FastAPI lifespan — runs startup code before serving requests.
    Models load ONCE here. All requests reuse the same instances.
    Replaces deprecated @app.on_event("startup").
    
    global _query_engine, _generator, _embedder, _vector_store

    logger.info("Starting up — loading models...")
    _query_engine = QueryEngine()         # loads bi-encoder + cross-encoder
    _generator    = Generator()           # initialises Groq client
    _embedder     = DocumentEmbedder()    # loads sentence-transformers
    _vector_store = VectorStore()         # connects to Pinecone
    logger.info("All models loaded. Server ready.")

    yield   # server runs here

    logger.info("Shutting down.")
    """


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
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] POST /query received")
    start = time.time()

    try:
        chunks = await asyncio.wait_for(
            _query_engine.run(
                query      = request.query,
                patient_id = request.patient_id,
                max_results= request.max_results,
            ),
            timeout=30.0
        )
        result = await asyncio.wait_for(
            _generator.generate(
                query      = request.query,
                chunks     = chunks,
                patient_id = request.patient_id,
            ),
            timeout=30.0
        )

    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Query timed out after 30s")
        raise HTTPException(status_code=504, detail="Request timed out — try again")

    except Exception as e:
        # HIPAA: never log raw query or exception message — may contain PHI
        logger.error(f"[{request_id}] Query failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal server error")

    processing_time = round((time.time() - start) * 1000, 2)
    logger.info(f"[{request_id}] Query complete in {processing_time}ms")

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

# ── POST /agent/query ─────────────────────────────────────────────────────────
@app.post("/agent/query", response_model=AgentQueryResponse)
async def agent_query(request: AgentQueryRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] POST /agent/query received")
    start = time.time()

    try:
        result = await asyncio.wait_for(
            run_clinical_agent(
                query      = request.query,
                patient_id = request.patient_id,
                max_results= request.max_results,
            ),
            timeout=45.0   # agent gets more time — it makes an extra LLM call
        )

    except asyncio.TimeoutError:
        logger.error(f"[{request_id}] Agent query timed out after 45s")
        raise HTTPException(status_code=504, detail="Request timed out — try again")

    except Exception as e:
        # HIPAA: never log raw query or exception message — may contain PHI
        logger.error(f"[{request_id}] Agent query failed: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Internal server error")

    processing_time = round((time.time() - start) * 1000, 2)
    logger.info(f"[{request_id}] Agent complete in {processing_time}ms")

    return AgentQueryResponse(
        answer            = result["answer"],
        query             = request.query,
        sources           = result["sources"],
        needs_retrieval   = result["needs_retrieval"],
        reasoning         = result["reasoning"],
        chunks_used       = result["chunks_used"],
        model_used        = result["model_used"],
        processing_time_ms= round((time.time() - start) * 1000, 2),
        confidence_score  = result.get("confidence_score", 0.0),   # ADD
        requires_review   = result.get("requires_review", False),   # ADD

    )