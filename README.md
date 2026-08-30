# healthcare-rag-pipeline
Production RAG pipeline for healthcare documents with HIPAA compliance Visibility: Public

# Healthcare RAG Pipeline 🏥

Production-grade Retrieval-Augmented Generation pipeline 
for healthcare documents with HIPAA compliance.

## Architecture

Ingestion Pipeline:
Document → File Type Check → OCR → Classifier → 
Chunker → Embedder → Vector DB (Pinecone)

Query Pipeline (WIP):
Query → Query Processor → Hybrid Search → 
Re-ranker → Prompt → LLM → RAGAS Evaluation

## Tech Stack

- **Vector DB:** Pinecone
- **Embeddings:** Sentence Transformers (ClinicalBERT)
- **LLM:** OpenAI GPT-4
- **Evaluation:** RAGAS
- **OCR:** Tesseract + AWS Textract
- **Framework:** LangChain
- **API:** FastAPI (coming)

## Key Features

- HIPAA-compliant chunk deletion by document source
- Dynamic chunk sizing based on document length
- Hybrid search: semantic + BM25 keyword
- Cross-encoder re-ranking
- Domain-specific medical embeddings
- OCR support for scanned healthcare documents
- RAGAS automated evaluation

## Project Structure

ingestion/
  loader.py      - Document loading + OCR
  chunker.py     - 4 chunking strategies
  embedder.py    - Domain-specific embeddings
  vector_store.py - Pinecone operations

query/           - Coming
generation/      - Coming
evaluation/      - Coming
api/             - FastAPI layer (coming)

## Setup

1. Clone repo
2. Create virtual environment: python -m venv venv
3. Activate: venv\Scripts\activate.bat
4. Install: pip install -r requirements.txt
5. Copy .env.example to .env and add your API keys
6. Run: python main.py

------Updated---------------

# Healthcare RAG Pipeline

![CI](https://github.com/AyushiAwasthi-AI/healthcare-rag-pipeline/actions/workflows/ci.yml/badge.svg)

A production-grade Retrieval-Augmented Generation system for clinical decision support. Processes WHO diabetes and hypertension guidelines to answer clinical queries with page-level citations.

---

## Evaluation Results

| Metric | Score | Meaning |
|--------|-------|---------|
| Faithfulness | **1.0000** | Zero hallucination — every claim grounded in retrieved context |
| Answer Relevancy | **0.9458** | Answers directly address the clinical question |
| Context Precision | **0.8978** | Retrieved chunks are highly relevant |
| Context Recall | **0.9000** | No critical clinical information missed |

Evaluated with RAGAS 0.2.6 on WHO diabetes guidelines. All metrics above 0.89 — production-ready threshold is 0.70.

---

## Architecture

```
User Query
    │
    ▼
FastAPI (POST /query)
    │
    ▼
QueryProcessor          ← normalize query, build HIPAA metadata filters
    │
    ▼
Retriever               ← embed with all-MiniLM-L6-v2, search Pinecone (top 10)
    │
    ▼
Reranker                ← cross-encoder ms-marco-MiniLM-L-6-v2 (top 10 → top 5)
    │
    ▼
Generator               ← clinical system prompt + numbered context → Groq Llama3
    │
    ▼
QueryResponse           ← answer + page-level citations + processing time
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI + uvicorn (async) |
| Embedding | sentence-transformers all-MiniLM-L6-v2 (384-dim) |
| Vector DB | Pinecone (cosine similarity, 1455 vectors) |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 3.1 8B via Groq API |
| Evaluation | RAGAS 0.2.6 |
| Validation | Pydantic v2 |
| Config | pydantic-settings |
| Containerisation | Docker + docker-compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```
healthcare-rag-pipeline/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI app — /health, /query, /ingest
├── ingestion/
│   ├── loader.py            # PDF loader with page-level extraction
│   ├── chunker.py           # Dynamic chunk sizing (512–1024 tokens)
│   ├── embedder.py          # Sentence-transformers, uuid5 chunk IDs
│   └── vector_store.py      # Pinecone upsert + similarity search
├── query/
│   ├── processor.py         # Query normalisation + metadata filters
│   ├── retriever.py         # Async embed + Pinecone search
│   ├── reranker.py          # Cross-encoder reranking
│   └── query_engine.py      # Pipeline orchestrator
├── generation/
│   └── generator.py         # Clinical prompt builder + Groq LLM call
├── evaluation/
│   ├── ragas_evaluator.py   # RAGAS 4-metric evaluation
│   └── test_dataset.py      # 3 ground-truth QA pairs
├── config.py                # pydantic-settings, .env loading
├── models.py                # Pydantic v2 request/response contracts
├── main.py                  # Ingestion orchestrator (async, concurrent)
├── run_evaluation.py        # RAGAS evaluation runner
├── Dockerfile               # CPU-only torch, python:3.10-slim
├── docker-compose.yml       # Service definition with healthcheck
├── .dockerignore            # Excludes venv/, .env, data/
└── requirements.txt         # Pinned LangChain-compatible versions
```

---

## Setup

### Prerequisites
- Python 3.10
- Pinecone account (free tier works)
- Groq API key (free, 6000 req/day)

### Installation

```bash
git clone https://github.com/AyushiAwasthi-AI/healthcare-rag-pipeline.git
cd healthcare-rag-pipeline
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configuration

Create `.env` at project root:

```
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=healthcare-rag
GROQ_API_KEY=your_key_here
ENVIRONMENT=development
```

### Ingest documents

```bash
# Place PDFs in data/documents/
python main.py
```

### Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

### Run with Docker

```bash
docker-compose up --build
```

---

## API Reference

### GET /health
Liveness probe for Docker, Kubernetes, Azure Container Apps.

```json
{
  "status": "healthy",
  "pinecone_connected": true,
  "embedding_model_loaded": true,
  "version": "1.0.0",
  "environment": "development"
}
```

### POST /query
Full RAG pipeline. Returns clinical answer with page-level citations.

**Request:**
```json
{
  "query": "What are the symptoms of Type 2 diabetes?",
  "patient_id": "P001",
  "max_results": 5,
  "include_sources": true
}
```

**Response:**
```json
{
  "answer": "According to the provided documents, symptoms include polyuria, thirst, blurred vision [1]...",
  "query": "What are the symptoms of Type 2 diabetes?",
  "sources": [
    {
      "chunk_id": "3aaf7ccf-...",
      "text": "Classic symptoms of polyuria...",
      "source": "BTN_D1_Diabetes Management guidelines.pdf",
      "score": 9.6695,
      "page_number": 23
    }
  ],
  "retrieved_chunks_count": 5,
  "model_used": "llama-3.1-8b-instant",
  "processing_time_ms": 1842.5
}
```

### POST /ingest
Ingest one document into the vector store.

**Request:**
```json
{
  "file_path": "/app/data/documents/guidelines.pdf",
  "document_type": "clinical_notes"
}
```

---

## Key Design Decisions

**Two-stage retrieval:** Bi-encoder (all-MiniLM-L6-v2) retrieves top 10 by cosine similarity in milliseconds. Cross-encoder (ms-marco-MiniLM-L-6-v2) reranks to top 5 with much higher accuracy. Fast retrieval at scale + precise ranking on candidates.

**Deterministic chunk IDs (uuid5):** Chunk IDs are derived from `source + page_number + chunk_index`. Re-ingesting the same document produces identical IDs — Pinecone upsert safely overwrites without duplicates. Critical for HIPAA right-to-erasure.

**Page-level citations:** Loader processes PDFs page by page, storing `page_number` in Pinecone metadata. Every answer cites the exact page from a 100+ page clinical document.

**Clinical system prompt:** LLM instructed to answer only from provided context, cite sources by number, and return a specific "insufficient information" response when context is inadequate. Prevents hallucination in a safety-critical system.

---

## Roadmap

### Phase 2 (September–October 2026)
- LangGraph clinical decision agent
- LangSmith observability and tracing
- Azure Container Apps deployment
- Kafka streaming ingestion for real-time FHIR feeds
- Airflow DAG for scheduled re-ingestion
- Hybrid BM25 + vector search

---

## Documents Ingested

5 WHO diabetes and hypertension guidelines (1455 chunks, 384-dimensional embeddings).

## 9. LangGraph Clinical Agent

Graph structure:
  [decide] → (needs_retrieval=True)  → [retrieve] → [generate] → END
  [decide] → (needs_retrieval=False) → [generate] → END

- Clinical query: needs_retrieval=True, full RAG pipeline runs
- General greeting: needs_retrieval=False, Pinecone skipped entirely
- Agent reuses existing QueryEngine and Generator — adds decision layer on top
- State (ClinicalAgentState TypedDict) carries data between all nodes
- route_after_decision() is the conditional edge function LangGraph calls

Interview answer: "The agent adds a reasoning step before retrieval.
A pipeline always retrieves — an agent decides first. For a clinical system
this means non-medical queries never hit Pinecone, reducing latency and cost.
The decision is transparent — the response includes the agent's reasoning string
so clinicians can audit why the system chose each path."

## Observability — LangSmith Trace

![LangSmith Trace](https://github.com/user-attachments/assets/93f99b63-8f94-4d3b-a91a-a043f8d15686)

## Clinical Safety Design Principles

**Zero hallucination policy:** RAGAS faithfulness score of 1.0 across evaluation
set. The LLM is instructed to respond with "The provided documents do not contain
sufficient information" rather than generating beyond retrieved context.

**Auditability:** Every response includes exact page numbers from source documents,
enabling clinicians to verify citations in the original guidelines. Required for
clinical AI deployment under FDA SaMD (Software as a Medical Device) guidelines.

**HIPAA-compliant error handling:** Error logs emit only exception type names,
never raw query content, which may contain Protected Health Information (PHI).

**Deterministic document IDs:** uuid5-based chunk IDs enable reliable document
deletion for HIPAA right-to-erasure compliance without orphaned vectors.
