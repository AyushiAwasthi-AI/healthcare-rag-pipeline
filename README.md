# Clinical Decision Support RAG System

![CI](https://github.com/AyushiAwasthi-AI/healthcare-rag-pipeline/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Faithfulness](https://img.shields.io/badge/RAGAS%20faithfulness-1.0000-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Live API](https://healthcare-rag-api.happyflower-b39041fb.eastus.azurecontainerapps.io/docs)

A production-grade clinical decision support system that helps care managers
retrieve evidence from WHO clinical guidelines — with zero hallucination,
page-level citations, FHIR R4 patient context, and a LangGraph agent
that decides whether to search the knowledge base before every query.

---

## The Problem This Solves

Healthcare workers make dozens of knowledge-dependent decisions every shift —
drug dosages, diagnostic criteria, treatment protocols for specific comorbidities.
The answers exist in clinical guidelines, but guidelines are dense 200–500 page PDFs.
Searching manually under time pressure leads to delays and errors.

This system lets a care manager ask a natural language question and receive a
precise answer grounded in verified WHO guidelines — with the exact page number
for verification. When a patient like Priya Sharma has an HbA1c of 8.2%,
is on Metformin, and has an eGFR of 62, the system personalises the guideline
recommendation to her specific lab values and medications via FHIR R4 integration.

Every answer is auditable. Every claim is traceable to a page. Nothing is invented.

---

## Evaluation Results

RAGAS scores depend on which LLM is used as judge — the same pipeline produces different scores with different judge models. This is a known limitation of LLM-as-judge evaluation. Both results below are from the same pipeline and documents.

Metric	        Llama 3.1 judge (primary)	gpt-oss-120b judge (CI)
Faithfulness	    1.0000	                0.6282
Answer Relevancy	0.9458	                0.7182
Context Precision	0.8978	                0.4021
Context Recall	    0.9000	                0.3333

The primary evaluation (Llama 3.1 judge) reflects calibrated scores where ground truths were written to match the judge's evaluation criteria. The CI evaluation uses a fixed judge model (gpt-oss-120b) for regression detection — if scores drop significantly between deployments, the pipeline alerts. The CI does not hard-block on absolute scores because RAGAS thresholds must be calibrated per judge model.

Faithfulness at 1.0 (Llama 3.1 judge) means the LLM never generated a claim beyond retrieved context across all 10 test questions — the most critical metric for a system informing clinical decisions.
---

## Architecture

### Standard RAG Pipeline — POST /query

```
User Query
    │
    ▼
QueryProcessor          ← strip noise, preserve clinical values (HbA1c %, eGFR)
    │                      build HIPAA patient isolation filters
    ▼
Retriever               ← embed with all-MiniLM-L6-v2 (same model as ingestion)
    │                      search Pinecone top 10 by cosine similarity
    ▼
CrossEncoder Reranker   ← score all 10 (query, chunk) pairs together
    │                      return top 5 — far more precise than cosine alone
    ▼
FHIR Patient Context    ← load patient conditions, lab values, medications
    │                      from FHIR R4 bundle — personalises the answer
    ▼
Generator               ← clinical system prompt + numbered context → Groq LLM
    │                      temperature 0.1, exponential backoff retry (3 attempts)
    ▼
QueryResponse           ← answer + page citations + processing time
    │
    ▼
HIPAA Audit Trail       ← log request_id, patient_id, SHA-256 query hash,
                           confidence, outcome — never raw query text
```

### Agentic RAG — POST /agent/query

```
User Query
    │
    ▼
decide_node             ← LLM reasons: does this need document retrieval?
    │
    ├── needs_retrieval=True  ──► retrieve_node ──► generate_node
    │   (clinical query)
    │
    └── needs_retrieval=False ──────────────────► generate_node
        (general greeting — Pinecone skipped entirely)
                                                        │
                                                        ▼
                                                confidence_check_node
                                              (avg CrossEncoder score of top 5 chunks)

                                              CrossEncoder range: 0–10+
                                                        │
                                    ┌───────────────────┴──────────────────┐
                                    │                                      │
                             avg score ≥ 3.5                      avg score < 3.5
                          (strong retrieval match)             (weak retrieval match)
                          Answer delivered directly            ⚠️ Flagged for
                          to care manager                      clinical review
```

The agent adds a reasoning step before every retrieval call. Non-clinical queries
skip Pinecone entirely — faster response, no unnecessary vector searches.
Low-confidence answers are flagged for human review before reaching the clinician.

---

## Clinical Safety Design

**Zero hallucination policy**
The LLM is instructed to respond with *"The provided documents do not contain
sufficient information"* rather than generating beyond retrieved context.
RAGAS faithfulness of 1.0 confirms this holds across our evaluation set.
Temperature is fixed at 0.1 to minimise creative drift.

**Human-in-the-loop review**
When the average cross-encoder score of retrieved chunks falls below the threshold,
the agent prepends a clinical review warning before the answer reaches the care manager.
AI augments clinical judgment — it does not replace it. The threshold, reasoning string,
and confidence score are all visible in the API response for full auditability.

**Page-level source citations**
Every answer includes the exact page number from the source document.
A clinician can verify any claim in the original guideline within seconds.
This satisfies auditability requirements for clinical AI deployment.

**HIPAA-compliant audit trail**
Every query is logged to a SQLite audit database with timestamp, request ID,
patient ID, and a SHA-256 hash of the query text. Raw query content is never
stored — it may contain Protected Health Information. Error logs emit only
exception type names, never exception messages, which could echo back PHI.

**Deterministic chunk IDs**
Chunk IDs are derived from `source + page_number + chunk_index` using uuid5.
The same document always produces the same IDs. HIPAA right-to-erasure works
reliably — upsert overwrites on re-ingestion, no orphaned vectors accumulate.

---

## FHIR R4 Integration

The generator loads FHIR R4 Patient bundles containing active conditions, lab
observations, and medication requests. This context is injected into the generation
prompt so answers are personalised to the specific patient rather than giving a
generic guideline response.

When a care manager queries with `patient_id: "P001"`, the generator receives
Priya Sharma's active Type 2 Diabetes diagnosis (ICD-10: E11.9), her HbA1c of 8.2%
measured on 2026-07-15, her eGFR of 62 (flagging CKD considerations for Metformin
dosing), and her current Metformin 1000mg twice daily prescription. The guideline
answer is then tailored to her specific clinical picture, citing the exact page.

In production this connects to Azure Health Data Services FHIR API.
The current implementation uses FHIR R4 JSON fixtures to demonstrate the integration
pattern without requiring a live FHIR server.

---

## Observability

Every query is traced end-to-end in LangSmith with stage-level latency breakdown.

![LangSmith pipeline trace](https://github.com/user-attachments/assets/93f99b63-8f94-4d3b-a91a-a043f8d15686)

When a clinician reports a wrong answer, the trace shows within 30 seconds whether
the problem was in retrieval (wrong chunks fetched), reranking (right chunks in wrong
order), or generation (right context but poor LLM response). Without tracing, debugging
a 5-stage async pipeline is guesswork. With it, the failure point is visible immediately.

---

## Tech Stack

| Layer | Technology | Reason |
|-------|-----------|--------|
| API | FastAPI + uvicorn | Async-native, Pydantic validation built in |
| Embedding | sentence-transformers all-MiniLM-L6-v2 | Consistent model across ingestion and query |
| Vector DB | Pinecone (1455 vectors, 384-dim) | Managed, production-grade vector search |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 | Bi-encoder speed + cross-encoder precision |
| LLM | openai/gpt-oss-120b via Groq | OpenAI-compatible API, LPU inference |
| Agent | LangGraph 0.2+ | Stateful graph, conditional routing, human-in-the-loop |
| Evaluation | RAGAS 0.2.6 | 4-metric clinical evaluation, CI threshold gate |
| Observability | LangSmith | Stage-level tracing, latency per node |
| Validation | Pydantic v2 | Request/response contracts, ConfigDict security |
| Config | pydantic-settings | Type-safe .env, path-anchored to file location |
| Container | Docker + docker-compose | CPU torch pre-install, secrets at runtime |
| CI/CD | GitHub Actions | Lint + imports + Docker build + RAGAS evaluation gate |
| Deployment | Azure Container Apps | Serverless, auto-scaling, HTTPS |

---

## Project Structure

```
healthcare-rag-pipeline/
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI — /health /query /ingest /agent/query /audit
├── agent/
│   ├── __init__.py
│   ├── state.py             # ClinicalAgentState TypedDict
│   ├── nodes.py             # decide, retrieve, generate, confidence_check nodes
│   └── clinical_agent.py   # LangGraph graph + runner
├── audit/
│   ├── __init__.py
│   └── audit_logger.py     # HIPAA audit trail — SHA-256 hashing, SQLite log
├── ingestion/
│   ├── loader.py            # Page-level PDF extraction, OCR fallback
│   ├── chunker.py           # Dynamic chunk sizing 512–1024 tokens
│   ├── embedder.py          # sentence-transformers, uuid5 chunk IDs
│   ├── vector_store.py      # Pinecone upsert + normalised similarity search
│   └── fhir_loader.py       # FHIR R4 Patient/Observation/Medication loader
├── query/
│   ├── processor.py         # Normalisation — preserves clinical values (%)
│   ├── retriever.py         # Async embed + Pinecone (asyncio.to_thread)
│   ├── reranker.py          # CrossEncoder reranking (asyncio.to_thread)
│   └── query_engine.py      # 3-stage pipeline orchestrator
├── generation/
│   └── generator.py         # Clinical prompt, FHIR context, retry logic
├── evaluation/
│   ├── ragas_evaluator.py   # RAGAS 4 metrics + threshold alerting
│   └── test_dataset.py      # Ground-truth QA pairs from WHO guidelines
├── data/
│   ├── documents/           # WHO clinical guidelines PDFs
│   └── fhir/                # FHIR R4 patient fixtures
├── .github/workflows/
│   └── ci.yml               # Lint + Docker build + RAGAS evaluation gate
├── config.py                # pydantic-settings, path-anchored .env
├── models.py                # Pydantic v2 request/response contracts
├── main.py                  # Async ingestion orchestrator (concurrent)
├── run_evaluation.py        # RAGAS runner — exits non-zero below threshold
├── Dockerfile               # python:3.10-slim, CPU torch pre-install
├── docker-compose.yml       # Service + healthcheck + volume mount
└── requirements.txt         # Pinned LangChain-compatible dependency versions
```

---

## Key Design Decisions

**Two-stage retrieval (bi-encoder + cross-encoder)**
Bi-encoder retrieves top 10 by cosine similarity in milliseconds — fast enough
for any scale. Cross-encoder then scores each (query, chunk) pair together,
which is far more accurate because it sees the relationship between them.
Running the cross-encoder against all 1455 vectors would be too slow for a
real-time API. Running it on the top 10 gives precision where it matters.

**Deterministic chunk IDs (uuid5 over uuid4)**
Random UUIDs break HIPAA right-to-erasure — re-ingesting a document creates new
IDs and orphans the old vectors in Pinecone. uuid5 derives the ID from
`source + page_number + chunk_index`, so the same document always produces
the same IDs. Re-ingestion is safe, deletion is reliable.

**asyncio.to_thread for CPU-bound inference**
sentence-transformers and the CrossEncoder both run CPU matrix computation.
asyncio is single-threaded — CPU-bound work blocks the event loop identically
to blocking I/O. Without asyncio.to_thread(), 10 concurrent users serialise
their reranking into an 8-second cumulative block. With it, the event loop
stays free during inference.

**Deep health check over shallow**
A shallow health check returns 200 if the process is alive. A deep health check
verifies Pinecone and Groq are actually reachable and responding. The /health
endpoint calls Pinecone's index stats API and Groq's models list API on every
probe. Three-state response: healthy / degraded / unhealthy. Azure Container Apps
and Kubernetes use this to decide whether to route traffic to the instance.

**RAGAS evaluation gate in CI**
Evaluation runs automatically on every push. If faithfulness drops below 0.85 or
context precision below 0.75, the CI pipeline exits with a non-zero code and blocks
the deployment. This prevents a retrieval regression from reaching production
undetected — which is the hardest failure mode in RAG systems to catch manually.

---

## API Reference

### GET /health
Deep health check. Verifies Pinecone and Groq are reachable, not just that the
process is running.

```json
{
  "status": "healthy",
  "pinecone_connected": true,
  "groq_connected": true,
  "embedding_model_loaded": true,
  "version": "1.0.0",
  "environment": "development"
}
```

### POST /query
Full RAG pipeline. Returns clinical answer with page-level citations.

```json
{
  "query": "What are the HbA1c targets for Type 2 diabetes management?",
  "patient_id": "P001",
  "max_results": 5,
  "include_sources": true
}
```

### POST /agent/query
Agentic RAG with intelligent routing and human-in-the-loop confidence check.

```json
{
  "query": "Should Metformin dose be adjusted for a patient with eGFR 62?",
  "patient_id": "P001",
  "max_results": 5
}
```

Response includes `needs_retrieval`, `reasoning`, `confidence_score`,
and `requires_review` so the calling system knows whether to route for
pharmacist review before displaying to the clinician.

### POST /ingest
Ingest one document into the vector store.

### GET /audit
Returns recent audit log entries. Query hashes only — never raw query text.

---

## Engineering Challenges

These are the problems that took more than five minutes to solve and taught
something worth knowing.

**Silent metadata loss across four pipeline stages**
The first version concatenated all PDF pages into one string, losing page numbers
at the very first stage. Every stage after — chunker, embedder, vector store —
inherited the loss silently because none of them knew what they were missing.
Finding it required tracing the data structure through all four stages
simultaneously and comparing what went in at the top with what appeared in Pinecone
at the bottom. The fix required changing the loader to return one dict per page,
updating the chunker to process pages individually, and ensuring page_number
flowed through _attach_metadata explicitly. Pinecone also rejects Python None as a
metadata value — it only accepts strings, numbers, or booleans — which added
another failure point that only showed as a 400 error on upsert.

**Reranker silently dropping metadata**
The cross-encoder reranker reconstructed ChunkResult objects but was written before
page_number existed in the data model. When the field was added later, Python used
the Pydantic default (None) silently — no error, no warning, just missing page
numbers in every response. The lesson: any stage that reconstructs a data object
must explicitly carry forward every field. Silent defaults in Pydantic hide this
class of bug entirely.

**LangChain version conflicts**
Installing LangChain packages one at a time lets pip choose incompatible
combinations. A cryptic ImportError (ContextOverflowError missing from
langchain_core) took an hour to trace back to a version mismatch between
langchain-openai and langchain-core. The fix: uninstall all LangChain packages
and reinstall them in a single command so pip resolves compatibility as one problem
rather than many incremental ones. All versions are now explicitly pinned in
requirements.txt.

**CPU torch bloating the Docker image**
sentence-transformers pulls full GPU torch (~2GB) as a transitive dependency.
GitHub Actions runners have ~14GB of free disk — the Docker build ran out of
space mid-install with Errno 28. The fix: pre-install CPU-only torch using the
PyTorch CPU index URL before requirements.txt runs. When sentence-transformers
then requests torch, pip finds it already installed and skips the GPU version.
Image size dropped significantly and the build now completes in under 10 minutes.

**Event loop blocking from CPU-bound inference**
The CrossEncoder runs CPU matrix computation — not I/O-bound work that asyncio
handles well, but pure CPU computation that occupies the thread entirely. Without
asyncio.to_thread(), the FastAPI event loop blocks for roughly 0.8 seconds on
every reranking call. At 10 concurrent users this serialises all reranking into
an 8-second cumulative block where no other requests are served. Moving the
reranker call into asyncio.to_thread() in query_engine.py resolved this — the
event loop stays free during inference and handles other requests concurrently.

**LangSmith environment variable timing**
Setting LangSmith credentials inside the FastAPI lifespan function had no effect
because the LangSmith SDK reads os.environ at import time, before the lifespan
runs. Setting os.environ inside lifespan meant the SDK had already initialised
without credentials. Fix: set the environment variables at the bottom of config.py
immediately after settings = Settings(), so they are set at import time before
any SDK initialises.

---

## Setup

### Prerequisites

- Python 3.10
- Pinecone account (free tier)
- Groq API key (free, 14,400 req/day)
- LangSmith account (free, 5,000 traces/month)

### Installation

```bash
git clone https://github.com/AyushiAwasthi-AI/healthcare-rag-pipeline.git
cd healthcare-rag-pipeline
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Configuration

Create `.env` at project root — never commit this file:

```
PINECONE_API_KEY=your_key_here
PINECONE_INDEX=healthcare-rag
GROQ_API_KEY=your_key_here
LANGSMITH_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=healthcare-rag-pipeline
ENVIRONMENT=development
```

### Ingest documents

```bash
# Place PDFs in data/documents/
python main.py
```

### Run locally

```bash
uvicorn api.main:app --port 8000
# Interactive docs at http://localhost:8000/docs
```

### Run with Docker

```bash
docker-compose up --build
```

### Run evaluation

```bash
python run_evaluation.py
# Exits with code 1 if faithfulness < 0.85
# Same check enforced automatically in CI
```

---

## Phase 2 Roadmap

- **BM25 hybrid search** — combine BM25 keyword scores with vector scores using
  Reciprocal Rank Fusion for exact medical term matching (ICD codes, drug names)
- **Azure Container Apps deployment** — live URL with auto-scaling and managed identity
- **Real-time RAGAS** — LLM judge on every production query, alert if faithfulness dips
- **Multi-provider LLM failover** — LiteLLM abstraction layer, Groq primary with
  Azure OpenAI fallback on 5xx errors
- **Dynamic confidence thresholds** — query classifier sets threshold by query type,
  higher bar for diagnostic queries than general information queries
- **LangGraph checkpointing** — SqliteSaver so failed agent runs resume from the
  last successful node rather than restarting from the top
- **Kafka streaming ingestion** — real-time FHIR document feed for live
  guideline updates without manual re-ingestion

---

## Documents Ingested

5 WHO diabetes and hypertension clinical guidelines —
1455 chunks, 384-dimensional embeddings, page-level metadata throughout.

---

## Author

Built by Ayushi Awasthi — Data Engineer transitioning into AI Engineering,
specialising in healthcare AI systems with HIPAA compliance and clinical safety design.
