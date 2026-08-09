# Healthcare RAG Pipeline — Interview Notes
## Complete reference: bugs, decisions, concepts, and expected questions

---

## 1. Production Bugs Fixed (every one is an interview story)

### Bug 1: `datetime.utcoffset()` instead of `datetime.utcnow()`
**File:** `main.py`
**Symptom:** Pipeline crashed immediately on start.
**Root cause:** `utcoffset()` is a method on timedelta objects, not datetime. Silent naming similarity.
**Fix:** `datetime.utcnow()`
**Interview answer:** "This is why integration tests matter more than unit tests. Unit tests mock the clock. An integration test that actually runs the pipeline catches this class of error immediately."

---

### Bug 2: Embedder and VectorStore re-instantiated inside the ingestion loop
**File:** `main.py` (original version)
**Symptom:** Every document reloaded the embedding model from disk — 5x slower ingestion.
**Root cause:** `embedder = DocumentEmbedder()` was inside `run_ingestion()` instead of created once and passed in as parameters.
**Fix:** Singleton pattern — create once in `__main__`, pass as arguments.
**Interview answer:** "Model loading takes 2–10 seconds. In a loop over 5 documents that's 10–50 seconds of wasted time. The fix is the singleton pattern: create expensive objects once, share them across all work units."

---

### Bug 3: `asyncio.run()` inside an async function
**Symptom:** `RuntimeError: This event loop is already running`
**Root cause:** `asyncio.run()` creates a new event loop. Calling it inside an existing async context crashes because Python cannot nest event loops.
**Fix:** Use `await` inside async functions. Only call `asyncio.run()` at the top level of a standalone script.
**Interview answer:** "asyncio.run() is the entry point — it creates and owns the event loop. Once inside async def, the loop already exists. You signal 'pause here' with await, not by creating another loop. In FastAPI, uvicorn owns the loop and you never call asyncio.run() at all."

---

### Bug 4: `config.py` relative `.env` path
**Symptom:** `ValidationError: pinecone_api_key Field required` despite `.env` existing.
**Root cause:** `env_file=".env"` resolves relative to the current working directory, not the project root.
**Fix:** `env_file=Path(__file__).resolve().parent / ".env"`
**Interview answer:** "Relative paths are a deployment anti-pattern. The fix anchors the path to the file's own location using `__file__`. This works identically on local dev, Docker, and Azure Container Apps regardless of working directory."

---

### Bug 5: Pinecone SDK returns objects, not dicts
**File:** `ingestion/vector_store.py` → `query/retriever.py`
**Symptom:** `AttributeError: 'list' object has no attribute 'id'`
**Root cause:** Pinecone SDK version changed internal response types across versions.
**Fix:** Normalize all Pinecone responses to plain dicts inside `vector_store.py`. Retriever never touches SDK objects directly.
**Interview answer:** "This is a coupling problem. The retriever was coupled to Pinecone's internal SDK format. The normalization layer in vector_store.py decouples the retriever from Pinecone versioning. When Pinecone ships a breaking change, only one file changes."

---

### Bug 6: Pinecone rejects Python `None` metadata values
**Symptom:** `ApiError: [400] Metadata value must be a string, number, boolean or list of strings, got 'null'`
**Root cause:** `chunk.get("section_header")` returns Python `None` when key missing. JSON serializes this as `null`. Pinecone's metadata store only accepts strings, numbers, booleans, or lists.
**Fix:** `chunk.get("section_header") or ""` for strings, `chunk.get("page_number") or 0` for integers.
**Interview answer:** "Pinecone metadata is a flat key-value store, not a document store. It has strict type constraints. The defensive pattern is always provide type-safe defaults: empty string for strings, 0 for integers. Never insert None."

---

### Bug 7: All PDF pages concatenated — page numbers lost
**File:** `ingestion/loader.py`
**Symptom:** Citations showed `None` for page number.
**Root cause:** Original loader concatenated all page text into one string with `text += page_text`. Page-level metadata was lost at the first processing stage.
**Fix:** Loader returns a list of dicts — one per page — each with `page_number`. Chunker processes one page at a time and inherits `page_number` through `_attach_metadata`.
**Interview answer:** "This is a data lineage problem. Metadata that exists at the source (PDF page numbers) was discarded at the first transformation. The fix requires tracing metadata through every stage: load → chunk → embed → store. Each stage must carry metadata forward explicitly."

---

### Bug 8: Reranker silently drops `page_number`
**File:** `query/reranker.py`
**Root cause:** Reranker reconstructs ChunkResult objects but was written before `page_number` field existed. The field was added to the model later but never added to the reranker's constructor call. Python uses the default value (None) silently.
**Fix:** Add `page_number=chunk.page_number` and `section_header=chunk.section_header` to reranker's ChunkResult constructor.
**Interview answer:** "Metadata fields added to data models must be explicitly carried through every stage that reconstructs objects. This is why code review matters — the field was in the model, in the storage, and in the retriever, but the reranker was a silent gap. Silent defaults in Pydantic hide this class of bug."

---

### Bug 9: `requirements.txt` out of sync with installed packages
**Symptom:** GitHub Actions CI failed — `ModuleNotFoundError` for packages installed locally but not in requirements.
**Root cause:** Packages installed manually with `pip install X` locally were never added to `requirements.txt`. CI installs only what's listed.
**Fix:** Audited every import in the codebase and added all missing packages.
**Interview answer:** "requirements.txt must be the source of truth, not an afterthought. In production, the CI environment has nothing except what's in requirements.txt. This is exactly what CI is for — catching the gap between local dev and a clean environment."

---

### Bug 10: LangChain version conflicts
**Symptom:** `ImportError: cannot import name 'ContextOverflowError' from 'langchain_core'`
**Root cause:** LangChain releases many tightly-coupled packages (langchain-core, langchain-community, langchain-openai, etc.) that must all be at compatible versions. Installing them one at a time lets pip choose incompatible combinations.
**Fix:** Uninstall all LangChain packages, reinstall as one pinned set in a single command.
**Interview answer:** "LangChain's package ecosystem is fragile — version mismatches cause cryptic import errors. The pattern for any tightly-coupled package family is to pin all versions explicitly and install them together so pip can resolve compatibility in one pass."

---

### Bug 11: Docker build fails — no space left on device
**Symptom:** `OSError: [Errno 28] No space left on device` in GitHub Actions
**Root cause:** `sentence-transformers` depends on `torch`. Without specifying otherwise, pip installs full GPU torch (~2GB) including all nvidia-* CUDA packages. GitHub Actions runner has ~14GB free disk.
**Fix:** Install CPU-only torch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. When sentence-transformers then requests torch, pip finds it already installed and skips the GPU version.
**Interview answer:** "ML dependencies have hidden size traps. sentence-transformers pulls in full CUDA torch by default. In a container running on CPU-only infrastructure, you need the CPU wheel — ~250MB instead of ~2GB. Always pre-install torch with the CPU index URL before other ML packages."

---

### Bug 12: HuggingFace free serverless inference removed support for most models (2025)
**Symptom:** `model_not_supported` error for Mistral-7B and Phi-3
**Root cause:** HuggingFace changed their free tier in 2025 — most popular models now route through paid providers (Featherless AI).
**Fix:** Switched to Groq free tier — 6000 req/day, OpenAI-compatible API, supports Llama 3.1 and Mixtral. Zero code change beyond client initialization.
**Interview answer:** "External API dependencies in production are risks. When HuggingFace broke their free tier, I evaluated alternatives and chose Groq — it uses the OpenAI-compatible API format, so switching required changing only the client initialization, not any pipeline logic."

---

## 2. Architecture Decisions

### Two-stage retrieval: bi-encoder + cross-encoder
**Why not use cross-encoder for everything?**
Cross-encoder looks at query + chunk together — more accurate but O(n) comparisons. Against 1455 vectors = too slow for a real-time API.
Bi-encoder embeds query and chunks separately — cosine similarity search in milliseconds at any scale.
**Pattern:** Retrieve top 10 fast with bi-encoder → rerank to top 5 with cross-encoder.
**Interview answer:** "Bi-encoder gives speed at scale. Cross-encoder gives precision on a small candidate set. Two-stage retrieval gives you both. This is the standard production RAG architecture."

---

### Deterministic chunk IDs (uuid5 over uuid4)
**Why not random UUIDs?**
HIPAA right-to-erasure: if a patient requests deletion, you must find and delete all their chunks. With uuid4 (random), re-ingesting creates new IDs — old vectors are orphaned in Pinecone, impossible to identify.
With uuid5 (deterministic from `source + page_number + chunk_index`), same document always produces same IDs. Upsert safely overwrites. Deletion is reliable.
**Interview answer:** "HIPAA compliance requires reliable deletion. Deterministic IDs make re-ingestion idempotent — same input, same output. Random IDs make deletion unreliable, which is a critical HIPAA risk."

---

### `asyncio.to_thread()` for sentence-transformers
**Why not just `await model.encode()`?**
`model.encode()` is CPU-bound matrix computation. asyncio is single-threaded — CPU-bound work blocks the event loop identically to blocking I/O. A blocked event loop means no other requests are served.
`asyncio.to_thread()` offloads CPU-bound work to a thread pool, returning control to the event loop immediately.
**Interview answer:** "asyncio cannot help with CPU-bound work — the GIL still blocks. `asyncio.to_thread()` pushes CPU work to a thread pool executor, keeping the event loop free. The distinction: await is for I/O-bound waiting, to_thread is for CPU-bound computing."

---

### `extra="forbid"` in Pydantic request models
**Why reject unknown fields?**
In a healthcare API, unexpected fields in a request could be injection attempts or indicate a client bug. Silent acceptance hides bugs. Explicit rejection surfaces them with a 422 immediately.
**Interview answer:** "In healthcare systems, unexpected data is a red flag. `extra='forbid'` means any unknown field in a POST body returns 422 rather than being silently ignored. This is both a security and correctness decision."

---

### `Path(__file__).resolve().parent` for all config paths
**Used in:** `config.py`, `main.py`
**Why:** Relative paths break in Docker, CI, and Azure Container Apps where the working directory is different from the project root.
**Interview answer:** "File path resolution is a common source of environment-specific bugs. Anchoring to `__file__` makes the code portable — it resolves relative to the file's own location regardless of where Python was invoked from."

---

### CPU-only torch in Docker
**Pattern:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` before `pip install -r requirements.txt`
**Why:** sentence-transformers pulls full GPU torch (~2GB) by default. Healthcare RAG inference runs on CPU. Installing CPU wheel first prevents pip from downloading CUDA packages.
**Interview answer:** "Container images should be minimal. For a CPU inference workload, full CUDA torch adds 1.7GB of unnecessary weight and causes disk issues in CI. Pre-installing the CPU wheel is a standard optimization for ML containers."

---

## 3. Key Technical Concepts for Interviews

### asyncio in 4 lines
- `async def` — function can pause while waiting
- `await` — pause here, let others run, come back with result
- `asyncio.gather()` — start multiple tasks simultaneously, wait for slowest
- `asyncio.run()` — entry point that starts the event loop (scripts only, never in FastAPI)

### Why async matters for FastAPI
Sync endpoint: server blocks during LLM call (~1500ms). 10 users → last user waits 15 seconds.
Async endpoint: server handles other users during LLM wait. 10 users → all wait ~1500ms.

### RAG pipeline — full interview answer
"The pipeline has 5 stages. QueryProcessor normalizes the input and builds Pinecone metadata filters for patient isolation. Retriever embeds the query using the same model as ingestion — mixing models breaks cosine similarity — then searches Pinecone for top 10. Reranker runs a cross-encoder on each query+chunk pair to rerank top 10 down to top 5 with higher precision. Generator builds a numbered context block with page citations and calls Llama 3.1 via Groq with a clinical system prompt that enforces answer-only-from-context. Response includes answer, page-level source citations, processing time, and token count."

### RAGAS metrics — full interview answer
"I evaluate with 4 RAGAS metrics. Faithfulness measures what fraction of claims in the answer are supported by the retrieved context — it catches hallucination. Answer relevancy measures whether the generated answer actually addresses the question asked. Context precision measures what fraction of retrieved chunks were actually relevant — it catches retrieval noise. Context recall measures whether all the information needed to answer correctly was retrieved — requires ground truth answers. Our first run: faithfulness 1.0, answer relevancy 0.946, context precision 0.898, context recall 0.900. Faithfulness at 1.0 is the critical metric for a HIPAA-compliant clinical system."

### HIPAA-relevant engineering decisions in this project
- Deterministic uuid5 chunk IDs — reliable deletion for right-to-erasure
- `extra="forbid"` — reject unexpected PHI fields at the API boundary
- Patient ID metadata filters in Pinecone — query isolation per patient
- Page-level citations — clinical auditability, every claim traceable to source
- Clinical system prompt — LLM prohibited from answering beyond retrieved context
- `.env` excluded from Docker image — secrets injected at runtime via env_file

### Docker — interview answer
"Docker packages the application, runtime, and all dependencies into a single immutable image. For a healthcare RAG system: the sentence-transformers and LangChain dependency tree is fragile — exact versions must match across dev, CI, and production or inference breaks silently. Docker guarantees identical behavior everywhere. Azure Container Apps and AKS both consume Docker images as the deployment unit. Environment variables like API keys are injected at runtime via env_file, never baked into the image — satisfying HIPAA's requirement that secrets are not stored in code or artifacts."

---

## 4. RAGAS Evaluation Results

```
faithfulness:      1.0000  ← no hallucination, critical for healthcare
answer_relevancy:  0.9458  ← answers directly address the clinical question
context_precision: 0.8978  ← retrieved chunks are highly relevant
context_recall:    0.9000  ← pipeline finds most needed clinical information
```

Test set: 3 questions on WHO diabetes guidelines
Model: llama-3.1-8b-instant via Groq
Embedding: all-MiniLM-L6-v2 (384-dim)
Evaluation: RAGAS 0.2.6

---

## 5. Questions You Will Be Asked

**"Why Pinecone over pgvector?"**
Pinecone for managed vector similarity search at scale — no infrastructure to maintain. pgvector for relational data, audit logs, and document metadata where SQL joins are needed. They serve different purposes. In Phase 2 the design adds pgvector for audit logging alongside Pinecone for search.

**"How do you handle HIPAA right-to-erasure?"**
uuid5 IDs are deterministic — same source + page + chunk_index always produces the same chunk ID. To delete a patient's documents, call `index.delete(filter={"source": patient_file})`. Re-ingestion after deletion safely overwrites without duplicates because IDs are deterministic.

**"What happens if the LLM hallucinates?"**
The clinical system prompt instructs the model to answer only from the numbered context block. Temperature is 0.1 to minimize creativity. If the context doesn't contain sufficient information, the model is instructed to return "The provided documents do not contain sufficient information" rather than guessing. RAGAS faithfulness score of 1.0 confirms this is working across our test set.

**"What is the difference between bi-encoder and cross-encoder?"**
Bi-encoder embeds query and document independently and compares with cosine similarity — fast, scales to any number of vectors, less precise. Cross-encoder reads query and document together in one forward pass — much more accurate because it sees the relationship between them, but O(n) complexity makes it infeasible at scale. Production pattern: bi-encoder retrieves candidates, cross-encoder reranks them.

**"Why does your embedder use `asyncio.to_thread()`?"**
sentence-transformers.encode() is CPU-bound matrix computation. asyncio is single-threaded — CPU-bound work blocks the event loop the same as synchronous I/O. `asyncio.to_thread()` offloads the CPU work to a thread pool executor, keeping the event loop free to handle other requests while inference runs.

**"Walk me through what happens when I call POST /query"**
Request hits FastAPI. Pydantic validates the request body — wrong type or unknown field returns 422 immediately. QueryProcessor strips whitespace and builds a Pinecone metadata filter if patient_id is present. Retriever embeds the query using all-MiniLM-L6-v2 in a thread pool (CPU-bound), then queries Pinecone for top 10 cosine matches. Reranker scores all 10 (query, chunk) pairs with a cross-encoder and returns top 5 sorted by score. Generator builds a numbered context block with source filenames and page numbers, prepends the clinical system prompt, calls Groq API with temperature 0.1, and returns the answer. FastAPI serializes the QueryResponse with answer, chunks, page citations, model used, and processing time.

**"How would you scale this to 10,000 concurrent users?"**
Horizontal scaling behind a load balancer — each container is stateless (all state in Pinecone and the LLM provider). The embedding model and cross-encoder are the bottleneck since they're CPU-bound. In production I'd separate the embedding service and run it on a dedicated instance with asyncio.to_thread workers. The LLM call scales naturally since it's an external API call that the event loop handles concurrently.

**"What would you add in Phase 2?"**
LangGraph clinical decision agent with tools for guideline lookup and drug interaction checking. LangSmith for production observability — trace every request with retrieval scores and token counts. Kafka streaming ingestion for real-time FHIR document feeds. Hybrid BM25 + vector search for better keyword matching on medical terminology. Azure Container Apps for production deployment with auto-scaling.

---

## 6. Commit History Pattern

```
feat: add ingestion pipeline with page-level PDF processing
feat: add config.py with pydantic-settings
feat: add models.py — Pydantic v2 request/response contracts
feat: add 3-stage query pipeline with cross-encoder reranking
fix: reranker now carries page_number through ChunkResult reconstruction
feat: add generation layer with clinical prompt and Groq integration
feat: add page-level citations to generation output
feat: add RAGAS evaluation — faithfulness 1.0, all metrics above 0.89
feat: add FastAPI layer — /health, /query, /ingest endpoints
chore: add Dockerfile, docker-compose.yml and .dockerignore
ci: add GitHub Actions workflow for import validation and linting
ci: add Docker image build validation to CI pipeline
fix: use CPU-only torch in Docker, free disk space in CI
```

This commit history tells a story. Each commit is a working, testable increment. No "WIP" commits. No "fix bug" commits without context. This is what production git history looks like.

---

## 7. What This Project Demonstrates to an Interviewer

| Skill | Evidence |
|-------|---------|
| Production RAG | End-to-end pipeline, not a tutorial |
| HIPAA awareness | Deterministic IDs, patient filtering, clinical prompt guardrails |
| Async Python | asyncio.gather, to_thread, FastAPI lifespan pattern |
| Pydantic v2 | BaseModel, ConfigDict, field_validator, pydantic-settings |
| Evaluation discipline | RAGAS scores on record, not just "it works" |
| Docker | Multi-stage awareness, CPU torch optimization, secrets via env_file |
| CI/CD | GitHub Actions with lint, import validation, Docker build |
| Debugging methodology | Diagnostic scripts to isolate failures, not random trial and error |
| System design thinking | Singleton pattern, normalization layers, defensive metadata handling |
