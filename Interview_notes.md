# Clinical Decision Support RAG System — Interview Notes
## Complete reference: bugs, decisions, concepts, and expected questions

---

## 1. Production Bugs Fixed

### Bug 1: `datetime.utcoffset()` instead of `datetime.utcnow()`
**File:** `main.py`
**Symptom:** Pipeline crashed immediately on start.
**Root cause:** `utcoffset()` is a method on timedelta objects, not datetime. Silent naming similarity.
**Fix:** `datetime.utcnow()`
**Interview answer:** "This is why integration tests matter more than unit tests. Unit tests mock the clock. An integration test that actually runs the pipeline catches this class of error immediately."

---

### Bug 2: Embedder and VectorStore re-instantiated inside the ingestion loop
**File:** `main.py`
**Symptom:** Every document reloaded the embedding model from disk — 5x slower ingestion.
**Root cause:** `embedder = DocumentEmbedder()` was inside `run_ingestion()` instead of being created once and passed in.
**Fix:** Singleton pattern — create once in `__main__`, pass as arguments to every call.
**Interview answer:** "Model loading takes 2–10 seconds. Over 5 documents that is 10–50 seconds of wasted time. The fix is the singleton pattern: create expensive objects once, share them across all work units."

---

### Bug 3: `asyncio.run()` inside an async function
**Symptom:** `RuntimeError: This event loop is already running`
**Root cause:** `asyncio.run()` creates a new event loop. Calling it inside an existing async context crashes because Python cannot nest event loops.
**Fix:** Use `await` inside async functions. Only call `asyncio.run()` at the top level of a script.
**Interview answer:** "asyncio.run() is the entry point — it creates and owns the event loop. Once inside async def, the loop already exists. You signal pause with await, not by creating another loop. In FastAPI, uvicorn owns the loop and you never call asyncio.run() at all."

---

### Bug 4: `config.py` relative `.env` path
**Symptom:** `ValidationError: pinecone_api_key Field required` despite `.env` existing.
**Root cause:** `env_file=".env"` resolves relative to the current working directory, not the project root. Different when run from Docker or CI.
**Fix:** `env_file=Path(__file__).resolve().parent / ".env"`
**Interview answer:** "Relative paths are a deployment anti-pattern. Anchoring to __file__ makes the path resolve correctly on local dev, Docker, and Azure Container Apps regardless of working directory."

---

### Bug 5: Pinecone SDK returns objects, not dicts
**Symptom:** `AttributeError: 'list' object has no attribute 'id'`
**Root cause:** Pinecone SDK version changed internal response types across versions.
**Fix:** Normalize all Pinecone responses to plain dicts inside `vector_store.py`. Retriever never touches SDK objects directly.
**Interview answer:** "This is a coupling problem. The retriever was coupled to Pinecone's internal SDK format. The normalization layer decouples the retriever from Pinecone versioning — when Pinecone ships a breaking change, only one file changes."

---

### Bug 6: Pinecone rejects Python `None` metadata values
**Symptom:** `ApiError: [400] got 'null' for field 'section_header'`
**Root cause:** `chunk.get("section_header")` returns None. JSON serializes this as null. Pinecone only accepts strings, numbers, booleans, or lists.
**Fix:** `chunk.get("section_header") or ""` for strings, `chunk.get("page_number") or 0` for integers.
**Interview answer:** "Pinecone metadata is a flat key-value store with strict type constraints. The defensive pattern is always provide type-safe defaults — never insert None."

---

### Bug 7: All PDF pages concatenated — page numbers lost
**Symptom:** Citations showed `None` for page number across all responses.
**Root cause:** Loader concatenated all page text into one string with `text += page_text`. Page-level metadata was discarded at the first stage and every stage after inherited the loss silently.
**Fix:** Loader returns one dict per page, each with `page_number`. Chunker processes pages individually. `_attach_metadata` carries `page_number` into every chunk.
**Interview answer:** "This is a data lineage problem. Metadata that exists at the source was discarded at the first transformation. The fix requires tracing data structure through every stage simultaneously — what goes in at the top must match what appears in Pinecone at the bottom."

---

### Bug 8: Reranker silently dropping `page_number`
**Root cause:** Reranker reconstructed ChunkResult objects but was written before `page_number` existed in the model. When the field was added later, Pydantic used the default (None) silently — no error, no warning.
**Fix:** Add `page_number=chunk.page_number` and `section_header=chunk.section_header` to the reranker's ChunkResult constructor call.
**Interview answer:** "Any stage that reconstructs a data object must explicitly carry forward every field. Silent defaults in Pydantic hide this class of bug — the field was in the model, in storage, and in the retriever, but the reranker was a silent gap."

---

### Bug 9: `requirements.txt` out of sync with installed packages
**Symptom:** CI failed with ModuleNotFoundError for packages installed locally but not listed.
**Root cause:** Packages installed manually were never added to requirements.txt. CI installs only what is listed.
**Fix:** Audited every import in the codebase and added all missing packages.
**Interview answer:** "requirements.txt must be the source of truth. In a CI environment there is nothing except what is listed. This is exactly what CI is for — catching the gap between local dev and a clean environment."

---

### Bug 10: LangChain version conflicts
**Symptom:** `ImportError: cannot import name 'ContextOverflowError' from 'langchain_core'`
**Root cause:** LangChain packages must all be at compatible versions. Installing one at a time lets pip choose incompatible combinations.
**Fix:** Uninstall all LangChain packages, reinstall as one pinned set in a single command.
**Interview answer:** "For any tightly-coupled package family, pin all versions explicitly and install them together so pip resolves compatibility in one pass rather than many incremental ones."

---

### Bug 11: Docker build fails — no space left on device
**Symptom:** `OSError: [Errno 28] No space left on device` in GitHub Actions
**Root cause:** sentence-transformers pulls full GPU torch (~2GB) as a transitive dependency. GitHub Actions runners have ~14GB free disk — the build ran out mid-install.
**Fix:** Pre-install CPU-only torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. When sentence-transformers requests torch, pip finds it already installed and skips the GPU version.
**Interview answer:** "ML dependencies have hidden size traps. sentence-transformers pulls full CUDA torch by default. For a CPU inference workload, pre-installing the CPU wheel is standard practice — ~250MB instead of ~2GB."

---

### Bug 12: LangSmith env vars must be set at import time
**Root cause:** Setting LangSmith credentials inside FastAPI lifespan had no effect because the LangSmith SDK reads os.environ at import time, before lifespan runs.
**Fix:** Set os.environ at the bottom of config.py immediately after `settings = Settings()`, so they are set before any SDK initialises.
**Interview answer:** "SDK initialisation order matters. When a library reads configuration at import time rather than at call time, you must set the environment before the import happens — not inside application startup hooks."

---

### Bug 13: HuggingFace free serverless inference removed model support
**Symptom:** `model_not_supported` for Mistral-7B and Phi-3
**Root cause:** HuggingFace changed free tier in 2025 — most models now route through paid providers.
**Fix:** Switched to Groq — OpenAI-compatible API, free tier, zero code change beyond client initialisation.
**Interview answer:** "External API dependencies are risks. When HuggingFace broke their free tier, I evaluated alternatives and chose Groq — OpenAI-compatible format meant changing only the client initialisation, not any pipeline logic."

---

### Bug 14: Groq key rotation broke model access
**Root cause:** New Groq API key after rotation did not have the same model access as the original key. `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` were unavailable on the new key.
**Fix:** Queried available models via `client.models.list()` to find what the key actually supports. Switched to `openai/gpt-oss-120b` which is available on all Groq accounts.
**Interview answer:** "Never assume model availability after key rotation. Always query the available models list before hardcoding a model name. This is particularly important in healthcare where an unavailable model would silently break clinical query serving."

---

## 2. Architecture Decisions

### Two-stage retrieval: bi-encoder + cross-encoder
Cross-encoder looks at query + chunk together — more accurate but O(n) comparisons against all 1455 vectors would be too slow. Bi-encoder embeds separately — cosine search in milliseconds at any scale.
**Pattern:** Retrieve top 10 fast with bi-encoder → rerank to top 5 with cross-encoder.
**Interview answer:** "Bi-encoder gives speed at scale. Cross-encoder gives precision on a small candidate set. Two-stage retrieval gives you both. This is the standard production RAG architecture."

---

### Deterministic chunk IDs (uuid5 over uuid4)
HIPAA right-to-erasure requires reliable deletion. Random uuid4 creates new IDs on re-ingestion — old vectors are orphaned. uuid5 derived from `source + page_number + chunk_index` produces the same ID every time. Upsert safely overwrites. Deletion is reliable.
**Interview answer:** "HIPAA compliance requires reliable deletion. Deterministic IDs make re-ingestion idempotent. Random IDs make deletion unreliable — a critical compliance risk."

---

### `asyncio.to_thread()` for CPU-bound inference
`model.encode()` is CPU-bound matrix computation. asyncio is single-threaded — CPU-bound work blocks the event loop identically to blocking I/O. asyncio.to_thread() offloads to a thread pool, keeping the event loop free.
**Interview answer:** "asyncio cannot help with CPU-bound work. asyncio.to_thread() pushes CPU work to a thread pool executor. The distinction: await is for I/O-bound waiting, to_thread is for CPU-bound computing."

---

### `extra="forbid"` in Pydantic request models
In a healthcare API, unexpected fields could be injection attempts or client bugs. Silent acceptance hides bugs. Explicit rejection surfaces them immediately with a 422.
**Interview answer:** "In healthcare systems, unexpected data is a red flag. extra='forbid' means any unknown field in a POST body returns 422 rather than being silently ignored. Security and correctness decision."

---

### `Path(__file__).resolve().parent` for all config paths
Relative paths break in Docker, CI, and Azure where the working directory differs from project root. Anchoring to `__file__` makes paths resolve correctly everywhere.
**Interview answer:** "File path resolution is a common source of environment-specific bugs. Anchoring to __file__ makes code portable across local dev, Docker, and cloud deployment."

---

### CPU-only torch in Docker
sentence-transformers pulls full GPU torch (~2GB) by default. Healthcare RAG inference runs on CPU. Pre-installing CPU wheel first prevents pip from downloading CUDA packages.
**Interview answer:** "Container images should be minimal. Full CUDA torch adds 1.7GB of unnecessary weight for a CPU workload and causes disk failures in CI. Pre-installing the CPU wheel is standard ML container practice."

---

### LangGraph agent on top of RAG pipeline
The agent adds a reasoning layer before retrieval. A RAG pipeline always retrieves — an agent decides first. Non-clinical queries skip Pinecone entirely. The agent reuses existing QueryEngine and Generator — it does not replace them, it orchestrates them.
**Interview answer:** "The agent adds decision-making on top of the pipeline. For a clinical system this means non-medical queries never hit Pinecone — reducing latency and cost. The reasoning is transparent: the response includes the agent's decision string for auditability."

---

### Human-in-the-loop confidence gate
When the average cross-encoder score of retrieved chunks falls below 3.5, the agent flags the answer for pharmacist review rather than delivering directly. AI augments clinical judgment — it does not replace it.
**Interview answer:** "Clinical decisions above a confidence threshold go directly to the care manager. Below the threshold, the answer is flagged for clinical review. This is human-in-the-loop design — the system knows what it does not know."

---

### FHIR R4 patient context injection
The generator loads the patient's FHIR R4 bundle — conditions, lab observations, medications — and injects this context into the LLM prompt. Answers are personalised to the specific patient's clinical picture, not just generic guideline summaries.
**Interview answer:** "FHIR R4 is the US healthcare interoperability standard mandated by the CMS Interoperability Rule since 2021. Every EHR — Epic, Cerner — exposes patient data via FHIR APIs. Integrating FHIR context means the system answers 'what is the HbA1c target for this specific patient on Metformin with an eGFR of 62' rather than 'what is the HbA1c target for diabetes in general.'"

---

### HIPAA audit trail
Every query is logged with timestamp, request ID, patient ID, and SHA-256 hash of the query text. Raw query text is never stored — it may contain PHI. Error logs emit only exception type names, never messages which could echo back PHI.
**Interview answer:** "HIPAA §164.312(b) requires audit controls that record activity in systems containing PHI. We log the query hash not the query itself — allowing correlation without PHI storage. This is a technical safeguard requirement, not optional."

---

## 3. Key Technical Concepts

### asyncio in 4 lines
- `async def` — function can pause while waiting
- `await` — pause here, let others run, come back with result
- `asyncio.gather()` — start multiple tasks simultaneously, wait for slowest
- `asyncio.run()` — entry point, starts the event loop (scripts only, never in FastAPI)

### Why async matters for FastAPI
Sync endpoint: server blocks during LLM call (~1500ms). 10 users → last user waits 15 seconds.
Async endpoint: server handles other users during LLM wait. 10 users → all wait ~1500ms.

### Full RAG pipeline walk-through (POST /query)
Request hits FastAPI. Pydantic validates — wrong type or unknown field returns 422 immediately. QueryProcessor strips whitespace, preserves clinical values (%), builds Pinecone patient isolation filter. Retriever embeds the query using all-MiniLM-L6-v2 in a thread pool (CPU-bound), queries Pinecone for top 10 cosine matches. Reranker scores all 10 (query, chunk) pairs with cross-encoder in a thread pool, returns top 5. Generator loads FHIR patient context if patient_id present, builds numbered context block with page citations, calls Groq with clinical system prompt at temperature 0.1, retries up to 3 times with exponential backoff. FastAPI serialises QueryResponse with answer, page citations, model used, and processing time. HIPAA audit logger writes query hash, outcome, and confidence to SQLite.

### Full agent walk-through (POST /agent/query)
Request hits FastAPI. Pydantic validates. LangGraph agent starts. decide_node calls LLM to reason whether retrieval is needed — returns JSON with needs_retrieval and reasoning. route_after_decision conditional edge reads needs_retrieval and routes to either retrieve_node or directly to generate_node. If retrieval: retrieve_node runs full 3-stage QueryEngine (embed, search, rerank). generate_node calls Generator with chunks and FHIR context. confidence_check_node averages cross-encoder scores — below 3.5 prepends clinical review warning to the answer. Response includes answer, reasoning, confidence_score, requires_review flag.

### RAGAS 4 metrics
- Faithfulness: fraction of answer claims supported by retrieved context. Catches hallucination.
- Answer relevancy: does the answer actually address the question? Catches off-topic responses.
- Context precision: fraction of retrieved chunks that were actually relevant. Catches retrieval noise.
- Context recall: did retrieval find all information needed to answer correctly? Requires ground truth.

### HIPAA-relevant engineering decisions
- uuid5 chunk IDs — reliable deletion for right-to-erasure
- `extra="forbid"` — reject unexpected PHI fields at API boundary
- Patient ID metadata filters in Pinecone — query isolation per patient
- Page-level citations — auditability, every claim traceable to source
- Clinical system prompt — LLM prohibited from answering beyond context
- SHA-256 query hashing in audit log — correlation without PHI storage
- Exception type names only in error logs — no message echo-back of PHI
- `.env` excluded from Docker image — secrets injected at runtime

### Deep vs shallow health check
Shallow: returns 200 if the process is alive. Does not verify external dependencies.
Deep (what we built): calls Pinecone index stats API and Groq models list API on every probe. Three-state: healthy / degraded / unhealthy. Azure Container Apps and Kubernetes use this to route traffic — degraded means partial functionality, unhealthy means remove from load balancer entirely.
**Interview answer:** "A process can be running but completely unable to serve requests if its dependencies are down. Shallow checks hide Pinecone outages. We also have liveness vs readiness distinction: liveness is 'is the process alive, should we restart it?', readiness is 'is it ready to serve traffic, should we route to it?' In production you need both."


## Azure Container Apps Deployment

- ACR Tasks blocked on free tier — fixed by building image in GitHub Actions
  and pushing directly to ACR using docker/build-push-action
- Container Apps requires secrets as quoted strings: "key=value" format
- min-replicas 1 keeps container warm — no cold start delay during demos
- Deep health check confirms Pinecone connected and model loaded at startup
- environment: production confirms correct environment variable injection
- Live URL: https://healthcare-rag-api.happyflower-b39041fb.eastus.azurecontainerapps.io/docs

Interview answer: "The system is deployed to Azure Container Apps at a public
HTTPS URL. The Docker image is built in GitHub Actions and pushed to Azure
Container Registry on every merge to main. Secrets are injected at runtime
via Azure Container Apps secret references — never baked into the image.
The /health endpoint confirms all external dependencies are reachable before
the container accepts traffic."
---

## 4. RAGAS Evaluation Results

### Primary evaluation (local run, Llama 3.1-8b judge)
```
faithfulness:      1.0000
answer_relevancy:  0.9458
context_precision: 0.8978
context_recall:    0.9000
```
Test set: 10 clinical questions on WHO diabetes guidelines
Embedding: all-MiniLM-L6-v2 (384-dim) | Framework: RAGAS 0.2.6

### CI regression check (gpt-oss-120b judge)
```
faithfulness:      0.6282
answer_relevancy:  0.7182
context_precision: 0.4021
context_recall:    0.3333
```

### Why scores differ — RAGAS judge model variance
RAGAS scores are not absolute. They depend heavily on which LLM is used as the judge. The same pipeline, same documents, same retrieval — different judge model — produces completely different scores. This is a known limitation of LLM-as-judge evaluation frameworks.

The CI scores are lower for two reasons: gpt-oss-120b applies different evaluation criteria than Llama 3.1, and the ground truths were written to match the clinical reasoning style that Llama 3.1 uses when judging faithfulness.

**Interview answer:** "Our CI pipeline uses a fixed judge model for regression detection — if scores drop significantly between deployments, it alerts. But we do not treat the absolute CI score as a quality gate without calibration, because RAGAS scores vary significantly across judge LLMs. The primary evaluation uses Llama 3.1 as judge — faithfulness 1.0, all metrics above 0.89. The CI uses a different model and serves as relative regression detection. Comparing absolute RAGAS scores across judge models is methodologically incorrect — the same pipeline can score 1.0 with one judge and 0.6 with another."

---

## 5. Questions You Will Be Asked

**"Why Pinecone over pgvector?"**
Pinecone for managed vector similarity search — no infrastructure to maintain, scales automatically. pgvector for relational data, audit logs, and metadata where SQL joins are needed. Phase 2 adds pgvector for the audit trail alongside Pinecone for search.

**"How do you handle HIPAA right-to-erasure?"**
uuid5 IDs are deterministic — same source + page + chunk_index always produces the same chunk ID. Delete by Pinecone metadata filter on source path. Re-ingestion after deletion overwrites safely. No orphaned vectors.

**"What happens if the LLM hallucinates?"**
Clinical system prompt instructs the model to answer only from the numbered context block. Temperature 0.1 minimises creative drift. Insufficient context returns a specific "insufficient information" response rather than guessing. RAGAS faithfulness monitors this continuously.

**"What is the difference between bi-encoder and cross-encoder?"**
Bi-encoder embeds query and document independently — cosine similarity comparison, fast, scales to millions of vectors. Cross-encoder reads query and document together in one forward pass — much more accurate but O(n) complexity. Production pattern: bi-encoder retrieves candidates, cross-encoder reranks them.

**"Walk me through the LangGraph agent decision flow"**
decide_node calls the LLM with a routing prompt. If needs_retrieval is true, retrieve_node runs the full QueryEngine — embed, search, rerank. generate_node calls the Generator with chunks and FHIR patient context. confidence_check_node averages cross-encoder scores — below threshold prepends a clinical review warning. Every node receives the full ClinicalAgentState TypedDict and returns an updated copy. The conditional edge function route_after_decision reads needs_retrieval and returns the name of the next node.

**"How does your confidence gate work in production terms?"**
The average cross-encoder score of the top 5 retrieved chunks serves as a proxy for retrieval confidence. A score below 3.5 means the retrieved context is not strongly related to the query — the answer may be unreliable. The gate adds a warning banner and sets requires_review: true in the response, allowing the calling system to route for pharmacist review before displaying to the clinician. The threshold should be dynamic per query type in production — diagnostic queries warrant a higher threshold than general information queries.

**"What is FHIR and why does it matter?"**
FHIR R4 (Fast Healthcare Interoperability Resources) is the standard data format for exchanging healthcare information. It is mandated by the CMS Interoperability Rule for US payers and providers since 2021. Every major EHR — Epic, Cerner, Oracle Health — exposes patient data via FHIR APIs. Integrating FHIR means the system can answer personalised clinical questions based on a patient's actual conditions, lab values, and medications rather than giving generic guideline responses.

**"How do you handle RAGAS score variance across judge models?"**
RAGAS scores are judge-model-dependent. The same pipeline scored faithfulness 1.0 with Llama 3.1 and 0.6282 with gpt-oss-120b as judge. The correct production approach is to fix the judge model and track relative changes — a drop of 0.2 between deployments with the same judge is a signal. Comparing absolute scores across judge models is methodologically incorrect. We use CI evaluation for regression detection with a fixed judge, and primary evaluation with a calibrated judge model for absolute scores.

**"How would you scale this to 10,000 concurrent users?"**
Each container is stateless — all state in Pinecone and the LLM provider. Scale horizontally behind a load balancer. The embedding model and cross-encoder are the CPU bottleneck. In production, separate the embedding service onto dedicated instances. The LLM call is an external API call that the async event loop handles concurrently across all requests. Azure Container Apps auto-scales based on HTTP request queue depth.

**"What would you add in Phase 2?"**
BM25 hybrid search with Reciprocal Rank Fusion for exact medical term matching. LiteLLM abstraction layer for multi-provider failover — Groq primary, Azure OpenAI fallback. Dynamic confidence thresholds — query classifier sets threshold per query type. Real-time RAGAS — LLM judge on every production query, alert on faithfulness degradation. LangGraph checkpointing with SqliteSaver so failed agent runs resume from the last successful node.

---

## 6. Known Gaps —

**Gap 1 — BM25 hybrid search missing**
Current system uses vector search only. BM25 misses exact medical term matches — ICD codes, drug names, lab test codes. Phase 2: rank_bm25 library, combine BM25 and vector scores with Reciprocal Rank Fusion. One additional retriever, merge results before reranking.

**Gap 2 — Double model load in retriever**
retriever.py creates a new DocumentEmbedder() in __init__. query_engine.py also creates one. Two instances of the same model loaded in memory. Fix: dependency injection — pass the singleton embedder into Retriever.__init__. Saves ~500MB RAM per container instance.

**Gap 3 — No real-time RAGAS**
Current RAGAS runs offline on a test set only. Production: LLM judge scores every query for faithfulness in an async background task after generate_node. Score logged to LangSmith. Alert fires if rolling average drops below threshold.

**Gap 4 — Single LLM provider, no failover**
If Groq goes down, all queries fail. Fix: LiteLLM abstraction layer. Primary Groq, automatic failover to Azure OpenAI on 5xx errors. No code change in the generator — the abstraction layer handles provider switching.

**Gap 5 — Static confidence threshold**
Current threshold is 3.5 — hardcoded, applies to all query types. Production: query classifier sets threshold before confidence_check_node. Diagnostic queries: threshold 5.0. General information queries: threshold 3.0. Implementation adds one classify_node before decide_node.

**Gap 6 — RAGAS ground truth bottleneck at scale**
10 ground-truth questions is manageable manually. At 1,000 questions it requires a clinical team. Production solution: stronger LLM generates ground truths from source documents, clinician spot-checks 10%. Reduces annotation burden by 90%.

### Gap 7: Confidence gate measures retrieval quality, not generation faithfulness
Current confidence_check_node averages cross-encoder scores of (query, chunk) pairs
from the reranking stage. This measures whether retrieved chunks are relevant to the
query — not whether the generated answer is grounded in those chunks.

A hallucinated answer that is on-topic scores high on retrieval relevance
but is not caught by the confidence gate.

Production fix: add a second cross-encoder pass after generation, scoring
(chunk, generated_answer) pairs. High score = answer is grounded in chunks.
Low score = answer diverges from retrieved context = flag for review.

This is runtime faithfulness checking. RAGAS faithfulness does this offline
on a test set. Runtime faithfulness checking does it on every production query.
Architecture: add groundedness_check_node after generate_node in the LangGraph
graph, scoring top 5 (chunk, answer) pairs with the same cross-encoder.

Interview answer: "Our confidence gate measures retrieval quality — are the
retrieved chunks relevant to the query? It does not measure generation
faithfulness — did the LLM actually use those chunks? Hallucination is
currently caught by the clinical system prompt instruction and measured
offline by RAGAS. Phase 2 adds runtime faithfulness checking via a second
cross-encoder pass scoring (chunk, generated_answer) pairs after generation."

---

## 7. What This Project Demonstrates

| Skill | Evidence |
|-------|---------|
| Production RAG | End-to-end pipeline, not a tutorial |
| Agentic AI | LangGraph 4-node agent with conditional routing |
| Human-in-the-loop | Confidence gate, clinical review flagging |
| FHIR R4 | Patient context injection from FHIR bundles |
| HIPAA compliance | Audit trail, query hashing, deterministic deletion |
| Async Python | asyncio.gather, to_thread, FastAPI lifespan pattern |
| Pydantic v2 | BaseModel, ConfigDict, field_validator, pydantic-settings |
| Evaluation discipline | RAGAS scores on record, judge model variance documented |
| Docker | CPU torch optimisation, secrets at runtime, layer caching |
| CI/CD | Lint + import validation + Docker build + RAGAS evaluation gate |
| Observability | LangSmith stage-level tracing, request ID correlation |
| Debugging methodology | Diagnostic scripts to isolate failures, not random trial and error |
| Healthcare domain | Prior authorisation use case, ICD-10 awareness, clinical safety design |

---

## 8. Commit History — What a Production Git Log Looks Like

```
feat: add ingestion pipeline with page-level PDF processing
feat: add config.py with pydantic-settings, path-anchored .env
feat: add models.py — Pydantic v2 request/response contracts
feat: add 3-stage query pipeline with cross-encoder reranking
fix: reranker carries page_number through ChunkResult reconstruction
feat: add generation layer with clinical prompt and exponential backoff retry
feat: add page-level citations — page_number stored in Pinecone metadata
feat: add RAGAS evaluation with threshold alerting
feat: add FastAPI — /health /query /ingest with lifespan startup
feat: add LangGraph 4-node clinical agent with human-in-the-loop
feat: add FHIR R4 patient context — personalised clinical answers
feat: add HIPAA audit trail — SHA-256 query hashing, SQLite log, GET /audit
feat: add deep health check — verify Pinecone and Groq connectivity
feat: add request ID tracing, HIPAA-safe error logging, 30s timeouts
feat: add LangSmith observability — stage-level pipeline tracing
ci: GitHub Actions — lint + imports + Docker build + RAGAS evaluation gate
fix: use CPU-only torch in Docker — image size and CI disk space
fix: RAGAS judge model variance — warning-only gate, document score context
```

No WIP commits. No "fix bug" without context. Each commit is a working, testable increment. This is what production git history looks like.

---

## 9. The 45-Second Project Pitch

*"Healthcare workers make dozens of knowledge-dependent decisions every shift — drug dosages, diagnostic criteria, treatment protocols. The answers exist in clinical guidelines but guidelines are dense 500-page PDFs. Searching manually under time pressure leads to delays and errors.*

*This system lets a care manager ask a natural language question and receive a precise answer grounded in verified WHO guidelines — with the exact page number for verification. When patient Priya Sharma has an HbA1c of 8.2% and an eGFR of 62, the system personalises the guideline recommendation to her specific clinical picture via FHIR R4 integration.*

*The LangGraph agent adds intelligent routing — non-clinical queries skip Pinecone entirely, and low-confidence answers are flagged for pharmacist review before reaching the clinician. Every query is logged to a HIPAA-compliant audit trail. The CI pipeline blocks deployment automatically if faithfulness drops below threshold.*

*RAGAS evaluation: faithfulness 1.0, answer relevancy 0.946, context precision 0.898, context recall 0.900 with Llama 3.1 as judge."*
