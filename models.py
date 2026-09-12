"""models.py
Pydantic v2 request/response models for the clinical Decision Spport RAG API.
These are the data contracts for all API endpoints.
"""
from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator

# ----------------Shared-------------------------------------

class ChunkResult(BaseModel):
    """A single retrieved chunk returned with its similarity score."""
    model_config = ConfigDict(frozen=True) #response chunks are immutable

    chunk_id: str
    text: str
    source: str
    score: float   # cosine similarity — 0.0 (unrelated) to 1.0 (identical)
    chunk_index: int
    chunk_type: str = "text.pdf"
    page_number: Optional[int] = None       # ADD — page in source PDF
    section_header: Optional[str] = None    # ADD — section title if available

#----------------------------Query endpoint---------------------

class QueryRequest(BaseModel):
    """
    Request model for POST/query.
    Validates and cleans all incoming data before the pipeline touches it. 
    """
    model_config = ConfigDict(protected_namespaces=())   # ADD THIS
    model_config = ConfigDict(
        extra="forbid",   # reject unknown fields — security
        str_strip_whitespace=True  # "  HbA1c  " → "HbA1c" automatically
    )
    query: str
    patient_id: Optional[str] = None  # will become required when HIPAA layer added
    max_results: int = 5
    include_sources: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, v:str)-> str:
        if len(v.strip()) < 3:
            raise ValueError("Query too short - minimum 3 characters")
        if len(v) > 2000:
            raise ValueError("Query too long - maximum 2000 characters")
        return v.strip()

    @field_validator("max_results")
    @classmethod
    def validate_max_results(cls, v:int) -> int:
        if v < 1 or v > 20:
            raise ValueError("max_results must be between 1 and 20")
        return v
    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, v: Optional[str])-> Optional[str]:
        if v is None:
            return v
        normalized = v.upper()
        if not normalized.startswith("P"):
            raise ValueError("patient_id must start with 'P' - e.g. P001")
        return normalized

class QueryResponse(BaseModel):
    """Response model for POST /query."""
    model_config = ConfigDict(protected_namespaces=())  
    answer: str
    query: str
    patient_id: Optional[str] = None
    sources: list[ChunkResult] = []
    retrieved_chunks_count: int
    model_used: str
    processing_time_ms: float

#--------Ingestion endpoint------------------------------

class IngestionRequest(BaseModel):
    """Request model for POST/ ingest - triggers pipeline for one document."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    file_path: str
    document_type: str = "general"

    @field_validator("document_type")
    @classmethod
    def validate_document_type(cls, v:str)-> str:
        allowed = {"general", "clinical_notes", "research_papers", "hl7_messages"}
        if v not in allowed:
            raise ValueError (f"document_type must be one of:{allowed}")
        return v

class IngestionResponse(BaseModel):
    """Response model for POST/ ingest."""
    status: str
    source: str
    chunks_stored: int
    processing_time_ms: float

# ── Agent endpoint ────────────────────────────────────────────────────────────

class AgentQueryRequest(BaseModel):
    """Request model for POST /agent/query."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, protected_namespaces=())

    query:       str
    patient_id:  Optional[str] = None
    max_results: int = 5

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError("Query too short — minimum 3 characters")
        return v.strip()


class AgentQueryResponse(BaseModel):
    """Response model for POST /agent/query."""
    model_config = ConfigDict(protected_namespaces=())
    answer:           str
    query:            str
    sources:          list[str]
    needs_retrieval:  bool           # did agent decide to search documents?
    reasoning:        str            # agent's decision explanation
    chunks_used:      int
    model_used:       str
    processing_time_ms: float
    confidence_score:   float = 0.0    # ADD
    requires_review:    bool  = False  # ADD

#-----------------Health check---------------------------

class HealthCheckResponse(BaseModel):
    """
    Response model for GET /health.
    Every production deploymemt - Azure, AWS, Docker - pings this endpoint
    to know if the service is alive. Required for kubernetes, load balancers,
    and Azure Container Apps health probes.
    """
    status: str
    pinecone_connected: bool
    embedding_model_loaded: bool
    version: str = "1.0.0"
    environment: str
    

