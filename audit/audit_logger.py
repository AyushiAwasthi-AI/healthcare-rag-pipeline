"""
audit/audit_logger.py
HIPAA-compliant audit trail for all clinical queries.

HIPAA §164.312(b) requires audit controls that record and examine
activity in systems containing PHI. Every query to this system
is logged with: timestamp, request_id, patient_id (if present),
query_hash (not raw query), response_confidence, and outcome.

Note: Raw query text is NOT stored — it may contain PHI.
Query is hashed (SHA-256) for correlation without PHI exposure.
"""
import hashlib
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger  = logging.getLogger(__name__)
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "audit.db"


def _get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            request_id       TEXT    NOT NULL,
            patient_id       TEXT,
            query_hash       TEXT    NOT NULL,
            endpoint         TEXT    NOT NULL,
            needs_retrieval  INTEGER,
            confidence_score REAL,
            requires_review  INTEGER,
            chunks_used      INTEGER,
            model_used       TEXT,
            processing_ms    REAL,
            outcome          TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def log_query(
    request_id:       str,
    query:            str,
    endpoint:         str,
    patient_id:       Optional[str] = None,
    needs_retrieval:  Optional[bool] = None,
    confidence_score: float = 0.0,
    requires_review:  bool  = False,
    chunks_used:      int   = 0,
    model_used:       str   = "",
    processing_ms:    float = 0.0,
    outcome:          str   = "success",
) -> None:
    """
    Log a clinical query to the HIPAA audit trail.
    Raw query is hashed — PHI is never stored in the audit log.
    """
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

    try:
        conn = _get_connection()
        conn.execute("""
            INSERT INTO audit_log (
                timestamp, request_id, patient_id, query_hash,
                endpoint, needs_retrieval, confidence_score,
                requires_review, chunks_used, model_used,
                processing_ms, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            request_id,
            patient_id,
            query_hash,
            endpoint,
            int(needs_retrieval) if needs_retrieval is not None else None,
            confidence_score,
            int(requires_review),
            chunks_used,
            model_used,
            processing_ms,
            outcome,
        ))
        conn.commit()
        conn.close()
        logger.info(f"[{request_id}] Audit log written: {outcome}")
    except Exception as e:
        logger.error(f"[{request_id}] Audit log failed: {type(e).__name__}")