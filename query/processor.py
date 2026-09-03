"""
query/processor.py
Stage 1: cleans the raw query and extracts Pinecone metadata filters.
A noisy query degrades retrieval. Clean first, retrieve second.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Output of QueryProcessor — passed to Retriever."""
    original_query: str
    clean_query: str
    patient_id: Optional[str] = None
    filters: dict = field(default_factory=dict)


class QueryProcessor:
    """
    Normalizes the query and builds Pinecone metadata filters.
    Keeps retrieval clean and HIPAA-aware from the start.
    """

    def process(
        self,
        query: str,
        patient_id: Optional[str] = None,
    ) -> ProcessedQuery:
        """Clean query and build filters. Returns ProcessedQuery."""
        original = query
        clean = self._clean(query)
        filters = self._build_filters(patient_id)

        logger.info(f"Query processed: '{original[:50]}' → '{clean[:50]}'")

        return ProcessedQuery(
            original_query=original,
            clean_query=clean,
            patient_id=patient_id,
            filters=filters,
        )

    def _clean(self, query: str) -> str:
        """Normalize whitespace, strip junk characters."""
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)
        query = re.sub(r'[^\w\s\?\.\,\-\/\%]', '', query)
        # Ensure query ends with ? for better semantic matching
        if not query.endswith('?') and len(query.split()) > 3:
            query = query + '?'
        return query

    def _build_filters(self, patient_id: Optional[str]) -> dict:
        """
        Build Pinecone metadata filter dict.
        Empty dict = no filter = search all 706 vectors.
        Patient filter = search only that patient's chunks (HIPAA).
        """
        if patient_id:
            return {"patient_id": {"$eq": patient_id.upper()}}
        return {}