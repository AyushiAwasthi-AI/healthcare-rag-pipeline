import re
import logging
from pathlib import Path
from typing import List
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

logger = logging.getLogger(__name__)

class DocumentChunker:
    '''Chunks documents into precise, meaningful pieces.
    Applies different strategies based on document type.
    Cleans text before chunking based on source quality.'''

    def __init__(self, chunk_size: int =512, chunk_overlap: int =None):
        self.chunk_size = chunk_size
    #if overlap not specificed - always 25% of chunk size
        self.chunk_overlap = chunk_overlap or (chunk_size//4)

    def _clean_text(self, text: str, doc_type: str) -> str:
        """
        Clean text based on source quality.
        OCR'd text needs aggressive cleaning.
        Clean PDFs need minimal cleaning.
        """
        # Always normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Aggressive cleaning for OCR'd documents
        if doc_type in ("scanned_pdf", "image_ocr"):
            text = re.sub(r'[^\x20-\x7E\n]', '', text)
            text = text.replace('|', 'I')
            logger.info("Applied aggressive OCR cleaning")

        return text    

    

    def chunk(self, document: dict) -> List[dict]:
    # Step 1 - clean text
        cleaned_text = self._clean_text(
            document["text"],
            document["type"]
        )

        if not cleaned_text.strip():
            logger.warning(f"Empty text after cleaning: {document['source']}")
            return []

        # Step 2 - dynamic chunk size based on document length
        self.chunk_size = self._calculate_chunk_size(cleaned_text)
        self.chunk_overlap = self.chunk_size // 4

        logger.info(
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

        # Step 3 - select strategy
        strategy = self._select_strategy(document["type"])
        logger.info(f"Strategy: {strategy}")

        # Step 4 - apply strategy
        if strategy == "structure":
            chunks = self._structure_chunk(cleaned_text)
        elif strategy == "semantic":
            chunks = self._semantic_chunk(cleaned_text)
        else:
            chunks = self._recursive_chunk(cleaned_text)

        # Step 5 - attach metadata to every chunk
        return self._attach_metadata(chunks, document)
    
    def _select_strategy(self, doc_type: str) -> str:
        strategy_map = {
            "text_pdf": "structure",
            "scanned_pdf": "recursive",
            "image_ocr": "recursive",
            "mixed_pdf": "structure"
        }
        return strategy_map.get(doc_type, "recursive")

    def _structure_chunk(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    def _semantic_chunk(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[". ", "? ", "! ", "\n", " ", ""]
        )
        return splitter.split_text(text)

    def _recursive_chunk(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return splitter.split_text(text)

    def _attach_metadata(self, chunks: List[str], document: dict) -> List[dict]:
        return [
            {
                "text": chunk,
                "source": document["source"],
                "type": document["type"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "page_number": document.get("page_number"),   # ADD
                "total_pages": document.get("total_pages"),   # ADD
            }
            for i, chunk in enumerate(chunks)
    ]


    def _calculate_chunk_size(self, text:str) -> int:
        '''Dynamically adjust chunk size based on document length.
        Prevents retrieval coverage problem on long documents.'''

        doc_length = len(text)
        if doc_length < 5000:
            return 512
        elif doc_length < 20000:
            return 800
        else:
            return 1024