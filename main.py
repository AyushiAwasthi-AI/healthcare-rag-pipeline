"""
main.py
Ingestion orchestrator — processes all PDFs from data/documents/ into Pinecone.
Run this whenever you change the ingestion pipeline or need to re-ingest documents.
"""
import os
import asyncio
import logging
from datetime import datetime
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import VectorStore

from pathlib import Path

# Resolves relative to main.py location — never breaks
BASE_DIR   = Path(__file__).resolve().parent
docs_folder = str(BASE_DIR / "data" / "documents")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ingestion(
    file_path: str,
    embedder: DocumentEmbedder,
    vector_store: VectorStore,
) -> dict:
    """
    Process one document end to end — synchronous.
    Loader returns a list of page dicts, each with page_number.
    Called inside asyncio.to_thread() so it never blocks the event loop.
    """
    start_time = datetime.utcnow()

    # Stage 1 — load: returns list of page dicts
    loader = DocumentLoader()
    pages = loader.load(file_path)

    # Filter out scanned/empty pages
    valid_pages = [p for p in pages if p.get("text", "").strip()]

    if not valid_pages:
        logger.warning(f"Empty document skipped: {os.path.basename(file_path)}")
        return {"source": file_path, "status": "skipped", "chunks": 0}

    # Stage 2 — chunk each page separately (preserves page_number per chunk)
    chunker = DocumentChunker()
    all_chunks = []
    for page in valid_pages:
        page_chunks = chunker.chunk(page)
        all_chunks.extend(page_chunks)

    if not all_chunks:
        return {"source": file_path, "status": "skipped", "chunks": 0}

    # Stage 3 — embed
    embedded_chunks = embedder.embed(all_chunks)

    # Stage 4 — store in Pinecone
    stored = vector_store.upsert_chunks(embedded_chunks)

    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info(
        f"Completed {os.path.basename(file_path)}: "
        f"{stored} chunks in {duration:.1f}s"
    )

    return {
        "source": file_path,
        "status": "success",
        "chunks": stored,
        "duration_seconds": round(duration, 2),
    }


async def process_all_documents(
    docs_folder: str,
    embedder: DocumentEmbedder,
    vector_store: VectorStore,
    max_concurrent: int = 3,
) -> list[dict]:
    """
    Process multiple documents concurrently.
    Semaphore limits memory — never more than max_concurrent docs at once.
    """
    pdf_files = [
        os.path.join(docs_folder, f)
        for f in os.listdir(docs_folder)
        if f.endswith(".pdf") and not f.startswith("~$")
    ]

    if not pdf_files:
        logger.warning(f"No PDFs found in {docs_folder}")
        return []

    print(f"\nFound {len(pdf_files)} documents to process")

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(file_path: str):
        async with semaphore:
            print(f"  Starting: {os.path.basename(file_path)}")
            result = await asyncio.to_thread(
                run_ingestion, file_path, embedder, vector_store
            )
            print(f"  Done: {os.path.basename(file_path)} — {result['chunks']} chunks")
            return result

    tasks = [process_with_limit(fp) for fp in pdf_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = [r for r in results if isinstance(r, dict)]
    failures  = [r for r in results if isinstance(r, Exception)]
    for f in failures:
        print(f"FAILED: {f}")   # print forces it to show regardless of log level
        import traceback
        traceback.print_exception(type(f), f, f.__traceback__)

    return successes


if __name__ == "__main__":
    #docs_folder = r"C:\Users\ayushi.awasthi\Documents\Python\rag-pipeline\data\documents"

    # Singletons — created once, shared across all documents
    embedder      = DocumentEmbedder()
    vector_store  = VectorStore()

    results = asyncio.run(
        process_all_documents(
            docs_folder,
            embedder,
            vector_store,
            max_concurrent=3,
        )
    )

    total_chunks = sum(r["chunks"] for r in results)
    stats        = vector_store.get_index_stats()

    print(f"\n{'='*50}")
    print(f"Ingestion complete")
    print(f"Documents processed : {len(results)}")
    print(f"Total chunks stored : {total_chunks}")
    print(f"Vectors in Pinecone : {stats['total_vectors']}")
    print(f"{'='*50}")