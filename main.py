import os
import asyncio
import logging
from datetime import datetime
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Synchronous ingestion logic (unchanged) ──────────────────────────
'''def run_ingestion(file_path: str,
                  embedder: DocumentEmbedder,
                  vector_store: VectorStore) -> dict:
    """
    Process one document end to end.
    Returns ingestion result dict for audit logging.
    """
    start_time = datetime.utcnow()

    loader = DocumentLoader()
    document = loader.load(file_path)

    if not document["text"].strip():
        logger.warning(f"Empty document skipped: {file_path}")
        return {"source": file_path, "status": "skipped", "chunks": 0}

    chunker = DocumentChunker()
    chunks = chunker.chunk(document)

    embedded_chunks = embedder.embed(chunks)
    stored = vector_store.upsert_chunks(embedded_chunks)

    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Completed {os.path.basename(file_path)} in {duration:.1f}s")

    return {
        "source": file_path,
        "status": "success",
        "chunks": stored,
        "duration_seconds": duration
    }

# ── Async orchestration layer ─────────────────────────────────────────
async def process_all_documents(docs_folder: str,
                                 embedder: DocumentEmbedder,
                                 vector_store: VectorStore,
                                 max_concurrent: int = 3):
    """
    Process multiple documents concurrently.
    max_concurrent controls memory usage — never process more than
    this many documents simultaneously.
    """
    pdf_files = [
        os.path.join(docs_folder, f)
        for f in os.listdir(docs_folder)
        if f.endswith(".pdf") and not f.startswith("~$")
    ]

    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_folder}")
        return []

    print(f"\nFound {len(pdf_files)} documents")
    print(f"Processing up to {max_concurrent} simultaneously\n")

    # Semaphore = traffic light for coroutines
    # max_concurrent = max green lights at once
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(file_path: str):
        """Wrap sync function in async context with concurrency limit."""
        async with semaphore:
            print(f"Starting: {os.path.basename(file_path)}")
            # asyncio.to_thread runs sync function in thread pool
            # without blocking the event loop
            result = await asyncio.to_thread(
                run_ingestion,
                file_path,
                embedder,
                vector_store
            )
            print(f"Done: {os.path.basename(file_path)} "
                  f"— {result['chunks']} chunks")
            return result

    # gather runs all coroutines concurrently
    # return_exceptions=True means one failure doesn't stop others
    tasks = [process_with_limit(fp) for fp in pdf_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successes from failures
    successes = [r for r in results if isinstance(r, dict)]
    failures = [r for r in results if isinstance(r, Exception)]

    for i, failure in enumerate(failures):
        logger.error(f"Failed: {failure}")

    return successes

# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    docs_folder = r"data/documents"

    # Singletons — created ONCE, shared across all documents
    embedder = DocumentEmbedder()
    vector_store = VectorStore()

    # asyncio.run() starts the event loop
    # everything async runs inside here
    results = asyncio.run(
        process_all_documents(
            docs_folder,
            embedder,
            vector_store,
            max_concurrent=3
        )
    )

    # Summary
    total_chunks = sum(r["chunks"] for r in results)
    print(f"\n{'='*50}")
    print(f"Pipeline complete")
    print(f"Documents processed: {len(results)}")
    print(f"Total chunks stored: {total_chunks}")
    print(f"{'='*50}")'''


#-------------------------Old code-----------------------------

import os
import asyncio
import logging
from datetime import datetime
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import DocumentEmbedder
from ingestion.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#------------- Synchronous ingestion logic-----------------------
def run_ingestion(file_path: str, embedder: DocumentEmbedder, vector_store: VectorStore) -> dict:
    
    #Process one document end to end.Returns ingestion result dict for audit logging

    start_time =datetime.utcoffset()
    loader = DocumentLoader()
    document = loader.load(file_path)

    if not document["text"].strip():
        print(f"Skipping empty document: {file_path}")
        return {"source":file_path, "status": "skipped", "chunks": 0 }

    chunker = DocumentChunker()
    chunks = chunker.chunk(document)

    embedder = DocumentEmbedder()
    embedded_chunks = embedder.embed(chunks)

    vector_store = VectorStore()
    stored = vector_store.upsert_chunks(embedded_chunks)

    duration = (datetime.utcoffset() - start_time).total_seconds()
    logger.info(f"Completed {os.path.basename(file_path)} in {duration: 1f}s")

    return {
        "source": file_path,
        "status": "success",
        "chunks": stored,
        "duration_Seconds": duration
    }
#-----------Async orchestration layer-------------------------
async def process_all_documents(docs_folder: str, embedder: DocumentEmbedder, vector_store: VectorStore, max_concurrent: int=3):
    """Process multiple docs concurrently.
    max_concurrent contrils memory usage - never process more than 60% of RAM.
    How to Pick the RIGHT max_concurrent
    # Professional approach:
    # Test with small batch first!

    # Step 1: test with 10 documents
    # measure: time, memory usage, errors

    # Step 2: gradually increase
    # concurrent=1  → baseline
    # concurrent=3  → 3x faster? (ideal)
    # concurrent=5  → 5x faster? (or memory issues?)
    # concurrent=10 → 10x faster? (or API rate limits?)

    # Step 3: find the breaking point
    # where does it stop getting faster?
    # where does memory spike?
    # where do API errors start?

    # That breaking point - 1 = your max_concurrent! ✅

    # Also consider API rate limits:
    # OpenAI limit: 3000 requests/minute
    # Each doc = 10 embedding calls
    # concurrent=3 → 30 calls/second → 1800/min ✅
    # concurrent=10 → 100 calls/second → 6000/min ❌ RATE LIMITED!
    
    """
    
    pdf_files = [os.path.join(docs_folder, f) for f in os.listdir(docs_folder) if f.endswith(".pdf") and not f.startswith("~$")]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {docs_folder}")
        return []
    
    print(f"\nFound {len(pdf_files)} documents")
    print(f"Processing up to {max_concurrent} simultaneously\n")

    #Semaphore = traffic light for coroutines
    #max_concurrent = max green lights at once
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(file_path: str):
        """Wrap sync function in async context with concurrency limit."""
        async with semaphore:
            print(f"Starting: {os.path.basename(file_path)}")
            #asyncio.to_thread runs sync function in thread pool
            #without blocking the event loop
            result = await asyncio.to_thread(run_ingestion, file_path, embedder, vector_store)
            print(f"Done:{os.path.basename(file_path)}" f"- {result['chunks']} chunks")
            return result
    
    tasks = [process_with_limit(fp) for fp in pdf_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    
    stats = vector_store.get_index_stats()

    print(f"\n{'='*50}")
    print(f"Document: {os.path.basename(file_path)}")
    print(f"Chunks stored: {stored}")
    print(f"Total vectors in DB: {stats['total_vectors']}")
    print(f"{'='*50}")

    return embedded_chunks


if __name__ == "__main__":
    docs_folder = r"C:\Users\ayushi.awasthi\Documents\Python\rag-pipeline\data\documents"

    # Filter both ~$ temp files AND non-PDFs
    pdf_files = [
        os.path.join(docs_folder, f)
        for f in os.listdir(docs_folder)
        if f.endswith(".pdf") and not f.startswith("~$")
    ]

    print(f"Found {len(pdf_files)} documents to process")

    for file_path in pdf_files:
        print(f"\nProcessing: {os.path.basename(file_path)}")
        try:
            run_ingestion(file_path)
        except Exception as e:
            logger.error(f"Failed: {os.path.basename(file_path)}: {e}")
            print(f"Skipping: {os.path.basename(file_path)}")
            continue

    print("\nAll documents processed successfully.")